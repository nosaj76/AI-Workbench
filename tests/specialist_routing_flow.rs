//! Actual HTTP inference checks for routing, isolated translation and teams.
use ollama_workbench::{
    agents::{self, AgentRequest},
    ollama::ChatMessage,
    routing::RoutingConfig,
    runtime::{spawn_chat, ChatEvent, ChatJob},
    settings::Settings,
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

type Responder = Arc<dyn Fn(&Value) -> Value + Send + Sync>;
struct Server {
    url: String,
    requests: Arc<Mutex<Vec<(String, Value)>>>,
    max_active: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Server {
    fn new(responder: impl Fn(&Value) -> Value + Send + Sync + 'static, delay: Duration) -> Self {
        Self::with_models(models(), responder, delay)
    }
    fn with_models(
        metadata: BTreeMap<String, Value>,
        responder: impl Fn(&Value) -> Value + Send + Sync + 'static,
        delay: Duration,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(vec![]));
        let seen = requests.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let peak = max_active.clone();
        let metadata = Arc::new(metadata);
        let responder: Responder = Arc::new(responder);
        let worker = thread::spawn(move || {
            let mut connections = vec![];
            while !stopped.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut socket, _)) => {
                        let seen = seen.clone();
                        let metadata = metadata.clone();
                        let responder = responder.clone();
                        let active = active.clone();
                        let peak = peak.clone();
                        connections.push(thread::spawn(move || {
                            socket.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                            let (path, body) = read_request(&mut socket);
                            seen.lock().unwrap().push((path.clone(), body.clone()));
                            let chat = path == "/api/chat";
                            if chat {
                                let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                                peak.fetch_max(count, Ordering::SeqCst);
                                if !delay.is_zero() { thread::sleep(delay); }
                            }
                            let response = match path.as_str() {
                                "/api/tags" => json!({"models":metadata.iter().map(|(name, info)| json!({"name":name,"size":2_000_000_000_u64,"details":info["details"]})).collect::<Vec<_>>()}),
                                "/api/ps" => json!({"models":[]}),
                                "/api/show" => metadata.get(body["model"].as_str().unwrap()).expect("unknown model").clone(),
                                "/api/chat" => responder(&body),
                                _ => panic!("Unexpected endpoint {path}"),
                            };
                            let data = format!("{response}\n");
                            let header = format!("HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", data.len());
                            let _ = socket.write_all(header.as_bytes());
                            let _ = socket.write_all(data.as_bytes());
                            if chat { active.fetch_sub(1, Ordering::SeqCst); }
                        }));
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2))
                    }
                    Err(error) => panic!("{error}"),
                }
            }
            for connection in connections {
                connection.join().unwrap();
            }
        });
        Self {
            url,
            requests,
            max_active,
            stop,
            worker: Some(worker),
        }
    }
    fn chats(&self) -> Vec<Value> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|(path, _)| path == "/api/chat")
            .map(|(_, body)| body.clone())
            .collect()
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap();
        }
    }
}
fn read_request(socket: &mut TcpStream) -> (String, Value) {
    let mut bytes = vec![];
    let mut buffer = [0u8; 4096];
    let end = loop {
        let n = socket.read(&mut buffer).unwrap();
        assert!(n > 0);
        bytes.extend_from_slice(&buffer[..n]);
        if let Some(index) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            break index + 4;
        }
        assert!(bytes.len() < 128_000);
    };
    let header = String::from_utf8_lossy(&bytes[..end]);
    let path = header
        .lines()
        .next()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .to_string();
    let length = header
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().unwrap())
        })
        .unwrap_or(0);
    while bytes.len() < end + length {
        let n = socket.read(&mut buffer).unwrap();
        assert!(n > 0);
        bytes.extend_from_slice(&buffer[..n]);
    }
    (
        path,
        if length == 0 {
            Value::Null
        } else {
            serde_json::from_slice(&bytes[end..end + length]).unwrap()
        },
    )
}
fn models() -> BTreeMap<String, Value> {
    [
        ("coordinator:4b", json!({"capabilities":["completion","tools","vision","thinking"],"model_info":{"general.parameter_count":4_000_000_000_u64}})),
        ("coder:7b", json!({"capabilities":["completion","tools"],"model_info":{"general.parameter_count":7_000_000_000_u64}})),
        ("translategemma:4b-it-q4_K_M", json!({"capabilities":["completion","vision"],"model_info":{"general.parameter_count":4_300_000_000_u64}})),
    ].into_iter().map(|(name, value)| (name.into(), value)).collect()
}
fn config() -> RoutingConfig {
    RoutingConfig {
        general: vec!["coordinator:4b".into()],
        orchestrator: vec!["coordinator:4b".into()],
        reasoning: vec!["coordinator:4b".into()],
        coding: vec!["coder:7b".into()],
        translation: vec!["translategemma:4b-it-q4_K_M".into()],
        max_parameters_b: 10.0,
        single_model_mode: true,
        ..Default::default()
    }
}
fn settings(server: &Server) -> Settings {
    Settings {
        native_tool_probe: false,
        base_url: server.url.clone(),
        model: "coordinator:4b".into(),
        model_routing: true,
        routing_json: serde_json::to_string(&config()).unwrap(),
        automatic_memory: false,
        agent_max_workers: 4,
        agent_max_tasks: 2,
        ..Default::default()
    }
}
fn answer(text: &str) -> Value {
    json!({"message":{"role":"assistant","content":text},"done":true,"eval_count":10,"eval_duration":1_000_000_000_u64})
}
fn user(body: &Value) -> &str {
    body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .rev()
        .find(|m| m["role"] == "user")
        .unwrap()["content"]
        .as_str()
        .unwrap()
}
fn job(settings: Settings, dir: &Path, prompt: &str, workspace: bool) -> ChatJob {
    ChatJob {
        model: "coordinator:4b".into(),
        settings,
        messages: vec![
            ChatMessage {
                role: "system".into(),
                content: "Private prior system context".into(),
                ..Default::default()
            },
            ChatMessage {
                role: "user".into(),
                content: prompt.into(),
                ..Default::default()
            },
        ],
        workspace: workspace.then(|| dir.to_path_buf()),
        engine: None,
        retrieval_path: dir.join("knowledge.sqlite3"),
    }
}
#[derive(Default)]
struct Outcome {
    models: Vec<String>,
    answers: Vec<String>,
    // Native assistant tool-call messages are part of the conversation, but are
    // not user-facing final answers (their content may legitimately be empty).
    native_tool_calls: usize,
    errors: Vec<String>,
}
fn finish(rx: mpsc::Receiver<ChatEvent>) -> Outcome {
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut out = Outcome::default();
    loop {
        assert!(Instant::now() < deadline, "inference did not finish");
        match rx
            .recv_timeout(Duration::from_secs(5))
            .expect("worker event timeout")
        {
            ChatEvent::Begin(model) => out.models.push(model),
            ChatEvent::Complete(message) => {
                if message.tool_calls.is_empty() {
                    out.answers.push(message.content);
                } else {
                    out.native_tool_calls += message.tool_calls.len();
                }
            }
            ChatEvent::Error(error) => out.errors.push(error),
            ChatEvent::Approval { reply, .. } => {
                let _ = reply.send(false);
            }
            ChatEvent::Finished => return out,
            _ => {}
        }
    }
}
fn run(job: ChatJob) -> Outcome {
    let (tx, rx) = mpsc::channel();
    spawn_chat(job, tx, Arc::new(AtomicBool::new(false)));
    finish(rx)
}

#[test]
fn translation_handoff_has_one_exact_native_message_and_no_tools_or_memory() {
    let server = Server::new(
        |body| {
            match body["model"].as_str().unwrap() {
        "coordinator:4b" => answer(&json!({"ready":true,"source_code":"en","target_code":"fr","text":"Hello, friend!"}).to_string()),
        "translategemma:4b-it-q4_K_M" => answer("Bonjour, mon ami !"),
        other => panic!("Wrong specialist {other}"),
    }
        },
        Duration::ZERO,
    );
    let dir = tempfile::tempdir().unwrap();
    let mut settings = settings(&server);
    settings.automatic_memory = true;
    settings.thinking = "on".into();
    settings.response_format = "json".into();
    let input = job(
        settings,
        dir.path(),
        "Translate from English to French: Hello, friend!",
        true,
    );
    let original = input.messages.clone();
    let outcome = run(input);
    assert!(outcome.errors.is_empty(), "{:?}", outcome.errors);
    assert_eq!(outcome.models, ["translategemma:4b-it-q4_K_M"]);
    assert_eq!(outcome.answers, ["Bonjour, mon ami !"]);
    let chats = server.chats();
    assert_eq!(
        chats.len(),
        2,
        "specialized translation must not add memory extraction"
    );
    assert_eq!(chats[0]["think"], false);
    assert!(chats[0].get("tools").is_none());
    let messages = chats[1]["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
    let prompt = messages[0]["content"].as_str().unwrap();
    assert!(prompt.starts_with("You are a professional English (en) to French (fr) translator."));
    assert!(prompt.ends_with("French:\n\n\nHello, friend!"));
    assert!(!prompt.contains("Private prior system context"));
    assert!(chats[1].get("tools").is_none());
    assert!(chats[1].get("think").is_none());
    assert!(chats[1].get("format").is_none());
    assert!(chats.iter().all(|body| body["keep_alive"] == "0"));
    assert_eq!(
        original[1].content,
        "Translate from English to French: Hello, friend!"
    );
    let recorded: Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("last-route.json")).unwrap())
            .unwrap();
    assert_eq!(recorded["model"], "translategemma:4b-it-q4_K_M");
}

#[test]
fn invented_source_and_unread_files_use_full_coordinator_context() {
    for prompt in [
        "Translate from English to French: Hello, friend!",
        "Translate README.md into French",
    ] {
        let server = Server::new(
            |body| {
                assert_eq!(body["model"], "coordinator:4b");
                if body.get("format").is_some() {
                    answer(&json!({"ready":true,"source_code":"en","target_code":"fr","text":"Fabricated source"}).to_string())
                } else {
                    answer("I need the source evidence before translating.")
                }
            },
            Duration::ZERO,
        );
        let dir = tempfile::tempdir().unwrap();
        let outcome = run(job(settings(&server), dir.path(), prompt, true));
        assert!(outcome.errors.is_empty(), "{:?}", outcome.errors);
        assert_eq!(outcome.models, ["coordinator:4b"]);
        let chats = server.chats();
        let final_request = chats.last().unwrap();
        assert_eq!(
            final_request["messages"][0]["content"],
            "Private prior system context"
        );
        assert_eq!(user(final_request), prompt);
        assert!(!final_request["tools"].as_array().unwrap().is_empty());
        if prompt.contains("README.md") {
            assert_eq!(chats.len(), 1, "unread file detection should skip planning");
        }
    }
}

#[test]
fn automatic_and_manual_teams_respect_role_assignments_and_worker_choices() {
    for manual in [false, true] {
        let server = Server::new(
            move |body| {
                let prompt = user(body);
                if prompt.starts_with("Decompose") {
                    let suggested = if manual { "coder:7b" } else { "coordinator:4b" };
                    answer(&json!({"tasks":[
                    {"title":"Code inspection","prompt":"Review the Rust code and unit tests","model":suggested},
                    {"title":"Evidence analysis","prompt":"Analyze the evidence and compare outcomes","model":suggested}
                ]}).to_string())
                } else if prompt.starts_with("TEAM OBJECTIVE:") {
                    answer("Inspected the assigned evidence; coverage is limited to this fixture.")
                } else {
                    answer("The two completed child findings support this synthesis.")
                }
            },
            Duration::from_millis(15),
        );
        let dir = tempfile::tempdir().unwrap();
        let mut request = AgentRequest::new(
            settings(&server),
            Some(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
            "Review the code and analyze the evidence".into(),
            "coordinator:4b".into(),
        );
        request.concurrency = 4;
        if manual {
            request.worker_models = vec!["coder:7b".into()];
        }
        let (tx, _rx) = mpsc::channel();
        let run = agents::run(request, tx, Arc::new(AtomicBool::new(false))).unwrap();
        assert_eq!(run.state, "done", "{:?}", run.error);
        assert_eq!(run.planner_model, "coordinator:4b");
        assert_eq!(run.tasks[0].model, "coder:7b");
        assert_eq!(
            run.tasks[1].model,
            if manual { "coder:7b" } else { "coordinator:4b" }
        );
        assert_eq!(run.automatic_routing, !manual);
        assert_eq!(!run.tasks[0].routing_reason.is_empty(), !manual);
        assert_eq!(server.max_active.load(Ordering::SeqCst), 1);
        let chats = server.chats();
        assert_eq!(chats.len(), 4);
        if !manual {
            assert_eq!(chats[0]["think"], false);
            assert_eq!(chats[3]["think"], false);
        }
        assert!(chats[1]["tools"].as_array().is_some_and(|v| !v.is_empty()));
        assert!(chats[2]["tools"].as_array().is_some_and(|v| !v.is_empty()));
    }
}

#[test]
fn manual_models_cannot_bypass_the_total_parameter_cap() {
    let mut metadata = models();
    metadata.insert("oversized:14b".into(), json!({"capabilities":["completion","tools"],"model_info":{"general.parameter_count":14_000_000_000_u64}}));
    let server = Server::with_models(
        metadata,
        |_| panic!("No oversized model should receive a prompt"),
        Duration::ZERO,
    );
    let dir = tempfile::tempdir().unwrap();
    let mut settings = settings(&server);
    settings.model_routing = false;
    let mut input = job(settings, dir.path(), "Hello", false);
    input.model = "oversized:14b".into();
    let outcome = run(input);
    assert_eq!(outcome.errors.len(), 1);
    assert!(outcome.errors[0].contains("exceeds the 10.00B routing cap"));
    assert!(server.chats().is_empty());
}

#[test]
fn single_model_mode_serializes_real_requests_from_separate_runtimes() {
    let server = Server::new(|_| answer("Complete"), Duration::from_millis(100));
    let dir = tempfile::tempdir().unwrap();
    let mut settings = settings(&server);
    settings.model_routing = false;
    let (tx1, rx1) = mpsc::channel();
    let (tx2, rx2) = mpsc::channel();
    spawn_chat(
        job(settings.clone(), dir.path(), "First request", false),
        tx1,
        Arc::new(AtomicBool::new(false)),
    );
    // The same API root with a trailing /api shares the endpoint's permit.
    settings.base_url.push_str("/api/");
    spawn_chat(
        job(settings, dir.path(), "Second request", false),
        tx2,
        Arc::new(AtomicBool::new(false)),
    );
    assert!(finish(rx1).errors.is_empty());
    assert!(finish(rx2).errors.is_empty());
    assert_eq!(server.chats().len(), 2);
    assert_eq!(server.max_active.load(Ordering::SeqCst), 1);
    assert!(server.chats().iter().all(|body| body["keep_alive"] == "0"));
}

#[test]
fn parent_releases_inference_permit_before_automatic_delegation() {
    let server = Server::new(
        |body| {
            let prompt = user(body);
            if prompt.starts_with("Decompose") {
                answer(&json!({"tasks":[{"title":"Rust code review","prompt":"Inspect Rust code and cite inspected evidence","model":"coder:7b"}]}).to_string())
            } else if prompt.starts_with("TEAM OBJECTIVE:") {
                answer("The code inspection completed with limited fixture evidence.")
            } else if prompt.starts_with("OBJECTIVE:") {
                answer("The child result is synthesized with its limits.")
            } else if body["messages"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m["tool_name"] == "delegate_tasks")
            {
                answer("The delegated review is complete.")
            } else {
                json!({"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"delegate_tasks","arguments":{"objective":"Inspect the Rust code","max_agents":1}}}]},"done":true})
            }
        },
        Duration::from_millis(10),
    );
    let dir = tempfile::tempdir().unwrap();
    let outcome = run(job(
        settings(&server),
        dir.path(),
        "Analyze this project using a team",
        true,
    ));
    assert!(outcome.errors.is_empty(), "{:?}", outcome.errors);
    assert_eq!(
        outcome.answers.last().unwrap(),
        "The delegated review is complete."
    );
    let chats = server.chats();
    assert_eq!(chats.len(), 5);
    assert!(chats.iter().any(|body| body["model"] == "coder:7b"));
    assert_eq!(server.max_active.load(Ordering::SeqCst), 1);
    assert!(chats.iter().all(|body| body["keep_alive"] == "0"));
}

#[test]
fn coded_review_with_text_only_tool_requests_retries_on_verified_coordinator() {
    let server = Server::new(
        |body| {
            if body["model"] == "coder:7b" {
                return answer(concat!(
                    r#"{"name":"project_tree","arguments":{"path":"."}}"#,
                    r#"{"name":"read_file","arguments":{"path":"main.rs"}}"#
                ));
            }
            assert_eq!(body["model"], "coordinator:4b");
            let messages = body["messages"].as_array().unwrap();
            if messages.iter().any(|message| message["role"] == "tool") {
                answer("I inspected main.rs using the actual read_file tool.")
            } else {
                assert!(messages.iter().any(|message| message["role"] == "system"
                    && message["content"]
                        .as_str()
                        .unwrap()
                        .contains("native tool_calls")));
                json!({"message":{"role":"assistant","tool_calls":[{"function":{"name":"read_file","arguments":{"path":"main.rs"}}}]} ,"done":true})
            }
        },
        Duration::ZERO,
    );
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("main.rs"), "fn main() {}\n").unwrap();
    let mut configured = settings(&server);
    let mut routing = config();
    // Reproduce a misconfigured coordinator preference that points back to the
    // coding model first. Runtime exclusion must still select the backup.
    routing.orchestrator = vec!["coder:7b".into(), "coordinator:4b".into()];
    configured.routing_json = serde_json::to_string(&routing).unwrap();
    let outcome = run(job(
        configured,
        tmp.path(),
        "Review the Rust code in main.rs",
        true,
    ));
    assert!(outcome.errors.is_empty(), "{:?}", outcome.errors);
    assert_eq!(
        outcome.models,
        ["coder:7b", "coordinator:4b", "coordinator:4b"]
    );
    assert_eq!(outcome.answers.len(), 1);
    assert!(
        outcome.answers[0].starts_with("I inspected main.rs using the actual read_file tool.")
            && outcome.answers[0].contains("`main.rs`: lines 1-1 of 1"),
        "{:?}",
        outcome.answers
    );
    assert_eq!(
        outcome.native_tool_calls, 1,
        "The retry must make an actual native read_file call"
    );
    let calls = server.chats();
    assert_eq!(calls.len(), 3);
    assert_eq!(calls[0]["model"], "coder:7b");
    assert_eq!(calls[1]["model"], "coordinator:4b");
    assert_eq!(calls[2]["model"], "coordinator:4b");
    assert!(calls[2]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .any(|message| message["role"] == "tool"));
}
