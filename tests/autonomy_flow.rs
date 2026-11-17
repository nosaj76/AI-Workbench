//! Full planner/worker/review/research flows against actual loopback protocol
//! fixtures and temporary local files. No real-model quality claims are made.
use ollama_workbench::{
    agents::{self, AgentEvent, AgentRequest},
    files::Workspace,
    ollama::ChatMessage,
    research::{self, ResearchRequest},
    review_queue::{self, ReviewQueue, ReviewRequest},
    runtime::{self, ChatEvent, ChatJob},
    settings::Settings,
};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

type RequestHook = Box<dyn FnMut(&Value) + Send>;

struct MockOllama {
    url: String,
    requests: Arc<Mutex<Vec<Value>>>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl MockOllama {
    fn new(responses: Vec<Value>) -> Self {
        Self::with_hook(responses, None)
    }
    fn with_hook(responses: Vec<Value>, mut hook: Option<RequestHook>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = requests.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let worker = thread::spawn(move || {
            let mut responses = VecDeque::from(responses);
            while !stopping.load(Ordering::Relaxed) {
                let mut socket = match listener.accept() {
                    Ok((socket, _)) => socket,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(e) => panic!("mock accept: {e}"),
                };
                socket
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let (path, body) = read_request(&mut socket);
                let response = match path.as_str() {
                    "/api/show" => {
                        json!({"capabilities":["completion","tools"],"details":{"family":"fixture"}})
                    }
                    "/api/chat" => {
                        seen.lock().unwrap().push(body.clone());
                        if let Some(hook) = &mut hook {
                            hook(&body);
                        }
                        responses.pop_front().unwrap_or_else(
                            || json!({"error":"Unexpected inference beyond test budget"}),
                        )
                    }
                    other => json!({"error":format!("Unexpected fixture request {other}")}),
                };
                let body = format!("{response}\n");
                let _=socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).as_bytes());
            }
        });
        Self {
            url,
            requests,
            stop,
            worker: Some(worker),
        }
    }
    fn chats(&self) -> Vec<Value> {
        self.requests.lock().unwrap().clone()
    }
}

impl Drop for MockOllama {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap();
        }
    }
}

struct WebSources {
    url: String,
    requests: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}
impl WebSources {
    fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let url = format!("http://127.0.0.1:{port}");
        let base = url.clone();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = requests.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let worker = thread::spawn(move || {
            while !stopping.load(Ordering::Relaxed) {
                let mut socket = match listener.accept() {
                    Ok((socket, _)) => socket,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(e) => panic!("web fixture accept: {e}"),
                };
                socket
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let (path, _) = read_request(&mut socket);
                seen.lock().unwrap().push(path.clone());
                let (headers, body) = if path.starts_with("/search?") {
                    ("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n".to_string(),json!({"results":[{"title":"Reference","url":format!("{base}/source"),"content":"Discovery hint only"},{"title":"Duplicate fragment","url":format!("{base}/source#section"),"content":"Not another source"},{"title":"Unreadable","url":format!("{base}/binary"),"content":"Do not cite this snippet as evidence"}]}).to_string())
                } else if path == "/source" {
                    ("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n".to_string(),"<html><head><title>Reference fixture</title></head><body><main>The reference specifies a twelve-day orbit.<script>UNTRUSTED_SCRIPT_DO_NOT_USE</script></main></body></html>".to_string())
                } else if path == "/redirect" {
                    (
                        format!(
                            "HTTP/1.1 302 Found\r\nLocation: http://localhost:{port}/source\r\n"
                        ),
                        String::new(),
                    )
                } else {
                    (
                        "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\n".to_string(),
                        "not-readable-image-bytes".to_string(),
                    )
                };
                let _ = socket.write_all(
                    format!(
                        "{headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                );
            }
        });
        Self {
            url,
            requests,
            stop,
            worker: Some(worker),
        }
    }
}
impl Drop for WebSources {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap();
        }
    }
}

fn conversation_len(request: &Value) -> usize {
    request["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| {
            // Request-only project context is not part of the worker's own conversation.
            !m["content"].as_str().is_some_and(|c| {
                c.starts_with("<project_files>")
                    || c.starts_with("<repo_map>")
                    || c.starts_with("<project_instructions")
            })
        })
        .count()
}

fn read_request(socket: &mut TcpStream) -> (String, Value) {
    let mut bytes = Vec::new();
    let mut buffer = [0u8; 4096];
    loop {
        let count = socket.read(&mut buffer).unwrap();
        assert!(count > 0);
        bytes.extend_from_slice(&buffer[..count]);
        if let Some(end) = bytes.windows(4).position(|b| b == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..end]);
            let length = headers
                .lines()
                .find_map(|line| {
                    line.split_once(':')
                        .filter(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                        .map(|(_, value)| value.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0);
            if bytes.len() >= end + 4 + length {
                let path = headers
                    .lines()
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .nth(1)
                    .unwrap()
                    .to_string();
                return (
                    path,
                    if length == 0 {
                        Value::Null
                    } else {
                        serde_json::from_slice(&bytes[end + 4..end + 4 + length]).unwrap()
                    },
                );
            }
        }
    }
}

fn answer(text: &str) -> Value {
    json!({"message":{"role":"assistant","content":text},"done":true,"eval_count":8})
}
fn tool(name: &str, args: Value) -> Value {
    json!({"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":name,"arguments":args}}]},"done":true})
}
fn settings(server: &MockOllama) -> Settings {
    Settings {
        native_tool_probe: false,
        base_url: server.url.clone(),
        model: "fixture".into(),
        automatic_memory: false,
        agents_enabled: false,
        tts_program: String::new(),
        stt_program: String::new(),
        ..Default::default()
    }
}
fn last_prompt(request: &Value) -> &str {
    request["messages"].as_array().unwrap().last().unwrap()["content"]
        .as_str()
        .unwrap()
}

fn paths(tmp: &Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let project = tmp.join("project");
    let data = tmp.join("data");
    fs::create_dir(&project).unwrap();
    fs::create_dir(&data).unwrap();
    (project, data)
}

#[test]
fn autonomous_plan_uses_independent_tool_contexts_and_synthesizes_real_child_results() {
    let tmp = tempfile::tempdir().unwrap();
    let (project, data) = paths(tmp.path());
    fs::write(
        project.join("logic.rs"),
        "pub fn valid() -> bool { true }\n",
    )
    .unwrap();
    fs::write(project.join("README.md"), "Documented interface\n").unwrap();
    let server = MockOllama::new(vec![
        answer(
            r#"{"tasks":[{"title":"Source","prompt":"Inspect logic.rs","model":"fixture"},{"title":"Documentation","prompt":"Inspect README.md","model":"fixture"}]}"#,
        ),
        tool("read_file", json!({"path":"logic.rs"})),
        answer("The source returns true at logic.rs:1."),
        tool("read_file", json!({"path":"README.md"})),
        answer("README.md:1 documents the interface."),
        answer("The source and documentation agree; inspected logic.rs:1 and README.md:1."),
    ]);
    let mut request = AgentRequest::new(
        settings(&server),
        Some(project.clone()),
        data.clone(),
        "Review the function and its documentation".into(),
        "fixture".into(),
    );
    request.max_agents = 2;
    request.concurrency = 1;
    let (tx, rx) = mpsc::channel();
    let run = agents::run(request, tx, Arc::new(AtomicBool::new(false))).unwrap();
    assert_eq!(run.state, "done");
    assert_eq!(run.tasks.len(), 2);
    assert!(run.tasks.iter().all(|t| t.state == "done"));
    assert_eq!(
        agents::load_run(&data, &run.id).unwrap().tasks[0].answer,
        run.tasks[0].answer
    );
    let chats = server.chats();
    assert_eq!(chats.len(), 6);
    // Each worker sees only its system prompt and its own task (plus the
    // request-only project context), never a sibling's context.
    assert_eq!(conversation_len(&chats[1]), 2);
    assert_eq!(conversation_len(&chats[3]), 2);
    assert!(!last_prompt(&chats[3]).contains("The source returns true"));
    assert!(last_prompt(&chats[5]).contains("The source returns true"));
    assert!(last_prompt(&chats[5]).contains("documents the interface"));
    assert!(rx
        .try_iter()
        .any(|event| matches!(event,AgentEvent::Task{state,..}if state=="done")));
}

#[test]
fn read_only_subagent_cannot_turn_a_model_requested_write_into_authority() {
    let tmp = tempfile::tempdir().unwrap();
    let (project, data) = paths(tmp.path());
    fs::write(project.join("file.txt"), "original").unwrap();
    let server = MockOllama::new(vec![
        answer(r#"{"tasks":[{"title":"Inspect","prompt":"Inspect file.txt"}]}"#),
        tool(
            "write_file",
            json!({"path":"file.txt","content":"unauthorized"}),
        ),
        answer("The requested write was denied; no edit was made."),
        answer("No edit was made because this team was read-only."),
    ]);
    let request = AgentRequest::new(
        settings(&server),
        Some(project.clone()),
        data,
        "Inspect the file".into(),
        "fixture".into(),
    );
    let (tx, rx) = mpsc::channel();
    let run = agents::run(request, tx, Arc::new(AtomicBool::new(false))).unwrap();
    assert_eq!(run.state, "done");
    assert_eq!(run.tasks[0].denied_actions, 1);
    assert_eq!(
        fs::read_to_string(project.join("file.txt")).unwrap(),
        "original"
    );
    assert!(!rx
        .try_iter()
        .any(|event| matches!(event, AgentEvent::Approval { .. })));
    let chats = server.chats();
    let result = chats[2]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "tool")
        .unwrap();
    let result: Value = serde_json::from_str(result["content"].as_str().unwrap()).unwrap();
    assert_eq!(result["denied"], true);
}

#[test]
fn cancelling_a_team_waiting_for_approval_keeps_files_and_saved_resume_state() {
    let tmp = tempfile::tempdir().unwrap();
    let (project, data) = paths(tmp.path());
    fs::write(project.join("file.txt"), "original").unwrap();
    let server = MockOllama::new(vec![
        answer(r#"{"tasks":[{"title":"Edit","prompt":"Propose an edit"}]}"#),
        tool("write_file", json!({"path":"file.txt","content":"changed"})),
    ]);
    let mut request = AgentRequest::new(
        settings(&server),
        Some(project.clone()),
        data.clone(),
        "Propose a file edit".into(),
        "fixture".into(),
    );
    request.allow_actions = true;
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let stopping = cancel.clone();
    let worker = thread::spawn(move || agents::run(request, tx, stopping).unwrap());
    loop {
        if let AgentEvent::Approval { reply, .. } = rx.recv_timeout(Duration::from_secs(5)).unwrap()
        {
            cancel.store(true, Ordering::Relaxed);
            drop(reply);
            break;
        }
    }
    let run = worker.join().unwrap();
    assert_eq!(run.state, "cancelled");
    assert_eq!(
        fs::read_to_string(project.join("file.txt")).unwrap(),
        "original"
    );
    assert_eq!(agents::load_run(&data, &run.id).unwrap().state, "cancelled");
    assert_eq!(server.chats().len(), 2);
}

#[test]
fn cancelling_parent_chat_unblocks_active_delegation_and_commits_cancelled_checkpoint() {
    let tmp = tempfile::tempdir().unwrap();
    let (project, data) = paths(tmp.path());
    fs::write(project.join("file.txt"), "original").unwrap();
    let (pending_tx, pending_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let mut chat_count = 0;
    let server = MockOllama::with_hook(
        vec![
            tool(
                "delegate_tasks",
                json!({"objective":"Inspect local code","max_agents":1}),
            ),
            answer(r#"{"tasks":[{"title":"Inspection","prompt":"Inspect file.txt"}]}"#),
            answer("This response is deliberately held until after cancellation."),
        ],
        Some(Box::new(move |_| {
            chat_count += 1;
            if chat_count == 3 {
                pending_tx.send(()).unwrap();
                let _ = release_rx.recv_timeout(Duration::from_secs(5));
            }
        })),
    );
    let mut configuration = settings(&server);
    configuration.agents_enabled = true;
    configuration.retrieval_enabled = false;
    let cancel = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    runtime::spawn_chat(
        ChatJob {
            settings: configuration,
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "Use one independent child to inspect the workspace.".into(),
                ..Default::default()
            }],
            workspace: Some(project.clone()),
            model: "fixture".into(),
            engine: None,
            retrieval_path: data.join("knowledge.sqlite3"),
        },
        tx,
        cancel.clone(),
    );
    pending_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    cancel.store(true, Ordering::Relaxed);
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .expect("Parent chat did not finish after cancellation");
        match rx.recv_timeout(remaining).unwrap() {
            ChatEvent::Finished => break,
            ChatEvent::Approval { reply, .. } => {
                let _ = reply.send(false);
                panic!("Automatic delegation must not forward child side-effect approvals");
            }
            ChatEvent::Error(error) => panic!("Unexpected parent error: {error}"),
            _ => {}
        }
    }
    // Parent completion occurs while the fixture still holds the child's HTTP
    // response, proving the worker/forwarder receivers were not left waiting.
    release_tx.send(()).unwrap();
    let runs = agents::list_runs(&data, 10).unwrap();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].state, "cancelled");
    assert_eq!(runs[0].tasks.len(), 1);
    assert_eq!(runs[0].tasks[0].state, "cancelled");
    assert_eq!(server.chats().len(), 3);
    assert_eq!(
        fs::read_to_string(project.join("file.txt")).unwrap(),
        "original"
    );
}

#[test]
fn comprehensive_review_resumes_the_remaining_chunks_without_repeating_committed_work() {
    let tmp = tempfile::tempdir().unwrap();
    let (project, data) = paths(tmp.path());
    let source = (1..=160)
        .map(|i| format!("line {i}: concrete local content\n"))
        .collect::<String>();
    fs::write(project.join("notes.txt"), &source).unwrap();
    let workspace = Workspace::new(project.clone()).unwrap();
    let mut queue = ReviewQueue::open(data.join("review.sqlite3")).unwrap();
    queue
        .refresh(&workspace, 500, &AtomicBool::new(false))
        .unwrap();
    let items = queue.items(&workspace, None, 1000).unwrap();
    assert_eq!(
        items.iter().map(|i| i.text.as_str()).collect::<String>(),
        source
    );
    let total = items.len();
    assert!(total > 2);
    let server = MockOllama::new(
        (0..total)
            .map(|i| answer(&format!("Reviewed this exact queued excerpt, result {i}.")))
            .collect(),
    );
    let mut request =
        ReviewRequest::new(settings(&server), project, data.clone(), "fixture".into());
    request.chunk_chars = 500;
    request.max_items = 2;
    request.synthesize = false;
    let (tx, _) = mpsc::channel();
    let first =
        review_queue::run_review(request.clone(), tx, Arc::new(AtomicBool::new(false))).unwrap();
    assert_eq!(first.processed, 2);
    assert_eq!(first.coverage.done, 2);
    assert_eq!(first.state, "partial");
    request.max_items = 0;
    request.refresh_first = false;
    let (tx, _) = mpsc::channel();
    let second = review_queue::run_review(request, tx, Arc::new(AtomicBool::new(false))).unwrap();
    assert_eq!(second.processed, total - 2);
    assert_eq!(second.coverage.done, total);
    assert!(second.coverage.complete());
    assert_eq!(second.state, "done");
    assert_eq!(server.chats().len(), total);
    assert!(fs::read_to_string(second.full_report_path)
        .unwrap()
        .contains("result 0"));
}

#[test]
fn source_change_during_inference_invalidates_the_review_instead_of_claiming_completion() {
    let tmp = tempfile::tempdir().unwrap();
    let (project, data) = paths(tmp.path());
    let file = project.join("changing.txt");
    fs::write(&file, "old text").unwrap();
    let mut changed = false;
    let server = MockOllama::with_hook(
        vec![answer("Review of the old snapshot")],
        Some(Box::new(move |_| {
            if !changed {
                fs::write(&file, "new outside edit").unwrap();
                changed = true;
            }
        })),
    );
    let mut request = ReviewRequest::new(settings(&server), project, data, "fixture".into());
    request.synthesize = false;
    let (tx, _) = mpsc::channel();
    let run = review_queue::run_review(request, tx, Arc::new(AtomicBool::new(false))).unwrap();
    assert_eq!(run.coverage.stale_files, 1);
    assert_eq!(run.coverage.done, 0);
    assert!(!run.coverage.complete());
    assert_eq!(run.state, "partial");
}

#[test]
fn iterative_local_research_retains_real_evidence_and_validates_final_citations() {
    let tmp = tempfile::tempdir().unwrap();
    let (project, data) = paths(tmp.path());
    fs::write(
        project.join("facts.md"),
        "The fictional satellite has an orbit period of twelve days.\n",
    )
    .unwrap();
    let server=MockOllama::new(vec![
        answer(r#"{"queries":["orbit"],"done":false,"reason":"Find the measured period"}"#),
        answer("The local fictional scenario specifies a twelve-day orbit [S1]."),
        answer(r#"{"queries":[],"done":true,"reason":"The local source answers the narrow question"}"#),
        answer("The fictional satellite's orbit period is twelve days [S1]. A fabricated source [S99] is unsupported."),
    ]);
    let request = ResearchRequest::new(
        settings(&server),
        Some(project),
        data.clone(),
        "What orbit period does the local scenario specify?".into(),
        "fixture".into(),
    );
    let (tx, _) = mpsc::channel();
    let run = research::run_research(request, tx, Arc::new(AtomicBool::new(false))).unwrap();
    assert_eq!(run.state, "done");
    assert_eq!(run.sources.len(), 1);
    assert_eq!(run.sources[0].path.as_deref(), Some("facts.md"));
    assert!(run.sources[0].content.contains("twelve days"));
    assert!(run.answer.contains("[S1]"));
    assert!(!run.answer.contains("[S99]"));
    assert!(run.warnings.iter().any(|w| w.contains("S99")));
    assert_eq!(
        research::load_run(&data, &run.id).unwrap().sources[0].content_sha256,
        run.sources[0].content_sha256
    );
    assert_eq!(server.chats().len(), 4);
}

#[test]
fn web_research_fetches_real_pages_deduplicates_urls_and_does_not_cite_search_only_results() {
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("data");
    let web = WebSources::new();
    let server = MockOllama::new(vec![
        answer(r#"{"queries":["orbit reference"],"done":false}"#),
        answer("The fetched reference specifies twelve days [S1]."),
        answer(
            r#"{"queries":[],"done":true,"reason":"The fetched reference answers the question"}"#,
        ),
        answer("The reference specifies a twelve-day orbit [S1]."),
    ]);
    let mut config = settings(&server);
    config.network_tools = true;
    config.searxng_url = web.url.clone();
    let mut request = ResearchRequest::new(
        config,
        None,
        data,
        "What orbit period does the reference specify?".into(),
        "fixture".into(),
    );
    request.web_enabled = true;
    request.allowed_domains = vec!["127.0.0.1".into()];
    let (tx, _) = mpsc::channel();
    let run = research::run_research(request, tx, Arc::new(AtomicBool::new(false))).unwrap();
    assert_eq!(run.state, "done");
    assert_eq!(run.sources.len(), 1);
    assert_eq!(run.sources[0].url, format!("{}/source", web.url));
    assert!(run.sources[0].content.contains("twelve-day"));
    assert!(!run.sources[0].content.contains("UNTRUSTED_SCRIPT"));
    assert!(run
        .warnings
        .iter()
        .any(|w| w.contains("not cited") && w.contains("binary")));
    let requests = web.requests.lock().unwrap();
    assert_eq!(
        requests.iter().filter(|p| p.as_str() == "/source").count(),
        1
    );
    assert_eq!(
        requests
            .iter()
            .filter(|p| p.starts_with("/search?"))
            .count(),
        1
    );
    assert_eq!(server.chats().len(), 4);
}

#[test]
fn research_domain_restriction_blocks_redirect_before_the_other_host_is_contacted() {
    let tmp = tempfile::tempdir().unwrap();
    let web = WebSources::new();
    let server = MockOllama::new(vec![answer(r#"{"queries":[],"done":true}"#)]);
    let mut config = settings(&server);
    config.network_tools = true;
    let mut request = ResearchRequest::new(
        config,
        None,
        tmp.path().join("data"),
        "Read the permitted source".into(),
        "fixture".into(),
    );
    request.web_enabled = true;
    request.max_rounds = 1;
    request.seed_urls = vec![format!("{}/redirect", web.url)];
    request.allowed_domains = vec!["127.0.0.1".into()];
    let (tx, _) = mpsc::channel();
    let run = research::run_research(request, tx, Arc::new(AtomicBool::new(false))).unwrap();
    assert_eq!(run.state, "error");
    assert!(run.sources.is_empty());
    assert_eq!(*web.requests.lock().unwrap(), vec!["/redirect"]);
}
