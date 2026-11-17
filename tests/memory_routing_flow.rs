//! Full chat-worker checks for automatic memory and capability-aware routing.
use ollama_workbench::{
    memory::MemoryStore,
    ollama::ChatMessage,
    runtime::{spawn_chat, ChatEvent, ChatJob},
    settings::Settings,
};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

struct Server {
    url: String,
    requests: Arc<Mutex<Vec<(String, Value)>>>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}
impl Server {
    fn new(replies: Vec<Value>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(vec![]));
        let seen = requests.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let worker = thread::spawn(move || {
            let mut replies = VecDeque::from(replies);
            while !stopped.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut s, _)) => {
                        s.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
                        let (path, body) = request(&mut s);
                        seen.lock().unwrap().push((path.clone(), body.clone()));
                        let response = match path.as_str() {
                            "/api/tags" => {
                                json!({"models":[{"name":"general:7b","size":4_000_000_000_u64},{"name":"qwen-coder:7b","size":4_000_000_000_u64},{"name":"local-renamed","size":100}]})
                            }
                            "/api/ps" => json!({"models":[]}),
                            "/api/show" => {
                                if body["model"] == "local-renamed" {
                                    json!({"capabilities":["completion","tools"],"remote_host":"https://ollama.com","remote_model":"remote"})
                                } else {
                                    json!({"capabilities":["completion","tools"]})
                                }
                            }
                            "/api/chat" => {
                                replies.pop_front().expect("Unexpected inference request")
                            }
                            _ => panic!("Unexpected endpoint {path}"),
                        };
                        let data = format!("{}\n", serde_json::to_string(&response).unwrap());
                        let header=format!("HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",data.len());
                        s.write_all(header.as_bytes()).unwrap();
                        s.write_all(data.as_bytes()).unwrap();
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2))
                    }
                    Err(e) => panic!("{e}"),
                }
            }
        });
        Self {
            url,
            requests,
            stop,
            thread: Some(worker),
        }
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            t.join().unwrap();
        }
    }
}
fn request(socket: &mut TcpStream) -> (String, Value) {
    let mut all = vec![];
    let mut b = [0u8; 4096];
    let end = loop {
        let n = socket.read(&mut b).unwrap();
        assert!(n > 0);
        all.extend_from_slice(&b[..n]);
        if let Some(i) = all.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        assert!(all.len() < 128_000);
    };
    let header = String::from_utf8_lossy(&all[..end]);
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
        .find_map(|l| {
            let (k, v) = l.split_once(':')?;
            if k.eq_ignore_ascii_case("content-length") {
                v.trim().parse::<usize>().ok()
            } else {
                None
            }
        })
        .unwrap_or(0);
    while all.len() < end + length {
        let n = socket.read(&mut b).unwrap();
        assert!(n > 0);
        all.extend_from_slice(&b[..n]);
    }
    let value = if length == 0 {
        Value::Null
    } else {
        serde_json::from_slice(&all[end..end + length]).unwrap()
    };
    (path, value)
}
fn answer(text: &str) -> Value {
    json!({"message":{"role":"assistant","content":text},"done":true,"eval_count":10,"eval_duration":1_000_000_000_u64})
}
fn run(settings: Settings, dir: &std::path::Path, prompt: &str) {
    let (tx, rx) = mpsc::channel();
    spawn_chat(
        ChatJob {
            model: "general:7b".into(),
            settings,
            messages: vec![
                ChatMessage {
                    role: "system".into(),
                    content: "Help the user".into(),
                    ..Default::default()
                },
                ChatMessage {
                    role: "user".into(),
                    content: prompt.into(),
                    ..Default::default()
                },
            ],
            workspace: None,
            engine: None,
            retrieval_path: dir.join("knowledge.sqlite3"),
        },
        tx,
        Arc::new(AtomicBool::new(false)),
    );
    let end = Instant::now() + Duration::from_secs(10);
    loop {
        assert!(Instant::now() < end, "Worker timed out");
        match rx
            .recv_timeout(Duration::from_secs(3))
            .expect("Worker did not finish")
        {
            ChatEvent::Error(e) => panic!("{e}"),
            ChatEvent::Approval { .. } => panic!("Unexpected approval"),
            ChatEvent::Finished => break,
            _ => {}
        }
    }
}

#[test]
fn memory_is_automatically_learned_and_recalled_in_a_separate_chat() {
    let server = Server::new(vec![
        answer("I will use Rust."),
        answer(
            r#"{"memories":[{"key":"desktop language","fact":"Prefers Rust for desktop applications","quote":"I prefer Rust for desktop applications","confidence":0.95}]}"#,
        ),
        answer("You prefer Rust."),
        answer(r#"{"memories":[]}"#),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let settings = Settings {
        native_tool_probe: false,
        base_url: server.url.clone(),
        tools_enabled: false,
        agents_enabled: false,
        automatic_memory: true,
        ..Default::default()
    };
    run(
        settings.clone(),
        dir.path(),
        "Remember my language preference: I prefer Rust for desktop applications.",
    );
    let memories = MemoryStore::open(dir.path())
        .unwrap()
        .list(None, "", 20)
        .unwrap();
    assert_eq!(memories.len(), 1);
    assert_eq!(memories[0].quote, "I prefer Rust for desktop applications");
    run(
        settings,
        dir.path(),
        "Which language should this desktop application use?",
    );
    let requests = server.requests.lock().unwrap();
    let chats = requests
        .iter()
        .filter(|(path, _)| path == "/api/chat")
        .collect::<Vec<_>>();
    assert_eq!(chats.len(), 4);
    let second = chats[2].1["messages"].as_array().unwrap();
    assert!(second.iter().any(|m| m["role"] == "system"
        && m["content"]
            .as_str()
            .unwrap_or("")
            .contains("Prefers Rust for desktop applications")));
    assert!(
        MemoryStore::open(dir.path())
            .unwrap()
            .list(None, "", 20)
            .unwrap()[0]
            .used
            > 0
    );
}
#[test]
fn automatic_router_ignores_a_preferred_cloud_alias_before_sending_the_prompt() {
    let server = Server::new(vec![answer("The Rust review is complete.")]);
    let dir = tempfile::tempdir().unwrap();
    let settings = Settings {
        native_tool_probe: false,
        base_url: server.url.clone(),
        model_routing: true,
        routing_json: r#"{"coding":["local-renamed","qwen-coder:7b"]}"#.into(),
        automatic_memory: false,
        agents_enabled: false,
        tools_enabled: false,
        ..Default::default()
    };
    run(
        settings,
        dir.path(),
        "Review this Rust code for error handling.",
    );
    let requests = server.requests.lock().unwrap();
    let chats = requests
        .iter()
        .filter(|(path, _)| path == "/api/chat")
        .collect::<Vec<_>>();
    assert_eq!(chats.len(), 1);
    assert_eq!(chats[0].1["model"], "qwen-coder:7b");
    let route: Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("last-route.json")).unwrap())
            .unwrap();
    assert_eq!(route["model"], "qwen-coder:7b");
    assert!(route["rejected"]
        .as_array()
        .unwrap()
        .iter()
        .any(|s| s.as_str().unwrap().contains("cloud")));
}
