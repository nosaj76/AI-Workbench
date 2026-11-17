//! Protocol-level regression tests for the screenshot's prose-only CLI handoff.
//! The server is scripted; actual terminal calls still pass through ToolEngine,
//! approval, and the real PTY. No live model or installed ModernBASIC is assumed.

use super::*;
use serde_json::json;
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::Mutex,
    thread::JoinHandle,
    time::Instant,
};

struct MockOllama {
    endpoint: String,
    requests: Arc<Mutex<Vec<Value>>>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl MockOllama {
    fn start(mut response: impl FnMut(usize, &Value) -> Value + Send + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let log = requests.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let server_stop = stop.clone();
        let worker = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(30);
            let mut round = 0;
            while !server_stop.load(Ordering::Relaxed) {
                let (mut stream, _) = match listener.accept() {
                    Ok(connection) => connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < deadline, "Native terminal mock timed out");
                        std::thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    Err(error) => panic!("{error}"),
                };
                let (path, payload) = read_request(&mut stream);
                let body = match path.as_str() {
                    "/api/show" => json!({"capabilities":["completion","tools"],"details":{}}),
                    "/api/chat" => {
                        log.lock().unwrap().push(payload.clone());
                        let result = response(round, &payload);
                        round += 1;
                        result
                    }
                    _ => panic!("Unexpected API request: {path}"),
                };
                let body = format!("{body}\n");
                write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            }
        });
        Self {
            endpoint,
            requests,
            stop,
            worker: Some(worker),
        }
    }

    fn finish(mut self) -> Vec<Value> {
        self.stop.store(true, Ordering::Relaxed);
        self.worker.take().unwrap().join().unwrap();
        let requests = self.requests.lock().unwrap().clone();
        requests
    }
}

impl Drop for MockOllama {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn read_request(stream: &mut TcpStream) -> (String, Value) {
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    let mut bytes = Vec::new();
    let mut buffer = [0u8; 4096];
    loop {
        let count = stream.read(&mut buffer).unwrap();
        assert!(count > 0 && bytes.len() < 1_048_576);
        bytes.extend_from_slice(&buffer[..count]);
        let Some(end) = bytes
            .windows(4)
            .position(|part| part == b"\r\n\r\n")
            .map(|at| at + 4)
        else {
            continue;
        };
        let headers = String::from_utf8_lossy(&bytes[..end]);
        let length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().unwrap())
            })
            .unwrap_or(0);
        if bytes.len() >= end + length {
            let path = headers
                .lines()
                .next()
                .unwrap()
                .split_whitespace()
                .nth(1)
                .unwrap()
                .to_owned();
            return (
                path,
                serde_json::from_slice(&bytes[end..end + length]).unwrap(),
            );
        }
    }
}

fn answer(content: &str) -> Value {
    json!({"message":{"role":"assistant","content":content},"done":true})
}

fn call(name: &str, arguments: Value) -> Value {
    json!({"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":name,"arguments":arguments}}]},"done":true})
}

fn settings(endpoint: &str) -> Settings {
    Settings {
        base_url: endpoint.into(),
        model: "native-terminal-test".into(),
        native_tool_probe: false,
        automatic_memory: false,
        knowledge_notes: false,
        agents_enabled: false,
        code_example_search: false,
        ..Default::default()
    }
}

#[derive(Default)]
struct Transcript {
    complete: Vec<ChatMessage>,
    tools: Vec<ChatMessage>,
    superseded: Vec<String>,
    errors: Vec<String>,
    approvals: usize,
}

fn chat(job: ChatJob, approve: bool) -> Transcript {
    let (tx, rx) = mpsc::channel();
    spawn_chat(job, tx, Arc::new(AtomicBool::new(false)));
    let mut transcript = Transcript::default();
    loop {
        match rx
            .recv_timeout(Duration::from_secs(20))
            .expect("Chat did not finish")
        {
            ChatEvent::Approval { reply, .. } => {
                transcript.approvals += 1;
                reply.send(approve).unwrap();
            }
            ChatEvent::Complete(message) => transcript.complete.push(message),
            ChatEvent::Append(entry) => transcript.tools.push(entry.message),
            ChatEvent::Superseded(reason) => transcript.superseded.push(reason),
            ChatEvent::Error(error) => transcript.errors.push(error),
            ChatEvent::Finished => return transcript,
            _ => {}
        }
    }
}

fn job(endpoint: &str, user: &str, data: &std::path::Path, engine: Option<ToolEngine>) -> ChatJob {
    let settings = settings(endpoint);
    ChatJob {
        workspace: engine.as_ref().map(|engine| engine.root().to_path_buf()),
        model: settings.model.clone(),
        messages: vec![
            ChatMessage {
                role: "system".into(),
                content: settings.system_prompt.clone(),
                ..Default::default()
            },
            ChatMessage {
                role: "user".into(),
                content: user.into(),
                ..Default::default()
            },
        ],
        settings,
        engine,
        retrieval_path: data.join("retrieval.sqlite3"),
    }
}

#[cfg(unix)]
fn fixture(directory: &std::path::Path) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let program = directory.join("modernbasic");
    std::fs::write(&program, "#!/bin/sh\nprintf 'started\\n' > started.txt\nprintf 'ModernBASIC fixture\\nREADY> '\nwhile IFS= read -r line; do printf 'received: %s\\nREADY> ' \"$line\"; done\n").unwrap();
    std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
    program
}

fn engine(workspace: &std::path::Path, data: &std::path::Path) -> ToolEngine {
    ToolEngine::with_backup_dir(
        crate::files::Workspace::new(workspace.to_path_buf()).unwrap(),
        data.join("backups"),
    )
}

#[test]
#[cfg(unix)]
fn prose_handoff_is_discarded_then_native_launch_reaches_real_approval_and_pty() {
    let workspace = tempfile::tempdir().unwrap();
    let data = tempfile::tempdir().unwrap();
    let program = fixture(workspace.path());
    let server = MockOllama::start(move |round, payload| {
        assert!(payload["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|definition| definition["function"]["name"] == "terminal_open"));
        match round {
            0 => answer("Please open your CLI and type modernbasic."),
            1 => {
                assert!(payload["messages"]
                    .to_string()
                    .contains("Submit pending input with Enter"));
                assert!(!payload["messages"]
                    .to_string()
                    .contains("Please open your CLI"));
                call("terminal_open", json!({"program":program,"wait_ms":750}))
            }
            2 => {
                assert!(
                    !payload["messages"]
                        .to_string()
                        .contains("Submit pending input with Enter"),
                    "Recovery must last for one request only"
                );
                assert!(payload["messages"]
                    .to_string()
                    .contains("ModernBASIC fixture"));
                answer("The terminal returned the ModernBASIC fixture startup output.")
            }
            _ => panic!("Unexpected extra model request"),
        }
    });
    let engine = engine(workspace.path(), data.path());
    let terminals = engine.terminal_manager();
    let transcript = chat(
        job(
            &server.endpoint,
            "use modern basic",
            data.path(),
            Some(engine),
        ),
        true,
    );
    let requests = server.finish();
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(requests.len(), 3);
    assert_eq!(transcript.approvals, 1);
    assert_eq!(transcript.superseded.len(), 1);
    assert!(transcript
        .tools
        .iter()
        .any(|message| message.tool_name.as_deref() == Some("terminal_open")));
    assert!(transcript
        .complete
        .iter()
        .all(|message| !message.content.contains("Please open your CLI")));
    assert_eq!(
        std::fs::read_to_string(workspace.path().join("started.txt")).unwrap(),
        "started\n"
    );
    assert_eq!(terminals.list().len(), 1);
    assert!(terminals.list()[0].alive);
    terminals.close(&terminals.list()[0].id).unwrap();
}

#[test]
fn repeated_prose_handoff_stops_once_with_honest_incomplete_status() {
    let workspace = tempfile::tempdir().unwrap();
    let data = tempfile::tempdir().unwrap();
    let server = MockOllama::start(|round, _| {
        assert!(round < 2, "The repair must be bounded to one retry");
        answer("Please open your CLI and type modernbasic.")
    });
    let engine = engine(workspace.path(), data.path());
    let terminals = engine.terminal_manager();
    let transcript = chat(
        job(
            &server.endpoint,
            "use modern basic",
            data.path(),
            Some(engine),
        ),
        true,
    );
    let requests = server.finish();
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(requests.len(), 2);
    assert_eq!(transcript.approvals, 0);
    assert!(terminals.list().is_empty());
    assert_eq!(transcript.superseded.len(), 2);
    assert!(transcript
        .complete
        .last()
        .unwrap()
        .content
        .contains("CLI task incomplete"));
    assert!(transcript
        .complete
        .iter()
        .all(|message| !message.content.contains("Please open your CLI")));
}

#[test]
#[cfg(unix)]
fn opening_bash_alone_does_not_complete_the_requested_modernbasic_launch() {
    let workspace = tempfile::tempdir().unwrap();
    let data = tempfile::tempdir().unwrap();
    let program = fixture(workspace.path());
    let server = MockOllama::start(move |round, payload| match round {
        0 => call(
            "terminal_open",
            json!({"program":"bash","args":["--noprofile","--norc","-i"],"wait_ms":250}),
        ),
        1 => answer("Bash is ready. Please type modernbasic yourself."),
        2 => {
            let output = payload["messages"]
                .as_array()
                .unwrap()
                .iter()
                .rev()
                .find(|message| message["role"] == "tool")
                .unwrap();
            let output: Value = serde_json::from_str(output["content"].as_str().unwrap()).unwrap();
            call(
                "terminal_write",
                json!({"session_id":output["session_id"],"text":program.to_str().unwrap(),"wait_ms":750}),
            )
        }
        3 => answer("The interpreter startup output was observed in the same Bash session."),
        _ => panic!("Unexpected extra model request"),
    });
    let engine = engine(workspace.path(), data.path());
    let terminals = engine.terminal_manager();
    let transcript = chat(
        job(
            &server.endpoint,
            "use modernbasic",
            data.path(),
            Some(engine),
        ),
        true,
    );
    let requests = server.finish();
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(requests.len(), 4);
    assert_eq!(transcript.approvals, 2);
    assert_eq!(transcript.superseded.len(), 1);
    assert_eq!(
        std::fs::read_to_string(workspace.path().join("started.txt")).unwrap(),
        "started\n"
    );
    assert_eq!(
        terminals.list().len(),
        1,
        "The existing Bash session must be reused"
    );
    terminals.close(&terminals.list()[0].id).unwrap();
}

#[test]
fn denied_launch_is_reported_without_a_repair_or_a_second_approval() {
    let workspace = tempfile::tempdir().unwrap();
    let data = tempfile::tempdir().unwrap();
    let server = MockOllama::start(|round, payload| match round {
        0 => call("terminal_open", json!({"program":"modernbasic"})),
        1 => {
            assert!(payload["messages"].to_string().contains("denied"));
            answer("The launch was denied, so it was not run.")
        }
        _ => panic!("Denied actions must not be retried"),
    });
    let engine = engine(workspace.path(), data.path());
    let terminals = engine.terminal_manager();
    let transcript = chat(
        job(
            &server.endpoint,
            "use modern basic",
            data.path(),
            Some(engine),
        ),
        false,
    );
    let requests = server.finish();
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(requests.len(), 2);
    assert_eq!(transcript.approvals, 1);
    assert!(transcript.superseded.is_empty());
    assert!(terminals.list().is_empty());
}

#[test]
fn how_to_question_does_not_require_a_launch_or_a_workspace() {
    let data = tempfile::tempdir().unwrap();
    let server = MockOllama::start(|round, _| {
        assert_eq!(round, 0);
        answer("To use ModernBASIC, launch its installed executable and read help.")
    });
    let transcript = chat(
        job(
            &server.endpoint,
            "How do I use modern basic?",
            data.path(),
            None,
        ),
        true,
    );
    assert_eq!(server.finish().len(), 1);
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(transcript.approvals, 0);
    assert!(transcript.superseded.is_empty());
    assert!(!data.path().join("workspaces").exists());
}

#[test]
fn explicit_cli_setup_failures_are_reported_before_contacting_a_model() {
    let data = tempfile::tempdir().unwrap();
    for tools_enabled in [false, true] {
        let mut request = job("http://127.0.0.1:1", "use modern basic", data.path(), None);
        request.settings.tools_enabled = tools_enabled;
        let transcript = chat(request, true);
        assert_eq!(transcript.errors.len(), 1);
        let expected = if tools_enabled {
            "Open folder"
        } else {
            "Tools are disabled"
        };
        assert!(
            transcript.errors[0].contains(expected),
            "{:?}",
            transcript.errors
        );
        assert!(transcript.complete.is_empty());
        assert_eq!(transcript.approvals, 0);
        assert!(!data.path().join("workspaces").exists());
    }
}
