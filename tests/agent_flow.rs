//! End-to-end worker tests use real TCP and ordinary temporary workspace files.
//! No model weights or external network are required.
use ollama_workbench::{
    files::Workspace,
    ollama::ChatMessage,
    runtime::{self, ChatEvent, ChatJob},
    settings::Settings,
    tools::{PendingAction, ToolEngine},
};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::Duration,
};

struct MockOllama {
    url: String,
    requests: Arc<Mutex<Vec<(String, Value)>>>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl MockOllama {
    fn new(responses: Vec<Value>) -> Self {
        Self::with_metadata(
            responses,
            json!({"capabilities":["completion","tools"],"details":{"family":"test"}}),
        )
    }

    fn with_metadata(responses: Vec<Value>, metadata: Value) -> Self {
        Self::with_tags(responses, metadata, &["test"])
    }

    fn with_tags(responses: Vec<Value>, metadata: Value, tags: &[&str]) -> Self {
        let listed = json!({"models":tags.iter().map(|name| {
            json!({"name":name,"size":1_000_000,"details":{"family":"test"}})
        }).collect::<Vec<_>>()});
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let observed = requests.clone();
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
                socket
                    .set_write_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                // A bare connect/close (e.g. a reachability probe) carries no
                // request; real servers simply drop it, and so does the mock.
                let Some((path, request)) = read_request(&mut socket) else {
                    continue;
                };
                observed.lock().unwrap().push((path.clone(), request));
                let response = match path.as_str() {
                    "/api/chat" => responses
                        .pop_front()
                        .unwrap_or_else(|| json!({"error":"Unexpected extra inference request"})),
                    "/api/show" => metadata.clone(),
                    "/api/tags" => listed.clone(),
                    path if path.starts_with("/search?") => {
                        if path.contains("provider-failure") {
                            json!({"error":"synthetic search provider failure"})
                        } else if path.contains("loopback-fixture") {
                            json!({"results":[{
                                "title":"Local fixture", "url":"https://127.0.0.1/ai",
                                "content":"A result for the URL scoping test."
                            }]})
                        } else if path.contains("kernel-notes") {
                            json!({"results":[{
                                "title":"Kernel configuration", "url":"https://example.invalid/kernel",
                                "content":"The Linux kernel is configured with a .config file that selects which features are built. Running make menuconfig opens a text menu for choosing options, and make olddefconfig fills in defaults for any new options after an upgrade."
                            }]})
                        } else {
                            json!({"results":[{
                                "title":"AI fixture", "url":"https://example.invalid/ai",
                                "content":"Controlled loopback search result. Not real internet evidence."
                            }]})
                        }
                    }
                    _ => json!({"error":format!("Unexpected mock API path {path}")}),
                };
                let body = format!("{response}\n");
                let _=socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).as_bytes());
            }
        });
        Self {
            url,
            requests,
            stop,
            thread: Some(worker),
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

impl Drop for MockOllama {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.thread.take() {
            worker.join().unwrap();
        }
    }
}

fn read_request(socket: &mut TcpStream) -> Option<(String, Value)> {
    let mut bytes = Vec::new();
    let mut buffer = [0u8; 4096];
    loop {
        let count = socket.read(&mut buffer).unwrap_or(0);
        if count == 0 && bytes.is_empty() {
            return None;
        }
        assert!(count > 0, "request ended before body");
        bytes.extend_from_slice(&buffer[..count]);
        if let Some(end) = bytes.windows(4).position(|b| b == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..end]);
            let length = headers
                .lines()
                .find_map(|l| {
                    l.split_once(':')
                        .filter(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                        .map(|(_, v)| v.trim().parse::<usize>().unwrap())
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
                    .to_owned();
                let body = if length == 0 {
                    Value::Null
                } else {
                    serde_json::from_slice(&bytes[end + 4..end + 4 + length]).unwrap()
                };
                return Some((path, body));
            }
        }
    }
}

/// A request's messages without the injected project context (orientation,
/// repo map, project instructions), which is request-only and never part of the
/// saved conversation.
fn conversation(request: &Value) -> Value {
    Value::Array(
        request["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| {
                !(m["role"] == "system"
                    && m["content"].as_str().is_some_and(|c| {
                        c.starts_with("<project_files>")
                            || c.starts_with("<repo_map>")
                            || c.starts_with("<project_instructions")
                    }))
            })
            .cloned()
            .collect(),
    )
}

fn message(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.into(),
        content: content.into(),
        ..Default::default()
    }
}
fn call(name: &str, args: Value) -> Value {
    json!({"message":{"role":"assistant","thinking":"Inspect evidence first","content":"","tool_calls":[{"function":{"name":name,"arguments":args}}]},"done":true})
}
fn final_answer(text: &str) -> Value {
    json!({"message":{"role":"assistant","content":text},"done":true,"eval_count":5})
}

fn job(server: &MockOllama, root: &Path, messages: Vec<ChatMessage>) -> ChatJob {
    let settings = Settings {
        native_tool_probe: false,
        automatic_memory: false,
        knowledge_notes: false,
        notes_review_auto: false,
        agents_enabled: false,
        base_url: server.url.clone(),
        model: "test".into(),
        tts_program: String::new(),
        stt_program: String::new(),
        ..Default::default()
    };
    let engine = ToolEngine::with_backup_dir(
        Workspace::new(root.to_path_buf()).unwrap(),
        root.parent().unwrap().join("recovery"),
    );
    ChatJob {
        settings,
        messages,
        workspace: Some(root.to_path_buf()),
        model: "test".into(),
        engine: Some(engine),
        retrieval_path: root.parent().unwrap().join("knowledge.sqlite3"),
    }
}

#[derive(Default)]
struct Transcript {
    complete: Vec<ChatMessage>,
    tools: Vec<ChatMessage>,
    approvals: Vec<PendingAction>,
    audits: Vec<String>,
    errors: Vec<String>,
    evidence: Vec<(String, Vec<String>)>,
    unexecuted_tool_text: Vec<(Vec<String>, bool)>,
    /// Live tool activity and transcript appends, in arrival order.
    live: Vec<String>,
    statuses: Vec<String>,
    stats: Vec<Value>,
}

enum Decision {
    Approve,
    Deny,
    Cancel,
    CancelAndSendLateApproval,
}

fn run(job: ChatJob, mut decide: impl FnMut(&PendingAction) -> Decision) -> Transcript {
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    runtime::spawn_chat(job, tx, cancel.clone());
    let mut transcript = Transcript::default();
    loop {
        match rx
            .recv_timeout(Duration::from_secs(5))
            .expect("agent worker did not finish")
        {
            ChatEvent::Complete(message) => transcript.complete.push(message),
            ChatEvent::UnexecutedToolText {
                names, retrying, ..
            } => {
                transcript.unexecuted_tool_text.push((names, retrying));
            }
            ChatEvent::Append(entry) => {
                transcript
                    .live
                    .push(format!("append:{}", entry.message.role));
                transcript.tools.push(entry.message)
            }
            ChatEvent::ToolStarted { name, summary, .. } => {
                transcript.live.push(format!("start:{name}:{summary}"))
            }
            ChatEvent::ToolOutput { chunk, .. } => transcript.live.push(format!("out:{chunk}")),
            ChatEvent::ToolFinished { ok, .. } => transcript.live.push(format!("end:{ok}")),
            ChatEvent::Approval { action, reply } => {
                let choice = decide(&action);
                transcript.approvals.push(action);
                match choice {
                    Decision::Approve => reply.send(true).unwrap(),
                    Decision::Deny => reply.send(false).unwrap(),
                    Decision::Cancel => {
                        cancel.store(true, Ordering::Relaxed);
                    }
                    Decision::CancelAndSendLateApproval => {
                        cancel.store(true, Ordering::Relaxed);
                        let _ = reply.send(true);
                    }
                }
            }
            ChatEvent::Audit { action, .. } => transcript.audits.push(action),
            ChatEvent::Status(status) => transcript.statuses.push(status),
            ChatEvent::Stats(stats) => transcript.stats.push(stats),
            ChatEvent::Error(error) => transcript.errors.push(error),
            ChatEvent::Evidence { context, sources } => {
                transcript.evidence.push((context, sources))
            }
            ChatEvent::Finished => break,
            _ => {}
        }
    }
    transcript
}

fn fixture() -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("project");
    std::fs::create_dir(&root).unwrap();
    (directory, root)
}

#[test]
fn read_tool_round_trip_retains_entire_multiturn_history_and_evidence() {
    let (_directory, root) = fixture();
    std::fs::write(root.join("main.rs"), "fn meaning() -> u32 {\n    42\n}\n").unwrap();
    let server = MockOllama::new(vec![
        call("read_file", json!({"path":"main.rs"})),
        final_answer("main.rs:2 returns 42."),
    ]);
    let history = vec![
        message("system", "Cite files and preserve history"),
        message("user", "Remember I want detailed code reviews"),
        message("assistant", "I will explain specific defects and evidence."),
        message("user", "Review main.rs"),
    ];
    let expected: Value = serde_json::to_value(&history).unwrap();
    let transcript = run(job(&server, &root, history), |_| {
        panic!("Read-only tool requested approval")
    });
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(transcript.complete.len(), 2);
    assert!(
        transcript.complete[1]
            .content
            .starts_with("main.rs:2 returns 42.")
            && transcript.complete[1]
                .content
                .contains("recorded by Workbench"),
        "{}",
        transcript.complete[1].content
    );
    assert_eq!(transcript.tools.len(), 1);
    let evidence: Value = serde_json::from_str(&transcript.tools[0].content).unwrap();
    assert_eq!(evidence["path"], "main.rs");
    assert!(evidence["content"].as_str().unwrap().contains("42"));
    assert_eq!(evidence["truncated"], false);
    let requests = server.chats();
    assert_eq!(requests.len(), 2);
    assert_eq!(conversation(&requests[0]), expected);
    let followup_messages = conversation(&requests[1]);
    let followup = followup_messages.as_array().unwrap();
    assert_eq!(&followup[..4], expected.as_array().unwrap());
    assert_eq!(followup[4]["thinking"], "Inspect evidence first");
    assert_eq!(
        followup[4]["tool_calls"][0]["function"]["name"],
        "read_file"
    );
    assert_eq!(followup[5]["role"], "tool");
    assert_eq!(followup[5]["tool_name"], "read_file");
    assert_eq!(
        serde_json::from_str::<Value>(followup[5]["content"].as_str().unwrap()).unwrap(),
        evidence
    );
    assert!(requests[0]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|t| t["function"]["name"] == "read_file"));
}

#[test]
fn denied_mutation_has_concrete_preview_and_preserves_workspace() {
    let (_directory, root) = fixture();
    let target = root.join("notes.txt");
    std::fs::write(&target, "Original text\n").unwrap();
    let server = MockOllama::new(vec![
        call(
            "replace_text",
            json!({"path":"notes.txt","find":"Original","replacement":"Revised"}),
        ),
        final_answer("The requested edit was denied; the file remains unchanged."),
    ]);
    let transcript = run(
        job(&server, &root, vec![message("user", "Revise notes.txt")]),
        |action| {
            assert_eq!(std::fs::read_to_string(&target).unwrap(), "Original text\n");
            assert_eq!(action.details["before"], "Original text\n");
            assert_eq!(action.details["after"], "Revised text\n");
            Decision::Deny
        },
    );
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(transcript.approvals.len(), 1);
    assert_eq!(transcript.audits, ["Denied", "Operation"]);
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "Original text\n");
    let result: Value = serde_json::from_str(&transcript.tools[0].content).unwrap();
    assert_eq!(result["operation"]["state"], "denied");
    assert_eq!(result["operation"]["verified"], false);
    assert_eq!(result["denied"], true);
    assert_eq!(server.chats().len(), 2);
}

#[test]
fn approved_exact_edit_runs_once_retains_backup_and_returns_result_to_model() {
    let (_directory, root) = fixture();
    let target = root.join("notes.txt");
    std::fs::write(&target, "Original text\n").unwrap();
    let server = MockOllama::new(vec![
        call(
            "replace_text",
            json!({"path":"notes.txt","find":"Original","replacement":"Revised"}),
        ),
        final_answer("Updated notes.txt after your approval."),
    ]);
    let transcript = run(
        job(&server, &root, vec![message("user", "Revise notes.txt")]),
        |action| {
            assert_eq!(std::fs::read_to_string(&target).unwrap(), "Original text\n");
            assert_eq!(action.details["after"], "Revised text\n");
            Decision::Approve
        },
    );
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(transcript.approvals.len(), 1);
    assert_eq!(transcript.audits, ["Approved", "Operation"]);
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "Revised text\n");
    let result: Value = serde_json::from_str(&transcript.tools[0].content).unwrap();
    assert_eq!(result["ok"], true);
    // The change was confirmed on disk by Workbench, not taken from the tool's word.
    assert_eq!(result["operation"]["state"], "verified");
    assert_eq!(result["replacements"], 1);
    assert_eq!(
        std::fs::read_to_string(result["backup"].as_str().unwrap()).unwrap(),
        "Original text\n"
    );
    let requests = server.chats();
    let tool = requests[1]["messages"].as_array().unwrap().last().unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(tool["content"].as_str().unwrap()).unwrap(),
        result
    );
    assert_eq!(
        transcript.complete.last().unwrap().content,
        "Updated notes.txt after your approval."
    );
}

#[test]
fn cancellation_while_awaiting_approval_prevents_edits_and_followup_inference() {
    for late_approval in [false, true] {
        let (_directory, root) = fixture();
        let target = root.join("notes.txt");
        std::fs::write(&target, "Original text\n").unwrap();
        let server = MockOllama::new(vec![call(
            "write_file",
            json!({"path":"notes.txt","content":"Changed text\n"}),
        )]);
        let transcript = run(
            job(&server, &root, vec![message("user", "Change notes.txt")]),
            |_| {
                if late_approval {
                    Decision::CancelAndSendLateApproval
                } else {
                    Decision::Cancel
                }
            },
        );
        assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
        assert_eq!(transcript.approvals.len(), 1);
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "Original text\n");
        assert_eq!(
            server.chats().len(),
            1,
            "Cancellation sent another inference request"
        );
    }
}

#[test]
fn changed_file_after_preview_rejects_approved_edit_without_overwriting_new_content() {
    let (_directory, root) = fixture();
    let target = root.join("notes.txt");
    std::fs::write(&target, "Original text\n").unwrap();
    let server = MockOllama::new(vec![
        call(
            "write_file",
            json!({"path":"notes.txt","content":"Model edit\n"}),
        ),
        final_answer("The file changed after preview, so I did not overwrite it."),
    ]);
    let transcript = run(
        job(&server, &root, vec![message("user", "Edit notes.txt")]),
        |_| {
            std::fs::write(&target, "User edited while preview was open\n").unwrap();
            Decision::Approve
        },
    );
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(
        std::fs::read_to_string(&target).unwrap(),
        "User edited while preview was open\n"
    );
    let result: Value = serde_json::from_str(&transcript.tools[0].content).unwrap();
    assert!(result["error"]
        .as_str()
        .unwrap()
        .contains("changed after the preview"));
    assert_eq!(server.chats().len(), 2);
}

#[test]
fn local_only_worker_rejects_renamed_cloud_alias_before_sending_private_content() {
    let (_directory, root) = fixture();
    let server = MockOllama::with_metadata(
        vec![final_answer("This must never be requested")],
        json!({"remote_host":"https://ollama.com","remote_model":"cloud-model"}),
    );
    let transcript = run(
        job(
            &server,
            &root,
            vec![message("user", "PRIVATE_LOCAL_CONTENT")],
        ),
        |_| panic!("No action should be proposed"),
    );
    assert_eq!(
        server.chats().len(),
        0,
        "Private content was sent before cloud-alias verification"
    );
    assert_eq!(transcript.errors.len(), 1);
    assert!(
        transcript.errors[0].contains("remote"),
        "{:?}",
        transcript.errors
    );
    let requests = server.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0], ("/api/show".into(), json!({"model":"test"})));
}

#[test]
fn persisted_retrieval_snapshot_survives_retry_and_new_queries_use_fresh_evidence() {
    use ollama_workbench::{
        retrieval::RetrievalIndex,
        storage::{Entry, Session, Store},
    };
    let (directory, root) = fixture();
    let source = root.join("guide.txt");
    std::fs::write(&source, "Aster mechanism uses ledger snapshots.\n").unwrap();
    let workspace = Workspace::new(root.clone()).unwrap();
    let mut index = RetrievalIndex::open(directory.path().join("knowledge.sqlite3")).unwrap();
    index
        .index_workspace(&workspace, &AtomicBool::new(false), |_| {})
        .unwrap();
    let server = MockOllama::new(vec![
        final_answer("Aster uses ledger snapshots."),
        final_answer("Retried against the same evidence."),
        final_answer("Aster now uses changed live state."),
    ]);
    let original = message("user", "Explain the Aster mechanism");
    let mut first = job(&server, &root, vec![original.clone()]);
    first.settings.retrieval_enabled = true;
    let first_result = run(first, |_| panic!("No tools requested"));
    assert!(first_result.errors.is_empty(), "{:?}", first_result.errors);
    assert_eq!(first_result.evidence.len(), 1);
    assert!(first_result.evidence[0].0.contains("ledger snapshots"));

    // Exercise the exact persistence/request reconstruction used by the GUI.
    let mut entry = Entry::new(original);
    entry.context = first_result.evidence[0].0.clone();
    entry.sources = first_result.evidence[0].1.clone();
    let mut session = Session {
        entries: vec![entry],
        ..Default::default()
    };
    let store = Store::open(&directory.path().join("history")).unwrap();
    store.save(&mut session).unwrap();
    let saved = store.load(&session.id).unwrap();
    let saved_request = saved.entries[0].request_message();
    assert!(saved_request.content.contains("ledger snapshots"));

    // A retry uses the stored source snapshot even if the current index changed.
    std::fs::write(&source, "Aster mechanism now uses changed live state.\n").unwrap();
    index
        .index_workspace(&workspace, &AtomicBool::new(false), |_| {})
        .unwrap();
    let mut retry = job(&server, &root, vec![saved_request.clone()]);
    retry.settings.retrieval_enabled = true;
    let retry_result = run(retry, |_| panic!("No tools requested"));
    assert!(retry_result.errors.is_empty(), "{:?}", retry_result.errors);
    assert!(
        retry_result.evidence.is_empty(),
        "Retry replaced its saved evidence snapshot"
    );

    // A new current user question retrieves fresh evidence while retaining the
    // old snapshot as part of the earlier conversation turn.
    let mut next = job(
        &server,
        &root,
        vec![
            saved_request,
            message("assistant", "Aster uses ledger snapshots."),
            message("user", "Explain the Aster mechanism now"),
        ],
    );
    next.settings.retrieval_enabled = true;
    let next_result = run(next, |_| panic!("No tools requested"));
    assert!(next_result.errors.is_empty(), "{:?}", next_result.errors);
    assert_eq!(next_result.evidence.len(), 1);
    assert!(next_result.evidence[0].0.contains("changed live state"));
    let chats = server.chats();
    assert_eq!(chats.len(), 3);
    let retried_messages = conversation(&chats[1]);
    let retried = retried_messages[0]["content"].as_str().unwrap();
    assert!(retried.contains("ledger snapshots"));
    assert!(!retried.contains("changed live state"));
    assert_eq!(retried.matches("<retrieved_context>").count(), 1);
    let current_messages = conversation(&chats[2]);
    let current = current_messages.as_array().unwrap();
    assert!(current[0]["content"]
        .as_str()
        .unwrap()
        .contains("ledger snapshots"));
    assert!(current[2]["content"]
        .as_str()
        .unwrap()
        .contains("changed live state"));
}

#[test]
fn tool_round_limit_is_enforced_before_any_final_round_action() {
    let (_directory, root) = fixture();
    let target = root.join("notes.txt");
    std::fs::write(&target, "Original text\n").unwrap();
    let server = MockOllama::new(vec![
        call("read_file", json!({"path":"notes.txt"})),
        // Deliberately violate the model contract by asking for another tool
        // after the worker has stopped advertising them.
        call(
            "write_file",
            json!({"path":"notes.txt","content":"Unexpected edit\n"}),
        ),
    ]);
    let mut task = job(&server, &root, vec![message("user", "Review notes.txt")]);
    task.settings.max_tool_rounds = 1;
    let transcript = run(task, |_| Decision::Deny);
    assert!(
        transcript.approvals.is_empty(),
        "The exhausted round proposed another action"
    );
    assert_eq!(transcript.tools.len(), 2);
    let refused: Value = serde_json::from_str(&transcript.tools[1].content).unwrap();
    assert_eq!(refused["denied"], true);
    assert!(refused["reason"]
        .as_str()
        .unwrap()
        .contains("no action was executed"));
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert!(transcript
        .complete
        .last()
        .unwrap()
        .content
        .contains("Review incomplete"));
    assert!(transcript
        .complete
        .last()
        .unwrap()
        .content
        .contains("`notes.txt`"));
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "Original text\n");
    let chats = server.chats();
    // The model ignored the tool-free instruction, so one more attempt restates
    // the evidence with no tool history (it is not scripted here, so it fails
    // and the honest progress report above is what the user gets).
    assert_eq!(chats.len(), 3);
    let last = chats[2]["messages"].as_array().unwrap();
    assert!(
        last.iter()
            .all(|m| m["role"] != "tool" && m.get("tool_calls").is_none()),
        "{last:?}"
    );
    assert!(last.last().unwrap()["content"]
        .as_str()
        .unwrap()
        .contains("Tool use has ended"));
    assert!(chats[2]
        .get("tools")
        .is_none_or(|tools| tools.as_array().is_some_and(Vec::is_empty)));
    assert!(chats[1]
        .get("tools")
        .is_none_or(|tools| tools.as_array().is_some_and(Vec::is_empty)));
    assert!(chats[1]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .any(|m| m["role"] == "system"
            && m["content"]
                .as_str()
                .unwrap_or("")
                .contains("FINAL RESPONSE REQUIRED")));
}

#[test]
fn textual_tool_json_is_repaired_once_through_the_native_protocol() {
    let (_directory, root) = fixture();
    std::fs::write(root.join("main.rs"), "fn main() {}\n").unwrap();
    let server = MockOllama::new(vec![
        final_answer(r#"{"name":"read_file","arguments":{"path":"main.rs"}}"#),
        call("read_file", json!({"path":"main.rs"})),
        final_answer("Inspected main.rs:1 using the actual file tool."),
    ]);
    let transcript = run(
        job(&server, &root, vec![message("user", "Read main.rs")]),
        |_| panic!("Read requested approval"),
    );
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(transcript.tools.len(), 1);
    assert_eq!(server.chats().len(), 3);
    assert!(server.chats()[1]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .any(|m| m["role"] == "system"
            && m["content"]
                .as_str()
                .unwrap()
                .contains("No operation was performed")));
}

#[test]
fn adjacent_tool_json_is_not_executed_but_native_retry_can_inspect_real_files() {
    let (_directory, root) = fixture();
    std::fs::write(root.join("main.rs"), "fn main() {}\n").unwrap();
    // Reproduces the user's screenshot: several adjacent JSON objects, no
    // delimiters, and no actual tool_calls in the first Ollama response.
    let printed = [
        r#"{"name":"project_tree","arguments":{"path":"."}}"#,
        r#"{"name":"read_file","arguments":{"path":"main.rs"}}"#,
        r#"{"name":"search_files","arguments":{"query":"test"}}"#,
    ]
    .concat();
    let server = MockOllama::new(vec![
        final_answer(&printed),
        call("project_tree", json!({"path":"."})),
        final_answer("Verified the actual project tree with the file tool."),
    ]);
    let transcript = run(
        job(&server, &root, vec![message("user", "Review the project")]),
        |_| panic!("No side-effect approval expected"),
    );
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(transcript.unexecuted_tool_text.len(), 1);
    assert_eq!(
        transcript.unexecuted_tool_text[0].0,
        vec![
            "project_tree".to_string(),
            "read_file".to_string(),
            "search_files".to_string()
        ]
    );
    assert!(transcript.unexecuted_tool_text[0].1);
    assert_eq!(transcript.tools.len(), 1, "only the native tool call runs");
    assert_eq!(
        transcript.tools[0].tool_name.as_deref(),
        Some("project_tree")
    );
    assert!(
        transcript
            .complete
            .last()
            .unwrap()
            .content
            .starts_with("Verified the actual project tree with the file tool.")
            && transcript
                .complete
                .last()
                .unwrap()
                .content
                .contains("no file contents were read"),
        "{}",
        transcript.complete.last().unwrap().content
    );
    let requests = server.chats();
    assert_eq!(requests.len(), 3);
    assert!(requests[1]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|d| d["function"]["name"] == "project_tree"));
    assert!(requests[1]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .any(|m| m["role"] == "system"
            && m["content"].as_str().unwrap().contains("native tool_calls")));
}

#[test]
fn repeated_textual_actions_never_execute_or_claim_completion() {
    let (_directory, root) = fixture();
    let target = root.join("keep.txt");
    std::fs::write(&target, "original").unwrap();
    let text =
        r#"{"name":"write_file","arguments":{"path":"keep.txt","content":"not authorized"}}"#;
    let server = MockOllama::new(vec![final_answer(text), final_answer(text)]);
    let transcript = run(
        job(&server, &root, vec![message("user", "Review keep.txt")]),
        |_| panic!("Text is not an authorization"),
    );
    assert_eq!(std::fs::read_to_string(target).unwrap(), "original");
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(transcript.unexecuted_tool_text.len(), 2);
    assert!(transcript.unexecuted_tool_text[0].1);
    assert!(!transcript.unexecuted_tool_text[1].1);
    assert!(transcript.tools.is_empty());
    assert!(transcript
        .complete
        .last()
        .unwrap()
        .content
        .contains("Review incomplete"));
    assert!(transcript
        .complete
        .last()
        .unwrap()
        .content
        .contains("bounded protocol retry did not recover"));
    assert_eq!(server.chats().len(), 2);
}

#[test]
fn unknown_textual_tool_name_is_rejected_without_retrying_or_execution() {
    let (_directory, root) = fixture();
    let printed = r#"{"name":"not_a_real_tool","arguments":{"path":"main.rs"}}"#;
    let server = MockOllama::new(vec![final_answer(printed)]);
    let transcript = run(
        job(&server, &root, vec![message("user", "Review the files")]),
        |_| panic!("Plain text must never create an approval"),
    );
    assert_eq!(transcript.unexecuted_tool_text.len(), 1);
    assert!(!transcript.unexecuted_tool_text[0].1);
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert!(transcript
        .complete
        .last()
        .unwrap()
        .content
        .contains("unregistered tool name"));
    assert!(transcript
        .complete
        .last()
        .unwrap()
        .content
        .contains("Review incomplete"));
    assert!(transcript.tools.is_empty());
    assert_eq!(server.chats().len(), 1);
}

#[test]
fn prose_mentioning_tool_json_is_not_mistaken_for_an_action() {
    let (_directory, root) = fixture();
    let text = r#"Here is an example: {"name":"project_tree","arguments":{"path":"."}}"#;
    let server = MockOllama::new(vec![final_answer(text)]);
    let transcript = run(
        job(
            &server,
            &root,
            vec![message("user", "Describe a tool JSON example")],
        ),
        |_| panic!("No approval expected"),
    );
    assert!(transcript.errors.is_empty());
    assert!(transcript.unexecuted_tool_text.is_empty());
    assert!(transcript.tools.is_empty());
    assert_eq!(transcript.complete.len(), 1);
    assert_eq!(transcript.complete[0].content, text);
}

#[test]
fn repeated_native_reads_are_skipped_and_force_an_early_final_response() {
    let (_directory, root) = fixture();
    std::fs::write(root.join("main.rs"), "fn main() {}\n").unwrap();
    let repeated = || call("read_file", json!({"path":"main.rs"}));
    let server = MockOllama::new(vec![
        repeated(),
        repeated(),
        repeated(),
        final_answer("Used the first verified read of main.rs and stopped repeating it."),
    ]);
    let transcript = run(
        job(&server, &root, vec![message("user", "Review main.rs")]),
        |_| panic!("Read-only tool requested approval"),
    );
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(transcript.tools.len(), 3);
    let first: Value = serde_json::from_str(&transcript.tools[0].content).unwrap();
    assert_eq!(first["path"], "main.rs");
    for skipped in &transcript.tools[1..] {
        let value: Value = serde_json::from_str(&skipped.content).unwrap();
        assert_eq!(value["duplicate"].as_bool(), Some(true));
        assert_eq!(value["executed"].as_bool(), Some(false));
    }
    assert!(
        transcript
            .complete
            .last()
            .unwrap()
            .content
            .starts_with("Used the first verified read of main.rs and stopped repeating it.")
            && transcript
                .complete
                .last()
                .unwrap()
                .content
                .contains("recorded by Workbench"),
        "{}",
        transcript.complete.last().unwrap().content
    );
    let requests = server.chats();
    assert_eq!(
        requests.len(),
        4,
        "the no-progress guard should stop well before the normal 12-round limit"
    );
    assert!(requests[3]
        .get("tools")
        .is_none_or(|tools| tools.as_array().is_some_and(Vec::is_empty)));
}

#[test]
fn successful_write_opens_a_new_epoch_for_a_real_verification_reread() {
    let (_directory, root) = fixture();
    std::fs::write(root.join("main.rs"), "fn old() {}\n").unwrap();
    let server = MockOllama::new(vec![
        call("read_file", json!({"path":"main.rs"})),
        call(
            "write_file",
            json!({"path":"main.rs","content":"fn updated() {}\n"}),
        ),
        call("read_file", json!({"path":"main.rs"})),
        final_answer("Verified the approved update in main.rs."),
    ]);
    let transcript = run(
        job(
            &server,
            &root,
            vec![message("user", "Update and verify main.rs")],
        ),
        |_| Decision::Approve,
    );
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(transcript.approvals.len(), 1);
    assert_eq!(transcript.tools.len(), 3);
    let reread: Value = serde_json::from_str(&transcript.tools[2].content).unwrap();
    assert_eq!(reread["path"], "main.rs");
    assert!(reread["content"].as_str().unwrap().contains("updated"));
    assert!(reread.get("duplicate").is_none());
    assert_eq!(
        std::fs::read_to_string(root.join("main.rs")).unwrap(),
        "fn updated() {}\n"
    );
    assert_eq!(server.chats().len(), 4);
}

#[test]
fn large_project_tree_is_saved_in_full_but_sent_to_model_with_explicit_limits() {
    let (_directory, root) = fixture();
    for index in 0..300 {
        std::fs::write(
            root.join(format!("file_{index:04}.rs")),
            "pub fn sample() {}\n",
        )
        .unwrap();
    }
    let server = MockOllama::new(vec![
        call(
            "project_tree",
            json!({"path":".", "max_depth":2, "limit":500}),
        ),
        final_answer(
            "I received a partial project listing; deeper coverage needs more tool calls.",
        ),
    ]);
    let result = run(
        job(
            &server,
            &root,
            vec![message("user", "Inspect this project structure")],
        ),
        |_| panic!("Read-only project_tree must not require approval"),
    );
    assert!(result.errors.is_empty(), "{:?}", result.errors);
    let full: Value = serde_json::from_str(&result.tools[0].content).unwrap();
    assert_eq!(full["entries"].as_array().unwrap().len(), 300);
    let requests = server.chats();
    assert_eq!(requests.len(), 2);
    let second_turn = requests[1]["messages"].as_array().unwrap();
    let model_tool = second_turn.iter().find(|m| m["role"] == "tool").unwrap();
    let limited: Value = serde_json::from_str(model_tool["content"].as_str().unwrap()).unwrap();
    assert_eq!(limited["context_limited"], true);
    assert_eq!(limited["truncated"], true);
    assert!(limited["entries"].as_array().unwrap().len() < 300);
    assert!(limited["context_note"]
        .as_str()
        .unwrap()
        .contains("full actual tool output"));
}

#[test]
fn tool_limit_can_end_with_real_summary_without_executing_more_tools() {
    let (_directory, root) = fixture();
    std::fs::write(root.join("main.rs"), "fn main() {}\n").unwrap();
    let server = MockOllama::new(vec![
        call("read_file", json!({"path":"main.rs"})),
        final_answer("Inspected main.rs, lines 1–1. The broader project remains unreviewed."),
    ]);
    let mut task = job(&server, &root, vec![message("user", "Review this project")]);
    task.settings.max_tool_rounds = 1;
    let transcript = run(task, |_| panic!("No write approval expected"));
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(transcript.tools.len(), 1);
    assert!(
        transcript
            .complete
            .last()
            .unwrap()
            .content
            .starts_with("Inspected main.rs, lines 1–1. The broader project remains unreviewed.")
            && transcript
                .complete
                .last()
                .unwrap()
                .content
                .contains("recorded by Workbench"),
        "{}",
        transcript.complete.last().unwrap().content
    );
    let requests = server.chats();
    assert_eq!(requests.len(), 2);
    assert!(requests[1].get("tools").is_none());
}

#[test]
fn tool_limit_with_empty_model_response_still_records_honest_progress() {
    let (_directory, root) = fixture();
    std::fs::write(root.join("main.rs"), "fn main() {}\n").unwrap();
    let server = MockOllama::new(vec![
        call("read_file", json!({"path":"main.rs"})),
        final_answer(""),
    ]);
    let mut task = job(&server, &root, vec![message("user", "Review this project")]);
    task.settings.max_tool_rounds = 1;
    let transcript = run(task, |_| panic!("No approval expected"));
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(transcript.complete.len(), 2);
    let final_message = &transcript.complete.last().unwrap().content;
    assert!(final_message.contains("Review incomplete"));
    assert!(final_message.contains("`main.rs`"));
    assert!(final_message.contains("not a model-generated code review"));
}

#[test]
fn text_tool_request_in_final_pass_is_not_reported_as_an_inspection() {
    let (_directory, root) = fixture();
    std::fs::write(root.join("main.rs"), "fn main() {}\n").unwrap();
    let server = MockOllama::new(vec![
        call("read_file", json!({"path":"main.rs"})),
        final_answer(r#"{"name":"read_file","arguments":{"path":"unread.rs"}}"#),
    ]);
    let mut task = job(&server, &root, vec![message("user", "Review this project")]);
    task.settings.max_tool_rounds = 1;
    let transcript = run(task, |_| panic!("No approval expected"));
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(transcript.unexecuted_tool_text.len(), 1);
    assert!(!transcript.unexecuted_tool_text[0].1);
    let final_message = &transcript.complete.last().unwrap().content;
    assert!(final_message.contains("Review incomplete"));
    assert!(final_message.contains("`main.rs`"));
    assert!(!final_message.contains("unread.rs"));
    assert_eq!(server.chats().len(), 2);
}

fn web_job(server: &MockOllama, root: &Path) -> ChatJob {
    let mut job = job(server, root, vec![message("user", "web search ai")]);
    job.settings.network_tools = true;
    let engine = job.engine.as_ref().unwrap();
    engine.set_network_enabled(true);
    engine.set_search_url(Some(server.url.clone())).unwrap();
    // No fallback browsers: the tests must never reach the real DuckDuckGo.
    engine.set_text_browser_fallbacks(Some(vec![])).unwrap();
    job
}

#[test]
fn offline_code_request_keeps_local_answer_and_blocks_provider_calls() {
    for unsolicited_call in [false, true] {
        let (_directory, root) = fixture();
        let mut replies = vec![];
        if unsolicited_call {
            replies.push(call("web_search", json!({"query":"sort example"})));
        }
        replies.push(final_answer("def ordered(values): return sorted(values)"));
        let server = MockOllama::new(replies);
        let mut task = web_job(&server, &root);
        task.settings.auto_approve_web = true;
        task.messages = vec![message(
            "user",
            "Write a Python function to sort a list. Do not search the web.",
        )];
        let transcript = run(task, |_| Decision::Approve);
        assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
        assert!(transcript.approvals.is_empty());
        assert_eq!(
            transcript.complete.last().unwrap().content,
            "def ordered(values): return sorted(values)"
        );
        assert!(!server
            .requests
            .lock()
            .unwrap()
            .iter()
            .any(|(path, _)| path.starts_with("/search")));
        for chat in server.chats() {
            assert!(chat["tools"]
                .as_array()
                .unwrap()
                .iter()
                .all(|t| t["function"]["name"] != "web_search"));
            assert!(!chat["messages"]
                .to_string()
                .contains("Before writing any code, call web_search"));
        }
        if unsolicited_call {
            let result: Value = serde_json::from_str(&transcript.tools[0].content).unwrap();
            assert_eq!(result["denied"], true);
            assert_eq!(result["executed"], false);
        }
    }
}

#[test]
fn lower_memory_preset_reads_workspace_file_with_the_full_default_system_prompt() {
    let (_directory, root) = fixture();
    std::fs::write(root.join("README.md"), "Workshop line 42.").unwrap();
    let server = MockOllama::new(vec![
        call("read_file", json!({"path":"README.md"})),
        final_answer("Workshop line 42."),
    ]);
    let mut task = job(&server, &root, vec![]);
    task.settings.agents_enabled = true;
    task.settings.num_ctx = 4096;
    task.settings.num_predict = -1;
    task.engine = None;
    task.messages = vec![
        message("system", &task.settings.system()),
        message("user", "Read README.md"),
    ];
    let transcript = run(task, |_| panic!("Reading a file is read-only"));
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(
        transcript.complete.last().unwrap().content,
        "Workshop line 42."
    );
    let chats = server.chats();
    assert_eq!(chats.len(), 2);
    assert!(chats[0]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|t| t["function"]["name"] == "read_file"));
    assert_eq!(chats[0]["options"]["num_ctx"], 4096);
    assert!(chats[1]["messages"].as_array().unwrap().iter().any(
        |m| m["role"] == "tool" && m["content"].as_str().unwrap().contains("Workshop line 42.")
    ));
}

#[test]
fn explicit_web_search_missing_configuration_never_runs_local_tools_or_inference() {
    let (_directory, root) = fixture();
    let server = MockOllama::new(vec![]);
    let request = job(&server, &root, vec![message("user", "web search ai")]);
    let transcript = run(request, |_| panic!("No action should need approval"));
    assert!(transcript
        .errors
        .iter()
        .any(|error| error.contains("network tools are disabled")));
    assert!(transcript.tools.is_empty());
    assert!(server.requests.lock().unwrap().is_empty());

    let mut request = job(
        &server,
        &root,
        vec![message("user", "search the web about Linux")],
    );
    request.settings.network_tools = true;
    request.engine.as_ref().unwrap().set_network_enabled(true);
    let transcript = run(request, |_| panic!("No provider is configured"));
    assert!(transcript
        .errors
        .iter()
        .any(|error| error.contains("no enabled search provider")));
    assert!(server.requests.lock().unwrap().is_empty());
}

#[test]
fn explicit_web_search_tools_disabled_is_reported_before_model_routing() {
    let (_directory, root) = fixture();
    let server = MockOllama::new(vec![]);
    let mut request = web_job(&server, &root);
    request.settings.tools_enabled = false;
    request.settings.model_routing = true;
    let transcript = run(request, |_| panic!("Tools are disabled"));
    assert!(transcript
        .errors
        .iter()
        .any(|error| error.contains("Tools are disabled")));
    assert!(server.requests.lock().unwrap().is_empty());
}

#[test]
fn web_search_scope_blocks_native_local_read_then_preserves_real_search_roundtrip() {
    let (_directory, root) = fixture();
    std::fs::write(root.join("private.txt"), "PRIVATE-MUST-NOT-BE-READ").unwrap();
    let server = MockOllama::new(vec![
        call("read_file", json!({"path":"private.txt"})),
        call("web_search", json!({"query":"ai","limit":3})),
        final_answer("The controlled search returned the AI fixture."),
    ]);
    let request = web_job(&server, &root);
    let shared_engine = request.engine.clone().unwrap();
    let transcript = run(request, |_| Decision::Approve);
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(transcript.approvals.len(), 1);
    let blocked: Value = serde_json::from_str(&transcript.tools[0].content).unwrap();
    assert_eq!(blocked["denied"], true);
    assert_eq!(blocked["executed"], false);
    assert!(!transcript.tools[0]
        .content
        .contains("PRIVATE-MUST-NOT-BE-READ"));
    assert_eq!(transcript.tools[1].tool_name.as_deref(), Some("web_search"));
    assert!(transcript.tools[1].content.contains("AI fixture"));
    for request in server.chats() {
        for tool in request["tools"].as_array().unwrap() {
            assert!(matches!(
                tool["function"]["name"].as_str(),
                Some("web_search" | "fetch_url")
            ));
        }
    }
    assert!(server.chats()[2]["messages"]
        .to_string()
        .contains("AI fixture"));
    // Scope only this turn; do not disable tools on the shared chat engine.
    assert!(shared_engine
        .definitions()
        .iter()
        .any(|tool| tool["function"]["name"] == "read_file"));
}

#[test]
fn unsearched_web_answer_is_retried_once_not_presented_as_researched() {
    let (_directory, root) = fixture();
    let server = MockOllama::new(vec![
        final_answer("I found sources even though I did not call a tool."),
        call("web_search", json!({"query":"ai"})),
        final_answer("The real tool result in this test is AI fixture."),
    ]);
    let transcript = run(web_job(&server, &root), |_| Decision::Approve);
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert!(transcript.complete[0]
        .content
        .contains("No web search was performed"));
    assert!(!server.chats()[1]["messages"]
        .to_string()
        .contains("I found sources even though"));
    assert_eq!(transcript.approvals.len(), 1);
    assert_eq!(server.chats().len(), 3);
}

#[test]
fn denied_web_search_is_not_retried_or_presented_as_a_success() {
    let (_directory, root) = fixture();
    let server = MockOllama::new(vec![
        call("web_search", json!({"query":"ai"})),
        final_answer("I searched successfully."),
    ]);
    let transcript = run(web_job(&server, &root), |_| Decision::Deny);
    assert_eq!(transcript.approvals.len(), 1);
    assert!(transcript
        .complete
        .last()
        .unwrap()
        .content
        .contains("Web search incomplete"));
    assert!(!transcript
        .complete
        .last()
        .unwrap()
        .content
        .contains("I searched successfully"));
    assert!(!server
        .requests
        .lock()
        .unwrap()
        .iter()
        .any(|(path, _)| path.starts_with("/search?")));
}

#[test]
fn provider_error_cannot_be_reported_as_a_successful_web_search() {
    let (_directory, root) = fixture();
    let server = MockOllama::new(vec![
        call("web_search", json!({"query":"provider-failure"})),
        final_answer("The web search succeeded."),
    ]);
    let transcript = run(web_job(&server, &root), |_| Decision::Approve);
    let result: Value = serde_json::from_str(&transcript.tools[0].content).unwrap();
    assert!(result["error"].as_str().unwrap().contains("results array"));
    assert!(transcript
        .complete
        .last()
        .unwrap()
        .content
        .contains("Web search incomplete"));
}

#[test]
fn web_search_wrong_tool_loop_is_bounded_even_with_different_arguments() {
    let (_directory, root) = fixture();
    let server = MockOllama::new(vec![
        call("list_files", json!({"path":"."})),
        call("list_files", json!({"path":"src"})),
        final_answer("Here is a guess about AI."),
    ]);
    let transcript = run(web_job(&server, &root), |_| {
        panic!("Local tool must be rejected")
    });
    assert_eq!(server.chats().len(), 3);
    assert!(server.chats()[2].get("tools").is_none());
    assert!(transcript
        .complete
        .last()
        .unwrap()
        .content
        .contains("No online search was performed"));
    for tool in transcript.tools {
        let result: Value = serde_json::from_str(&tool.content).unwrap();
        assert_eq!(result["executed"], false);
    }
}

#[test]
fn thinking_only_reply_after_file_read_gets_one_bounded_answer_retry() {
    let (_directory, root) = fixture();
    std::fs::write(root.join("main.rs"), "fn main() {}\n").unwrap();
    let server = MockOllama::new(vec![
        call("read_file", json!({"path":"main.rs"})),
        json!({"message":{"role":"assistant","thinking":"Thinking only","content":""},"done":true,"done_reason":"length"}),
        final_answer("main.rs:1 defines main."),
    ]);
    let transcript = run(
        job(&server, &root, vec![message("user", "Review main.rs")]),
        |_| panic!("Read only"),
    );
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(server.chats().len(), 3);
    assert_eq!(transcript.tools.len(), 1);
    assert!(transcript.complete[1]
        .content
        .contains("done_reason: length"));
    assert!(
        transcript
            .complete
            .last()
            .unwrap()
            .content
            .starts_with("main.rs:1 defines main.")
            && transcript
                .complete
                .last()
                .unwrap()
                .content
                .contains("recorded by Workbench"),
        "{}",
        transcript.complete.last().unwrap().content
    );
    assert!(server.chats()[2]["messages"]
        .to_string()
        .contains("fn main()"));
}

#[test]
fn repeated_empty_model_answers_end_with_explicit_report() {
    let (_directory, root) = fixture();
    let empty = json!({"message":{"role":"assistant","content":""},"done":true});
    let server = MockOllama::new(vec![empty.clone(), empty]);
    let transcript = run(
        job(&server, &root, vec![message("user", "Review project")]),
        |_| panic!("No tool call"),
    );
    assert_eq!(server.chats().len(), 2);
    assert!(transcript
        .complete
        .last()
        .unwrap()
        .content
        .contains("bounded recovery attempt"));
    assert!(transcript
        .complete
        .iter()
        .all(|message| !message.content.is_empty()));
}

#[test]
fn first_inference_failure_is_an_error_not_an_incomplete_review_report() {
    let (_directory, root) = fixture();
    let server = MockOllama::new(vec![json!({"error":"model 'test' not found"})]);
    let transcript = run(
        job(&server, &root, vec![message("user", "Review main.rs")]),
        |_| panic!("No tool call"),
    );
    assert!(transcript.complete.is_empty(), "{:?}", transcript.complete);
    assert_eq!(transcript.errors.len(), 1);
    assert!(transcript.errors[0].contains("not found"));
}

#[test]
fn inference_failure_after_tool_retains_evidence_and_exact_error() {
    let (_directory, root) = fixture();
    std::fs::write(root.join("main.rs"), "fn main() {}\n").unwrap();
    let server = MockOllama::new(vec![
        call("read_file", json!({"path":"main.rs"})),
        json!({"error":"synthetic inference failure"}),
    ]);
    let transcript = run(
        job(&server, &root, vec![message("user", "Review main.rs")]),
        |_| panic!("Read only"),
    );
    let report = &transcript.complete.last().unwrap().content;
    assert!(report.contains("synthetic inference failure"));
    assert!(report.contains("main.rs"));
    assert!(report.contains("1 successful tool result"));
    assert_eq!(transcript.tools.len(), 1);
}

#[test]
fn approved_command_output_streams_live_before_the_saved_result() {
    let (_directory, root) = fixture();
    let script = "echo first; sleep 0.3; echo second";
    let server = MockOllama::new(vec![
        call("run_command", json!({"program":"sh","args":["-c",script]})),
        final_answer("Ran the command."),
    ]);
    let transcript = run(
        job(&server, &root, vec![message("user", "Run the check")]),
        |_| Decision::Approve,
    );
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    let live = &transcript.live;
    assert_eq!(
        live[0],
        format!("start:run_command:$ sh -c {script}"),
        "{live:?}"
    );
    let outputs: Vec<&str> = live.iter().filter_map(|e| e.strip_prefix("out:")).collect();
    // The first line must arrive on its own, before the command printed the
    // second one: output is streamed while the process runs, not at the end.
    assert!(
        outputs.len() >= 2,
        "expected separate live chunks: {live:?}"
    );
    assert!(
        outputs[0].contains("first") && !outputs[0].contains("second"),
        "{live:?}"
    );
    assert_eq!(outputs.concat(), "first\nsecond\n");
    let end = live
        .iter()
        .position(|e| e == "end:true")
        .expect("tool finished ok");
    let append = live
        .iter()
        .position(|e| e == "append:tool")
        .expect("saved result");
    let last_output = live.iter().rposition(|e| e.starts_with("out:")).unwrap();
    assert!(last_output < end && end < append, "{live:?}");
}

#[test]
fn duplicate_tool_requests_do_not_open_live_activity() {
    let (_directory, root) = fixture();
    std::fs::write(root.join("a.txt"), "alpha").unwrap();
    let server = MockOllama::new(vec![
        call("read_file", json!({"path":"a.txt"})),
        call("read_file", json!({"path":"a.txt"})),
        final_answer("Done."),
    ]);
    let transcript = run(
        job(&server, &root, vec![message("user", "Read a.txt")]),
        |_| Decision::Approve,
    );
    let starts = transcript
        .live
        .iter()
        .filter(|e| e.starts_with("start:read_file:a.txt"))
        .count();
    assert_eq!(starts, 1, "{:?}", transcript.live);
    assert_eq!(
        transcript
            .live
            .iter()
            .filter(|e| e.starts_with("end:"))
            .count(),
        1
    );
}

#[test]
fn project_file_paths_are_given_to_the_model_up_front() {
    let (_directory, root) = fixture();
    std::fs::create_dir(root.join("src")).unwrap();
    std::fs::write(root.join("src/lib.rs"), "pub fn add() {}").unwrap();
    std::fs::write(root.join(".env"), "SECRET=1").unwrap();
    std::fs::write(root.join(".gitignore"), "ignored.txt\n").unwrap();
    std::fs::write(root.join("ignored.txt"), "x").unwrap();
    let server = MockOllama::new(vec![final_answer("ok")]);
    let transcript = run(
        job(&server, &root, vec![message("user", "What does add do?")]),
        |_| Decision::Approve,
    );
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    let first = &server.chats()[0];
    let orientation = first["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["role"] == "system")
        .filter_map(|m| m["content"].as_str())
        .find(|c| c.contains("<project_files>"))
        .expect("project orientation was not sent");
    assert!(orientation.contains("\nsrc/lib.rs\n"), "{orientation}");
    assert!(
        !orientation.contains(".env"),
        "protected file listed: {orientation}"
    );
    assert!(
        !orientation.contains("ignored.txt"),
        "gitignored file listed: {orientation}"
    );
    // The orientation sits just before the user's request.
    let messages = first["messages"].as_array().unwrap();
    let user = messages.iter().rposition(|m| m["role"] == "user").unwrap();
    assert!(messages[user - 1]["content"]
        .as_str()
        .unwrap()
        .contains("<project_files>"));
}

#[test]
fn web_search_turn_only_fetches_urls_from_results_or_the_user() {
    let (_directory, root) = fixture();
    let server = MockOllama::new(vec![
        call("web_search", json!({"query":"loopback-fixture"})),
        // Guessed from memory: must not reach approval or the network.
        call("fetch_url", json!({"url":"https://guessed.example/page"})),
        // Listed in the search result (with cosmetic differences).
        call("fetch_url", json!({"url":"http://127.0.0.1/ai/"})),
        final_answer("Done."),
    ]);
    let transcript = run(web_job(&server, &root), |action| {
        // Getting as far as approval proves the returned URL was accepted.
        // Do not send a real request to the fixture URL.
        if action.details["url"].as_str().is_some() {
            Decision::Deny
        } else {
            Decision::Approve
        }
    });
    let titles: Vec<&str> = transcript
        .approvals
        .iter()
        .map(|a| a.title.as_str())
        .collect();
    assert_eq!(titles.len(), 2, "{titles:?}");
    assert!(transcript.approvals[1]
        .details
        .to_string()
        .contains("127.0.0.1/ai"));
    assert!(!transcript
        .approvals
        .iter()
        .any(|a| a.details.to_string().contains("guessed.example")));
    let guessed: Value = serde_json::from_str(&transcript.tools[1].content).unwrap();
    assert_eq!(guessed["executed"], false);
    assert!(guessed["result_urls"]
        .as_array()
        .unwrap()
        .iter()
        .any(|u| u == "127.0.0.1/ai"));
}

#[cfg(unix)]
#[test]
fn search_refused_by_the_provider_is_not_retried_in_the_same_turn() {
    use std::os::unix::fs::PermissionsExt;
    let (directory, root) = fixture();
    let program = directory.path().join("lynx");
    std::fs::write(
        &program,
        "#!/bin/sh\necho 'If this persists, please <a href=\"mailto:x@duckduckgo.com?subject=Error getting results\">email us</a>.'\n",
    )
    .unwrap();
    std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
    let server = MockOllama::new(vec![
        call("web_search", json!({"query":"first"})),
        call("web_search", json!({"query":"second"})),
        call("web_search", json!({"query":"third"})),
        final_answer("Search is unavailable right now."),
    ]);
    let mut job = job(&server, &root, vec![message("user", "web search rust")]);
    job.settings.network_tools = true;
    let engine = job.engine.as_ref().unwrap();
    engine.set_network_enabled(true);
    engine
        .set_text_browser_program(Some(ollama_workbench::integrations::TextBrowser {
            name: "lynx".into(),
            program,
        }))
        .unwrap();
    engine.set_text_browser_fallbacks(Some(vec![])).unwrap();
    let transcript = run(job, |_| Decision::Approve);
    assert_eq!(
        transcript.approvals.len(),
        1,
        "only the first query may be sent"
    );
    let first: Value = serde_json::from_str(&transcript.tools[0].content).unwrap();
    assert!(
        first["error"].as_str().unwrap().contains("rate limited"),
        "{first}"
    );
    for later in &transcript.tools[1..3] {
        let value: Value = serde_json::from_str(&later.content).unwrap();
        assert_eq!(value["executed"], false, "{value}");
        assert!(value["reason"]
            .as_str()
            .unwrap()
            .contains("already refused"));
    }
}

#[test]
fn a_profile_with_an_explicit_model_is_not_overridden_by_routing() {
    let (_directory, root) = fixture();
    let server = MockOllama::new(vec![final_answer("Reviewed.")]);
    let mut request = job(
        &server,
        &root,
        vec![message("user", "Review this function")],
    );
    request.settings.model_routing = true;
    request.settings.profiles = vec![ollama_workbench::settings::Profile {
        name: "Code review".into(),
        model: "test".into(),
        instruction: String::new(),
    }];
    request.settings.selected_profile = 0;
    let transcript = run(request, |_| Decision::Approve);
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert!(
        transcript
            .statuses
            .iter()
            .any(|s| s.contains("Using test from the Code review profile")),
        "{:?}",
        transcript.statuses
    );
    assert!(
        !transcript
            .statuses
            .iter()
            .any(|s| s.starts_with("Routed to")),
        "routing overrode the profile: {:?}",
        transcript.statuses
    );
    assert_eq!(server.chats()[0]["model"], "test");
}

#[test]
fn reply_that_runs_out_of_room_continues_automatically_until_finished() {
    let (_directory, root) = fixture();
    let server = MockOllama::new(vec![
        json!({
            "message":{"role":"assistant","content":"Step 1 is to open the [Arch"},
            "done":true,"done_reason":"length","eval_count":2048
        }),
        json!({
            "message":{"role":"assistant","content":" Wiki] and read it."},
            "done":true,"done_reason":"stop","eval_count":9
        }),
    ]);
    let transcript = run(
        job(
            &server,
            &root,
            vec![message("user", "How do I configure it?")],
        ),
        |_| panic!("No tools"),
    );
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert!(
        transcript
            .statuses
            .iter()
            .any(|s| s.contains("continuing automatically")),
        "{:?}",
        transcript.statuses
    );
    let last = transcript.stats.last().expect("stats event");
    assert!(last.get("incomplete").is_none(), "{last:?}");
    assert_eq!(server.chats().len(), 2);
}

#[test]
fn a_normally_finished_reply_is_not_marked_incomplete() {
    let (_directory, root) = fixture();
    let server = MockOllama::new(vec![json!({
        "message":{"role":"assistant","content":"Done."},
        "done":true,"done_reason":"stop","eval_count":3
    })]);
    let transcript = run(job(&server, &root, vec![message("user", "Hi")]), |_| {
        panic!("No tools")
    });
    assert!(
        transcript
            .stats
            .iter()
            .all(|s| s.get("incomplete").is_none()),
        "{:?}",
        transcript.stats
    );
}

fn long_answer() -> String {
    "The Linux kernel is configured with a .config file that selects which features are built. \
     Running make menuconfig opens a text menu for choosing options, and make olddefconfig fills \
     in defaults for any new options after an upgrade. Options can be built in or as loadable \
     modules, which affects the size of the kernel image and what hardware works at boot."
        .into()
}

#[test]
fn only_facts_from_web_pages_become_notes_that_a_later_chat_recalls() {
    let (_directory, root) = fixture();
    let data_dir = root.parent().unwrap().to_path_buf();
    // A long answer from the model alone is unverified: no note is saved.
    let server = MockOllama::new(vec![final_answer(&long_answer())]);
    let mut first = job(
        &server,
        &root,
        vec![message("user", "How do I configure the Linux kernel?")],
    );
    first.settings.knowledge_notes = true;
    let transcript = run(first, |_| panic!("No tools"));
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert!(
        !transcript
            .statuses
            .iter()
            .any(|s| s.starts_with("Saved to knowledge notes")),
        "{:?}",
        transcript.statuses
    );
    assert!(!data_dir
        .join("knowledge-notes/configure-linux-kernel.md")
        .exists());

    // A turn backed by a web search saves the page's facts, credited to the page.
    let server = MockOllama::new(vec![
        call("web_search", json!({"query":"kernel-notes"})),
        final_answer(&long_answer()),
    ]);
    let mut searched = web_job(&server, &root);
    searched.messages = vec![message(
        "user",
        "web search how the Linux kernel is configured",
    )];
    searched.settings.knowledge_notes = true;
    let transcript = run(searched, |_| Decision::Approve);
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    let saved = transcript
        .statuses
        .iter()
        .find(|s| s.starts_with("Saved to knowledge notes"))
        .unwrap_or_else(|| panic!("{:?}", transcript.statuses))
        .clone();
    let notes = std::fs::read_dir(data_dir.join("knowledge-notes"))
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| std::fs::read_to_string(e.path()).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        notes.contains("https://example.invalid/kernel"),
        "{saved}\n{notes}"
    );
    assert!(notes.contains("configured with a .config file"), "{notes}");
    assert!(!notes.contains("model answer, unverified"), "{notes}");

    // A later conversation on the subject is given the sourced note.
    let server = MockOllama::new(vec![final_answer("Short answer.")]);
    let mut later = job(
        &server,
        &root,
        vec![message(
            "user",
            "Remind me how the Linux kernel config file works",
        )],
    );
    later.settings.knowledge_notes = true;
    let transcript = run(later, |_| panic!("No tools"));
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    let request = server.chats()[0]["messages"].to_string();
    assert!(
        request.contains("<knowledge_notes>") && request.contains("example.invalid/kernel"),
        "{request}"
    );
}

#[test]
fn turns_that_read_the_project_or_are_short_never_become_notes() {
    let (_directory, root) = fixture();
    std::fs::write(root.join("secret.txt"), "internal plan").unwrap();
    let server = MockOllama::new(vec![
        call("read_file", json!({"path":"secret.txt"})),
        final_answer(&long_answer()),
    ]);
    let mut private = job(
        &server,
        &root,
        vec![message("user", "Summarise secret.txt for me please")],
    );
    private.settings.knowledge_notes = true;
    let transcript = run(private, |_| panic!("Read only"));
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(
        server.chats().len(),
        2,
        "no extraction call may follow a project-reading turn"
    );
    assert!(
        !root.parent().unwrap().join("knowledge-notes").exists()
            || std::fs::read_dir(root.parent().unwrap().join("knowledge-notes"))
                .unwrap()
                .all(|e| e.unwrap().file_name() == "INDEX.md")
    );

    let server = MockOllama::new(vec![final_answer("Sure, hello there.")]);
    let mut short = job(
        &server,
        &root,
        vec![message("user", "Hello, how are you today?")],
    );
    short.settings.knowledge_notes = true;
    let transcript = run(short, |_| panic!("No tools"));
    assert!(transcript.errors.is_empty());
    assert_eq!(server.chats().len(), 1);
}

#[test]
fn a_model_that_keeps_calling_tools_still_gets_a_real_final_answer_from_the_evidence() {
    let (_directory, root) = fixture();
    std::fs::write(root.join("notes.txt"), "Original text\n").unwrap();
    let server = MockOllama::new(vec![
        call("read_file", json!({"path":"notes.txt"})),
        // Violates the tool-free final instruction, as real models do.
        call("read_file", json!({"path":"other.txt"})),
        final_answer("notes.txt:1 says Original text; I did not read any other file."),
    ]);
    let mut task = job(&server, &root, vec![message("user", "Review notes.txt")]);
    task.settings.max_tool_rounds = 1;
    let transcript = run(task, |_| Decision::Deny);
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    let answer = &transcript.complete.last().unwrap().content;
    assert!(answer.contains("did not read any other file"), "{answer}");
    assert!(!answer.contains("Review incomplete"), "{answer}");
    let chats = server.chats();
    assert_eq!(chats.len(), 3);
    let flat = chats[2]["messages"].to_string();
    assert!(
        flat.contains("Original text") && flat.contains("Evidence gathered"),
        "{flat}"
    );
    assert!(!flat.contains("\"role\":\"tool\""));
}

fn verdict_reply(votes: [&str; 3]) -> Value {
    let list: Vec<Value> = votes
        .iter()
        .enumerate()
        .map(|(i, v)| json!({"id": i + 1, "verdict": v}))
        .collect();
    json!({"message":{"role":"assistant","content":json!({"verdicts":list}).to_string()},"done":true})
}

fn note_with_three_facts(data_dir: &Path) -> ollama_workbench::notes::NoteStore {
    use ollama_workbench::notes::{IncomingFact, NoteStore, BASIS_MODEL};
    let store = NoteStore::open(data_dir).unwrap();
    let facts: Vec<IncomingFact> = [
        "Each value in Rust has a single owner at a time.",
        "Rust frees memory with a tracing garbage collector at runtime.",
        "Moving a value transfers ownership to the new binding.",
    ]
    .iter()
    .map(|t| IncomingFact {
        text: (*t).into(),
        basis: BASIS_MODEL.into(),
        source_ix: vec![],
    })
    .collect();
    store.absorb("Rust Ownership", &facts, &[]).unwrap();
    store
}

fn review_with(
    server: &MockOllama,
    data_dir: &Path,
    models: &[&str],
    yield_after: Option<u64>,
) -> ollama_workbench::notes::ReviewReport {
    let settings = Settings {
        native_tool_probe: false,
        base_url: server.url.clone(),
        ..Default::default()
    };
    let client = ollama_workbench::ollama::OllamaClient::new(&server.url).unwrap();
    let models: Vec<String> = models.iter().map(|m| m.to_string()).collect();
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(ollama_workbench::notes::review_note(
            data_dir,
            &client,
            &settings,
            "rust-ownership",
            &models,
            yield_after,
            Arc::new(AtomicBool::new(false)),
        ))
        .unwrap()
}

#[test]
fn several_models_judge_each_fact_separately_and_their_votes_become_consensus() {
    use ollama_workbench::notes::ReviewStatus;
    let (_directory, root) = fixture();
    let data_dir = root.parent().unwrap().to_path_buf();
    let store = note_with_three_facts(&data_dir);
    let server = MockOllama::new(vec![
        verdict_reply(["true", "false", "true"]),
        verdict_reply(["true", "false", "unsure"]),
        verdict_reply(["true", "false", "true"]),
    ]);
    let report = review_with(
        &server,
        &data_dir,
        &["alpha:1b", "beta:2b", "gamma:3b"],
        None,
    );
    assert_eq!(
        (
            report.reviewed,
            report.agreed,
            report.mixed,
            report.disputed
        ),
        (3, 1, 1, 1),
        "{report:?}"
    );
    assert_eq!(report.models, ["alpha:1b", "beta:2b", "gamma:3b"]);

    // One question per model, each to a different model, with the topic and every fact.
    let chats = server.chats();
    assert_eq!(chats.len(), 3);
    let asked: Vec<&str> = chats.iter().map(|c| c["model"].as_str().unwrap()).collect();
    assert_eq!(asked, ["alpha:1b", "beta:2b", "gamma:3b"]);
    let prompt = chats[0]["messages"].to_string();
    assert!(
        prompt.contains("Rust Ownership") && prompt.contains("tracing garbage collector"),
        "{prompt}"
    );
    assert!(chats[0]
        .get("tools")
        .is_none_or(|t| t.as_array().is_some_and(Vec::is_empty)));

    let note = store.get("rust-ownership").unwrap().unwrap();
    let status: Vec<ReviewStatus> = note
        .entries
        .iter()
        .map(|e| e.consensus.unwrap().status())
        .collect();
    assert_eq!(
        status,
        [
            ReviewStatus::Agreed,
            ReviewStatus::Disputed,
            ReviewStatus::Mixed
        ]
    );
    assert!(
        note.reviews[0].contains("models: alpha:1b, beta:2b, gamma:3b"),
        "{:?}",
        note.reviews
    );

    // Reviewed facts are not reviewed again.
    let again = review_with(
        &server,
        &data_dir,
        &["alpha:1b", "beta:2b", "gamma:3b"],
        None,
    );
    assert_eq!(again.reviewed, 0);
    assert_eq!(server.chats().len(), 3);
}

#[test]
fn a_model_that_cannot_answer_is_not_counted_and_one_voter_is_not_a_consensus() {
    let (_directory, root) = fixture();
    let data_dir = root.parent().unwrap().to_path_buf();
    let store = note_with_three_facts(&data_dir);
    let garbage =
        json!({"message":{"role":"assistant","content":"They all look right to me."},"done":true});
    let server = MockOllama::new(vec![
        verdict_reply(["true", "true", "true"]),
        garbage.clone(),
        garbage,
    ]);
    let report = review_with(
        &server,
        &data_dir,
        &["alpha:1b", "beta:2b", "gamma:3b"],
        None,
    );
    assert_eq!(report.reviewed, 0, "{report:?}");
    assert_eq!(report.failed.len(), 2, "{report:?}");
    assert!(
        report.summary().contains("beta:2b could not answer"),
        "{}",
        report.summary()
    );
    let note = store.get("rust-ownership").unwrap().unwrap();
    assert!(
        note.entries.iter().all(|e| e.consensus.is_none()),
        "{:?}",
        note.entries
    );

    // With only one model there is nothing to compare, and no model is even asked.
    let server = MockOllama::new(vec![]);
    let single = review_with(&server, &data_dir, &["alpha:1b"], None);
    assert!(single.too_few_models && single.reviewed == 0);
    assert_eq!(server.chats().len(), 0);
}

#[test]
fn a_background_review_stops_when_another_chat_has_started() {
    let (_directory, root) = fixture();
    let data_dir = root.parent().unwrap().to_path_buf();
    note_with_three_facts(&data_dir);
    let server = MockOllama::new(vec![verdict_reply(["true", "true", "true"])]);
    // A marker that can never match the current chat counter stands for "a chat has started since".
    let report = review_with(&server, &data_dir, &["alpha:1b", "beta:2b"], Some(u64::MAX));
    assert!(
        report.stopped_for_chat && report.reviewed == 0,
        "{report:?}"
    );
    assert_eq!(
        server.chats().len(),
        0,
        "no model may be asked after a chat has started"
    );
}

#[test]
fn the_projects_own_instructions_reach_the_model_before_the_orientation() {
    let (_directory, root) = fixture();
    std::fs::write(
        root.join("AGENTS.md"),
        "Always run cargo test before saying a change is done.\n",
    )
    .unwrap();
    std::fs::write(root.join("main.rs"), "fn main() {}\n").unwrap();
    let server = MockOllama::new(vec![final_answer("ok")]);
    let transcript = run(
        job(&server, &root, vec![message("user", "What does main do?")]),
        |_| Decision::Approve,
    );
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert!(
        transcript
            .statuses
            .iter()
            .any(|s| s.contains("project's own instructions from AGENTS.md")),
        "{:?}",
        transcript.statuses
    );
    let sent = server.chats()[0]["messages"].as_array().unwrap().clone();
    let at = |needle: &str| {
        sent.iter()
            .position(|m| m["content"].as_str().is_some_and(|c| c.contains(needle)))
            .unwrap()
    };
    let (instructions, orientation, user) = (
        at("<project_instructions file=\"AGENTS.md\">"),
        at("<project_files>"),
        at("What does main do?"),
    );
    assert!(sent[instructions]["content"]
        .as_str()
        .unwrap()
        .contains("Always run cargo test"));
    assert_eq!(sent[instructions]["role"], "system");
    assert!(
        instructions < orientation && orientation + 1 == user,
        "{instructions} {orientation} {user}"
    );
}

#[test]
fn a_chat_about_a_project_is_given_a_ranked_map_of_its_code() {
    let (_directory, root) = fixture();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("src/engine.rs"),
        "pub struct Engine;\npub fn start_engine_loop() {}\n",
    )
    .unwrap();
    std::fs::write(
        root.join("src/app.rs"),
        "fn run() { start_engine_loop(); let _ = Engine; }\n",
    )
    .unwrap();
    let server = MockOllama::new(vec![final_answer("ok")]);
    let transcript = run(
        job(
            &server,
            &root,
            vec![message("user", "How does the engine start?")],
        ),
        |_| Decision::Approve,
    );
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert!(
        transcript
            .statuses
            .iter()
            .any(|s| s.contains("ranked map of the project's code")),
        "{:?}",
        transcript.statuses
    );
    let sent = server.chats()[0]["messages"].as_array().unwrap().clone();
    let at = |needle: &str| {
        sent.iter()
            .position(|m| m["content"].as_str().is_some_and(|c| c.contains(needle)))
            .unwrap()
    };
    let (map, orientation, user) = (
        at("<repo_map>"),
        at("<project_files>"),
        at("How does the engine start?"),
    );
    let text = sent[map]["content"].as_str().unwrap();
    assert!(
        text.contains("src/engine.rs\n  1: pub struct Engine"),
        "{text}"
    );
    assert!(
        text.contains("not file content")
            && text.contains("2 of 2 files with definitions are shown"),
        "{text}"
    );
    assert!(
        map < orientation && orientation + 1 == user,
        "{map} {orientation} {user}"
    );
}

#[test]
fn a_review_answered_without_reading_anything_is_retried_then_refused() {
    let (_directory, root) = fixture();
    std::fs::write(root.join("main.rs"), "fn main() {}\n").unwrap();
    let server = MockOllama::new(vec![
        final_answer("The code looks great, no bugs."),
        final_answer("Still looks great."),
    ]);
    let transcript = run(
        job(
            &server,
            &root,
            vec![message("user", "Review the code in this project for bugs")],
        ),
        |_| panic!("No approvals"),
    );
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(server.chats().len(), 2, "one bounded retry");
    let last = &transcript.complete.last().unwrap().content;
    assert!(last.contains("without reading any project file"), "{last}");
    assert!(!transcript
        .complete
        .iter()
        .any(|m| m.content.contains("looks great")));
}

#[test]
fn a_review_that_reads_files_lists_exactly_what_was_read() {
    let (_directory, root) = fixture();
    std::fs::write(root.join("main.rs"), "fn main() {}\nfn other() {}\n").unwrap();
    let server = MockOllama::new(vec![
        final_answer("Looks fine."),
        call("read_file", json!({"path":"main.rs"})),
        final_answer("main.rs:1 defines main."),
    ]);
    let transcript = run(
        job(
            &server,
            &root,
            vec![message("user", "Please review main.rs")],
        ),
        |_| panic!("No approvals"),
    );
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    let last = &transcript.complete.last().unwrap().content;
    assert!(last.starts_with("main.rs:1 defines main."), "{last}");
    assert!(last.contains("`main.rs`: lines 1-2 of 2"), "{last}");
}

#[test]
fn a_general_question_about_reviewing_is_not_forced_to_read_files() {
    let (_directory, root) = fixture();
    let server = MockOllama::new(vec![final_answer("Read it top to bottom.")]);
    let transcript = run(
        job(
            &server,
            &root,
            vec![message("user", "How do I review code well?")],
        ),
        |_| panic!("No approvals"),
    );
    assert_eq!(
        transcript.complete.last().unwrap().content,
        "Read it top to bottom."
    );
}

#[test]
fn gpu_out_of_memory_retries_the_same_request_with_half_the_context() {
    let (_directory, root) = fixture();
    let server = MockOllama::new(vec![
        json!({"error":"an error was encountered while running the model: CUDA error\nCUDA error: out of memory"}),
        final_answer("Answered after the retry."),
    ]);
    let mut request = job(&server, &root, vec![message("user", "Hello")]);
    request.settings.num_ctx = 16384;
    let transcript = run(request, |_| panic!("No approvals"));
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    let chats = server.chats();
    assert_eq!(chats.len(), 2);
    assert_eq!(chats[0]["options"]["num_ctx"], 16384);
    assert_eq!(chats[1]["options"]["num_ctx"], 8192);
    assert!(
        transcript
            .statuses
            .iter()
            .any(|s| s.contains("ran out of GPU memory")),
        "{:?}",
        transcript.statuses
    );
    assert_eq!(
        transcript.complete.last().unwrap().content,
        "Answered after the retry."
    );
}

#[test]
fn tool_json_inside_prose_is_not_run_and_gets_a_native_retry() {
    let (_directory, root) = fixture();
    std::fs::write(root.join("main.rs"), "fn main() {}\n").unwrap();
    let server = MockOllama::new(vec![
        final_answer("Sure, I'll open it first: {\"name\": \"read_file\", \"arguments\": {\"path\": \"main.rs\"}} and then summarize."),
        call("read_file", json!({"path":"main.rs"})),
        final_answer("main.rs defines an empty main."),
    ]);
    let transcript = run(
        job(&server, &root, vec![message("user", "Summarize main.rs")]),
        |_| panic!("No approvals"),
    );
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(
        transcript.unexecuted_tool_text,
        [(vec!["read_file".to_string()], true)]
    );
    assert_eq!(server.chats().len(), 3);
    assert_eq!(transcript.tools.len(), 1, "only the native call ran");
    assert!(transcript
        .complete
        .last()
        .unwrap()
        .content
        .starts_with("main.rs defines an empty main."));
}

#[test]
fn a_native_tool_call_that_runs_marks_the_model_verified_by_use() {
    let (_directory, root) = fixture();
    std::fs::write(root.join("main.rs"), "fn main() {}\n").unwrap();
    let server = MockOllama::new(vec![
        call("read_file", json!({"path":"main.rs"})),
        final_answer("Done."),
    ]);
    let transcript = run(
        job(&server, &root, vec![message("user", "Summarize main.rs")]),
        |_| panic!("No approvals"),
    );
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(
        ollama_workbench::routing::native_tool_record(root.parent().unwrap(), "test").unwrap(),
        ollama_workbench::routing::NativeToolRecord::Verified { successes: 1 }
    );
}

#[test]
fn a_model_that_writes_the_probe_call_as_text_is_recorded_and_the_user_is_told() {
    let (_directory, root) = fixture();
    std::fs::write(root.join("main.rs"), "fn main() {}\n").unwrap();
    let server = MockOllama::new(vec![
        // The probe: the model prints the call instead of making it.
        final_answer("{\"name\": \"report_ready\", \"arguments\": {\"ready\": true}}"),
        call("read_file", json!({"path":"main.rs"})),
        final_answer("Done."),
    ]);
    let mut request = job(&server, &root, vec![message("user", "Summarize main.rs")]);
    request.settings.native_tool_probe = true;
    let transcript = run(request, |_| panic!("No approvals"));
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert!(
        transcript
            .statuses
            .iter()
            .any(|s| s.contains("wrote its tool call as text")),
        "{:?}",
        transcript.statuses
    );
    let probe = &server.chats()[0];
    assert_eq!(probe["tools"][0]["function"]["name"], "report_ready");
    // The later real call verified it by use, which supersedes the probe failure.
    assert_eq!(
        ollama_workbench::routing::native_tool_record(root.parent().unwrap(), "test").unwrap(),
        ollama_workbench::routing::NativeToolRecord::Verified { successes: 1 }
    );
}

#[test]
fn the_probe_runs_once_per_model_and_a_native_answer_verifies_it() {
    let (_directory, root) = fixture();
    let server = MockOllama::new(vec![
        call("report_ready", json!({"ready": true})),
        final_answer("Hello."),
        final_answer("Hello again."),
    ]);
    let mut first = job(&server, &root, vec![message("user", "Hi")]);
    first.settings.native_tool_probe = true;
    run(first, |_| panic!("No approvals"));
    let mut second = job(&server, &root, vec![message("user", "Hi again")]);
    second.settings.native_tool_probe = true;
    let transcript = run(second, |_| panic!("No approvals"));
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(server.chats().len(), 3, "one probe, then two real requests");
    assert_eq!(
        ollama_workbench::routing::native_tool_record(root.parent().unwrap(), "test").unwrap(),
        ollama_workbench::routing::NativeToolRecord::Verified { successes: 1 }
    );
}

#[test]
fn a_code_request_searches_for_open_source_examples_before_writing() {
    let (_directory, root) = fixture();
    let server = MockOllama::new(vec![
        // The model writes code without searching: not accepted, asked to search.
        final_answer("<html>my own guess</html>"),
        call(
            "web_search",
            json!({"query":"open source html flight simulator"}),
        ),
        final_answer("<html>built on the example</html> Credits: https://example.invalid/ai"),
    ]);
    let mut request = web_job(&server, &root);
    request.settings.knowledge_notes = true;
    request.messages = vec![message("user", "write an html game flight simulator")];
    let transcript = run(request, |_| Decision::Approve);
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    let first = server.chats()[0]["messages"].to_string();
    assert!(first.contains("CODE REQUEST"), "{first}");
    assert!(
        transcript
            .statuses
            .iter()
            .any(|s| s.contains("asking it to search first")),
        "{:?}",
        transcript.statuses
    );
    assert_eq!(server.chats().len(), 3);
    let last = &transcript.complete.last().unwrap().content;
    assert!(last.contains("built on the example"), "{last}");
    assert!(
        !last.contains("no open-source examples were searched"),
        "{last}"
    );
    assert!(
        !transcript
            .statuses
            .iter()
            .any(|s| s.starts_with("Saved to knowledge notes")),
        "example code pages are not common facts: {:?}",
        transcript.statuses
    );
}

#[test]
fn a_reply_stuck_in_a_loop_is_stopped_and_its_repeats_removed() {
    let (_directory, root) = fixture();
    let mut looping = String::from("Here is the code:\n");
    for _ in 0..80 {
        looping.push_str("const targetVelocity = Math.sqrt(\n  this.aircraft.position.x ** 2 +\n  this.aircraft.position.z ** 2\n);\n");
    }
    let server = MockOllama::new(vec![json!({
        "message":{"role":"assistant","content":looping},
        "done":true,"done_reason":"length","eval_count":20000
    })]);
    let transcript = run(job(&server, &root, vec![message("user", "Hi")]), |_| {
        panic!("No tools")
    });
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    let last = &transcript.complete.last().unwrap().content;
    assert_eq!(last.matches("targetVelocity").count(), 1, "{last}");
    assert!(last.contains("started repeating"), "{last}");
    assert_eq!(server.chats().len(), 1, "a loop is not continued");
    assert!(transcript
        .stats
        .iter()
        .any(|s| s["interruption_kind"] == "repetition_loop"));
}

#[test]
fn a_long_partial_reply_is_continued_from_its_end_without_losing_the_request() {
    let (_directory, root) = fixture();
    let long: String = (0..2000)
        .map(|n| format!("line {n} of the program\n"))
        .collect();
    let server = MockOllama::new(vec![
        json!({"message":{"role":"assistant","content":long},"done":true,"done_reason":"length","eval_count":9000}),
        final_answer("the rest"),
    ]);
    let transcript = run(
        job(
            &server,
            &root,
            vec![message("user", "Write the whole program")],
        ),
        |_| panic!("No tools"),
    );
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    let second = server.chats()[1]["messages"].as_array().unwrap().clone();
    assert!(second
        .iter()
        .any(|m| m["content"] == "Write the whole program"));
    let partial = second
        .iter()
        .rev()
        .find(|m| m["role"] == "assistant")
        .unwrap();
    let text = partial["content"].as_str().unwrap();
    assert!(text.starts_with("[The first "), "{}", &text[..80]);
    assert!(text.ends_with("line 1999 of the program\n"));
}

#[test]
fn routing_keeps_the_model_that_answered_the_conversation_before() {
    let (_directory, root) = fixture();
    let server = MockOllama::with_tags(
        vec![final_answer("Same voice as before.")],
        json!({"capabilities":["completion","tools"]}),
        &["test", "earlier-model"],
    );
    let mut request = job(&server, &root, vec![message("user", "expand on that idea")]);
    request.settings.model_routing = true;
    request.settings.conversation_model = Some("earlier-model".into());
    let transcript = run(request, |_| panic!("No tools"));
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(server.chats()[0]["model"], "earlier-model");
    assert!(
        transcript
            .statuses
            .iter()
            .any(|s| s.contains("Keeping earlier-model for this conversation")),
        "{:?}",
        transcript.statuses
    );
}

#[test]
fn a_file_that_fits_in_the_context_is_read_whole_in_one_call() {
    let (_directory, root) = fixture();
    let body: String = (1..=1000)
        .map(|n| format!("let value_{n} = {n};\n"))
        .collect();
    std::fs::write(root.join("big.rs"), &body).unwrap();
    let server = MockOllama::new(vec![
        // The model asks for a small slice; Workbench returns all that fits.
        call("read_file", json!({"path":"big.rs","max_chars":2000})),
        final_answer("Read it all."),
    ]);
    let mut request = job(&server, &root, vec![message("user", "Explain big.rs")]);
    request.settings.num_ctx = 32768;
    let transcript = run(request, |_| panic!("No approvals"));
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    let saved: Value = serde_json::from_str(&transcript.tools[0].content).unwrap();
    assert_eq!(saved["complete"], true, "{}", saved["coverage"]);
    assert_eq!(saved["returned_end_line"], 1000);
    // And the model was given all of it, not a shortened view.
    let sent = server.chats()[1]["messages"].to_string();
    assert!(
        sent.contains("let value_1000 = 1000;"),
        "the model did not get the end of the file"
    );
}

#[test]
fn general_search_mode_uses_the_configured_web_provider() {
    let (_directory, root) = fixture();
    let server = MockOllama::new(vec![
        call(
            "web_search",
            json!({"query":"lisp open source", "mode":"general"}),
        ),
        final_answer("Found it."),
    ]);
    let transcript = run(web_job(&server, &root), |_| Decision::Approve);
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    let result: Value = serde_json::from_str(&transcript.tools[0].content).unwrap();
    assert_eq!(result["provider"]["type"], "searxng", "{result}");
}

#[test]
fn varied_outside_workspace_calls_stop_after_three_failures() {
    let (_directory, root) = fixture();
    let server = MockOllama::new(vec![
        call("list_files", json!({"path":"/outside-one"})),
        call("list_files", json!({"path":"/outside-two"})),
        call("list_files", json!({"path":"/outside-three"})),
        final_answer("These paths are outside the connected workspace; no files were inspected."),
    ]);
    let transcript = run(
        job(
            &server,
            &root,
            vec![message("user", "List files in the selected folder")],
        ),
        |_| Decision::Approve,
    );
    assert!(transcript.errors.is_empty(), "{:?}", transcript.errors);
    assert_eq!(transcript.tools.len(), 3);
    assert!(transcript
        .statuses
        .iter()
        .any(|s| s.contains("Three attempts")));
    let chats = server.chats();
    assert_eq!(chats.len(), 4);
    assert!(chats[3]
        .get("tools")
        .is_none_or(|v| v.is_null() || v.as_array().is_some_and(Vec::is_empty)));
}

#[test]
fn disabling_chat_tools_suppresses_the_automatic_health_lookup() {
    let server = MockOllama::new(vec![final_answer("An ordinary chat response.")]);
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("workspace");
    std::fs::create_dir(&root).unwrap();
    let prompt = ChatMessage { role: "user".into(), content: "What is chamomile?".into(), ..Default::default() };
    let mut job = job(&server, &root, vec![prompt]);
    job.settings.tools_enabled = false;
    job.settings.network_tools = true;
    job.settings.auto_approve_web = true;
    job.settings.local_sites_url = "http://127.0.0.1".into();
    let result = run(job, |_| panic!("unexpected approval"));
    assert!(result.errors.is_empty(), "{:?}", result.errors);
    assert!(!result.statuses.iter().any(|s| s.contains("Herb Helper")));
    assert_eq!(server.chats().len(), 1);
}
