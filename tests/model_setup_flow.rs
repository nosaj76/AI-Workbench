//! Real loopback pull/show/list exchanges for the first-run specialist pack.
//! Fixtures provide model metadata and small progress streams, never weights.
use ollama_workbench::{
    model_setup::{self, Candidate, ModelRole, ModelSetupStatus, SetupRequest, SetupState},
    settings::Settings,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

#[derive(Clone)]
struct FakeModel {
    tag: String,
    size: u64,
    metadata: Value,
    digest: String,
}

impl FakeModel {
    fn local(tag: &str, parameters: u64, size: u64, caps: &[&str]) -> Self {
        Self {
            tag: tag.into(),
            size,
            metadata: json!({
                "details":{"parameter_size":format!("{:.1}B",parameters as f64 / 1_000_000_000.0),"quantization_level":"Q4_K_M","family":"fixture"},
                "model_info":{"general.parameter_count":parameters},
                "capabilities":caps,
            }),
            digest: format!("{:x}", Sha256::digest(tag)),
        }
    }

    fn listed(&self) -> Value {
        json!({"name":self.tag,"size":self.size,"digest":self.digest,"details":self.metadata["details"]})
    }
}

struct ServerState {
    version: String,
    available: BTreeMap<String, FakeModel>,
    installed: BTreeMap<String, FakeModel>,
    requests: Vec<(String, Value)>,
    fail_once: Option<String>,
    stall_once: Option<String>,
    remove_on_final_pull: Option<String>,
}

struct Server {
    url: String,
    state: Arc<Mutex<ServerState>>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Server {
    fn new(initial: Vec<FakeModel>, available: Vec<FakeModel>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let state = Arc::new(Mutex::new(ServerState {
            version: "0.18.0".into(),
            installed: initial.into_iter().map(|m| (m.tag.clone(), m)).collect(),
            available: available.into_iter().map(|m| (m.tag.clone(), m)).collect(),
            requests: vec![],
            fail_once: None,
            stall_once: None,
            remove_on_final_pull: None,
        }));
        let shared = state.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let worker = thread::spawn(move || {
            while !stopping.load(Ordering::Relaxed) {
                let mut socket = match listener.accept() {
                    Ok((socket, _)) => socket,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("Fixture listener: {error}"),
                };
                socket
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let (path, body) = read_request(&mut socket);
                let mut current = shared.lock().unwrap();
                current.requests.push((path.clone(), body.clone()));
                let mut status = "200 OK";
                let mut stalled = false;
                let response = match path.as_str() {
                    "/api/version" => format!("{}\n", json!({"version":current.version})),
                    "/api/tags" => format!("{}\n", json!({"models":current.installed.values().map(FakeModel::listed).collect::<Vec<_>>()})),
                    "/api/show" => {
                        let name = body["model"].as_str().unwrap();
                        if let Some(model) = current.installed.get(name) {
                            format!("{}\n", model.metadata)
                        } else {
                            status = "404 Not Found";
                            format!("{}\n", json!({"error":"model is not installed"}))
                        }
                    }
                    "/api/pull" => {
                        assert_eq!(body["stream"], true);
                        let name = body["model"].as_str().unwrap();
                        if current.fail_once.as_deref() == Some(name) {
                            current.fail_once = None;
                            status = "503 Service Unavailable";
                            format!("{}\n", json!({"error":"fixture download interrupted"}))
                        } else if current.stall_once.as_deref() == Some(name) {
                            current.stall_once = None;
                            stalled = true;
                            format!("{}\n", json!({"status":"pulling layer","digest":"sha256:fixture-layer","completed":1024,"total":2048}))
                        } else if let Some(model) = current.available.get(name).cloned() {
                            current.installed.insert(name.into(), model);
                            if name == "translator:4b" {
                                if let Some(remove) = current.remove_on_final_pull.take() {
                                    current.installed.remove(&remove);
                                }
                            }
                            format!("{}\n{}\n{}\n",
                                json!({"status":"pulling layer","digest":"sha256:fixture-layer","completed":1024,"total":2048}),
                                json!({"status":"verifying sha256 digest"}),
                                json!({"status":"success"}))
                        } else {
                            status = "404 Not Found";
                            format!("{}\n", json!({"error":"unknown catalog model"}))
                        }
                    }
                    other => panic!("Unexpected setup endpoint {other}; setup must not infer, unload, or delete"),
                };
                drop(current);
                let length = if stalled {
                    response.len() + 4096
                } else {
                    response.len()
                };
                let header = format!("HTTP/1.1 {status}\r\nContent-Type: application/x-ndjson\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n");
                let _ = socket.write_all(header.as_bytes());
                let _ = socket.write_all(response.as_bytes());
                if stalled {
                    socket
                        .set_read_timeout(Some(Duration::from_millis(50)))
                        .unwrap();
                    let mut b = [0u8; 1];
                    while !stopping.load(Ordering::Relaxed) {
                        match socket.read(&mut b) {
                            Ok(0) => break,
                            Ok(_) => {}
                            Err(e)
                                if matches!(
                                    e.kind(),
                                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                                ) => {}
                            Err(_) => break,
                        }
                    }
                }
            }
        });
        Self {
            url,
            state,
            stop,
            worker: Some(worker),
        }
    }

    fn standard() -> Self {
        let general = FakeModel::local(
            "general:latest",
            4_210_000_000,
            3_300_000_000,
            &["completion", "tools", "thinking", "vision"],
        );
        let mut general = general;
        general.metadata["projector_info"] = json!({"general.parameter_count":334_000_000});
        let coder = FakeModel::local(
            "coder:7b",
            7_620_000_000,
            4_700_000_000,
            &["completion", "tools"],
        );
        let translator = FakeModel::local(
            "translator:4b",
            4_300_000_000,
            3_300_000_000,
            &["completion", "vision"],
        );
        Self::new(vec![general], vec![coder, translator])
    }

    fn requests(&self) -> Vec<(String, Value)> {
        self.state.lock().unwrap().requests.clone()
    }

    fn pulls(&self) -> Vec<String> {
        self.requests()
            .iter()
            .filter(|(p, _)| p == "/api/pull")
            .map(|(_, b)| b["model"].as_str().unwrap().to_owned())
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
        let read = socket.read(&mut buffer).unwrap();
        assert!(read > 0);
        bytes.extend_from_slice(&buffer[..read]);
        if let Some(at) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
            break at + 4;
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
        .to_owned();
    let length = header
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            if name.eq_ignore_ascii_case("content-length") {
                value.trim().parse::<usize>().ok()
            } else {
                None
            }
        })
        .unwrap_or(0);
    while bytes.len() < end + length {
        let read = socket.read(&mut buffer).unwrap();
        assert!(read > 0);
        bytes.extend_from_slice(&buffer[..read]);
    }
    let value = if length == 0 {
        Value::Null
    } else {
        serde_json::from_slice(&bytes[end..end + length]).unwrap()
    };
    (path, value)
}

fn candidate(
    id: &str,
    tag: &str,
    parameters: f64,
    bytes: u64,
    roles: Vec<ModelRole>,
    caps: &[&str],
) -> Candidate {
    Candidate {
        id: id.into(),
        tag: tag.into(),
        roles,
        total_parameters_b: parameters,
        download_bytes: bytes,
        license: "Fixture license; no real model weights".into(),
        source_url: "https://ollama.com/library/fixture".into(),
        required_capabilities: caps.iter().map(|s| (*s).into()).collect(),
        minimum_ollama: Some("0.17.5".into()),
        installed_aliases: vec![],
        required_quantization: Some("Q4_K_M".into()),
    }
}

fn request(server: &Server, dir: &Path) -> SetupRequest {
    let general = candidate(
        "general",
        "general:4b",
        4.6,
        3_300_000_000,
        vec![
            ModelRole::Orchestrator,
            ModelRole::General,
            ModelRole::Reasoning,
            ModelRole::Writing,
            ModelRole::Vision,
        ],
        &["completion", "tools", "thinking", "vision"],
    );
    let coder = candidate(
        "coder",
        "coder:7b",
        7.62,
        4_700_000_000,
        vec![ModelRole::Coding],
        &["completion", "tools"],
    );
    let translator = candidate(
        "translator",
        "translator:4b",
        4.3,
        3_300_000_000,
        vec![ModelRole::Translation],
        &["completion"],
    );
    let mut request = SetupRequest::new(
        Settings {
            native_tool_probe: false,
            base_url: server.url.clone(),
            ..Default::default()
        },
        dir.to_path_buf(),
        vec![general, coder, translator],
    );
    request.candidates[0]
        .installed_aliases
        .push("general:latest".into());
    request
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

#[test]
fn starter_pack_reuses_verified_aliases_pulls_sequentially_and_only_then_exposes_routes() {
    let server = Server::standard();
    let temp = tempfile::tempdir().unwrap();
    let request = request(&server, temp.path());
    let original_settings = serde_json::to_value(&request.settings).unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    let mut events = vec![];
    runtime().block_on(async {
        let plan = model_setup::prepare(&request, cancel.clone(), |_| {})
            .await
            .unwrap();
        assert!(plan.can_run());
        assert_eq!(plan.models.len(), 3);
        assert_eq!(
            plan.models[0].installed_name.as_deref(),
            Some("general:latest")
        );
        assert_eq!(
            plan.models[0]
                .verification
                .as_ref()
                .unwrap()
                .total_parameters,
            4_544_000_000
        );
        assert_eq!(plan.estimated_download_bytes, 8_000_000_000);
        assert_eq!(plan.total_disk_bytes, 11_300_000_000);
        assert!(server.pulls().is_empty(), "Preview must not download");
        let run = model_setup::run(&request, &plan, cancel.clone(), |event| events.push(event))
            .await
            .unwrap();
        assert!(run.completed(), "{:?}", run.error);
        let roles = run.verified_models().unwrap();
        assert_eq!(roles[&ModelRole::Orchestrator], vec!["general:latest"]);
        assert_eq!(roles[&ModelRole::Coding], vec!["coder:7b"]);
        assert_eq!(roles[&ModelRole::Translation], vec!["translator:4b"]);
        assert!(run.models[0].reused);
        assert!(!run.models[1].reused);
        assert!(run
            .models
            .iter()
            .all(|m| m.verification.as_ref().unwrap().digest.len() == 64));
        let mut inconsistent_receipt = run.clone();
        inconsistent_receipt.models[0]
            .verification
            .as_mut()
            .unwrap()
            .total_parameters = 1;
        assert!(!inconsistent_receipt.completed());
        assert!(inconsistent_receipt.verified_models().is_err());
        assert_eq!(
            model_setup::load_run(temp.path(), &run.id)
                .unwrap()
                .unwrap()
                .state,
            SetupState::Complete
        );
        let again = model_setup::run(&request, &plan, cancel, |_| {})
            .await
            .unwrap();
        assert!(again.completed());
        assert!(again.models.iter().all(|m| m.reused));
    });
    assert_eq!(server.pulls(), vec!["coder:7b", "translator:4b"]);
    let requests = server.requests();
    let code_pull = requests
        .iter()
        .position(|(p, b)| p == "/api/pull" && b["model"] == "coder:7b")
        .unwrap();
    let translation_pull = requests
        .iter()
        .position(|(p, b)| p == "/api/pull" && b["model"] == "translator:4b")
        .unwrap();
    assert!(requests[code_pull + 1..translation_pull]
        .iter()
        .any(|(p, b)| p == "/api/show" && b["model"] == "coder:7b"));
    assert!(events
        .iter()
        .any(|e| e.completed_bytes == 1024 && e.total_bytes == Some(2048)));
    assert_eq!(
        serde_json::to_value(&request.settings).unwrap(),
        original_settings,
        "Backend must not activate routing itself"
    );
}

#[test]
fn failed_pull_keeps_completed_models_and_resume_only_pulls_the_missing_member() {
    let server = Server::standard();
    server.state.lock().unwrap().fail_once = Some("translator:4b".into());
    let temp = tempfile::tempdir().unwrap();
    let request = request(&server, temp.path());
    runtime().block_on(async {
        let cancel = Arc::new(AtomicBool::new(false));
        let plan = model_setup::prepare(&request, cancel.clone(), |_| {})
            .await
            .unwrap();
        let partial = model_setup::run(&request, &plan, cancel.clone(), |_| {})
            .await
            .unwrap();
        assert_eq!(partial.state, SetupState::Failed);
        assert!(!partial.completed());
        assert!(partial.verified_models().is_err());
        assert_eq!(partial.models[1].state, ModelSetupStatus::Ready);
        assert_eq!(partial.models[2].state, ModelSetupStatus::Failed);
        let saved = model_setup::load_run(temp.path(), &partial.id)
            .unwrap()
            .unwrap();
        assert_eq!(
            saved.models[1].verification.as_ref().unwrap().model_name,
            "coder:7b"
        );
        let refreshed = model_setup::prepare(&request, cancel.clone(), |_| {})
            .await
            .unwrap();
        assert_eq!(refreshed.estimated_download_bytes, 3_300_000_000);
        let resumed = model_setup::run(&request, &refreshed, cancel, |_| {})
            .await
            .unwrap();
        assert!(resumed.completed(), "{:?}", resumed.error);
        assert!(resumed.models[1].reused);
        assert_eq!(resumed.models[2].attempts, 2);
    });
    assert_eq!(
        server.pulls(),
        vec!["coder:7b", "translator:4b", "translator:4b"]
    );
}

#[test]
fn cancellation_during_a_pull_releases_the_endpoint_and_can_resume_without_fabricated_success() {
    let server = Server::standard();
    server.state.lock().unwrap().stall_once = Some("coder:7b".into());
    let temp = tempfile::tempdir().unwrap();
    let request = request(&server, temp.path());
    runtime().block_on(async {
        let cancel = Arc::new(AtomicBool::new(false));
        let plan = model_setup::prepare(&request, cancel.clone(), |_| {})
            .await
            .unwrap();
        let start = Instant::now();
        let cancelled = model_setup::run(&request, &plan, cancel.clone(), |p| {
            if p.stage == "pulling layer" {
                cancel.store(true, Ordering::Relaxed);
            }
        })
        .await
        .unwrap();
        assert!(start.elapsed() < Duration::from_secs(2));
        assert_eq!(cancelled.state, SetupState::Cancelled);
        assert!(!cancelled.completed());
        assert_eq!(cancelled.models[0].state, ModelSetupStatus::Ready);
        assert_eq!(cancelled.models[1].state, ModelSetupStatus::Cancelled);
        assert!(cancelled.models[1].verification.is_none());
        assert_eq!(cancelled.models[2].state, ModelSetupStatus::Queued);
        let resumed = model_setup::run(&request, &plan, Arc::new(AtomicBool::new(false)), |_| {})
            .await
            .unwrap();
        assert!(resumed.completed(), "{:?}", resumed.error);
    });
    assert_eq!(
        server.pulls(),
        vec!["coder:7b", "coder:7b", "translator:4b"]
    );
}

#[test]
fn actual_total_parameters_after_pull_override_the_catalog_and_block_activation() {
    let server = Server::standard();
    server
        .state
        .lock()
        .unwrap()
        .available
        .get_mut("coder:7b")
        .unwrap()
        .metadata["projector_info"] = json!({"general.parameter_count":3_000_000_000_u64});
    let temp = tempfile::tempdir().unwrap();
    let request = request(&server, temp.path());
    runtime().block_on(async {
        let cancel = Arc::new(AtomicBool::new(false));
        let plan = model_setup::prepare(&request, cancel.clone(), |_| {})
            .await
            .unwrap();
        let run = model_setup::run(&request, &plan, cancel, |_| {})
            .await
            .unwrap();
        assert_eq!(run.state, SetupState::Failed);
        assert!(run.error.as_deref().unwrap().contains("10.620B"));
        assert!(!run.completed());
        assert!(run.verified_models().is_err());
    });
    assert_eq!(server.pulls(), vec!["coder:7b"]);
    assert!(
        server
            .state
            .lock()
            .unwrap()
            .installed
            .contains_key("coder:7b"),
        "Rejected weights must not be deleted automatically"
    );
}

#[test]
fn setup_version_gate_stops_before_any_model_download_or_listing() {
    let server = Server::standard();
    server.state.lock().unwrap().version = "0.16.9".into();
    let temp = tempfile::tempdir().unwrap();
    let request = request(&server, temp.path());
    let result = runtime().block_on(model_setup::prepare(
        &request,
        Arc::new(AtomicBool::new(false)),
        |_| {},
    ));
    assert!(format!("{:#}", result.unwrap_err()).contains("0.17.5"));
    assert_eq!(
        server
            .requests()
            .iter()
            .map(|(p, _)| p.as_str())
            .collect::<Vec<_>>(),
        vec!["/api/version"]
    );
}

#[test]
fn rounded_parameter_labels_cannot_authorize_a_hard_ten_billion_cap() {
    let server = Server::standard();
    {
        let mut state = server.state.lock().unwrap();
        let model = state.available.get_mut("coder:7b").unwrap();
        model.metadata["model_info"] = json!({});
        model.metadata["details"]["parameter_size"] = json!("10B");
    }
    let temp = tempfile::tempdir().unwrap();
    let request = request(&server, temp.path());
    runtime().block_on(async {
        let cancel = Arc::new(AtomicBool::new(false));
        let plan = model_setup::prepare(&request, cancel.clone(), |_| {})
            .await
            .unwrap();
        let run = model_setup::run(&request, &plan, cancel, |_| {})
            .await
            .unwrap();
        assert_eq!(run.state, SetupState::Failed);
        assert!(!run.completed());
        assert!(run.verified_models().is_err());
        assert!(run.error.as_deref().unwrap().contains("exact"));
        assert_eq!(run.models[1].state, ModelSetupStatus::Failed);
    });
    assert_eq!(server.pulls(), vec!["coder:7b"]);
}

#[test]
fn reviewed_setup_cannot_be_retargeted_and_cloud_exact_tags_are_never_overwritten() {
    let server = Server::standard();
    let another = Server::standard();
    let temp = tempfile::tempdir().unwrap();
    let mut request = request(&server, temp.path());
    runtime().block_on(async {
        let cancel = Arc::new(AtomicBool::new(false));
        let plan = model_setup::prepare(&request, cancel.clone(), |_| {})
            .await
            .unwrap();
        request.settings.base_url = another.url.clone();
        assert!(model_setup::run(&request, &plan, cancel.clone(), |_| {})
            .await
            .is_err());
        assert!(another.requests().is_empty());
        request.settings.base_url = server.url.clone();
        let mut remote = FakeModel::local(
            "coder:7b",
            7_620_000_000,
            4_700_000_000,
            &["completion", "tools"],
        );
        remote.metadata["remote_model"] = json!("large-cloud");
        remote.metadata["remote_host"] = json!("https://ollama.com");
        server
            .state
            .lock()
            .unwrap()
            .installed
            .insert(remote.tag.clone(), remote);
        let blocked = model_setup::prepare(&request, cancel.clone(), |_| {})
            .await
            .unwrap();
        assert!(!blocked.can_run());
        assert!(blocked.models[1]
            .issue
            .as_deref()
            .unwrap()
            .contains("remote"));
        assert!(model_setup::run(&request, &blocked, cancel, |_| {})
            .await
            .is_err());
    });
    assert!(server.pulls().is_empty());
}

#[test]
fn removing_an_earlier_model_during_later_downloads_prevents_complete_role_assignment() {
    let server = Server::standard();
    server.state.lock().unwrap().remove_on_final_pull = Some("general:latest".into());
    let temp = tempfile::tempdir().unwrap();
    let request = request(&server, temp.path());
    runtime().block_on(async {
        let cancel = Arc::new(AtomicBool::new(false));
        let plan = model_setup::prepare(&request, cancel.clone(), |_| {})
            .await
            .unwrap();
        let run = model_setup::run(&request, &plan, cancel, |_| {})
            .await
            .unwrap();
        assert_eq!(run.state, SetupState::Failed);
        assert!(!run.completed());
        assert_eq!(run.models[0].state, ModelSetupStatus::Failed);
        assert!(run.error.as_deref().unwrap().contains("removed"));
        assert!(run.verified_models().is_err());
    });
}
