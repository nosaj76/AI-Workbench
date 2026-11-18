//! Live smoke tests against a real local Ollama server, real models and (for
//! search) the real internet. They are ignored by default; run explicitly:
//!
//!   cargo test --test live_ollama -- --ignored --nocapture --test-threads=1
//!
//! Settings come from the user's saved Workbench settings when present (so the
//! same models and routing are exercised), with network tools enabled for the
//! search tests only. File changes in the temporary workspace and web requests
//! are approved; arbitrary commands are denied. Saved settings are never modified.
use ollama_workbench::{
    ollama::ChatMessage,
    runtime::{self, ChatEvent, ChatJob},
    settings::Settings,
};
use std::{
    path::Path,
    sync::{atomic::AtomicBool, mpsc, Arc},
    time::{Duration, Instant},
};

fn live_settings() -> Settings {
    let mut settings = ollama_workbench::settings::data_dir()
        .ok()
        .and_then(|dir| Settings::load(&dir).ok())
        .unwrap_or_default();
    if let Ok(model) = std::env::var("WORKBENCH_LIVE_MODEL") {
        settings.model = model;
        settings.model_routing = false;
        // An explicit test model must not be replaced by a profile's model.
        for profile in &mut settings.profiles {
            profile.model.clear();
        }
    }
    settings.tools_enabled = true;
    settings
}

#[derive(Default, Debug)]
struct Outcome {
    tools: Vec<(String, String)>,
    statuses: Vec<String>,
    approvals: Vec<String>,
    answer: String,
    errors: Vec<String>,
}

fn run_live(settings: Settings, workspace: Option<&Path>, data: &Path, prompt: &str) -> Outcome {
    let (tx, rx) = mpsc::channel();
    // Same as the desktop app: the selected profile's model, if it names one.
    let model = settings.active_model().to_owned();
    runtime::spawn_chat(
        ChatJob {
            settings,
            messages: vec![ChatMessage {
                role: "user".into(),
                content: prompt.into(),
                ..Default::default()
            }],
            workspace: workspace.map(Path::to_path_buf),
            model,
            engine: None,
            retrieval_path: data.join("knowledge.sqlite3"),
        },
        tx,
        Arc::new(AtomicBool::new(false)),
    );
    let started = Instant::now();
    let mut outcome = Outcome::default();
    loop {
        let event = rx
            .recv_timeout(Duration::from_secs(600))
            .expect("live chat did not finish within 10 minutes");
        let t = started.elapsed().as_secs_f32();
        match event {
            ChatEvent::Begin(model) => eprintln!("[{t:6.1}s] model: {model}"),
            ChatEvent::Status(s) => {
                eprintln!("[{t:6.1}s] status: {s}");
                outcome.statuses.push(s);
            }
            ChatEvent::ToolStarted { name, summary, .. } => {
                eprintln!("[{t:6.1}s] tool start: {name} {summary}")
            }
            ChatEvent::ToolFinished { ok, preview, .. } => {
                let short: String = preview.chars().take(300).collect();
                eprintln!("[{t:6.1}s] tool end ok={ok}: {short}")
            }
            ChatEvent::Approval { action, reply } => {
                eprintln!("[{t:6.1}s] approve: {}", action.title);
                outcome.approvals.push(action.title.clone());
                let approved = action.operation.approval_kind()
                    != ollama_workbench::tools::ApprovalKind::Command;
                reply.send(approved).unwrap();
            }
            ChatEvent::Append(entry) if entry.message.role == "tool" => {
                outcome.tools.push((
                    entry.message.tool_name.clone().unwrap_or_default(),
                    entry.message.content.clone(),
                ));
            }
            ChatEvent::Complete(message) => outcome.answer = message.content,
            ChatEvent::Error(e) => {
                eprintln!("[{t:6.1}s] ERROR: {e}");
                outcome.errors.push(e)
            }
            ChatEvent::Finished => break,
            _ => {}
        }
    }
    eprintln!("--- answer ---\n{}\n--------------", outcome.answer);
    outcome
}

fn succeeded(content: &str) -> bool {
    let value: serde_json::Value = serde_json::from_str(content).unwrap_or_default();
    value.get("error").is_none()
        && value.get("denied").is_none()
        && value.get("ok").and_then(|v| v.as_bool()) != Some(false)
}

#[test]
#[ignore = "needs a local Ollama server, a tool-capable model and internet access"]
fn live_web_search_without_a_project_returns_real_results() {
    let data = tempfile::tempdir().unwrap();
    let mut settings = live_settings();
    settings.network_tools = true;
    let outcome = run_live(
        settings,
        None,
        data.path(),
        "Search the web for the latest stable Rust release and tell me its version number.",
    );
    assert!(outcome.errors.is_empty(), "{:?}", outcome.errors);
    let searches: Vec<_> = outcome
        .tools
        .iter()
        .filter(|(name, _)| name == "web_search")
        .collect();
    assert!(
        !searches.is_empty(),
        "model never called web_search: {outcome:?}"
    );
    // An empty result list is not a working search (a rate-limited provider
    // used to be reported that way), so require real results.
    let with_results = searches.iter().any(|(_, content)| {
        succeeded(content)
            && serde_json::from_str::<serde_json::Value>(content)
                .ok()
                .and_then(|v| {
                    v.get("results")
                        .and_then(|r| r.as_array())
                        .map(|r| !r.is_empty())
                })
                .unwrap_or(false)
    });
    assert!(
        with_results,
        "no web_search returned any results: {searches:?}"
    );
    let has_version = outcome
        .answer
        .split(|c: char| !(c.is_ascii_digit() || c == '.'))
        .any(|word| word.starts_with("1.") && word.matches('.').count() >= 1 && word.len() >= 4);
    assert!(
        has_version,
        "answer names no Rust version: {}",
        outcome.answer
    );
}

#[test]
#[ignore = "needs a local Ollama server and a tool-capable model"]
fn live_project_file_read_and_write() {
    let data = tempfile::tempdir().unwrap();
    let project = tempfile::tempdir().unwrap();
    std::fs::create_dir(project.path().join("src")).unwrap();
    std::fs::write(
        project.path().join("src/lib.rs"),
        "/// Adds two numbers.\npub fn add(a: i32, b: i32) -> i32 { a + b }\n\n\
         /// Multiplies two numbers.\npub fn multiply(a: i32, b: i32) -> i32 { a * b }\n",
    )
    .unwrap();
    let mut settings = live_settings();
    settings.network_tools = false;
    let outcome = run_live(
        settings,
        Some(project.path()),
        data.path(),
        "Read src/lib.rs, then create a file named NOTES.md in the project that lists \
         each public function name with a one-line description.",
    );
    assert!(outcome.errors.is_empty(), "{:?}", outcome.errors);
    let read = outcome
        .tools
        .iter()
        .any(|(name, content)| name == "read_file" && succeeded(content));
    assert!(read, "src/lib.rs was not read successfully: {outcome:?}");
    let notes = std::fs::read_to_string(project.path().join("NOTES.md"))
        .unwrap_or_else(|e| panic!("NOTES.md was not written ({e}): {outcome:?}"));
    assert!(
        notes.contains("add") && notes.contains("multiply"),
        "NOTES.md does not describe both functions:\n{notes}"
    );
}

#[test]
#[ignore = "needs a local Ollama server and several installed models with routing enabled"]
fn live_routing_remembers_models_that_fail_native_tool_calls() {
    let data = tempfile::tempdir().unwrap();
    let project = tempfile::tempdir().unwrap();
    std::fs::write(
        project.path().join("lib.rs"),
        "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
    )
    .unwrap();
    let settings = live_settings();
    if !settings.model_routing {
        eprintln!("routing is disabled in the saved settings; nothing to check");
        return;
    }
    // A coding task, so routing considers the coding specialist.
    let prompt = "Read lib.rs and explain what the Rust function add returns.";
    let retried = |o: &Outcome| o.statuses.iter().any(|s| s.contains("were not executed"));
    let first = run_live(settings.clone(), Some(project.path()), data.path(), prompt);
    assert!(first.errors.is_empty(), "{:?}", first.errors);
    let second = run_live(settings, Some(project.path()), data.path(), prompt);
    assert!(second.errors.is_empty(), "{:?}", second.errors);
    if retried(&first) {
        assert!(
            !retried(&second),
            "the model that failed native tool calls was routed again: {:?}",
            second.statuses
        );
    }
    if !retried(&first) {
        // Every routed model called tools natively, so there was no failure to
        // remember. The store itself is covered by routing's unit tests.
        eprintln!("no native tool-call failure occurred; routing memory not exercised");
    }
    assert!(second
        .tools
        .iter()
        .any(|(name, c)| name == "read_file" && succeeded(c)));
}

#[test]
#[ignore = "needs a local Ollama server; uses the saved Code review profile"]
fn live_code_review_profile_finds_a_real_bug() {
    let data = tempfile::tempdir().unwrap();
    let project = tempfile::tempdir().unwrap();
    std::fs::create_dir(project.path().join("src")).unwrap();
    std::fs::write(
        project.path().join("src/lib.rs"),
        "/// Returns the last item.\npub fn last(items: &[i32]) -> i32 {\n    items[items.len()]\n}\n",
    )
    .unwrap();
    let mut settings = live_settings();
    settings.network_tools = false;
    match settings
        .profiles
        .iter()
        .position(|p| p.name == "Code review")
    {
        Some(index) => settings.selected_profile = index,
        None => {
            eprintln!("no Code review profile in the saved settings");
            return;
        }
    }
    let expected = settings.active_model().to_owned();
    let outcome = run_live(
        settings,
        Some(project.path()),
        data.path(),
        "Review src/lib.rs for bugs.",
    );
    assert!(outcome.errors.is_empty(), "{:?}", outcome.errors);
    assert!(
        outcome.statuses.iter().all(|s| !s.starts_with("Routed to")),
        "routing replaced the profile model {expected}: {:?}",
        outcome.statuses
    );
    assert!(outcome
        .tools
        .iter()
        .any(|(name, c)| name == "read_file" && succeeded(c)));
    let answer = outcome.answer.to_lowercase();
    assert!(
        [
            "out of bounds",
            "out-of-bounds",
            "len() - 1",
            "len()-1",
            "panic",
            "off-by-one",
            "off by one"
        ]
        .iter()
        .any(|k| answer.contains(k)),
        "review missed the out-of-bounds index:\n{}",
        outcome.answer
    );
}

#[test]
#[ignore = "needs a local Ollama server, a text browser and internet access"]
fn live_qa_workflow_answers_then_reuses_the_cache() {
    use ollama_workbench::qa;
    let data = tempfile::tempdir().unwrap();
    let mut settings = live_settings();
    settings.network_tools = true;
    let model = settings.active_model().to_owned();
    let (tx, rx) = mpsc::channel();
    let printer = std::thread::spawn(move || {
        while let Ok(event) = rx.recv() {
            if let ollama_workbench::agents::AgentEvent::Status(s) = event {
                eprintln!("status: {s}");
            }
        }
    });
    let ask = |question: &str| {
        let request = qa::QaRequest::new(
            settings.clone(),
            data.path().to_path_buf(),
            question.into(),
            model.clone(),
        );
        qa::run_qa(request, &tx, Arc::new(AtomicBool::new(false))).expect("Q&A run")
    };

    // A settled how-to: answered online, kept as a markdown file.
    let first = ask("How do I undo the last git commit?");
    eprintln!("{}", first.report());
    assert!(!first.cached);
    assert!(first.sources.iter().any(|s| s.method != "search_snippet"));
    let lower = first.answer.to_lowercase();
    assert!(
        lower.contains("git reset") || lower.contains("git revert"),
        "{}",
        first.answer
    );
    assert!(!first.summary.is_empty(), "no short answer");
    let file = data.path().join(qa::LIBRARY_DIR).join(&first.path);
    let markdown = std::fs::read_to_string(&file).expect("library file written");
    assert!(markdown.contains("# How do I undo the last git commit?"));
    assert!(data.path().join(qa::LIBRARY_DIR).join("INDEX.md").exists());
    eprintln!(
        "kind: {:?} topic: {} file: {}",
        first.kind, first.topic, first.path
    );

    // A reworded question is answered from the library, without any search.
    let second = ask("undo last git commit");
    assert!(second.cached, "the reworded question was searched again");
    assert_eq!(second.answer, first.answer);

    // A question about "the latest" release is always time-sensitive.
    let latest = ask("What is the latest stable Rust release?");
    eprintln!("{}", latest.report());
    assert_eq!(latest.kind, qa::AnswerKind::TimeSensitive);
    assert!(latest.refresh_after.is_some());

    let cache = qa::QaCache::open(data.path()).unwrap();
    let facts = cache.facts().unwrap();
    eprintln!("facts table: {facts:#?}");
    if first.kind == qa::AnswerKind::Factual {
        assert_eq!(facts.len(), 1);
        assert!(!facts[0].question.is_empty() && !facts[0].answer.is_empty());
    }
    assert!(facts.iter().all(|f| f.id
        != cache
            .lookup("latest stable rust release")
            .unwrap()
            .unwrap()
            .id));
    drop(tx);
    printer.join().unwrap();
}

#[test]
#[ignore = "needs a local Ollama server, a text browser and internet access"]
fn live_chat_uses_qa_lookup_and_its_cache() {
    let data = tempfile::tempdir().unwrap();
    // The chat engine keeps its Q&A cache in the application data directory;
    // point it at a temporary one so the real cache is untouched.
    // (Saved settings are loaded first, from the real directory.)
    let mut settings = live_settings();
    // Restored on every exit, including a failed assertion: later tests in the same
    // run must load the real saved settings, not an empty temporary directory.
    struct RestoreDataDir(Option<std::ffi::OsString>);
    impl Drop for RestoreDataDir {
        fn drop(&mut self) {
            match self.0.take() {
                Some(value) => std::env::set_var("OLLAMA_WORKBENCH_DATA_DIR", value),
                None => std::env::remove_var("OLLAMA_WORKBENCH_DATA_DIR"),
            }
        }
    }
    let _restore = RestoreDataDir(std::env::var_os("OLLAMA_WORKBENCH_DATA_DIR"));
    std::env::set_var("OLLAMA_WORKBENCH_DATA_DIR", data.path());
    settings.network_tools = true;
    let prompt = "How do I rename a local git branch? Check community Q&A sites.";
    let first = run_live(settings.clone(), None, data.path(), prompt);
    assert!(first.errors.is_empty(), "{:?}", first.errors);
    let lookups: Vec<_> = first
        .tools
        .iter()
        .filter(|(n, _)| n == "qa_lookup")
        .collect();
    assert!(
        !lookups.is_empty(),
        "the model never used qa_lookup: {:?}",
        first.tools
    );
    assert!(lookups.iter().any(|(_, c)| succeeded(c)), "{lookups:?}");
    assert!(first.answer.contains("git branch -m"), "{}", first.answer);

    let second = run_live(settings, None, data.path(), prompt);
    // The repeated question must not go online again. It may be answered from
    // the Q&A library (a cached lookup) or from the knowledge note built from
    // the first lookup's sources.
    let online_again = second
        .tools
        .iter()
        .filter(|(n, _)| n == "qa_lookup")
        .any(|(_, c)| c.contains("\"cached\":false"));
    let cached = second
        .tools
        .iter()
        .any(|(n, c)| n == "qa_lookup" && c.contains("\"cached\":true"));
    let from_note = second
        .statuses
        .iter()
        .any(|s| s.starts_with("Using ") && s.contains("saved knowledge note"));
    assert!(
        !online_again && (cached || from_note),
        "the repeated question was looked up online again: {:?} {:?}",
        second.tools,
        second.statuses
    );
    assert!(
        second.approvals.iter().all(|a| !a.contains("Q&A")),
        "{:?}",
        second.approvals
    );
}

#[test]
#[ignore = "needs a local Ollama server, a web search provider and internet access"]
fn live_knowledge_notes_are_built_from_a_chat_and_recalled_in_the_next() {
    let data = tempfile::tempdir().unwrap();
    let mut settings = live_settings();
    settings.knowledge_notes = true;
    settings.automatic_memory = false;
    // Notes keep facts taken from real pages; a model's own answer is never
    // saved (see answers_from_the_model_alone_are_never_saved_as_notes).
    settings.network_tools = true;
    settings.retrieval_enabled = false;
    settings.num_predict = 1500;
    let first = run_live(
        settings.clone(),
        None,
        data.path(),
        "Search the web: how does the Rust borrow checker prevent data races, including ownership, shared and mutable references, and the Send and Sync traits?",
    );
    assert!(first.errors.is_empty(), "{:?}", first.errors);
    let store = ollama_workbench::notes::NoteStore::open(data.path()).unwrap();
    let notes = store.list().unwrap();
    eprintln!("notes: {notes:#?}");
    assert!(
        !notes.is_empty(),
        "no note was created; statuses: {:?}",
        first.statuses
    );
    for summary in &notes {
        let note = store.get(&summary.slug).unwrap().unwrap();
        eprintln!(
            "--- {} ---\n{}",
            note.title,
            std::fs::read_to_string(store.path(&summary.slug)).unwrap()
        );
        assert!(
            note.entries
                .iter()
                .all(|e| e.basis == ollama_workbench::notes::BASIS_WEB && !e.sources.is_empty()),
            "every saved fact must come from a cited web page"
        );
    }
    let second = run_live(
        settings,
        None,
        data.path(),
        "Remind me what Send and Sync mean for the Rust borrow checker and threads.",
    );
    assert!(second.errors.is_empty(), "{:?}", second.errors);
    assert!(
        second
            .statuses
            .iter()
            .any(|s| s.starts_with("Using ") && s.contains("saved knowledge note")),
        "the saved note was not recalled: {:?}",
        second.statuses
    );
}

#[test]
#[ignore = "needs a local Ollama server with several different models installed"]
fn live_several_models_separate_true_facts_from_false_ones() {
    use ollama_workbench::notes::{IncomingFact, NoteStore, ReviewStatus, BASIS_MODEL};
    let data = tempfile::tempdir().unwrap();
    let mut settings = live_settings();
    settings.notes_review_models = vec![
        "qwen3.5:9b".into(),
        "ministral-3:8b".into(),
        "gemma4:e4b".into(),
    ];
    settings.notes_review_count = 3;
    let store = NoteStore::open(data.path()).unwrap();
    let texts = [
        "Rust's borrow checker enforces ownership rules at compile time.",
        "The Linux kernel is written primarily in the Java programming language.",
        "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "The Rust compiler rejects every valid program when it is run on a Tuesday.",
    ];
    let facts: Vec<IncomingFact> = texts
        .iter()
        .map(|t| IncomingFact {
            text: (*t).into(),
            basis: BASIS_MODEL.into(),
            source_ix: vec![],
        })
        .collect();
    store
        .absorb("Programming And Science Facts", &facts, &[])
        .unwrap();
    let reports = ollama_workbench::notes::review_notes_blocking(
        &settings,
        data.path(),
        &["programming-and-science-facts".to_string()],
        None,
        None,
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    eprintln!("{}", reports[0].summary());
    let note = store.get("programming-and-science-facts").unwrap().unwrap();
    for entry in &note.entries {
        let label = entry
            .consensus
            .map(|c| c.label())
            .unwrap_or_else(|| "not reviewed".into());
        eprintln!("{label:50} | {}", entry.text);
    }
    assert_eq!(reports[0].reviewed, 4, "{:?}", reports[0]);
    let status = |i: usize| note.entries[i].consensus.unwrap().status();
    assert_ne!(
        status(0),
        ReviewStatus::Disputed,
        "a true fact was disputed"
    );
    assert_ne!(status(1), ReviewStatus::Agreed, "a false fact was agreed");
    assert_ne!(status(3), ReviewStatus::Agreed, "a false fact was agreed");
}

#[test]
#[ignore = "needs a local Ollama server and a tool-capable model"]
fn live_the_repo_map_lets_a_model_find_code_without_exploring() {
    let data = tempfile::tempdir().unwrap();
    let project = tempfile::tempdir().unwrap();
    let write = |path: &str, text: &str| {
        let full = project.path().join(path);
        std::fs::create_dir_all(full.parent().unwrap()).unwrap();
        std::fs::write(full, text).unwrap();
    };
    for n in 0..12 {
        write(
            &format!("src/module_{n}.rs"),
            &format!("pub fn handler_{n}(input: &str) -> usize {{ input.len() + {n} }}\n"),
        );
    }
    write("src/billing/invoice.rs", "pub struct Invoice { pub lines: Vec<u32> }\n\npub fn compute_invoice_total(invoice: &Invoice) -> u32 {\n    invoice.lines.iter().sum()\n}\n");
    write("src/app.rs", "use crate::billing::invoice::{compute_invoice_total, Invoice};\nfn main() { let _ = compute_invoice_total(&Invoice { lines: vec![] }); }\n");
    let mut settings = live_settings();
    settings.knowledge_notes = false;
    settings.automatic_memory = false;
    settings.max_tool_rounds = 4;
    let outcome = run_live(
        settings,
        Some(project.path()),
        data.path(),
        "Which file defines compute_invoice_total? Give the path and the line number.",
    );
    assert!(outcome.errors.is_empty(), "{:?}", outcome.errors);
    assert!(
        outcome
            .statuses
            .iter()
            .any(|s| s.contains("ranked map of the project's code")),
        "{:?}",
        outcome.statuses
    );
    assert!(
        outcome.answer.contains("src/billing/invoice.rs"),
        "wrong or missing path: {}",
        outcome.answer
    );
    eprintln!(
        "tool calls made: {:?}",
        outcome
            .tools
            .iter()
            .map(|t| t.0.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "needs a local Ollama server, a tool-capable model and a search provider"]
fn live_a_code_request_searches_for_open_source_examples_first() {
    let data = tempfile::tempdir().unwrap();
    let mut settings = live_settings();
    settings.network_tools = true;
    settings.code_example_search = true;
    let outcome = run_live(
        settings,
        None,
        data.path(),
        "Write a small python script that prints the first 20 Fibonacci numbers.",
    );
    assert!(outcome.errors.is_empty(), "{:?}", outcome.errors);
    let searched = outcome.tools.iter().any(|(name, _)| name == "web_search");
    assert!(
        searched,
        "no web_search before writing code: {:?}",
        outcome.tools
    );
    assert!(
        outcome.answer.contains("def ") || outcome.answer.contains("print"),
        "{}",
        outcome.answer
    );
}

#[test]
#[ignore = "needs a local Ollama server; checks isolated memory extraction"]
fn live_isolated_memory_capture_preserves_a_user_preference() {
    let dir = tempfile::tempdir().unwrap();
    let mut settings = live_settings();
    settings.automatic_memory = true;
    let client = ollama_workbench::ollama::OllamaClient::new(&settings.base_url).unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let started = Instant::now();
    let count = rt.block_on(async {
        tokio::time::timeout(Duration::from_secs(45), ollama_workbench::memory::capture_automatic(
            dir.path(), &client, &settings, settings.active_model(), None,
            "For all my Rust projects, I prefer four spaces for indentation and explicit error handling instead of unwrap.",
            Arc::new(AtomicBool::new(false)),
        )).await.expect("isolated memory capture exceeded the production 45-second deadline").unwrap()
    });
    eprintln!("memory capture: {count} records in {:?}", started.elapsed());
    assert!(count > 0);
    let store = ollama_workbench::memory::MemoryStore::open(dir.path()).unwrap();
    assert!(!store.list(None, "", 20).unwrap().is_empty());
}
