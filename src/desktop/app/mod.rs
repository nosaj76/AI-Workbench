mod chat;
mod code_export;
mod conversations;
mod diagnostics;
mod dialogs;
mod events;
mod inspector;
mod knowledge;
mod lifecycle;
mod models;
mod preferences;
mod shell;
mod tool_console;
mod tools;
mod workspace;

use crate::{
    files::{FileEntry, Workspace},
    ollama::{ChatMessage, ModelInfo, OllamaClient},
    runtime::{self, ChatEvent, ChatJob},
    settings::Settings,
    storage::{self, Entry, Session, SessionSummary, Store},
    tools::{PendingAction, PreparedTool, ToolEngine},
};
use anyhow::{bail, Context, Result};
use base64::Engine as _;
use eframe::egui::{self, RichText};
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc,
    },
    time::{Duration, Instant},
};

#[derive(Clone, Copy, PartialEq)]
enum Page {
    Chat,
    Files,
    Models,
    Setup,
    Knowledge,
    Tools,
    Tasks,
    Advanced,
    Settings,
    Diagnostics,
}

#[derive(Clone)]
struct Attachment {
    name: String,
    source: String,
    text: String,
    image: Option<String>,
}

enum WorkEvent {
    Models {
        request_id: u64,
        endpoint: String,
        models: Vec<ModelInfo>,
        version: String,
        running: Value,
    },
    ConnectionError {
        request_id: u64,
        endpoint: String,
        message: String,
    },
    PreviewError {
        request_id: u64,
        root: String,
        message: String,
    },
    ModelInfo(Value),
    Skill(crate::automation::LocalSkill),
    Knowledge {
        root: String,
        hits: Vec<crate::retrieval::Evidence>,
    },
    FileSearch {
        root: String,
        text: String,
    },
    PrepareTool {
        name: String,
        args: Value,
    },
    Prepared {
        engine: ToolEngine,
        prepared: PreparedTool,
    },
    PickedWorkspace(PathBuf),
    PickedFiles(Vec<PathBuf>),
    Tree {
        root: String,
        entries: Vec<FileEntry>,
    },
    Preview {
        request_id: u64,
        root: String,
        path: String,
        text: String,
        note: String,
    },
    Attached(Vec<Attachment>),
    Text(String),
    Status(String),
    Import(Session),
    Error(String),
    Done,
}

#[derive(Clone)]
enum ConversationAction {
    Open(String),
    TogglePin(String),
    ToggleArchive(String),
    Trash(String),
    Restore(String),
    DeleteForever(String),
}

enum ApprovalOrigin {
    Chat(Sender<bool>),
    Manual(ToolEngine),
}
struct Approval {
    action: PendingAction,
    origin: ApprovalOrigin,
}

pub struct Workbench {
    advanced: crate::advanced_ui::AdvancedUi,
    setup: crate::setup_ui::SetupUi,
    dir: PathBuf,
    settings: Settings,
    settings_corrupt: bool,
    settings_draft: Option<Settings>,
    settings_draft_original: Option<Settings>,
    show_sidebar: bool,
    show_inspector: bool,
    tool_console: tool_console::ToolConsole,
    selected_entry_id: Option<String>,
    visible_message_limit: usize,
    code_export_preview: Option<code_export::CodeExportSelection>,
    connection_request_id: u64,
    preview_request_id: u64,
    diagnostics: Vec<crate::diagnostics::Check>,
    store: Store,
    session: Session,
    summaries: Vec<SessionSummary>,
    session_search: String,
    include_archived: bool,
    show_trash: bool,
    page: Page,
    input: String,
    /// Prompts typed while a reply was being written, sent in order afterwards.
    queued_prompts: Vec<String>,
    attachments: Vec<Attachment>,
    models: Vec<ModelInfo>,
    version: String,
    running_models: Value,
    status: String,
    error: Option<String>,
    busy: bool,
    work_busy: bool,
    active_work: usize,
    cancel: Arc<AtomicBool>,
    work_cancel: Arc<AtomicBool>,
    work_cancels: Vec<Arc<AtomicBool>>,
    chat_tx: Sender<ChatEvent>,
    chat_rx: Receiver<ChatEvent>,
    work_tx: Sender<WorkEvent>,
    work_rx: Receiver<WorkEvent>,
    streaming: Option<usize>,
    approval: Option<Approval>,
    manual_engine: Option<ToolEngine>,
    queued_tool: Option<(String, Value)>,
    markdown: CommonMarkCache,
    workspace_input: String,
    tree: Vec<FileEntry>,
    file_filter: String,
    file_query: String,
    file_results: String,
    preview_path: String,
    preview: String,
    preview_original: String,
    preview_note: String,
    new_path: String,
    model_pull: String,
    model_new: String,
    model_from: String,
    model_system: String,
    model_details: String,
    model_delete: Option<String>,
    knowledge_query: String,
    knowledge_results: String,
    knowledge_hits: Vec<crate::retrieval::Evidence>,
    /// Saved versions of changed files (None until first listed).
    checkpoints: Option<crate::tools::CheckpointList>,
    tool_name: String,
    tool_args: String,
    tool_result: String,
    external_agent_id: String,
    external_task: String,
    external_files: String,
    external_context: String,
    command_program: String,
    command_args: String,
    profiles_json: String,
    prompts_json: String,
    audit: Vec<String>,
    compare_models: String,
    compare_active: bool,
    compare_queue: Vec<String>,
    compare_messages: Vec<ChatMessage>,
    last_save: Instant,
    dirty: bool,
    pending_trash_id: Option<String>,
    pending_permanent_delete_id: Option<String>,
    demo: bool,
    skill_preview: Option<crate::automation::LocalSkill>,
}

impl Workbench {
    pub fn new(cc: &eframe::CreationContext<'_>, dir: PathBuf, demo: bool) -> Result<Self> {
        Self::from_context(&cc.egui_ctx, dir, demo)
    }

    /// Show a message from before the window existed (such as a crash notice)
    /// without hiding an error the app found while starting.
    pub fn show_startup_notice(&mut self, notice: String) {
        self.error = Some(match self.error.take() {
            Some(existing) => format!("{existing}\n\n{notice}"),
            None => notice,
        });
    }

    fn from_context(ctx: &egui::Context, dir: PathBuf, demo: bool) -> Result<Self> {
        let (settings, settings_corrupt, error) = match Settings::load(&dir) {
            Ok(s) => (s, false, None),
            Err(e) => (Settings::default(), true, Some(format!("{e:#}"))),
        };
        let store = Store::open(&dir)?;
        let summaries = store.list("", false)?;
        let session = settings
            .last_session
            .as_deref()
            .and_then(|id| store.load(id).ok())
            .or_else(|| summaries.first().and_then(|s| store.load(&s.id).ok()))
            .unwrap_or_default();
        let (chat_tx, chat_rx) = mpsc::channel();
        let (work_tx, work_rx) = mpsc::channel();
        let profiles_json = serde_json::to_string_pretty(&settings.profiles)?;
        let prompts_json = serde_json::to_string_pretty(&settings.prompts)?;
        let workspace_input = session.project.clone().unwrap_or_default();
        let first_page = if !demo && !settings.model_setup_seen {
            Page::Setup
        } else {
            Page::Chat
        };
        let mut app = Self {
            advanced: crate::advanced_ui::AdvancedUi::new(dir.clone()),
            setup: crate::setup_ui::SetupUi::new(dir.clone()),
            dir,
            settings,
            settings_corrupt,
            settings_draft: None,
            settings_draft_original: None,
            show_sidebar: true,
            show_inspector: false,
            tool_console: tool_console::ToolConsole::default(),
            selected_entry_id: None,
            visible_message_limit: 80,
            code_export_preview: None,
            connection_request_id: 0,
            preview_request_id: 0,
            diagnostics: vec![],
            store,
            session,
            summaries,
            session_search: String::new(),
            include_archived: false,
            show_trash: false,
            page: first_page,
            input: String::new(),
            queued_prompts: vec![],
            attachments: vec![],
            models: vec![],
            version: String::new(),
            running_models: Value::Null,
            status: "Ready to connect to local Ollama".into(),
            error,
            busy: false,
            work_busy: false,
            active_work: 0,
            cancel: Arc::new(AtomicBool::new(false)),
            work_cancel: Arc::new(AtomicBool::new(false)),
            work_cancels: vec![],
            chat_tx,
            chat_rx,
            work_tx,
            work_rx,
            streaming: None,
            approval: None,
            manual_engine: None,
            queued_tool: None,
            markdown: CommonMarkCache::default(),
            workspace_input,
            tree: vec![],
            file_filter: String::new(),
            file_query: String::new(),
            file_results: String::new(),
            preview_path: String::new(),
            preview: String::new(),
            preview_original: String::new(),
            preview_note: String::new(),
            new_path: String::new(),
            model_pull: String::new(),
            model_new: String::new(),
            model_from: String::new(),
            model_system: String::new(),
            model_details: String::new(),
            model_delete: None,
            knowledge_query: String::new(),
            knowledge_results: String::new(),
            knowledge_hits: vec![],
            checkpoints: None,
            tool_name: "list_files".into(),
            tool_args: "{\"path\":\".\"}".into(),
            tool_result: String::new(),
            external_agent_id: "codex".into(),
            external_task: String::new(),
            external_files: String::new(),
            external_context: String::new(),
            command_program: "git".into(),
            command_args: "[\"status\",\"--short\"]".into(),
            profiles_json,
            prompts_json,
            audit: vec![],
            compare_models: String::new(),
            compare_active: false,
            compare_queue: vec![],
            compare_messages: vec![],
            last_save: Instant::now(),
            dirty: false,
            pending_trash_id: None,
            pending_permanent_delete_id: None,
            demo,
            skill_preview: None,
        };
        app.apply_theme(ctx);
        app.advanced.set_preview_mode(demo);
        if demo {
            app.seed_demo();
        } else if app.page != Page::Setup {
            app.refresh_models();
        }
        if app.session.project.is_some() && !demo {
            app.refresh_tree();
        }
        Ok(app)
    }

    fn seed_demo(&mut self) {
        self.status = "Preview mode · sample data · no inference requests".into();
        self.version = "preview".into();
        self.settings.model = "qwen2.5-coder:7b".into();
        self.session = Session {
            title: "Review the local workspace".into(),
            // Inside the isolated preview storage: never a real user path.
            project: Some(self.dir.join("example").to_string_lossy().into_owned()),
            ..Default::default()
        };
        self.workspace_input = self.session.project.clone().unwrap();
        self.session.entries.push(Entry::new(ChatMessage {
            role: "user".into(),
            content: "Review this Rust project and explain what needs attention.".into(),
            ..Default::default()
        }));
        for (tool, content) in [
            (
                "project_tree",
                r#"{"entries":["Cargo.toml","src/main.rs","src/lib.rs"],"note":"Sample tool result for the interface preview"}"#,
            ),
            (
                "run_command",
                r#"{"ok":false,"error":"Sample failure: the preview does not run commands"}"#,
            ),
        ] {
            self.session.entries.push(Entry::new(ChatMessage {
                role: "tool".into(),
                tool_name: Some(tool.into()),
                content: content.into(),
                ..Default::default()
            }));
        }
        let mut entry=Entry::new(ChatMessage {role:"assistant".into(),content:"## A review grounded in your files\n\nOpen a workspace to inspect source code and documents, then ask for a focused review. Every tool result appears in the conversation.\n\n- **Local inference** through your Ollama installation.\n- **Evidence** with file paths and line references.\n- **Changes you can review** before files or commands are executed.\n\n```rust\nfn main() {\n    println!(\"Your models. Your files. Your workspace.\");\n}\n```\n\nThis is a labeled interface preview; it is not a model-generated review.".into(),..Default::default()});
        entry.model = "qwen2.5-coder:7b".into();
        entry.sources = vec!["Example source references appear here".into()];
        self.session.entries.push(entry);
    }
}

impl eframe::App for Workbench {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.render(ctx);
    }
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.setup.cancel_all();
        self.advanced.cancel_all();
        self.cancel.store(true, Ordering::Relaxed);
        if let Some(index) = self.streaming.take() {
            let entry = &mut self.session.entries[index];
            if !entry.stats.is_object() {
                entry.stats = json!({});
            }
            entry.stats["incomplete"] = json!(true);
            entry.stats["interruption_kind"] = json!("app_closed");
            entry.stats["failure_reason"] =
                json!("Workbench was closed while this reply was still being written.");
            if entry.message.content.is_empty() {
                entry.message.content = "Request stopped before an answer was completed.".into();
            }
        }
        for cancel in &self.work_cancels {
            cancel.store(true, Ordering::Relaxed);
        }
        self.approve(false);
        self.persist();
    }
}

fn is_editable_text(path: &str) -> bool {
    let ext = Path::new(path)
        .extension()
        .unwrap_or_default()
        .to_string_lossy()
        .to_ascii_lowercase();
    !matches!(
        ext.as_str(),
        "pdf"
            | "docx"
            | "odt"
            | "pptx"
            | "xlsx"
            | "epub"
            | "png"
            | "jpg"
            | "jpeg"
            | "webp"
            | "gif"
            | "doc"
            | "xls"
            | "ppt"
    )
}

#[cfg(test)]
mod tests;
