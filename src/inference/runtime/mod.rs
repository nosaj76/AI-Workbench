mod tool_protocol;

/// Read-only identification of tool-shaped JSON in legacy assistant replies.
/// This helper never constructs or executes a tool call.
pub(crate) fn inspect_tool_text(content: &str) -> Option<Vec<String>> {
    tool_protocol::textual_request_names(content)
}

use crate::{
    ollama::{ChatMessage, ChatOptions, OllamaClient, StreamEvent},
    settings::Settings,
    storage::Entry,
    tools::{PendingAction, PreparedTool, ToolEngine},
};
use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Sender},
        Arc,
    },
    time::Duration,
};

pub enum ChatEvent {
    Begin(String),
    Content(String),
    Thinking(String),
    Stats(Value),
    Complete(ChatMessage),
    /// The answer being streamed was not accepted and will be retried; the
    /// reason replaces it, and it is never sent back to the model.
    Superseded(String),
    /// Raw assistant text resembling tool calls, not native Ollama tool_calls.
    /// The original text is retained for inspection, but is NEVER executed.
    UnexecutedToolText {
        message: ChatMessage,
        names: Vec<String>,
        retrying: bool,
    },
    Append(Entry),
    Status(String),
    /// Optional post-answer work failed; preserve it without marking the answer incomplete.
    BackgroundFailure {
        task: String,
        reason: String,
    },
    Approval {
        action: PendingAction,
        reply: Sender<bool>,
    },
    Audit {
        action: String,
        details: String,
    },
    Evidence {
        context: String,
        sources: Vec<String>,
    },
    /// A tool began running (live activity display only; the saved result
    /// still arrives through `Append`).
    ToolStarted {
        id: String,
        name: String,
        summary: String,
    },
    /// Live stdout/stderr from a running command or adapter.
    ToolOutput {
        id: String,
        chunk: String,
    },
    ToolFinished {
        id: String,
        ok: bool,
        preview: String,
    },
    Error(String),
    Finished,
}

pub struct ChatJob {
    pub settings: Settings,
    pub messages: Vec<ChatMessage>,
    pub workspace: Option<PathBuf>,
    pub model: String,
    pub engine: Option<ToolEngine>,
    pub retrieval_path: PathBuf,
}

mod configuration;
mod context;
mod delegation;
mod execution;
mod operations;
mod terminal_intent;
#[cfg(test)]
mod terminal_workflow_tests;
pub use configuration::{configured_engine, configured_network_engine, options};
pub use context::fit_context;
use delegation::delegate_run;
use execution::run_chat;
pub(crate) use terminal_intent::{
    modernbasic_installed, requests_modernbasic_for_turn, requests_terminal_action_for_turn,
};

pub fn spawn_chat(job: ChatJob, tx: Sender<ChatEvent>, cancel: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        let result = crate::jobs::guarded(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .context("Start inference worker")
                .and_then(|rt| rt.block_on(run_chat(job, &tx, cancel.clone())))
        });
        if let Err(e) = result {
            if !cancel.load(Ordering::Relaxed) {
                let _ = tx.send(ChatEvent::Error(format!("{e:#}")));
            }
        }
        if cancel.load(Ordering::Relaxed) {
            let _ = tx.send(ChatEvent::Status("Stopped".into()));
        }
        let _ = tx.send(ChatEvent::Finished);
    });
}
