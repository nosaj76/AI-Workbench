//! Self-review team: Workbench reviews its own recent conversations for
//! workflow problems and incomplete answers, optionally reads its own source to
//! explain the mechanisms involved, and recommends improvements.
//!
//! It is built on the autonomous agents framework with an application-fixed
//! plan: one read-only child per conversation (given a digest Workbench writes
//! from the saved transcript plus the rule-based diagnostic incidents), one
//! optional child that reads the Workbench source tree, and the framework's
//! normal synthesis pass. Nothing is edited; the output is a report.

use crate::{
    agents::{AgentRequest, MAX_AGENTS, MAX_PLANNED_TASK_CHARS},
    conversation_diagnostics::{self, Incident},
    settings::Settings,
    storage::{Session, Store},
};
use anyhow::{bail, Result};
use serde_json::Value;
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

/// Conversations one self-review may cover: one child each, plus the code child.
pub const MAX_CONVERSATIONS: usize = MAX_AGENTS - 1;
/// Room in a conversation task for the transcript digest itself.
const DIGEST_CHARS: usize = 14_000;
const INCIDENT_CHARS: usize = 4_000;

pub const OBJECTIVE: &str = "Self-review of AI Workbench. Find workflow problems and incomplete or unsatisfying answers in its most recent conversations, explain why they happened (model behaviour, tool design, prompt guidance, runtime limits and recovery, settings), and recommend specific, prioritised improvements to Workbench itself. Where the source code was read, ground each recommendation in the files and lines involved.";

const CONVERSATION_TASK: &str = "Review one saved Workbench conversation for workflow problems. Workbench wrote the digest below from the saved transcript: [n] is the entry number, U = user, A = assistant (with model and stop reason), T = tool result. Rule-based diagnostic incidents follow it. Use only this evidence; do not invent events.

Report, citing entry numbers:
1. Each user request and whether it was delivered: complete, partial or not delivered, and why.
2. Workflow problems: failed, repeated or wasted tool calls; loops and dead ends; context or output exhaustion; the wrong tool, language or format; needless clarifying questions; giving up or refusing when tools could have helped; claims not backed by tool results; ignoring files or examples already in the workspace.
3. The most likely cause of each problem: model behaviour, tool design, prompt guidance, runtime limit or recovery, or a setting.
4. Concrete improvements to Workbench (not advice for the user): guidance text, tool behaviour or error messages, runtime recovery, limits, defaults. Say what a good assistant turn would have done instead.";

const CODE_TASK: &str = "The workspace is the AI Workbench source tree. Using read-only tools (list, search, read), find how Workbench produces the behaviour behind the problems below, explain the mechanism briefly with file:line citations, and propose specific code or prompt changes that would prevent each one. Start with docs/ARCHITECTURE.md and the likely source files listed, then follow the code. Prefer a few well-evidenced recommendations over a long speculative list; say which problems you could not trace.";

/// The Workbench source tree, when this binary was built from one that still
/// exists here. It lets the code child read how Workbench works.
pub fn default_source_root() -> Option<PathBuf> {
    let root = PathBuf::from(option_env!("CARGO_MANIFEST_DIR")?);
    (root.join("Cargo.toml").is_file() && root.join("src/inference/runtime").is_dir())
        .then_some(root)
}

/// The most recently updated conversations that contain at least one answer,
/// newest first. Trashed conversations are excluded; archived ones are kept.
pub fn recent_sessions(store: &Store, count: usize) -> Result<Vec<Session>> {
    store.recent_answered(count)
}

fn clip(text: &str, max: usize) -> String {
    let text = text.trim();
    if text.chars().count() <= max {
        return text.to_string();
    }
    let head = max * 2 / 3;
    let tail = max.saturating_sub(head + 5);
    let chars: Vec<char> = text.chars().collect();
    format!(
        "{} […] {}",
        chars[..head].iter().collect::<String>(),
        chars[chars.len() - tail..].iter().collect::<String>()
    )
}

fn one_line(text: &str, max: usize) -> String {
    clip(&text.split_whitespace().collect::<Vec<_>>().join(" "), max)
}

/// One line saying what a tool result amounted to, never its full payload.
fn tool_outcome(content: &str) -> String {
    let Ok(result) = serde_json::from_str::<Value>(content) else {
        return format!("unparsed: {}", one_line(content, 200));
    };
    let text = |key: &str| result.get(key).and_then(Value::as_str);
    if result.get("duplicate").and_then(Value::as_bool) == Some(true) {
        return "SKIPPED duplicate request (not executed)".into();
    }
    if result.get("denied").and_then(Value::as_bool) == Some(true) {
        return format!("DENIED: {}", one_line(text("reason").unwrap_or(""), 200));
    }
    if let Some(error) = text("error").filter(|e| !e.is_empty()) {
        return format!("FAILED: {}", one_line(error, 240));
    }
    let state = result
        .pointer("/operation/state")
        .and_then(Value::as_str)
        .unwrap_or("ok");
    let mut parts = vec![state.to_string()];
    for key in ["path", "url", "query", "result_count", "exit_code", "total_lines"] {
        if let Some(value) = result.get(key).filter(|v| !v.is_null()) {
            let value = value.as_str().map_or_else(|| value.to_string(), str::to_string);
            parts.push(format!("{key}={}", one_line(&value, 120)));
        }
    }
    for flag in ["truncated", "context_limited", "has_more", "output_closed"] {
        if result.get(flag).and_then(Value::as_bool) == Some(true) {
            parts.push(flag.into());
        }
    }
    if result.pointer("/summary/alive").and_then(Value::as_bool) == Some(false) {
        parts.push("terminal exited".into());
    }
    for key in ["warning", "note", "coverage"] {
        if let Some(note) = text(key).filter(|n| n.contains("NOT ") || key == "warning") {
            parts.push(format!("{key}: {}", one_line(note, 200)));
        }
    }
    parts.push(format!("{} chars", content.chars().count()));
    parts.join(", ")
}

/// A compact, faithful outline of a conversation for a reviewer model. When it
/// does not fit, the newest entries are kept and the omission is stated.
pub fn digest(session: &Session, max_chars: usize) -> String {
    let mut lines = Vec::new();
    for (index, entry) in session.entries.iter().enumerate() {
        let message = &entry.message;
        let line = match message.role.as_str() {
            "user" => format!("[{index}] U: {}", clip(&message.content, 700)),
            "tool" => format!(
                "[{index}] T {}: {}",
                message.tool_name.as_deref().unwrap_or("tool"),
                tool_outcome(&message.content)
            ),
            "assistant" => {
                let mut meta = vec![];
                if !entry.model.is_empty() {
                    meta.push(entry.model.clone());
                }
                let stat = |key: &str| entry.stats.get(key).filter(|v| !v.is_null());
                if let Some(reason) = stat("done_reason").and_then(Value::as_str) {
                    meta.push(format!("stop={reason}"));
                }
                if let Some(tokens) = stat("prompt_eval_count").and_then(Value::as_u64) {
                    meta.push(format!("prompt={tokens}tok"));
                }
                if let Some(kind) = stat("interruption_kind").and_then(Value::as_str) {
                    meta.push(format!("interrupted={kind}"));
                }
                let mut text = format!("[{index}] A ({})", meta.join(", "));
                for call in &message.tool_calls {
                    text.push_str(&format!(
                        " CALL {}({})",
                        call.function.name,
                        one_line(&call.function.arguments.to_string(), 220)
                    ));
                }
                if !message.content.trim().is_empty() {
                    text.push_str(&format!(": {}", clip(&message.content, 900)));
                }
                if let Some(reason) = stat("failure_reason").and_then(Value::as_str) {
                    text.push_str(&format!(" [failure: {}]", one_line(reason, 240)));
                }
                text
            }
            other => format!("[{index}] {other}: {}", one_line(&message.content, 300)),
        };
        lines.push(line);
    }
    let header = format!(
        "Conversation \"{}\" · workspace {} · {} entries · updated {}",
        one_line(&session.title, 120),
        session.project.as_deref().unwrap_or("none"),
        session.entries.len(),
        session.updated_at
    );
    let mut kept = Vec::new();
    let mut used = header.chars().count();
    for line in lines.iter().rev() {
        let cost = line.chars().count() + 1;
        if used + cost > max_chars {
            break;
        }
        used += cost;
        kept.push(line.as_str());
    }
    kept.reverse();
    let omitted = lines.len() - kept.len();
    let mut text = header;
    if omitted > 0 {
        text.push_str(&format!(
            "\n(The first {omitted} entries are omitted for length; the newest are shown.)"
        ));
    }
    for line in kept {
        text.push('\n');
        text.push_str(line);
    }
    text
}

fn incident_lines(incidents: &[Incident], max_chars: usize) -> String {
    let mut text = String::new();
    for incident in incidents {
        let line = format!(
            "- {} ({}, {}): {}\n",
            incident.code,
            incident.severity,
            incident.confidence,
            one_line(&incident.summary, 260)
        );
        if text.chars().count() + line.chars().count() > max_chars {
            text.push_str("- … further incidents omitted for length\n");
            break;
        }
        text.push_str(&line);
    }
    if text.is_empty() {
        text.push_str("- None detected by the rules (problems may still be semantic).\n");
    }
    text
}

/// Builds the self-review team for these conversations. With `source_root`,
/// that tree becomes the read-only workspace and a code child is added.
pub fn request(
    settings: &Settings,
    data_dir: &Path,
    sessions: &[Session],
    source_root: Option<PathBuf>,
    planner_model: String,
) -> Result<AgentRequest> {
    if sessions.is_empty() {
        bail!("There are no saved conversations with an answer to review");
    }
    let sessions = &sessions[..sessions.len().min(MAX_CONVERSATIONS)];
    let mut tasks = Vec::new();
    let mut all_incidents = Vec::new();
    for session in sessions {
        let incidents = conversation_diagnostics::detect(session, false);
        let prompt = format!(
            "{CONVERSATION_TASK}\n\nDIGEST:\n{}\n\nDIAGNOSTIC INCIDENTS:\n{}",
            digest(session, DIGEST_CHARS),
            incident_lines(&incidents, INCIDENT_CHARS)
        );
        let title = format!("Review conversation: {}", one_line(&session.title, 100));
        tasks.push((title, clip(&prompt, MAX_PLANNED_TASK_CHARS)));
        all_incidents.extend(incidents);
    }
    if source_root.is_some() {
        let mut hints = BTreeSet::new();
        let mut codes = BTreeSet::new();
        let mut problems = String::new();
        for incident in &all_incidents {
            hints.extend(incident.source_hints.iter().cloned());
            // One example per rule keeps the list readable.
            if codes.insert(incident.code.clone()) {
                problems.push_str(&format!(
                    "- {} in \"{}\": {}\n",
                    incident.code,
                    one_line(&incident.title, 80),
                    one_line(&incident.summary, 260)
                ));
            }
        }
        if problems.is_empty() {
            problems.push_str("- No rule-based incidents; review how the turn loop handles tool failures, context limits and incomplete answers (src/inference/runtime/execution.rs).\n");
        }
        let hints = if hints.is_empty() {
            "src/inference/runtime/execution.rs, src/inference/runtime/context.rs, src/inference/runtime/turn_helpers.rs".to_string()
        } else {
            hints.into_iter().take(20).collect::<Vec<_>>().join(", ")
        };
        tasks.push((
            "Trace the causes in Workbench's code".into(),
            clip(
                &format!("{CODE_TASK}\n\nPROBLEMS SEEN:\n{problems}\nLIKELY SOURCE FILES: {hints}"),
                MAX_PLANNED_TASK_CHARS,
            ),
        ));
    }
    let mut settings = settings.clone();
    // Reviewers work only from the digest and the source tree. Saved knowledge
    // notes and the home-server health lookup match on keywords, and a long
    // transcript digest matches nearly anything ("treat", "pain"), so they only
    // add unrelated context. The children are read-only and offline anyway.
    settings.knowledge_notes = false;
    settings.network_tools = false;
    let mut request = AgentRequest::new(
        settings,
        source_root,
        data_dir.to_path_buf(),
        OBJECTIVE.into(),
        planner_model,
    );
    request.max_agents = tasks.len().clamp(1, MAX_AGENTS);
    request.planned_tasks = tasks;
    // A review only reads; it never asks for approvals.
    request.allow_actions = false;
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ollama::{ChatMessage, FunctionCall, ToolCall},
        storage::Entry,
    };

    fn entry(role: &str, content: &str) -> Entry {
        Entry::new(ChatMessage {
            role: role.into(),
            content: content.into(),
            ..Default::default()
        })
    }

    fn session() -> Session {
        let mut call = entry("assistant", "");
        call.message.tool_calls = vec![ToolCall {
            function: FunctionCall {
                name: "terminal_write".into(),
                arguments: serde_json::json!({"text":"modernbasic repl"}),
            },
        }];
        let mut tool = entry(
            "tool",
            r#"{"error":"This terminal shell has exited","operation":{"state":"failed"}}"#,
        );
        tool.message.tool_name = Some("terminal_write".into());
        let mut answer = entry("assistant", "I cannot complete your request.");
        answer.model = "small:4b".into();
        answer.stats = serde_json::json!({"done_reason":"length","prompt_eval_count":32766});
        Session {
            title: "Song".into(),
            entries: vec![entry("user", "write the song"), call, tool, answer],
            ..Default::default()
        }
    }

    #[test]
    fn digest_keeps_calls_failures_and_stop_reasons() {
        let text = digest(&session(), 10_000);
        assert!(text.contains("[0] U: write the song"), "{text}");
        assert!(text.contains("CALL terminal_write"), "{text}");
        assert!(text.contains("[2] T terminal_write: FAILED: This terminal shell has exited"), "{text}");
        assert!(text.contains("stop=length"), "{text}");
        assert!(text.contains("prompt=32766tok"), "{text}");
    }

    #[test]
    fn long_digests_keep_the_newest_entries_and_say_so() {
        let mut long = session();
        for n in 0..200 {
            long.entries.push(entry("user", &format!("request number {n} {}", "x".repeat(200))));
        }
        let text = digest(&long, 3_000);
        assert!(text.chars().count() <= 3_200);
        assert!(text.contains("request number 199"));
        assert!(!text.contains("[0] U: write the song"));
        assert!(text.contains("omitted for length"));
    }

    #[test]
    fn request_has_one_child_per_conversation_plus_an_optional_code_child() {
        let dir = tempfile::tempdir().unwrap();
        let settings = Settings::default();
        let sessions = vec![session(), session()];
        let plain = request(&settings, dir.path(), &sessions, None, "m".into()).unwrap();
        assert_eq!(plain.planned_tasks.len(), 2);
        assert!(plain.workspace.is_none());
        assert!(!plain.allow_actions);
        assert!(!plain.settings.knowledge_notes);
        assert!(!plain.settings.network_tools);
        assert!(plain.planned_tasks[0].1.contains("FAILED: This terminal shell has exited"));
        plain.validate().unwrap();

        let source = tempfile::tempdir().unwrap();
        let with_code = request(
            &settings,
            dir.path(),
            &sessions,
            Some(source.path().to_path_buf()),
            "m".into(),
        )
        .unwrap();
        assert_eq!(with_code.planned_tasks.len(), 3);
        assert!(with_code.planned_tasks[2].0.contains("code"));
        with_code.validate().unwrap();
    }

    #[test]
    fn no_conversations_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(request(&Settings::default(), dir.path(), &[], None, "m".into()).is_err());
    }
}
