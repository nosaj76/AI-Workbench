//! Local, evidence-based conversation incident detection. Never executes source
//! content or sends conversation data to a model/network. Heuristics are labelled.
use crate::storage::{Entry, Session};
use anyhow::{Context, Result};
use rusqlite::{params, Connection, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::OnceLock,
};

pub const RETAINED_INCIDENTS: usize = 1000;
const RULE_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Incident {
    pub schema_version: u32,
    pub rule_version: u32,
    pub id: String,
    pub code: String,
    pub severity: String,
    pub confidence: String,
    pub summary: String,
    pub session_id: String,
    pub title: String,
    pub turn_entry_id: String,
    pub anchor_entry_id: String,
    pub model: String,
    pub workspace: Option<String>,
    pub user_request: String,
    pub evidence: Vec<Value>,
    pub investigation: String,
    pub source_hints: Vec<String>,
    pub detector_build: String,
    pub configuration_at_detection: Value,
}

pub fn log_path(dir: &Path) -> PathBuf {
    dir.join("diagnostics/conversation-failures.jsonl")
}

pub fn initialize(conn: &Connection) -> Result<()> {
    conn.execute_batch("CREATE TABLE IF NOT EXISTS conversation_incidents (
        id TEXT PRIMARY KEY, session_id TEXT NOT NULL, code TEXT NOT NULL,
        first_seen TEXT NOT NULL, last_changed TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_conversation_incidents_session ON conversation_incidents(session_id);
        CREATE TABLE IF NOT EXISTS conversation_diagnostic_state (id INTEGER PRIMARY KEY CHECK(id=1), dirty INTEGER NOT NULL);
        INSERT OR IGNORE INTO conversation_diagnostic_state VALUES(1,0);")?;
    Ok(())
}

fn clipped(text: &str, limit: usize) -> String {
    let mut s: String = text.chars().take(limit).collect();
    if text.chars().count() > limit {
        s.push_str("… [truncated; full evidence remains in saved conversation]");
    }
    s
}

/// Best-effort redaction, not a guarantee arbitrary prose contains no secrets.
/// Omit entire credential-bearing lines rather than guessing where a value ends.
fn scrub(text: &str, limit: usize) -> String {
    if text.contains("-----BEGIN") && text.contains("PRIVATE KEY-----") {
        return "[private key material omitted]".into();
    }
    static SECRET: OnceLock<regex::Regex> = OnceLock::new();
    let secret = SECRET.get_or_init(|| regex::Regex::new(
        r#"(?i)(bearer\s+\S+|(?:api[ _-]?key|access[_-]?token|password|passwd|secret|authorization|credential|token)[\s\"']*[:=]|--(?:password|token|api-key|secret)\s|-----BEGIN .*PRIVATE KEY|https?://[^\s/]+:[^\s/]+@)"#
    ).expect("constant redaction regex"));
    let bounded = clipped(text, limit * 4);
    clipped(
        &bounded
            .lines()
            .map(|line| {
                if secret.is_match(line) {
                    "[credential-bearing line omitted]"
                } else {
                    line
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        limit,
    )
}

fn safe_value(value: &Value, depth: usize) -> Value {
    if depth > 5 {
        return json!("[nested value omitted]");
    }
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .take(24)
                .map(|(k, v)| {
                    let key = k.to_ascii_lowercase();
                    let shown = if [
                        "password",
                        "secret",
                        "token",
                        "api_key",
                        "apikey",
                        "authorization",
                        "credential",
                        "cookie",
                    ]
                    .iter()
                    .any(|s| key.contains(s))
                    {
                        json!("[redacted]")
                    } else if [
                        "content",
                        "stdin_text",
                        "replacement",
                        "before",
                        "after",
                        "thinking",
                        "context",
                    ]
                    .contains(&key.as_str())
                    {
                        json!("[body omitted; see saved entry]")
                    } else {
                        safe_value(v, depth + 1)
                    };
                    (clipped(k, 80), shown)
                })
                .collect(),
        ),
        Value::Array(a) => Value::Array(
            a.iter()
                .take(16)
                .enumerate()
                .map(|(i, v)| {
                    let previous = i
                        .checked_sub(1)
                        .and_then(|i| a[i].as_str())
                        .unwrap_or("")
                        .to_ascii_lowercase();
                    if [
                        "--password",
                        "--token",
                        "--api-key",
                        "--secret",
                        "--authorization",
                    ]
                    .contains(&previous.as_str())
                    {
                        json!("[redacted]")
                    } else {
                        safe_value(v, depth + 1)
                    }
                })
                .collect(),
        ),
        Value::String(s) => json!(scrub(s, 400)),
        v => v.clone(),
    }
}

fn evidence(entry: &Entry, index: usize) -> Value {
    let message = &entry.message;
    let calls: Vec<_> = message
        .tool_calls
        .iter()
        .take(3)
        .map(|call| {
            json!({
                "tool":call.function.name,"arguments":safe_value(&call.function.arguments,0)
            })
        })
        .collect();
    let output = if message.role == "tool" {
        match serde_json::from_str::<Value>(&message.content) {
            Ok(v) => json!({"error":v.get("error").map(|v|safe_value(v,0)),
                "operation":v.get("operation").map(|v|safe_value(v,0)),"ok":v.get("ok").map(|v|safe_value(v,0)),
                "result":v.get("result").map(|v|safe_value(v,0)),
                "output":v.get("output").and_then(Value::as_str).map(|s|scrub(s,900)),
                "path":v.get("path").map(|v|safe_value(v,0)),"absolute_path":v.get("absolute_path").map(|v|safe_value(v,0)),
                "has_more":v.get("has_more").map(|v|safe_value(v,0)),"session_id":v.get("session_id").map(|v|safe_value(v,0))}),
            Err(_) => json!(scrub(&message.content, 700)),
        }
    } else {
        json!(scrub(&message.content, 700))
    };
    let result = json!({"entry_id":entry.id,"entry_index":index,"role":message.role,
        "tool_name":message.tool_name,"model":scrub(&entry.model,200),"content":output,"tool_calls":calls,
        "interruption_kind":entry.stats.get("interruption_kind").map(|v|safe_value(v,0)),
        "background_failures":entry.stats.get("background_failures").map(|v|safe_value(v,0)),
        "failure_reason":entry.stats.get("failure_reason").map(|v|safe_value(v,0))});
    let encoded = result.to_string();
    if encoded.chars().count() > 2400 {
        json!({"entry_id":entry.id,"entry_index":index,"role":message.role,
            "preview":clipped(&encoded,2000),"truncated":true})
    } else {
        result
    }
}

fn add(
    out: &mut Vec<Incident>,
    session: &Session,
    turn: usize,
    at: usize,
    rule: (&str, &str, &str, &str, &str),
    related: &[usize],
) {
    let (code, severity, confidence, summary, investigation) = rule;
    let anchor = &session.entries[at];
    let id = format!(
        "{:x}",
        Sha256::digest(format!(
            "{}:{}:{}:{}",
            RULE_VERSION, session.id, anchor.id, code
        ))
    );
    let mut indices = related.to_vec();
    indices.push(at);
    indices.sort_unstable();
    indices.dedup();
    let model = session.entries[turn..=at]
        .iter()
        .rev()
        .find(|e| !e.model.is_empty())
        .map(|e| e.model.clone())
        .or_else(|| session.selected_model.clone())
        .unwrap_or_default();
    out.push(Incident {
        schema_version: 1,
        rule_version: RULE_VERSION,
        id,
        code: code.into(),
        severity: severity.into(),
        confidence: confidence.into(),
        summary: summary.into(),
        session_id: session.id.clone(),
        title: scrub(&session.title, 200),
        turn_entry_id: session.entries[turn].id.clone(),
        anchor_entry_id: anchor.id.clone(),
        model,
        workspace: session.project.as_ref().map(|p| scrub(p, 500)),
        user_request: scrub(&session.entries[turn].message.content, 1000),
        evidence: indices
            .into_iter()
            .rev()
            .take(5)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|i| evidence(&session.entries[i], i))
            .collect(),
        investigation: investigation.into(),
        source_hints: source_hints(code),
        detector_build: format!(
            "{} / {}",
            env!("CARGO_PKG_VERSION"),
            env!("WORKBENCH_SOURCE_ID")
        ),
        configuration_at_detection: Value::Null,
    });
}

fn source_hints(code: &str) -> Vec<String> {
    let paths: &[&str] = match code {
        "background_task_failed" => &[
            "src/orchestration/memory.rs",
            "src/inference/runtime/execution.rs",
            "src/desktop/app/events.rs",
        ],
        "workspace_mismatch" => &[
            "src/desktop/app/workspace.rs",
            "src/workspace/tools/preparation.rs",
        ],
        "invalid_shell_invocation" => &[
            "src/services/integrations.rs",
            "src/workspace/tools/terminal.rs",
        ],
        "file_verification_failed" => &[
            "src/inference/runtime/operations.rs",
            "src/workspace/tools/execution.rs",
        ],
        "unexecuted_tool_text" => &[
            "src/inference/runtime/tool_protocol.rs",
            "src/inference/runtime/execution.rs",
        ],
        "possible_capability_denial" => &[
            "src/inference/runtime/terminal_intent.rs",
            "src/desktop/app/chat.rs",
        ],
        "input_before_output_drained" => &[
            "src/workspace/tools/terminal.rs",
            "src/inference/runtime/execution.rs",
        ],
        _ => &[
            "src/inference/runtime/execution.rs",
            "src/desktop/app/events.rs",
        ],
    };
    paths.iter().map(|p| (*p).into()).collect()
}

fn failure(v: &Value) -> bool {
    if v.get("denied").and_then(Value::as_bool) == Some(true)
        || v.get("cancelled").and_then(Value::as_bool) == Some(true)
        || v.pointer("/result/cancelled").and_then(Value::as_bool) == Some(true)
        || matches!(
            v.pointer("/operation/state").and_then(Value::as_str),
            Some("denied" | "cancelled" | "refused" | "skipped_duplicate")
        )
    {
        return false;
    }
    v.get("error")
        .is_some_and(|e| !e.is_null() && e.as_str() != Some(""))
        || v.get("ok").and_then(Value::as_bool) == Some(false)
        || matches!(
            v.pointer("/operation/state").and_then(Value::as_str),
            Some("failed" | "verification_failed")
        )
        || v.pointer("/result/exit_code")
            .and_then(Value::as_i64)
            .is_some_and(|n| n != 0)
        || v.pointer("/result/timed_out").and_then(Value::as_bool) == Some(true)
        || v.pointer("/summary/exit_code")
            .and_then(Value::as_i64)
            .is_some_and(|n| n != 0)
        || v.get("output")
            .and_then(Value::as_str)
            .is_some_and(|s| s.contains("import: command not found"))
}

/// Analyse saved evidence only. A streaming draft is not a completed answer.
pub fn detect(session: &Session, latest_only: bool) -> Vec<Incident> {
    let mut out = Vec::new();
    if session.entries.is_empty() {
        return out;
    }
    let start = if latest_only {
        session
            .entries
            .iter()
            .rposition(|e| e.message.role == "user")
            .unwrap_or(0)
    } else {
        0
    };
    let mut turn = start;
    let mut pending: Vec<(String, usize)> = Vec::new();
    let mut failed = Vec::new();
    let mut tools = 0;
    let mut successful_commands = 0;
    let mut code_writes = Vec::new();
    let mut check_observed = false;
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    let mut unread: HashMap<String, usize> = HashMap::new();
    for (i, e) in session.entries.iter().enumerate().skip(start) {
        let m = &e.message;
        if m.role == "user" {
            turn = i;
            pending.clear();
            failed.clear();
            groups.clear();
            tools = 0;
            successful_commands = 0;
            code_writes.clear();
            check_observed = false;
            unread.clear();
            let lower = m.content.to_ascii_lowercase();
            if [
                "not working",
                "doesn't work",
                "does not work",
                "different directory",
                "wrong folder",
                "syntax error:",
            ]
            .iter()
            .any(|s| lower.contains(s))
                || lower.trim_start().starts_with("you did not ")
                || lower.trim_start().starts_with("you didn't ")
            {
                add(&mut out,session,turn,i,("user_reported_problem","warning","heuristic","The user reported a possible failure or corrected the assistant.","Review the preceding answer and this correction. Quoted examples can trigger this rule; do not assume the complaint proves a defect."),&[i.saturating_sub(1)]);
            }
        }
        if e.stats
            .get("background_failures")
            .and_then(Value::as_array)
            .is_some_and(|f| !f.is_empty())
        {
            add(&mut out,session,turn,i,("background_task_failed","warning","observed","Optional post-answer processing failed.","Inspect background_failures for the task and reason. Check inference contention, model latency and timeout handling; this does not mean the completed answer failed."),&[]);
        }
        if e.stats.get("unexecuted_tool_text").and_then(Value::as_bool) == Some(true) {
            add(&mut out,session,turn,i,("unexecuted_tool_text","error","observed","The model emitted tool-shaped text that did not execute.","Inspect native tool-call support, tool schemas, prompt assembly, and the protocol recovery path."),&[]);
        }
        if let Some(kind) = e
            .stats
            .get("interruption_kind")
            .and_then(Value::as_str)
            .filter(|k| !matches!(*k, "cancelled" | "user_cancelled"))
        {
            add(&mut out,session,turn,i,("interrupted_answer","error","observed","The runtime recorded an incomplete or interrupted answer.","Inspect the entry's interruption_kind/failure_reason, transport, context budget, and runtime completion handling."),&[]);
            if let Some(last) = out.last_mut().filter(|x| x.anchor_entry_id == e.id) {
                last.summary = format!("Runtime interruption: {}", scrub(kind, 120));
            }
        }
        if m.role == "assistant" {
            if !m.tool_calls.is_empty() {
                pending.clear();
            }
            for call in &m.tool_calls {
                tools += 1;
                pending.push((call.function.name.clone(), i));
                if matches!(
                    call.function.name.as_str(),
                    "terminal_write" | "terminal_key"
                ) {
                    if let Some(previous) = call
                        .function
                        .arguments
                        .get("session_id")
                        .and_then(Value::as_str)
                        .and_then(|id| unread.get(id))
                    {
                        add(&mut out,session,turn,i,("input_before_output_drained","warning","heuristic","Input was sent while the previous terminal result still had unread output.","Read after=next_cursor until has_more is false; inspect the foreground prompt before typing. Interrupt keys may be intentional."),&[*previous]);
                    }
                }
            }
            let complete = m.tool_calls.is_empty()
                && e.stats.get("done").and_then(Value::as_bool) == Some(true)
                && e.stats.get("incomplete").and_then(Value::as_bool) != Some(true)
                && e.stats.get("superseded").and_then(Value::as_bool) != Some(true);
            let lower = m.content.to_ascii_lowercase();
            if complete
                && tools == 0
                && ["cannot access", "don't have access", "do not have access"]
                    .iter()
                    .any(|s| lower.contains(s))
                && ["terminal", "cli", "local"].iter().any(|s| {
                    session.entries[turn]
                        .message
                        .content
                        .to_ascii_lowercase()
                        .contains(s)
                })
            {
                add(&mut out,session,turn,i,("possible_capability_denial","warning","heuristic","The model denied local access without attempting a tool.","Check which tools were actually offered on this turn. This may be correct if tools were disabled; detection-time settings are not historical request evidence."),&[]);
            }
            if complete && !code_writes.is_empty() && successful_commands == 0 && !check_observed {
                add(&mut out, session, turn, i, ("generated_code_without_validation", "warning", "heuristic",
                    "Code was written, but this turn contains no subsequent successful command result or observed BASIC structure check.",
                    "Inspect validation evidence for the generated file. A persistent process may still be running; absence of a structured success is not proof the program is broken."), &code_writes);
            }
            if complete
                && !failed.is_empty()
                && successful_commands == 0
                && positive_execution_claim(&lower)
            {
                add(&mut out,session,turn,i,("possible_unsupported_success","warning","heuristic","The answer claims execution or validation despite tool failures and no successful command result in the turn.","Compare the exact claim with returned tool evidence. A successful file write does not prove execution. Inspect completion verification; do not treat this heuristic as a proven hallucination."),&failed.iter().rev().take(3).copied().collect::<Vec<_>>());
            }
        }
        if m.role != "tool" {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(&m.content) else {
            continue;
        };
        let name = m
            .tool_name
            .as_deref()
            .or_else(|| pending.first().map(|p| p.0.as_str()))
            .unwrap_or("unknown")
            .to_owned();
        let call = pending
            .iter()
            .position(|p| p.0 == name)
            .map(|p| pending.remove(p).1);
        if let Some(id) = v.get("session_id").and_then(Value::as_str) {
            if v.get("has_more").and_then(Value::as_bool) == Some(true) {
                unread.insert(id.into(), i);
            } else {
                unread.remove(id);
            }
        }
        if !failure(&v) {
            if matches!(name.as_str(), "write_file" | "replace_text" | "edit_file") {
                if let Some(path) = v.get("path").and_then(Value::as_str).or_else(|| {
                    call.and_then(|index| {
                        session.entries[index]
                            .message
                            .tool_calls
                            .iter()
                            .find(|c| c.function.name == name)
                            .and_then(|c| c.function.arguments.get("path"))
                            .and_then(Value::as_str)
                    })
                }) {
                    if Path::new(path)
                        .extension()
                        .and_then(|s| s.to_str())
                        .is_some_and(|s| {
                            ["bas", "py", "rs", "js", "ts", "c", "cpp", "sh"].contains(&s)
                        })
                    {
                        code_writes.push(i);
                        successful_commands = 0;
                        check_observed = false;
                    }
                }
            }
            if matches!(name.as_str(), "terminal_read" | "terminal_write")
                && v.get("output")
                    .and_then(Value::as_str)
                    .is_some_and(|s| s.lines().any(|line| line.trim() == "PROGRAM STRUCTURE OK."))
            {
                check_observed = true;
            }

            if matches!(name.as_str(), "run_command" | "run_python")
                && v.pointer("/result/exit_code").and_then(Value::as_i64) == Some(0)
            {
                successful_commands += 1;
            }
            continue;
        }
        failed.push(i);
        let error = v
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_ascii_lowercase();
        let stderr = v
            .pointer("/result/stderr")
            .and_then(Value::as_str)
            .unwrap_or("");
        let (code, summary, hint) = if error.contains("outside the connected project")
            || error.contains("workspace-relative")
        {
            ("workspace_mismatch","A tool request targeted a path outside its connected workspace.","Compare session workspace, requested path/cwd, and the user's intended destination. Resolve the workspace before retrying; do not strip an absolute path and claim the same location.")
        } else if error.contains("bash would interpret")
            || stderr.contains("cannot execute binary file")
            || v.get("output")
                .and_then(Value::as_str)
                .is_some_and(|s| s.contains("import: command not found"))
        {
            ("invalid_shell_invocation","Bash received an executable or non-shell script as a script argument.","Inspect program/args separation. Open Bash and type a command, or invoke the executable directly; preserve approval semantics.")
        } else if v.pointer("/operation/state").and_then(Value::as_str)
            == Some("verification_failed")
        {
            ("file_verification_failed","The tool claimed a change that disk verification could not confirm.","Inspect operation verification evidence, root/path normalization, atomic writes, and concurrent modifications.")
        } else {
            ("tool_failure","A tool returned an execution or validation failure.","Inspect the paired call and result; distinguish environmental failure, invalid arguments, and runtime defects before proposing a fix.")
        };
        add(
            &mut out,
            session,
            turn,
            i,
            (code, "error", "observed", summary, hint),
            &call.into_iter().collect::<Vec<_>>(),
        );
        let group = groups.entry(format!("{name}:{code}")).or_default();
        group.push(i);
        if group.len() == 3 {
            add(&mut out,session,turn,i,("repeated_tool_failure","error","observed","The same tool failed three times in one user turn.","Compare arguments and results for all attempts. Check recovery-loop stopping, path corrections, and whether the model adapted to the error."),group);
        }
    }
    if out.len() > 128 {
        out.drain(..out.len() - 128);
    }
    out
}

fn positive_execution_claim(text: &str) -> bool {
    text.split(['.', '!', '\n']).any(|sentence| {
        ![
            "not ",
            "cannot",
            "couldn't",
            "failed",
            "unverified",
            "unvalidated",
        ]
        .iter()
        .any(|s| sentence.contains(s))
            && [
                "runs successfully",
                "ran successfully",
                "tested successfully",
                "verified working",
                "game is working",
                "fully working",
                "working game",
                "successfully executed",
            ]
            .iter()
            .any(|s| sentence.contains(s))
    })
}

fn configuration(dir: &Path) -> Value {
    let parsed = fs::read(dir.join("settings.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .unwrap_or(Value::Null);
    let mut map = serde_json::Map::new();
    for key in [
        "model",
        "model_routing",
        "tools_enabled",
        "network_tools",
        "num_ctx",
        "num_predict",
        "max_tool_rounds",
        "auto_approve_commands",
        "auto_approve_unsandboxed",
        "auto_approve_file_changes",
    ] {
        if let Some(value) = parsed.get(key) {
            map.insert(key.into(), safe_value(value, 0));
        }
    }
    Value::Object(map)
}

pub fn manual(session: &Session) -> Option<Incident> {
    let at = session.entries.len().checked_sub(1)?;
    let turn = session
        .entries
        .iter()
        .rposition(|e| e.message.role == "user")
        .unwrap_or(0);
    let mut out = Vec::new();
    add(&mut out,session,turn,at,("user_flagged_conversation","warning","user_report","The user flagged this conversation for coder review.","Review the referenced turn and its saved history. This flag reports dissatisfaction, not a verified root cause."),&[turn]);
    out.pop()
}

/// SQLite supplies cross-process serialization and durable deduplication. JSONL
/// is an atomically replaced, bounded review snapshot, not an ever-growing trace.
pub fn record(conn: &Connection, dir: &Path, mut incidents: Vec<Incident>) -> Result<usize> {
    if incidents.is_empty() {
        refresh_log(conn, dir)?;
        return Ok(0);
    }
    let config = configuration(dir);
    let tx = rusqlite::Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
    let mut changed = 0;
    for incident in &mut incidents {
        incident.configuration_at_detection = config.clone();
        let payload = serde_json::to_string(incident)?;
        changed+=tx.execute("INSERT INTO conversation_incidents(id,session_id,code,first_seen,last_changed,payload) VALUES(?1,?2,?3,?4,?4,?5)
            ON CONFLICT(id) DO UPDATE SET last_changed=excluded.last_changed,payload=excluded.payload WHERE conversation_incidents.payload<>excluded.payload",
            params![incident.id,incident.session_id,incident.code,chrono::Utc::now().to_rfc3339(),payload])?;
    }
    let removed=tx.execute("DELETE FROM conversation_incidents WHERE id NOT IN (SELECT id FROM conversation_incidents ORDER BY last_changed DESC,id DESC LIMIT ?1)",[RETAINED_INCIDENTS as i64])?;
    if changed > 0 || removed > 0 {
        tx.execute(
            "UPDATE conversation_diagnostic_state SET dirty=1 WHERE id=1",
            [],
        )?;
    }
    tx.commit()?;
    // Commit incidents before export; an unwritable log cannot lose evidence.
    refresh_log(conn, dir)?;
    Ok(changed)
}

pub fn remove_session(conn: &Connection, dir: &Path, id: &str) -> Result<()> {
    let tx = rusqlite::Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
    tx.execute(
        "DELETE FROM conversation_incidents WHERE session_id=?1",
        [id],
    )?;
    tx.execute(
        "UPDATE conversation_diagnostic_state SET dirty=1 WHERE id=1",
        [],
    )?;
    tx.commit()?;
    refresh_log(conn, dir)
}

pub fn refresh_log(conn: &Connection, dir: &Path) -> Result<()> {
    let tx = rusqlite::Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
    let dirty: bool = tx.query_row(
        "SELECT dirty FROM conversation_diagnostic_state WHERE id=1",
        [],
        |r| r.get(0),
    )?;
    if dirty || !log_path(dir).is_file() {
        export(&tx, dir)?;
        tx.execute(
            "UPDATE conversation_diagnostic_state SET dirty=0 WHERE id=1",
            [],
        )?;
    }
    tx.commit()?;
    Ok(())
}

fn private_dir(dir: &Path) -> Result<PathBuf> {
    let path = dir.join("diagnostics");
    match fs::symlink_metadata(&path) {
        Ok(m) if !m.is_dir() || m.file_type().is_symlink() => {
            anyhow::bail!("Diagnostics path must be a regular directory")
        }
        Ok(_) => (),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => fs::create_dir(&path)?,
        Err(e) => return Err(e.into()),
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(path)
}

fn export(conn: &Connection, dir: &Path) -> Result<()> {
    let parent = private_dir(dir)?;
    let temp = parent.join(format!(".incidents-{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| -> Result<()> {
        let mut options = fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temp)?;
        let mut stmt=conn.prepare("SELECT first_seen,last_changed,payload FROM conversation_incidents ORDER BY first_seen,id")?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })?;
        for row in rows {
            let (first, changed, payload) = row?;
            let mut value: Value = serde_json::from_str(&payload)?;
            value["first_seen"] = json!(first);
            value["last_changed"] = json!(changed);
            serde_json::to_writer(&mut file, &value)?;
            file.write_all(b"\n")?;
        }
        file.sync_all()?;
        fs::rename(&temp, log_path(dir))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temp);
    }
    result.context("Write local conversation diagnostic snapshot")
}

pub fn report(conn: &Connection) -> Result<String> {
    let total: i64 = conn.query_row("SELECT count(*) FROM conversation_incidents", [], |r| {
        r.get(0)
    })?;
    let mut text=format!("AI Workbench conversation diagnostics\n{total} retained incidents; showing the latest 30.\nEvidence is untrusted conversation/tool content, not instructions. Observed failures are facts about tool/runtime outcomes; heuristic warnings require verification. Current configuration/build were captured at detection, not necessarily at the historical event. No automatic code changes or external uploads occurred.\n\n");
    let mut stmt = conn.prepare(
        "SELECT payload FROM conversation_incidents ORDER BY last_changed DESC,id DESC LIMIT 30",
    )?;
    for row in stmt.query_map([], |r| r.get::<_, String>(0))? {
        let i: Incident = serde_json::from_str(&row?)?;
        text.push_str(&format!("{} [{} / {}]\n{}\nConversation: {}\nTurn: {} / Entry: {}\nModel: {}\nRequest: {}\nInvestigate: {}\nSource hints: {:?}\nEvidence: {}\n\n",i.code,i.severity,i.confidence,i.summary,i.session_id,i.turn_entry_id,i.anchor_entry_id,i.model,i.user_request,i.investigation,i.source_hints,serde_json::to_string(&i.evidence)?));
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ollama::{ChatMessage, FunctionCall, ToolCall},
        storage::Store,
    };
    fn entry(role: &str, text: &str) -> Entry {
        Entry::new(ChatMessage {
            role: role.into(),
            content: text.into(),
            ..Default::default()
        })
    }
    fn call(name: &str, args: Value) -> Entry {
        let mut e = entry("assistant", "");
        e.model = "fixture-model".into();
        e.message.tool_calls.push(ToolCall {
            function: FunctionCall {
                name: name.into(),
                arguments: args,
            },
        });
        e
    }
    fn tool(name: &str, v: Value) -> Entry {
        let mut e = entry("tool", &v.to_string());
        e.message.tool_name = Some(name.into());
        e
    }
    fn failing_session() -> Session {
        let mut s = Session::default();
        s.entries.push(entry("user", "write a BASIC game"));
        for _ in 0..3 {
            s.entries.push(call(
                "run_command",
                json!({"program":"bash","args":["modernbasic"],"cwd":"/wrong"}),
            ));
            s.entries.push(tool("run_command",json!({"error":"That absolute path is outside the connected project folder","operation":{"state":"failed"}})));
        }
        s
    }
    #[test]
    fn reports_real_failure_chain_with_exact_entry_references() {
        let s = failing_session();
        let findings = detect(&s, false);
        assert_eq!(
            findings
                .iter()
                .filter(|i| i.code == "workspace_mismatch")
                .count(),
            3
        );
        let repeat = findings
            .iter()
            .find(|i| i.code == "repeated_tool_failure")
            .unwrap();
        assert_eq!(repeat.evidence.len(), 3);
        assert_eq!(repeat.model, "fixture-model");
        assert_eq!(repeat.turn_entry_id, s.entries[0].id);
        assert_eq!(repeat.evidence[0]["entry_id"], s.entries[2].id);
    }
    #[test]
    fn refusal_and_user_cancellation_are_not_tool_defects() {
        for result in [
            json!({"ok":false,"denied":true}),
            json!({"ok":false,"operation":{"state":"refused"}}),
            json!({"ok":false,"result":{"cancelled":true}}),
        ] {
            let s = Session {
                entries: vec![entry("user", "run this"), tool("run_command", result)],
                ..Default::default()
            };
            assert!(detect(&s, false).is_empty());
        }
        let mut s = Session::default();
        let mut stopped = entry("assistant", "Stopped");
        stopped.stats = json!({"interruption_kind":"cancelled","incomplete":true});
        s.entries = vec![entry("user", "run"), stopped];
        assert!(detect(&s, false).is_empty());
    }
    #[test]
    fn flags_shell_errors_even_when_terminal_delivery_succeeded() {
        let s = Session {
            entries: vec![
                entry("user", "open BASIC"),
                tool(
                    "terminal_open",
                    json!({"ok":true,"output":"import: command not found","summary":{"exit_code":2}}),
                ),
            ],
            ..Default::default()
        };
        assert_eq!(detect(&s, false)[0].code, "invalid_shell_invocation");
    }
    #[test]
    fn unsupported_claim_is_heuristic_only_on_completed_positive_answers() {
        let mut s = failing_session();
        let mut answer = entry("assistant", "The game is working!");
        s.entries.push(answer.clone());
        assert!(!detect(&s, false)
            .iter()
            .any(|i| i.code == "possible_unsupported_success"));
        answer.stats = json!({"done":true});
        *s.entries.last_mut().unwrap() = answer;
        assert!(detect(&s, false)
            .iter()
            .any(|i| i.code == "possible_unsupported_success" && i.confidence == "heuristic"));
        s.entries.last_mut().unwrap().message.content =
            "The game is not working; execution failed.".into();
        assert!(!detect(&s, false)
            .iter()
            .any(|i| i.code == "possible_unsupported_success"));
    }
    #[test]
    fn detects_input_before_unread_terminal_output_is_drained() {
        let mut s = Session {
            entries: vec![
                entry("user", "use BASIC"),
                tool(
                    "terminal_read",
                    json!({"session_id":"shell","has_more":true}),
                ),
                call("terminal_write", json!({"session_id":"shell","text":"RUN"})),
            ],
            ..Default::default()
        };
        assert!(detect(&s, false)
            .iter()
            .any(|i| i.code == "input_before_output_drained"));
        s.entries.insert(
            2,
            tool(
                "terminal_read",
                json!({"session_id":"shell","has_more":false}),
            ),
        );
        assert!(!detect(&s, false)
            .iter()
            .any(|i| i.code == "input_before_output_drained"));
    }
    #[test]
    fn persistence_deduplicates_redacts_and_rebuilds_missing_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path()).unwrap();
        let mut s = failing_session();
        s.entries[0].message.content =
            "run this\napi_key=DO-NOT-LOG-ME\nAuthorization: Bearer ALSO-SECRET".into();
        s.entries[1].message.tool_calls[0].function.arguments["token"] = json!("TOKEN-VALUE");
        store.save(&mut s).unwrap();
        let first = fs::read_to_string(log_path(dir.path())).unwrap();
        assert_eq!(first.lines().count(), 4);
        assert!(!first.contains("DO-NOT-LOG-ME"));
        assert!(!first.contains("ALSO-SECRET"));
        assert!(!first.contains("TOKEN-VALUE"));
        store.save(&mut s).unwrap();
        assert_eq!(first, fs::read_to_string(log_path(dir.path())).unwrap());
        fs::remove_file(log_path(dir.path())).unwrap();
        store.save(&mut s).unwrap();
        assert_eq!(first, fs::read_to_string(log_path(dir.path())).unwrap());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(log_path(dir.path()))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
    }
    #[test]
    fn log_failure_does_not_lose_chat_or_incidents_and_retry_recovers() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path()).unwrap();
        fs::write(dir.path().join("diagnostics"), "blocked").unwrap();
        let mut s = failing_session();
        store.save(&mut s).unwrap();
        assert_eq!(store.load(&s.id).unwrap().entries.len(), s.entries.len());
        assert!(store.diagnostic_error().is_some());
        assert!(store
            .conversation_diagnostic_report()
            .unwrap()
            .contains("workspace_mismatch"));
        fs::remove_file(dir.path().join("diagnostics")).unwrap();
        store.save(&mut s).unwrap();
        assert!(store.diagnostic_error().is_none());
        assert_eq!(
            fs::read_to_string(log_path(dir.path()))
                .unwrap()
                .lines()
                .count(),
            4
        );
    }
    #[test]
    fn historical_scan_preserves_chats_and_permanent_delete_purges_incidents() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path()).unwrap();
        let mut s = failing_session();
        store.save(&mut s).unwrap();
        let before = serde_json::to_value(store.load(&s.id).unwrap()).unwrap();
        let (count, _) = store.scan_conversation_diagnostics().unwrap();
        assert_eq!(count, 1);
        assert_eq!(
            before,
            serde_json::to_value(store.load(&s.id).unwrap()).unwrap()
        );
        store.move_to_trash(&s.id).unwrap();
        store.delete_trashed_permanently(&s.id).unwrap();
        assert!(fs::read_to_string(log_path(dir.path())).unwrap().is_empty());
    }
    #[test]
    fn retention_and_concurrent_writers_keep_valid_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let conn = Connection::open(dir.path().join("workbench.sqlite3")).unwrap();
        initialize(&conn).unwrap();
        let base = detect(&failing_session(), false).remove(0);
        let incidents = (0..RETAINED_INCIDENTS + 10)
            .map(|n| {
                let mut i = base.clone();
                i.id = format!("{n:06}");
                i
            })
            .collect();
        record(&conn, dir.path(), incidents).unwrap();
        let log = fs::read_to_string(log_path(dir.path())).unwrap();
        assert_eq!(log.lines().count(), RETAINED_INCIDENTS);
        for line in log.lines() {
            assert!(serde_json::from_str::<Incident>(line).is_ok());
        }
        // Separate connections exercise the same cross-process SQLite locking path.
        let root = dir.path().to_path_buf();
        let threads: Vec<_> = (0..2)
            .map(|_| {
                let root = root.clone();
                let store = Store::open(&root).unwrap();
                std::thread::spawn(move || {
                    store.save(&mut failing_session()).unwrap();
                    assert!(store.diagnostic_error().is_none());
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        for line in fs::read_to_string(log_path(dir.path())).unwrap().lines() {
            assert!(serde_json::from_str::<Value>(line).is_ok());
        }
    }
}

#[cfg(test)]
mod redaction_tests {
    use super::*;
    #[test]
    fn hides_split_flags_and_multiline_private_keys() {
        let value = safe_value(
            &json!({"args":["--token","VERY-SECRET","--password","ALSO-SECRET"]}),
            0,
        )
        .to_string();
        assert!(!value.contains("VERY-SECRET"));
        assert!(!value.contains("ALSO-SECRET"));
        assert!(!scrub(
            "-----BEGIN RSA PRIVATE KEY-----\nPRIVATE-DATA\n-----END RSA PRIVATE KEY-----",
            1000
        )
        .contains("PRIVATE-DATA"));
        assert!(!scrub("API key: SENSITIVE", 1000).contains("SENSITIVE"));
    }
}

#[cfg(test)]
mod generated_code_tests {
    use super::*;
    use crate::ollama::ChatMessage;
    #[test]
    fn a_write_is_not_validation_and_a_successful_command_is_recorded() {
        let mut answer = Entry::new(ChatMessage {
            role: "assistant".into(),
            content: "Saved your program.".into(),
            ..Default::default()
        });
        answer.stats = json!({"done":true});
        let mut s = Session {
            entries: vec![
                Entry::new(ChatMessage {
                    role: "user".into(),
                    content: "write BASIC".into(),
                    ..Default::default()
                }),
                Entry::new(ChatMessage {
                    role: "tool".into(),
                    tool_name: Some("write_file".into()),
                    content: json!({"ok":true,"path":"game.bas"}).to_string(),
                    ..Default::default()
                }),
                answer,
            ],
            ..Default::default()
        };
        assert!(detect(&s, false)
            .iter()
            .any(|i| i.code == "generated_code_without_validation"));
        s.entries.insert(
            2,
            Entry::new(ChatMessage {
                role: "tool".into(),
                tool_name: Some("run_command".into()),
                content: json!({"ok":true,"result":{"exit_code":0}}).to_string(),
                ..Default::default()
            }),
        );
        assert!(!detect(&s, false)
            .iter()
            .any(|i| i.code == "generated_code_without_validation"));
    }
}

/// Explicit full re-analysis replaces obsolete automatic findings but retains
/// manual flags. Live partial-turn scans never retract historical incidents.
pub fn reconcile(
    conn: &Connection,
    dir: &Path,
    session: &str,
    incidents: Vec<Incident>,
) -> Result<usize> {
    let keep: std::collections::HashSet<_> = incidents.iter().map(|i| i.id.clone()).collect();
    let changed = record(conn, dir, incidents)?;
    let tx = rusqlite::Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
    let stale = {
        let mut stmt=tx.prepare("SELECT id FROM conversation_incidents WHERE session_id=?1 AND code<>'user_flagged_conversation'")?;
        let ids = stmt
            .query_map([session], |r| r.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        ids.into_iter()
            .filter(|id| !keep.contains(id))
            .collect::<Vec<_>>()
    };
    for id in &stale {
        tx.execute("DELETE FROM conversation_incidents WHERE id=?1", [id])?;
    }
    if !stale.is_empty() {
        tx.execute(
            "UPDATE conversation_diagnostic_state SET dirty=1 WHERE id=1",
            [],
        )?;
    }
    tx.commit()?;
    refresh_log(conn, dir)?;
    Ok(changed)
}

#[cfg(test)]
mod complaint_regressions {
    use super::*;
    use crate::ollama::ChatMessage;
    #[test]
    fn review_template_is_not_a_user_complaint_and_stale_findings_are_retracted() {
        let dir = tempfile::tempdir().unwrap();
        let conn = Connection::open_in_memory().unwrap();
        initialize(&conn).unwrap();
        let mut s = Session {
            entries: vec![Entry::new(ChatMessage {
                role: "user".into(),
                content: "Review the project. End with the files or ranges you did not read."
                    .into(),
                ..Default::default()
            })],
            ..Default::default()
        };
        assert!(detect(&s, false).is_empty());
        let mut obsolete = manual(&s).unwrap();
        obsolete.code = "user_reported_problem".into();
        obsolete.id = "old-false-positive".into();
        record(&conn, dir.path(), vec![obsolete, manual(&s).unwrap()]).unwrap();
        reconcile(&conn, dir.path(), &s.id, detect(&s, false)).unwrap();
        assert_eq!(
            fs::read_to_string(log_path(dir.path()))
                .unwrap()
                .lines()
                .count(),
            1
        );
        s.entries[0].message.content = "you did not read the file I asked for".into();
        assert_eq!(detect(&s, false)[0].code, "user_reported_problem");
    }
}
