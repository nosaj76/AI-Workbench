//! Text detectors, tool-result helpers and report builders used by the chat
//! turn loop in `execution.rs`. Kept apart so the loop itself stays readable.

use super::terminal_intent::TerminalActionKind;
use super::*;
use std::collections::HashSet;

/// A request to write or build something in code ("write an html game flight
/// simulator", "build a python script that…"). Questions and requests to explain,
/// fix or review existing code are not.
pub(super) fn requests_code_creation(text: &str) -> bool {
    let user = crate::memory::user_text(text);
    let lower = user.trim().to_ascii_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric() && c != '+' && c != '#')
        .filter(|w| !w.is_empty())
        .collect();
    let Some(first) = words.first() else {
        return false;
    };
    if [
        "how",
        "what",
        "why",
        "when",
        "where",
        "who",
        "explain",
        "describe",
        "is",
        "are",
        "does",
        "do",
        "fix",
        "debug",
        "review",
        "summarize",
    ]
    .contains(first)
    {
        return false;
    }
    let verb = words.iter().any(|w| {
        [
            "write",
            "build",
            "create",
            "make",
            "implement",
            "code",
            "program",
            "develop",
            "generate",
        ]
        .contains(w)
    });
    let object = words.iter().any(|w| {
        [
            "game",
            "app",
            "application",
            "script",
            "program",
            "website",
            "webpage",
            "site",
            "simulator",
            "tool",
            "library",
            "function",
            "class",
            "module",
            "api",
            "bot",
            "cli",
            "server",
            "plugin",
            "extension",
            "parser",
            "engine",
            "html",
            "javascript",
            "js",
            "python",
            "rust",
            "c++",
            "java",
            "typescript",
            "css",
            "shader",
            "component",
            "code",
        ]
        .contains(w)
    });
    verb && object
}

/// A request to review, audit or inspect project code or files. Questions about how
/// to review something in general are not review requests.
pub(super) fn requests_project_review(text: &str) -> bool {
    let user = crate::memory::user_text(text);
    let lower = user.trim().to_ascii_lowercase();
    if ["how ", "what ", "why ", "when "]
        .iter()
        .any(|q| lower.starts_with(q))
        || [
            "without reading",
            "don't read",
            "do not read",
            "from memory",
        ]
        .iter()
        .any(|phrase| lower.contains(phrase))
    {
        return false;
    }
    let verb = [
        "review",
        "audit",
        "inspect",
        "examine",
        "analyze",
        "analyse",
        "go through",
        "look over",
    ]
    .iter()
    .any(|word| lower.contains(word));
    let object = [
        "project", "code", "repo", "source", "src/", "file", "crate", "module", "readme", ".rs",
        ".py", ".js", ".ts", ".go", ".c", ".h", ".toml", ".json", ".md",
    ]
    .iter()
    .any(|word| lower.contains(word));
    verb && object
}

pub(super) const CONTINUE_TAIL_CHARS: usize = 6_000;

pub(super) const CONTINUE_INSTRUCTION: &str = "Your previous answer was cut off. Continue exactly from the end of the assistant text below to finish the original user request. Do not restart or repeat earlier text; preserve any unfinished code block. Earlier omitted answer text remains saved.";

/// Generated answer fragments are saved in full by `Complete`, but only their
/// latest end belongs in the next inference request. This state is neither a
/// new user turn nor a permanent system instruction.
#[derive(Default)]
pub(super) struct AnswerContinuation {
    pub(super) tail: String,
    pub(super) generated_characters: usize,
    pub(super) after_history_messages: usize,
}

impl AnswerContinuation {
    pub(super) fn observe(&mut self, content: &str, messages: &[ChatMessage]) {
        self.generated_characters = self
            .generated_characters
            .saturating_add(content.chars().count());
        self.tail.push_str(content);
        let characters = self.tail.chars().count();
        if characters > CONTINUE_TAIL_CHARS {
            self.tail = self
                .tail
                .chars()
                .skip(characters - CONTINUE_TAIL_CHARS)
                .collect();
        }
        self.after_history_messages = messages
            .iter()
            .filter(|message| message.role != "system")
            .count();
    }

    pub(super) fn content(&self, keep: usize) -> String {
        let characters = self.tail.chars().count();
        let keep = keep.min(characters);
        let tail: String = self.tail.chars().skip(characters - keep).collect();
        let omitted = self.generated_characters.saturating_sub(keep);
        if omitted == 0 {
            tail
        } else {
            format!("[The first {omitted} characters of this reply are saved but left out here to fit the context. It ends with:]\n{tail}")
        }
    }

    pub(super) fn fit_request(
        &self,
        messages: &[ChatMessage],
        num_ctx: u32,
        num_predict: i32,
        tool_chars: usize,
    ) -> Result<(Vec<ChatMessage>, usize, usize, usize)> {
        let following_history = messages
            .iter()
            .filter(|message| message.role != "system")
            .count()
            .saturating_sub(self.after_history_messages);
        // First fit the genuine current turn and its evidence with only the
        // smallest continuation marker. The normal context policy may omit
        // older completed turns, but the original current user remains pinned.
        let mut minimum = messages.to_vec();
        let insertion = if following_history == 0 {
            minimum.len()
        } else {
            minimum
                .iter()
                .enumerate()
                .rev()
                .filter(|(_, message)| message.role != "system")
                .nth(following_history - 1)
                .map(|(index, _)| index)
                .unwrap_or(minimum.len())
        };
        minimum.insert(
            insertion,
            ChatMessage {
                role: "assistant".into(),
                content: self.content(1),
                ..Default::default()
            },
        );
        minimum.push(ChatMessage {
            role: "system".into(),
            content: CONTINUE_INSTRUCTION.into(),
            ..Default::default()
        });
        let (minimum, removed, compacted, thoughts_omitted) =
            context::fit_context_for_tool_round(&minimum, num_ctx, num_predict, tool_chars)?;
        // The fitter retains the complete current turn and puts system
        // messages first. Keep the partial answer before any tool exchanges
        // that occurred after it, rather than moving it past their results.
        let tail_index = minimum
            .len()
            .checked_sub(following_history + 1)
            .context("Continuation context lost its assistant tail")?;
        if minimum[tail_index].role != "assistant" || minimum[tail_index].content != self.content(1)
        {
            bail!("Continuation context did not preserve its answer position");
        }
        let fit_tail = |keep: usize| -> Result<(Vec<ChatMessage>, usize)> {
            let mut candidate = minimum.clone();
            candidate[tail_index].content = self.content(keep);
            context::fit_context_reserved(&candidate, num_ctx, num_predict, tool_chars)
        };
        let maximum = self.tail.chars().count();
        if let Ok((request, extra_removed)) = fit_tail(maximum) {
            return Ok((
                request,
                removed + extra_removed,
                compacted,
                thoughts_omitted,
            ));
        }
        // A fixed 6,000-character tail can itself exceed a small model's
        // usable context. Grow only generated text into the available room;
        // never shorten the real user request or compact further tool evidence.
        let mut best = (minimum.clone(), 0);
        let mut low = 2usize;
        let mut high = maximum.saturating_sub(1);
        while low <= high {
            let middle = low + (high - low) / 2;
            match fit_tail(middle) {
                Ok(request) => {
                    best = request;
                    low = middle + 1;
                }
                Err(_) => high = middle - 1,
            }
        }
        Ok((best.0, removed + best.1, compacted, thoughts_omitted))
    }
}

/// The files a review actually read this turn, appended to the answer so coverage
/// is recorded by the application rather than claimed by the model.
pub(super) fn review_coverage_footer(
    files_read: &std::collections::BTreeMap<String, Vec<String>>,
    inspected: &[String],
) -> String {
    if files_read.is_empty() {
        return format!(
            "\n\n---\n**Recorded by Workbench:** no file contents were read in this review; only {} results were seen. Statements about code inside files are not backed by a file read.",
            inspected.join(", ")
        );
    }
    let mut footer = format!(
        "\n\n---\n**Files actually read in this review ({}), recorded by Workbench:**\n",
        files_read.len()
    );
    for (path, ranges) in files_read {
        footer.push_str(&format!("- `{path}`: {}\n", ranges.join("; ")));
    }
    footer.push_str("Anything not listed here was not read and is not covered by this review.");
    footer
}

pub(super) fn explicitly_requests_web_search(text: &str) -> bool {
    let user = crate::memory::user_text(text);
    let lower = user.trim().to_ascii_lowercase();
    if [
        "do not search",
        "don't search",
        "dont search",
        "without web search",
        "no web search",
        "offline only",
        "do not browse",
        "don't browse",
    ]
    .iter()
    .any(|phrase| lower.contains(phrase))
    {
        return false;
    }
    let command = lower.as_str();
    let command = ["please ", "can you ", "could you ", "would you "]
        .iter()
        .find_map(|prefix| command.strip_prefix(*prefix))
        .unwrap_or(command);
    let command = command.strip_prefix("please ").unwrap_or(command);
    [
        "search the web",
        "search online",
        "search the internet",
        "internet search",
        "web search",
        "browse the web",
        "browse online",
        "look it up online",
        "look this up online",
    ]
    .iter()
    .any(|phrase| {
        command.strip_prefix(*phrase).is_some_and(|rest| {
            rest.chars()
                .next()
                .is_none_or(|ch| ch.is_whitespace() || matches!(ch, ':' | '?'))
        })
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RepeatScope {
    /// Deterministic local observations may be repeated after a successful
    /// state-changing call, because their result may then legitimately differ.
    Observation,
    /// Network/research/delegation requests are expensive but do not establish
    /// a new local workspace state. Keep exact repeats suppressed for the turn.
    Persistent,
    /// Approved actions may change files, processes, adapters, or MCP state.
    StateChanging,
}

pub(super) fn repeat_scope(name: &str) -> RepeatScope {
    match name {
        "calculator"
        | "list_files"
        | "list_directory"
        | "project_tree"
        | "repo_map"
        | "read_file"
        | "search_files"
        | "local_search"
        | "mcp_list"
        | "terminal_list"
        | "terminal_read"
        | "list_adapters"
        | "list_external_agents" => RepeatScope::Observation,
        "write_file" | "replace_text" | "edit_file" | "copy_file" | "move_file" | "delete_file"
        | "make_directory" | "mkdir" | "run_command" | "run_python" | "mcp_connect"
        | "mcp_call" | "consult_agent" | "run_adapter" | "terminal_open" | "terminal_write"
        | "terminal_key" | "terminal_resize" | "terminal_close" => RepeatScope::StateChanging,
        _ => RepeatScope::Persistent,
    }
}

pub(super) fn delegation_tool_definition() -> Value {
    serde_json::json!({"type":"function","function":{"name":"delegate_tasks","description":"Ask a bounded team of independent local subagents to investigate a complex objective, inspect workspace evidence and synthesize findings. Workers have independent context and read-only tools. No edits or commands are executed through delegation. Use at most twice per user request.","parameters":{"type":"object","properties":{"objective":{"type":"string"},"max_agents":{"type":"integer","minimum":1,"maximum":8}},"required":["objective"]}}})
}

pub(super) const INTERACTIVE_TERMINAL_GUIDANCE: &str = "MODERNBASIC WORK: Write in the requested BASIC dialect. Operate the terminal tools yourself. Open terminal_open with no arguments, then terminal_write text=\"modernbasic repl\" for a terminal session. Never pass a command name as Bash script arguments. Use cwd=\".\" in the connected library, not its parent directory. The only BASIC program library is ~/ModernBASIC/programs; create, load, and save .bas files there, using its existing category folders. Never create a BASIC program in the Workbench workspace, a Codex output folder, /tmp, or the ModernBASIC project source tree. If the user wants a visible graphical console, run modernbasic-console list. Use modernbasic-console send 'COMMAND' or modernbasic-console type --stdin to type into an open window. When no window exists, launch modernbasic from Bash; use another Bash session to run modernbasic-console while the graphical shell is open. For a terminal READY environment, run modernbasic repl. Read HELP for F1 Help, relevant MANUAL sections for F2 Manual, and GUIDES for F3 Guides when an example helps. Enter the program, CHECK, LIST, RUN, correct errors, and SAVE \"filename.bas\" only when the user asks to save. Wait for READY before typing the next command. Preserve the active session. Never claim a run or save without observing its output. Honor approvals and the tool budget.";

pub(super) const COMPACT_TERMINAL_GUIDANCE: &str = "Use the requested BASIC dialect. Keep every .bas file in the sole library ~/ModernBASIC/programs, using its category folders. For a visible console, use modernbasic-console list and send/type into a graphical ModernBASIC window, launching modernbasic if needed. For a terminal READY prompt, use modernbasic repl. Read HELP/F1 Help and relevant MANUAL/F2 Manual sections; use GUIDES/F3 Guides for beginner examples. Enter, CHECK, RUN and save only on request with SAVE \"filename.bas\". Wait for READY; inspect output.";

pub(super) const GENERAL_TERMINAL_GUIDANCE: &str = "Perform the requested CLI work through native terminal tools yourself. List/read existing sessions and reuse the correct session_id, or open Bash when needed. Inspect the current foreground program and its help before typing. Follow the user's requested action and exclusions; do not launch an unrelated program. terminal_write sends input to the foreground application. Read after=next_cursor and drain has_more; reread omitted output from start_cursor without resending input. Honor approvals. Silence or an alive shell is not proof of command success; report observed output and preserve a waiting session.";

pub(super) const COMPACT_GENERAL_TERMINAL_GUIDANCE: &str = "Use terminal tools for the requested CLI action yourself. Read/reuse the existing session or open Bash. Inspect its current program before typing; honor the user's exclusions and command approvals. Read after=next_cursor; do not repeat possibly delivered input or claim success from silence.";

pub(super) fn terminal_not_performed(tx: &Sender<ChatEvent>, reason: &str) {
    let _ = tx.send(ChatEvent::Stats(serde_json::json!({
        "incomplete": true,
        "interruption_kind": "no_terminal_action",
        "failure_reason": reason,
    })));
    let _ = tx.send(ChatEvent::Complete(ChatMessage {
        role: "assistant".into(),
        content: format!("**CLI task incomplete.** {reason}\n\nWorkbench provided terminal tools, but the requested CLI work has not been established by their results. Any terminal already opened remains available. Choose a model that uses native tools; if a profile pins the model, change it under Chat → Options before retrying."),
        ..Default::default()
    }));
}

/// A corrective instruction lasts for one inference, not the saved transcript.
/// Like optional coaching, it must fit without displacing selected evidence.
pub(super) fn add_terminal_recovery(
    messages: &mut Vec<ChatMessage>,
    definitions: &[Value],
    kind: TerminalActionKind,
    num_ctx: u32,
    num_predict: i32,
) -> bool {
    let instruction = match kind {
        TerminalActionKind::ModernBasic => "The user asked for ModernBASIC work. Use terminal tools now and keep every BASIC program under ~/ModernBASIC/programs. If a visible graphical console was requested, run modernbasic-console list, then modernbasic-console send/type into that window; launch modernbasic first if needed. Otherwise inspect/reuse a BASIC session or run modernbasic repl. Read HELP, relevant MANUAL sections, and GUIDES when beginner examples help. A Bash prompt alone is not BASIC. Submit pending input with Enter instead of retyping it. Do not repeat denied or possibly delivered input.",
        TerminalActionKind::Interactive => "The user asked you to operate a CLI. Use the native terminal tools to perform that action now; do not ask the user to open a CLI or type commands. Inspect and reuse an existing session when appropriate. Do not repeat denied or possibly delivered input.",
    };
    let Ok(tool_chars) = serde_json::to_string(definitions).map(|text| text.chars().count()) else {
        return false;
    };
    let Ok((_, baseline_removed)) =
        context::fit_context_reserved(messages, num_ctx, num_predict, tool_chars)
    else {
        return false;
    };
    let position = messages
        .iter()
        .rposition(|message| message.role == "user")
        .unwrap_or(messages.len());
    messages.insert(
        position,
        ChatMessage {
            role: "system".into(),
            content: instruction.into(),
            ..Default::default()
        },
    );
    if context::fit_context_reserved(messages, num_ctx, num_predict, tool_chars)
        .is_ok_and(|(_, removed)| removed <= baseline_removed)
    {
        true
    } else {
        messages.remove(position);
        false
    }
}

/// Add optional instructions only to a relevant turn and only when they fit
/// alongside the existing evidence and real tool schemas. Prefer concise help
/// over discarding another earlier conversation turn to make room for it.
pub(super) fn add_terminal_guidance(
    messages: &mut Vec<ChatMessage>,
    definitions: &[Value],
    intent: Option<TerminalActionKind>,
    num_ctx: u32,
    num_predict: i32,
) -> Option<&'static str> {
    if !definitions.iter().any(|definition| {
        definition.pointer("/function/name").and_then(Value::as_str) == Some("terminal_open")
    }) || !(intent.is_some() || has_terminal_session_history(messages))
    {
        return None;
    }
    let tool_chars = serde_json::to_string(definitions).ok()?.chars().count();
    let (_, baseline_removed) =
        context::fit_context_reserved(messages, num_ctx, num_predict, tool_chars).ok()?;
    let position = messages
        .iter()
        .rposition(|message| message.role == "user")
        .unwrap_or(messages.len());
    let candidates = if intent == Some(TerminalActionKind::ModernBasic) {
        [INTERACTIVE_TERMINAL_GUIDANCE, COMPACT_TERMINAL_GUIDANCE]
    } else {
        [GENERAL_TERMINAL_GUIDANCE, COMPACT_GENERAL_TERMINAL_GUIDANCE]
    };
    for guidance in candidates {
        messages.insert(
            position,
            ChatMessage {
                role: "system".into(),
                content: guidance.into(),
                ..Default::default()
            },
        );
        if context::fit_context_reserved(messages, num_ctx, num_predict, tool_chars)
            .is_ok_and(|(_, removed)| removed <= baseline_removed)
        {
            return Some(guidance);
        }
        messages.remove(position);
    }
    // Tool definitions still describe their use. Preserve the user's existing
    // evidence rather than making an otherwise valid request exceed its cap.
    None
}

/// Only genuine terminal tool results establish a continuing CLI session.
/// The newest observation of each id wins, so a closed/exited terminal does
/// not add terminal coaching to unrelated later work.
pub(super) fn has_terminal_session_history(messages: &[ChatMessage]) -> bool {
    let mut observed = HashSet::new();
    for message in messages
        .iter()
        .rev()
        .filter(|message| message.role == "tool")
    {
        let Some(name) = message.tool_name.as_deref().filter(|name| {
            matches!(
                *name,
                "terminal_open"
                    | "terminal_list"
                    | "terminal_read"
                    | "terminal_write"
                    | "terminal_key"
                    | "terminal_resize"
                    | "terminal_close"
            )
        }) else {
            continue;
        };
        let Ok(result) = serde_json::from_str::<Value>(&message.content) else {
            continue;
        };
        if name == "terminal_list" {
            if let Some(sessions) = result.get("sessions").and_then(Value::as_array) {
                for session in sessions {
                    if let Some(id) = session.get("id").and_then(Value::as_str) {
                        if observed.insert(id.to_owned())
                            && session.get("alive").and_then(Value::as_bool) == Some(true)
                        {
                            return true;
                        }
                    }
                }
            }
            continue;
        }
        if let Some(id) = result.get("session_id").and_then(Value::as_str) {
            let alive = result
                .pointer("/summary/alive")
                .or_else(|| result.get("alive"))
                .and_then(Value::as_bool);
            if observed.insert(id.to_owned())
                && result.get("closed").and_then(Value::as_bool) != Some(true)
                && alive != Some(false)
                && (alive == Some(true)
                    || result.get("ok").and_then(Value::as_bool) == Some(true)
                    || result.get("action_applied").and_then(Value::as_bool) == Some(true))
            {
                return true;
            }
        }
    }
    false
}

/// An explicit request to learn/use an installed CLI should start with that
/// program's observed help, even when generic online code examples are enabled.
#[cfg(test)]
pub(super) fn requests_interactive_cli(text: &str) -> bool {
    super::terminal_intent::mentions_interactive_cli(text)
}

/// A live terminal can produce output independently, and identical input may
/// answer successive prompts. The normal turn budget and bounded read waits
/// still apply; opening duplicate shells remains protected by the usual guard.
pub(super) fn duplicate_tool_request(
    seen: &mut HashSet<String>,
    name: &str,
    arguments: &Value,
    state_epoch: u64,
    scope: RepeatScope,
) -> bool {
    if matches!(
        name,
        "terminal_list" | "terminal_read" | "terminal_write" | "terminal_key" | "terminal_resize"
    ) {
        return false;
    }
    !seen.insert(tool_request_key(name, arguments, state_epoch, scope))
}

/// http(s) URLs written in free text.
pub(super) fn urls_in_text(text: &str) -> Vec<String> {
    text.split(|c: char| c.is_whitespace() || matches!(c, '<' | '>' | '"' | '\'' | '(' | ')'))
        .filter(|word| word.starts_with("http://") || word.starts_with("https://"))
        .map(|word| {
            word.trim_end_matches(['.', ',', ';', ':', '!', '?'])
                .to_owned()
        })
        .collect()
}

/// A URL reduced for matching: scheme, a leading `www.`, fragment and trailing
/// slash are ignored, so `https://www.x.org/a/` and `http://x.org/a` match.
pub(super) fn comparable_url(url: &str) -> String {
    let url = url.trim();
    let without_fragment = url.split('#').next().unwrap_or(url);
    let without_scheme = without_fragment
        .strip_prefix("https://")
        .or_else(|| without_fragment.strip_prefix("http://"))
        .unwrap_or(without_fragment);
    let without_www = without_scheme
        .strip_prefix("www.")
        .unwrap_or(without_scheme);
    without_www.trim_end_matches('/').to_ascii_lowercase()
}

/// One-line description of a tool call's arguments for the live activity view.
pub(super) fn tool_call_summary(arguments: &Value) -> String {
    let pick = |key: &str| arguments.get(key).and_then(Value::as_str);
    let text = if let Some(program) = pick("program") {
        let args = arguments
            .get("args")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_owned))
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_default();
        format!("$ {program} {args}")
    } else if let Some(value) = [
        // Argument keys used by preparation.rs tools, plus delegate_tasks.
        "path",
        "query",
        "question",
        "url",
        "source",
        "expression",
        "task",
        "objective",
        "tool_name",
        "adapter_id",
        "agent_id",
        "server_id",
    ]
    .iter()
    .find_map(|key| pick(key))
    {
        value.to_owned()
    } else if arguments.as_object().is_some_and(|o| o.is_empty()) {
        String::new()
    } else {
        arguments.to_string()
    };
    let text = text.trim().replace('\n', " ");
    match text.char_indices().nth(160) {
        Some((cut, _)) => format!("{}…", &text[..cut]),
        None => text,
    }
}

/// A short readable excerpt of a finished tool's result for the activity view.
pub(super) fn tool_result_preview(result: &Value) -> String {
    let text = if let Some(error) = result.get("error").and_then(Value::as_str) {
        format!("error: {error}")
    } else if let Some(message) = result
        .get("message")
        .or_else(|| result.get("reason"))
        .and_then(Value::as_str)
    {
        message.to_owned()
    } else {
        serde_json::to_string_pretty(result).unwrap_or_default()
    };
    match text.char_indices().nth(1_500) {
        Some((cut, _)) => format!("{}\n…", &text[..cut]),
        None => text,
    }
}

/// Where a reply that is stuck repeating the same block of lines started its
/// second copy, so the first copy is kept. A block of 1 to 12 lines repeated at
/// the end of the text (at least 6 times, and over at least 24 lines) counts;
/// blocks with no real content ("}", "0, 0,") are ignored.
pub(super) fn repetition_loop(text: &str) -> Option<usize> {
    let mut lines: Vec<(usize, &str)> = Vec::new();
    let mut start = 0;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim();
        if !trimmed.is_empty() && line.ends_with('\n') {
            lines.push((start, trimmed));
        }
        start += line.len();
    }
    let n = lines.len();
    for period in 1..=12usize {
        let needed = (24 / period).max(6);
        if n < period * needed {
            continue;
        }
        let block = &lines[n - period..];
        if !block
            .iter()
            .any(|(_, t)| t.chars().filter(|c| c.is_alphanumeric()).count() >= 8)
        {
            continue;
        }
        let mut repeats = 1;
        while (repeats + 1) * period <= n {
            let earlier = n - (repeats + 1) * period;
            let later = n - repeats * period;
            if (0..period).all(|k| lines[earlier + k].1 == lines[later + k].1) {
                repeats += 1;
            } else {
                break;
            }
        }
        if repeats >= needed {
            return Some(lines[n - (repeats - 1) * period].0);
        }
    }
    None
}

/// A GitHub source file (raw.githubusercontent.com, or a blob/tree/raw page)
/// inside a repository whose page a search result or the user listed.
pub(super) fn github_file_of_listed_repo(url: &str, listed: &HashSet<String>) -> bool {
    let comparable = comparable_url(url);
    let mut parts = comparable.split('/');
    let host = parts.next().unwrap_or("");
    let (owner, repo) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    if owner.is_empty() || repo.is_empty() {
        return false;
    }
    let inside = match host {
        "raw.githubusercontent.com" => true,
        "github.com" => matches!(parts.next(), Some("blob" | "tree" | "raw")),
        _ => false,
    };
    inside && listed.contains(&format!("github.com/{owner}/{repo}"))
}

/// Smallest context an out-of-memory retry will shrink to.
pub(super) const MIN_RETRY_CONTEXT: u32 = 4096;

/// Ollama's wording when a model or its context does not fit in memory.
pub(super) fn out_of_memory(error: &str) -> bool {
    let lower = error.to_ascii_lowercase();
    [
        "out of memory",
        "cudamalloc failed",
        "failed to allocate",
        "insufficient memory",
        "unable to allocate",
    ]
    .iter()
    .any(|phrase| lower.contains(phrase))
}

pub(super) fn tool_succeeded(result: &Value) -> bool {
    tool_result_succeeded(result)
}

pub(super) fn tool_result_succeeded(result: &Value) -> bool {
    if result.get("error").is_some_and(|error| !error.is_null())
        || result.get("denied").and_then(Value::as_bool) == Some(true)
        || result.get("cancelled").and_then(Value::as_bool) == Some(true)
        || result.get("executed").and_then(Value::as_bool) == Some(false)
        || result.get("success").and_then(Value::as_bool) == Some(false)
    {
        return false;
    }
    result.get("ok").and_then(Value::as_bool).unwrap_or(true)
}

pub(super) fn tool_request_key(
    name: &str,
    arguments: &Value,
    state_epoch: u64,
    repeat_scope: RepeatScope,
) -> String {
    fn canonical(value: &Value) -> Value {
        match value {
            Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
            Value::Object(object) => {
                let mut keys = object.keys().collect::<Vec<_>>();
                keys.sort_unstable();
                let mut normalized = serde_json::Map::new();
                for key in keys {
                    normalized.insert(key.as_str().to_owned(), canonical(&object[key.as_str()]));
                }
                Value::Object(normalized)
            }
            other => other.clone(),
        }
    }

    let scope = match repeat_scope {
        RepeatScope::Persistent => "persistent".to_string(),
        RepeatScope::Observation | RepeatScope::StateChanging => {
            format!("state:{state_epoch}")
        }
    };
    format!(
        "{scope}\n{}\n{}",
        name.trim(),
        serde_json::to_string(&canonical(arguments)).unwrap_or_else(|_| "null".into())
    )
}

/// A bounded, evidence-only progress report when the local model fails to
/// produce a final answer before tool use is stopped by a configured limit or
/// a no-progress guard. Tool successes are counted from genuine results, never
/// from tool requests or the model's own claims.
/// One last attempt at a real answer when the model will not stop asking for
/// tools: the evidence is restated without tool history (see
/// `context::flattened_finalization`). Returns the answer only when it is
/// plain text; the caller falls back to the progress report otherwise.
pub(super) async fn flattened_final_answer(
    client: &OllamaClient,
    settings: &Settings,
    model: &str,
    options: &ChatOptions,
    messages: &[ChatMessage],
    tx: &Sender<ChatEvent>,
    cancel: &Arc<AtomicBool>,
) -> Option<ChatMessage> {
    let flat = context::flattened_finalization(messages, settings.num_ctx, settings.num_predict)?;
    let _ = tx.send(ChatEvent::Status(
        "The model kept requesting tools; asking once more for a final answer from the evidence collected…".into(),
    ));
    let _ = tx.send(ChatEvent::Begin(model.to_string()));
    let permit = crate::routing::inference_permit(settings, cancel.clone())
        .await
        .ok()?;
    let answer = client
        .chat_stream(model, &flat, options, &[], cancel.clone(), |event| {
            let _ = tx.send(match event {
                StreamEvent::Content(s) => ChatEvent::Content(s),
                StreamEvent::Thinking(s) => ChatEvent::Thinking(s),
                StreamEvent::Stats(v) => ChatEvent::Stats(v),
            });
        })
        .await;
    drop(permit);
    let answer = answer.ok()?;
    (answer.tool_calls.is_empty()
        && !answer.content.trim().is_empty()
        && tool_protocol::textual_request_names(&answer.content).is_none())
    .then_some(answer)
}

pub(super) fn incomplete_review_report(
    messages: &[ChatMessage],
    limit: usize,
    reason: &str,
) -> String {
    use std::collections::BTreeSet;
    let mut opened = BTreeSet::new();
    let mut result_count = 0usize;
    let mut failures = 0usize;
    let mut skipped = 0usize;
    let mut partial = 0usize;
    // Never attribute evidence from previous conversations/turns to this review.
    let current_start = messages
        .iter()
        .rposition(|message| message.role == "user")
        .map_or(0, |index| index + 1);
    for message in messages
        .iter()
        .skip(current_start)
        .filter(|message| message.role == "tool")
    {
        let Ok(result) = serde_json::from_str::<Value>(&message.content) else {
            failures += 1;
            continue;
        };
        if result.get("duplicate").and_then(Value::as_bool) == Some(true)
            || result.get("executed").and_then(Value::as_bool) == Some(false)
        {
            skipped += 1;
            continue;
        }
        if !tool_result_succeeded(&result) {
            failures += 1;
            continue;
        }
        result_count += 1;
        if message.tool_name.as_deref() != Some("read_file")
            || result.get("content").and_then(Value::as_str).is_none()
        {
            continue;
        }
        let Some(path) = result.get("path").and_then(Value::as_str) else {
            continue;
        };
        // Tool-returned paths are display text, not trusted Markdown.
        let path = path
            .chars()
            .take(160)
            .collect::<String>()
            .replace('`', "'")
            .replace('\n', " ");
        let first = result
            .get("requested_start_line")
            .and_then(Value::as_u64)
            .unwrap_or(1);
        let last = result
            .get("requested_end_line")
            .and_then(Value::as_u64)
            .unwrap_or(first);
        let limited = result.get("truncated").and_then(Value::as_bool) == Some(true)
            || result.get("context_limited").and_then(Value::as_bool) == Some(true);
        if limited {
            partial += 1;
        }
        // An open-ended read asks for lines 1..u64::MAX; never print that.
        let range = if last >= u64::from(u32::MAX) {
            format!("lines {first}–end")
        } else {
            format!("lines {first}–{last}")
        };
        opened.insert(format!(
            "`{path}` (requested {range}{})",
            if limited { ", partial result" } else { "" }
        ));
    }
    if messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .is_some_and(|message| explicitly_requests_web_search(&message.content))
    {
        return format!(
            "## Web search incomplete\n\n{reason}\n\nWorkbench recorded {result_count} successful tool result(s), {failures} denied/failed result(s), and {skipped} skipped request(s) in this turn. This is an application status report, not a researched answer.\n\nInspect the web_search/fetch_url results above for provider errors or returned sources. A local directory listing is not a web search. Search requires Tools, network access, and an enabled configured provider; these settings are not enabled automatically."
        );
    }
    let rounds = messages
        .iter()
        .skip(current_start)
        .filter(|message| message.role == "assistant" && !message.tool_calls.is_empty())
        .count();
    let review = messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .is_some_and(|message| {
            let lower = crate::memory::user_text(&message.content).to_ascii_lowercase();
            requests_project_review(&message.content)
                || ["review", "audit", "inspect", "examine", "analyze", "analyse"]
                    .iter()
                    .any(|verb| lower.contains(verb))
        });
    // Only a review request gets the review wording. Writing a song or a
    // program that stops early is an unfinished answer, not a failed review.
    let (heading, product) = if review {
        ("Review incomplete — tool use stopped", "code review")
    } else {
        ("Answer incomplete — tool use stopped", "answer")
    };
    let mut text = format!(
        "## {heading}\n\n\
        {reason}\n\n\
        Workbench stopped after {rounds} tool round(s) (limit {limit}) without obtaining a final model {product}. \
        It recorded {result_count} successful tool result(s), {failures} denied/failed result(s), and {skipped} duplicate/skipped request(s). \
        This is a progress report from the application, **not a model-generated {product}**.\n\n"
    );
    if opened.is_empty() {
        text.push_str("**File contents verified:** None recorded through a successful `read_file` result in this turn.\n\n");
    } else {
        text.push_str(
            "**Files with recorded read results (requested ranges, not full-file coverage):**\n\n",
        );
        for file in opened.iter().take(12) {
            text.push_str(&format!("- {file}\n"));
        }
        if opened.len() > 12 {
            text.push_str(&format!(
                "- ... and {} further recorded ranges\n",
                opened.len() - 12
            ));
        }
        text.push('\n');
    }
    if partial > 0 {
        text.push_str(&format!("**Partial reads:** At least {partial} tool result(s) were marked truncated or context-limited.\n\n"));
    }
    if review {
        text.push_str("**Next step:** Use **Activities → Review queue** for a complete, checkpointed project review, or ask a smaller question about an explicitly named file. Tool results remain available in this conversation; no additional file changes were authorized by this report.");
    } else {
        text.push_str("**Next step:** Reply \"continue\" to retry with the evidence gathered so far, ask for a smaller piece (one file, section or step), or start a new chat if this one has grown long. Tool results remain available in this conversation; no additional file changes were authorized by this report.");
    }
    text
}

/// Existing folders named by absolute path in the user's message.
pub(super) fn local_dirs_in(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|token| token.trim_matches(|c: char| ",.;:)(\"'`".contains(c)))
        .filter(|token| {
            token.len() > 3 && token.starts_with('/') && std::path::Path::new(token).is_dir()
        })
        .map(str::to_owned)
        .collect()
}

/// What a finished chat turn contributes to the knowledge notes.
pub(super) struct NotesTurn {
    pub(super) question: String,
    /// The conversation's subject, for a follow-up question.
    pub(super) subject: Option<String>,
    /// Text to take sentences from, each with the index of its source.
    pub(super) parts: Vec<(String, Option<usize>)>,
    pub(super) basis: &'static str,
    pub(super) sources: Vec<crate::notes::IncomingSource>,
}

/// Tools whose results are public web material. A turn that used any other
/// tool touched the user's files, commands or accounts and is never noted.
pub(super) const NOTE_SAFE_TOOLS: [&str; 3] = ["web_search", "fetch_url", "qa_lookup"];

pub(super) const NOTE_MAX_SOURCES: usize = 8;

/// Web pages in a tool result: each page's title and URL as a source, and its
/// text (snippet or fetched content), aligned by index.
pub(super) fn collect_web_sources(
    value: &Value,
    sources: &mut Vec<crate::notes::IncomingSource>,
    bodies: &mut Vec<String>,
) {
    match value {
        Value::Object(map) => {
            let url = map.get("url").and_then(Value::as_str).unwrap_or("");
            if url.starts_with("http")
                && sources.len() < NOTE_MAX_SOURCES
                && !sources.iter().any(|s| s.url == url)
            {
                let title = map.get("title").and_then(Value::as_str).unwrap_or(url);
                let body: String = ["snippet", "text", "content", "excerpt", "answer"]
                    .iter()
                    .filter_map(|k| map.get(*k).and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("\n\n");
                bodies.push(body);
                sources.push(crate::notes::IncomingSource {
                    kind: "web".into(),
                    label: title.to_string(),
                    url: url.to_string(),
                });
            }
            for child in map.values() {
                collect_web_sources(child, sources, bodies);
            }
        }
        Value::Array(items) => {
            for child in items {
                collect_web_sources(child, sources, bodies);
            }
        }
        _ => {}
    }
}

/// The source label a conversation turn gets in knowledge notes.
pub(super) fn conversation_note_label(user: &str) -> String {
    let first: String = user
        .lines()
        .next()
        .unwrap_or("")
        .chars()
        .take(200)
        .collect();
    format!("Conversation: {first}")
}

pub(super) fn notes_turn(messages: &[ChatMessage], user: &str) -> Option<NotesTurn> {
    let start = messages.iter().rposition(|m| m.role == "user")?;
    let turn = &messages[start + 1..];
    let answer = turn
        .iter()
        .rev()
        .find(|m| m.role == "assistant" && !m.content.trim().is_empty())?
        .content
        .trim()
        .to_string();
    // Short replies are chat, not knowledge.
    if answer.chars().count() < 300 || user.chars().count() < 12 {
        return None;
    }
    let mut sources = Vec::new();
    let mut bodies = Vec::new();
    for message in turn.iter().filter(|m| m.role == "tool") {
        let name = message.tool_name.as_deref().unwrap_or("");
        if !NOTE_SAFE_TOOLS.contains(&name) {
            return None;
        }
        if let Ok(value) = serde_json::from_str::<Value>(&message.content) {
            collect_web_sources(&value, &mut sources, &mut bodies);
        }
    }
    // Knowledge notes keep facts taken from real pages. An answer from the model
    // alone is not saved: it is unverified, and saved model answers were the
    // source of wrong "facts" and of fragments from follow-up questions.
    if sources.is_empty() {
        return None;
    }
    // A follow-up ("soil requirements and environment conditions") belongs to
    // the subject the conversation started with, so its facts join that note
    // instead of starting a topic that does not even name the subject.
    let subject = messages[..start]
        .iter()
        .find(|m| m.role == "user")
        .map(|m| crate::memory::user_text(&m.content))
        .filter(|first| !first.trim().is_empty());
    // With web sources the notes come from the pages themselves, each fact
    // credited to the page it was taken from, not from the model's summary.
    Some(NotesTurn {
        question: user.to_string(),
        subject,
        parts: bodies
            .into_iter()
            .enumerate()
            .map(|(i, body)| (body, Some(i)))
            .collect(),
        basis: crate::notes::BASIS_WEB,
        sources,
    })
}

// Observe returned process output, never model prose or a stale terminal screen.
pub(super) fn basic_structure_checked(name: &str, result: &Value) -> bool {
    if !matches!(name, "terminal_read" | "terminal_write" | "run_command")
        || !tool_succeeded(result)
    {
        return false;
    }
    let output = result
        .get("output")
        .or_else(|| result.pointer("/result/stdout"))
        .and_then(Value::as_str)
        .unwrap_or("");
    output
        .lines()
        .any(|line| line.trim() == "PROGRAM STRUCTURE OK.")
        && !output.contains("?syntax error")
        && !output.contains("error:")
}

pub(super) fn boundary_failure_class(result: &Value) -> Option<&'static str> {
    let error = result
        .get("error")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    if error.contains("outside the connected project")
        || error.contains("do not expand '~'")
        || error.contains("cwd must be an existing")
    {
        Some("workspace")
    } else if error.contains("bash would interpret")
        || error.contains("interpreter is repeated")
        || error.contains("python is not installed")
    {
        Some("invocation")
    } else {
        None
    }
}

const TERMINAL_TOOLS: [&str; 7] = [
    "terminal_open",
    "terminal_list",
    "terminal_read",
    "terminal_write",
    "terminal_key",
    "terminal_resize",
    "terminal_close",
];
const ADAPTER_TOOLS: [&str; 2] = ["list_adapters", "run_adapter"];
const GIT_TOOLS: [&str; 4] = ["git_status", "git_diff", "git_log", "git_commit"];

fn mentions_any(text: &str, words: &[&str]) -> bool {
    let lower = text.to_ascii_lowercase();
    let tokens: HashSet<&str> = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    words.iter().any(|word| {
        if word.contains(' ') {
            lower.contains(word)
        } else {
            tokens.contains(word)
        }
    })
}

/// Drop tool groups this turn clearly does not need, so their schemas do not
/// crowd the conversation out of a small context. Only the advertised list
/// changes: a native call to a dropped tool still goes through approval as
/// usual. `terminal_in_use` covers a terminal request, earlier terminal
/// results in the conversation, or terminals the AI already has open.
pub(super) fn scope_tools_to_turn(definitions: &mut Vec<Value>, user: &str, terminal_in_use: bool) {
    let terminal = terminal_in_use
        || mentions_any(
            user,
            &[
                "terminal",
                "shell",
                "bash",
                "zsh",
                "console",
                "repl",
                "interactive",
                "tty",
                "cli",
                "command line",
                "modernbasic",
            ],
        );
    let adapters = mentions_any(
        user,
        &[
            "speak",
            "speech",
            "aloud",
            "transcribe",
            "transcription",
            "audio",
            "voice",
            "tts",
            "adapter",
            "adapters",
        ],
    );
    let git = mentions_any(
        user,
        &[
            "git", "commit", "commits", "branch", "branches", "diff", "staged", "stage", "changes",
            "changed", "revert", "merge", "history",
        ],
    );
    definitions.retain(|definition| {
        let name = definition
            .pointer("/function/name")
            .and_then(Value::as_str)
            .unwrap_or("");
        (terminal || !TERMINAL_TOOLS.contains(&name))
            && (adapters || !ADAPTER_TOOLS.contains(&name))
            && (git || !GIT_TOOLS.contains(&name))
    });
}
