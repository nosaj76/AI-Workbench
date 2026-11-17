//! Conservative recognition of requests to operate an interactive program.
//! Recognition only enables the ordinary tool workflow; it never turns text
//! into an executable command or bypasses an action's approval.

use super::ChatMessage;
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TerminalActionKind {
    Interactive,
    ModernBasic,
}

/// Shared by the desktop's workspace preparation and the runtime's evidence
/// guard. Explanations, quoted material and attachments are not launch requests.
#[cfg(test)]
pub(crate) fn requests_terminal_action(text: &str) -> bool {
    terminal_action_kind(text).is_some()
}

/// A brief correction belongs to the preceding BASIC task in this chat.
/// Follow a bounded chain of short corrections; unrelated tasks break the chain.
pub(super) fn terminal_action_for_turn(messages: &[ChatMessage]) -> Option<TerminalActionKind> {
    let mut users = messages
        .iter()
        .rev()
        .filter(|message| message.role == "user");
    let latest = direct_user_text(&users.next()?.content);
    if let Some(kind) = terminal_action_kind(&latest) {
        return Some(kind);
    }
    let words = words(&latest);
    let continuation = words.len() <= 18
        && (["proceed", "continue", "try again", "run it", "save it"].contains(&latest.as_str())
            || [
                "not ",
                "no ",
                "instead ",
                "make it ",
                "keep it ",
                "use ",
                "produce ",
                "write a ",
                "place the ",
                "you must be working",
                "/",
            ]
            .iter()
            .any(|prefix| latest.starts_with(prefix)))
        && ![
            "python",
            "javascript",
            "typescript",
            "rust",
            "html",
            "webpage",
        ]
        .iter()
        .any(|language| words.contains(language) && !latest.starts_with("not "));
    if !continuation || latest.contains("not basic") || latest.contains("no basic") {
        return None;
    }
    // Walk a bounded chain of short corrections, never cross an unrelated task.
    let index = messages.iter().rposition(|m| m.role == "user")?;
    let prior = &messages[..index];
    let previous = prior.iter().rev().find(|m| m.role == "user")?;
    if terminal_action_kind(&previous.content) == Some(TerminalActionKind::ModernBasic) {
        return Some(TerminalActionKind::ModernBasic);
    }
    if prior.iter().filter(|m| m.role == "user").count() > 32 {
        return None;
    }
    terminal_action_for_turn(prior)
}

/// ModernBASIC guidance only makes sense where its `modernbasic` launcher is
/// installed; elsewhere a BASIC request is ordinary chat. Unit tests assume it.
pub(crate) fn modernbasic_installed() -> bool {
    static INSTALLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    cfg!(test)
        || *INSTALLED.get_or_init(|| crate::diagnostics::find_executable("modernbasic").is_some())
}

pub(crate) fn requests_modernbasic_for_turn(messages: &[ChatMessage]) -> bool {
    terminal_action_for_turn(messages) == Some(TerminalActionKind::ModernBasic)
}

pub(crate) fn requests_terminal_action_for_turn(messages: &[ChatMessage]) -> bool {
    terminal_action_for_turn(messages).is_some()
}

fn direct_user_text_with_case(text: &str) -> String {
    let user = crate::memory::user_text(text);
    let mut fenced = false;
    let mut lines = Vec::new();
    for line in user.lines() {
        let line = line.trim();
        if line.starts_with("```") || line.starts_with("~~~") {
            fenced = !fenced;
        } else if !fenced && !line.starts_with('>') {
            lines.push(line);
        }
    }
    let text = lines.join(" ").trim().to_owned();
    if [('"', '"'), ('\'', '\''), ('“', '”')]
        .iter()
        .any(|(start, end)| text.starts_with(*start) && text.ends_with(*end))
    {
        String::new()
    } else {
        text
    }
}

fn direct_user_text(text: &str) -> String {
    direct_user_text_with_case(text).to_ascii_lowercase()
}

fn words(text: &str) -> Vec<&str> {
    text.split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect()
}

#[cfg(test)]
fn names_modernbasic(words: &[&str]) -> bool {
    words.contains(&"modernbasic") || words.windows(2).any(|pair| pair == ["modern", "basic"])
}

fn requests_modernbasic_program(user: &str, words: &[&str], names_basic_language: bool) -> bool {
    // "basic" can also mean simple. A positively requested different
    // language wins, while "not HTML" does not cancel a BASIC request.
    if words.iter().enumerate().any(|(index, word)| {
        ["python", "javascript", "typescript", "rust", "html", "java"].contains(word)
            && !words[index.saturating_sub(3)..index]
                .iter()
                .any(|prior| ["not", "no", "without", "instead"].contains(prior))
    }) {
        return false;
    }
    let has_basic = words.contains(&"basic")
        || words.contains(&"modernbasic")
        || user.split_whitespace().any(|word| word.ends_with(".bas"));
    let explicit_dialect = [
        "in basic",
        "with basic",
        "using basic",
        "as basic",
        "in modernbasic",
        "with modernbasic",
        "using modernbasic",
        "in modern basic",
        "with modern basic",
        "using modern basic",
    ]
    .iter()
    .any(|phrase| user.contains(phrase))
        || user.split_whitespace().any(|word| word.ends_with(".bas"));
    let has_program_request = [
        "script",
        "program",
        "code",
        "listing",
        "exercise",
        "app",
        "game",
        "maze",
        "calculator",
        "converter",
        "tool",
        "utility",
    ]
    .iter()
    .any(|word| words.contains(word))
        || (names_basic_language
            && [
                "music",
                "melody",
                "song",
                "sound",
                "effect",
                "effects",
                "graphics",
                "graphic",
                "animation",
                "art",
            ]
            .iter()
            .any(|word| words.contains(word)));
    has_basic
        && (has_program_request || explicit_dialect)
        && [
            "write", "create", "make", "build", "show", "run", "execute", "draw", "compose",
            "produce", "design", "develop", "generate", "code",
        ]
        .iter()
        .any(|verb| words.contains(verb))
}

#[cfg(test)]
pub(super) fn mentions_interactive_cli(text: &str) -> bool {
    let user = direct_user_text(text);
    let words = words(&user);
    names_modernbasic(&words)
        || words.contains(&"repl")
        || [
            "interactive cli",
            "interactive terminal",
            "interactive shell",
        ]
        .iter()
        .any(|phrase| user.contains(phrase))
        || terminal_action_kind(text).is_some()
}

pub(super) fn terminal_action_kind(text: &str) -> Option<TerminalActionKind> {
    let direct = direct_user_text_with_case(text);
    let user = direct.to_ascii_lowercase();
    if [
        "without opening",
        "without running",
        "without launching",
        "without using",
        "do not open it",
        "don't open it",
        "dont open it",
        "do not launch it",
        "don't launch it",
        "dont launch it",
        "do not run it",
        "don't run it",
        "dont run it",
    ]
    .iter()
    .any(|phrase| user.contains(phrase))
    {
        return None;
    }
    let mut command = user.trim();
    for _ in 0..3 {
        let Some(rest) = [
            "please ",
            "just ",
            "can you ",
            "could you ",
            "would you ",
            "will you ",
            "i want you to ",
            "i need you to ",
            "i would like you to ",
            "i'd like you to ",
            "i want to ",
            "i need to ",
            "help me ",
            "let's ",
            "lets ",
        ]
        .iter()
        .find_map(|prefix| command.strip_prefix(*prefix)) else {
            break;
        };
        command = rest.trim_start();
    }
    let words = words(command);
    if requests_modernbasic_program(
        &user,
        &words,
        direct.contains("BASIC") || words.contains(&"modernbasic"),
    ) {
        return Some(TerminalActionKind::ModernBasic);
    }
    let first = *words.first()?;
    if ![
        "open", "launch", "start", "run", "use", "type", "enter", "execute", "continue",
    ]
    .contains(&first)
    {
        return None;
    }
    // An instruction to use a word in prose is not an instruction to run its
    // namesake. Keep this classifier narrower than optional topic guidance.
    if [
        "phrase",
        "word",
        "title",
        "quotation",
        "quote",
        "example",
        "pseudocode",
    ]
    .iter()
    .any(|word| words.iter().take(5).any(|part| part == word))
    {
        return None;
    }
    let mut modernbasic = None;
    let mut interactive = None;
    for (index, word) in words.iter().enumerate() {
        if ![
            "use", "run", "start", "launch", "open", "type", "execute", "continue",
        ]
        .contains(word)
        {
            continue;
        }
        let prior = &words[index.saturating_sub(3)..index];
        let negated = prior
            .iter()
            .any(|part| ["not", "dont", "never"].contains(part))
            || prior.windows(2).any(|pair| pair == ["don", "t"]);
        let tail = &words[index + 1..];
        let start = tail
            .iter()
            .position(|part| {
                ![
                    "a",
                    "an",
                    "the",
                    "my",
                    "our",
                    "your",
                    "installed",
                    "local",
                    "existing",
                    "new",
                    "persistent",
                    "interactive",
                    "in",
                ]
                .contains(part)
            })
            .unwrap_or(tail.len());
        if tail.get(start) == Some(&"modernbasic")
            || tail.get(start..start.saturating_add(2)) == Some(&["modern", "basic"])
        {
            // A later explicit countermand wins for this target, without
            // cancelling a separately requested Bash-only action.
            modernbasic = Some(!negated);
        } else if tail
            .get(start)
            .is_some_and(|word| ["bash", "terminal", "shell", "cli", "repl"].contains(word))
        {
            interactive = Some(!negated);
        }
    }
    if modernbasic == Some(true) {
        return Some(TerminalActionKind::ModernBasic);
    }
    (interactive == Some(true)
        || (modernbasic != Some(false)
            && interactive != Some(false)
            && ["bash", "terminal", "shell", "cli", "repl"]
                .iter()
                .any(|word| words.contains(word))))
    .then_some(TerminalActionKind::Interactive)
}

/// This tracks observed attempts, not a claim that the foreground application
/// succeeded. An idle/alive Bash, a title, or a model's prose is never evidence
/// that ModernBASIC was launched.
pub(super) struct TerminalProgress {
    kind: TerminalActionKind,
    handled: bool,
    blocked: bool,
    target_sessions: HashSet<String>,
    pending_input: HashMap<String, String>,
}

impl TerminalProgress {
    pub(super) fn new(kind: TerminalActionKind, history: &[ChatMessage]) -> Self {
        let mut progress = Self {
            kind,
            handled: false,
            blocked: false,
            target_sessions: HashSet::new(),
            pending_input: HashMap::new(),
        };
        let mut pending = VecDeque::new();
        for message in history {
            if message.role == "assistant" {
                pending.clear();
                pending.extend(message.tool_calls.iter().map(|call| call.function.clone()));
            } else if message.role == "user" {
                pending.clear();
            } else if message.role == "tool" {
                let Some(name) = message.tool_name.as_deref() else {
                    continue;
                };
                let Some(index) = pending.iter().position(|call| call.name == name) else {
                    continue;
                };
                let call = pending.remove(index).expect("matched pending call");
                if let Ok(result) = serde_json::from_str::<Value>(&message.content) {
                    progress.observe(name, &call.arguments, &result);
                }
            }
        }
        // Previous attempts cannot satisfy this new request. They only identify
        // sessions that may be inspected and reused without duplicate input.
        progress.handled = false;
        progress.blocked = false;
        progress
    }

    pub(super) fn needs_action(&self) -> bool {
        !self.handled && !self.blocked
    }

    pub(super) fn observe(&mut self, name: &str, args: &Value, result: &Value) {
        if !name.starts_with("terminal_") {
            return;
        }
        let id = result
            .get("session_id")
            .and_then(Value::as_str)
            .or_else(|| args.get("session_id").and_then(Value::as_str));
        let alive = result
            .pointer("/summary/alive")
            .or_else(|| result.get("alive"))
            .and_then(Value::as_bool);
        let closing =
            result.get("closed").and_then(Value::as_bool) == Some(true) || alive == Some(false);
        if let Some(id) = id.filter(|_| closing) {
            self.target_sessions.remove(id);
            self.pending_input.remove(id);
        }
        if name == "terminal_list" {
            if let Some(sessions) = result.get("sessions").and_then(Value::as_array) {
                self.target_sessions.retain(|id| {
                    sessions.iter().any(|session| {
                        session.get("id").and_then(Value::as_str) == Some(id)
                            && session.get("alive").and_then(Value::as_bool) == Some(true)
                    })
                });
            }
        }
        if !super::execution::tool_succeeded(result)
            || result
                .get("input_may_have_been_sent")
                .and_then(Value::as_bool)
                == Some(true)
        {
            // A denial, failure or uncertain input is a real outcome. Never
            // pressure the model into repeating it merely to satisfy this guard.
            self.blocked = true;
            if let Some(id) = id {
                self.pending_input.remove(id);
            }
            return;
        }
        let launch = self.submitted_launch(name, args, id);
        if name == "terminal_close" {
            if let Some(id) = id {
                self.target_sessions.remove(id);
                self.pending_input.remove(id);
            }
        }
        if launch {
            self.handled = true;
            if let Some(id) = id.filter(|_| alive == Some(true)) {
                self.target_sessions.insert(id.to_owned());
            }
        } else if (self.kind == TerminalActionKind::Interactive
            && matches!(
                name,
                "terminal_open" | "terminal_write" | "terminal_key" | "terminal_close"
            ))
            || (matches!(name, "terminal_read" | "terminal_write" | "terminal_key")
                && alive == Some(true)
                && id.is_some_and(|id| self.target_sessions.contains(id)))
        {
            // A generic request was operated, or this is fresh observation of
            // a natively launched target. An arbitrary title is not evidence.
            self.handled = true;
        }
        if result.get("input_sent").and_then(Value::as_bool) == Some(true)
            && (name == "terminal_key"
                || (name == "terminal_write"
                    && args
                        .get("text")
                        .and_then(Value::as_str)
                        .is_some_and(|text| {
                            !text.is_empty()
                                && (args.get("enter").and_then(Value::as_bool) != Some(false)
                                    || text.contains(['\r', '\n']))
                        })))
        {
            // A manually opened/shared session may have no launch in this
            // chat's history; complex shell input cannot be interpreted safely
            // here either. Actual submitted input is work already attempted,
            // not permission to pressure a second launch into its foreground
            // program. Preserve the result without claiming target success.
            self.blocked = true;
        }
        if let Some(id) = id {
            let leaves_program = (name == "terminal_key"
                && matches!(
                    args.get("key").and_then(Value::as_str),
                    Some("eof" | "interrupt")
                ))
                || (name == "terminal_write"
                    && args
                        .get("text")
                        .and_then(Value::as_str)
                        .is_some_and(|text| {
                            matches!(
                                text.trim().to_ascii_lowercase().as_str(),
                                "bye" | "quit" | "exit" | "system"
                            )
                        }));
            if leaves_program {
                self.target_sessions.remove(id);
                self.pending_input.remove(id);
            }
        }
    }

    /// Observe native input submissions only. A partial write followed by Enter
    /// is equivalent to a submitted write, but editing keys make the pending
    /// buffer unknown. This tracker never sends or reconstructs executable input.
    fn submitted_launch(&mut self, name: &str, args: &Value, id: Option<&str>) -> bool {
        if name == "terminal_open" {
            return launches_modernbasic(name, args);
        }
        let Some(id) = id else { return false };
        if name == "terminal_key" {
            let pending = self.pending_input.remove(id);
            return args.get("key").and_then(Value::as_str) == Some("enter")
                && pending.as_deref().is_some_and(modernbasic_command);
        }
        if name != "terminal_write" {
            return false;
        }
        let Some(text) = args.get("text").and_then(Value::as_str) else {
            return false;
        };
        let mut combined = self.pending_input.remove(id).unwrap_or_default();
        combined.push_str(text);
        if args.get("enter").and_then(Value::as_bool) != Some(false) {
            return modernbasic_command(&combined);
        }
        let end = combined
            .rfind(['\r', '\n'])
            .map(|index| index + 1)
            .unwrap_or(0);
        let submitted = modernbasic_command(&combined[..end]);
        let partial = &combined[end..];
        if !partial.is_empty() && partial.len() <= 1024 && self.pending_input.len() < 64 {
            self.pending_input.insert(id.to_owned(), partial.to_owned());
        }
        submitted
    }
}

fn is_modernbasic_program(program: &str) -> bool {
    program
        .trim_matches(['\'', '"', '`'])
        .rsplit('/')
        .next()
        .is_some_and(|name| name.eq_ignore_ascii_case("modernbasic"))
}

fn modernbasic_command(text: &str) -> bool {
    // Only remember simple, unambiguous invocations. This is not a shell
    // parser: quotes, substitutions and operators remain uncertain. Delivered
    // input receipts suppress retries separately, including `env ...` forms.
    if text.contains(['\'', '"', '`', '$', ';', '&', '|', '<', '>', '\\', '#']) {
        return false;
    }
    text.split(['\n', '\r']).any(|command| {
        let mut parts = command.split_whitespace();
        let Some(mut program) = parts.next() else {
            return false;
        };
        if matches!(program, "command" | "exec") {
            program = parts.next().unwrap_or("");
        }
        is_modernbasic_program(program)
    })
}

fn launches_modernbasic(name: &str, args: &Value) -> bool {
    if name != "terminal_open" {
        return false;
    }
    let program = args
        .get("program")
        .and_then(Value::as_str)
        .unwrap_or("bash");
    if is_modernbasic_program(program) {
        return true;
    }
    if !matches!(program.rsplit('/').next(), Some("bash" | "sh")) {
        return false;
    }
    let Some(arguments) = args.get("args").and_then(Value::as_array) else {
        return false;
    };
    arguments.windows(2).any(|pair| {
        pair[0]
            .as_str()
            .is_some_and(|flag| matches!(flag, "-c" | "-ic" | "-lc" | "-lic"))
            && pair[1].as_str().is_some_and(modernbasic_command)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn terminal_action_requests_include_spaced_alias_and_polite_requests() {
        for request in [
            "use modern basic",
            "use modernbasic",
            "Please open ModernBASIC",
            "Can you launch modern basic?",
            "I want to use modernbasic",
            "write a BASIC program",
            "create a BASIC script",
            "build a BASIC game",
            "make a maze in BASIC",
            "write a song in BASIC",
            "create a BASIC melody",
            "produce graphics in modernbasic",
            "compose a BASIC sound effect",
            "draw a picture in BASIC",
            "build a calculator using BASIC",
            "write a BASIC game, not HTML",
            "run demo.bas",
            "I need you to open a persistent Bash terminal",
            "Open Bash and type modernbasic",
            "Run python in an interactive terminal",
            "Start a REPL",
            "Please use Bash CLI tools",
            "use modern basic and don't ask me to open the CLI",
            "use modernbasic and do not ask me to type commands",
        ] {
            assert!(requests_terminal_action(request), "{request}");
        }
        assert_eq!(
            terminal_action_kind("use modern basic"),
            Some(TerminalActionKind::ModernBasic)
        );
        assert_eq!(
            terminal_action_kind("Open Bash and type modernbasic"),
            Some(TerminalActionKind::ModernBasic)
        );
        assert_eq!(
            terminal_action_kind("Open Bash"),
            Some(TerminalActionKind::Interactive)
        );
        assert_eq!(
            terminal_action_kind("Open Bash but do not run modernbasic"),
            Some(TerminalActionKind::Interactive)
        );
        assert_eq!(
            terminal_action_kind("Use modernbasic but do not launch modernbasic"),
            None
        );
        assert_eq!(
            terminal_action_kind("use modern basic and don't ask me to open the CLI"),
            Some(TerminalActionKind::ModernBasic)
        );
    }

    #[test]
    fn basic_corrections_keep_the_task_without_leaking_to_unrelated_intents() {
        let turn = |latest: &str| {
            vec![
                ChatMessage {
                    role: "user".into(),
                    content: "write a BASIC program for a maze game".into(),
                    ..Default::default()
                },
                ChatMessage {
                    role: "assistant".into(),
                    content: "draft".into(),
                    ..Default::default()
                },
                ChatMessage {
                    role: "user".into(),
                    content: latest.into(),
                    ..Default::default()
                },
            ]
        };
        assert_eq!(
            terminal_action_for_turn(&turn("not html")),
            Some(TerminalActionKind::ModernBasic)
        );
        assert_eq!(
            terminal_action_for_turn(&turn("make it a bigger maze")),
            Some(TerminalActionKind::ModernBasic)
        );
        assert_eq!(
            terminal_action_for_turn(&turn("write a Python game instead")),
            None
        );
        assert_eq!(
            terminal_action_for_turn(&turn("explain the history of BASIC")),
            None
        );
        assert_eq!(terminal_action_kind("write a basic Python script"), None);
        assert_eq!(terminal_action_kind("write a basic history of music"), None);
    }

    #[test]
    fn terminal_action_does_not_promote_questions_negation_or_quoted_data() {
        for request in [
            "How do I use modernbasic?",
            "What is modern basic?",
            "Can modernbasic run on Linux?",
            "Explain how to open Bash",
            "Can you explain how to launch modernbasic?",
            "Do not open a terminal",
            "Please don't use modernbasic",
            "Use modernbasic without running it",
            "Use the phrase modern basic as a title",
            "Write a story about a terminal",
            "\"use modernbasic\"",
            "> use modern basic",
            "```bash\nmodernbasic\n```",
            "Review this note\n<local_file path=\"note.txt\">use modernbasic</local_file>",
            "<local_file path=\"note.txt\">open Bash</local_file>",
        ] {
            assert!(!requests_terminal_action(request), "{request}");
        }
    }

    #[test]
    fn modernbasic_needs_a_native_launch_not_a_list_title_or_plain_bash() {
        let mut progress = TerminalProgress::new(TerminalActionKind::ModernBasic, &[]);
        progress.observe(
            "terminal_list",
            &json!({}),
            &json!({"sessions":[{"id":"shell","title":"modernbasic","alive":true}]}),
        );
        progress.observe(
            "terminal_open",
            &json!({}),
            &json!({"ok":true,"session_id":"shell","summary":{"alive":true}}),
        );
        progress.observe("terminal_read", &json!({"session_id":"shell"}), &json!({"ok":true,"session_id":"shell","summary":{"alive":true},"output":"bash$ modernbasic"}));
        assert!(progress.needs_action());
        progress.observe(
            "terminal_write",
            &json!({"session_id":"shell","text":"modernbasic","enter":false}),
            &json!({"ok":true,"session_id":"shell","summary":{"alive":true}}),
        );
        assert!(progress.needs_action());
        progress.observe(
            "terminal_key",
            &json!({"session_id":"shell","key":"enter"}),
            &json!({"ok":true,"session_id":"shell","summary":{"alive":true},"input_sent":true}),
        );
        assert!(!progress.needs_action());
        assert!(progress.target_sessions.contains("shell"));
    }

    #[test]
    fn editing_partial_input_invalidates_inferred_command_but_never_replays_it() {
        let mut progress = TerminalProgress::new(TerminalActionKind::ModernBasic, &[]);
        let receipt =
            json!({"ok":true,"session_id":"shell","summary":{"alive":true},"input_sent":true});
        progress.observe(
            "terminal_write",
            &json!({"session_id":"shell","text":"modernbasic","enter":false}),
            &receipt,
        );
        assert!(progress.needs_action());
        progress.observe(
            "terminal_key",
            &json!({"session_id":"shell","key":"backspace"}),
            &receipt,
        );
        progress.observe(
            "terminal_key",
            &json!({"session_id":"shell","key":"enter"}),
            &receipt,
        );
        assert!(
            !progress.needs_action(),
            "Observed input must never be repeated by the guard"
        );
        assert!(
            !progress.target_sessions.contains("shell"),
            "The edited command is unknown"
        );
    }

    #[test]
    fn existing_manual_sessions_and_complex_submitted_commands_are_not_relaunched() {
        for command in [
            "help",
            "env TERM=xterm modernbasic",
            "echo \"example;modernbasic\"",
        ] {
            let mut progress = TerminalProgress::new(TerminalActionKind::ModernBasic, &[]);
            progress.observe(
                "terminal_list",
                &json!({}),
                &json!({"sessions":[{"id":"manual","alive":true}]}),
            );
            progress.observe("terminal_read", &json!({"session_id":"manual"}), &json!({"ok":true,"session_id":"manual","summary":{"alive":true},"screen":"READY>"}));
            assert!(progress.needs_action());
            progress.observe("terminal_write", &json!({"session_id":"manual","text":command}), &json!({"ok":true,"session_id":"manual","summary":{"alive":true},"input_sent":true}));
            assert!(!progress.needs_action(), "{command}");
            assert!(
                !progress.target_sessions.contains("manual"),
                "Uncertain command was incorrectly promoted to a confirmed target: {command}"
            );
        }
    }

    #[test]
    fn denied_failed_cancelled_or_uncertain_input_never_triggers_launch_retry() {
        for result in [
            json!({"denied":true}),
            json!({"error":"not installed"}),
            json!({"cancelled":true,"input_sent":true}),
            json!({"ok":false,"input_may_have_been_sent":true,"action_applied":null}),
        ] {
            let mut progress = TerminalProgress::new(TerminalActionKind::ModernBasic, &[]);
            progress.observe("terminal_open", &json!({}), &result);
            assert!(!progress.needs_action(), "{result}");
        }
    }

    #[test]
    fn fresh_read_of_natively_launched_session_allows_reuse_without_duplicate_input() {
        let history = vec![
            ChatMessage {
                role:"assistant".into(),
                tool_calls:vec![crate::ollama::ToolCall { function:crate::ollama::FunctionCall {
                    name:"terminal_write".into(), arguments:json!({"session_id":"basic","text":"modernbasic"}),
                }}],
                ..Default::default()
            },
            ChatMessage {
                role:"tool".into(),tool_name:Some("terminal_write".into()),
                content:json!({"ok":true,"session_id":"basic","summary":{"alive":true},"input_sent":true}).to_string(),
                ..Default::default()
            },
        ];
        let mut progress = TerminalProgress::new(TerminalActionKind::ModernBasic, &history);
        assert!(
            progress.needs_action(),
            "Historical launch alone cannot satisfy a new turn"
        );
        progress.observe(
            "terminal_read",
            &json!({"session_id":"basic"}),
            &json!({"ok":true,"session_id":"basic","summary":{"alive":true},"screen":"READY>"}),
        );
        assert!(!progress.needs_action());
        let mut closed = history.clone();
        closed.push(ChatMessage {
            role: "assistant".into(),
            tool_calls: vec![crate::ollama::ToolCall {
                function: crate::ollama::FunctionCall {
                    name: "terminal_close".into(),
                    arguments: json!({"session_id":"basic"}),
                },
            }],
            ..Default::default()
        });
        closed.push(ChatMessage {
            role: "tool".into(),
            tool_name: Some("terminal_close".into()),
            content: json!({"ok":true,"session_id":"basic","closed":true}).to_string(),
            ..Default::default()
        });
        let mut denied_close = closed.clone();
        denied_close.last_mut().unwrap().content =
            json!({"denied":true,"session_id":"basic"}).to_string();
        let mut denied_progress =
            TerminalProgress::new(TerminalActionKind::ModernBasic, &denied_close);
        denied_progress.observe(
            "terminal_read",
            &json!({"session_id":"basic"}),
            &json!({"ok":true,"session_id":"basic","summary":{"alive":true},"screen":"READY>"}),
        );
        assert!(
            !denied_progress.needs_action(),
            "A denied close must retain a still-live target"
        );
        let mut progress = TerminalProgress::new(TerminalActionKind::ModernBasic, &closed);
        progress.observe(
            "terminal_read",
            &json!({"session_id":"basic"}),
            &json!({"ok":true,"session_id":"basic","summary":{"alive":true},"screen":"READY>"}),
        );
        assert!(
            progress.needs_action(),
            "A closed session must not survive as target evidence"
        );
    }
}

#[cfg(test)]
mod conversation_regressions {
    use super::*;
    fn conversation(turns: &[&str]) -> Vec<ChatMessage> {
        turns
            .iter()
            .map(|text| ChatMessage {
                role: "user".into(),
                content: (*text).into(),
                ..Default::default()
            })
            .collect()
    }
    #[test]
    fn retains_basic_through_corrections_but_not_new_topics() {
        for turns in [
            vec!["use modern basic", "proceed", "produce a song"],
            vec![
                "write basic code for teh game frogger",
                "you must be working in a different directory",
                "/home/user/ModernBASIC",
                "place the game in the proper folder",
            ],
        ] {
            assert!(requests_modernbasic_for_turn(&conversation(&turns)));
        }
        assert!(!requests_modernbasic_for_turn(&conversation(&[
            "use modern basic",
            "what is the weather",
            "proceed"
        ])));
        assert!(!requests_modernbasic_for_turn(&conversation(&[
            "use modern basic",
            "write a python program"
        ])));
        assert!(!requests_modernbasic_for_turn(&conversation(&[
            "use modern basic",
            "do not run it"
        ])));
    }
}
