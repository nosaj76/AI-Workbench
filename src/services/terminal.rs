//! Native persistent PTY sessions. Shell state survives individual commands and
//! page changes. Closing a session closes its terminal and reaps its shell.
//! A terminal is intentionally a real user process, not a filesystem sandbox.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::io::{Read, Write};
use std::num::NonZeroU16;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const MAX_SESSIONS: usize = 12;
const MAX_INPUT: usize = 256 * 1024;
const MAX_TRANSCRIPT: usize = 1024 * 1024;
const SCROLLBACK_LINES: usize = 10_000;
const MAX_AGENT_OUTPUT: usize = 64 * 1024;
const MAX_AGENT_SCREEN: usize = 16 * 1024;
const MAX_AGENT_WAIT_MS: u64 = 10_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TerminalConfig {
    pub title: String,
    pub cwd: PathBuf,
    pub program: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub rows: u16,
    pub cols: u16,
    /// Run the session inside bubblewrap (terminals the model opens).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox: Option<crate::sandbox::SandboxPolicy>,
    /// The model may list, read and (with approval) type into this session.
    /// Only terminals the model opened, or ones the user shares, have it.
    pub ai_access: bool,
}

impl Default for TerminalConfig {
    fn default() -> Self {
        Self {
            title: "Terminal".into(),
            cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")),
            program: std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".into()),
            args: vec!["-i".into()],
            env: BTreeMap::new(),
            rows: 28,
            cols: 100,
            sandbox: None,
            ai_access: false,
        }
    }
}

impl TerminalConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.title.len() > 256 || self.program.is_empty() || self.program.contains('\0') {
            bail!("Terminal title or executable is invalid");
        }
        if self.args.len() > 128
            || self
                .args
                .iter()
                .any(|s| s.len() > 64_000 || s.contains('\0'))
        {
            bail!("Terminal arguments exceed their limits");
        }
        if self.env.len() > 128
            || self.env.iter().any(|(k, v)| {
                k.is_empty()
                    || k.contains(['=', '\0'])
                    || v.contains('\0')
                    || k.len() + v.len() > 64_000
            })
        {
            bail!("Terminal environment is invalid");
        }
        crate::integrations::validate_shell_argv(&self.program, &self.args, Some(&self.cwd))?;
        validate_size(self.rows, self.cols)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalSummary {
    pub id: String,
    pub title: String,
    pub initial_directory: PathBuf,
    pub pid: u32,
    pub alive: bool,
    pub exit_code: Option<i32>,
    /// The session runs inside the bubblewrap sandbox.
    #[serde(default)]
    pub sandboxed: bool,
    /// The model may use this session (it opened it, or the user shared it).
    #[serde(default)]
    pub ai_access: bool,
}

#[derive(Debug, Clone)]
pub struct TerminalCell {
    pub text: String,
    pub foreground: Option<[u8; 3]>,
    pub background: Option<[u8; 3]>,
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub inverse: bool,
    pub wide: bool,
    pub wide_continuation: bool,
}

#[derive(Debug, Clone)]
pub struct TerminalSnapshot {
    pub summary: TerminalSummary,
    pub text: String,
    pub cells: Vec<Vec<TerminalCell>>,
    pub rows: u16,
    pub cols: u16,
    pub cursor: (u16, u16),
    pub cursor_hidden: bool,
    pub alternate_screen: bool,
    pub scrollback: usize,
    pub bytes_received: u64,
    pub dropped_transcript_bytes: u64,
    pub error: Option<String>,
}

/// A bounded observation of a persistent terminal. Cursors count raw bytes in
/// the session's output stream, not characters and not positions on the screen.
/// `output` preserves ANSI/control sequences (escaped when serialized to JSON);
/// `screen` is the parsed, current screen even while the user views scrollback.
/// Silence is only an idle observation: it is never proof a command finished.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalRead {
    pub summary: TerminalSummary,
    pub output: String,
    /// True when this raw byte range contains invalid or partial UTF-8. Raw
    /// cursor offsets still advance by bytes; `screen` is the parsed view.
    pub output_lossy: bool,
    pub screen: String,
    pub screen_truncated: bool,
    pub start_cursor: u64,
    pub next_cursor: u64,
    pub bytes_received: u64,
    pub dropped_transcript_bytes: u64,
    pub lost_bytes: u64,
    pub truncated: bool,
    pub has_more: bool,
    pub cancelled: bool,
    pub timed_out: bool,
    pub output_closed: bool,
    pub rows: u16,
    pub cols: u16,
    pub cursor: (u16, u16),
    pub cursor_hidden: bool,
    pub alternate_screen: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub enum TerminalKey {
    Enter,
    Backspace,
    Tab,
    Escape,
    Up,
    Down,
    Left,
    Right,
    Home,
    End,
    Delete,
    PageUp,
    PageDown,
    Interrupt,
    Eof,
    Clear,
    Function(u8),
}

#[derive(Default)]
struct TerminalCallbacks {
    replies: Vec<u8>,
}

impl TerminalCallbacks {
    fn reply(&mut self, bytes: &[u8]) {
        if self.replies.len() + bytes.len() <= 4096 {
            self.replies.extend_from_slice(bytes);
        }
    }
}

impl vt100::Callbacks for TerminalCallbacks {
    fn unhandled_csi(
        &mut self,
        screen: &mut vt100::Screen,
        i1: Option<u8>,
        i2: Option<u8>,
        params: &[&[u16]],
        command: char,
    ) {
        if i2.is_some() {
            return;
        }
        let parameter = params.first().and_then(|p| p.first()).copied().unwrap_or(0);
        match (i1, command, parameter) {
            (None, 'n', 5) => self.reply(b"\x1b[0n"),
            (None, 'n', 6) => {
                let (row, col) = screen.cursor_position();
                self.reply(format!("\x1b[{};{}R", row + 1, col + 1).as_bytes());
            }
            (None, 'c', 0) => self.reply(b"\x1b[?1;2c"),
            (Some(b'>'), 'c', 0) => self.reply(b"\x1b[>0;0;0c"),
            _ => {}
        }
    }
    fn unhandled_escape(
        &mut self,
        _screen: &mut vt100::Screen,
        i1: Option<u8>,
        i2: Option<u8>,
        byte: u8,
    ) {
        if i1.is_none() && i2.is_none() && byte == b'Z' {
            self.reply(b"\x1b[?1;2c");
        }
    }
}

struct TerminalState {
    parser: vt100::Parser<TerminalCallbacks>,
    transcript: VecDeque<u8>,
    bytes_received: u64,
    dropped: u64,
    error: Option<String>,
}

struct TerminalSession {
    id: String,
    title: String,
    initial_directory: PathBuf,
    pid: u32,
    child: Mutex<Child>,
    writer: Mutex<Option<std::fs::File>>,
    state: Arc<Mutex<TerminalState>>,
    stop: Arc<AtomicBool>,
    output_closed: Arc<AtomicBool>,
    exit_code: Mutex<Option<i32>>,
    sandboxed: bool,
    ai_access: AtomicBool,
}

#[derive(Default)]
struct TerminalRegistry {
    sessions: BTreeMap<String, Arc<TerminalSession>>,
    opening: usize,
}

/// An opening consumes a slot without holding the registry lock across process
/// creation. Failed opens release their slot, including errors before `spawn`.
struct OpenReservation {
    registry: Arc<Mutex<TerminalRegistry>>,
}

impl OpenReservation {
    fn new(registry: &Arc<Mutex<TerminalRegistry>>) -> Result<Self> {
        let mut guard = registry
            .lock()
            .map_err(|_| anyhow::anyhow!("Terminal registry unavailable"))?;
        if guard.sessions.len() + guard.opening >= MAX_SESSIONS {
            bail!("Close a terminal before opening more than {MAX_SESSIONS} sessions");
        }
        guard.opening += 1;
        Ok(Self {
            registry: registry.clone(),
        })
    }
}

impl Drop for OpenReservation {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.registry.lock() {
            guard.opening = guard.opening.saturating_sub(1);
        }
    }
}

/// Clones refer to the same sessions. Dropping a temporary tool-engine clone
/// does not close terminals still owned by the desktop or another caller.
#[derive(Clone, Default)]
pub struct TerminalManager {
    registry: Arc<Mutex<TerminalRegistry>>,
}

impl TerminalManager {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(unix)]
    pub fn open(&self, config: TerminalConfig) -> Result<String> {
        use std::os::fd::{AsRawFd, FromRawFd};
        use std::os::unix::process::CommandExt;
        config.validate()?;
        let _reservation = OpenReservation::new(&self.registry)?;
        let cwd = config
            .cwd
            .canonicalize()
            .context("Terminal working directory is unavailable")?;
        if !cwd.is_dir() {
            bail!("Terminal working directory must be a directory");
        }
        let mut master_fd = -1;
        let mut slave_fd = -1;
        let dimensions = libc::winsize {
            ws_row: config.rows,
            ws_col: config.cols,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        // openpty initializes both owned descriptors or reports failure. They are
        // immediately placed in File owners; no borrowed descriptor is closed.
        if unsafe {
            libc::openpty(
                &mut master_fd,
                &mut slave_fd,
                std::ptr::null_mut(),
                std::ptr::null(),
                &dimensions,
            )
        } != 0
        {
            return Err(std::io::Error::last_os_error()).context("Allocate native pseudo-terminal");
        }
        let master = unsafe { std::fs::File::from_raw_fd(master_fd) };
        let slave = unsafe { std::fs::File::from_raw_fd(slave_fd) };
        for file in [&master, &slave] {
            let old = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFD) };
            if old < 0
                || unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFD, old | libc::FD_CLOEXEC) }
                    < 0
            {
                return Err(std::io::Error::last_os_error())
                    .context("Protect terminal descriptors across exec");
            }
        }
        crate::integrations::nonblocking(&master)?;
        let mut reader = master.try_clone()?;
        let sandboxed = config.sandbox.is_some();
        let (program, args) = match &config.sandbox {
            Some(policy) => {
                // The sandbox clears the environment; keep the terminal's own settings.
                let mut env = config.env.clone();
                env.insert("TERM".into(), "xterm-256color".into());
                env.insert("COLORTERM".into(), "truecolor".into());
                env.insert("COLUMNS".into(), config.cols.to_string());
                env.insert("LINES".into(), config.rows.to_string());
                policy.wrap(&config.program, &config.args, &cwd, &env, true)?
            }
            None => (config.program.clone(), config.args.clone()),
        };
        let mut command = Command::new(&program);
        command
            .args(&args)
            .current_dir(&cwd)
            .envs(&config.env)
            .env("TERM", "xterm-256color")
            .env("COLORTERM", "truecolor")
            .env("COLUMNS", config.cols.to_string())
            .env("LINES", config.rows.to_string())
            .stdin(Stdio::from(slave.try_clone()?))
            .stdout(Stdio::from(slave.try_clone()?))
            .stderr(Stdio::from(slave));
        // Only async-signal-safe syscalls run after fork. Child stdio has been
        // connected to the slave before this hook; fd 0 becomes controlling tty.
        unsafe {
            command.pre_exec(|| {
                if libc::setsid() < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::ioctl(0, libc::TIOCSCTTY as _, 0) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let child = command
            .spawn()
            .with_context(|| format!("Launch terminal executable {}", config.program))?;
        let pid = child.id();
        let state = Arc::new(Mutex::new(TerminalState {
            parser: vt100::Parser::new_with_callbacks(
                dimension(config.rows),
                dimension(config.cols),
                SCROLLBACK_LINES,
                TerminalCallbacks::default(),
            ),
            transcript: VecDeque::new(),
            bytes_received: 0,
            dropped: 0,
            error: None,
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let output_closed = Arc::new(AtomicBool::new(false));
        let id = uuid::Uuid::new_v4().to_string();
        let session = Arc::new(TerminalSession {
            id: id.clone(),
            title: config.title,
            initial_directory: cwd,
            pid,
            child: Mutex::new(child),
            writer: Mutex::new(Some(master)),
            state: state.clone(),
            stop: stop.clone(),
            output_closed: output_closed.clone(),
            exit_code: Mutex::new(None),
            sandboxed,
            ai_access: AtomicBool::new(config.ai_access),
        });
        let output_state = state.clone();
        let output_stop = stop.clone();
        std::thread::Builder::new()
            .name("workbench-pty-reader".into())
            .spawn(move || {
                let mut buf = [0u8; 16 * 1024];
                while !output_stop.load(Ordering::Relaxed) {
                    match reader.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            let Ok(mut state) = output_state.lock() else {
                                break;
                            };
                            state.parser.process(&buf[..n]);
                            state.bytes_received += n as u64;
                            state.transcript.extend(&buf[..n]);
                            if state.transcript.len() > MAX_TRANSCRIPT {
                                let excess = state.transcript.len() - MAX_TRANSCRIPT;
                                state.transcript.drain(..excess);
                                state.dropped += excess as u64;
                            }
                            let replies = std::mem::take(&mut state.parser.callbacks_mut().replies);
                            drop(state);
                            if !replies.is_empty() {
                                let _ = reader.write(&replies);
                            }
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            let mut pollfd = libc::pollfd {
                                fd: reader.as_raw_fd(),
                                events: libc::POLLIN,
                                revents: 0,
                            };
                            unsafe {
                                libc::poll(&mut pollfd, 1, 80);
                            }
                        }
                        // Linux signals the last slave closing as EIO.
                        Err(e) if e.raw_os_error() == Some(libc::EIO) => break,
                        Err(e) => {
                            if let Ok(mut state) = output_state.lock() {
                                state.error = Some(e.to_string());
                            }
                            break;
                        }
                    }
                }
                output_closed.store(true, Ordering::Release);
            })
            .context("Start terminal output reader")?;
        self.registry
            .lock()
            .map_err(|_| anyhow::anyhow!("Terminal registry unavailable"))?
            .sessions
            .insert(id.clone(), session);
        Ok(id)
    }

    #[cfg(not(unix))]
    pub fn open(&self, _config: TerminalConfig) -> Result<String> {
        bail!("Native persistent terminals are implemented for Unix desktops in this release")
    }

    pub fn list(&self) -> Vec<TerminalSummary> {
        let sessions: Vec<_> = match self.registry.lock() {
            Ok(registry) => registry.sessions.values().cloned().collect(),
            Err(_) => return Vec::new(),
        };
        sessions.iter().filter_map(|s| s.summary().ok()).collect()
    }

    pub fn summary(&self, id: &str) -> Result<TerminalSummary> {
        self.session(id)?.summary()
    }

    /// Sessions the model may use: the ones it opened and the ones the user shared.
    pub fn ai_list(&self) -> Vec<TerminalSummary> {
        self.list().into_iter().filter(|s| s.ai_access).collect()
    }

    /// The summary of a session the model may use; any other session is refused
    /// with the same message as an unknown id, so its existence is not revealed.
    pub fn ai_summary(&self, id: &str) -> Result<TerminalSummary> {
        match self.summary(id) {
            Ok(summary) if summary.ai_access => Ok(summary),
            _ => bail!("No terminal session {id:?} is available to the AI. Terminals the user opens are private unless shared; use terminal_list, or terminal_open to start one."),
        }
    }

    /// Share a user-opened session with the model, or make it private again.
    pub fn set_ai_access(&self, id: &str, allowed: bool) -> Result<()> {
        self.session(id)?
            .ai_access
            .store(allowed, Ordering::Relaxed);
        Ok(())
    }

    pub fn bytes_received(&self, id: &str) -> Result<u64> {
        let session = self.session(id)?;
        let state = session
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Terminal state unavailable"))?;
        Ok(state.bytes_received)
    }

    pub fn snapshot(&self, id: &str) -> Result<TerminalSnapshot> {
        let session = self.session(id)?;
        let summary = session.summary()?;
        let state = session
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Terminal state unavailable"))?;
        let screen = state.parser.screen();
        let (rows, cols) = screen.size();
        let (rows, cols) = (rows.get(), cols.get());
        let mut cells = Vec::with_capacity(rows as usize);
        for row in 0..rows {
            let mut line = Vec::with_capacity(cols as usize);
            for col in 0..cols {
                if let Some(cell) = screen.cell(row, col) {
                    line.push(TerminalCell {
                        text: cell.contents().to_string(),
                        foreground: color(cell.fgcolor()),
                        background: color(cell.bgcolor()),
                        bold: cell.bold(),
                        italic: cell.italic(),
                        underline: cell.underline(),
                        inverse: cell.inverse(),
                        wide: cell.is_wide(),
                        wide_continuation: cell.is_wide_continuation(),
                    });
                }
            }
            cells.push(line);
        }
        Ok(TerminalSnapshot {
            summary,
            text: screen.contents(),
            cells,
            rows,
            cols,
            cursor: screen.cursor_position(),
            cursor_hidden: screen.hide_cursor(),
            alternate_screen: screen.alternate_screen(),
            scrollback: screen.scrollback(),
            bytes_received: state.bytes_received,
            dropped_transcript_bytes: state.dropped,
            error: state.error.clone(),
        })
    }

    /// Read output after an absolute byte cursor. `None` starts at the beginning
    /// of the stream; `lost_bytes` reports anything already evicted from the
    /// bounded transcript. Pass `next_cursor` to continue without replaying old
    /// text. Waiting never holds the registry, parser, writer, or child lock.
    pub fn read_output(
        &self,
        id: &str,
        after: Option<u64>,
        max_bytes: usize,
        wait_ms: u64,
        cancel: Option<&AtomicBool>,
    ) -> Result<TerminalRead> {
        if !(1..=MAX_AGENT_OUTPUT).contains(&max_bytes) {
            bail!("Terminal output limit must be 1–{MAX_AGENT_OUTPUT} bytes");
        }
        if wait_ms > MAX_AGENT_WAIT_MS {
            bail!("Terminal wait must be 0–{MAX_AGENT_WAIT_MS} milliseconds");
        }
        let session = self.session(id)?;
        let after = after.unwrap_or(0);
        let deadline = Instant::now() + Duration::from_millis(wait_ms);
        let mut last_change = Instant::now();
        let mut last_received = {
            let state = session
                .state
                .lock()
                .map_err(|_| anyhow::anyhow!("Terminal state unavailable"))?;
            if after > state.bytes_received {
                bail!("Terminal cursor {after} is ahead of output {}; use a cursor returned by this session", state.bytes_received);
            }
            state.bytes_received
        };
        let mut cancelled = false;
        let mut timed_out = false;
        loop {
            if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                cancelled = true;
                break;
            }
            let (received, dropped, error) = {
                let state = session
                    .state
                    .lock()
                    .map_err(|_| anyhow::anyhow!("Terminal state unavailable"))?;
                (state.bytes_received, state.dropped, state.error.is_some())
            };
            let now = Instant::now();
            if received != last_received {
                last_received = received;
                last_change = now;
            }
            // Return promptly when a full page is ready, the stream has ended,
            // or output has been quiet briefly. A live shell is not a completed
            // foreground command, so no command-completion status is invented.
            if received.saturating_sub(after.max(dropped)) >= max_bytes as u64
                || session.output_closed.load(Ordering::Acquire)
                || error
                || (received > after
                    && now.duration_since(last_change) >= Duration::from_millis(80))
            {
                break;
            }
            if now >= deadline {
                timed_out = wait_ms > 0;
                break;
            }
            std::thread::sleep((deadline - now).min(Duration::from_millis(10)));
        }

        let summary = session.summary()?;
        let mut state = session
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Terminal state unavailable"))?;
        let start_cursor = after.max(state.dropped);
        let start = (start_cursor - state.dropped) as usize;
        let available = state.transcript.len().saturating_sub(start);
        let count = available.min(max_bytes);
        let mut bytes: Vec<u8> = state
            .transcript
            .iter()
            .skip(start)
            .take(count)
            .copied()
            .collect();
        // Ordinary Unicode text should not acquire replacement characters just
        // because a page boundary cuts its final code point. Keep the incomplete
        // suffix for the next page when a complete prefix can be returned. Raw
        // binary or caller-chosen mid-codepoint cursors remain explicitly lossy.
        if let Err(error) = std::str::from_utf8(&bytes) {
            if error.error_len().is_none() && error.valid_up_to() > 0 {
                bytes.truncate(error.valid_up_to());
            }
        }
        let output_lossy = std::str::from_utf8(&bytes).is_err();
        let next_cursor = start_cursor + bytes.len() as u64;
        let has_more = next_cursor < state.bytes_received;
        let lost_bytes = state.dropped.saturating_sub(after);

        // Reading the AI's live screen must neither show old user scrollback nor
        // change the user's chosen scrollback position.
        let scrollback = state.parser.screen().scrollback();
        state.parser.screen_mut().set_scrollback(0);
        let screen = state.parser.screen();
        let (rows, cols) = screen.size();
        let (rows, cols) = (rows.get(), cols.get());
        let cursor = screen.cursor_position();
        let cursor_hidden = screen.hide_cursor();
        let alternate_screen = screen.alternate_screen();
        let mut screen = screen.contents();
        let screen_truncated = screen.len() > MAX_AGENT_SCREEN;
        if screen_truncated {
            // Keep the bottom of a very large screen, where interactive prompts
            // usually appear, and make truncation explicit to the caller.
            let mut boundary = screen.len() - MAX_AGENT_SCREEN;
            while !screen.is_char_boundary(boundary) {
                boundary += 1;
            }
            screen = screen.split_off(boundary);
        }
        state.parser.screen_mut().set_scrollback(scrollback);

        Ok(TerminalRead {
            summary,
            output: String::from_utf8_lossy(&bytes).into_owned(),
            output_lossy,
            screen,
            screen_truncated,
            start_cursor,
            next_cursor,
            bytes_received: state.bytes_received,
            dropped_transcript_bytes: state.dropped,
            lost_bytes,
            truncated: lost_bytes > 0 || has_more,
            has_more,
            cancelled,
            timed_out,
            output_closed: session.output_closed.load(Ordering::Acquire),
            rows,
            cols,
            cursor,
            cursor_hidden,
            alternate_screen,
            error: state.error.clone(),
        })
    }

    /// Input is written to the existing PTY. No fresh shell or implicit command
    /// wrapper is created. A newline/Enter is an explicit caller decision.
    pub fn input(&self, id: &str, bytes: &[u8]) -> Result<()> {
        if bytes.len() > MAX_INPUT {
            bail!("Terminal input exceeds 256 KiB");
        }
        let session = self.session(id)?;
        if !session.summary()?.alive {
            bail!("Terminal shell has exited");
        }
        let mut guard = session
            .writer
            .lock()
            .map_err(|_| anyhow::anyhow!("Terminal writer unavailable"))?;
        let file = guard.as_mut().context("Terminal was closed")?;
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut rest = bytes;
        while !rest.is_empty() {
            match file.write(rest) {
                Ok(0) => bail!("Terminal stopped accepting input"),
                Ok(n) => rest = &rest[n..],
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e)
                    if e.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < deadline =>
                {
                    std::thread::sleep(Duration::from_millis(5))
                }
                Err(e) => return Err(e).context("Write to terminal"),
            }
        }
        Ok(())
    }

    pub fn paste(&self, id: &str, text: &str) -> Result<()> {
        if text.len() > MAX_INPUT - 16 {
            bail!("Terminal paste exceeds 256 KiB");
        }
        let bracketed = self
            .session(id)?
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Terminal state unavailable"))?
            .parser
            .screen()
            .bracketed_paste();
        if bracketed {
            self.input(
                id,
                format!("\x1b[200~{}\x1b[201~", text.replace("\x1b[201~", "")).as_bytes(),
            )
        } else {
            self.input(id, text.as_bytes())
        }
    }

    pub fn key(&self, id: &str, key: TerminalKey) -> Result<()> {
        let app_cursor = self
            .session(id)?
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Terminal state unavailable"))?
            .parser
            .screen()
            .application_cursor();
        let bytes: Vec<u8> = match key {
            TerminalKey::Enter => b"\r".to_vec(),
            TerminalKey::Backspace => vec![127],
            TerminalKey::Tab => vec![9],
            TerminalKey::Escape => vec![27],
            TerminalKey::Up
            | TerminalKey::Down
            | TerminalKey::Right
            | TerminalKey::Left
            | TerminalKey::Home
            | TerminalKey::End => {
                let suffix = match key {
                    TerminalKey::Up => 'A',
                    TerminalKey::Down => 'B',
                    TerminalKey::Right => 'C',
                    TerminalKey::Left => 'D',
                    TerminalKey::Home => 'H',
                    _ => 'F',
                };
                format!("\x1b{}{suffix}", if app_cursor { "O" } else { "[" }).into_bytes()
            }
            TerminalKey::Delete => b"\x1b[3~".to_vec(),
            TerminalKey::PageUp => b"\x1b[5~".to_vec(),
            TerminalKey::PageDown => b"\x1b[6~".to_vec(),
            TerminalKey::Interrupt => vec![3],
            TerminalKey::Eof => vec![4],
            TerminalKey::Clear => vec![12],
            TerminalKey::Function(n) => match n {
                1..=4 => format!("\x1bO{}", (b'P' + n - 1) as char).into_bytes(),
                5..=12 => format!(
                    "\x1b[{}~",
                    [15, 17, 18, 19, 20, 21, 23, 24][(n - 5) as usize]
                )
                .into_bytes(),
                _ => bail!("Only F1 through F12 are supported"),
            },
        };
        self.input(id, &bytes)
    }

    pub fn scrollback(&self, id: &str, rows: usize) -> Result<()> {
        self.session(id)?
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Terminal state unavailable"))?
            .parser
            .screen_mut()
            .set_scrollback(rows.min(SCROLLBACK_LINES));
        Ok(())
    }

    pub fn transcript(&self, id: &str) -> Result<Vec<u8>> {
        Ok(self
            .session(id)?
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Terminal state unavailable"))?
            .transcript
            .iter()
            .copied()
            .collect())
    }

    #[cfg(unix)]
    pub fn resize(&self, id: &str, rows: u16, cols: u16) -> Result<()> {
        use std::os::fd::AsRawFd;
        validate_size(rows, cols)?;
        let session = self.session(id)?;
        let dimensions = libc::winsize {
            ws_row: rows,
            ws_col: cols,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        let writer = session
            .writer
            .lock()
            .map_err(|_| anyhow::anyhow!("Terminal writer unavailable"))?;
        let fd = writer.as_ref().context("Terminal was closed")?.as_raw_fd();
        if unsafe { libc::ioctl(fd, libc::TIOCSWINSZ as _, &dimensions) } < 0 {
            return Err(std::io::Error::last_os_error()).context("Resize terminal");
        }
        session
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Terminal state unavailable"))?
            .parser
            .set_size(dimension(rows), dimension(cols));
        Ok(())
    }

    #[cfg(not(unix))]
    pub fn resize(&self, _id: &str, _rows: u16, _cols: u16) -> Result<()> {
        bail!("Native PTY resizing is unavailable")
    }

    pub fn close(&self, id: &str) -> Result<()> {
        let session = self
            .registry
            .lock()
            .map_err(|_| anyhow::anyhow!("Terminal registry unavailable"))?
            .sessions
            .remove(id)
            .context("Unknown terminal session")?;
        session.terminate();
        Ok(())
    }

    pub fn close_all(&self) {
        let sessions = match self.registry.lock() {
            Ok(mut registry) => std::mem::take(&mut registry.sessions),
            Err(_) => return,
        };
        for session in sessions.values() {
            session.terminate();
        }
    }

    fn session(&self, id: &str) -> Result<Arc<TerminalSession>> {
        self.registry
            .lock()
            .map_err(|_| anyhow::anyhow!("Terminal registry unavailable"))?
            .sessions
            .get(id)
            .cloned()
            .context("Unknown terminal session")
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        self.terminate();
    }
}

impl TerminalSession {
    fn summary(&self) -> Result<TerminalSummary> {
        let mut child = self
            .child
            .lock()
            .map_err(|_| anyhow::anyhow!("Terminal process unavailable"))?;
        let status = child.try_wait()?;
        if let Some(status) = status {
            *self
                .exit_code
                .lock()
                .map_err(|_| anyhow::anyhow!("Terminal process unavailable"))? =
                Some(status.code().unwrap_or(-1));
        }
        Ok(TerminalSummary {
            id: self.id.clone(),
            title: self.title.clone(),
            initial_directory: self.initial_directory.clone(),
            pid: self.pid,
            alive: status.is_none() && !self.stop.load(Ordering::Relaxed),
            exit_code: *self
                .exit_code
                .lock()
                .map_err(|_| anyhow::anyhow!("Terminal process unavailable"))?,
            sandboxed: self.sandboxed,
            ai_access: self.ai_access.load(Ordering::Relaxed),
        })
    }
    fn terminate(&self) {
        if self.stop.swap(true, Ordering::Relaxed) {
            return;
        }
        if let Ok(mut writer) = self.writer.lock() {
            #[cfg(unix)]
            if let Some(file) = writer.as_ref() {
                use std::os::fd::AsRawFd;
                let mut foreground: libc::pid_t = 0;
                if unsafe { libc::ioctl(file.as_raw_fd(), libc::TIOCGPGRP as _, &mut foreground) }
                    == 0
                    && foreground > 1
                    && foreground != self.pid as libc::pid_t
                {
                    // Close is destructive by contract. Stop the *currently*
                    // attached foreground job while this PTY still identifies
                    // its group, before closing the master or hanging up the
                    // shell. HUP alone is insufficient for a program that
                    // ignores it. Do not save this group id for a later kill:
                    // it could have exited and been reused during a grace wait.
                    unsafe {
                        libc::kill(-foreground, libc::SIGKILL);
                    }
                }
            }
            writer.take();
        }
        if let Ok(mut child) = self.child.lock() {
            // A previously reaped shell PID may have been reused by the OS.
            // The live PTY foreground group was handled above; never signal a
            // stale shell group after the child has already exited.
            if child.try_wait().ok().flatten().is_some() {
                return;
            }
            #[cfg(unix)]
            unsafe {
                libc::kill(-(self.pid as i32), libc::SIGHUP);
            }
            let deadline = Instant::now() + Duration::from_millis(400);
            while Instant::now() < deadline {
                if child.try_wait().ok().flatten().is_some() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            crate::integrations::kill_process_group(&mut child);
            let _ = child.wait();
        }
    }
}

fn validate_size(rows: u16, cols: u16) -> Result<()> {
    if !(2..=300).contains(&rows) || !(2..=500).contains(&cols) {
        bail!("Terminal dimensions must be 2–300 rows and 2–500 columns");
    }
    Ok(())
}

fn dimension(value: u16) -> NonZeroU16 {
    // Every public size entry point validates dimensions before reaching here.
    NonZeroU16::new(value).expect("validated terminal dimension")
}

fn color(color: vt100::Color) -> Option<[u8; 3]> {
    match color {
        vt100::Color::Default => None,
        vt100::Color::Rgb(r, g, b) => Some([r, g, b]),
        vt100::Color::Idx(i) => {
            const ANSI: [[u8; 3]; 16] = [
                [0, 0, 0],
                [205, 49, 49],
                [13, 188, 121],
                [229, 229, 16],
                [36, 114, 200],
                [188, 63, 188],
                [17, 168, 205],
                [229, 229, 229],
                [102, 102, 102],
                [241, 76, 76],
                [35, 209, 139],
                [245, 245, 67],
                [59, 142, 234],
                [214, 112, 214],
                [41, 184, 219],
                [255, 255, 255],
            ];
            if i < 16 {
                Some(ANSI[i as usize])
            } else if i < 232 {
                let n = i - 16;
                let v = |n: u8| if n == 0 { 0 } else { 55 + n * 40 };
                Some([v(n / 36), v(n / 6 % 6), v(n % 6)])
            } else {
                let v = 8 + (i - 232) * 10;
                Some([v, v, v])
            }
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    fn wait_text(manager: &TerminalManager, id: &str, needle: &str) -> String {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let text = String::from_utf8_lossy(&manager.transcript(id).unwrap()).into_owned();
            if text.contains(needle) {
                return text;
            }
            assert!(Instant::now() < deadline, "Missing {needle:?} in {text:?}");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    #[test]
    fn interactive_shell_preserves_environment_directory_and_tty_size() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("child")).unwrap();
        let manager = TerminalManager::new();
        let id = manager
            .open(TerminalConfig {
                program: "/bin/sh".into(),
                cwd: dir.path().into(),
                ..Default::default()
            })
            .unwrap();
        manager.input(&id,b"stty -echo; export WORKBENCH_PTY_VALUE=retained; cd child; printf 'FIRST_READY\\n'\r").unwrap();
        wait_text(&manager, &id, "FIRST_READY\r\n");
        manager.resize(&id, 37, 91).unwrap();
        manager.input(&id,b"printf 'VALUE=%s DIR=%s\\n' \"$WORKBENCH_PTY_VALUE\" \"${PWD##*/}\"; stty size; printf 'SECOND_DONE\\n'\r").unwrap();
        let text = wait_text(&manager, &id, "SECOND_DONE\r\n");
        assert!(text.contains("VALUE=retained DIR=child"));
        assert!(text.contains("37 91"));
        assert_eq!(
            (
                manager.snapshot(&id).unwrap().rows,
                manager.snapshot(&id).unwrap().cols
            ),
            (37, 91)
        );
        manager.close(&id).unwrap();
        assert!(manager.list().is_empty());
    }

    fn bash(manager: &TerminalManager, cwd: &std::path::Path) -> String {
        let id = manager
            .open(TerminalConfig {
                program: "/bin/bash".into(),
                args: vec!["--noprofile".into(), "--norc".into(), "-i".into()],
                cwd: cwd.into(),
                ..Default::default()
            })
            .unwrap();
        manager
            .input(&id, b"stty -echo; printf 'SHELL_READY\\n'\r")
            .unwrap();
        wait_text(manager, &id, "SHELL_READY\r\n");
        id
    }

    #[test]
    fn shared_manager_survives_tool_clone_drop_and_shares_close() {
        let dir = tempfile::tempdir().unwrap();
        let manager = TerminalManager::new();
        let id = bash(&manager, dir.path());
        let tools = manager.clone();
        manager
            .input(
                &id,
                b"export SHARED_VALUE=retained; printf 'SHARED_READY\\n'\r",
            )
            .unwrap();
        wait_text(&tools, &id, "SHARED_READY\r\n");
        drop(manager);
        tools
            .input(&id, b"printf 'SHARED=%s\\n' \"$SHARED_VALUE\"\r")
            .unwrap();
        assert!(wait_text(&tools, &id, "SHARED=retained\r\n").contains("SHARED=retained"));
        let desktop = tools.clone();
        desktop.close(&id).unwrap();
        assert!(tools.summary(&id).is_err());
        assert!(tools.list().is_empty());
    }

    #[test]
    fn bash_can_launch_an_interactive_basic_fixture_then_return_to_shell() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        // This is a deliberately small TTY/REPL fixture, not ModernBASIC. It
        // proves that later tool calls reach the foreground child in the same
        // terminal, and that a running program can itself request input.
        let fixture = dir.path().join("modernbasic");
        std::fs::write(
            &fixture,
            r##"#!/bin/bash
[[ -t 0 && -t 1 ]] || exit 42
program=()
printf 'BASIC_FIXTURE_READY> '
while IFS= read -r line; do
    case "$line" in
        help) printf 'FIXTURE_HELP: numbered lines; LIST; RUN; BYE\n' ;;
        LIST) printf '%s\n' "${program[@]}" ;;
        RUN)
            [[ "${program[0]}" == '10 INPUT A$' && "${program[1]}" == '20 PRINT A$' ]] || exit 43
            printf 'FIXTURE_INPUT? '
            IFS= read -r answer || exit 44
            printf 'FIXTURE_RESULT=%s\n' "$answer"
            ;;
        BYE) printf 'FIXTURE_EXIT\n'; exit 0 ;;
        [0-9]*) program+=("$line") ;;
        *) printf 'FIXTURE_UNKNOWN=%s\n' "$line" ;;
    esac
    printf 'BASIC_FIXTURE_READY> '
done
"##,
        )
        .unwrap();
        std::fs::set_permissions(&fixture, std::fs::Permissions::from_mode(0o755)).unwrap();
        let manager = TerminalManager::new();
        let mut env = BTreeMap::new();
        env.insert(
            "PATH".into(),
            format!("{}:/usr/bin:/bin", dir.path().display()),
        );
        let id = manager
            .open(TerminalConfig {
                program: "/bin/bash".into(),
                args: vec!["--noprofile".into(), "--norc".into(), "-i".into()],
                cwd: dir.path().into(),
                env,
                ..Default::default()
            })
            .unwrap();
        manager.input(&id, b"stty -echo; modernbasic\r").unwrap();
        wait_text(&manager, &id, "BASIC_FIXTURE_READY>");
        manager.input(&id, b"help\r").unwrap();
        let help = wait_text(&manager, &id, "FIXTURE_HELP:");
        assert!(help.contains("numbered lines; LIST; RUN; BYE"));
        manager
            .input(&id, b"10 INPUT A$\r20 PRINT A$\rLIST\rRUN\r")
            .unwrap();
        wait_text(&manager, &id, "FIXTURE_INPUT?");
        manager.input(&id, b"typed into BASIC\r").unwrap();
        wait_text(&manager, &id, "FIXTURE_RESULT=typed into BASIC\r\n");
        manager.input(&id, b"BYE\r").unwrap();
        wait_text(&manager, &id, "FIXTURE_EXIT\r\n");
        manager.input(&id, b"printf 'BACK_IN_BASH\\n'\r").unwrap();
        wait_text(&manager, &id, "BACK_IN_BASH\r\n");
        assert!(manager.summary(&id).unwrap().alive);
        assert_eq!(manager.list().len(), 1);
    }

    #[test]
    fn incremental_reads_page_output_report_eviction_and_keep_scrollback() {
        let dir = tempfile::tempdir().unwrap();
        let manager = TerminalManager::new();
        let id = manager
            .open(TerminalConfig {
                program: "/bin/sh".into(),
                args: vec!["-c".into(), "printf 'INITIAL\\n'".into()],
                cwd: dir.path().into(),
                rows: 4,
                cols: 20,
                ..Default::default()
            })
            .unwrap();
        assert!(
            manager
                .read_output(&id, None, 1024, 1000, None)
                .unwrap()
                .output_closed
        );
        let session = manager.session(&id).unwrap();
        // Inject output under the same parser lock as the reader so pagination
        // and eviction can be checked deterministically without shell timing.
        {
            let mut state = session.state.lock().unwrap();
            state.transcript = b"discarded kept output".iter().copied().collect();
            state.bytes_received = state.transcript.len() as u64;
            state.transcript.drain(..10);
            state.dropped = 10;
            state
                .parser
                .process(b"\r\nold\r\nold\r\nold\r\nold\r\nLIVE_SCREEN");
        }
        manager.scrollback(&id, 2).unwrap();
        let scrollback = manager.snapshot(&id).unwrap().scrollback;
        let first = manager.read_output(&id, Some(0), 5, 0, None).unwrap();
        assert_eq!(first.start_cursor, 10);
        assert_eq!(first.next_cursor, 15);
        assert_eq!(first.output, "kept ");
        assert_eq!(first.lost_bytes, 10);
        assert!(first.truncated && first.has_more);
        assert!(first.screen.contains("LIVE_SCREEN"));
        assert_eq!(manager.snapshot(&id).unwrap().scrollback, scrollback);
        let second = manager
            .read_output(&id, Some(first.next_cursor), 64, 0, None)
            .unwrap();
        assert_eq!(second.output, "output");
        assert_eq!(second.lost_bytes, 0);
        assert!(!second.has_more);
        assert!(manager
            .read_output(&id, Some(u64::MAX), 64, 0, None)
            .is_err());
    }

    #[test]
    fn waiting_read_is_cancellable_without_closing_the_terminal() {
        let dir = tempfile::tempdir().unwrap();
        let manager = TerminalManager::new();
        let id = bash(&manager, dir.path());
        // Drain startup and prompt output before starting a bounded wait.
        let cursor = manager
            .read_output(&id, None, 65_536, 300, None)
            .unwrap()
            .next_cursor;
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_manager = manager.clone();
        let worker_id = id.clone();
        let worker_cancel = cancel.clone();
        let start = Instant::now();
        let read = std::thread::spawn(move || {
            worker_manager
                .read_output(&worker_id, Some(cursor), 1024, 10_000, Some(&worker_cancel))
                .unwrap()
        });
        std::thread::sleep(Duration::from_millis(30));
        // A long read must not monopolize the registry: inspection is immediate.
        assert_eq!(manager.list().len(), 1);
        cancel.store(true, Ordering::Relaxed);
        assert!(read.join().unwrap().cancelled);
        assert!(start.elapsed() < Duration::from_secs(2));
        assert!(manager.summary(&id).unwrap().alive);
        assert!(manager.read_output(&id, None, 65_537, 0, None).is_err());
        assert!(manager.read_output(&id, None, 1024, 10_001, None).is_err());
    }

    #[test]
    fn interrupt_reaches_foreground_child_and_eof_exits_shell() {
        use std::os::fd::AsRawFd;
        let dir = tempfile::tempdir().unwrap();
        let manager = TerminalManager::new();
        let id = bash(&manager, dir.path());
        let session = manager.session(&id).unwrap();
        manager.input(&id, b"sleep 30\r").unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            let mut foreground: libc::pid_t = 0;
            let result = {
                let writer = session.writer.lock().unwrap();
                unsafe {
                    libc::ioctl(
                        writer.as_ref().unwrap().as_raw_fd(),
                        libc::TIOCGPGRP as _,
                        &mut foreground,
                    )
                }
            };
            assert_eq!(result, 0);
            if foreground > 1 && foreground != session.pid as libc::pid_t {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "sleep did not become the foreground job"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        manager.key(&id, TerminalKey::Interrupt).unwrap();
        manager
            .input(&id, b"printf 'AFTER_INTERRUPT\\n'\r")
            .unwrap();
        wait_text(&manager, &id, "AFTER_INTERRUPT\r\n");
        assert!(manager.summary(&id).unwrap().alive);
        manager.key(&id, TerminalKey::Eof).unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        while manager.summary(&id).unwrap().alive {
            assert!(Instant::now() < deadline, "EOF did not exit the shell");
            std::thread::sleep(Duration::from_millis(10));
        }
        let last = manager.read_output(&id, None, 65_536, 500, None).unwrap();
        assert!(last.output_closed);
        assert!(!last.summary.alive);
        assert!(last.output.contains("AFTER_INTERRUPT"));
    }

    #[test]
    fn close_stops_foreground_job_that_ignores_hangup_and_termination() {
        use std::os::fd::AsRawFd;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("stubborn.sh"),
            "trap '' HUP TERM\nprintf 'STUBBORN_READY\\n'\nwhile :; do printf x >> heartbeat; sleep 0.02; done\n",
        )
        .unwrap();
        let manager = TerminalManager::new();
        let id = bash(&manager, dir.path());
        manager.input(&id, b"/bin/bash stubborn.sh\r").unwrap();
        wait_text(&manager, &id, "STUBBORN_READY\r\n");
        let session = manager.session(&id).unwrap();
        let mut foreground: libc::pid_t = 0;
        {
            let writer = session.writer.lock().unwrap();
            assert_eq!(
                unsafe {
                    libc::ioctl(
                        writer.as_ref().unwrap().as_raw_fd(),
                        libc::TIOCGPGRP as _,
                        &mut foreground,
                    )
                },
                0
            );
        }
        assert!(foreground > 1 && foreground != session.pid as libc::pid_t);
        let heartbeat = dir.path().join("heartbeat");
        let size = || {
            std::fs::metadata(&heartbeat)
                .map(|meta| meta.len())
                .unwrap_or(0)
        };
        let deadline = Instant::now() + Duration::from_secs(3);
        while size() == 0 {
            assert!(
                Instant::now() < deadline,
                "foreground heartbeat did not start"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        let started = Instant::now();
        manager.close(&id).unwrap();
        assert!(started.elapsed() < Duration::from_secs(2));
        // A dead child can briefly be a zombie while the OS reaps it; an
        // execution heartbeat distinguishes that from a surviving application.
        std::thread::sleep(Duration::from_millis(50));
        let stopped_size = size();
        std::thread::sleep(Duration::from_millis(150));
        let survived = size() > stopped_size;
        if survived {
            // Clean up the demonstrably still-running fixture before failing.
            unsafe { libc::kill(-foreground, libc::SIGKILL) };
        }
        assert!(
            !survived,
            "close left the HUP/TERM-ignoring foreground job executing"
        );
        assert!(session.child.lock().unwrap().try_wait().unwrap().is_some());
        assert!(manager.list().is_empty());
    }

    #[test]
    fn byte_pages_preserve_complete_unicode_and_flag_invalid_text() {
        let dir = tempfile::tempdir().unwrap();
        let manager = TerminalManager::new();
        let id = manager
            .open(TerminalConfig {
                program: "/bin/sh".into(),
                args: vec!["-c".into(), "exit 0".into()],
                cwd: dir.path().into(),
                ..Default::default()
            })
            .unwrap();
        assert!(
            manager
                .read_output(&id, None, 1024, 1000, None)
                .unwrap()
                .output_closed
        );
        let session = manager.session(&id).unwrap();
        {
            let mut state = session.state.lock().unwrap();
            state.transcript = "abc界def".as_bytes().iter().copied().collect();
            state.bytes_received = state.transcript.len() as u64;
            state.dropped = 0;
        }
        let first = manager.read_output(&id, None, 5, 0, None).unwrap();
        assert_eq!(first.output, "abc");
        assert_eq!(first.next_cursor, 3);
        assert!(!first.output_lossy);
        let second = manager
            .read_output(&id, Some(first.next_cursor), 64, 0, None)
            .unwrap();
        assert_eq!(second.output, "界def");
        assert!(!second.output_lossy);
        assert!(!second.has_more);
        assert!(
            manager
                .read_output(&id, Some(4), 64, 0, None)
                .unwrap()
                .output_lossy
        );
    }
    #[test]
    fn terminal_parser_handles_screen_rewrites_and_scrollback() {
        let mut parser = vt100::Parser::new(dimension(4), dimension(20), 100);
        parser.process(b"obsolete\r\x1b[2Kcurrent\r\n\x1b[31mred\x1b[0m");
        assert_eq!(parser.screen().contents(), "current\nred");
        assert_eq!(
            color(parser.screen().cell(1, 0).unwrap().fgcolor()),
            Some([205, 49, 49])
        );
        parser.process(b"\x1b[?1049hfull screen");
        assert!(parser.screen().alternate_screen());
        parser.process(b"\x1b[?1049l");
        assert!(!parser.screen().alternate_screen());
        assert!(validate_size(0, 80).is_err());
        let mut wide = vt100::Parser::new(dimension(2), dimension(4), 0);
        wide.process("ab界".as_bytes());
        wide.set_size(dimension(2), dimension(3));
        wide.process(b"\x1b[1;3H!");
        assert!(wide.screen().contents().contains('!'));
        let mut queries = vt100::Parser::new_with_callbacks(
            dimension(4),
            dimension(20),
            10,
            TerminalCallbacks::default(),
        );
        queries.process(b"\x1b[2;3H\x1b[6n\x1b[5n");
        assert_eq!(queries.callbacks().replies, b"\x1b[2;3R\x1b[0n");
    }
}
