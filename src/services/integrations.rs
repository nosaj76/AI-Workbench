//! Optional local executable and network adapters. No service is launched on startup.
//! Callers must obtain user approval before invoking an executable or network request.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const MAX_CAPTURE: usize = 2 * 1024 * 1024;
const MAX_HTTP_BYTES: usize = 3 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CommandSpec {
    pub program: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub cwd: Option<PathBuf>,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    #[serde(default = "default_output_limit")]
    pub max_output_bytes: usize,
    #[serde(default)]
    pub stdin_text: Option<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Run inside bubblewrap with this access instead of the user's full permissions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox: Option<crate::sandbox::SandboxPolicy>,
}

fn default_timeout() -> u64 {
    60
}
fn default_output_limit() -> usize {
    64 * 1024
}

impl Default for CommandSpec {
    fn default() -> Self {
        Self {
            program: String::new(),
            args: Vec::new(),
            cwd: None,
            timeout_secs: default_timeout(),
            max_output_bytes: default_output_limit(),
            stdin_text: None,
            env: BTreeMap::new(),
            sandbox: None,
        }
    }
}

/// Catch the common model error `bash ls ...` without rewriting approved argv.
/// Bash accepts shell scripts, not an executable name to launch through PATH.
pub(crate) fn validate_shell_argv(
    program: &str,
    args: &[String],
    cwd: Option<&Path>,
) -> Result<()> {
    if !matches!(
        Path::new(program).file_name().and_then(|s| s.to_str()),
        Some("bash" | "sh")
    ) {
        return Ok(());
    }
    let Some(script) = args.first().filter(|s| !s.starts_with('-')) else {
        return Ok(());
    };
    let candidate = cwd.unwrap_or_else(|| Path::new(".")).join(script);
    let target = if candidate.is_file() {
        Some(candidate)
    } else {
        std::env::var_os("PATH").and_then(|path| {
            std::env::split_paths(&path)
                .map(|dir| dir.join(script))
                .find(|p| p.is_file())
        })
    };
    if let Some(target) = target {
        let mut bytes = [0_u8; 256];
        let count = std::fs::File::open(target)
            .and_then(|mut f| f.read(&mut bytes))
            .unwrap_or(0);
        let head = String::from_utf8_lossy(&bytes[..count]);
        let first_line = head.lines().next().unwrap_or("");
        if bytes[..count].starts_with(b"\x7fELF")
            || (first_line.starts_with("#!")
                && !first_line
                    .split(|c: char| c.is_whitespace() || c == '/')
                    .any(|p| matches!(p, "bash" | "sh" | "dash")))
        {
            bail!("Bash would interpret {script:?} as a shell script, not launch it. Set program to the executable and args to its arguments, or open terminal_open with no args and type the command using terminal_write. No command was executed.");
        }
    }
    Ok(())
}

impl CommandSpec {
    pub fn validate(&self) -> Result<()> {
        validate_interpreter_argv(&self.program, &self.args, self.cwd.as_deref())?;
        validate_shell_argv(&self.program, &self.args, self.cwd.as_deref())?;
        if self.program.trim().is_empty() || self.program.contains('\0') {
            bail!("An executable name or absolute path is required");
        }
        if self.args.len() > 256
            || self
                .args
                .iter()
                .any(|a| a.contains('\0') || a.len() > 128 * 1024)
        {
            bail!("Invalid or excessive command arguments");
        }
        if self
            .stdin_text
            .as_ref()
            .is_some_and(|s| s.len() > 1024 * 1024)
        {
            bail!("Command standard input exceeds 1 MiB");
        }
        if self
            .env
            .iter()
            .any(|(k, v)| k.is_empty() || k.contains(['\0', '=']) || v.contains('\0'))
        {
            bail!("Invalid environment entry");
        }
        Ok(())
    }

    /// A presentation only: this string is never passed to a shell.
    pub fn display_command(&self) -> String {
        std::iter::once(&self.program)
            .chain(self.args.iter())
            .map(|v| serde_json::to_string(v).unwrap_or_else(|_| "\"?\"".to_owned()))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResult {
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
    pub cancelled: bool,
    pub truncated: bool,
    pub elapsed_ms: u64,
}

/// Uses a separate process group on Unix so timeout includes children.
pub(crate) fn configure_process_group(command: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let _ = command;
}

pub(crate) fn kill_process_group(child: &mut Child) {
    #[cfg(unix)]
    unsafe {
        // The child was deliberately created as leader of its own process group.
        libc::kill(-(child.id() as i32), libc::SIGKILL);
    }
    let _ = child.kill();
}

#[cfg(unix)]
pub(crate) fn nonblocking<T: std::os::fd::AsRawFd>(pipe: &T) -> Result<()> {
    let fd = pipe.as_raw_fd();
    let old = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if old < 0 || unsafe { libc::fcntl(fd, libc::F_SETFL, old | libc::O_NONBLOCK) } < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(())
}

/// Receives raw output bytes as a running command produces them, for live display.
/// Only bytes within the capture limit are forwarded.
pub type OutputSink = Arc<dyn Fn(&[u8]) + Send + Sync>;

fn capture<R: Read>(
    mut pipe: R,
    cap: usize,
    stop: Arc<AtomicBool>,
    sink: Option<OutputSink>,
) -> Result<(Vec<u8>, bool)> {
    let mut captured = Vec::new();
    let mut truncated = false;
    let mut buffer = [0u8; 8192];
    let mut stop_deadline = None;
    loop {
        if stop.load(Ordering::Relaxed) {
            let deadline =
                stop_deadline.get_or_insert_with(|| Instant::now() + Duration::from_millis(100));
            if Instant::now() >= *deadline {
                break;
            }
        }
        match pipe.read(&mut buffer) {
            Ok(0) => break,
            Ok(n) => {
                let available = cap.saturating_sub(captured.len());
                let kept = &buffer[..n.min(available)];
                if let (Some(sink), false) = (&sink, kept.is_empty()) {
                    sink(kept);
                }
                captured.extend_from_slice(kept);
                truncated |= n > available;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                thread::sleep(Duration::from_millis(5));
            }
            Err(e) => return Err(e.into()),
        }
    }
    Ok((captured, truncated))
}

/// Executes argv directly, with bounded capture and deadline. Without a
/// `sandbox` policy a command has the current user's ordinary OS permissions.
pub fn run_command(spec: &CommandSpec, default_cwd: &Path) -> Result<CommandResult> {
    run_command_cancellable(spec, default_cwd, None)
}

pub fn run_command_cancellable(
    spec: &CommandSpec,
    default_cwd: &Path,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<CommandResult> {
    run_command_streaming(spec, default_cwd, cancel, None)
}

/// Like [`run_command_cancellable`], additionally forwarding stdout and stderr
/// to `sink` while the command runs. The returned result is identical.
pub fn run_command_streaming(
    spec: &CommandSpec,
    default_cwd: &Path,
    cancel: Option<Arc<AtomicBool>>,
    sink: Option<OutputSink>,
) -> Result<CommandResult> {
    spec.validate()?;
    if cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
        bail!("Command was cancelled before launch");
    }
    let cwd = spec.cwd.as_deref().unwrap_or(default_cwd);
    if !cwd.is_dir() {
        bail!(
            "Command working directory is not a directory: {}",
            cwd.display()
        );
    }
    let cap = spec.max_output_bytes.clamp(1024, MAX_CAPTURE);
    let (program, args) = match &spec.sandbox {
        Some(policy) => policy.wrap(&spec.program, &spec.args, cwd, &spec.env, false)?,
        None => (spec.program.clone(), spec.args.clone()),
    };
    let mut command = Command::new(&program);
    command
        .args(&args)
        .current_dir(cwd)
        .envs(&spec.env)
        .stdin(if spec.stdin_text.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    configure_process_group(&mut command);
    let started = Instant::now();
    let mut child = command
        .spawn()
        .with_context(|| format!("Could not start {}", spec.program))?;
    let stdout = child.stdout.take().context("Missing stdout pipe")?;
    let stderr = child.stderr.take().context("Missing stderr pipe")?;
    #[cfg(unix)]
    {
        if let Err(e) = nonblocking(&stdout).and_then(|_| nonblocking(&stderr)) {
            kill_process_group(&mut child);
            let _ = child.wait();
            return Err(e);
        }
    }
    let stop = Arc::new(AtomicBool::new(false));
    let out_stop = stop.clone();
    let err_stop = stop.clone();
    let err_sink = sink.clone();
    let out = thread::spawn(move || capture(stdout, cap, out_stop, sink));
    let err = thread::spawn(move || capture(stderr, cap, err_stop, err_sink));
    let input_writer =
        if let (Some(mut pipe), Some(input)) = (child.stdin.take(), spec.stdin_text.clone()) {
            #[cfg(unix)]
            if let Err(e) = nonblocking(&pipe) {
                kill_process_group(&mut child);
                let _ = child.wait();
                stop.store(true, Ordering::Relaxed);
                let _ = out.join();
                let _ = err.join();
                return Err(e);
            }
            let input_stop = stop.clone();
            Some(thread::spawn(move || {
                let mut written = 0;
                while written < input.len() && !input_stop.load(Ordering::Relaxed) {
                    match pipe.write(&input.as_bytes()[written..]) {
                        Ok(0) => break,
                        Ok(n) => written += n,
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(5))
                        }
                        Err(_) => break,
                    }
                }
            }))
        } else {
            None
        };
    let timeout = Duration::from_secs(spec.timeout_secs.clamp(1, 3600));
    let mut timed_out = false;
    let mut cancelled = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None)
                if started.elapsed() >= timeout
                    || cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) =>
            {
                cancelled = cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed));
                timed_out = !cancelled;
                kill_process_group(&mut child);
                break child.wait();
            }
            Ok(None) => thread::sleep(Duration::from_millis(10)),
            Err(e) => {
                kill_process_group(&mut child);
                let _ = child.wait();
                break Err(e);
            }
        }
    };
    // No detached work is part of the command contract. Close inherited pipes too.
    kill_process_group(&mut child);
    stop.store(true, Ordering::Relaxed);
    if let Some(writer) = input_writer {
        let _ = writer.join();
    }
    let (stdout, out_truncated) = out
        .join()
        .map_err(|_| anyhow::anyhow!("stdout reader failed"))??;
    let (stderr, err_truncated) = err
        .join()
        .map_err(|_| anyhow::anyhow!("stderr reader failed"))??;
    let status = status.context("Could not wait for command")?;
    Ok(CommandResult {
        exit_code: status.code(),
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
        timed_out,
        cancelled,
        truncated: out_truncated || err_truncated,
        elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AdapterKind {
    SpeechToText,
    TextToSpeech,
    Ocr,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CommandAdapter {
    pub id: String,
    pub name: String,
    pub kind: AdapterKind,
    pub enabled: bool,
    pub program: String,
    pub args: Vec<String>,
    #[serde(default)]
    pub text_to_stdin: bool,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

impl CommandAdapter {
    /// Editable templates; the engine/model paths must exist on the user's computer.
    pub fn examples() -> Vec<Self> {
        vec![
            Self {
                id: "whisper".into(),
                name: "Whisper.cpp transcription".into(),
                kind: AdapterKind::SpeechToText,
                enabled: false,
                program: "whisper-cli".into(),
                args: vec![
                    "-m".into(),
                    "/path/to/ggml-model.bin".into(),
                    "-f".into(),
                    "{input}".into(),
                    "-nt".into(),
                ],
                text_to_stdin: false,
                timeout_secs: 600,
            },
            Self {
                id: "piper".into(),
                name: "Piper voice to WAV".into(),
                kind: AdapterKind::TextToSpeech,
                enabled: false,
                program: "piper".into(),
                args: vec![
                    "--model".into(),
                    "/path/to/voice.onnx".into(),
                    "--output_file".into(),
                    "{output}".into(),
                ],
                text_to_stdin: true,
                timeout_secs: 120,
            },
            Self {
                id: "espeak".into(),
                name: "eSpeak voice to WAV".into(),
                kind: AdapterKind::TextToSpeech,
                enabled: false,
                program: "espeak-ng".into(),
                args: vec!["-w".into(), "{output}".into(), "--stdin".into()],
                text_to_stdin: true,
                timeout_secs: 120,
            },
            Self {
                id: "ocr".into(),
                name: "Tesseract OCR".into(),
                kind: AdapterKind::Ocr,
                enabled: false,
                program: "tesseract".into(),
                args: vec!["{input}".into(), "stdout".into()],
                text_to_stdin: false,
                timeout_secs: 120,
            },
        ]
    }

    pub fn command(
        &self,
        input: Option<&Path>,
        output: Option<&Path>,
        text: Option<&str>,
        cwd: &Path,
    ) -> Result<CommandSpec> {
        if !self.enabled {
            bail!("Enable and configure the {} adapter first", self.name);
        }
        for (provided, placeholder) in [
            (input.is_some(), "{input}"),
            (output.is_some(), "{output}"),
            (text.is_some() && !self.text_to_stdin, "{text}"),
        ] {
            if provided && !self.args.iter().any(|arg| arg.contains(placeholder)) {
                bail!(
                    "{} does not use {}. Configure its argument template before supplying this value.",
                    self.name,
                    placeholder
                );
            }
        }
        let mut args = Vec::new();
        for template in &self.args {
            let mut arg = template.clone();
            for (key, value) in [
                ("{input}", input.map(|p| p.to_string_lossy().into_owned())),
                ("{output}", output.map(|p| p.to_string_lossy().into_owned())),
                ("{text}", text.map(str::to_owned)),
            ] {
                if arg.contains(key) {
                    let value = value
                        .as_ref()
                        .with_context(|| format!("{} requires {}", self.name, key))?;
                    arg = arg.replace(key, value);
                }
            }
            args.push(arg);
        }
        let spec = CommandSpec {
            program: self.program.clone(),
            args,
            cwd: Some(cwd.to_path_buf()),
            timeout_secs: self.timeout_secs,
            stdin_text: if self.text_to_stdin {
                Some(text.context("The adapter requires text input")?.to_owned())
            } else {
                None
            },
            ..Default::default()
        };
        spec.validate()?;
        Ok(spec)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebPage {
    pub url: String,
    pub title: String,
    pub content: String,
    pub truncated: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    /// Image searches: the full image, and a smaller preview when the provider has one.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub image_url: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub thumbnail: String,
}

/// A SearXNG answer: the results and the related searches it suggests.
#[derive(Debug, Clone, Default)]
pub struct SearxngResponse {
    pub results: Vec<SearchResult>,
    pub suggestions: Vec<String>,
}

/// Optional search refinements every provider is asked to honour. Each search
/// result reports which ones the answering provider applied, so a model never
/// assumes a filter was used when the provider could not apply it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchOptions {
    /// "day", "week", "month" or "year": only pages from that recent period.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_range: Option<String>,
    /// BCP 47 language, optionally with a region: "en", "de", "en-US", "pt-BR".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// "off", "moderate" or "strict".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safe_search: Option<String>,
}

pub const TIME_RANGES: [&str; 4] = ["day", "week", "month", "year"];
pub const SAFE_SEARCH_LEVELS: [&str; 3] = ["off", "moderate", "strict"];

impl SearchOptions {
    /// Reads and validates `time_range`, `language` and `safe_search` from tool arguments.
    pub fn from_args(args: &Value) -> Result<Self> {
        let text = |key: &str| -> Result<Option<String>> {
            match args.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::String(s)) if s.trim().is_empty() => Ok(None),
                Some(Value::String(s)) => Ok(Some(s.trim().to_owned())),
                Some(_) => bail!("{key} must be a string"),
            }
        };
        let time_range = text("time_range")?.map(|v| v.to_ascii_lowercase());
        if let Some(value) = &time_range {
            if !TIME_RANGES.contains(&value.as_str()) {
                bail!("time_range must be one of {}", TIME_RANGES.join(", "));
            }
        }
        let safe_search = text("safe_search")?.map(|v| v.to_ascii_lowercase());
        if let Some(value) = &safe_search {
            if !SAFE_SEARCH_LEVELS.contains(&value.as_str()) {
                bail!("safe_search must be one of {}", SAFE_SEARCH_LEVELS.join(", "));
            }
        }
        let language = text("language")?
            .map(|value| normalize_language(&value))
            .transpose()?;
        Ok(Self { time_range, language, safe_search })
    }

    pub fn is_empty(&self) -> bool {
        self.time_range.is_none() && self.language.is_none() && self.safe_search.is_none()
    }

    /// The primary language subtag: "en" for "en-US".
    pub fn language_code(&self) -> Option<&str> {
        self.language.as_deref().map(|l| l.split('-').next().unwrap_or(l))
    }

    /// The region subtag, upper case: "US" for "en-US".
    pub fn region(&self) -> Option<&str> {
        self.language.as_deref().and_then(|l| l.split_once('-')).map(|(_, r)| r)
    }

    /// The requested options, plus which of them a provider applied or could not apply.
    pub fn report(&self, applied: &[&str]) -> Value {
        if self.is_empty() {
            return Value::Null;
        }
        let requested: Vec<&str> = [
            ("time_range", self.time_range.is_some()),
            ("language", self.language.is_some()),
            ("safe_search", self.safe_search.is_some()),
        ]
        .into_iter()
        .filter_map(|(name, set)| set.then_some(name))
        .collect();
        let used: Vec<&str> = requested.iter().copied().filter(|n| applied.contains(n)).collect();
        let not_applied: Vec<&str> = requested.iter().copied().filter(|n| !applied.contains(n)).collect();
        serde_json::json!({
            "requested": self,
            "applied": used,
            "not_applied": not_applied,
        })
    }

    /// The options SearXNG can apply: all of them.
    pub fn searxng_applied(&self) -> Vec<&'static str> {
        vec!["time_range", "language", "safe_search"]
    }

    /// DuckDuckGo applies a language only through a region ("en-US" → us-en).
    pub fn duckduckgo_applied(&self) -> Vec<&'static str> {
        let mut applied = vec!["time_range", "safe_search"];
        if self.region().is_some() {
            applied.push("language");
        }
        applied
    }
}

/// "EN_us" → "en-US". Accepts a 2–3 letter language with an optional 2-letter region.
fn normalize_language(value: &str) -> Result<String> {
    let value = value.replace('_', "-");
    let mut parts = value.split('-');
    let language = parts.next().unwrap_or_default().to_ascii_lowercase();
    let region = parts.next().map(str::to_ascii_uppercase);
    let valid_language = (2..=3).contains(&language.len()) && language.chars().all(|c| c.is_ascii_alphabetic());
    let valid_region = region
        .as_ref()
        .is_none_or(|r| r.len() == 2 && r.chars().all(|c| c.is_ascii_alphabetic()));
    if !valid_language || !valid_region || parts.next().is_some() {
        bail!("language must be a language code such as \"en\" or \"en-US\"");
    }
    Ok(match region {
        Some(region) => format!("{language}-{region}"),
        None => language,
    })
}

/// Whether something accepts TCP connections at the URL's host and port.
/// Used to skip a configured-but-stopped local service (such as SearXNG)
/// before asking the user to approve a request that could only fail.
pub fn endpoint_accepts_connections(value: &str, timeout: Duration) -> bool {
    use std::net::{TcpStream, ToSocketAddrs};
    let Ok(url) = url::Url::parse(value) else {
        return false;
    };
    let (Some(host), Some(port)) = (url.host_str(), url.port_or_known_default()) else {
        return false;
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    (host, port)
        .to_socket_addrs()
        .map(|addrs| {
            addrs
                .take(4)
                .any(|a| TcpStream::connect_timeout(&a, timeout).is_ok())
        })
        .unwrap_or(false)
}

pub fn validate_http_url(value: &str) -> Result<url::Url> {
    if value.len() > 4096 {
        bail!("URL exceeds 4096 bytes");
    }
    let url = url::Url::parse(value).context("Invalid URL")?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
    {
        bail!("Use an HTTP(S) URL without embedded credentials");
    }
    Ok(url)
}

const MAX_REDIRECTS: usize = 3;

/// True for loopback, private, link-local, CGNAT, unique-local and unspecified
/// addresses — destinations a public page must not be able to redirect into.
fn non_public_ip(ip: std::net::IpAddr) -> bool {
    use std::net::IpAddr;
    match ip {
        IpAddr::V4(v4) => {
            let [a, b, ..] = v4.octets();
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || (a == 100 && (b & 0xc0) == 64)
        }
        IpAddr::V6(v6) => {
            let first = v6.segments()[0];
            v6.is_loopback()
                || v6.is_unspecified()
                || (first & 0xfe00) == 0xfc00
                || (first & 0xffc0) == 0xfe80
                || v6
                    .to_ipv4_mapped()
                    .is_some_and(|v4| non_public_ip(v4.into()))
        }
    }
}

/// Whether a URL's host is (or resolves to) a non-public address. Unresolvable
/// names are treated as public; the connection attempt will fail on its own.
pub(crate) fn is_local_destination(url: &url::Url) -> bool {
    use std::net::ToSocketAddrs;
    match url.host() {
        None => true,
        Some(url::Host::Ipv4(ip)) => non_public_ip(ip.into()),
        Some(url::Host::Ipv6(ip)) => non_public_ip(ip.into()),
        Some(url::Host::Domain(domain)) => {
            let domain = domain.trim_end_matches('.').to_ascii_lowercase();
            if domain == "localhost" || domain.ends_with(".localhost") {
                return true;
            }
            let port = url.port_or_known_default().unwrap_or(80);
            (domain.as_str(), port)
                .to_socket_addrs()
                .is_ok_and(|mut addrs| addrs.any(|a| non_public_ip(a.ip())))
        }
    }
}

/// The user approves one URL. A redirect may stay local if the approved URL was
/// local (e.g. a LAN SearXNG), but a public page may not bounce the request to
/// loopback, LAN or metadata addresses the user never saw.
fn redirect_allowed(origin: &url::Url, next: &url::Url) -> bool {
    is_local_destination(origin) || !is_local_destination(next)
}

/// Resolves names for requests whose approved destination is public, and drops
/// every local or private address from the answer. The check happens on the very
/// addresses the connection uses, so a name that resolved publicly when it was
/// checked cannot be re-pointed at loopback, the LAN or a metadata service before
/// the connection (DNS rebinding), including on redirects.
struct PublicOnlyResolver;

impl reqwest::dns::Resolve for PublicOnlyResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let host = name.as_str().to_owned();
        Box::pin(async move {
            let found: Vec<std::net::SocketAddr> =
                tokio::net::lookup_host((host.as_str(), 0)).await?.collect();
            let public: Vec<std::net::SocketAddr> = found
                .iter()
                .copied()
                .filter(|address| !non_public_ip(address.ip()))
                .collect();
            if public.is_empty() {
                return Err(format!(
                    "{host} resolves only to local or private addresses; refused for a request that was approved as public"
                )
                .into());
            }
            Ok(Box::new(public.into_iter()) as reqwest::dns::Addrs)
        })
    }
}

/// A client for a request to `approved`. If the user approved a local address
/// (for example a LAN SearXNG), local destinations stay reachable; otherwise every
/// connection is limited to public addresses.
pub(crate) fn network_client_for(approved: &url::Url) -> Result<reqwest::blocking::Client> {
    build_network_client(is_local_destination(approved))
}

/// A client for public services: connections to local or private addresses are refused.
pub(crate) fn network_client() -> Result<reqwest::blocking::Client> {
    build_network_client(false)
}

fn build_network_client(allow_local: bool) -> Result<reqwest::blocking::Client> {
    let builder = reqwest::blocking::Client::builder();
    let builder = if allow_local {
        builder
    } else {
        builder.dns_resolver(std::sync::Arc::new(PublicOnlyResolver))
    };
    Ok(builder
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            if attempt.previous().len() > MAX_REDIRECTS {
                return attempt.error("too many redirects");
            }
            let allowed = attempt
                .previous()
                .first()
                .is_none_or(|origin| redirect_allowed(origin, attempt.url()));
            if allowed {
                attempt.follow()
            } else {
                let blocked = attempt.url().to_string();
                attempt.error(format!(
                    "Blocked redirect from a public page to a local or private address: {blocked}"
                ))
            }
        }))
        .user_agent(concat!(
            "Ollama-Workbench/",
            env!("CARGO_PKG_VERSION"),
            " (user-requested retrieval)"
        ))
        .build()?)
}

pub(crate) fn limited_body(response: reqwest::blocking::Response) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    response
        .take((MAX_HTTP_BYTES + 1) as u64)
        .read_to_end(&mut body)?;
    if body.len() > MAX_HTTP_BYTES {
        bail!("HTTP body exceeds 3 MiB; use a smaller document");
    }
    Ok(body)
}

/// Fetch an approved URL. `allow_local` is what the user approved: when false,
/// every connection (including the first, and after any DNS change since the
/// preview) is limited to public addresses.
pub fn fetch_url(value: &str, max_chars: usize, allow_local: bool) -> Result<WebPage> {
    let url = validate_http_url(value)?;
    // Literal IP addresses bypass the public-only resolver; check them here.
    if !allow_local && is_local_destination(&url) {
        bail!("{url} is a local or private address, but the request was approved as public. Nothing was sent.");
    }
    let response = build_network_client(allow_local)?
        .get(url)
        .send()?
        .error_for_status()?;
    let final_url = response.url().to_string();
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("text/plain")
        .to_lowercase();
    if !content_type.starts_with("text/")
        && !content_type.contains("json")
        && !content_type.contains("xml")
    {
        bail!("URL content type {content_type} is not a readable text document");
    }
    let body = limited_body(response)?;
    let raw = String::from_utf8_lossy(&body);
    let (title, content) = if content_type.contains("html") {
        html_text(&raw)
    } else {
        (String::new(), raw.into_owned())
    };
    let limit = max_chars.clamp(1000, 100_000);
    let truncated = content.chars().count() > limit;
    Ok(WebPage {
        url: final_url,
        title,
        content: content.chars().take(limit).collect(),
        truncated,
    })
}

fn html_text(raw: &str) -> (String, String) {
    let document = scraper::Html::parse_document(raw);
    let title_selector = scraper::Selector::parse("title").expect("constant selector");
    let title = document
        .select(&title_selector)
        .next()
        .map(|e| e.text().collect::<Vec<_>>().join(" "))
        .unwrap_or_default();
    // Selector lists match in DOCUMENT order, so "main, article, body" would
    // always return <body>. Try each content root in priority order instead
    // and skip one whose readable text is too small to be the real content.
    let extract = |root: scraper::ElementRef, drop_page_chrome: bool| {
        let mut pieces = Vec::new();
        for node in root.descendants() {
            if let Some(text) = node.value().as_text() {
                let excluded = node.ancestors().any(|p| {
                    p.value().as_element().is_some_and(|e| {
                        matches!(
                            e.name(),
                            "script"
                                | "style"
                                | "noscript"
                                | "svg"
                                | "template"
                                | "nav"
                                | "aside"
                                | "form"
                                | "button"
                                | "select"
                        ) || (drop_page_chrome && matches!(e.name(), "header" | "footer"))
                            || e.attr("hidden").is_some()
                            || e.attr("aria-hidden") == Some("true")
                            || e.attr("role").is_some_and(|role| {
                                matches!(
                                    role,
                                    "navigation"
                                        | "menu"
                                        | "menubar"
                                        | "toolbar"
                                        | "search"
                                        | "banner"
                                        | "contentinfo"
                                        | "complementary"
                                )
                            })
                            || e.classes().any(|class| {
                                let class = class.to_ascii_lowercase();
                                [
                                    "dropdown",
                                    "navbox",
                                    "sidebar",
                                    "breadcrumb",
                                    "toolbar",
                                    "cookie",
                                ]
                                .iter()
                                .any(|marker| class.contains(marker))
                            })
                    })
                });
                if !excluded {
                    let clean = text.split_whitespace().collect::<Vec<_>>().join(" ");
                    if !clean.is_empty() {
                        pieces.push(clean);
                    }
                }
            }
        }
        pieces.join("\n")
    };
    for selector in ["main", "[role=main]", "article"] {
        let selector = scraper::Selector::parse(selector).expect("constant selector");
        if let Some(root) = document.select(&selector).next() {
            let text = extract(root, false);
            if text.chars().count() >= 200 {
                return (title, text);
            }
        }
    }
    let body_selector = scraper::Selector::parse("body").expect("constant selector");
    let root = document
        .select(&body_selector)
        .next()
        .unwrap_or_else(|| document.root_element());
    let text = extract(root, true);
    if text.trim().is_empty() {
        // Never return nothing just because a site puts content in <nav> etc.
        let root_text = root
            .text()
            .map(|t| t.split_whitespace().collect::<Vec<_>>().join(" "))
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        return (title, root_text);
    }
    (title, text)
}

/// Largest local-sites index read (search-index.json grows with the user's sites).
const MAX_LOCAL_INDEX_BYTES: usize = 32 * 1024 * 1024;

/// Searches the user's own websites on the home server through the index the
/// home page uses (`<base>/search-index.json`, rebuilt every 15 minutes by
/// build-site-search.py). The query never leaves this computer or the LAN:
/// the index is downloaded and matched here. Results link to
/// `http://SITE.localhost/PATH`, which the workbench can open as local pages.
pub fn search_local_sites(base_url: &str, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
    if query.trim().is_empty() || query.len() > 4096 {
        bail!("Search query is empty or too long");
    }
    let base = validate_http_url(base_url)?;
    if !is_local_destination(&base) {
        bail!("The local sites index must be on this computer or the local network");
    }
    let index_url = base
        .join("search-index.json")
        .context("Invalid local sites address")?;
    let response = network_client_for(&index_url)?
        .get(index_url)
        .send()?
        .error_for_status()
        .context("The local sites index could not be read")?;
    let mut body = Vec::new();
    response
        .take((MAX_LOCAL_INDEX_BYTES + 1) as u64)
        .read_to_end(&mut body)?;
    if body.len() > MAX_LOCAL_INDEX_BYTES {
        bail!("The local sites index exceeds 32 MiB");
    }
    let index: Value =
        serde_json::from_slice(&body).context("The local sites index is not valid JSON")?;
    Ok(rank_local_pages(&index, query, limit))
}

fn query_terms(query: &str) -> Vec<String> {
    let mut terms: Vec<String> = Vec::new();
    for term in query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
    {
        if !terms.iter().any(|t| t == term) {
            terms.push(term.to_owned());
        }
    }
    terms
}

/// Ranks the pages in a search-index.json: every query word must appear as a whole word
/// (as on the home page); title matches count ten times a body match.
fn rank_local_pages(index: &Value, query: &str, limit: usize) -> Vec<SearchResult> {
    let terms = query_terms(query);
    if terms.is_empty() {
        return Vec::new();
    }
    let valid_site = |s: &str| {
        !s.is_empty()
            && s.len() <= 41
            && s.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    };
    let mut hits: Vec<(usize, SearchResult)> = Vec::new();
    for page in index
        .get("pages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let site = page.get("site").and_then(Value::as_str).unwrap_or("");
        let path = page.get("path").and_then(Value::as_str).unwrap_or("/");
        if !valid_site(site) || !path.starts_with('/') || path.contains("//") {
            continue;
        }
        let title = page.get("title").and_then(Value::as_str).unwrap_or(path);
        let text = page.get("text").and_then(Value::as_str).unwrap_or("");
        let (title_l, text_l) = (title.to_lowercase(), text.to_lowercase());
        let mut score = 0usize;
        if !terms.iter().all(|t| {
            let (a, b) = (word_count(&title_l, t), word_count(&text_l, t));
            score += a * 10 + b.min(50);
            a + b > 0
        }) {
            continue;
        }
        hits.push((
            score,
            SearchResult {
                title: title.to_owned(),
                url: format!("http://{site}.localhost{path}"),
                snippet: local_snippet(text, &text_l, &terms),
                ..Default::default()
            },
        ));
    }
    hits.sort_by_key(|h| std::cmp::Reverse(h.0));
    hits.into_iter().take(limit).map(|(_, r)| r).collect()
}

/// Whole-word matches of `term` in lowercase `text` (a plural -s/-es is allowed), so "ai" never
/// matches inside "said" or "pain". Returns each match's byte offset.
fn word_matches<'a>(text: &'a str, term: &'a str) -> impl Iterator<Item = usize> + 'a {
    text.match_indices(term).filter_map(move |(at, _)| {
        let before = text[..at].chars().next_back();
        let rest = &text[at + term.len()..];
        let rest = rest
            .strip_prefix("es")
            .filter(|r| !r.starts_with(char::is_alphanumeric))
            .or_else(|| {
                rest.strip_prefix('s')
                    .filter(|r| !r.starts_with(char::is_alphanumeric))
            })
            .unwrap_or(rest);
        let after = rest.chars().next();
        (!before.is_some_and(char::is_alphanumeric) && !after.is_some_and(char::is_alphanumeric))
            .then_some(at)
    })
}

fn word_count(text: &str, term: &str) -> usize {
    word_matches(text, term).count()
}

fn local_snippet(text: &str, lower: &str, terms: &[String]) -> String {
    let at = terms
        .iter()
        .find_map(|t| word_matches(lower, t).next())
        .unwrap_or(0);
    // Work in characters so multi-byte text is never split.
    let chars: Vec<char> = text.chars().collect();
    let at_char = lower[..at].chars().count().min(chars.len());
    let start = at_char.saturating_sub(100);
    let end = (start + 300).min(chars.len());
    let mut snippet: String = chars[start..end].iter().collect();
    if start > 0 {
        snippet.insert_str(0, "...");
    }
    if end < chars.len() {
        snippet.push_str("...");
    }
    snippet
}

/// Requires a user-configured SearXNG instance with JSON enabled.
pub fn search_searxng(base_url: &str, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
    search_searxng_with(base_url, query, limit, &SearchOptions::default())
}

/// SearXNG search with its standard `time_range`, `language` and `safesearch` parameters.
pub fn search_searxng_with(
    base_url: &str,
    query: &str,
    limit: usize,
    options: &SearchOptions,
) -> Result<Vec<SearchResult>> {
    Ok(search_searxng_full(base_url, query, limit, options, false)?.results)
}

/// SearXNG search returning results and suggested related searches. `images`
/// searches SearXNG's image category; each result then carries `image_url`.
pub fn search_searxng_full(
    base_url: &str,
    query: &str,
    limit: usize,
    options: &SearchOptions,
    images: bool,
) -> Result<SearxngResponse> {
    if query.trim().is_empty() || query.len() > 4096 {
        bail!("Search query is empty or too long");
    }
    let mut url = validate_http_url(base_url)?;
    let base_path = url.path().trim_end_matches('/');
    if !base_path.ends_with("/search") {
        url.set_path(&format!("{base_path}/search"));
    }
    url.set_query(None);
    url.query_pairs_mut()
        .append_pair("q", query)
        .append_pair("format", "json");
    if images {
        url.query_pairs_mut().append_pair("categories", "images");
    }
    if let Some(range) = &options.time_range {
        url.query_pairs_mut().append_pair("time_range", range);
    }
    if let Some(language) = &options.language {
        url.query_pairs_mut().append_pair("language", language);
    }
    if let Some(level) = &options.safe_search {
        let value = match level.as_str() {
            "off" => "0",
            "strict" => "2",
            _ => "1",
        };
        url.query_pairs_mut().append_pair("safesearch", value);
    }
    let response = network_client_for(&url)?
        .get(url)
        .send()?
        .error_for_status()
        .context(
            "SearXNG request failed; ensure JSON is enabled in its search.formats configuration",
        )?;
    let body = limited_body(response)?;
    let json: Value = serde_json::from_slice(&body).context("SearXNG returned non-JSON data")?;
    let results = json
        .get("results")
        .and_then(Value::as_array)
        .context("SearXNG response has no results array")?;
    // No results because every engine failed is a provider error, not an answer:
    // say which engines failed and why instead of returning an empty list.
    if results.is_empty() {
        let failed: Vec<String> = json
            .get("unresponsive_engines")
            .and_then(Value::as_array)
            .map(|engines| {
                engines
                    .iter()
                    .filter_map(|engine| {
                        let pair = engine.as_array()?;
                        let name = pair.first()?.as_str()?;
                        let why = pair.get(1).and_then(Value::as_str).unwrap_or("failed");
                        Some(format!("{name} ({why})"))
                    })
                    .collect()
            })
            .unwrap_or_default();
        if !failed.is_empty() {
            bail!(
                "SearXNG returned no results because its search engines failed: {}. This is a provider error, not an empty result",
                failed.join(", ")
            );
        }
    }
    let text = |r: &Value, key: &str| r.get(key).and_then(Value::as_str).unwrap_or("").trim().to_owned();
    let results = results
        .iter()
        .filter_map(|r| {
            let url = r.get("url")?.as_str()?;
            if validate_http_url(url).is_err() {
                return None;
            }
            let image = |key: &str| {
                let value = text(r, key);
                // SearXNG gives some image links without a scheme ("//host/x.jpg").
                let value = if value.starts_with("//") { format!("https:{value}") } else { value };
                if validate_http_url(&value).is_ok() { value } else { String::new() }
            };
            Some(SearchResult {
                title: text(r, "title"),
                url: url.to_owned(),
                snippet: text(r, "content").chars().take(4000).collect(),
                image_url: if images { image("img_src") } else { String::new() },
                thumbnail: if images { image("thumbnail_src") } else { String::new() },
            })
        })
        .filter(|r| !images || !r.image_url.is_empty())
        .take(limit.clamp(1, 20))
        .collect();
    let suggestions = json
        .get("suggestions")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty() && s.chars().count() <= 200)
                .take(8)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    Ok(SearxngResponse { results, suggestions })
}

/// Terminal text browsers that can fetch a page source non-interactively,
/// in automatic-detection order.
pub const TEXT_BROWSERS: [&str; 4] = ["lynx", "w3m", "elinks", "links"];

/// A text browser resolved to an executable for keyless web search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextBrowser {
    pub name: String,
    pub program: PathBuf,
}

/// Resolve the `text_browser` setting: "auto" picks the first installed
/// browser, "off"/blank disables, a browser name requires that browser.
pub fn resolve_text_browser(choice: &str) -> Option<TextBrowser> {
    let choice = choice.trim().to_ascii_lowercase();
    let candidates: Vec<&str> = match choice.as_str() {
        "" | "off" | "none" => return None,
        "auto" => TEXT_BROWSERS.to_vec(),
        name if TEXT_BROWSERS.contains(&name) => vec![TEXT_BROWSERS
            .iter()
            .copied()
            .find(|candidate| *candidate == name)?],
        _ => return None,
    };
    candidates.into_iter().find_map(|name| {
        crate::diagnostics::find_executable(name).map(|program| TextBrowser {
            name: name.to_owned(),
            program,
        })
    })
}

/// Every installed text browser, in preference order.
pub fn installed_text_browsers() -> Vec<TextBrowser> {
    TEXT_BROWSERS
        .iter()
        .filter_map(|name| {
            crate::diagnostics::find_executable(name).map(|program| TextBrowser {
                name: (*name).to_owned(),
                program,
            })
        })
        .collect()
}

/// DuckDuckGo's keyless, JavaScript-free results page.
pub fn duckduckgo_html_url(query: &str) -> Result<url::Url> {
    duckduckgo_html_url_with(query, &SearchOptions::default())
}

/// DuckDuckGo's HTML results page: `df` (date), `kl` (region-language, e.g. us-en)
/// and `kp` (safe search: 1 strict, -1 moderate, -2 off).
pub fn duckduckgo_html_url_with(query: &str, options: &SearchOptions) -> Result<url::Url> {
    if query.trim().is_empty() || query.len() > 4096 {
        bail!("Search query is empty or too long");
    }
    let mut url = url::Url::parse("https://html.duckduckgo.com/html/").expect("constant URL");
    url.query_pairs_mut().append_pair("q", query.trim());
    if let Some(range) = &options.time_range {
        url.query_pairs_mut().append_pair("df", &range[..1]);
    }
    if let (Some(language), Some(region)) = (options.language_code(), options.region()) {
        url.query_pairs_mut()
            .append_pair("kl", &format!("{}-{language}", region.to_ascii_lowercase()));
    }
    if let Some(level) = &options.safe_search {
        let value = match level.as_str() {
            "strict" => "1",
            "off" => "-2",
            _ => "-1",
        };
        url.query_pairs_mut().append_pair("kp", value);
    }
    Ok(url)
}

/// The exact non-interactive command: the browser prints the results page
/// source, which Workbench parses. No shell is involved.
pub fn text_browser_search_command(browser: &TextBrowser, query: &str) -> Result<CommandSpec> {
    text_browser_search_command_with(browser, query, &SearchOptions::default())
}

pub fn text_browser_search_command_with(
    browser: &TextBrowser,
    query: &str,
    options: &SearchOptions,
) -> Result<CommandSpec> {
    let url = duckduckgo_html_url_with(query, options)?.to_string();
    let mut args: Vec<String> = match browser.name.as_str() {
        "lynx" => vec!["-source".into(), "-accept_all_cookies".into()],
        "w3m" => vec!["-dump_source".into()],
        "elinks" => vec!["-no-home".into(), "1".into(), "-source".into(), "1".into()],
        "links" => vec!["-source".into()],
        other => bail!("Unsupported text browser {other}"),
    };
    args.push(url);
    let spec = CommandSpec {
        program: browser.program.to_string_lossy().into_owned(),
        args,
        timeout_secs: 30,
        max_output_bytes: MAX_HTTP_BYTES.min(MAX_CAPTURE),
        ..Default::default()
    };
    spec.validate()?;
    Ok(spec)
}

/// Search with each browser in turn until one returns a DuckDuckGo results
/// page. Returns the browser that succeeded, its results, and the failures of
/// browsers tried before it.
pub fn search_text_browsers(
    browsers: &[TextBrowser],
    query: &str,
    limit: usize,
    cwd: &Path,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<(String, Vec<SearchResult>, Vec<String>)> {
    let mut failures = Vec::new();
    for browser in browsers {
        if cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
            bail!("Search cancelled");
        }
        match search_text_browser(browser, query, limit, cwd, cancel.clone()) {
            Ok(results) => return Ok((browser.name.clone(), results, failures)),
            Err(error) => failures.push(format!("{}: {error:#}", browser.name)),
        }
    }
    if failures.is_empty() {
        bail!("No text browser (lynx, w3m, elinks or links) is installed");
    }
    bail!("Every text browser failed: {}", failures.join("; "))
}

/// The exact non-interactive command that renders a page as plain text.
pub fn text_browser_dump_command(browser: &TextBrowser, url: &str) -> Result<CommandSpec> {
    let mut args: Vec<String> = match browser.name.as_str() {
        "lynx" => vec![
            "-dump".into(),
            "-nolist".into(),
            "-width=110".into(),
            "-accept_all_cookies".into(),
        ],
        "w3m" => vec!["-dump".into(), "-cols".into(), "110".into()],
        "elinks" => vec![
            "-no-home".into(),
            "1".into(),
            "-dump".into(),
            "1".into(),
            "-no-references".into(),
            "1".into(),
        ],
        "links" => vec!["-dump".into()],
        other => bail!("Unsupported text browser {other}"),
    };
    args.push(url.to_owned());
    let spec = CommandSpec {
        program: browser.program.to_string_lossy().into_owned(),
        args,
        timeout_secs: 30,
        max_output_bytes: MAX_HTTP_BYTES.min(MAX_CAPTURE),
        ..Default::default()
    };
    spec.validate()?;
    Ok(spec)
}

/// True when rendered text is a bot/JavaScript wall rather than content.
pub fn looks_like_script_wall(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let trimmed = lower.trim();
    trimmed.chars().count() < 200
        || [
            "enable javascript",
            "javascript is disabled",
            "just a moment",
            "checking your browser",
            "verify you are human",
            "please enable cookies",
        ]
        .iter()
        .any(|marker| trimmed.chars().count() < 3000 && trimmed.contains(marker))
}

/// Render a public page as text with each browser in turn. Pages behind a
/// JavaScript/bot wall are reported as errors, not as content.
pub fn fetch_page_text_browsers(
    browsers: &[TextBrowser],
    url: &str,
    cwd: &Path,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<(String, String)> {
    let parsed = validate_http_url(url)?;
    if is_local_destination(&parsed) {
        bail!("Refusing to render a local or private address: {url}");
    }
    let mut failures = Vec::new();
    for browser in browsers {
        if cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
            bail!("Page retrieval cancelled");
        }
        let spec = text_browser_dump_command(browser, parsed.as_str())?;
        let attempt = run_command_cancellable(&spec, cwd, cancel.clone()).and_then(|result| {
            if result.cancelled {
                bail!("cancelled");
            }
            if result.timed_out {
                bail!("timed out after {} seconds", spec.timeout_secs);
            }
            if result.exit_code != Some(0) {
                bail!("exit {:?}", result.exit_code);
            }
            if looks_like_script_wall(&result.stdout) {
                bail!("the site requires JavaScript or blocked the text browser");
            }
            Ok(result.stdout)
        });
        match attempt {
            Ok(text) => return Ok((browser.name.clone(), text)),
            Err(error) => failures.push(format!("{}: {error:#}", browser.name)),
        }
    }
    if failures.is_empty() {
        bail!("No text browser (lynx, w3m, elinks or links) is installed");
    }
    bail!("Could not read {url}: {}", failures.join("; "))
}

/// A Stack Exchange question and its top answers, from the public API.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StackExchangeThread {
    pub site: String,
    pub question_id: u64,
    pub title: String,
    pub question: String,
    pub answers: Vec<StackExchangeAnswer>,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StackExchangeAnswer {
    pub score: i64,
    pub accepted: bool,
    pub body: String,
}

/// Map a Stack Exchange question URL to its API site parameter and ID.
/// Stack Exchange serves text browsers a JavaScript wall, so its content is
/// read through the public API instead.
pub fn stack_exchange_question(url: &str) -> Option<(String, u64)> {
    let parsed = url::Url::parse(url).ok()?;
    let host = parsed
        .host_str()?
        .trim_start_matches("www.")
        .to_ascii_lowercase();
    let site = match host.as_str() {
        "stackoverflow.com" => "stackoverflow".to_owned(),
        "superuser.com" => "superuser".to_owned(),
        "serverfault.com" => "serverfault".to_owned(),
        "askubuntu.com" => "askubuntu".to_owned(),
        "mathoverflow.net" => "mathoverflow.net".to_owned(),
        "stackapps.com" => "stackapps".to_owned(),
        other => {
            let name = other.strip_suffix(".stackexchange.com")?;
            if name.is_empty() || name == "meta" || name.contains(".") {
                return None;
            }
            name.to_owned()
        }
    };
    let mut segments = parsed.path_segments()?;
    let kind = segments.next()?;
    if kind != "questions" && kind != "q" {
        return None;
    }
    let id = segments.next()?.parse().ok()?;
    Some((site, id))
}

fn stack_exchange_get(path: &str, site: &str, params: &[(&str, &str)]) -> Result<Value> {
    let mut url = url::Url::parse("https://api.stackexchange.com/2.3/").expect("constant URL");
    url.set_path(&format!("/2.3/{path}"));
    url.query_pairs_mut().append_pair("site", site);
    url.query_pairs_mut().append_pair("filter", "withbody");
    for (key, value) in params {
        url.query_pairs_mut().append_pair(key, value);
    }
    let response = network_client()?
        .get(url)
        .header(reqwest::header::ACCEPT_ENCODING, "gzip")
        .send()?;
    let gzip = response
        .headers()
        .get(reqwest::header::CONTENT_ENCODING)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.eq_ignore_ascii_case("gzip"));
    let status = response.status();
    let body = limited_body(response)?;
    // The API always compresses; this client has no automatic decoding.
    let body = if gzip || body.starts_with(&[0x1f, 0x8b]) {
        let mut decoded = Vec::new();
        flate2::read::GzDecoder::new(&body[..])
            .take(MAX_HTTP_BYTES as u64 + 1)
            .read_to_end(&mut decoded)?;
        decoded
    } else {
        body
    };
    let value: Value =
        serde_json::from_slice(&body).context("Stack Exchange returned invalid JSON")?;
    if let Some(message) = value.get("error_message").and_then(Value::as_str) {
        bail!("Stack Exchange API error: {message}");
    }
    if !status.is_success() {
        bail!("Stack Exchange API returned HTTP {status}");
    }
    Ok(value)
}

/// Fetch a question and its highest-voted answers through the Stack Exchange API.
pub fn stack_exchange_thread(
    site: &str,
    question_id: u64,
    max_answers: usize,
) -> Result<StackExchangeThread> {
    let question = stack_exchange_get(&format!("questions/{question_id}"), site, &[])?;
    let item = question
        .get("items")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .context("Stack Exchange question not found")?;
    let page_size = max_answers.clamp(1, 5).to_string();
    let answers = stack_exchange_get(
        &format!("questions/{question_id}/answers"),
        site,
        &[
            ("order", "desc"),
            ("sort", "votes"),
            ("pagesize", &page_size),
        ],
    )?;
    Ok(parse_stack_exchange(
        site,
        question_id,
        item,
        &answers,
        max_answers,
    ))
}

fn parse_stack_exchange(
    site: &str,
    question_id: u64,
    question: &Value,
    answers: &Value,
    max_answers: usize,
) -> StackExchangeThread {
    let text = |value: &Value, key: &str| {
        html_fragment_text(value.get(key).and_then(Value::as_str).unwrap_or_default())
    };
    let mut parsed: Vec<StackExchangeAnswer> = answers
        .get("items")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|a| StackExchangeAnswer {
                    score: a.get("score").and_then(Value::as_i64).unwrap_or(0),
                    accepted: a
                        .get("is_accepted")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    body: text(a, "body"),
                })
                .collect()
        })
        .unwrap_or_default();
    parsed.sort_by(|a, b| b.accepted.cmp(&a.accepted).then(b.score.cmp(&a.score)));
    parsed.truncate(max_answers.clamp(1, 5));
    StackExchangeThread {
        site: site.to_owned(),
        question_id,
        title: text(question, "title"),
        question: text(question, "body"),
        answers: parsed,
        tags: question
            .get("tags")
            .and_then(Value::as_array)
            .map(|tags| {
                tags.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default(),
    }
}

/// Plain text from an HTML fragment, keeping code blocks and paragraphs apart.
pub fn html_fragment_text(html: &str) -> String {
    let fragment = scraper::Html::parse_fragment(html);
    let mut out = String::new();
    for node in fragment.root_element().descendants() {
        if let Some(text) = node.value().as_text() {
            out.push_str(text);
        } else if let Some(element) = node.value().as_element() {
            if matches!(
                element.name(),
                "p" | "pre" | "li" | "br" | "h1" | "h2" | "h3" | "blockquote"
            ) && !out.ends_with('\n')
                && !out.is_empty()
            {
                out.push('\n');
            }
        }
    }
    out.lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
        .split("\n\n\n")
        .collect::<Vec<_>>()
        .join("\n\n")
        .trim()
        .to_owned()
}

/// Run an approved text-browser search command and parse its results.
pub fn search_text_browser(
    browser: &TextBrowser,
    query: &str,
    limit: usize,
    cwd: &Path,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<Vec<SearchResult>> {
    search_text_browser_with(browser, query, limit, cwd, cancel, &SearchOptions::default())
}

pub fn search_text_browser_with(
    browser: &TextBrowser,
    query: &str,
    limit: usize,
    cwd: &Path,
    cancel: Option<Arc<AtomicBool>>,
    options: &SearchOptions,
) -> Result<Vec<SearchResult>> {
    let spec = text_browser_search_command_with(browser, query, options)?;
    let result = run_command_cancellable(&spec, cwd, cancel)?;
    if result.cancelled {
        bail!("{} search was cancelled", browser.name);
    }
    if result.timed_out {
        bail!(
            "{} did not return search results within {} seconds",
            browser.name,
            spec.timeout_secs
        );
    }
    if result.exit_code != Some(0) || result.stdout.trim().is_empty() {
        let stderr = result.stderr.trim().chars().take(600).collect::<String>();
        bail!(
            "{} could not fetch DuckDuckGo results (exit {:?}){}",
            browser.name,
            result.exit_code,
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        );
    }
    if result.truncated {
        bail!(
            "{} returned more than the search page size limit",
            browser.name
        );
    }
    parse_duckduckgo_html(&result.stdout, limit)
}

/// Reads an MCP search tool's reply into the standard result list. Handles
/// `structuredContent`, JSON text items (one result or a list, under keys such
/// as results/web/items), "Title:/URL:/Description:" text blocks, Markdown
/// links and bare URLs. Duplicate and non-HTTP(S) URLs are dropped.
pub fn normalize_mcp_search(reply: &Value, limit: usize) -> Vec<SearchResult> {
    let mut out = Vec::new();
    if let Some(structured) = reply.get("structuredContent") {
        collect_search_results(structured, &mut out, 0);
    }
    if let Some(items) = reply.get("content").and_then(Value::as_array) {
        for item in items {
            if let Some(text) = item.get("text").and_then(Value::as_str) {
                match serde_json::from_str::<Value>(text.trim()) {
                    Ok(json) if json.is_object() || json.is_array() => {
                        collect_search_results(&json, &mut out, 0)
                    }
                    _ => parse_search_text(text, &mut out),
                }
            } else if let Some(resource) = item.get("resource").or(Some(item)) {
                collect_search_results(resource, &mut out, 0);
            }
        }
    }
    if out.is_empty() && reply.get("content").is_none() && reply.get("structuredContent").is_none() {
        collect_search_results(reply, &mut out, 0);
    }
    let mut seen = std::collections::HashSet::new();
    out.retain(|r| validate_http_url(&r.url).is_ok() && seen.insert(r.url.clone()));
    out.truncate(limit.clamp(1, 100));
    out
}

/// Related searches a provider suggested, from keys such as `suggestions`,
/// `related_searches` or `query_suggestions` anywhere in the reply's JSON.
pub fn mcp_suggestions(reply: &Value) -> Vec<String> {
    fn walk(value: &Value, out: &mut Vec<String>, depth: usize) {
        if depth > 5 || out.len() >= 8 {
            return;
        }
        match value {
            Value::Object(map) => {
                for (key, child) in map {
                    let key = key.to_ascii_lowercase();
                    if matches!(key.as_str(), "suggestions" | "related_searches" | "query_suggestions" | "related") {
                        for item in child.as_array().into_iter().flatten() {
                            let text = item.as_str().or_else(|| {
                                ["query", "text", "title"].iter().find_map(|k| item.get(*k).and_then(Value::as_str))
                            });
                            if let Some(text) = text.map(str::trim).filter(|t| !t.is_empty() && t.chars().count() <= 200) {
                                if !out.iter().any(|o| o == text) && out.len() < 8 {
                                    out.push(text.to_owned());
                                }
                            }
                        }
                    } else {
                        walk(child, out, depth + 1);
                    }
                }
            }
            Value::Array(items) => items.iter().for_each(|i| walk(i, out, depth + 1)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    if let Some(structured) = reply.get("structuredContent") {
        walk(structured, &mut out, 0);
    }
    for text in reply
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|i| i.get("text").and_then(Value::as_str))
    {
        if let Ok(json) = serde_json::from_str::<Value>(text.trim()) {
            walk(&json, &mut out, 0);
        }
    }
    out
}

/// Plain text from an MCP reply's content items, for when no results could be read.
pub fn mcp_reply_text(reply: &Value, max_chars: usize) -> String {
    let text = reply
        .get("content")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|i| i.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .unwrap_or_else(|| reply.to_string());
    text.chars().take(max_chars).collect()
}

fn collect_search_results(value: &Value, out: &mut Vec<SearchResult>, depth: usize) {
    if depth > 5 || out.len() >= 200 {
        return;
    }
    match value {
        Value::Array(items) => {
            for item in items {
                collect_search_results(item, out, depth + 1);
            }
        }
        Value::Object(map) => {
            let field = |keys: &[&str]| {
                keys.iter().find_map(|k| map.get(*k).and_then(Value::as_str)).map(str::trim)
            };
            if let Some(url) = field(&["url", "link", "href", "uri"]).filter(|u| u.starts_with("http")) {
                let mut snippet = field(&["description", "snippet", "content", "summary", "text", "body"])
                    .unwrap_or_default()
                    .to_owned();
                // Exa returns "highlights", Brave "extra_snippets": short passages.
                for key in ["highlights", "extra_snippets"] {
                    for more in map.get(key).and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).take(2) {
                        if !snippet.is_empty() {
                            snippet.push(' ');
                        }
                        snippet.push_str(more.trim());
                    }
                }
                let image = |keys: &[&str]| {
                    keys.iter()
                        .find_map(|k| match map.get(*k) {
                            Some(Value::String(s)) => Some(s.trim().to_owned()),
                            Some(Value::Object(o)) => o.get("url").and_then(Value::as_str).map(str::to_owned),
                            _ => None,
                        })
                        .filter(|u| u.starts_with("http"))
                        .unwrap_or_default()
                };
                out.push(SearchResult {
                    title: strip_tags(field(&["title", "name", "headline"]).unwrap_or_default()),
                    url: url.to_owned(),
                    snippet: strip_tags(&snippet).chars().take(4000).collect(),
                    image_url: image(&["image_url", "img_src", "image", "properties"]),
                    thumbnail: image(&["thumbnail", "thumbnail_src", "thumbnail_url"]),
                });
                return;
            }
            for (_, child) in map {
                if child.is_array() || child.is_object() {
                    collect_search_results(child, out, depth + 1);
                }
            }
        }
        _ => {}
    }
}

fn parse_search_text(text: &str, out: &mut Vec<SearchResult>) {
    let before = out.len();
    // "Title: …\nURL: …\nDescription: …" blocks. Exa adds "Highlights:" followed
    // by passages on the next lines, and separates results with "---".
    for block in text.split("\n\n") {
        let mut title = String::new();
        let mut url = String::new();
        let mut snippet = String::new();
        let mut continues_snippet = false;
        for line in block.lines().map(str::trim) {
            let key = line
                .split_once(':')
                .map(|(k, _)| k.to_ascii_lowercase())
                .filter(|k| k.len() <= 12 && k.chars().all(|c| c.is_ascii_alphabetic()));
            let value = || {
                let v = line.split_once(':').map_or("", |(_, v)| v).trim();
                if v.eq_ignore_ascii_case("n/a") { String::new() } else { v.to_owned() }
            };
            match key.as_deref() {
                Some("title") => {
                    title = value();
                    continues_snippet = false;
                }
                Some("url" | "link") => {
                    url = value();
                    continues_snippet = false;
                }
                Some("description" | "snippet" | "highlights" | "text" | "summary" | "content") => {
                    let more = value();
                    if !more.is_empty() {
                        if !snippet.is_empty() {
                            snippet.push(' ');
                        }
                        snippet.push_str(&more);
                    }
                    continues_snippet = true;
                }
                Some("published" | "author" | "date" | "score" | "id") => continues_snippet = false,
                _ if continues_snippet && !line.is_empty() && line != "---" => {
                    if !snippet.is_empty() {
                        snippet.push(' ');
                    }
                    snippet.push_str(line);
                }
                _ => {}
            }
        }
        if url.starts_with("http") {
            out.push(SearchResult {
                title,
                url,
                snippet: snippet.chars().take(4000).collect(),
                ..Default::default()
            });
        }
    }
    if out.len() > before {
        return;
    }
    // Markdown links, then bare URLs with the preceding line as the title.
    let mut previous = "";
    for line in text.lines().map(str::trim) {
        if let Some(start) = line.find("](http") {
            if let (Some(open), Some(close)) = (line[..start].rfind('['), line[start + 2..].find(')')) {
                let url = &line[start + 2..start + 2 + close];
                let rest = line[start + 3 + close..].trim_start_matches([' ', '-', ':', '—']).trim();
                out.push(SearchResult {
                    title: line[open + 1..start].to_owned(),
                    url: url.to_owned(),
                    snippet: rest.to_owned(),
                    ..Default::default()
                });
            }
        } else if line.starts_with("http://") || line.starts_with("https://") {
            let url = line.split_whitespace().next().unwrap_or(line);
            out.push(SearchResult {
                title: previous.to_owned(),
                url: url.to_owned(),
                ..Default::default()
            });
        }
        if !line.is_empty() {
            previous = line;
        }
    }
}

fn strip_tags(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut inside = false;
    for c in text.chars() {
        match c {
            '<' => inside = true,
            '>' if inside => inside = false,
            _ if !inside => out.push(c),
            _ => {}
        }
    }
    out.replace("&amp;", "&").replace("&quot;", "\"").replace("&#39;", "'").trim().to_owned()
}

/// Parse DuckDuckGo's HTML results page. Ads and non-HTTP(S) links are
/// dropped; DuckDuckGo redirect links are decoded to the destination URL.
pub fn parse_duckduckgo_html(html: &str, limit: usize) -> Result<Vec<SearchResult>> {
    let document = scraper::Html::parse_document(html);
    let result_selector = scraper::Selector::parse(".result").expect("constant selector");
    let link_selector = scraper::Selector::parse("a.result__a").expect("constant selector");
    let snippet_selector = scraper::Selector::parse(".result__snippet").expect("constant selector");
    let squash = |text: scraper::element_ref::Text| {
        text.collect::<Vec<_>>()
            .join(" ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    };
    let mut results = Vec::new();
    for item in document.select(&result_selector) {
        if item.value().classes().any(|class| class == "result--ad") {
            continue;
        }
        let Some(link) = item.select(&link_selector).next() else {
            continue;
        };
        let Some(url) = link.value().attr("href").and_then(duckduckgo_destination) else {
            continue;
        };
        if results
            .iter()
            .any(|existing: &SearchResult| existing.url == url)
        {
            continue;
        }
        results.push(SearchResult {
            title: squash(link.text()),
            url,
            snippet: item
                .select(&snippet_selector)
                .next()
                .map(|snippet| squash(snippet.text()))
                .unwrap_or_default()
                .chars()
                .take(4000)
                .collect(),
            ..Default::default()
        });
        if results.len() >= limit.clamp(1, 20) {
            break;
        }
    }
    if results.is_empty() {
        let lower = html.to_ascii_lowercase();
        if lower.contains("anomaly")
            || lower.contains("captcha")
            || lower.contains("challenge-form")
        {
            bail!("DuckDuckGo refused this automated search (bot check). Wait a few minutes and retry, or configure SearXNG/MCP search");
        }
        if lower.contains("error getting results") || lower.contains("if this persists") {
            bail!("DuckDuckGo refused this search (rate limited after too many automated queries). This is a provider error, not an empty result: wait several minutes and retry, or configure SearXNG/MCP search");
        }
        // Only DuckDuckGo's explicit no-results marker means "nothing found".
        // Any other page without results is an error page, not an empty search.
        if !lower.contains("no-results") && !lower.contains("no results") {
            bail!("The text browser did not return a DuckDuckGo results page");
        }
    }
    Ok(results)
}

fn duckduckgo_destination(href: &str) -> Option<String> {
    let absolute = if href.starts_with("//") {
        format!("https:{href}")
    } else if href.starts_with('/') {
        format!("https://duckduckgo.com{href}")
    } else {
        href.to_owned()
    };
    let parsed = url::Url::parse(&absolute).ok()?;
    let destination = if parsed
        .host_str()
        .is_some_and(|host| host.ends_with("duckduckgo.com"))
    {
        if parsed.path() != "/l/" {
            return None; // ads (/y.js) and internal links
        }
        parsed
            .query_pairs()
            .find(|(key, _)| key == "uddg")
            .map(|(_, value)| value.into_owned())?
    } else {
        absolute
    };
    validate_http_url(&destination)
        .ok()
        .map(|url| url.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn searxng_image_search_returns_pictures_and_suggestions() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut request = [0u8; 4096];
            let n = socket.read(&mut request).unwrap();
            let body = r#"{"results":[
                {"title":"Red fox","url":"https://example.org/fox","img_src":"//images.example.org/fox.jpg","thumbnail_src":"https://images.example.org/fox_t.jpg"},
                {"title":"No picture","url":"https://example.org/text"}
            ],"suggestions":["red fox habitat"," arctic fox ",""]}"#;
            let _ = write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
            String::from_utf8_lossy(&request[..n]).lines().next().unwrap_or_default().to_owned()
        });
        let found = search_searxng_full(&url, "red fox", 5, &SearchOptions::default(), true).unwrap();
        assert!(server.join().unwrap().contains("categories=images"));
        assert_eq!(found.results.len(), 1, "a result without an image is dropped");
        assert_eq!(found.results[0].image_url, "https://images.example.org/fox.jpg");
        assert_eq!(found.results[0].thumbnail, "https://images.example.org/fox_t.jpg");
        assert_eq!(found.suggestions, vec!["red fox habitat", "arctic fox"]);
        // Web results never carry empty image fields in their JSON.
        let web = SearchResult { title: "t".into(), url: "https://a.example/".into(), ..Default::default() };
        assert!(serde_json::to_value(&web).unwrap().get("image_url").is_none());
    }

    #[test]
    fn exa_and_kagi_replies_are_read_into_results() {
        // Exa: text blocks with multi-line highlights, N/A fields and --- separators.
        let exa = serde_json::json!({"content":[{"type":"text","text":
            "Title: Tokio tutorial\nURL: https://tokio.rs/tokio/tutorial\nPublished: N/A\nAuthor: N/A\nHighlights:\nTokio is an async runtime.\nIt schedules tasks.\n\n---\n\nTitle: N/A\nURL: https://docs.rs/tokio\nPublished: 2026-01-02\nAuthor: Tokio team\nText: API docs"}]});
        let results = normalize_mcp_search(&exa, 10);
        assert_eq!(results.len(), 2, "{results:?}");
        assert_eq!(results[0].title, "Tokio tutorial");
        assert_eq!(results[0].snippet, "Tokio is an async runtime. It schedules tasks.");
        assert_eq!((results[1].title.as_str(), results[1].snippet.as_str()), ("", "API docs"));
        // Kagi JSON: results grouped by category, images with image_url.
        let kagi = serde_json::json!({"content":[{"type":"text","text":
            "{\"search\":[{\"title\":\"Fox\",\"url\":\"https://en.wikipedia.org/wiki/Fox\",\"snippet\":\"A canid\"}],\"image\":[{\"title\":\"Fox photo\",\"url\":\"https://photos.example/fox\",\"image_url\":\"https://photos.example/fox.jpg\"}],\"related_searches\":[\"fox species\"]}"}]});
        let results = normalize_mcp_search(&kagi, 10);
        assert_eq!(results.len(), 2);
        let photo = results.iter().find(|r| r.title == "Fox photo").unwrap();
        assert_eq!(photo.image_url, "https://photos.example/fox.jpg");
        assert_eq!(mcp_suggestions(&kagi), vec!["fox species"]);
        assert!(mcp_suggestions(&exa).is_empty());
    }

    #[test]
    fn search_options_are_validated_and_normalized() {
        let options = SearchOptions::from_args(&serde_json::json!({
            "time_range":"Week","language":"EN_us","safe_search":"STRICT"
        }))
        .unwrap();
        assert_eq!(options.time_range.as_deref(), Some("week"));
        assert_eq!(options.language.as_deref(), Some("en-US"));
        assert_eq!((options.language_code(), options.region()), (Some("en"), Some("US")));
        assert_eq!(options.safe_search.as_deref(), Some("strict"));
        assert!(SearchOptions::from_args(&serde_json::json!({"language":""})).unwrap().is_empty());
        for bad in [
            serde_json::json!({"time_range":"fortnight"}),
            serde_json::json!({"safe_search":"high"}),
            serde_json::json!({"language":"english"}),
            serde_json::json!({"language":"en-USA"}),
            serde_json::json!({"time_range":7}),
        ] {
            assert!(SearchOptions::from_args(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn duckduckgo_receives_date_region_and_safe_search() {
        let options = SearchOptions {
            time_range: Some("month".into()),
            language: Some("en-GB".into()),
            safe_search: Some("off".into()),
        };
        let url = duckduckgo_html_url_with("rust", &options).unwrap();
        let pairs: Vec<(String, String)> = url.query_pairs().into_owned().collect();
        for expected in [("q", "rust"), ("df", "m"), ("kl", "gb-en"), ("kp", "-2")] {
            assert!(pairs.contains(&(expected.0.into(), expected.1.into())), "{pairs:?}");
        }
        // A language without a region cannot be applied by DuckDuckGo, and says so.
        let language_only = SearchOptions { language: Some("de".into()), ..Default::default() };
        assert!(!duckduckgo_html_url_with("rust", &language_only).unwrap().as_str().contains("kl="));
        let report = language_only.report(&language_only.duckduckgo_applied());
        assert_eq!(report["not_applied"], serde_json::json!(["language"]));
        assert_eq!(SearchOptions::default().report(&[]), Value::Null);
        assert_eq!(duckduckgo_html_url("rust").unwrap(), duckduckgo_html_url_with("rust", &SearchOptions::default()).unwrap());
    }

    #[test]
    fn searxng_receives_its_standard_option_parameters() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut request = [0u8; 4096];
            let n = socket.read(&mut request).unwrap();
            let body = r#"{"results":[{"title":"Rust","url":"https://www.rust-lang.org/","content":"A language"}]}"#;
            let _ = write!(
                socket,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            String::from_utf8_lossy(&request[..n]).lines().next().unwrap_or_default().to_owned()
        });
        let options = SearchOptions {
            time_range: Some("week".into()),
            language: Some("de-DE".into()),
            safe_search: Some("strict".into()),
        };
        let results = search_searxng_with(&url, "rust", 5, &options).unwrap();
        assert_eq!(results[0].url, "https://www.rust-lang.org/");
        let request_line = server.join().unwrap();
        for part in ["q=rust", "format=json", "time_range=week", "language=de-DE", "safesearch=2"] {
            assert!(request_line.contains(part), "{request_line}");
        }
    }

    #[test]
    fn mcp_search_replies_become_the_standard_result_list() {
        let brave_style = serde_json::json!({"content":[
            {"type":"text","text":"{\"url\":\"https://www.rust-lang.org/\",\"title\":\"Rust <strong>Language</strong>\",\"description\":\"Fast &amp; safe\",\"extra_snippets\":[\"Memory safety\"]}"},
            {"type":"text","text":"{\"url\":\"https://doc.rust-lang.org/book/\",\"title\":\"The Book\",\"description\":\"Learn Rust\"}"}
        ]});
        let results = normalize_mcp_search(&brave_style, 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Rust Language");
        assert_eq!(results[0].snippet, "Fast & safe Memory safety");

        let structured = serde_json::json!({
            "structuredContent":{"web":{"results":[{"link":"https://a.example/","title":"A","snippet":"first"}]}},
            "content":[{"type":"text","text":"[{\"href\":\"https://a.example/\",\"name\":\"A again\"},{\"href\":\"ftp://b.example/\",\"name\":\"not web\"}]"}]
        });
        let results = normalize_mcp_search(&structured, 10);
        assert_eq!(results.len(), 1, "{results:?}");
        assert_eq!((results[0].title.as_str(), results[0].snippet.as_str()), ("A", "first"));

        let blocks = serde_json::json!({"content":[{"type":"text","text":
            "Title: First\nURL: https://one.example/\nDescription: One\n\nTitle: Second\nURL: https://two.example/\nDescription: Two"}]});
        let results = normalize_mcp_search(&blocks, 1);
        assert_eq!(results.len(), 1);
        assert_eq!((results[0].title.as_str(), results[0].url.as_str(), results[0].snippet.as_str()), ("First", "https://one.example/", "One"));

        let markdown = serde_json::json!({"content":[{"type":"text","text":
            "1. [Tokio](https://tokio.rs/) - An async runtime\nSmol\nhttps://github.com/smol-rs/smol"}]});
        let results = normalize_mcp_search(&markdown, 10);
        assert_eq!(results.len(), 2);
        assert_eq!((results[0].title.as_str(), results[0].snippet.as_str()), ("Tokio", "An async runtime"));
        assert_eq!((results[1].title.as_str(), results[1].url.as_str()), ("Smol", "https://github.com/smol-rs/smol"));

        let unreadable = serde_json::json!({"content":[{"type":"text","text":"No results for that query."}]});
        assert!(normalize_mcp_search(&unreadable, 10).is_empty());
        assert_eq!(mcp_reply_text(&unreadable, 100), "No results for that query.");
    }

    fn local_index() -> Value {
        serde_json::json!({"pages": [
            {"site": "garden", "path": "/", "title": "Garden Notes", "text": "Plant tomatoes after the last frost."},
            {"site": "garden", "path": "/beds/", "title": "Raised beds", "text": "Tomatoes like raised beds and tomato cages."},
            {"site": "recipes", "path": "/bread.md", "title": "Bread", "text": "Flour, water, salt and yeast."},
            {"site": "Bad Site", "path": "/", "title": "tomato", "text": "tomato"},
            {"site": "evil", "path": "//attacker.example/", "title": "tomato", "text": "tomato"}
        ]})
    }

    #[test]
    fn local_pages_rank_by_title_then_body_and_need_every_word() {
        let hits = rank_local_pages(&local_index(), "tomato", 10);
        let urls: Vec<&str> = hits.iter().map(|h| h.url.as_str()).collect();
        assert_eq!(
            urls,
            ["http://garden.localhost/beds/", "http://garden.localhost/"]
        );
        assert!(rank_local_pages(&local_index(), "tomato yeast", 10).is_empty());
        assert_eq!(
            rank_local_pages(&local_index(), "BREAD flour", 10)[0].url,
            "http://recipes.localhost/bread.md"
        );
        assert_eq!(rank_local_pages(&local_index(), "tomato", 1).len(), 1);
        assert!(rank_local_pages(&local_index(), "  ", 10).is_empty());
    }

    #[test]
    fn local_pages_match_whole_words_only() {
        let index = serde_json::json!({"pages":[
            {"site":"a","path":"/said","title":"He said","text":"pain again, aid"},
            {"site":"b","path":"/ai","title":"AI notes","text":"Notes about ai."},
            {"site":"c","path":"/herbs","title":"Herbs","text":"Herbal teas"}]});
        let urls = |q| {
            rank_local_pages(&index, q, 10)
                .into_iter()
                .map(|h| h.url)
                .collect::<Vec<_>>()
        };
        assert_eq!(urls("ai"), ["http://b.localhost/ai"]);
        assert_eq!(
            urls("herb"),
            ["http://c.localhost/herbs"],
            "a plural counts; herbal does not"
        );
        assert!(urls("sai").is_empty());
    }

    #[test]
    fn local_snippets_center_on_the_match_without_splitting_characters() {
        let text = format!("{}é tomato here", "ä".repeat(400));
        let lower = text.to_lowercase();
        let snippet = local_snippet(&text, &lower, &["tomato".into()]);
        assert!(snippet.starts_with("...") && snippet.contains("tomato here"));
    }

    #[test]
    fn local_sites_index_must_be_local() {
        let error = search_local_sites("https://example.com/", "tomato", 5).unwrap_err();
        assert!(error.to_string().contains("local network"));
    }
    use serde_json::json;

    #[test]
    fn stack_exchange_urls_map_to_api_sites() {
        assert_eq!(
            stack_exchange_question("https://stackoverflow.com/questions/927358/how-do-i-undo"),
            Some(("stackoverflow".into(), 927358))
        );
        assert_eq!(
            stack_exchange_question("https://unix.stackexchange.com/questions/12/x"),
            Some(("unix".into(), 12))
        );
        assert_eq!(
            stack_exchange_question("https://askubuntu.com/q/99"),
            Some(("askubuntu".into(), 99))
        );
        assert_eq!(
            stack_exchange_question("https://stackoverflow.com/users/1/x"),
            None
        );
        assert_eq!(
            stack_exchange_question("https://meta.stackexchange.com/questions/1"),
            None
        );
        assert_eq!(
            stack_exchange_question("https://example.org/questions/1"),
            None
        );
    }

    #[test]
    fn stack_exchange_answers_put_accepted_first_and_strip_html() {
        let question = json!({"title":"Undo &amp; redo","body":"<p>How?</p>"});
        let answers = json!({"items":[
            {"score":5,"is_accepted":false,"body":"<p>Use <code>git revert</code>.</p>"},
            {"score":3,"is_accepted":true,"body":"<p>Run:</p><pre><code>git reset HEAD~1\n</code></pre>"}
        ]});
        let thread = parse_stack_exchange("stackoverflow", 1, &question, &answers, 3);
        assert_eq!(thread.title, "Undo & redo");
        assert_eq!(thread.question, "How?");
        assert!(thread.answers[0].accepted);
        assert!(thread.answers[0].body.contains("git reset HEAD~1"));
        assert_eq!(thread.answers[1].body, "Use git revert.");
    }

    #[test]
    fn script_walls_are_not_content() {
        assert!(looks_like_script_wall(
            "Enable JavaScript and cookies to continue"
        ));
        assert!(looks_like_script_wall("   "));
        let article = "A real answer. ".repeat(40);
        assert!(!looks_like_script_wall(&article));
    }

    #[test]
    fn page_rendering_refuses_local_addresses() {
        let error = fetch_page_text_browsers(&[], "http://127.0.0.1:11434/", Path::new("."), None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("local or private"), "{error}");
    }

    #[test]
    fn endpoint_probe_distinguishes_listening_and_closed_ports() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let open = format!("http://{}/", listener.local_addr().unwrap());
        assert!(endpoint_accepts_connections(
            &open,
            Duration::from_millis(500)
        ));
        // Port 0 can never accept connections. (Dropping the listener and
        // reusing its port races with parallel tests that may be handed it.)
        assert!(!endpoint_accepts_connections(
            "http://127.0.0.1:0/",
            Duration::from_millis(500)
        ));
        assert!(!endpoint_accepts_connections(
            "not a url",
            Duration::from_millis(500)
        ));
    }

    #[test]
    fn non_public_addresses_are_recognised() {
        for local in [
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",
            "100.64.0.1",
            "0.0.0.0",
            "::1",
            "fd00::1",
            "fe80::1",
            "::ffff:127.0.0.1",
        ] {
            assert!(non_public_ip(local.parse().unwrap()), "{local}");
        }
        for public in ["1.1.1.1", "93.184.216.34", "100.128.0.1", "2606:4700::1111"] {
            assert!(!non_public_ip(public.parse().unwrap()), "{public}");
        }
    }

    #[test]
    fn searxng_with_every_engine_failing_is_a_provider_error_not_an_empty_result() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            for body in [
                r#"{"results":[],"unresponsive_engines":[["brave","too many requests"],["google cse","timeout"]]}"#,
                r#"{"results":[],"unresponsive_engines":[]}"#,
            ] {
                let (mut socket, _) = listener.accept().unwrap();
                let mut request = [0u8; 2048];
                let _ = socket.read(&mut request);
                let _ = write!(
                    socket,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        let error = search_searxng(&url, "herbs", 5).unwrap_err().to_string();
        assert!(error.contains("brave (too many requests)"), "{error}");
        assert!(error.contains("google cse (timeout)"), "{error}");
        assert!(
            error.contains("provider error, not an empty result"),
            "{error}"
        );
        // Every engine answered and none matched: a genuine empty result.
        assert!(search_searxng(&url, "herbs", 5).unwrap().is_empty());
        server.join().unwrap();
    }

    #[test]
    fn public_requests_refuse_names_that_resolve_to_local_addresses_at_connect_time() {
        // "localhost" resolves to loopback: a client for a public destination must
        // refuse to connect, whatever the URL validation saw earlier.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let error = network_client()
            .unwrap()
            .get(format!("http://localhost:{port}/"))
            .send()
            .unwrap_err();
        let mut chain = error.to_string();
        let mut source = std::error::Error::source(&error);
        while let Some(cause) = source {
            chain.push_str(&format!(": {cause}"));
            source = cause.source();
        }
        assert!(chain.contains("local or private"), "{chain}");
        // A destination the user approved as local stays reachable.
        let approved = url::Url::parse(&format!("http://localhost:{port}/")).unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut socket, _) = listener.accept().unwrap();
            let mut buffer = [0u8; 1024];
            let _ = socket.read(&mut buffer);
            let _ = socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok");
        });
        let response = network_client_for(&approved)
            .unwrap()
            .get(approved)
            .send()
            .unwrap();
        assert_eq!(response.text().unwrap(), "ok");
        server.join().unwrap();
    }

    #[test]
    fn public_pages_cannot_redirect_into_local_addresses() {
        let url = |s: &str| url::Url::parse(s).unwrap();
        let public = url("http://1.1.1.1/page");
        for target in [
            "http://127.0.0.1:11434/api/tags",
            "http://localhost/",
            "http://app.localhost/",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]/",
            "http://192.168.0.10/",
        ] {
            assert!(!redirect_allowed(&public, &url(target)), "{target}");
        }
        assert!(redirect_allowed(&public, &url("https://1.0.0.1/next")));
        // An explicitly approved local URL (e.g. LAN SearXNG) may redirect locally.
        assert!(redirect_allowed(
            &url("http://192.168.0.5:8080/search"),
            &url("http://127.0.0.1/")
        ));
    }

    #[test]
    fn a_fetch_approved_as_public_never_reaches_local_addresses() {
        for target in [
            "http://127.0.0.1:9/",
            "http://localhost:9/",
            "http://[::1]:9/",
            "http://192.168.0.10/",
            "http://169.254.169.254/latest/meta-data/",
        ] {
            let error = fetch_url(target, 1000, false).unwrap_err().to_string();
            assert!(error.contains("local or private"), "{target}: {error}");
        }
    }

    #[test]
    fn approved_local_url_still_follows_local_redirects() {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = thread::spawn(move || {
            for response in [
                format!("HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:{port}/final\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"),
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhello".to_string(),
            ] {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buf = [0u8; 2048];
                let _ = stream.read(&mut buf);
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        let page = fetch_url(&format!("http://127.0.0.1:{port}/start"), 1000, true).unwrap();
        server.join().unwrap();
        assert_eq!(page.content, "hello");
        assert!(page.url.ends_with("/final"), "{}", page.url);
    }

    #[test]
    fn duckduckgo_results_decode_redirects_and_skip_ads() {
        let html = r#"<html><body>
          <div class="result results_links result--ad"><a class="result__a" href="https://duckduckgo.com/y.js?ad_domain=x">Ad</a></div>
          <div class="result results_links"><h2><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Frust%2Dlang.org%2F&amp;rut=abc">Rust <b>Programming</b> Language</a></h2>
            <a class="result__snippet" href="x">A language empowering   everyone.</a></div>
          <div class="result results_links"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Frust%2Dlang.org%2F&amp;rut=dup">Duplicate</a></div>
          <div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=javascript%3Aalert(1)">Bad</a></div>
          <div class="result"><a class="result__a" href="https://en.wikipedia.org/wiki/Rust">Rust - Wikipedia</a></div>
        </body></html>"#;
        let results = parse_duckduckgo_html(html, 10).unwrap();
        assert_eq!(results.len(), 2, "{results:?}");
        assert_eq!(results[0].url, "https://rust-lang.org/");
        assert_eq!(results[0].title, "Rust Programming Language");
        assert_eq!(results[0].snippet, "A language empowering everyone.");
        assert_eq!(results[1].url, "https://en.wikipedia.org/wiki/Rust");
        assert_eq!(parse_duckduckgo_html(html, 1).unwrap().len(), 1);
    }

    #[test]
    fn duckduckgo_bot_check_and_non_results_pages_are_errors() {
        let blocked = parse_duckduckgo_html("<html><div class=\"anomaly-modal\">Unfortunately, bots use DuckDuckGo too</div></html>", 5);
        assert!(blocked.unwrap_err().to_string().contains("bot check"));
        assert!(parse_duckduckgo_html("<html><body>Hello</body></html>", 5).is_err());
        // DuckDuckGo's rate-limit page mentions "results" but is an error.
        let limited = r#"If this persists, please <a href="mailto:error-lite+0480@duckduckgo.com?subject=Error getting results">email us</a>.<br />"#;
        let error = parse_duckduckgo_html(limited, 5).unwrap_err().to_string();
        assert!(error.contains("rate limited"), "{error}");
        assert!(
            parse_duckduckgo_html("<html><body>Search results unavailable</body></html>", 5)
                .is_err()
        );
        let empty = "<html><div class=\"no-results\">No results.</div></html>";
        assert!(parse_duckduckgo_html(empty, 5).unwrap().is_empty());
    }

    #[test]
    fn text_browser_commands_are_exact_argv_without_shell() {
        for (name, flag) in [
            ("lynx", "-source"),
            ("w3m", "-dump_source"),
            ("elinks", "-source"),
            ("links", "-source"),
        ] {
            let browser = TextBrowser {
                name: name.into(),
                program: PathBuf::from(format!("/usr/bin/{name}")),
            };
            let spec = text_browser_search_command(&browser, "rust; rm -rf ~").unwrap();
            assert_eq!(spec.program, format!("/usr/bin/{name}"));
            assert!(spec.args.contains(&flag.to_string()));
            let url = spec.args.last().unwrap();
            assert!(
                url.starts_with("https://html.duckduckgo.com/html/?q="),
                "{url}"
            );
            assert!(!url.contains(' ') && !url.contains(';'), "{url}");
        }
        assert!(resolve_text_browser("off").is_none());
        assert!(resolve_text_browser("not-a-browser").is_none());
    }

    #[test]
    fn adapter_substitutes_argv_without_shell_expansion() {
        let adapter = CommandAdapter {
            id: "x".into(),
            name: "x".into(),
            kind: AdapterKind::Custom,
            enabled: true,
            program: "printf".into(),
            args: vec!["%s".into(), "{text}".into()],
            text_to_stdin: false,
            timeout_secs: 10,
        };
        let spec = adapter
            .command(None, None, Some("$(touch /tmp/never)"), Path::new("/tmp"))
            .unwrap();
        assert_eq!(spec.args[1], "$(touch /tmp/never)");
        assert!(adapter
            .command(
                None,
                Some(Path::new("/tmp/output.txt")),
                Some("text"),
                Path::new("/tmp")
            )
            .is_err());
    }

    #[test]
    fn readable_html_omits_scripts() {
        let (title, text) = html_text("<title>Test</title><body><p>Hello</p><script>secret()</script><style>x{}</style></body>");
        assert_eq!(title, "Test");
        assert!(text.contains("Hello"));
        assert!(!text.contains("secret"));
    }

    #[test]
    fn readable_html_prefers_main_content_over_page_navigation() {
        let article = "Real article sentence. ".repeat(20);
        let (_, text) = html_text(&format!(
            "<body><header>Site banner</header><nav>Main menu Home About</nav>\
             <main><nav>Contents</nav><h1>Topic</h1><p>{article}</p></main>\
             <aside>Related links</aside><footer>Copyright</footer></body>"
        ));
        assert!(text.starts_with("Topic"), "{text}");
        assert!(text.contains("Real article sentence."));
        let (_, wiki_like) = html_text(&format!(
            "<body><main><div class=\"vector-dropdown\"><div class=\"vector-dropdown-content\">\
             Afrikaans Deutsch</div></div><div role=\"navigation\">Article Talk</div>\
             <div aria-hidden=\"true\">Hidden</div><p>{article}</p></main></body>"
        ));
        assert!(
            wiki_like.starts_with("Real article sentence."),
            "{wiki_like}"
        );
        for chrome in [
            "Site banner",
            "Main menu",
            "Contents",
            "Related links",
            "Copyright",
        ] {
            assert!(!text.contains(chrome), "{chrome} leaked into {text}");
        }
    }

    #[test]
    fn readable_html_falls_back_to_body_when_main_is_tiny_or_only_navigation() {
        let (_, text) = html_text(
            "<body><header>Banner</header><main>Hi</main><div><p>Body paragraph</p></div></body>",
        );
        assert!(text.contains("Body paragraph"));
        assert!(!text.contains("Banner"));
        let (_, text) = html_text("<body><nav>Only navigation text</nav></body>");
        assert!(text.contains("Only navigation text"));
    }

    #[test]
    fn rejects_non_http_and_credentials() {
        assert!(validate_http_url("file:///etc/passwd").is_err());
        assert!(validate_http_url("https://user:secret@example.com").is_err());
        assert!(validate_http_url("http://127.0.0.1:8080").is_ok());
    }

    #[test]
    #[cfg(unix)]
    fn command_output_and_timeout_are_bounded() {
        let temp = tempfile::tempdir().unwrap();
        let spec = CommandSpec {
            program: "/bin/sh".into(),
            args: vec!["-c".into(), "printf 'hello'; printf 'problem' >&2".into()],
            ..Default::default()
        };
        let result = run_command(&spec, temp.path()).unwrap();
        assert_eq!(result.stdout, "hello");
        assert_eq!(result.stderr, "problem");
        assert_eq!(result.exit_code, Some(0));
        let result = run_command(
            &CommandSpec {
                program: "/bin/sh".into(),
                args: vec!["-c".into(), "while :; do printf '1234567890'; done".into()],
                timeout_secs: 1,
                max_output_bytes: 1024,
                ..Default::default()
            },
            temp.path(),
        )
        .unwrap();
        assert!(result.timed_out);
        assert!(result.truncated);
        assert!(result.stdout.len() <= 1024);
        assert!(result.elapsed_ms < 5000);
    }

    #[test]
    #[cfg(unix)]
    fn command_cancellation_interrupts_unread_standard_input() {
        let temp = tempfile::tempdir().unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let notify = cancel.clone();
        let stopper = thread::spawn(move || {
            thread::sleep(Duration::from_millis(150));
            notify.store(true, Ordering::Relaxed);
        });
        let result = run_command_cancellable(
            &CommandSpec {
                program: "/bin/sleep".into(),
                args: vec!["10".into()],
                stdin_text: Some("a".repeat(1024 * 1024)),
                timeout_secs: 30,
                ..Default::default()
            },
            temp.path(),
            Some(cancel),
        )
        .unwrap();
        stopper.join().unwrap();
        assert!(result.cancelled);
        assert!(!result.timed_out);
        assert!(result.elapsed_ms < 2500);
        let echoed = run_command(
            &CommandSpec {
                program: "/bin/cat".into(),
                stdin_text: Some("Hello $USER; literal input".into()),
                ..Default::default()
            },
            temp.path(),
        )
        .unwrap();
        assert_eq!(echoed.stdout, "Hello $USER; literal input");
    }
}

#[cfg(test)]
mod shell_argv_regressions {
    use super::*;
    #[test]
    fn rejects_binary_or_python_as_bash_script_but_keeps_shell_scripts() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("ls"), b"\x7fELFfixture").unwrap();
        std::fs::write(
            dir.path().join("modernbasic"),
            b"#!/usr/bin/env python3\nimport os\n",
        )
        .unwrap();
        std::fs::write(dir.path().join("build.sh"), b"#!/bin/bash\necho ok\n").unwrap();
        for script in ["ls", "modernbasic"] {
            assert!(validate_shell_argv("bash", &[script.into()], Some(dir.path())).is_err());
        }
        for args in [
            vec!["build.sh".into()],
            vec!["-lc".into(), "ls".into()],
            vec!["-i".into()],
        ] {
            assert!(validate_shell_argv("bash", &args, Some(dir.path())).is_ok());
        }
        assert!(validate_shell_argv("modernbasic", &["repl".into()], Some(dir.path())).is_ok());
    }
}

fn validate_interpreter_argv(program: &str, args: &[String], cwd: Option<&Path>) -> Result<()> {
    let name = Path::new(program)
        .file_name()
        .and_then(|p| p.to_str())
        .unwrap_or("");
    if !matches!(name, "python" | "python3" | "python2") {
        return Ok(());
    }
    if let Some(first) = args.first() {
        if Path::new(first).file_name().and_then(|p| p.to_str()) == Some(name)
            && !cwd.unwrap_or_else(|| Path::new(".")).join(first).is_file()
        {
            bail!("The interpreter is repeated in args: use program={program:?} with only the script path/options in args. No command was executed.");
        }
    }
    if program == "python"
        && crate::diagnostics::find_executable("python").is_none()
        && crate::diagnostics::find_executable("python3").is_some()
    {
        bail!("python is not installed; python3 is available. Request program=python3 with only the script path/options in args, after checking the project interpreter requirements. No command was executed.");
    }
    Ok(())
}

#[cfg(test)]
mod interpreter_regressions {
    use super::*;
    #[test]
    fn duplicate_python_is_rejected_without_rewriting_real_script_names() {
        let dir = tempfile::tempdir().unwrap();
        assert!(validate_interpreter_argv(
            "python3",
            &["python3".into(), "train.py".into()],
            Some(dir.path())
        )
        .is_err());
        assert!(
            validate_interpreter_argv("python3", &["train.py".into()], Some(dir.path())).is_ok()
        );
        std::fs::write(dir.path().join("python3"), "print(1)").unwrap();
        assert!(
            validate_interpreter_argv("python3", &["python3".into()], Some(dir.path())).is_ok()
        );
    }
}
