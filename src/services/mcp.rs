//! MCP stdio client: newline-delimited JSON-RPC, version negotiation and tool calls.
//! Server processes have ordinary OS permissions; start only after explicit approval.
//! The client offers the newest protocol revision supported by this legacy stdio
//! implementation first and accepts the older revisions listed below.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const MAX_MESSAGE_BYTES: usize = 2 * 1024 * 1024;
const SUPPORTED_PROTOCOLS: &[&str] = &["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

pub const BRAVE_SEARCH_SERVER_ID: &str = "brave-search";
pub const BRAVE_SEARCH_PACKAGE: &str = "@brave/brave-search-mcp-server@2.1.4";
pub const EXA_SEARCH_SERVER_ID: &str = "exa-search";
pub const EXA_SEARCH_PACKAGE: &str = "exa-mcp-server@3.4.3";
pub const KAGI_SEARCH_SERVER_ID: &str = "kagi-search";
pub const KAGI_SEARCH_PACKAGE: &str = "kagimcp==1.1.0";

/// A ready-made web-search server: its pinned package, launcher and API key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchPreset {
    pub id: &'static str,
    pub provider: &'static str,
    pub package: &'static str,
    pub launcher: &'static str,
    pub key_env: &'static str,
    pub key_url: &'static str,
}

/// Official search MCP servers Workbench can add with one click. All start disabled.
pub const SEARCH_PRESETS: [SearchPreset; 3] = [
    SearchPreset {
        id: BRAVE_SEARCH_SERVER_ID,
        provider: "Brave Search",
        package: BRAVE_SEARCH_PACKAGE,
        launcher: "npx",
        key_env: "BRAVE_API_KEY",
        key_url: "https://brave.com/search/api/",
    },
    SearchPreset {
        id: EXA_SEARCH_SERVER_ID,
        provider: "Exa",
        package: EXA_SEARCH_PACKAGE,
        launcher: "npx",
        key_env: "EXA_API_KEY",
        key_url: "https://dashboard.exa.ai/api-keys",
    },
    SearchPreset {
        id: KAGI_SEARCH_SERVER_ID,
        provider: "Kagi",
        package: KAGI_SEARCH_PACKAGE,
        launcher: "uvx",
        key_env: "KAGI_API_KEY",
        key_url: "https://kagi.com/api/keys",
    },
];

pub fn search_preset(id: &str) -> Option<&'static SearchPreset> {
    SEARCH_PRESETS.iter().find(|preset| preset.id == id)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct McpServerConfig {
    pub id: String,
    pub name: String,
    pub program: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub cwd: Option<PathBuf>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Optional mapping that lets Workbench expose this server as the native
    /// `web_search` tool instead of making a small model reason through
    /// mcp_list -> mcp_connect -> mcp_call.
    #[serde(default)]
    pub web_search: Option<McpWebSearchConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct McpWebSearchConfig {
    /// Advertised MCP tool name, for example `brave_web_search`.
    pub tool: String,
    /// JSON argument that receives the search text.
    pub query_argument: String,
    /// Optional JSON argument that receives Workbench's bounded result count.
    pub limit_argument: Option<String>,
    /// Literal provider-specific arguments added to every search request.
    pub extra_arguments: BTreeMap<String, Value>,
    /// Provider-specific maximum query length, measured in Unicode scalar values.
    #[serde(default = "default_search_query_chars")]
    pub max_query_chars: usize,
    /// Optional provider-specific maximum number of whitespace-delimited words.
    #[serde(default)]
    pub max_query_words: Option<usize>,
    /// Provider-specific result-count ceiling. Workbench may impose a smaller bound.
    #[serde(default = "default_search_results")]
    pub max_results: usize,
    /// Argument that receives a requested time range (day/week/month/year).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_range_argument: Option<String>,
    /// Provider values for each time range, e.g. {"week": "pw"}. Empty: the standard word.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub time_range_values: BTreeMap<String, Value>,
    /// Argument that receives the language code ("en" for "en-US").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_argument: Option<String>,
    /// Argument that receives the region code ("US" for "en-US").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region_argument: Option<String>,
    /// Argument that receives a safe-search level (off/moderate/strict).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safe_search_argument: Option<String>,
    /// Provider values for each safe-search level. Empty: the standard word.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub safe_search_values: BTreeMap<String, Value>,
    /// How to search for images, when the provider can. Absent: no image search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_search: Option<McpImageSearchConfig>,
}

/// Image search through the same server: another tool, or the search tool with
/// extra arguments (for example Kagi's `{"workflow": "images"}`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct McpImageSearchConfig {
    /// A different advertised tool for images. Absent: the web-search tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    /// Arguments added to image searches.
    pub extra_arguments: BTreeMap<String, Value>,
}

impl Default for McpWebSearchConfig {
    fn default() -> Self {
        Self {
            tool: String::new(),
            query_argument: "query".into(),
            limit_argument: Some("count".into()),
            extra_arguments: BTreeMap::new(),
            max_query_chars: default_search_query_chars(),
            max_query_words: None,
            max_results: default_search_results(),
            time_range_argument: None,
            time_range_values: BTreeMap::new(),
            language_argument: None,
            region_argument: None,
            safe_search_argument: None,
            safe_search_values: BTreeMap::new(),
            image_search: None,
        }
    }
}

fn default_timeout() -> u64 {
    60
}

fn default_search_query_chars() -> usize {
    2_048
}

fn default_search_results() -> usize {
    10
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(rename = "inputSchema", default = "default_schema")]
    pub input_schema: Value,
}

/// A resource a server advertises through `resources/list`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct McpResource {
    pub uri: String,
    #[serde(default)]
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "mimeType", default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct McpPromptArgument {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub required: bool,
}

/// A prompt template a server advertises through `prompts/list`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct McpPrompt {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub arguments: Vec<McpPromptArgument>,
}

/// Text returned from one resource read or prompt, before it reaches a model.
const MAX_MCP_TEXT: usize = 64 * 1024;
const MAX_MCP_LISTED: usize = 1024;

fn default_schema() -> Value {
    json!({"type":"object","properties":{}})
}

pub struct McpClient {
    child: Child,
    input: Option<ChildStdin>,
    responses: mpsc::Receiver<std::result::Result<Value, String>>,
    stop: Arc<AtomicBool>,
    reader_error: Arc<Mutex<Option<String>>>,
    stderr: Arc<Mutex<Vec<u8>>>,
    next_id: u64,
    timeout: Duration,
    pub server_info: Value,
    pub protocol_version: String,
    pub tools: Vec<McpTool>,
    pub resources: Vec<McpResource>,
    pub prompts: Vec<McpPrompt>,
    closed: bool,
    cancel: Option<Arc<AtomicBool>>,
}

impl McpWebSearchConfig {
    pub fn validate(&self) -> Result<()> {
        validate_mcp_name(&self.tool, "MCP web-search tool")?;
        validate_mcp_name(&self.query_argument, "MCP web-search query argument")?;
        if let Some(argument) = &self.limit_argument {
            validate_mcp_name(argument, "MCP web-search limit argument")?;
        }
        if self.extra_arguments.len() > 32 {
            bail!("MCP web-search mapping has too many fixed arguments");
        }
        if self.extra_arguments.contains_key(&self.query_argument)
            || self
                .limit_argument
                .as_ref()
                .is_some_and(|name| self.extra_arguments.contains_key(name))
        {
            bail!("MCP web-search fixed arguments cannot replace query or limit arguments");
        }
        if serde_json::to_vec(&self.extra_arguments)?.len() > 16 * 1024 {
            bail!("MCP web-search fixed arguments exceed 16 KiB");
        }
        if !(1..=16_384).contains(&self.max_query_chars) {
            bail!("MCP web-search max_query_chars must be between 1 and 16384");
        }
        if self
            .max_query_words
            .is_some_and(|limit| !(1..=2_048).contains(&limit))
        {
            bail!("MCP web-search max_query_words must be between 1 and 2048");
        }
        if !(1..=100).contains(&self.max_results) {
            bail!("MCP web-search max_results must be between 1 and 100");
        }
        // Option arguments may share a name with a fixed argument (the fixed value
        // is then the default), but never with the query or the limit.
        for (argument, label) in [
            (&self.time_range_argument, "time-range"),
            (&self.language_argument, "language"),
            (&self.region_argument, "region"),
            (&self.safe_search_argument, "safe-search"),
        ] {
            if let Some(name) = argument {
                validate_mcp_name(name, &format!("MCP web-search {label} argument"))?;
                if *name == self.query_argument || self.limit_argument.as_ref() == Some(name) {
                    bail!("MCP web-search {label} argument cannot replace the query or limit argument");
                }
            }
        }
        for (values, allowed, label) in [
            (&self.time_range_values, &crate::integrations::TIME_RANGES[..], "time_range_values"),
            (&self.safe_search_values, &crate::integrations::SAFE_SEARCH_LEVELS[..], "safe_search_values"),
        ] {
            if let Some(key) = values.keys().find(|k| !allowed.contains(&k.as_str())) {
                bail!("MCP web-search {label} has unknown key {key:?}; use {}", allowed.join(", "));
            }
            if serde_json::to_vec(values)?.len() > 4 * 1024 {
                bail!("MCP web-search {label} exceed 4 KiB");
            }
        }
        if let Some(images) = &self.image_search {
            if let Some(tool) = &images.tool {
                validate_mcp_name(tool, "MCP image-search tool")?;
            }
            if images.extra_arguments.len() > 16
                || serde_json::to_vec(&images.extra_arguments)?.len() > 4 * 1024
            {
                bail!("MCP image-search arguments exceed 16 entries or 4 KiB");
            }
            if images.extra_arguments.contains_key(&self.query_argument)
                || self
                    .limit_argument
                    .as_ref()
                    .is_some_and(|name| images.extra_arguments.contains_key(name))
            {
                bail!("MCP image-search arguments cannot replace the query or limit argument");
            }
        }
        Ok(())
    }

    /// The tool and arguments for a web or image search, and which standard
    /// options the arguments carry.
    pub fn request(
        &self,
        query: &str,
        limit: usize,
        options: &crate::integrations::SearchOptions,
        images: bool,
    ) -> Result<(String, Value, Vec<&'static str>)> {
        let (mut arguments, applied) = self.arguments_with(query, limit, options)?;
        if !images {
            return Ok((self.tool.clone(), arguments, applied));
        }
        let config = self
            .image_search
            .as_ref()
            .context("This search server is not set up for image search")?;
        if let Some(object) = arguments.as_object_mut() {
            for (key, value) in &config.extra_arguments {
                object.insert(key.clone(), value.clone());
            }
        }
        Ok((config.tool.clone().unwrap_or_else(|| self.tool.clone()), arguments, applied))
    }

    pub fn arguments(&self, query: &str, limit: usize) -> Result<Value> {
        Ok(self
            .arguments_with(query, limit, &crate::integrations::SearchOptions::default())?
            .0)
    }

    /// Tool arguments for a search, and which standard options they carry.
    pub fn arguments_with(
        &self,
        query: &str,
        limit: usize,
        options: &crate::integrations::SearchOptions,
    ) -> Result<(Value, Vec<&'static str>)> {
        self.validate()?;
        let chars = query.chars().count();
        if chars == 0 || chars > self.max_query_chars {
            bail!(
                "MCP web-search query must contain 1 to {} characters; received {chars}",
                self.max_query_chars
            );
        }
        if let Some(max_words) = self.max_query_words {
            let words = query.split_whitespace().count();
            if words > max_words {
                bail!(
                    "MCP web-search query must contain at most {max_words} words; received {words}"
                );
            }
        }
        if limit == 0 || limit > self.max_results {
            bail!(
                "MCP web-search result count must be between 1 and {}; received {limit}",
                self.max_results
            );
        }
        let mut arguments = serde_json::Map::new();
        for (key, value) in &self.extra_arguments {
            arguments.insert(key.clone(), value.clone());
        }
        arguments.insert(self.query_argument.clone(), Value::String(query.to_owned()));
        if let Some(name) = &self.limit_argument {
            arguments.insert(name.clone(), json!(limit));
        }
        let mut applied = Vec::new();
        // A value of null marks a word the provider does not support: nothing
        // is sent and the option is reported as not applied.
        let mapped = |values: &BTreeMap<String, Value>, word: &str| match values.get(word) {
            Some(Value::Null) => None,
            Some(value) => Some(value.clone()),
            None => Some(json!(word)),
        };
        if let (Some(name), Some(range)) = (&self.time_range_argument, &options.time_range) {
            if let Some(value) = mapped(&self.time_range_values, range) {
                arguments.insert(name.clone(), value);
                applied.push("time_range");
            }
        }
        let mut language_used = false;
        if let (Some(name), Some(code)) = (&self.language_argument, options.language_code()) {
            arguments.insert(name.clone(), json!(code));
            language_used = true;
        }
        if let (Some(name), Some(region)) = (&self.region_argument, options.region()) {
            arguments.insert(name.clone(), json!(region));
            language_used = true;
        }
        if language_used {
            applied.push("language");
        }
        if let (Some(name), Some(level)) = (&self.safe_search_argument, &options.safe_search) {
            if let Some(value) = mapped(&self.safe_search_values, level) {
                arguments.insert(name.clone(), value);
                applied.push("safe_search");
            }
        }
        Ok((Value::Object(arguments), applied))
    }
}

impl McpServerConfig {
    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() || self.id.len() > 128 {
            bail!("MCP server needs a short nonempty ID");
        }
        if self.name.trim().is_empty() || self.name.len() > 256 {
            bail!("MCP server needs a short nonempty display name");
        }
        if let Some(search) = &self.web_search {
            search.validate()?;
        }
        if let Some(preset) = search_preset(&self.id) {
            if self.enabled
                && self
                    .env
                    .get(preset.key_env)
                    .is_none_or(|key| key.trim().is_empty())
            {
                bail!("{} MCP is enabled but {} is empty", preset.provider, preset.key_env);
            }
        }
        crate::integrations::CommandSpec {
            program: self.program.clone(),
            args: self.args.clone(),
            env: self.env.clone(),
            ..Default::default()
        }
        .validate()
    }

    /// Disabled, reproducible template for Brave's official stdio MCP server.
    /// The user must provide a Brave Search API key and explicitly enable it.
    pub fn brave_search_preset() -> Self {
        Self {
            id: BRAVE_SEARCH_SERVER_ID.into(),
            name: "Brave Search (official MCP)".into(),
            program: "npx".into(),
            args: vec![
                "-y".into(),
                BRAVE_SEARCH_PACKAGE.into(),
                "--transport".into(),
                "stdio".into(),
            ],
            cwd: None,
            env: BTreeMap::from([
                ("BRAVE_API_KEY".into(), String::new()),
                ("BRAVE_MCP_ENABLED_TOOLS".into(), "brave_web_search".into()),
            ]),
            enabled: false,
            timeout_secs: 90,
            web_search: Some(McpWebSearchConfig {
                tool: "brave_web_search".into(),
                query_argument: "query".into(),
                limit_argument: Some("count".into()),
                extra_arguments: BTreeMap::from([("safesearch".into(), json!("moderate"))]),
                max_query_chars: 400,
                max_query_words: Some(50),
                max_results: 10,
                // Brave's own parameter names and values.
                time_range_argument: Some("freshness".into()),
                time_range_values: BTreeMap::from([
                    ("day".into(), json!("pd")),
                    ("week".into(), json!("pw")),
                    ("month".into(), json!("pm")),
                    ("year".into(), json!("py")),
                ]),
                language_argument: Some("search_lang".into()),
                region_argument: Some("country".into()),
                safe_search_argument: Some("safesearch".into()),
                safe_search_values: BTreeMap::new(),
                image_search: None,
            }),
        }
    }

    /// The preset with this ID, disabled and without a key.
    pub fn search_preset(id: &str) -> Option<Self> {
        match id {
            BRAVE_SEARCH_SERVER_ID => Some(Self::brave_search_preset()),
            EXA_SEARCH_SERVER_ID => Some(Self::exa_search_preset()),
            KAGI_SEARCH_SERVER_ID => Some(Self::kagi_search_preset()),
            _ => None,
        }
    }

    /// Exa's official stdio MCP server (`web_search_exa`: query and numResults).
    pub fn exa_search_preset() -> Self {
        Self {
            id: EXA_SEARCH_SERVER_ID.into(),
            name: "Exa Search (official MCP)".into(),
            program: "npx".into(),
            args: vec!["-y".into(), EXA_SEARCH_PACKAGE.into()],
            cwd: None,
            env: BTreeMap::from([("EXA_API_KEY".into(), String::new())]),
            enabled: false,
            timeout_secs: 90,
            web_search: Some(McpWebSearchConfig {
                tool: "web_search_exa".into(),
                query_argument: "query".into(),
                limit_argument: Some("numResults".into()),
                extra_arguments: BTreeMap::new(),
                max_query_chars: 2_000,
                max_query_words: None,
                max_results: 10,
                ..Default::default()
            }),
        }
    }

    /// Kagi's official MCP server (`kagi_search_fetch`), run with uvx and asked
    /// for JSON results. Kagi filters by day, week or month, not by year.
    pub fn kagi_search_preset() -> Self {
        Self {
            id: KAGI_SEARCH_SERVER_ID.into(),
            name: "Kagi Search (official MCP)".into(),
            program: "uvx".into(),
            args: vec![KAGI_SEARCH_PACKAGE.into()],
            cwd: None,
            env: BTreeMap::from([
                ("KAGI_API_KEY".into(), String::new()),
                ("KAGI_SEARCH_FORMAT".into(), "json".into()),
            ]),
            enabled: false,
            timeout_secs: 90,
            web_search: Some(McpWebSearchConfig {
                tool: "kagi_search_fetch".into(),
                query_argument: "query".into(),
                limit_argument: Some("limit".into()),
                extra_arguments: BTreeMap::new(),
                max_query_chars: 1_000,
                max_query_words: None,
                max_results: 20,
                time_range_argument: Some("time_relative".into()),
                time_range_values: BTreeMap::from([
                    ("day".into(), json!("day")),
                    ("week".into(), json!("week")),
                    ("month".into(), json!("month")),
                    ("year".into(), Value::Null),
                ]),
                image_search: Some(McpImageSearchConfig {
                    tool: None,
                    extra_arguments: BTreeMap::from([("workflow".into(), json!("images"))]),
                }),
                ..Default::default()
            }),
        }
    }
}

fn validate_mcp_name(value: &str, label: &str) -> Result<()> {
    if value.trim().is_empty() || value.len() > 256 || value.chars().any(char::is_control) {
        bail!("{label} must be a nonempty name without control characters");
    }
    Ok(())
}

impl McpClient {
    pub fn connect(config: &McpServerConfig, default_cwd: &Path) -> Result<Self> {
        Self::connect_cancellable(config, default_cwd, None)
    }

    pub fn connect_cancellable(
        config: &McpServerConfig,
        default_cwd: &Path,
        cancel: Option<Arc<AtomicBool>>,
    ) -> Result<Self> {
        config.validate()?;
        if !config.enabled {
            bail!("MCP server {} is not enabled", config.name);
        }
        if cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
            bail!("MCP connection was cancelled before launch");
        }
        let mut command = Command::new(&config.program);
        command
            .args(&config.args)
            .current_dir(config.cwd.as_deref().unwrap_or(default_cwd))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // Do not silently hand all application API tokens to a subprocess.
        command.env_clear();
        for key in [
            "PATH",
            "HOME",
            "USER",
            "LANG",
            "LC_ALL",
            "TMPDIR",
            "XDG_RUNTIME_DIR",
            "SYSTEMROOT",
            "COMSPEC",
        ] {
            if let Some(value) = std::env::var_os(key) {
                command.env(key, value);
            }
        }
        command.envs(&config.env);
        crate::integrations::configure_process_group(&mut command);
        let mut child = command
            .spawn()
            .with_context(|| format!("Could not start MCP executable {}", config.program))?;
        let input = child.stdin.take().context("Missing MCP stdin")?;
        let stdout = child.stdout.take().context("Missing MCP stdout")?;
        let stderr_pipe = child.stderr.take().context("Missing MCP stderr")?;
        #[cfg(unix)]
        {
            if let Err(e) = crate::integrations::nonblocking(&input)
                .and_then(|_| crate::integrations::nonblocking(&stdout))
                .and_then(|_| crate::integrations::nonblocking(&stderr_pipe))
            {
                crate::integrations::kill_process_group(&mut child);
                let _ = child.wait();
                return Err(e);
            }
        }
        let (sender, responses) = mpsc::sync_channel(64);
        let stop = Arc::new(AtomicBool::new(false));
        let reader_error = Arc::new(Mutex::new(None));
        let stdout_stop = stop.clone();
        let error_copy = reader_error.clone();
        thread::spawn(move || {
            if let Err(error) = read_messages(stdout, sender, stdout_stop) {
                if let Ok(mut dest) = error_copy.lock() {
                    *dest = Some(error.to_string());
                }
            }
        });
        let stderr = Arc::new(Mutex::new(Vec::new()));
        let stderr_copy = stderr.clone();
        let stderr_stop = stop.clone();
        thread::spawn(move || capture_stderr(stderr_pipe, stderr_copy, stderr_stop));
        let mut client = Self {
            child,
            input: Some(input),
            responses,
            stop,
            reader_error,
            stderr,
            next_id: 0,
            timeout: Duration::from_secs(config.timeout_secs.clamp(1, 600)),
            server_info: Value::Null,
            protocol_version: String::new(),
            tools: Vec::new(),
            resources: Vec::new(),
            prompts: Vec::new(),
            closed: false,
            cancel,
        };
        let initialized = client.request(
            "initialize",
            json!({
                "protocolVersion": SUPPORTED_PROTOCOLS[0], "capabilities": {},
                "clientInfo": {"name":"ollama-workbench","version":env!("CARGO_PKG_VERSION")}
            }),
        )?;
        let version = initialized
            .get("protocolVersion")
            .and_then(Value::as_str)
            .context("MCP initialization omitted protocolVersion")?;
        if !SUPPORTED_PROTOCOLS.contains(&version) {
            bail!("Unsupported MCP protocol version {version}");
        }
        client.protocol_version = version.to_owned();
        client.server_info = initialized
            .get("serverInfo")
            .cloned()
            .unwrap_or(Value::Null);
        client.notify("notifications/initialized", json!({}))?;
        if initialized.pointer("/capabilities/tools").is_some() {
            client.refresh_tools()?;
        }
        // Resources and prompts are optional extras: a server that lists them
        // badly still connects, with its tools.
        if initialized.pointer("/capabilities/resources").is_some() {
            let _ = client.refresh_resources();
        }
        if initialized.pointer("/capabilities/prompts").is_some() {
            let _ = client.refresh_prompts();
        }
        Ok(client)
    }

    pub fn is_connected(&mut self) -> bool {
        if self.closed || self.stop.load(Ordering::Relaxed) {
            return false;
        }
        match self.child.try_wait() {
            Ok(None) => true,
            Ok(Some(_)) | Err(_) => {
                self.closed = true;
                self.input.take();
                self.stop.store(true, Ordering::Relaxed);
                false
            }
        }
    }

    pub fn refresh_tools(&mut self) -> Result<Vec<McpTool>> {
        let mut cursor: Option<String> = None;
        let mut seen_cursors = HashSet::new();
        let mut names = HashSet::new();
        let mut tools = Vec::new();
        for _ in 0..32 {
            let params = cursor
                .as_ref()
                .map(|v| json!({"cursor":v}))
                .unwrap_or_else(|| json!({}));
            let result = self.request("tools/list", params)?;
            let batch = result
                .get("tools")
                .and_then(Value::as_array)
                .context("MCP tools/list lacks tools array")?;
            for value in batch {
                let tool: McpTool =
                    serde_json::from_value(value.clone()).context("Invalid MCP tool definition")?;
                if tool.name.is_empty() || tool.name.len() > 256 || !names.insert(tool.name.clone())
                {
                    bail!("MCP tools/list returned an empty, duplicate or overlong name");
                }
                tools.push(tool);
                if tools.len() > 1024 {
                    bail!("MCP server exceeds 1024 tool limit");
                }
            }
            cursor = result
                .get("nextCursor")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_owned);
            match &cursor {
                None => {
                    self.tools = tools.clone();
                    return Ok(tools);
                }
                Some(value) if !seen_cursors.insert(value.clone()) => {
                    bail!("MCP pagination cursor repeated")
                }
                _ => {}
            }
        }
        bail!("MCP tool list exceeded 32 pages")
    }

    /// Every page of a `*/list` method, bounded in pages and items.
    fn list_all(&mut self, method: &str, key: &str) -> Result<Vec<Value>> {
        let mut cursor: Option<String> = None;
        let mut seen = HashSet::new();
        let mut items = Vec::new();
        for _ in 0..32 {
            let params = cursor
                .as_ref()
                .map(|v| json!({"cursor":v}))
                .unwrap_or_else(|| json!({}));
            let result = self.request(method, params)?;
            let batch = result
                .get(key)
                .and_then(Value::as_array)
                .with_context(|| format!("MCP {method} lacks a {key} array"))?;
            items.extend(batch.iter().cloned());
            if items.len() > MAX_MCP_LISTED {
                bail!("MCP {method} exceeds {MAX_MCP_LISTED} items");
            }
            cursor = result
                .get("nextCursor")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_owned);
            match &cursor {
                None => return Ok(items),
                Some(value) if !seen.insert(value.clone()) => {
                    bail!("MCP pagination cursor repeated")
                }
                _ => {}
            }
        }
        bail!("MCP {method} exceeded 32 pages")
    }

    pub fn refresh_resources(&mut self) -> Result<Vec<McpResource>> {
        let resources = self
            .list_all("resources/list", "resources")?
            .into_iter()
            .map(|v| serde_json::from_value(v).context("Invalid MCP resource"))
            .collect::<Result<Vec<McpResource>>>()?;
        self.resources = resources.clone();
        Ok(resources)
    }

    pub fn refresh_prompts(&mut self) -> Result<Vec<McpPrompt>> {
        let prompts = self
            .list_all("prompts/list", "prompts")?
            .into_iter()
            .map(|v| serde_json::from_value(v).context("Invalid MCP prompt"))
            .collect::<Result<Vec<McpPrompt>>>()?;
        self.prompts = prompts.clone();
        Ok(prompts)
    }

    /// Read an advertised resource. Text is bounded; binary contents are
    /// described, not returned.
    pub fn read_resource(&mut self, uri: &str) -> Result<Value> {
        if !self.resources.iter().any(|r| r.uri == uri) {
            bail!("Resource {uri} was not advertised by this MCP server; use mcp_list to see its resources");
        }
        let result = self.request("resources/read", json!({ "uri": uri }))?;
        let contents = result
            .get("contents")
            .and_then(Value::as_array)
            .context("MCP resources/read lacks a contents array")?;
        let mut budget = MAX_MCP_TEXT;
        let items: Vec<Value> = contents
            .iter()
            .map(|item| {
                let uri = item.get("uri").cloned().unwrap_or(Value::Null);
                let mime = item.get("mimeType").cloned().unwrap_or(Value::Null);
                if let Some(text) = item.get("text").and_then(Value::as_str) {
                    let kept: String = text.chars().take(budget).collect();
                    budget = budget.saturating_sub(kept.chars().count());
                    let truncated = kept.len() < text.len();
                    json!({"uri":uri,"mimeType":mime,"text":kept,"truncated":truncated})
                } else {
                    let bytes = item.get("blob").and_then(Value::as_str).map_or(0, |b| b.len() * 3 / 4);
                    json!({"uri":uri,"mimeType":mime,"binary_bytes":bytes,"note":"Binary content is not shown to the model"})
                }
            })
            .collect();
        Ok(json!({ "contents": items }))
    }

    /// Fill an advertised prompt template with string arguments.
    pub fn get_prompt(
        &mut self,
        name: &str,
        arguments: &serde_json::Map<String, Value>,
    ) -> Result<Value> {
        let prompt = self
            .prompts
            .iter()
            .find(|p| p.name == name)
            .with_context(|| format!("Prompt {name} was not advertised by this MCP server"))?;
        for argument in prompt.arguments.iter().filter(|a| a.required) {
            if !arguments.contains_key(&argument.name) {
                bail!("Prompt {name} requires the argument {}", argument.name);
            }
        }
        if arguments.values().any(|v| !v.is_string()) {
            bail!("MCP prompt arguments must be strings");
        }
        let mut result = self.request("prompts/get", json!({"name":name,"arguments":arguments}))?;
        let text = serde_json::to_string(&result)?;
        if text.len() > MAX_MCP_TEXT {
            result = json!({"truncated": true, "text": text.chars().take(MAX_MCP_TEXT).collect::<String>()});
        }
        Ok(result)
    }

    pub fn call_tool(&mut self, name: &str, arguments: Value) -> Result<Value> {
        if !arguments.is_object() {
            bail!("MCP tool arguments must be an object");
        }
        if !self.tools.iter().any(|t| t.name == name) {
            bail!("Tool {name} was not advertised by this MCP server");
        }
        // An isError result is still a valid MCP result; retain it for the caller.
        self.request("tools/call", json!({"name":name,"arguments":arguments}))
    }

    pub fn call_tool_cancellable(
        &mut self,
        name: &str,
        arguments: Value,
        cancel: Arc<AtomicBool>,
    ) -> Result<Value> {
        let previous = self.cancel.replace(cancel);
        let result = self.call_tool(name, arguments);
        self.cancel = previous;
        result
    }

    pub fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        if self.closed {
            bail!("MCP connection is closed");
        }
        self.next_id += 1;
        let id = self.next_id;
        let deadline = Instant::now() + self.timeout;
        if let Err(error) = self.send_json(
            &json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}),
            deadline,
        ) {
            self.shutdown();
            return Err(error);
        }
        loop {
            if self
                .cancel
                .as_ref()
                .is_some_and(|c| c.load(Ordering::Relaxed))
            {
                self.shutdown();
                bail!("MCP {method} cancelled; server stopped");
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                let _ = self.notify(
                    "notifications/cancelled",
                    json!({"requestId":id,"reason":"Request deadline exceeded"}),
                );
                self.shutdown();
                bail!("MCP {method} timed out; connection closed");
            }
            match self
                .responses
                .recv_timeout(remaining.min(Duration::from_millis(100)))
            {
                Ok(Ok(message)) => {
                    if message.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
                        self.shutdown();
                        bail!("MCP server sent an invalid JSON-RPC version");
                    }
                    if let Some(request_method) = message.get("method").and_then(Value::as_str) {
                        if let Some(request_id) = message.get("id") {
                            let response = if request_method == "ping" {
                                json!({"jsonrpc":"2.0","id":request_id,"result":{}})
                            } else {
                                // Sampling, roots and elicitation were not advertised.
                                json!({"jsonrpc":"2.0","id":request_id,"error":{"code":-32601,"message":"Client capability not supported"}})
                            };
                            self.send_json(&response, deadline)?;
                        }
                        continue;
                    }
                    if message.get("id").and_then(Value::as_u64) != Some(id) {
                        continue;
                    }
                    if let Some(error) = message.get("error") {
                        bail!(
                            "MCP {method} failed: {}",
                            error
                                .get("message")
                                .and_then(Value::as_str)
                                .unwrap_or("JSON-RPC error")
                        );
                    }
                    return message
                        .get("result")
                        .cloned()
                        .context("MCP response omitted result and error");
                }
                Ok(Err(error)) => {
                    self.shutdown();
                    bail!("MCP transport failed: {error}");
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    let error = self.reader_error.lock().ok().and_then(|mut e| e.take());
                    if let Some(error) = error {
                        self.shutdown();
                        bail!("MCP transport failed: {error}");
                    }
                    if self.child.try_wait()?.is_some() {
                        self.shutdown();
                        bail!("MCP server exited before replying to {method}");
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    let error = self
                        .reader_error
                        .lock()
                        .ok()
                        .and_then(|mut e| e.take())
                        .unwrap_or_else(|| "end of stream".into());
                    self.shutdown();
                    bail!("MCP stdout closed: {error}");
                }
            }
        }
    }

    pub fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        self.send_json(
            &json!({"jsonrpc":"2.0","method":method,"params":params}),
            Instant::now() + self.timeout,
        )
    }

    fn send_json(&mut self, value: &Value, deadline: Instant) -> Result<()> {
        let mut bytes = serde_json::to_vec(value)?;
        if bytes.len() > MAX_MESSAGE_BYTES {
            bail!("MCP message exceeds 2 MiB");
        }
        bytes.push(b'\n');
        let input = self.input.as_mut().context("MCP input stream is closed")?;
        let mut written = 0;
        while written < bytes.len() {
            if self
                .cancel
                .as_ref()
                .is_some_and(|c| c.load(Ordering::Relaxed))
            {
                bail!("MCP write was cancelled");
            }
            if Instant::now() >= deadline {
                bail!("MCP write timed out");
            }
            match input.write(&bytes[written..]) {
                Ok(0) => bail!("MCP input stream closed"),
                Ok(n) => written += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(5))
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }

    pub fn stderr_tail(&self) -> String {
        self.stderr
            .lock()
            .map(|v| String::from_utf8_lossy(&v).into_owned())
            .unwrap_or_default()
    }

    pub fn shutdown(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        self.input.take();
        self.stop.store(true, Ordering::Relaxed);
        let deadline = Instant::now() + Duration::from_millis(100);
        while Instant::now() < deadline {
            if self.child.try_wait().ok().flatten().is_some() {
                crate::integrations::kill_process_group(&mut self.child);
                return;
            }
            thread::sleep(Duration::from_millis(5));
        }
        #[cfg(unix)]
        unsafe {
            libc::kill(-(self.child.id() as i32), libc::SIGTERM);
        }
        let deadline = Instant::now() + Duration::from_millis(100);
        while Instant::now() < deadline {
            if self.child.try_wait().ok().flatten().is_some() {
                crate::integrations::kill_process_group(&mut self.child);
                return;
            }
            thread::sleep(Duration::from_millis(5));
        }
        crate::integrations::kill_process_group(&mut self.child);
        let _ = self.child.wait();
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn read_messages<R: Read>(
    mut pipe: R,
    tx: mpsc::SyncSender<std::result::Result<Value, String>>,
    stop: Arc<AtomicBool>,
) -> Result<()> {
    let mut pending = Vec::new();
    let mut buffer = [0u8; 8192];
    while !stop.load(Ordering::Relaxed) {
        match pipe.read(&mut buffer) {
            Ok(0) => {
                if !pending.is_empty() {
                    bail!("MCP stdout ended mid-message");
                }
                return Ok(());
            }
            Ok(n) => {
                pending.extend_from_slice(&buffer[..n]);
                while let Some(pos) = pending.iter().position(|b| *b == b'\n') {
                    if pos > MAX_MESSAGE_BYTES {
                        bail!("MCP response exceeds 2 MiB");
                    }
                    let bytes: Vec<u8> = pending.drain(..=pos).collect();
                    if bytes.iter().all(u8::is_ascii_whitespace) {
                        continue;
                    }
                    let parsed: Value = serde_json::from_slice(&bytes)
                        .context("MCP stdout contains invalid JSON")?;
                    if tx.try_send(Ok(parsed)).is_err() {
                        bail!("MCP server flooded the response queue or disconnected");
                    }
                }
                if pending.len() > MAX_MESSAGE_BYTES {
                    bail!("MCP response exceeds 2 MiB");
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(5))
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

fn capture_stderr<R: Read>(mut pipe: R, output: Arc<Mutex<Vec<u8>>>, stop: Arc<AtomicBool>) {
    let mut buffer = [0u8; 4096];
    while !stop.load(Ordering::Relaxed) {
        match pipe.read(&mut buffer) {
            Ok(0) => break,
            Ok(n) => {
                if let Ok(mut bytes) = output.lock() {
                    bytes.extend_from_slice(&buffer[..n]);
                    let excess = bytes.len().saturating_sub(8192);
                    if excess > 0 {
                        bytes.drain(..excess);
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(5))
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exa_and_kagi_presets_build_their_providers_requests() {
        let options = |range: &str| crate::integrations::SearchOptions { time_range: Some(range.into()), ..Default::default() };
        let exa = McpServerConfig::exa_search_preset();
        assert_eq!((exa.program.as_str(), exa.args.as_slice()), ("npx", &["-y".to_string(), EXA_SEARCH_PACKAGE.to_string()][..]));
        let exa_search = exa.web_search.unwrap();
        let (tool, arguments, applied) = exa_search.request("rust", 5, &options("week"), false).unwrap();
        assert_eq!((tool.as_str(), arguments), ("web_search_exa", json!({"query":"rust","numResults":5})));
        assert!(applied.is_empty(), "Exa's search tool has no date filter");
        assert!(exa_search.request("fox", 5, &Default::default(), true).is_err(), "Exa is not set up for images");

        let kagi = McpServerConfig::kagi_search_preset();
        assert_eq!((kagi.program.as_str(), kagi.env.get("KAGI_SEARCH_FORMAT").map(String::as_str)), ("uvx", Some("json")));
        let kagi_search = kagi.web_search.unwrap();
        let (_, arguments, applied) = kagi_search.request("rust", 5, &options("week"), false).unwrap();
        assert_eq!(arguments, json!({"query":"rust","limit":5,"time_relative":"week"}));
        assert_eq!(applied, vec!["time_range"]);
        // Kagi has no "year" filter: nothing is sent and the option is not applied.
        let (_, arguments, applied) = kagi_search.request("rust", 5, &options("year"), false).unwrap();
        assert_eq!(arguments, json!({"query":"rust","limit":5}));
        assert!(applied.is_empty());
        let (tool, arguments, _) = kagi_search.request("red fox", 4, &Default::default(), true).unwrap();
        assert_eq!((tool.as_str(), arguments), ("kagi_search_fetch", json!({"query":"red fox","limit":4,"workflow":"images"})));
    }

    #[test]
    fn every_preset_starts_disabled_and_needs_its_key_when_enabled() {
        for preset in SEARCH_PRESETS {
            let mut server = McpServerConfig::search_preset(preset.id).unwrap();
            assert!(!server.enabled);
            assert!(server.validate().is_ok(), "{}", preset.id);
            assert_eq!(server.program, preset.launcher);
            server.enabled = true;
            let error = server.validate().unwrap_err().to_string();
            assert!(error.contains(preset.key_env), "{error}");
            server.env.insert(preset.key_env.into(), "key".into());
            assert!(server.validate().is_ok());
        }
        assert!(McpServerConfig::search_preset("unknown").is_none());
        let defaults: Vec<McpServerConfig> = serde_json::from_str(&crate::settings::Settings::default().mcp_json).unwrap();
        assert_eq!(defaults.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(), vec![BRAVE_SEARCH_SERVER_ID, EXA_SEARCH_SERVER_ID, KAGI_SEARCH_SERVER_ID]);
        assert!(defaults.iter().all(|s| !s.enabled));
    }

    #[test]
    fn brave_preset_maps_standard_options_to_brave_parameters() {
        let mapping = McpServerConfig::brave_search_preset().web_search.unwrap();
        let options = crate::integrations::SearchOptions {
            time_range: Some("week".into()),
            language: Some("en-GB".into()),
            safe_search: Some("strict".into()),
        };
        let (arguments, applied) = mapping.arguments_with("rust", 5, &options).unwrap();
        assert_eq!(
            arguments,
            json!({"query":"rust","count":5,"freshness":"pw","search_lang":"en","country":"GB","safesearch":"strict"})
        );
        assert_eq!(applied, vec!["time_range", "language", "safe_search"]);
        // Without options the fixed default stays, and nothing is reported as applied.
        let (arguments, applied) = mapping
            .arguments_with("rust", 5, &crate::integrations::SearchOptions::default())
            .unwrap();
        assert_eq!(arguments, json!({"query":"rust","count":5,"safesearch":"moderate"}));
        assert!(applied.is_empty());
        assert_eq!(mapping.arguments("rust", 5).unwrap(), arguments);
    }

    #[test]
    fn option_mappings_cannot_replace_the_query_or_use_unknown_values() {
        let base = McpWebSearchConfig { tool: "search".into(), ..Default::default() };
        let replaces_query = McpWebSearchConfig { time_range_argument: Some("query".into()), ..base.clone() };
        assert!(replaces_query.validate().is_err());
        let replaces_limit = McpWebSearchConfig { safe_search_argument: Some("count".into()), ..base.clone() };
        assert!(replaces_limit.validate().is_err());
        let unknown = McpWebSearchConfig {
            time_range_argument: Some("recency".into()),
            time_range_values: BTreeMap::from([("fortnight".into(), json!("2w"))]),
            ..base.clone()
        };
        assert!(unknown.validate().is_err());
        // An unmapped option is simply not applied.
        let options = crate::integrations::SearchOptions { time_range: Some("day".into()), ..Default::default() };
        let (arguments, applied) = base.arguments_with("q", 3, &options).unwrap();
        assert_eq!(arguments, json!({"query":"q","count":3}));
        assert!(applied.is_empty());
        // Older saved mappings without the new fields still load.
        let legacy: McpWebSearchConfig = serde_json::from_value(json!({"tool":"search","query_argument":"q"})).unwrap();
        assert!(legacy.time_range_argument.is_none() && legacy.validate().is_ok());
    }

    #[test]
    #[cfg(unix)]
    fn stdio_handshake_list_and_call() {
        let script = r#"import sys,json
for line in sys.stdin:
 r=json.loads(line)
 if 'id' not in r: continue
 m=r['method']
 if m=='initialize': result={'protocolVersion':r['params']['protocolVersion'],'capabilities':{'tools':{}},'serverInfo':{'name':'test','version':'1'}}
 elif m=='tools/list': result={'tools':[{'name':'echo','description':'echo','inputSchema':{'type':'object'}}]}
 elif m=='tools/call': result={'content':[{'type':'text','text':r['params']['arguments']['text']}]}
 else: result={}
 print(json.dumps({'jsonrpc':'2.0','id':r['id'],'result':result}),flush=True)
"#;
        let config = McpServerConfig {
            id: "fixture".into(),
            name: "Fixture".into(),
            program: "python3".into(),
            args: vec!["-u".into(), "-c".into(), script.into()],
            cwd: None,
            env: BTreeMap::new(),
            enabled: true,
            timeout_secs: 3,
            web_search: None,
        };
        let mut client = McpClient::connect(&config, Path::new("/tmp")).unwrap();
        assert_eq!(client.tools[0].name, "echo");
        let result = client.call_tool("echo", json!({"text":"hello"})).unwrap();
        assert_eq!(result["content"][0]["text"], "hello");
        assert!(client.call_tool("not-listed", json!({})).is_err());
        client.shutdown();
        assert!(!client.is_connected());
    }

    #[test]
    #[cfg(unix)]
    fn resources_and_prompts_are_listed_read_and_bounded() {
        let script = r#"import sys,json
for line in sys.stdin:
 r=json.loads(line)
 if 'id' not in r: continue
 m=r['method']; p=r.get('params',{})
 if m=='initialize': result={'protocolVersion':p['protocolVersion'],'capabilities':{'resources':{},'prompts':{}},'serverInfo':{'name':'t','version':'1'}}
 elif m=='resources/list' and 'cursor' not in p: result={'resources':[{'uri':'file:///a.txt','name':'a','mimeType':'text/plain'}],'nextCursor':'2'}
 elif m=='resources/list': result={'resources':[{'uri':'mem://big','name':'big'},{'uri':'mem://img','name':'img'}]}
 elif m=='resources/read' and p['uri']=='mem://big': result={'contents':[{'uri':p['uri'],'text':'x'*200000}]}
 elif m=='resources/read' and p['uri']=='mem://img': result={'contents':[{'uri':p['uri'],'mimeType':'image/png','blob':'AAAA'}]}
 elif m=='resources/read': result={'contents':[{'uri':p['uri'],'mimeType':'text/plain','text':'hello'}]}
 elif m=='prompts/list': result={'prompts':[{'name':'review','arguments':[{'name':'file','required':True}]}]}
 elif m=='prompts/get': result={'messages':[{'role':'user','content':{'type':'text','text':'Review '+p['arguments']['file']}}]}
 else: result={}
 print(json.dumps({'jsonrpc':'2.0','id':r['id'],'result':result}),flush=True)
"#;
        let config = McpServerConfig {
            id: "fixture".into(),
            name: "Fixture".into(),
            program: "python3".into(),
            args: vec!["-u".into(), "-c".into(), script.into()],
            cwd: None,
            env: BTreeMap::new(),
            enabled: true,
            timeout_secs: 3,
            web_search: None,
        };
        let mut client = McpClient::connect(&config, Path::new("/tmp")).unwrap();
        assert!(client.tools.is_empty());
        let uris: Vec<_> = client.resources.iter().map(|r| r.uri.as_str()).collect();
        assert_eq!(uris, ["file:///a.txt", "mem://big", "mem://img"]);
        assert_eq!(
            client.read_resource("file:///a.txt").unwrap()["contents"][0]["text"],
            "hello"
        );
        let big = client.read_resource("mem://big").unwrap();
        assert_eq!(big["contents"][0]["truncated"], true);
        assert_eq!(
            big["contents"][0]["text"].as_str().unwrap().len(),
            MAX_MCP_TEXT
        );
        let image = client.read_resource("mem://img").unwrap();
        assert!(
            image["contents"][0].get("blob").is_none() && image["contents"][0]["binary_bytes"] == 3
        );
        assert!(
            client.read_resource("file:///etc/passwd").is_err(),
            "unadvertised URI"
        );

        let mut args = serde_json::Map::new();
        assert!(
            client.get_prompt("review", &args).is_err(),
            "missing required argument"
        );
        args.insert("file".into(), json!("main.rs"));
        let prompt = client.get_prompt("review", &args).unwrap();
        assert_eq!(prompt["messages"][0]["content"]["text"], "Review main.rs");
        assert!(client.get_prompt("other", &args).is_err());
        client.shutdown();
    }

    #[test]
    fn disabled_server_never_starts() {
        let config = McpServerConfig {
            id: "disabled".into(),
            name: "Disabled".into(),
            program: "missing-binary".into(),
            args: Vec::new(),
            cwd: None,
            env: BTreeMap::new(),
            enabled: false,
            timeout_secs: 1,
            web_search: None,
        };
        assert!(McpClient::connect(&config, Path::new("/tmp"))
            .err()
            .unwrap()
            .to_string()
            .contains("not enabled"));
    }

    #[test]
    #[cfg(unix)]
    fn handshake_timeout_stops_child() {
        let config = McpServerConfig {
            id: "timeout".into(),
            name: "Timeout".into(),
            program: "/bin/sleep".into(),
            args: vec!["10".into()],
            cwd: None,
            env: BTreeMap::new(),
            enabled: true,
            timeout_secs: 1,
            web_search: None,
        };
        let start = Instant::now();
        assert!(McpClient::connect(&config, Path::new("/tmp")).is_err());
        assert!(start.elapsed() < Duration::from_secs(4));
    }

    #[test]
    fn brave_search_preset_is_disabled_pinned_and_mapped() {
        let preset = McpServerConfig::brave_search_preset();
        assert_eq!(preset.id, BRAVE_SEARCH_SERVER_ID);
        assert!(!preset.enabled);
        assert!(preset.args.iter().any(|arg| arg == BRAVE_SEARCH_PACKAGE));
        assert_eq!(
            preset.env.get("BRAVE_API_KEY").map(String::as_str),
            Some("")
        );
        assert!(preset
            .args
            .windows(2)
            .any(|pair| pair[0] == "--transport" && pair[1] == "stdio"));
        assert_eq!(
            preset
                .env
                .get("BRAVE_MCP_ENABLED_TOOLS")
                .map(String::as_str),
            Some("brave_web_search")
        );
        let search = preset.web_search.as_ref().unwrap();
        assert_eq!(search.tool, "brave_web_search");
        assert_eq!(search.arguments("linux", 5).unwrap()["query"], "linux");
        assert_eq!(search.arguments("linux", 5).unwrap()["count"], 5);
        assert_eq!(
            search.arguments("linux", 5).unwrap()["safesearch"],
            "moderate"
        );
        preset.validate().unwrap();
    }

    #[test]
    fn enabled_brave_preset_requires_an_api_key() {
        let mut preset = McpServerConfig::brave_search_preset();
        preset.enabled = true;
        assert!(preset
            .validate()
            .unwrap_err()
            .to_string()
            .contains("BRAVE_API_KEY"));
        preset
            .env
            .insert("BRAVE_API_KEY".into(), "fixture-key".into());
        preset.validate().unwrap();
    }

    #[test]
    fn older_mcp_json_without_search_mapping_remains_compatible() {
        let config: McpServerConfig = serde_json::from_value(json!({
            "id":"legacy",
            "name":"Legacy",
            "program":"legacy-mcp",
            "enabled":false,
            "timeout_secs":60
        }))
        .unwrap();
        assert!(config.web_search.is_none());
        config.validate().unwrap();
    }

    #[test]
    fn older_search_mapping_json_gets_safe_limits() {
        let mapping: McpWebSearchConfig = serde_json::from_value(json!({
            "tool":"search",
            "query_argument":"q",
            "limit_argument":"n",
            "extra_arguments":{}
        }))
        .unwrap();
        assert_eq!(mapping.max_query_chars, 2_048);
        assert_eq!(mapping.max_query_words, None);
        assert_eq!(mapping.max_results, 10);
        mapping.validate().unwrap();
    }

    #[test]
    fn web_search_mapping_rejects_argument_collisions() {
        let mapping = McpWebSearchConfig {
            tool: "search".into(),
            query_argument: "query".into(),
            limit_argument: Some("limit".into()),
            extra_arguments: BTreeMap::from([("query".into(), json!("forged"))]),
            ..Default::default()
        };
        assert!(mapping.validate().is_err());
    }

    #[test]
    fn brave_search_mapping_enforces_provider_query_bounds() {
        let mapping = McpServerConfig::brave_search_preset().web_search.unwrap();
        assert!(mapping.arguments(&"x".repeat(400), 10).is_ok());
        assert!(mapping.arguments(&"x".repeat(401), 10).is_err());
        assert!(mapping.arguments(&vec!["word"; 50].join(" "), 10).is_ok());
        assert!(mapping.arguments(&vec!["word"; 51].join(" "), 10).is_err());
        assert!(mapping.arguments("linux", 11).is_err());
    }

    #[test]
    #[cfg(unix)]
    fn exited_server_is_not_reported_as_connected() {
        let script = r#"import json,sys
for line in sys.stdin:
 r=json.loads(line)
 if 'id' not in r: continue
 if r['method']=='initialize':
  result={'protocolVersion':r['params']['protocolVersion'],'capabilities':{'tools':{}},'serverInfo':{'name':'exit-fixture','version':'1'}}
 elif r['method']=='tools/list':
  result={'tools':[]}
 else:
  result={}
 print(json.dumps({'jsonrpc':'2.0','id':r['id'],'result':result}),flush=True)
 if r['method']=='tools/list': break
"#;
        let config = McpServerConfig {
            id: "exit-fixture".into(),
            name: "Exit Fixture".into(),
            program: "python3".into(),
            args: vec!["-u".into(), "-c".into(), script.into()],
            cwd: None,
            env: BTreeMap::new(),
            enabled: true,
            timeout_secs: 3,
            web_search: None,
        };
        let mut client = McpClient::connect(&config, Path::new("/tmp")).unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while client.is_connected() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        assert!(!client.is_connected());
    }
}
