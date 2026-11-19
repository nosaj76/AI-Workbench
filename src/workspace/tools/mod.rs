//! Workspace tools with concrete, one-use approvals and optimistic file checks.
//! Model-generated arguments are data, never authority to approve an operation.

mod checkpoints;
mod checks;
pub use checkpoints::{Checkpoint, CheckpointList, CheckpointState};
pub use checks::CheckMode;
mod execution;
mod filesystem;
mod git;
mod matching;
mod preparation;
mod reading;
mod terminal;

use crate::files::Workspace;
use crate::integrations::{self, CommandAdapter, CommandSpec};
use crate::mcp::{McpClient, McpServerConfig, McpWebSearchConfig};
use anyhow::{bail, Context, Result};
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

const MAX_FILE_BYTES: usize = 16 * 1024 * 1024;
const MAX_WRITE_BYTES: usize = 2 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PendingAction {
    pub id: String,
    pub title: String,
    pub description: String,
    pub details: Value,
    pub operation: Operation,
    pub workspace: PathBuf,
    pub expected: Vec<PathState>,
}

/// What kind of risk an action carries, for automatic approval settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalKind {
    /// Sends a query or fetches a page; changes nothing locally.
    Web,
    /// Changes files in the project (backed up and checked afterwards).
    FileChange,
    /// Runs a program inside the command sandbox.
    Command,
    /// Fetches from this computer or the local network (loopback, LAN, link-local
    /// metadata). Always asks: a web page cannot steer the model into local
    /// services without the user seeing the address.
    LocalNetwork,
    /// Runs a program or contacts an external service with the user's own
    /// permissions: unsandboxed commands and terminals, MCP calls, adapters
    /// and external agents. Automatic approval of sandboxed commands never
    /// covers these.
    UnsandboxedCommand,
}

impl Operation {
    pub fn approval_kind(&self) -> ApprovalKind {
        match self {
            Operation::FetchUrl { local: true, .. } => ApprovalKind::LocalNetwork,
            Operation::FetchUrl { .. }
            | Operation::WebSearch { .. }
            | Operation::McpWebSearch { .. }
            | Operation::TextBrowserSearch { .. }
            | Operation::KnowledgeSearch { .. }
            | Operation::QaLookup { .. } => ApprovalKind::Web,
            Operation::Write { .. }
            | Operation::Replace { .. }
            | Operation::Copy { .. }
            | Operation::Move { .. }
            | Operation::Delete { .. }
            | Operation::MakeDirectory { .. }
            | Operation::GitCommit { .. } => ApprovalKind::FileChange,
            Operation::Command { spec } if spec.sandbox.is_some() => ApprovalKind::Command,
            Operation::TerminalOpen { config, .. } if config.sandbox.is_some() => {
                ApprovalKind::Command
            }
            Operation::TerminalWrite { sandboxed, .. }
            | Operation::TerminalKey { sandboxed, .. }
                if *sandboxed =>
            {
                ApprovalKind::Command
            }
            // Resizing or closing a terminal the AI may use runs no new code.
            Operation::TerminalResize { .. } | Operation::TerminalClose { .. } => {
                ApprovalKind::Command
            }
            Operation::Command { .. }
            | Operation::TerminalOpen { .. }
            | Operation::TerminalWrite { .. }
            | Operation::TerminalKey { .. }
            | Operation::McpConnect { .. }
            | Operation::McpCall { .. }
            | Operation::ExternalAgent { .. }
            | Operation::Adapter { .. } => ApprovalKind::UnsandboxedCommand,
        }
    }
}

#[derive(Debug, Clone)]
pub enum PreparedTool {
    Immediate(Value),
    Approval(Box<PendingAction>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Operation {
    Write {
        path: String,
        content: String,
    },
    Replace {
        path: String,
        find: String,
        replacement: String,
        replace_all: bool,
    },
    Copy {
        source: String,
        destination: String,
    },
    Move {
        source: String,
        destination: String,
    },
    Delete {
        path: String,
    },
    MakeDirectory {
        path: String,
    },
    GitCommit {
        message: String,
        paths: Vec<String>,
        branch: Option<String>,
    },
    Command {
        spec: CommandSpec,
    },
    TerminalOpen {
        config: crate::terminal::TerminalConfig,
        wait_ms: u64,
        max_bytes: usize,
    },
    TerminalWrite {
        session_id: String,
        text: String,
        enter: bool,
        wait_ms: u64,
        max_bytes: usize,
        /// The session runs in the sandbox (fixed for a session's lifetime).
        sandboxed: bool,
    },
    TerminalKey {
        session_id: String,
        key: String,
        wait_ms: u64,
        max_bytes: usize,
        sandboxed: bool,
    },
    TerminalResize {
        session_id: String,
        rows: u16,
        cols: u16,
    },
    TerminalClose {
        session_id: String,
    },
    FetchUrl {
        url: String,
        max_chars: usize,
        /// The approved destination is loopback, LAN or another private address.
        /// A fetch approved as public may only connect to public addresses.
        local: bool,
    },
    WebSearch {
        query: String,
        base_url: String,
        limit: usize,
        /// Text browsers (name, program) approved to retry the query through
        /// DuckDuckGo if SearXNG fails.
        #[serde(default)]
        fallbacks: Vec<(String, String)>,
        /// Home server whose local-site index is searched too (approved in the preview).
        #[serde(default)]
        local_sites: Option<String>,
        #[serde(default)]
        options: Box<integrations::SearchOptions>,
        /// Search SearXNG's image category instead of the web.
        #[serde(default)]
        images: bool,
    },
    McpWebSearch {
        server_id: String,
        tool_name: String,
        query: String,
        limit: usize,
        arguments: Value,
        config_hash: String,
        #[serde(default)]
        local_sites: Option<String>,
        #[serde(default)]
        options: Box<integrations::SearchOptions>,
        /// Standard options the mapped arguments carry.
        #[serde(default)]
        applied_options: Vec<String>,
        #[serde(default)]
        images: bool,
    },
    TextBrowserSearch {
        browser: String,
        program: String,
        query: String,
        limit: usize,
        /// Other installed browsers (name, program) to retry the same query
        /// with if the first one is refused or fails. Shown in the preview.
        fallbacks: Vec<(String, String)>,
        /// Public knowledge sources the same query is sent to if every browser
        /// fails. Empty when they are not enabled. Shown in the preview.
        knowledge_fallback: Vec<String>,
        #[serde(default)]
        options: Box<integrations::SearchOptions>,
        #[serde(default)]
        local_sites: Option<String>,
    },
    /// A search of public knowledge sources (Wikipedia, arXiv, ...).
    KnowledgeSearch {
        query: String,
        sources: Vec<String>,
        limit: usize,
        #[serde(default)]
        options: Box<integrations::SearchOptions>,
    },
    QaLookup {
        question: String,
        /// Browsers (name, program) the approved lookup may use, in order.
        browsers: Vec<(String, String)>,
        /// The SearXNG server searched first, when one is configured.
        #[serde(default)]
        searxng: Option<String>,
        /// How the calling model expects the answer to age, if it said.
        kind: Option<crate::qa::AnswerKind>,
    },
    McpConnect {
        config: Box<McpServerConfig>,
    },
    McpCall {
        server_id: String,
        tool_name: String,
        arguments: Value,
        config_hash: String,
    },
    ExternalAgent {
        request: Box<crate::external_agents::PreparedConsultation>,
    },
    Adapter {
        adapter_id: String,
        spec: CommandSpec,
        output: Option<String>,
        config_hash: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PathState {
    #[serde(default)]
    pub mode: Option<u32>,
    pub path: String,
    pub resolved: PathBuf,
    pub kind: String,
    pub sha256: Option<String>,
    pub size: u64,
}

#[derive(Debug, Clone)]
enum SearchProvider {
    Searxng {
        base_url: String,
    },
    TextBrowser {
        browser: integrations::TextBrowser,
    },
    Mcp {
        config: Box<McpServerConfig>,
        mapping: Box<McpWebSearchConfig>,
        connected: bool,
    },
    Knowledge {
        sources: Vec<String>,
    },
}

/// How the engine isolates commands and terminals the model proposes.
#[derive(Clone, Debug, Default)]
pub struct SandboxSettings {
    pub mode: crate::sandbox::SandboxMode,
    pub network: bool,
    pub read_only: Vec<PathBuf>,
}

#[derive(Clone)]
pub struct ToolEngine {
    workspace: Arc<Workspace>,
    sandbox: Arc<Mutex<SandboxSettings>>,
    terminals: crate::terminal::TerminalManager,
    unread_terminal_output: Arc<Mutex<HashMap<String, u64>>>,
    backup_root: PathBuf,
    pending: Arc<Mutex<HashMap<String, PendingAction>>>,
    network_enabled: Arc<AtomicBool>,
    /// Whether public knowledge sources may be used as a search provider.
    knowledge_search: Arc<AtomicBool>,
    search_depth: Arc<Mutex<String>>,
    knowledge_endpoints: Arc<Mutex<crate::knowledge_search::Endpoints>>,
    workspace_tools_enabled: Arc<AtomicBool>,
    search_url: Arc<Mutex<Option<String>>>,
    /// Home server whose search-index.json web searches also read (local sites).
    local_sites_url: Arc<Mutex<Option<String>>>,
    text_browser: Arc<Mutex<Option<integrations::TextBrowser>>>,
    /// Override for the browsers a failed text-browser search may retry with;
    /// None means every installed browser.
    text_browser_fallbacks: Arc<Mutex<Option<Vec<integrations::TextBrowser>>>>,
    /// Where the Q&A cache lives; None uses the application data directory.
    qa_cache_dir: Arc<Mutex<Option<PathBuf>>>,
    mcp_configs: Arc<Mutex<BTreeMap<String, McpServerConfig>>>,
    mcp_clients: Arc<Mutex<HashMap<String, Arc<Mutex<McpClient>>>>>,
    adapters: Arc<Mutex<BTreeMap<String, CommandAdapter>>>,
    external_agents: Arc<Mutex<BTreeMap<String, crate::external_agents::AgentProfile>>>,
    external_runs: PathBuf,
    /// What to run after approved file changes (in the sandbox only).
    check_mode: Arc<Mutex<CheckMode>>,
}

impl ToolEngine {
    pub fn new(workspace: Workspace) -> Self {
        let root_hash = hash(workspace.root().to_string_lossy().as_bytes());
        let backup_root = crate::settings::data_dir()
            .unwrap_or_else(|_| workspace.root().join(".ollama-workbench"))
            .join("backups")
            .join(&root_hash[..16]);
        Self::with_backup_dir(workspace, backup_root)
    }

    pub fn with_backup_dir(workspace: Workspace, backup_root: PathBuf) -> Self {
        Self {
            workspace: Arc::new(workspace),
            // Unit tests and bare engines run commands directly; the desktop
            // configures the sandbox from Settings (auto by default).
            sandbox: Arc::new(Mutex::new(SandboxSettings {
                mode: crate::sandbox::SandboxMode::Off,
                ..Default::default()
            })),
            terminals: crate::terminal::TerminalManager::default(),
            unread_terminal_output: Arc::new(Mutex::new(HashMap::new())),
            external_runs: backup_root.join("external-agent-runs"),
            check_mode: Arc::new(Mutex::new(CheckMode::Off)),
            backup_root,
            pending: Arc::new(Mutex::new(HashMap::new())),
            network_enabled: Arc::new(AtomicBool::new(false)),
            knowledge_search: Arc::new(AtomicBool::new(false)),
            search_depth: Arc::new(Mutex::new("balanced".into())),
            knowledge_endpoints: Arc::new(
                Mutex::new(crate::knowledge_search::Endpoints::default()),
            ),
            workspace_tools_enabled: Arc::new(AtomicBool::new(true)),
            search_url: Arc::new(Mutex::new(None)),
            local_sites_url: Arc::new(Mutex::new(None)),
            text_browser: Arc::new(Mutex::new(None)),
            text_browser_fallbacks: Arc::new(Mutex::new(None)),
            qa_cache_dir: Arc::new(Mutex::new(None)),
            mcp_configs: Arc::new(Mutex::new(BTreeMap::new())),
            mcp_clients: Arc::new(Mutex::new(HashMap::new())),
            adapters: Arc::new(Mutex::new(BTreeMap::new())),
            external_agents: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    pub fn root(&self) -> &Path {
        self.workspace.root()
    }
    /// Share the selected workspace's sessions with its visible terminal panel.
    /// Cloning this handle does not launch or close a process.
    pub fn terminal_manager(&self) -> crate::terminal::TerminalManager {
        self.terminals.clone()
    }

    pub fn set_terminal_manager(&mut self, manager: crate::terminal::TerminalManager) {
        self.terminals = manager;
    }
    pub fn backup_directory(&self) -> &Path {
        &self.backup_root
    }
    pub fn set_network_enabled(&self, enabled: bool) {
        self.network_enabled.store(enabled, Ordering::Relaxed);
    }

    /// Allows public knowledge sources as a search provider and as the
    /// disclosed fallback when every text browser fails. Off by default.
    pub fn set_knowledge_search(&self, enabled: bool) {
        self.knowledge_search.store(enabled, Ordering::Relaxed);
    }

    /// "speed", "balanced" or "quality": sets how many results a search returns
    /// when the model does not ask for a number.
    pub fn set_search_depth(&self, depth: &str) {
        if let Ok(mut current) = self.search_depth.lock() {
            *current = depth.to_owned();
        }
    }

    fn default_search_limit(&self) -> usize {
        self.search_depth
            .lock()
            .map(|depth| crate::settings::search_depth_limit(&depth))
            .unwrap_or(8)
    }

    /// An image search goes to an MCP server set up for images, or to SearXNG.
    /// DuckDuckGo through a text browser and the knowledge sources cannot search images.
    fn image_search_provider(&self, requested: Option<&str>) -> Result<SearchProvider> {
        let configs: Vec<McpServerConfig> = self
            .mcp_configs
            .lock()
            .map_err(|_| anyhow::anyhow!("MCP configuration lock failed"))?
            .values()
            .filter(|config| {
                config.enabled
                    && config.web_search.as_ref().is_some_and(|s| s.image_search.is_some())
            })
            .cloned()
            .collect();
        let mut ordered: Vec<&McpServerConfig> = configs.iter().collect();
        ordered.sort_by_key(|config| {
            (
                requested != Some(config.id.as_str()),
                !self.mcp_is_connected(&config.id).unwrap_or(false),
            )
        });
        if let Some(config) = ordered.first() {
            let mapping = config.web_search.clone().context("MCP web-search mapping was removed")?;
            return Ok(SearchProvider::Mcp {
                connected: self.mcp_is_connected(&config.id)?,
                config: Box::new((*config).clone()),
                mapping: Box::new(mapping),
            });
        }
        let searxng = self
            .search_url
            .lock()
            .map_err(|_| anyhow::anyhow!("Search configuration lock failed"))?
            .clone();
        if let Some(base_url) = searxng {
            return Ok(SearchProvider::Searxng { base_url });
        }
        bail!("Image search needs SearXNG or an MCP search server set up for images (the Kagi preset). DuckDuckGo through a text browser and the public knowledge sources cannot search images")
    }

    /// Points the knowledge sources somewhere else; for tests.
    pub fn set_knowledge_endpoints(&self, endpoints: crate::knowledge_search::Endpoints) {
        if let Ok(mut current) = self.knowledge_endpoints.lock() {
            *current = endpoints;
        }
    }

    fn knowledge_enabled(&self) -> bool {
        self.knowledge_search.load(Ordering::Relaxed)
    }

    /// The knowledge provider for a mode, if sources are allowed.
    fn knowledge_provider(&self, mode: &str) -> Result<SearchProvider> {
        let sources = crate::knowledge_search::mode_sources(mode).with_context(|| {
            format!(
                "Unknown search mode {mode}; use one of {}",
                crate::knowledge_search::MODES.join(", ")
            )
        })?;
        self.check_knowledge_sources(&sources)?;
        Ok(SearchProvider::Knowledge { sources })
    }

    /// Public sources need "Allow public knowledge sources"; the user's own
    /// sites (Herb Helper, NOMAD, the Bible lexicon) need the home server's
    /// local-sites address, which turns all of them on or off together.
    pub(super) fn check_knowledge_sources(&self, sources: &[String]) -> Result<()> {
        let local = sources.iter().any(|s| crate::knowledge_search::is_local(s));
        let public = sources
            .iter()
            .any(|s| !crate::knowledge_search::is_local(s));
        if public && !self.knowledge_enabled() {
            bail!("Public knowledge sources are not enabled; turn on \"Allow public knowledge sources\" in Settings");
        }
        if local && self.local_sites_url()?.is_none() {
            bail!(
                "Your home server's sites are turned off; set the local sites address in Settings"
            );
        }
        Ok(())
    }

    /// Restrict this engine to calculation plus explicitly enabled web fetch
    /// and native web search. Mapped MCP configuration remains available only
    /// behind `web_search`; generic MCP, file, command, adapter, and external-
    /// agent tools stay hidden when no project is open.
    pub fn set_workspace_tools_enabled(&self, enabled: bool) {
        self.workspace_tools_enabled
            .store(enabled, Ordering::Relaxed);
    }

    fn workspace_tools_enabled(&self) -> bool {
        self.workspace_tools_enabled.load(Ordering::Relaxed)
    }
    pub fn set_sandbox(&self, settings: SandboxSettings) {
        if let Ok(mut current) = self.sandbox.lock() {
            *current = settings;
        }
    }

    pub(super) fn sandbox_is_off(&self) -> Result<bool> {
        Ok(self
            .sandbox
            .lock()
            .map_err(|_| anyhow::anyhow!("Sandbox settings lock failed"))?
            .mode
            == crate::sandbox::SandboxMode::Off)
    }

    /// The policy for the next model-proposed command or terminal, if any.
    pub(super) fn command_sandbox(&self) -> Result<Option<crate::sandbox::SandboxPolicy>> {
        let settings = self
            .sandbox
            .lock()
            .map_err(|_| anyhow::anyhow!("Sandbox settings lock failed"))?
            .clone();
        let policy = || {
            crate::sandbox::SandboxPolicy::for_workspace(
                self.root(),
                &settings.read_only,
                settings.network,
            )
        };
        Ok(match settings.mode {
            crate::sandbox::SandboxMode::Off => None,
            crate::sandbox::SandboxMode::Auto => crate::sandbox::available().then(policy),
            crate::sandbox::SandboxMode::Required => {
                if !crate::sandbox::available() {
                    bail!("The command sandbox is required but bubblewrap cannot create one here. Install bubblewrap (bwrap) or change Settings → Tools → Command sandbox. Nothing was run.");
                }
                Some(policy())
            }
        })
    }

    pub fn set_local_sites_url(&self, url: Option<String>) -> Result<()> {
        if let Some(url) = url.as_ref() {
            integrations::validate_http_url(url)?;
        }
        *self
            .local_sites_url
            .lock()
            .map_err(|_| anyhow::anyhow!("Local sites configuration lock failed"))? = url;
        Ok(())
    }
    fn local_sites_url(&self) -> Result<Option<String>> {
        Ok(self
            .local_sites_url
            .lock()
            .map_err(|_| anyhow::anyhow!("Local sites configuration lock failed"))?
            .clone()
            .filter(|url| !url.trim().is_empty()))
    }
    pub fn set_search_url(&self, url: Option<String>) -> Result<()> {
        if let Some(url) = url.as_ref() {
            integrations::validate_http_url(url)?;
        }
        *self
            .search_url
            .lock()
            .map_err(|_| anyhow::anyhow!("Search configuration lock failed"))? = url;
        Ok(())
    }

    /// Configure keyless search through a terminal text browser (lynx, w3m,
    /// elinks or links). `choice` is the `text_browser` setting.
    pub fn set_text_browser(&self, choice: &str) -> Result<()> {
        *self
            .text_browser
            .lock()
            .map_err(|_| anyhow::anyhow!("Search configuration lock failed"))? =
            integrations::resolve_text_browser(choice);
        Ok(())
    }

    /// Set an explicit browser executable (tests and diagnostics).
    pub fn set_text_browser_program(
        &self,
        browser: Option<integrations::TextBrowser>,
    ) -> Result<()> {
        *self
            .text_browser
            .lock()
            .map_err(|_| anyhow::anyhow!("Search configuration lock failed"))? = browser;
        Ok(())
    }

    /// Set the fallback browsers explicitly (tests and diagnostics).
    pub fn set_text_browser_fallbacks(
        &self,
        browsers: Option<Vec<integrations::TextBrowser>>,
    ) -> Result<()> {
        *self
            .text_browser_fallbacks
            .lock()
            .map_err(|_| anyhow::anyhow!("Search configuration lock failed"))? = browsers;
        Ok(())
    }

    /// Use an explicit Q&A cache directory (tests and isolated runs).
    pub fn set_qa_cache_dir(&self, dir: Option<PathBuf>) -> Result<()> {
        *self
            .qa_cache_dir
            .lock()
            .map_err(|_| anyhow::anyhow!("Q&A configuration lock failed"))? = dir;
        Ok(())
    }

    fn qa_cache(&self) -> Result<crate::qa::QaCache> {
        let configured = self
            .qa_cache_dir
            .lock()
            .map_err(|_| anyhow::anyhow!("Q&A configuration lock failed"))?
            .clone();
        let dir = match configured {
            Some(dir) => dir,
            None => crate::settings::data_dir()?,
        };
        crate::qa::QaCache::open(&dir)
    }

    /// A saved library answer in compact form for the chat model: the short
    /// answer and answer with source links, without the source excerpts that
    /// would use up model context. Excerpts are included only when no answer
    /// has been written yet.
    fn qa_saved_result(entry: &crate::qa::CachedQa, question: &str, note: &str) -> Value {
        let answered = !entry.answer.trim().is_empty();
        let sources: Vec<Value> = entry
            .sources
            .iter()
            .map(|s| {
                if answered {
                    json!({"id":s.id,"title":s.title,"url":s.url,"site":s.site})
                } else {
                    json!({"id":s.id,"title":s.title,"url":s.url,"site":s.site,"content":s.content})
                }
            })
            .collect();
        json!({
            "ok": true,
            "cached": true,
            "question": question,
            "saved_question": entry.question,
            "saved": crate::qa::short_date(&entry.saved_at),
            "kind": entry.kind.as_str(),
            "refresh": entry.refresh_after.as_deref().map(crate::qa::short_date).unwrap_or_else(|| "never".into()),
            "summary": entry.summary,
            "answer": entry.answer.chars().take(4_000).collect::<String>(),
            "sources": sources,
            "library_file": format!("{}/{}", crate::qa::LIBRARY_DIR, entry.path),
            "note": note,
        })
    }

    /// The configured text browser followed by the fallback browsers.
    fn qa_browsers(&self) -> Result<Vec<integrations::TextBrowser>> {
        let mut browsers: Vec<integrations::TextBrowser> =
            self.configured_text_browser()?.into_iter().collect();
        for other in self.text_browser_fallback_candidates()? {
            if !browsers.iter().any(|b| b.name == other.name) {
                browsers.push(other);
            }
        }
        Ok(browsers)
    }

    /// Browsers a failed text-browser search may retry with, in order.
    fn text_browser_fallback_candidates(&self) -> Result<Vec<integrations::TextBrowser>> {
        let configured = self
            .text_browser_fallbacks
            .lock()
            .map_err(|_| anyhow::anyhow!("Search configuration lock failed"))?
            .clone();
        Ok(configured.unwrap_or_else(integrations::installed_text_browsers))
    }

    fn configured_text_browser(&self) -> Result<Option<integrations::TextBrowser>> {
        Ok(self
            .text_browser
            .lock()
            .map_err(|_| anyhow::anyhow!("Search configuration lock failed"))?
            .clone())
    }

    fn has_search_provider(&self) -> bool {
        let has_searxng = self
            .search_url
            .lock()
            .map(|url| url.is_some())
            .unwrap_or(false);
        let has_mcp = self
            .mcp_configs
            .lock()
            .map(|configs| {
                configs
                    .values()
                    .any(|config| config.enabled && config.web_search.is_some())
            })
            .unwrap_or(false);
        let has_text_browser = self
            .text_browser
            .lock()
            .map(|browser| browser.is_some())
            .unwrap_or(false);
        has_searxng || has_mcp || has_text_browser || self.knowledge_enabled()
    }

    fn search_provider(&self, requested: Option<&str>) -> Result<SearchProvider> {
        let searxng = self
            .search_url
            .lock()
            .map_err(|_| anyhow::anyhow!("Search configuration lock failed"))?
            .clone();
        let configs: Vec<McpServerConfig> = self
            .mcp_configs
            .lock()
            .map_err(|_| anyhow::anyhow!("MCP configuration lock failed"))?
            .values()
            .filter(|config| config.enabled && config.web_search.is_some())
            .cloned()
            .collect();

        if requested.map(str::trim) == Some("knowledge") {
            return self.knowledge_provider("general");
        }
        if let Some(provider) = requested.map(str::trim).filter(|value| !value.is_empty()) {
            // Models invent provider IDs (e.g. an unconfigured "brave-search")
            // or pick a stopped SearXNG. Use automatic selection instead of
            // failing a search that an available provider can perform.
            if let Ok(chosen) = self.requested_search_provider(provider, searxng.clone(), &configs)
            {
                return Ok(chosen);
            }
        }
        self.automatic_search_provider(searxng, configs)
    }

    fn requested_search_provider(
        &self,
        provider: &str,
        searxng: Option<String>,
        configs: &[McpServerConfig],
    ) -> Result<SearchProvider> {
        if provider == "searxng" {
            let base_url = searxng.context("The requested SearXNG provider is not configured")?;
            if !integrations::endpoint_accepts_connections(
                &base_url,
                std::time::Duration::from_millis(800),
            ) {
                bail!("The requested SearXNG provider is not accepting connections");
            }
            return Ok(SearchProvider::Searxng { base_url });
        }
        if provider == "text_browser" || integrations::TEXT_BROWSERS.contains(&provider) {
            let browser = self.configured_text_browser()?.context(
                "No text browser (lynx, w3m, elinks or links) is installed and enabled for search",
            )?;
            if provider != "text_browser" && browser.name != provider {
                bail!(
                    "Text browser {provider} is not the configured search browser ({})",
                    browser.name
                );
            }
            return Ok(SearchProvider::TextBrowser { browser });
        }
        let config = configs
            .iter()
            .find(|config| config.id == provider)
            .cloned()
            .with_context(|| format!("Unknown or disabled web-search provider {provider}"))?;
        let mapping = config
            .web_search
            .clone()
            .context("MCP web-search mapping was removed")?;
        let connected = self.mcp_is_connected(&config.id)?;
        Ok(SearchProvider::Mcp {
            config: Box::new(config),
            mapping: Box::new(mapping),
            connected,
        })
    }

    fn automatic_search_provider(
        &self,
        searxng: Option<String>,
        configs: Vec<McpServerConfig>,
    ) -> Result<SearchProvider> {
        // Prefer an already-running MCP provider, then an enabled MCP provider,
        // then the direct SearXNG fallback. This avoids silently ignoring the
        // provider the user explicitly configured for web search.
        for config in &configs {
            if self.mcp_is_connected(&config.id)? {
                return Ok(SearchProvider::Mcp {
                    config: Box::new(config.clone()),
                    mapping: Box::new(
                        config
                            .web_search
                            .clone()
                            .context("MCP web-search mapping was removed")?,
                    ),
                    connected: true,
                });
            }
        }
        if let Some(config) = configs.into_iter().next() {
            let mapping = config
                .web_search
                .clone()
                .context("MCP web-search mapping was removed")?;
            return Ok(SearchProvider::Mcp {
                config: Box::new(config),
                mapping: Box::new(mapping),
                connected: false,
            });
        }
        let browser = self.configured_text_browser()?;
        if let Some(base_url) = searxng {
            // A configured SearXNG that is not running would make every search
            // fail. Fall back to the text browser before the approval preview,
            // so the user approves the provider that will actually be used.
            let reachable = integrations::endpoint_accepts_connections(
                &base_url,
                std::time::Duration::from_millis(800),
            );
            if reachable || browser.is_none() {
                return Ok(SearchProvider::Searxng { base_url });
            }
        }
        if let Some(browser) = browser {
            return Ok(SearchProvider::TextBrowser { browser });
        }
        if self.knowledge_enabled() {
            // Nothing else is set up: public knowledge sources still answer.
            return self.knowledge_provider("general");
        }
        bail!("No web-search provider is configured. Install a text browser (lynx or w3m), enable an MCP search server, set a SearXNG URL, or allow public knowledge sources in Settings");
    }

    fn mcp_is_connected(&self, id: &str) -> Result<bool> {
        let client = self
            .mcp_clients
            .lock()
            .map_err(|_| anyhow::anyhow!("MCP lock failed"))?
            .get(id)
            .cloned();
        match client {
            Some(client) => {
                let mut client = client
                    .lock()
                    .map_err(|_| anyhow::anyhow!("MCP client lock failed"))?;
                Ok(client.is_connected())
            }
            None => Ok(false),
        }
    }

    /// Registers configuration only. Does not start a process or contact a server.
    pub fn register_mcp(&self, config: McpServerConfig) -> Result<()> {
        config.validate()?;
        if let Some(client) = self
            .mcp_clients
            .lock()
            .map_err(|_| anyhow::anyhow!("MCP lock failed"))?
            .remove(&config.id)
        {
            if let Ok(mut client) = client.lock() {
                client.shutdown();
            }
        }
        self.mcp_configs
            .lock()
            .map_err(|_| anyhow::anyhow!("MCP configuration lock failed"))?
            .insert(config.id.clone(), config);
        Ok(())
    }

    pub fn register_adapter(&self, adapter: CommandAdapter) -> Result<()> {
        if adapter.id.is_empty() || adapter.id.len() > 128 {
            bail!("Adapter requires a short ID");
        }
        CommandSpec {
            program: adapter.program.clone(),
            args: adapter.args.clone(),
            ..Default::default()
        }
        .validate()?;
        self.adapters
            .lock()
            .map_err(|_| anyhow::anyhow!("Adapter lock failed"))?
            .insert(adapter.id.clone(), adapter);
        Ok(())
    }

    /// Configuration only: never authenticates, starts a CLI, or sends content.
    pub fn register_external_agent(
        &self,
        profile: crate::external_agents::AgentProfile,
    ) -> Result<()> {
        profile.validate()?;
        self.external_agents
            .lock()
            .map_err(|_| anyhow::anyhow!("External agent configuration lock failed"))?
            .insert(profile.id.clone(), profile);
        Ok(())
    }

    pub fn definitions(&self) -> Vec<Value> {
        let string = || json!({"type":"string"});
        let path =
            || json!({"type":"string","description":"Path relative to the selected workspace."});
        let mut tools = vec![definition(
            "calculator",
            "Evaluate finite floating-point arithmetic locally: + - * / % ^, parentheses, pi/e/tau and standard math functions. Trigonometric angles use radians. No code or shell execution.",
            json!({"expression":string()}),
            &["expression"],
        )];
        if self.workspace_tools_enabled() {
            tools.extend([
                definition("list_files", "List readable entries in a directory. Use offset and next_offset to page without assuming the first response covers everything. Paths are workspace-relative; protected files and symlinks are excluded.", json!({"path":path(),"limit":{"type":"integer","minimum":1,"maximum":1000},"offset":{"type":"integer","minimum":0,"maximum":20000}}), &[]),
                definition("project_tree", "List a bounded part of the project tree, excluding dependency/build folders. Follow next_offset or ask about a narrower subdirectory. This is discovery, NOT a complete source review.", json!({"path":path(),"max_depth":{"type":"integer","minimum":1,"maximum":12},"limit":{"type":"integer","minimum":1,"maximum":2000},"offset":{"type":"integer","minimum":0,"maximum":20000}}), &[]),
                definition("repo_map", "A ranked outline of the project's code: the files the rest of the code relies on most, each with its key functions, types and classes and their line numbers. Use it first on a large project to decide what to read. `focus` is text naming what you care about (file names and identifiers in it raise those files). The result states how many ranked files it shows and the `offset` to continue from; it is an outline, not file content, so read_file before describing code.", json!({"focus":string(),"path":path(),"max_chars":{"type":"integer","minimum":1500,"maximum":40000},"offset":{"type":"integer","minimum":0,"maximum":100000}}), &[]),
                definition("read_file", "Read a file as numbered lines. The whole file is returned whenever it fits in your context; only a file larger than that is split into the largest pieces that fit. The result states exactly what came back: returned_start_line..returned_end_line of total_lines, `complete`, and `next_start_line`. If next_start_line is set, call again with that start_line to continue (and start_column=next_column when it is given, because a very long line was only partly returned); never describe a file you only partly read as fully reviewed. Line numbers for extracted documents refer to the extracted text.", json!({"path":path(),"start_line":{"type":"integer","minimum":1},"end_line":{"type":"integer","minimum":1},"start_column":{"type":"integer","minimum":0,"description":"Resume inside a line that was only partly returned: pass next_column from the previous result together with the same start_line."},"expected_sha256":{"type":"string","description":"Pass source_sha256 from the previous page to detect that the file changed between pages."},"max_chars":{"type":"integer","minimum":1000,"maximum":4000000}}), &["path"]),
                definition("search_files", "Find text in readable workspace files. The query is LITERAL text unless you set regex:true (needed for | alternation or patterns); a literal query that looks like a regex returns a warning (honoring ignore files; the result lists the excluded folder names). Check `complete`: only complete=true with zero matches means the text is absent. If complete=false the search did NOT cover everything: call again with the same query and `resume_after` set to the returned value and include `resume_line` when provided to continue inside that file, or narrow `path`. Files that need an external extractor (PDF/scans) are counted but not searched; read them with read_file.", json!({"query":string(),"path":path(),"case_sensitive":{"type":"boolean"},"limit":{"type":"integer","minimum":1,"maximum":200},"resume_after":string(),"resume_line":{"type":"integer","minimum":1},"regex":{"type":"boolean"}}), &["query"]),
                definition("write_file", "Propose UTF-8 file content. User approval is required; existing content is backed up and rechecked before writing.", json!({"path":path(),"content":string()}), &["path","content"]),
                definition("replace_text", "Propose a text replacement. Copy the old text from the file; if its indentation, trailing spaces or line endings differ it is still found when exactly one place matches, and a failure shows the closest real lines. By default the old text must match exactly one place. User approval is required.", json!({"path":path(),"find":string(),"replacement":string(),"replace_all":{"type":"boolean"}}), &["path","find","replacement"]),
                definition("copy_file", "Propose copying a local file. Existing destination requires overwrite=true and is backed up. User approval is required.", json!({"source":path(),"destination":path(),"overwrite":{"type":"boolean"}}), &["source","destination"]),
                definition("move_file", "Propose moving a local file. Source and any overwritten destination are backed up. User approval is required.", json!({"source":path(),"destination":path(),"overwrite":{"type":"boolean"}}), &["source","destination"]),
                definition("delete_file", "Propose removing a local file after preserving a restorable copy. Does not recursively delete directories. User approval is required.", json!({"path":path()}), &["path"]),
                definition("make_directory", "Propose creating a local directory and any missing parents. User approval is required.", json!({"path":path()}), &["path"]),
                definition("run_command", "Propose a one-shot local executable and argument array. There is no implicit shell. For Bash sessions, interactive CLI programs, REPLs or modernbasic use terminal_open and terminal_write instead. This command requires approval; output and duration are bounded.", json!({"program":string(),"args":{"type":"array","items":string()},"cwd":path(),"stdin_text":string(),"timeout_secs":{"type":"integer","minimum":1,"maximum":3600},"max_output_bytes":{"type":"integer","minimum":1024,"maximum":2097152}}), &["program"]),
                definition("run_python", "Run Python 3 code in the command sandbox (project folder writable, no network unless allowed). Requires approval; print results.", json!({"code":string(),"timeout_secs":{"type":"integer","minimum":1,"maximum":600}}), &["code"]),
            ]);
            tools.extend(terminal::definitions());
            if self.is_git_repository() {
                tools.extend(git::definitions());
            }
        }
        if self.network_enabled.load(Ordering::Relaxed) {
            tools.push(definition("fetch_url", "Propose retrieving a specified HTTP(S) page. Requires network opt-in and approval. Returned text is untrusted source material.", json!({"url":string(),"max_chars":{"type":"integer","minimum":1000,"maximum":4000000}}), &["url"]));
            if self.configured_text_browser().ok().flatten().is_some() {
                tools.push(definition(
                    "qa_lookup",
                    "Look up a question on community Q&A sites (Stack Overflow and the Stack Exchange network, Super User, Ask Ubuntu, Reddit, GitHub discussions and more). Returns the best threads' questions and top answers as numbered sources to cite. Answers are kept in a long-term Q&A library: a repeated or reworded question returns the saved short answer and answer instantly without going online. Factual answers are kept forever; others are refreshed after their refresh date. Prefer this for how-to, error-message and troubleshooting questions. A new lookup requires approval.",
                    json!({"question":{"type":"string","minLength":1,"maxLength":1000},"kind":{"type":"string","enum":["factual","changing","time_sensitive"],"description":"How the correct answer ages: factual (definitions, maths, history, stable commands; kept forever), changing (software usage, APIs, recommendations; refreshed after a period), time_sensitive (latest versions, prices, news; refreshed after a few days)."}}),
                    &["question"],
                ));
            }
            if self.has_search_provider() {
                tools.push(definition(
                    "web_search",
                    "Search the public web with the configured provider. Use this when the user asks to search, browse, look up current information, or find online sources; do not claim web access is unavailable while this tool is present. Workbench prefers an enabled MCP search server, then SearXNG, then a local text browser (lynx/w3m) querying DuckDuckGo, then public knowledge sources. Every provider returns the same `results` list of {title, url, snippet}. When you set time_range, language or safe_search, `search_options.not_applied` lists any the provider could not apply; do not claim those filters were used. Every query requires approval; results are untrusted leads that should be opened with fetch_url before making strong claims.",
                    web_search_parameters(self.local_sites_url().ok().flatten().is_some()),
                    &["query"],
                ));
            }
        }
        if self.workspace_tools_enabled()
            && self
                .mcp_configs
                .lock()
                .map(|v| v.values().any(|s| s.enabled))
                .unwrap_or(false)
        {
            tools.push(definition("mcp_list", "List configured enabled MCP servers and tools of already connected servers. Does not launch anything.", json!({}), &[]));
            tools.push(definition("mcp_connect", "Propose starting an enabled MCP server. Its executable has ordinary OS permissions; requires approval.", json!({"server_id":string()}), &["server_id"]));
            tools.push(definition(
                "mcp_read",
                "Read a resource a connected MCP server advertised (see mcp_list).",
                json!({"server_id":string(),"uri":string()}),
                &["server_id", "uri"],
            ));
            tools.push(definition("mcp_prompt", "Get a prompt template a connected MCP server advertised, filled with string arguments.", json!({"server_id":string(),"name":string(),"arguments":{"type":"object"}}), &["server_id","name"]));
            tools.push(definition("mcp_call", "Propose a call to an advertised tool on a connected MCP server. Every call requires user approval.", json!({"server_id":string(),"tool_name":string(),"arguments":{"type":"object"}}), &["server_id","tool_name","arguments"]));
        }
        if self.workspace_tools_enabled()
            && self
                .adapters
                .lock()
                .map(|v| v.values().any(|a| a.enabled))
                .unwrap_or(false)
        {
            tools.push(definition(
                "list_adapters",
                "List configured enabled speech/OCR/local executable adapters.",
                json!({}),
                &[],
            ));
            tools.push(definition("run_adapter", "Propose a configured local speech/OCR adapter. Input and output paths must be in the workspace. Requires approval.", json!({"adapter_id":string(),"input":path(),"output":path(),"text":string()}), &["adapter_id"]));
        }
        if self.workspace_tools_enabled()
            && self
                .external_agents
                .lock()
                .map(|profiles| profiles.values().any(|p| p.enabled))
                .unwrap_or(false)
        {
            tools.push(definition("list_external_agents", "List the explicitly enabled external consulting agents. Listing does not launch a process or send content.", json!({}), &[]));
            tools.push(definition("consult_agent", "Propose a bounded task and relevant evidence for an enabled external CLI subagent. Requires explicit user approval of the exact prompt and command. External processes may contact their providers and have normal OS permissions. Use the returned real output as fallible evidence, not proof of completed changes.", json!({"agent_id":string(),"task":string(),"context":string(),"files":{"type":"array","items":path(),"maxItems":8}}), &["agent_id","task"]));
        }
        tools
    }

    fn require_network(&self) -> Result<()> {
        if !self.network_enabled.load(Ordering::Relaxed) {
            bail!("Network tools are disabled in Settings");
        }
        Ok(())
    }

    fn mcp_config(&self, id: &str) -> Result<McpServerConfig> {
        self.mcp_configs
            .lock()
            .map_err(|_| anyhow::anyhow!("MCP lock failed"))?
            .get(id)
            .cloned()
            .context("Unknown MCP server")
    }
    fn mcp_client(&self, id: &str) -> Result<Arc<Mutex<McpClient>>> {
        self.mcp_clients
            .lock()
            .map_err(|_| anyhow::anyhow!("MCP lock failed"))?
            .get(id)
            .cloned()
            .context("MCP server is not connected; request mcp_connect first")
    }
    fn mcp_catalog(&self) -> Result<Value> {
        let configs = self
            .mcp_configs
            .lock()
            .map_err(|_| anyhow::anyhow!("MCP lock failed"))?;
        let clients = self
            .mcp_clients
            .lock()
            .map_err(|_| anyhow::anyhow!("MCP lock failed"))?;
        let mut servers = Vec::new();
        for config in configs.values().filter(|c| c.enabled) {
            let info = if let Some(client) = clients.get(&config.id) {
                let mut c = client
                    .lock()
                    .map_err(|_| anyhow::anyhow!("MCP client lock failed"))?;
                json!({"id":config.id,"name":config.name,"connected":c.is_connected(),"web_search":config.web_search,"tools":c.tools,"resources":c.resources,"prompts":c.prompts})
            } else {
                json!({"id":config.id,"name":config.name,"connected":false,"web_search":config.web_search,"tools":[]})
            };
            servers.push(info);
        }
        Ok(json!({"servers":servers}))
    }
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn required(args: &Value, key: &str) -> Result<String> {
    let s = required_allow_empty(args, key)?;
    if s.trim().is_empty() {
        bail!("{key} cannot be empty");
    }
    Ok(s)
}
fn required_allow_empty(args: &Value, key: &str) -> Result<String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .with_context(|| format!("{key} must be a string"))
}
fn uint(args: &Value, key: &str, default: usize, min: usize, max: usize) -> usize {
    // Models sometimes send numbers as text ("start_line": "753"). Ignoring that
    // silently re-read a file from line 1.
    args.get(key)
        .and_then(|value| {
            value.as_u64().or_else(|| {
                value
                    .as_str()
                    .and_then(|text| text.trim().parse::<u64>().ok())
            })
        })
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or(default)
        .clamp(min, max)
}
fn ensure_file(state: &PathState) -> Result<()> {
    if state.kind != "file" {
        bail!("{} must be an ordinary file", state.path);
    }
    Ok(())
}
fn ensure_file_or_missing(state: &PathState) -> Result<()> {
    if !matches!(state.kind.as_str(), "file" | "missing") {
        bail!("{} must be a file path", state.path);
    }
    Ok(())
}
fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}
fn parse_string_array(value: Option<&Value>) -> Result<Vec<String>> {
    match value {
        None => Ok(Vec::new()),
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .context("Every argument must be a string")
            })
            .collect(),
        _ => bail!("args must be a JSON array of strings"),
    }
}

fn definition(name: &str, description: &str, properties: Value, required: &[&str]) -> Value {
    json!({"type":"function","function":{"name":name,"description":description,"parameters":{"type":"object","properties":properties,"required":required,"additionalProperties":false}}})
}

fn command_preview(spec: &CommandSpec) -> Value {
    json!({"program":spec.program,"args":spec.args,"command":spec.display_command(),"cwd":spec.cwd,"timeout_secs":spec.timeout_secs,"max_output_bytes":spec.max_output_bytes,"stdin_text":spec.stdin_text,"environment_keys":spec.env.keys().collect::<Vec<_>>(),"sandbox":spec.sandbox.as_ref().map(|p| p.summary()).unwrap_or(json!({"sandboxed":false}))})
}

/// `read_limited` for a file inside the workspace: on Linux it is opened beneath the
/// root through descriptors, so a swapped directory cannot redirect the read.
fn read_limited_beneath(root: &Path, path: &Path, limit: usize) -> Result<Vec<u8>> {
    let Ok(relative) = path.strip_prefix(root) else {
        return read_limited(path, limit);
    };
    let Some(file) = crate::confined::open_read(root, relative)? else {
        return read_limited(path, limit);
    };
    if file.metadata()?.len() > limit as u64 {
        bail!("File exceeds {} bytes", limit);
    }
    let mut bytes = Vec::new();
    file.take((limit + 1) as u64).read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        bail!("File grew beyond {} bytes while reading", limit);
    }
    Ok(bytes)
}

fn read_limited(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let meta = fs::symlink_metadata(path)?;
    if !meta.is_file() || meta.file_type().is_symlink() {
        bail!("Path is not an ordinary file: {}", path.display());
    }
    if meta.len() > limit as u64 {
        bail!("File exceeds {} bytes", limit);
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let mut bytes = Vec::new();
    options
        .open(path)?
        .take((limit + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        bail!("File grew beyond {} bytes while reading", limit);
    }
    Ok(bytes)
}

fn private_create(path: &Path) -> Result<fs::File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    Ok(options.open(path)?)
}

/// The `web_search` schema. The four self-hosted modes (Herb Helper, NOMAD,
/// the Bible lexicon, the Knowledge Cache) are offered only when a local sites
/// address is configured, so other installations never see them.
fn web_search_parameters(local_sites: bool) -> Value {
    let mut modes = vec!["general", "images", "encyclopedia", "academic", "tech"];
    let mut description = String::from("Optional. ");
    if local_sites {
        modes.extend(["herbal", "offline", "bible", "knowledge"]);
        description.push_str("herbal = the user's own Herb Helper site (herbs, formulas, evidence, safety, interactions and preparation) plus NOMAD's health libraries; use it FIRST for any question about herbs, herbal or natural remedies, supplements or their safety, and prefer it to the web. offline = the NOMAD offline library on the user's home server (Kiwix: offline Wikipedia, survival and water treatment, programming docs and any other installed libraries); use it for reference questions those libraries cover, or when the web is unavailable. bible = the user's Ethiopian Bible Research Lexicon (Ge'ez and related biblical texts, editions and review status). knowledge = the user's Knowledge Cache: sourced articles and dictionary definitions for topics from the user's past conversations, grouped by domain; check it first when a topic was discussed before, and still cite the article's own sources. These four are the user's own sites on this computer; queries never leave it. ");
    }
    description.push_str("images = pictures (each result has image_url and the page url; needs SearXNG or an image-capable MCP server); ");
    description.push_str("encyclopedia = Wikipedia (definitions, concepts, history); academic = arXiv and Crossref (papers and abstracts); tech = Hacker News and GitHub (tools, libraries, engineering discussion); general = the ordinary web search (the same as omitting mode). Use a specialised mode when a topic clearly fits it; for research and anything current, omit mode.");
    json!({
        "query":{"type":"string","minLength":1,"maxLength":2048,"description":"Concise search keywords, not the user's whole sentence: \"rename local git branch\" rather than \"how do I rename my local git branch?\"."},
        "limit":{"type":"integer","minimum":1,"maximum":20},
        "mode":{"type":"string","enum":modes,"description":description},
        "time_range":{"type":"string","enum":integrations::TIME_RANGES,"description":"Optional. Only pages from the past day, week, month or year; use it for news and other recent events."},
        "language":{"type":"string","description":"Optional language code, with a region when it matters: \"en\", \"de\", \"en-GB\", \"pt-BR\"."},
        "safe_search":{"type":"string","enum":integrations::SAFE_SEARCH_LEVELS,"description":"Optional explicit-content filter. Leave it out to use the provider's default."}
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn engine(dir: &Path, backup: &Path) -> ToolEngine {
        let engine = ToolEngine::with_backup_dir(
            Workspace::new(dir.to_path_buf()).unwrap(),
            backup.to_path_buf(),
        );
        // Hermetic: never fall back to the real lynx/w3m on this machine.
        engine.set_text_browser_fallbacks(Some(vec![])).unwrap();
        engine
    }

    #[cfg(unix)]
    fn fake_browser(dir: &Path, name: &str, page: &str) -> integrations::TextBrowser {
        use std::os::unix::fs::PermissionsExt;
        let program = dir.join(name);
        std::fs::write(&program, format!("#!/bin/sh\ncat <<'HTML'\n{page}\nHTML\n")).unwrap();
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
        integrations::TextBrowser {
            name: name.into(),
            program,
        }
    }

    #[test]
    #[cfg(unix)]
    fn qa_lookup_reads_threads_caches_them_and_reuses_the_cache_without_approval() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let bin = tempfile::tempdir().unwrap();
        let cache_dir = tempfile::tempdir().unwrap();
        let calls = bin.path().join("calls.txt");
        // Stand-in lynx: a DuckDuckGo results page for -source, a rendered
        // thread for -dump. Each invocation is logged.
        let program = bin.path().join("lynx");
        std::fs::write(
            &program,
            format!(
                "#!/bin/sh\necho \"$1\" >> '{}'\nif [ \"$1\" = -source ]; then cat <<'HTML'\n<div class=\"result\"><a class=\"result__a\" href=\"//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fcli%2Fcli%2Fdiscussions%2F42&amp;rut=1\">Undo a commit</a><a class=\"result__snippet\">Use git reset</a></div>\n<div class=\"result\"><a class=\"result__a\" href=\"//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fblog&amp;rut=1\">Blog</a></div>\nHTML\nelse\nprintf 'Answer: run git reset --soft HEAD~1 to undo the last commit and keep the changes staged. %.0s' $(seq 1 10)\nfi\n",
                calls.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
        let lynx = integrations::TextBrowser {
            name: "lynx".into(),
            program,
        };
        let e = engine(dir.path(), backups.path());
        e.set_qa_cache_dir(Some(cache_dir.path().to_path_buf()))
            .unwrap();
        e.set_text_browser_program(Some(lynx)).unwrap();
        assert!(
            !names(&e).iter().any(|n| n == "qa_lookup"),
            "network opt-in required"
        );
        e.set_network_enabled(true);
        assert!(names(&e).iter().any(|n| n == "qa_lookup"));

        let action = approval(
            e.prepare(
                "qa_lookup",
                &json!({"question":"How do I undo the last git commit?"}),
            )
            .unwrap(),
        );
        assert_eq!(action.details["browsers"], json!(["lynx"]));
        let result = e.approve(&action).unwrap();
        assert_eq!(result["cached"], false);
        let sources = result["sources"].as_array().unwrap();
        assert_eq!(sources.len(), 1, "only the Q&A thread is read: {sources:?}");
        assert_eq!(
            sources[0]["url"],
            "https://github.com/cli/cli/discussions/42"
        );
        assert_eq!(sources[0]["method"], "text_browser:lynx");
        assert!(sources[0]["content"]
            .as_str()
            .unwrap()
            .contains("git reset --soft"));
        let logged = std::fs::read_to_string(&calls).unwrap();
        assert_eq!(logged.lines().collect::<Vec<_>>(), vec!["-source", "-dump"]);

        // A reworded question comes from the cache: no approval, no browser.
        match e
            .prepare("qa_lookup", &json!({"question":"undo last git commit"}))
            .unwrap()
        {
            PreparedTool::Immediate(value) => {
                assert_eq!(value["cached"], true);
                assert_eq!(
                    value["sources"][0]["url"],
                    "https://github.com/cli/cli/discussions/42"
                );
            }
            PreparedTool::Approval(_) => panic!("a cached question must not need approval"),
        }
        assert_eq!(
            std::fs::read_to_string(&calls).unwrap(),
            logged,
            "browser ran again"
        );
    }

    #[test]
    #[cfg(unix)]
    fn refused_text_browser_search_retries_with_the_next_browser() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let bin = tempfile::tempdir().unwrap();
        let limited = r#"If this persists, please <a href="mailto:x@duckduckgo.com?subject=Error getting results">email us</a>."#;
        let results = r#"<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fw&amp;rut=1">From w3m</a></div>"#;
        let lynx = fake_browser(bin.path(), "lynx", limited);
        let w3m = fake_browser(bin.path(), "w3m", results);
        let e = engine(dir.path(), backups.path());
        e.set_network_enabled(true);
        e.set_text_browser_program(Some(lynx.clone())).unwrap();
        e.set_text_browser_fallbacks(Some(vec![lynx.clone(), w3m.clone()]))
            .unwrap();

        let action = approval(e.prepare("web_search", &json!({"query":"rust"})).unwrap());
        // The preview tells the user about the fallback before approval.
        assert_eq!(action.details["fallback_browsers"], json!(["w3m"]));
        assert!(
            action.description.contains("retried with w3m"),
            "{}",
            action.description
        );
        let result = e.approve(&action).unwrap();
        assert_eq!(result["provider"]["browser"], "w3m");
        assert_eq!(result["results"][0]["url"], "https://example.org/w");
        assert!(result["failed_browsers"][0]
            .as_str()
            .unwrap()
            .contains("lynx: DuckDuckGo refused this search (rate limited"));

        // When every browser is refused, the error names each failure.
        let w3m_limited = fake_browser(bin.path(), "w3m", limited);
        e.set_text_browser_fallbacks(Some(vec![w3m_limited]))
            .unwrap();
        let action = approval(e.prepare("web_search", &json!({"query":"rust"})).unwrap());
        let error = e.approve(&action).unwrap_err().to_string();
        assert!(error.contains("lynx:") && error.contains("w3m:"), "{error}");
    }
    /// A local stand-in for the public knowledge APIs, answering by path.
    fn fake_knowledge(
        routes: Vec<(&'static str, u16, &'static str)>,
    ) -> crate::knowledge_search::Endpoints {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut buffer = [0u8; 4096];
                let n = stream.read(&mut buffer).unwrap_or(0);
                let request = String::from_utf8_lossy(&buffer[..n]).to_string();
                let line = request.lines().next().unwrap_or("").to_string();
                let (status, body) = routes
                    .iter()
                    .find(|(prefix, _, _)| line.contains(&format!(" {prefix}")))
                    .map_or((404, "not found"), |(_, s, b)| (*s, *b));
                let reply = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(reply.as_bytes());
            }
        });
        crate::knowledge_search::Endpoints {
            wikipedia: format!("{base}/wiki"),
            hackernews: format!("{base}/hn"),
            arxiv: format!("{base}/arxiv"),
            crossref: format!("{base}/crossref"),
            github: format!("{base}/github"),
            herbal: format!("{base}/herbal"),
            nomad: format!("{base}/nomad"),
            bible: format!("{base}/bible"),
            knowledgecache: format!("{base}/knowledgecache"),
        }
    }

    const FAKE_WIKI: &str = r#"{"query":{"pages":[{"title":"Borrow checker","index":1,"extract":"The borrow checker is the part of the Rust compiler that enforces ownership and borrowing rules at compile time.","fullurl":"https://en.wikipedia.org/wiki/Borrow_checker"}]}}"#;
    const FAKE_HN: &str = r#"{"hits":[{"title":"Understanding the borrow checker","url":"https://blog.example/bc","objectID":"7","points":50,"num_comments":9}]}"#;

    #[test]
    fn a_search_mode_previews_its_sources_and_returns_sourced_results() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        e.set_network_enabled(true);
        e.set_knowledge_search(true);
        e.set_knowledge_endpoints(fake_knowledge(vec![
            ("/wiki", 200, FAKE_WIKI),
            ("/hn", 200, FAKE_HN),
        ]));
        // No browser, SearXNG or MCP is configured, yet search is available.
        assert!(names(&e).iter().any(|n| n == "web_search"));
        let action = approval(
            e.prepare(
                "web_search",
                &json!({"query":"borrow checker","mode":"encyclopedia"}),
            )
            .unwrap(),
        );
        assert_eq!(action.title, "Search Wikipedia");
        assert_eq!(action.details["provider"], "knowledge");
        assert_eq!(action.details["hosts"], json!(["en.wikipedia.org"]));
        assert!(
            action
                .description
                .contains("public APIs that need no account"),
            "{}",
            action.description
        );
        let result = e.approve(&action).unwrap();
        assert_eq!(result["ok"], true);
        assert_eq!(
            result["results"][0]["url"],
            "https://en.wikipedia.org/wiki/Borrow_checker"
        );
        assert!(result["results"][0]["snippet"]
            .as_str()
            .unwrap()
            .contains("enforces ownership and borrowing rules"));
        assert_eq!(result["source_reports"][0]["source"], "wikipedia");
        // With no mode, the default is the general mix.
        let general = approval(
            e.prepare("web_search", &json!({"query":"borrow checker"}))
                .unwrap(),
        );
        assert_eq!(general.title, "Search Wikipedia and Hacker News");
        let both = e.approve(&general).unwrap();
        assert_eq!(both["results"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn home_server_modes_need_the_local_sites_setting_not_the_public_one() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        e.set_network_enabled(true);
        e.set_search_url(Some("http://127.0.0.1:18080".into()))
            .unwrap();
        let off = e
            .prepare("web_search", &json!({"query":"valerian","mode":"herbal"}))
            .unwrap_err()
            .to_string();
        assert!(off.contains("home server"), "{off}");
        let schema = |e: &ToolEngine| serde_json::to_string(&e.definitions()).unwrap();
        assert!(
            !schema(&e).contains("herbal"),
            "self-hosted modes must stay hidden"
        );
        e.set_local_sites_url(Some("http://127.0.0.1/".into()))
            .unwrap();
        assert!(schema(&e).contains("\"herbal\""));
        assert!(schema(&e).contains("Herb Helper"));
        let action = approval(
            e.prepare("web_search", &json!({"query":"valerian","mode":"herbal"}))
                .unwrap(),
        );
        assert_eq!(
            action.title,
            "Search Herb Helper (your site) and NOMAD health libraries"
        );
        assert!(
            action.description.contains("stays on this computer"),
            "{}",
            action.description
        );
        // Public modes still need their own setting.
        assert!(e
            .prepare("web_search", &json!({"query":"x","mode":"academic"}))
            .is_err());
    }

    #[test]
    fn search_modes_are_validated_and_need_the_setting() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        e.set_network_enabled(true);
        // Not enabled: no provider at all, and a mode says why.
        assert!(!names(&e).iter().any(|n| n == "web_search"));
        let off = e
            .prepare("web_search", &json!({"query":"x","mode":"academic"}))
            .unwrap_err()
            .to_string();
        assert!(off.contains("Allow public knowledge sources"), "{off}");
        e.set_knowledge_search(true);
        let bad = e
            .prepare("web_search", &json!({"query":"x","mode":"everything"}))
            .unwrap_err()
            .to_string();
        assert!(
            bad.contains("general, encyclopedia, academic, tech"),
            "{bad}"
        );
        // Network tools off: nothing is searched, whatever the mode.
        e.set_network_enabled(false);
        assert!(e
            .prepare("web_search", &json!({"query":"x","mode":"tech"}))
            .is_err());
    }

    #[test]
    fn when_every_browser_fails_the_disclosed_knowledge_fallback_answers() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let bin = tempfile::tempdir().unwrap();
        let limited = r#"If this persists, please <a href="mailto:x@duckduckgo.com?subject=Error getting results">email us</a>."#;
        let lynx = fake_browser(bin.path(), "lynx", limited);
        let e = engine(dir.path(), backups.path());
        e.set_network_enabled(true);
        e.set_text_browser_program(Some(lynx.clone())).unwrap();
        e.set_text_browser_fallbacks(Some(vec![lynx])).unwrap();

        // Fallback not allowed: the failure is reported exactly as before, and no other site is contacted.
        let action = approval(
            e.prepare("web_search", &json!({"query":"borrow checker"}))
                .unwrap(),
        );
        assert!(
            !action.description.contains("sent to Wikipedia"),
            "{}",
            action.description
        );
        assert!(e
            .approve(&action)
            .unwrap_err()
            .to_string()
            .starts_with("Every text browser failed:"));

        // Allowed: the preview names the sources before the user approves.
        e.set_knowledge_search(true);
        e.set_knowledge_endpoints(fake_knowledge(vec![
            ("/wiki", 200, FAKE_WIKI),
            ("/hn", 200, FAKE_HN),
        ]));
        let action = approval(
            e.prepare("web_search", &json!({"query":"borrow checker"}))
                .unwrap(),
        );
        assert!(action.description.contains("If every browser fails, the same query is sent to Wikipedia and Hacker News instead"), "{}", action.description);
        assert_eq!(
            action.details["fallback_knowledge_sources"],
            json!(["Wikipedia", "Hacker News"])
        );
        let result = e.approve(&action).unwrap();
        assert_eq!(result["provider"]["type"], "knowledge_fallback");
        assert!(result["failed_browsers"][0]
            .as_str()
            .unwrap()
            .contains("DuckDuckGo refused"));
        assert_eq!(result["results"][0]["source"], "wikipedia");

        // If the public sources fail too, both failures are named.
        e.set_knowledge_endpoints(fake_knowledge(vec![("/wiki", 500, "x"), ("/hn", 500, "x")]));
        let action = approval(
            e.prepare("web_search", &json!({"query":"borrow checker"}))
                .unwrap(),
        );
        let error = e.approve(&action).unwrap_err().to_string();
        assert!(
            error.contains("Every text browser failed:")
                && error.contains("public knowledge sources also failed:"),
            "{error}"
        );
    }

    fn names(engine: &ToolEngine) -> Vec<String> {
        engine
            .definitions()
            .iter()
            .filter_map(|definition| definition.pointer("/function/name").and_then(Value::as_str))
            .map(str::to_owned)
            .collect()
    }
    fn approval(result: PreparedTool) -> PendingAction {
        match result {
            PreparedTool::Approval(a) => *a,
            _ => panic!("Expected approval"),
        }
    }

    #[test]
    fn project_listing_pages_have_no_gaps_or_duplicate_paths() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        for i in 0..7 {
            fs::write(dir.path().join(format!("file_{i}.rs")), "fn main() {}\n").unwrap();
        }
        let e = engine(dir.path(), backups.path());
        let mut offset = 0usize;
        let mut seen = Vec::new();
        loop {
            let PreparedTool::Immediate(value) = e
                .prepare(
                    "project_tree",
                    &json!({"path":".", "max_depth":1, "limit":2, "offset":offset}),
                )
                .unwrap()
            else {
                panic!("Expected immediate tree result");
            };
            for entry in value["entries"].as_array().unwrap() {
                seen.push(entry["path"].as_str().unwrap().to_string());
            }
            match value["next_offset"].as_u64() {
                Some(next) => offset = next as usize,
                None => break,
            }
        }
        assert_eq!(seen.len(), 7);
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 7);
        let PreparedTool::Immediate(first) = e
            .prepare("list_files", &json!({"path":".", "limit":3, "offset":0}))
            .unwrap()
        else {
            panic!("Expected immediate list result");
        };
        let PreparedTool::Immediate(second) = e
            .prepare("list_files", &json!({"path":".", "limit":3, "offset":3}))
            .unwrap()
        else {
            panic!("Expected immediate list result");
        };
        assert_eq!(first["total"], 7);
        assert_eq!(first["next_offset"], 3);
        assert_eq!(second["offset"], 3);
        assert_ne!(first["entries"][0]["path"], second["entries"][0]["path"]);
    }

    #[test]
    fn writes_need_matching_one_use_approval_and_preserve_backup() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("note.txt"), "old").unwrap();
        let e = engine(dir.path(), backups.path());
        let a = approval(
            e.prepare("write_file", &json!({"path":"note.txt","content":"new"}))
                .unwrap(),
        );
        assert_eq!(
            fs::read_to_string(dir.path().join("note.txt")).unwrap(),
            "old"
        );
        let mut forged = a.clone();
        if let Operation::Write { content, .. } = &mut forged.operation {
            *content = "forged".into();
        }
        assert!(e.approve(&forged).is_err());
        let result = e.approve(&a).unwrap();
        assert_eq!(
            fs::read_to_string(dir.path().join("note.txt")).unwrap(),
            "new"
        );
        assert_eq!(
            fs::read_to_string(result["backup"].as_str().unwrap()).unwrap(),
            "old"
        );
        assert!(e.approve(&a).is_err());
    }

    #[test]
    fn stale_preview_refuses_to_overwrite_external_changes() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("note.txt"), "old").unwrap();
        let e = engine(dir.path(), backups.path());
        let a = approval(
            e.prepare(
                "replace_text",
                &json!({"path":"note.txt","find":"old","replacement":"new"}),
            )
            .unwrap(),
        );
        fs::write(dir.path().join("note.txt"), "external").unwrap();
        assert!(e.approve(&a).is_err());
        assert_eq!(
            fs::read_to_string(dir.path().join("note.txt")).unwrap(),
            "external"
        );
    }

    #[test]
    fn traversal_denial_and_network_opt_in_are_enforced() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        assert!(e
            .prepare("write_file", &json!({"path":"../escape","content":"x"}))
            .is_err());
        assert!(e
            .prepare("fetch_url", &json!({"url":"https://example.com"}))
            .is_err());
        let a = approval(
            e.prepare("write_file", &json!({"path":"new.txt","content":"x"}))
                .unwrap(),
        );
        e.deny(&a.id);
        assert!(e.approve(&a).is_err());
        assert!(!dir.path().join("new.txt").exists());
    }

    #[test]
    fn approved_copy_move_and_delete_have_real_results() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        fs::write(dir.path().join("a.txt"), "one").unwrap();
        let a = approval(
            e.prepare(
                "copy_file",
                &json!({"source":"a.txt","destination":"sub/b.txt"}),
            )
            .unwrap(),
        );
        e.approve(&a).unwrap();
        let a = approval(
            e.prepare(
                "move_file",
                &json!({"source":"sub/b.txt","destination":"c.txt"}),
            )
            .unwrap(),
        );
        e.approve(&a).unwrap();
        assert!(!dir.path().join("sub/b.txt").exists());
        assert_eq!(fs::read_to_string(dir.path().join("c.txt")).unwrap(), "one");
        let a = approval(e.prepare("delete_file", &json!({"path":"c.txt"})).unwrap());
        let result = e.approve(&a).unwrap();
        assert!(!dir.path().join("c.txt").exists());
        assert_eq!(
            fs::read_to_string(result["recoverable_copy"].as_str().unwrap()).unwrap(),
            "one"
        );
    }

    fn immediate(e: &ToolEngine, tool: &str, args: Value) -> Value {
        match e.prepare(tool, &args).unwrap() {
            PreparedTool::Immediate(v) => v,
            _ => panic!("expected an immediate result"),
        }
    }

    #[test]
    fn complete_search_says_so_and_zero_matches_is_conclusive() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        fs::write(dir.path().join("a.txt"), "alpha").unwrap();
        let none = immediate(&e, "search_files", json!({"query":"absent"}));
        assert_eq!(none["complete"], true);
        assert_eq!(none["truncated"], false);
        assert!(none["resume_after"].is_null());
        assert!(none["coverage"].as_str().unwrap().starts_with("COMPLETE"));
        assert!(none["excluded_directory_names"]
            .to_string()
            .contains("node_modules"));
    }

    #[test]
    fn search_that_stops_early_is_incomplete_and_resumes_without_gaps_or_repeats() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        for name in ["a", "b", "c", "d"] {
            fs::write(dir.path().join(format!("{name}.txt")), "needle\n").unwrap();
        }
        let mut seen = Vec::new();
        let mut cursor: Option<String> = None;
        let mut pages = 0;
        loop {
            let mut args = json!({"query":"needle","limit":2});
            if let Some(c) = &cursor {
                args["resume_after"] = json!(c);
            }
            let page = immediate(&e, "search_files", args);
            pages += 1;
            for m in page["matches"].as_array().unwrap() {
                seen.push(m["path"].as_str().unwrap().to_string());
            }
            if page["complete"] == true {
                break;
            }
            assert_eq!(page["truncated"], true);
            assert!(page["coverage"].as_str().unwrap().starts_with("INCOMPLETE"));
            cursor = Some(page["resume_after"].as_str().unwrap().to_string());
            assert!(pages < 10, "search never completed");
        }
        assert_eq!(seen, ["a.txt", "b.txt", "c.txt", "d.txt"]);
        assert!(pages >= 2);
    }

    #[test]
    fn search_reaches_large_text_tails_and_resumes_inside_dense_files() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        let mut text = "x".repeat(1_050_000);
        text.push_str("\nneedle one\nneedle two\n");
        fs::write(dir.path().join("large.txt"), text).unwrap();
        let first = immediate(&e, "search_files", json!({"query":"needle","limit":1}));
        assert_eq!(first["complete"], false);
        assert_eq!(first["matches"][0]["line"], 2);
        let second = immediate(&e, "search_files", json!({"query":"needle","limit":1,
            "resume_after":first["resume_after"],"resume_line":first["resume_line"]}));
        assert_eq!(second["complete"], true);
        assert_eq!(second["matches"][0]["line"], 3);
    }

    #[test]
    fn unreadable_files_are_reported_without_blocking_a_complete_search() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        fs::write(dir.path().join("notes.txt"), "plain text\n").unwrap();
        fs::write(dir.path().join("latin1.txt"), b"caf\xe9\n").unwrap();
        let result = immediate(&e, "search_files", json!({"query":"absent"}));
        assert_eq!(result["complete"], true, "{result}");
        assert_eq!(result["files_skipped"], 1);
        assert!(result["coverage"].as_str().unwrap().contains("could not be read"));
    }

    #[test]
    fn unknown_resume_cursor_is_an_error_not_an_empty_complete_result() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        fs::write(dir.path().join("a.txt"), "needle").unwrap();
        assert!(e
            .prepare(
                "search_files",
                &json!({"query":"needle","resume_after":"nope.txt"})
            )
            .is_err());
    }

    #[test]
    fn read_file_pages_on_line_boundaries_and_names_the_next_line() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        let body: String = (1..=300)
            .map(|n| format!("line number {n:03} of the file\n"))
            .collect();
        fs::write(dir.path().join("long.txt"), &body).unwrap();
        let first = immediate(
            &e,
            "read_file",
            json!({"path":"long.txt","end_line":300,"max_chars":1000}),
        );
        assert_eq!(first["complete"], false);
        assert_eq!(first["total_lines"], 300);
        let end = first["returned_end_line"].as_u64().unwrap();
        assert_eq!(first["next_start_line"], end + 1);
        // No line is cut in half: the last returned line is whole.
        let content = first["content"].as_str().unwrap();
        assert!(content.lines().last().unwrap().ends_with("of the file"));
        // Following next_start_line reaches the end with every line exactly once.
        let mut next = first["next_start_line"].as_u64();
        let mut last = end;
        while let Some(start) = next {
            let page = immediate(
                &e,
                "read_file",
                json!({"path":"long.txt","start_line":start,"end_line":300,"max_chars":1000}),
            );
            assert_eq!(page["returned_start_line"], start);
            last = page["returned_end_line"].as_u64().unwrap();
            next = page["next_start_line"].as_u64();
        }
        assert_eq!(last, 300);
    }

    #[test]
    fn line_numbers_sent_as_text_are_honoured() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        let body: String = (1..=20).map(|n| format!("line {n}\n")).collect();
        fs::write(dir.path().join("a.txt"), body).unwrap();
        let page = immediate(&e, "read_file", json!({"path":"a.txt","start_line":"15"}));
        assert_eq!(page["returned_start_line"], 15);
    }

    #[test]
    fn a_single_very_long_line_is_read_in_resumable_pieces_without_loss_or_duplication() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        // A minified-style line with multibyte characters, far longer than one page.
        let long: String = (0..3000).map(|n| format!("é{n}ü;")).collect();
        fs::write(dir.path().join("min.js"), format!("{long}\nshort tail")).unwrap();
        let (mut line, mut column) = (1u64, 0u64);
        let mut rebuilt = String::new();
        let mut hash: Option<String> = None;
        let mut pages = 0;
        loop {
            let mut args =
                json!({"path":"min.js","start_line":line,"start_column":column,"max_chars":1000});
            if let Some(h) = &hash {
                args["expected_sha256"] = json!(h);
            }
            let page = immediate(&e, "read_file", args);
            hash = page["source_sha256"].as_str().map(str::to_owned);
            let content = page["content"].as_str().unwrap();
            for text_line in content.split('\n') {
                let (_, body) = text_line.split_once(": ").unwrap();
                rebuilt.push_str(body);
                if page["next_column"].is_null()
                    || !text_line.starts_with(&format!("{}", page["returned_end_line"]))
                {
                    rebuilt.push('\n');
                }
            }
            pages += 1;
            assert!(pages < 200, "pagination must make progress");
            match (
                page["next_start_line"].as_u64(),
                page["next_column"].as_u64(),
            ) {
                (Some(l), Some(c)) => {
                    line = l;
                    column = c;
                }
                (Some(l), None) => {
                    line = l;
                    column = 0;
                }
                (None, _) => break,
            }
        }
        assert!(pages > 3, "the long line must span several pages");
        assert_eq!(
            rebuilt.trim_end_matches('\n'),
            format!("{long}\nshort tail")
        );
    }

    #[test]
    fn a_file_changed_between_pages_is_refused_instead_of_skipping_content() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        fs::write(dir.path().join("a.txt"), "one\ntwo\n").unwrap();
        let first = immediate(&e, "read_file", json!({"path":"a.txt"}));
        let hash = first["source_sha256"].as_str().unwrap().to_owned();
        fs::write(dir.path().join("a.txt"), "zero\none\ntwo\n").unwrap();
        let error = e
            .workspace
            .read_line_page("a.txt", 2, 0, 10, 6000, Some(&hash))
            .unwrap_err();
        assert!(error.to_string().contains("changed"), "{error}");
    }

    #[test]
    fn reading_a_slice_is_complete_for_that_slice_not_marked_truncated() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        fs::write(dir.path().join("f.txt"), "1\n2\n3\n4\n5\n").unwrap();
        let r = immediate(
            &e,
            "read_file",
            json!({"path":"f.txt","start_line":2,"end_line":3}),
        );
        assert_eq!(r["complete"], true);
        assert_eq!(r["truncated"], false);
        assert_eq!(r["next_start_line"], 4);
        assert!(r["coverage"]
            .as_str()
            .unwrap()
            .contains("continue with start_line=4"));
    }

    #[test]
    fn listing_a_directory_does_not_lose_entries_behind_a_deep_sibling() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        fs::create_dir_all(dir.path().join("a/deep/deeper")).unwrap();
        for n in 0..50 {
            fs::write(dir.path().join(format!("a/deep/deeper/f{n}.txt")), "x").unwrap();
        }
        fs::write(dir.path().join("zzz.txt"), "x").unwrap();
        let r = immediate(&e, "list_files", json!({}));
        let names = r["entries"].to_string();
        assert!(
            names.contains("zzz.txt") && names.contains("\"a\""),
            "{names}"
        );
        assert!(
            !names.contains("f0.txt"),
            "listing must not descend: {names}"
        );
    }

    #[test]
    fn regex_style_literal_query_is_flagged_and_regex_mode_really_matches_alternatives() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        fs::write(dir.path().join("a.rs"), "fn alpha() {}\nlet x = 1 || 2;\n").unwrap();
        fs::write(dir.path().join("b.rs"), "fn beta() {}\n").unwrap();
        let literal = immediate(&e, "search_files", json!({"query":"alpha|beta"}));
        assert!(literal["matches"].as_array().unwrap().is_empty());
        assert_eq!(literal["query_mode"], "literal");
        assert!(
            literal["warning"].as_str().unwrap().contains("regex"),
            "{literal}"
        );
        assert!(
            literal["coverage"]
                .as_str()
                .unwrap()
                .starts_with("NOT CONCLUSIVE"),
            "{literal}"
        );
        let real = immediate(
            &e,
            "search_files",
            json!({"query":"alpha|beta","regex":true}),
        );
        assert_eq!(real["query_mode"], "regex");
        assert_eq!(real["matches"].as_array().unwrap().len(), 2);
        assert!(real["warning"].is_null());
        // Ordinary code text, including `||`, is literal and raises no warning.
        let code = immediate(&e, "search_files", json!({"query":"1 || 2"}));
        assert_eq!(code["matches"].as_array().unwrap().len(), 1);
        assert!(code["warning"].is_null(), "{code}");
        assert!(e
            .prepare("search_files", &json!({"query":"(unclosed","regex":true}))
            .is_err());
    }

    #[test]
    fn an_absolute_path_inside_the_project_works_and_one_outside_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/lib.rs"), "pub fn x() {}\n").unwrap();
        let root = e.root().display().to_string();
        let listing = immediate(&e, "list_files", json!({"path":root}));
        assert!(listing.to_string().contains("src"), "{listing}");
        let read = immediate(
            &e,
            "read_file",
            json!({"path":format!("{root}/src/lib.rs")}),
        );
        assert!(read["content"].as_str().unwrap().contains("pub fn x"));
        let outside = e
            .prepare(
                "read_file",
                &json!({"path":other.path().join("x.txt").display().to_string()}),
            )
            .unwrap_err()
            .to_string();
        assert!(
            outside.contains("outside the connected project folder"),
            "{outside}"
        );
        // Protected files stay protected however the path is written.
        fs::write(dir.path().join(".env"), "SECRET=1").unwrap();
        assert!(e
            .prepare("read_file", &json!({"path":format!("{root}/.env")}))
            .is_err());
    }

    #[test]
    fn an_edit_with_wrong_indentation_is_previewed_as_the_real_text_and_applied_correctly() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        let original = "fn main() {\n    let a = 1;\n    let b = 2;\n}\n";
        fs::write(dir.path().join("m.rs"), original).unwrap();
        // No indentation in what the model sent.
        let proposal = approval(
            e.prepare(
                "replace_text",
                &json!({"path":"m.rs","find":"let a = 1;\nlet b = 2;","replacement":"let a = 10;\nlet b = 20;"}),
            )
            .unwrap(),
        );
        assert!(
            proposal.description.contains("indentation differed"),
            "{}",
            proposal.description
        );
        assert_eq!(
            proposal.details["match_note"],
            "matched lines 2-3 although indentation differed"
        );
        // The preview shows the file as it will be, not the model's version.
        assert_eq!(
            proposal.details["after"],
            "fn main() {\n    let a = 10;\n    let b = 20;\n}\n"
        );
        assert_eq!(
            fs::read_to_string(dir.path().join("m.rs")).unwrap(),
            original
        );
        e.approve(&proposal).unwrap();
        assert_eq!(
            fs::read_to_string(dir.path().join("m.rs")).unwrap(),
            "fn main() {\n    let a = 10;\n    let b = 20;\n}\n"
        );
        // A failing edit says what the closest real lines are, and changes nothing.
        let err = e
            .prepare(
                "replace_text",
                &json!({"path":"m.rs","find":"let a = 11;","replacement":"x"}),
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("closest real lines are 2-2"), "{err}");
    }

    #[test]
    fn the_repo_map_tool_ranks_files_pages_honestly_and_stays_inside_the_project() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::write(
            dir.path().join("src/core.rs"),
            "pub struct Engine;\npub fn start_engine_loop() {}\n",
        )
        .unwrap();
        for n in 0..4 {
            fs::write(
                dir.path().join(format!("src/user{n}.rs")),
                format!("fn use_{n}() {{ start_engine_loop(); let _ = Engine; }}\n"),
            )
            .unwrap();
        }
        fs::write(dir.path().join(".env"), "SECRET=1").unwrap();
        let result = immediate(&e, "repo_map", json!({}));
        assert!(
            result["map"].as_str().unwrap().starts_with("src/core.rs"),
            "{result}"
        );
        assert_eq!(result["complete"], true);
        assert!(
            result["coverage"]
                .as_str()
                .unwrap()
                .contains("5 of 5 files with definitions are shown"),
            "{result}"
        );
        assert!(!result.to_string().contains("SECRET"));
        // A small budget pages, and says where to continue.
        let first = immediate(&e, "repo_map", json!({"max_chars":1500}));
        let tiny = immediate(&e, "repo_map", json!({"max_chars":1500, "focus":"user3"}));
        assert!(
            tiny["map"].as_str().unwrap().starts_with("src/user3.rs"),
            "{tiny}"
        );
        assert!(first["files_shown"].as_u64().unwrap() >= 1);
        // Only paths inside the project are accepted.
        assert!(e
            .prepare("repo_map", &json!({"path":"../outside"}))
            .is_err());
        assert!(
            immediate(&e, "repo_map", json!({"path":"src"}))["files_shown"]
                .as_u64()
                .unwrap()
                == 5
        );
    }

    #[test]
    fn model_file_tools_honor_ignore_and_protected_paths() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        fs::write(dir.path().join(".gitignore"), "ignored.txt\n").unwrap();
        fs::write(dir.path().join(".env"), "needle-secret").unwrap();
        fs::write(dir.path().join("ignored.txt"), "needle-ignored").unwrap();
        fs::write(dir.path().join("visible.txt"), "needle-visible").unwrap();
        let PreparedTool::Immediate(result) = e
            .prepare("search_files", &json!({"query":"needle"}))
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(result["matches"].as_array().unwrap().len(), 1);
        assert_eq!(result["matches"][0]["path"], "visible.txt");
        let PreparedTool::Immediate(result) = e.prepare("list_files", &json!({})).unwrap() else {
            panic!()
        };
        assert!(!result.to_string().contains(".env"), "{result}");
        assert!(!result.to_string().contains("ignored.txt"), "{result}");
        assert!(e.prepare("read_file", &json!({"path":".env"})).is_err());
    }

    #[test]
    fn sandboxed_commands_change_only_the_project_and_say_so_in_the_preview() {
        if !crate::sandbox::available() {
            eprintln!("bubblewrap unavailable; skipping");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        e.set_sandbox(SandboxSettings {
            mode: crate::sandbox::SandboxMode::Required,
            ..Default::default()
        });
        let script = format!(
            "echo inside > made.txt; echo escaped > {}/x 2>/dev/null || echo blocked",
            outside.path().display()
        );
        let a = approval(
            e.prepare("run_command", &json!({"program":"sh","args":["-c",script]}))
                .unwrap(),
        );
        assert_eq!(a.details["sandbox"]["sandboxed"], true);
        assert_eq!(a.details["sandbox"]["network"], false);
        assert!(a.description.contains("sandbox"), "{}", a.description);
        let result = e.approve(&a).unwrap();
        assert!(result.to_string().contains("blocked"), "{result}");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("made.txt")).unwrap(),
            "inside\n"
        );
        assert!(!outside.path().join("x").exists());
    }

    #[test]
    fn local_fetches_always_ask_and_public_ones_stay_public() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        e.set_network_enabled(true);
        for url in [
            "http://127.0.0.1:11434/api/tags",
            "http://localhost/",
            "http://10.0.0.1/",
            "http://169.254.169.254/",
        ] {
            let a = approval(e.prepare("fetch_url", &json!({"url":url})).unwrap());
            assert_eq!(
                a.operation.approval_kind(),
                ApprovalKind::LocalNetwork,
                "{url}"
            );
            assert_eq!(a.details["destination"], "local");
        }
        let a = approval(
            e.prepare("fetch_url", &json!({"url":"https://1.1.1.1/"}))
                .unwrap(),
        );
        assert_eq!(a.operation.approval_kind(), ApprovalKind::Web);
        assert!(matches!(
            a.operation,
            Operation::FetchUrl { local: false, .. }
        ));
    }

    #[test]
    fn only_sandboxed_commands_count_as_sandboxed_approvals() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        // Sandbox off: commands and terminals have the user's permissions.
        let a = approval(
            e.prepare("run_command", &json!({"program":"true"}))
                .unwrap(),
        );
        assert_eq!(
            a.operation.approval_kind(),
            ApprovalKind::UnsandboxedCommand
        );
        let a = approval(e.prepare("terminal_open", &json!({})).unwrap());
        assert_eq!(
            a.operation.approval_kind(),
            ApprovalKind::UnsandboxedCommand
        );
        if crate::sandbox::available() {
            e.set_sandbox(SandboxSettings::default());
            for (name, args) in [
                ("run_command", json!({"program":"true"})),
                ("run_python", json!({"code":"print(1)"})),
                ("terminal_open", json!({})),
            ] {
                let a = approval(e.prepare(name, &args).unwrap());
                assert_eq!(a.operation.approval_kind(), ApprovalKind::Command, "{name}");
            }
        }
        // Configured integrations always run outside the sandbox.
        let adapter = Operation::Adapter {
            adapter_id: "speech".into(),
            spec: CommandSpec::default(),
            output: None,
            config_hash: String::new(),
        };
        assert_eq!(adapter.approval_kind(), ApprovalKind::UnsandboxedCommand);
    }

    #[test]
    fn run_python_executes_in_the_sandbox_and_shows_the_code() {
        if !crate::sandbox::available() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        e.set_sandbox(SandboxSettings::default());
        let code = "import os, socket\nopen('result.txt','w').write(str(6*7))\nprint(sorted(os.listdir(os.environ['HOME'])))\ntry:\n    socket.create_connection(('1.1.1.1', 53), timeout=2); print('NET')\nexcept OSError:\n    print('NONET')";
        let a = approval(e.prepare("run_python", &json!({"code":code})).unwrap());
        assert_eq!(a.title, "Run Python code");
        assert_eq!(a.details["code"], code);
        assert_eq!(a.details["sandbox"]["sandboxed"], true);
        let result = e.approve(&a).unwrap().to_string();
        assert!(
            result.contains("[]") && result.contains("NONET"),
            "{result}"
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("result.txt")).unwrap(),
            "42"
        );
    }

    #[test]
    fn run_python_refuses_to_run_unsandboxed_unless_the_sandbox_is_off() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        // Off: allowed, with the warning.
        let a = approval(
            e.prepare("run_python", &json!({"code":"print(1)"}))
                .unwrap(),
        );
        assert!(a.description.contains("sandbox is off"));
        assert!(e.prepare("run_python", &json!({"code":""})).is_err());
    }

    #[test]
    fn sandbox_off_keeps_the_unsandboxed_warning() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        let a = approval(
            e.prepare("run_command", &json!({"program":"true"}))
                .unwrap(),
        );
        assert_eq!(a.details["sandbox"]["sandboxed"], false);
        assert!(a.description.contains("not sandboxed"));
    }

    #[test]
    fn command_standard_input_is_part_of_the_preview() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        let a = approval(
            e.prepare(
                "run_command",
                &json!({"program":"cat","stdin_text":"literal input"}),
            )
            .unwrap(),
        );
        assert_eq!(a.details["stdin_text"], "literal input");
        let cancel = Arc::new(AtomicBool::new(true));
        assert!(e.approve_cancellable(&a, cancel).is_err());
    }

    #[cfg(unix)]
    fn mapped_search_server(script: &str) -> McpServerConfig {
        McpServerConfig {
            id: "fixture-search".into(),
            name: "Fixture Search".into(),
            program: "python3".into(),
            args: vec!["-u".into(), "-c".into(), script.into()],
            cwd: None,
            env: BTreeMap::new(),
            enabled: true,
            timeout_secs: 3,
            web_search: Some(McpWebSearchConfig {
                tool: "fixture_web_search".into(),
                query_argument: "query".into(),
                limit_argument: Some("count".into()),
                extra_arguments: BTreeMap::from([("safe".into(), json!(true))]),
                ..Default::default()
            }),
        }
    }

    #[test]
    fn restricted_engine_hides_and_rejects_workspace_actions() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        e.set_workspace_tools_enabled(false);
        e.set_network_enabled(true);
        e.set_search_url(Some("http://127.0.0.1:8080".into()))
            .unwrap();
        let mut mapped = McpServerConfig::brave_search_preset();
        mapped.enabled = true;
        mapped.program = std::env::current_exe()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        mapped.args.clear();
        mapped
            .env
            .insert("BRAVE_API_KEY".into(), "fixture-key".into());
        e.register_mcp(mapped).unwrap();

        let definitions = e.definitions();
        let names = definitions
            .iter()
            .filter_map(|definition| definition.pointer("/function/name").and_then(Value::as_str))
            .collect::<Vec<_>>();
        assert!(names.contains(&"calculator"));
        assert!(names.contains(&"fetch_url"));
        assert!(names.contains(&"web_search"));
        assert!(!names.contains(&"read_file"));
        assert!(!names.contains(&"write_file"));
        assert!(!names.contains(&"run_command"));
        assert!(!names.contains(&"mcp_list"));
        assert!(!names.contains(&"mcp_connect"));
        assert!(!names.contains(&"mcp_call"));
        assert!(e
            .prepare("read_file", &json!({"path":"secret.txt"}))
            .unwrap_err()
            .to_string()
            .contains("Open a workspace"));
        assert!(e
            .prepare("mcp_list", &json!({}))
            .unwrap_err()
            .to_string()
            .contains("generic MCP"));
    }

    #[test]
    #[cfg(unix)]
    fn text_browser_web_search_previews_exact_command_and_parses_results() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let bin = tempfile::tempdir().unwrap();
        // Stand-in for lynx: records its argv and prints a results page.
        let program = bin.path().join("lynx");
        let argv_log = bin.path().join("argv.txt");
        std::fs::write(&program, format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\ncat <<'HTML'\n<div class=\"result\"><a class=\"result__a\" href=\"//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa&amp;rut=1\">Example A</a><a class=\"result__snippet\">Snippet A</a></div>\nHTML\n",
            argv_log.display()
        )).unwrap();
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
        let e = engine(dir.path(), backups.path());
        let browser = integrations::TextBrowser {
            name: "lynx".into(),
            program: program.clone(),
        };
        e.set_text_browser_program(Some(browser.clone())).unwrap();
        assert!(
            !names(&e).iter().any(|name| name == "web_search"),
            "network opt-in is still required"
        );
        e.set_network_enabled(true);
        assert!(names(&e).iter().any(|name| name == "web_search"));

        let action = approval(
            e.prepare("web_search", &json!({"query":"rust lang","limit":3}))
                .unwrap(),
        );
        assert_eq!(action.details["provider"], "text_browser");
        assert_eq!(action.details["browser"], "lynx");
        assert_eq!(
            action.details["command"][0],
            program.to_string_lossy().as_ref()
        );
        let result = e.approve(&action).unwrap();
        assert_eq!(result["ok"], true);
        assert_eq!(result["results"][0]["url"], "https://example.org/a");
        assert_eq!(result["results"][0]["title"], "Example A");
        assert_eq!(result["results"][0]["snippet"], "Snippet A");
        let argv = std::fs::read_to_string(&argv_log).unwrap();
        assert!(argv.lines().any(|line| line == "-source"));
        assert!(argv.contains("https://html.duckduckgo.com/html/?q=rust+lang"));

        // A preview approved after the configured browser changes must not run.
        let stale = approval(e.prepare("web_search", &json!({"query":"x"})).unwrap());
        e.set_text_browser_program(None).unwrap();
        assert!(e
            .approve(&stale)
            .unwrap_err()
            .to_string()
            .contains("configuration changed"));

        // A running SearXNG keeps precedence over the text browser.
        e.set_text_browser_program(Some(browser)).unwrap();
        let searxng = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let searxng_url = format!("http://{}", searxng.local_addr().unwrap());
        e.set_search_url(Some(searxng_url)).unwrap();
        let preferred = approval(e.prepare("web_search", &json!({"query":"x"})).unwrap());
        assert_eq!(preferred.details["provider"], "searxng");
        // A configured SearXNG that is not running must not block search: the
        // preview names the text browser that will actually be used.
        drop(searxng);
        // Port 0 never accepts connections; reusing the dropped port would race
        // with parallel tests.
        e.set_search_url(Some("http://127.0.0.1:0".into())).unwrap();
        let fallback = approval(e.prepare("web_search", &json!({"query":"x"})).unwrap());
        assert_eq!(fallback.details["provider"], "text_browser");
        let explicit = approval(
            e.prepare(
                "web_search",
                &json!({"query":"x","provider":"text_browser"}),
            )
            .unwrap(),
        );
        assert_eq!(explicit.details["provider"], "text_browser");
    }

    #[test]
    #[cfg(unix)]
    fn mcp_web_search_is_first_class_and_reuses_connection() {
        let script = r#"import json,sys
for line in sys.stdin:
 r=json.loads(line)
 if 'id' not in r: continue
 method=r['method']
 if method=='initialize':
  result={'protocolVersion':r['params']['protocolVersion'],'capabilities':{'tools':{}},'serverInfo':{'name':'search-fixture','version':'1'}}
 elif method=='tools/list':
  result={'tools':[{'name':'fixture_web_search','description':'fixture search','inputSchema':{'type':'object'}}]}
 elif method=='tools/call':
  arguments=r['params']['arguments']
  result={'content':[{'type':'text','text':'query=%s;count=%s;safe=%s' % (arguments['query'],arguments['count'],arguments['safe'])}]}
 else:
  result={}
 print(json.dumps({'jsonrpc':'2.0','id':r['id'],'result':result}),flush=True)
"#;
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        e.set_network_enabled(true);
        e.register_mcp(mapped_search_server(script)).unwrap();

        assert!(e.definitions().iter().any(|definition| {
            definition.pointer("/function/name").and_then(Value::as_str) == Some("web_search")
        }));
        let first = approval(
            e.prepare("web_search", &json!({"query":"linux","limit":4}))
                .unwrap(),
        );
        assert_eq!(first.details["provider"], "mcp");
        assert_eq!(first.details["connected"], false);
        assert_eq!(first.details["tool"], "fixture_web_search");
        assert_eq!(first.details["arguments"]["query"], "linux");
        assert_eq!(first.details["arguments"]["count"], 4);
        assert_eq!(first.details["arguments"]["safe"], true);

        let result = e.approve(&first).unwrap();
        assert_eq!(result["provider"]["type"], "mcp");
        assert_eq!(result["provider"]["server_id"], "fixture-search");
        // A reply without readable results keeps the provider's text, never a fake list.
        assert_eq!(result["result_count"], 0);
        assert_eq!(result["results"], json!([]));
        assert_eq!(result["provider_text"], "query=linux;count=4;safe=True");

        let second = approval(
            e.prepare("web_search", &json!({"query":"rust linux","limit":2}))
                .unwrap(),
        );
        assert_eq!(second.details["connected"], true);
        assert!(second.details["launch_if_needed"].is_null());
        let result = e.approve(&second).unwrap();
        assert_eq!(result["provider_text"], "query=rust linux;count=2;safe=True");
    }

    #[test]
    fn searxng_search_carries_options_from_preview_to_request() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            // The first connection is the reachability probe made while preparing.
            let mut lines = Vec::new();
            for _ in 0..2 {
                let (mut socket, _) = listener.accept().unwrap();
                let mut request = [0u8; 4096];
                let n = socket.read(&mut request).unwrap_or(0);
                let line = String::from_utf8_lossy(&request[..n]).lines().next().unwrap_or_default().to_owned();
                let body = r#"{"results":[{"title":"Rust 2026","url":"https://blog.rust-lang.org/","content":"Release notes"}]}"#;
                let _ = write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                lines.push(line);
            }
            lines
        });
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        e.set_network_enabled(true);
        e.set_search_url(Some(url)).unwrap();
        let action = approval(
            e.prepare("web_search", &json!({"query":"rust release","time_range":"month","language":"en-US","safe_search":"moderate"}))
                .unwrap(),
        );
        assert_eq!(action.details["provider"], "searxng");
        assert_eq!(action.details["search_options"]["language"], "en-US");
        let result = e.approve(&action).unwrap();
        assert_eq!(result["provider"]["type"], "searxng");
        assert_eq!(result["results"][0]["title"], "Rust 2026");
        assert_eq!(result["search_options"]["applied"], json!(["time_range", "language", "safe_search"]));
        assert_eq!(result["search_options"]["not_applied"], json!([]));
        let lines = server.join().unwrap();
        let search = lines.iter().find(|l| l.contains("/search?")).expect("a search request");
        for part in ["time_range=month", "language=en-US", "safesearch=1"] {
            assert!(search.contains(part), "{search}");
        }
    }

    #[test]
    fn image_search_needs_an_image_capable_provider_and_depth_sets_the_default_count() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        e.set_network_enabled(true);
        e.set_knowledge_search(true);
        let error = e.prepare("web_search", &json!({"query":"red fox","mode":"images"})).unwrap_err().to_string();
        assert!(error.contains("Image search needs SearXNG"), "{error}");
        e.set_search_url(Some("http://127.0.0.1:9".into())).unwrap();
        let action = approval(e.prepare("web_search", &json!({"query":"red fox","mode":"images"})).unwrap());
        assert_eq!(action.details["kind"], "images");
        assert_eq!(action.details["limit"], 8, "balanced depth by default");
        let crate::tools::Operation::WebSearch { images, fallbacks, local_sites, .. } = &action.operation else {
            panic!("expected a SearXNG image search")
        };
        assert!(*images && fallbacks.is_empty() && local_sites.is_none());
        e.set_search_depth("speed");
        let action = approval(e.prepare("web_search", &json!({"query":"red fox","mode":"images"})).unwrap());
        assert_eq!(action.details["limit"], 5);
        e.set_search_depth("quality");
        let action = approval(e.prepare("web_search", &json!({"query":"red fox","mode":"images","limit":3})).unwrap());
        assert_eq!(action.details["limit"], 3, "an explicit count wins");
    }

    #[test]
    fn mcp_web_search_returns_the_standard_list_and_reports_options() {
        let script = r#"import json,sys
for line in sys.stdin:
 r=json.loads(line)
 if 'id' not in r: continue
 method=r['method']
 if method=='initialize':
  result={'protocolVersion':r['params']['protocolVersion'],'capabilities':{'tools':{}},'serverInfo':{'name':'search-fixture','version':'1'}}
 elif method=='tools/list':
  result={'tools':[{'name':'fixture_web_search','description':'fixture search','inputSchema':{'type':'object'}}]}
 elif method=='tools/call':
  hits=[{'url':'https://www.rust-lang.org/','title':'Rust','description':'A language'},{'url':'https://tokio.rs/','title':'Tokio','description':'Async runtime'}]
  result={'content':[{'type':'text','text':json.dumps(h)} for h in hits]}
 else:
  result={}
 print(json.dumps({'jsonrpc':'2.0','id':r['id'],'result':result}),flush=True)
"#;
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        e.set_network_enabled(true);
        e.register_mcp(mapped_search_server(script)).unwrap();
        let action = approval(
            e.prepare("web_search", &json!({"query":"rust","limit":5,"time_range":"week"}))
                .unwrap(),
        );
        assert_eq!(action.details["search_options"], json!({"time_range":"week"}));
        // The fixture mapping has no time-range argument, so none is sent.
        assert!(action.details["arguments"].get("time_range").is_none());
        let result = e.approve(&action).unwrap();
        assert_eq!(result["result_count"], 2);
        assert_eq!(result["results"][1], json!({"title":"Tokio","url":"https://tokio.rs/","snippet":"Async runtime"}));
        assert_eq!(result["search_options"]["not_applied"], json!(["time_range"]));
        assert!(result["provider_text"].is_null());
        // An invalid option is refused before anything is sent.
        assert!(e.prepare("web_search", &json!({"query":"rust","safe_search":"maximum"})).is_err());
    }

    #[test]
    #[cfg(unix)]
    fn mcp_search_requires_network_opt_in_and_mcp_is_default_provider() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        e.register_mcp(mapped_search_server("raise SystemExit(0)"))
            .unwrap();
        assert!(!e.definitions().iter().any(|definition| {
            definition.pointer("/function/name").and_then(Value::as_str) == Some("web_search")
        }));
        assert!(e
            .prepare("web_search", &json!({"query":"linux"}))
            .unwrap_err()
            .to_string()
            .contains("Network tools are disabled"));

        e.set_network_enabled(true);
        // A port nothing listens on (8080 was assumed free until a local service took it).
        let closed = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        e.set_search_url(Some(format!("http://127.0.0.1:{closed}")))
            .unwrap();
        let mcp = approval(e.prepare("web_search", &json!({"query":"linux"})).unwrap());
        assert_eq!(mcp.details["provider"], "mcp");
        e.deny(&mcp.id);

        // A requested SearXNG that is not running falls back to automatic
        // selection instead of failing the search.
        let fallback = approval(
            e.prepare("web_search", &json!({"query":"linux","provider":"searxng"}))
                .unwrap(),
        );
        assert_eq!(fallback.details["provider"], "mcp");
        e.deny(&fallback.id);
        // An invented provider ID also falls back rather than erroring.
        let invented = approval(
            e.prepare(
                "web_search",
                &json!({"query":"linux","provider":"brave-search"}),
            )
            .unwrap(),
        );
        assert_eq!(invented.details["provider"], "mcp");
        e.deny(&invented.id);

        // A running SearXNG is honored when explicitly requested.
        let running = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        e.set_search_url(Some(format!("http://{}", running.local_addr().unwrap())))
            .unwrap();
        let searxng = approval(
            e.prepare("web_search", &json!({"query":"linux","provider":"searxng"}))
                .unwrap(),
        );
        assert_eq!(searxng.details["provider"], "searxng");
        e.deny(&searxng.id);
        // The model-facing schema no longer offers a provider choice.
        let schema = e
            .definitions()
            .into_iter()
            .find(|d| d.pointer("/function/name").and_then(Value::as_str) == Some("web_search"))
            .unwrap();
        assert!(schema
            .pointer("/function/parameters/properties/provider")
            .is_none());
    }

    #[test]
    #[cfg(unix)]
    fn adapter_reports_the_actual_requested_output() {
        let dir = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), backups.path());
        let mut adapter = CommandAdapter {
            id: "fixture".into(),
            name: "Output fixture".into(),
            kind: integrations::AdapterKind::Custom,
            enabled: true,
            program: "/bin/true".into(),
            args: vec!["{output}".into()],
            text_to_stdin: false,
            timeout_secs: 10,
        };
        e.register_adapter(adapter.clone()).unwrap();
        let action = approval(
            e.prepare(
                "run_adapter",
                &json!({"adapter_id":"fixture","output":"out.txt"}),
            )
            .unwrap(),
        );
        let result = e.approve(&action).unwrap();
        assert_eq!(result["ok"], false);
        assert_eq!(result["output_result"]["status"], "missing");

        fs::write(dir.path().join("in.txt"), "actual output").unwrap();
        adapter.program = "/bin/cp".into();
        adapter.args = vec!["{input}".into(), "{output}".into()];
        e.register_adapter(adapter).unwrap();
        let action = approval(
            e.prepare(
                "run_adapter",
                &json!({"adapter_id":"fixture","input":"in.txt","output":"out.txt"}),
            )
            .unwrap(),
        );
        let result = e.approve(&action).unwrap();
        assert_eq!(result["ok"], true);
        assert_eq!(result["output_result"]["status"], "created");
        assert_eq!(
            fs::read_to_string(dir.path().join("out.txt")).unwrap(),
            "actual output"
        );
    }
}
