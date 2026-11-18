//! ToolEngine preparation responsibilities.
use super::*;

impl ToolEngine {
    pub fn prepare(&self, name: &str, args: &Value) -> Result<PreparedTool> {
        self.prepare_with_cancel(name, args, None)
    }

    /// The inference worker can stop a terminal read without destroying the
    /// shell or the interactive program whose output it is observing.
    pub fn prepare_cancellable(
        &self,
        name: &str,
        args: &Value,
        cancel: Arc<AtomicBool>,
    ) -> Result<PreparedTool> {
        self.prepare_with_cancel(name, args, Some(cancel.as_ref()))
    }

    fn prepare_with_cancel(
        &self,
        name: &str,
        args: &Value,
        cancel: Option<&AtomicBool>,
    ) -> Result<PreparedTool> {
        if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            bail!("Tool preparation cancelled");
        }
        if !args.is_object() {
            bail!("Tool arguments must be a JSON object");
        }
        if !self.workspace_tools_enabled()
            && matches!(
                name,
                "list_files"
                    | "list_directory"
                    | "project_tree"
                    | "repo_map"
                    | "read_file"
                    | "search_files"
                    | "local_search"
                    | "write_file"
                    | "replace_text"
                    | "edit_file"
                    | "copy_file"
                    | "move_file"
                    | "delete_file"
                    | "make_directory"
                    | "mkdir"
                    | "run_command"
                    | "run_python"
                    | "terminal_open"
                    | "terminal_list"
                    | "terminal_read"
                    | "terminal_write"
                    | "terminal_key"
                    | "terminal_resize"
                    | "terminal_close"
                    | "mcp_list"
                    | "mcp_connect"
                    | "mcp_call"
                    | "list_external_agents"
                    | "consult_agent"
                    | "list_adapters"
                    | "run_adapter"
            )
        {
            bail!("Open a workspace before using local files, commands, generic MCP calls, adapters, or external agents");
        }
        match name {
            "terminal_list" => {
                return Ok(PreparedTool::Immediate(json!({
                    "sessions": self.terminals.ai_list(),
                    "scope": "selected_workspace",
                    "note": "Use a returned session id to read or interact with that existing terminal."
                })));
            }
            "terminal_read" => {
                return Ok(PreparedTool::Immediate(self.read_terminal(args, cancel)?));
            }
            "calculator" => {
                let expression = required(args, "expression")?;
                return Ok(PreparedTool::Immediate(
                    json!({"expression":expression,"value":crate::calc::evaluate(&expression)?,"arithmetic":"f64","angles":"radians"}),
                ));
            }
            "list_files" | "list_directory" => {
                return Ok(PreparedTool::Immediate(self.list(args)?))
            }
            "project_tree" => return Ok(PreparedTool::Immediate(self.tree(args)?)),
            "repo_map" => return Ok(PreparedTool::Immediate(self.repo_map(args)?)),
            "read_file" => return Ok(PreparedTool::Immediate(self.read_file(args)?)),
            "search_files" | "local_search" => {
                return Ok(PreparedTool::Immediate(self.search(args)?))
            }
            "mcp_list" => return Ok(PreparedTool::Immediate(self.mcp_catalog()?)),
            "list_external_agents" => {
                let profiles = self
                    .external_agents
                    .lock()
                    .map_err(|_| anyhow::anyhow!("External agent configuration lock failed"))?;
                return Ok(PreparedTool::Immediate(
                    json!({"agents": profiles.values().filter(|p| p.enabled)
                    .map(|p| json!({"id":p.id,"name":p.name,"may_use_network":p.may_use_network,"requires_approval":true})).collect::<Vec<_>>()}),
                ));
            }
            "list_adapters" => {
                let adapters = self
                    .adapters
                    .lock()
                    .map_err(|_| anyhow::anyhow!("Adapter lock failed"))?;
                return Ok(PreparedTool::Immediate(
                    json!({"adapters":adapters.values().filter(|a| a.enabled)
                    .map(|a| json!({"id":a.id,"name":a.name,"kind":a.kind})).collect::<Vec<_>>()}),
                ));
            }
            _ => {}
        }
        let mut expected = Vec::new();
        let (title, description, details, operation) = match name {
            "terminal_open" | "terminal_write" | "terminal_key" | "terminal_resize"
            | "terminal_close" => self.prepare_terminal(name, args, &mut expected)?,
            "write_file" => {
                let path = required(args, "path")?;
                let content = required_allow_empty(args, "content")?;
                if content.len() > MAX_WRITE_BYTES {
                    bail!("File content exceeds 2 MiB");
                }
                let state = self.state(&path)?;
                ensure_file_or_missing(&state)?;
                let old = if state.kind == "file" {
                    read_limited_beneath(self.root(), &state.resolved, MAX_FILE_BYTES)?
                } else {
                    Vec::new()
                };
                let details = json!({"path":state.resolved,"old_sha256":state.sha256,"old_bytes":state.size,
                    "new_bytes":content.len(),"before":String::from_utf8_lossy(&old),"after":content,
                    "backup_directory":self.backup_root});
                expected.push(state);
                (
                    format!("Write {path}"),
                    "Write exactly the previewed content; preserve any previous file.".into(),
                    details,
                    Operation::Write { path, content },
                )
            }
            "replace_text" | "edit_file" => {
                let path = required(args, "path")?;
                let find = required(args, "find")?;
                let replacement = required_allow_empty(args, "replacement")?;
                if find.len() + replacement.len() > MAX_WRITE_BYTES {
                    bail!("Replacement text exceeds 2 MiB");
                }
                let replace_all = args
                    .get("replace_all")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let state = self.state(&path)?;
                ensure_file(&state)?;
                let before = String::from_utf8(read_limited_beneath(
                    self.root(),
                    &state.resolved,
                    MAX_FILE_BYTES,
                )?)
                .context("Exact replacement requires UTF-8 text")?;
                // Resolve the text to what the file really contains, so the
                // preview, the approval and the write all use the real bytes.
                let located = super::matching::locate(&before, &find, &replacement, replace_all)?;
                let (find, replacement, occurrences, note) = (
                    located.find,
                    located.replacement,
                    located.occurrences,
                    located.note,
                );
                let after = if replace_all {
                    before.replace(&find, &replacement)
                } else {
                    before.replacen(&find, &replacement, 1)
                };
                if after.len() > MAX_FILE_BYTES {
                    bail!("Result exceeds 16 MiB");
                }
                let details = json!({"path":state.resolved,"occurrences":occurrences,"match_note":note,"before":before,"after":after,"old_sha256":state.sha256,"backup_directory":self.backup_root});
                expected.push(state);
                (
                    format!("Edit {path}"),
                    match &note {
                        Some(note) => {
                            format!("Apply the replacement shown in this preview (it {note}).")
                        }
                        None => "Apply the exact replacement shown in this preview.".into(),
                    },
                    details,
                    Operation::Replace {
                        path,
                        find,
                        replacement,
                        replace_all,
                    },
                )
            }
            "copy_file" | "move_file" => {
                let source = required(args, "source")?;
                let destination = required(args, "destination")?;
                let from = self.state(&source)?;
                ensure_file(&from)?;
                let to = self.state(&destination)?;
                ensure_file_or_missing(&to)?;
                if from.resolved == to.resolved {
                    bail!("Source and destination are the same file");
                }
                if to.kind != "missing"
                    && !args
                        .get("overwrite")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                {
                    bail!("Destination exists; set overwrite=true to propose replacing it");
                }
                let details = json!({"source":from.resolved,"destination":to.resolved,"source_sha256":from.sha256,"destination_sha256":to.sha256,"backup_directory":self.backup_root});
                expected.extend([from, to]);
                let moving = name == "move_file";
                (
                    format!("{} {source}", if moving { "Move" } else { "Copy" }),
                    "Recheck both paths and preserve existing content before changing files."
                        .into(),
                    details,
                    if moving {
                        Operation::Move {
                            source,
                            destination,
                        }
                    } else {
                        Operation::Copy {
                            source,
                            destination,
                        }
                    },
                )
            }
            "delete_file" => {
                let path = required(args, "path")?;
                let state = self.state(&path)?;
                ensure_file(&state)?;
                let details = json!({"path":state.resolved,"bytes":state.size,"sha256":state.sha256,"backup_directory":self.backup_root});
                expected.push(state);
                (
                    format!("Remove {path}"),
                    "Preserve a recoverable copy, then remove this file from the workspace.".into(),
                    details,
                    Operation::Delete { path },
                )
            }
            "make_directory" | "mkdir" => {
                let path = required(args, "path")?;
                let state = self.state(&path)?;
                if state.kind != "missing" {
                    bail!("The directory path already exists");
                }
                let details = json!({"path":state.resolved});
                expected.push(state);
                (
                    format!("Create {path}"),
                    "Create this directory and any missing parent directories.".into(),
                    details,
                    Operation::MakeDirectory { path },
                )
            }
            "run_command" => {
                let program = required(args, "program")?;
                let arguments = parse_string_array(args.get("args"))?;
                let cwd_value = args.get("cwd").and_then(Value::as_str).unwrap_or(".");
                let state = self.state(cwd_value)?;
                if state.kind != "directory" {
                    bail!("Command cwd must be an existing workspace directory");
                }
                let spec = CommandSpec {
                    program,
                    args: arguments,
                    cwd: Some(state.resolved.clone()),
                    timeout_secs: uint(args, "timeout_secs", 60, 1, 3600) as u64,
                    max_output_bytes: uint(args, "max_output_bytes", 64000, 1024, 2 * 1024 * 1024),
                    stdin_text: args
                        .get("stdin_text")
                        .map(|v| {
                            v.as_str()
                                .map(str::to_owned)
                                .context("stdin_text must be a string")
                        })
                        .transpose()?,
                    ..Default::default()
                };
                let mut spec = spec;
                spec.sandbox = self.command_sandbox()?;
                spec.validate()?;
                let details = command_preview(&spec);
                expected.push(state);
                let description = match &spec.sandbox {
                    Some(policy) => format!("Runs in a sandbox: it can change files only in the project folder, cannot see your home folder{}. Only the exact executable and arguments shown are approved.", if policy.network { ", and may use the network" } else { ", and has no network" }),
                    None => "This command is not sandboxed. It can read, change files, or use the network with your OS permissions. Only the exact executable and arguments shown are approved.".into(),
                };
                (
                    "Run local command".into(),
                    description,
                    details,
                    Operation::Command { spec },
                )
            }
            "run_python" => {
                let code = required(args, "code")?;
                if code.len() > 200_000 {
                    bail!("Python code exceeds 200,000 bytes");
                }
                let sandbox = self.command_sandbox()?;
                if sandbox.is_none() && !self.sandbox_is_off()? {
                    bail!("run_python needs the command sandbox, and bubblewrap is not available here. Install bubblewrap, or use run_command if you accept running unsandboxed. Nothing was run.");
                }
                let state = self.state(".")?;
                let spec = CommandSpec {
                    program: "python3".into(),
                    // Isolated mode: ignore PYTHON* variables and the user site directory.
                    args: vec!["-I".into(), "-B".into(), "-".into()],
                    cwd: Some(state.resolved.clone()),
                    timeout_secs: uint(args, "timeout_secs", 60, 1, 600) as u64,
                    max_output_bytes: 256 * 1024,
                    stdin_text: Some(code.clone()),
                    sandbox,
                    ..Default::default()
                };
                spec.validate()?;
                let mut details = command_preview(&spec);
                // The code is the stdin; show it once, readably, under "code".
                if let Some(map) = details.as_object_mut() {
                    map.remove("stdin_text");
                }
                details["code"] = json!(code);
                details["language"] = json!("python");
                expected.push(state);
                let description = match &spec.sandbox {
                    Some(policy) => format!("Run this Python code in the sandbox: it can change files only in the project folder{}.", if policy.network { " and may use the network" } else { " and has no network" }),
                    None => "The command sandbox is off: this Python code runs with your OS permissions.".into(),
                };
                (
                    "Run Python code".into(),
                    description,
                    details,
                    Operation::Command { spec },
                )
            }
            "fetch_url" => {
                self.require_network()?;
                let url = required(args, "url")?;
                integrations::validate_http_url(&url)?;
                let max_chars = uint(args, "max_chars", 16000, 1000, 4_000_000);
                (
                    "Fetch a page".into(),
                    "Send an HTTP request to this URL. Redirects may contact another host.".into(),
                    json!({"url":url,"max_chars":max_chars}),
                    Operation::FetchUrl { url, max_chars },
                )
            }
            "qa_lookup" => {
                self.require_network()?;
                let question = required(args, "question")?;
                if question.chars().count() > 1_000 {
                    bail!("Q&A question exceeds 1,000 characters");
                }
                let kind = args
                    .get("kind")
                    .and_then(Value::as_str)
                    .and_then(crate::qa::AnswerKind::parse);
                // A saved answer that is not due for a refresh is local data:
                // return it without approval and without contacting any website.
                let cache = self.qa_cache()?;
                let saved = cache.lookup(&question)?;
                if let Some(hit) = saved.as_ref().filter(|hit| !hit.needs_refresh()) {
                    cache.record_hit(&hit.id)?;
                    return Ok(PreparedTool::Immediate(Self::qa_saved_result(
                        hit,
                        &question,
                        "From the local Q&A library; no website was contacted. Cite sources as [n].",
                    )));
                }
                let browsers = self.qa_browsers()?;
                let searxng = self
                    .search_url
                    .lock()
                    .map_err(|_| anyhow::anyhow!("Search lock failed"))?
                    .clone()
                    .filter(|url| !url.trim().is_empty());
                if browsers.is_empty() && searxng.is_none() {
                    bail!(
                        "No text browser (lynx, w3m, elinks or links) is installed for Q&A lookups"
                    );
                }
                let names: Vec<&str> = browsers.iter().map(|b| b.name.as_str()).collect();
                let engine = match &searxng {
                    Some(url) => format!(
                        "your SearXNG server ({url}), falling back to DuckDuckGo with {}",
                        if names.is_empty() {
                            "no installed text browser".to_owned()
                        } else {
                            names.join(", ")
                        }
                    ),
                    None => format!("DuckDuckGo with {}", names.join(", falling back to ")),
                };
                let (title, why) = match &saved {
                    Some(hit) => (
                        "Refresh a saved Q&A answer",
                        format!(
                            " A saved answer from {} is due for a refresh; if the refresh fails, the saved answer is used.",
                            crate::qa::short_date(&hit.saved_at)
                        ),
                    ),
                    None => ("Look up a question on Q&A sites", String::new()),
                };
                (
                    title.into(),
                    format!(
                        "Send this question (restricted to Q&A sites) to {engine}, then read the best threads: Stack Exchange threads through its public API, other sites with the text browser. The result is kept in your Q&A library.{why}",
                    ),
                    json!({
                        "question": question,
                        "search_engine": match &searxng {
                            Some(url) => format!("SearXNG ({url}), then DuckDuckGo (html.duckduckgo.com)"),
                            None => "DuckDuckGo (html.duckduckgo.com)".to_owned(),
                        },
                        "sites": crate::qa::QA_SITES,
                        "browsers": names,
                        "stack_exchange": "api.stackexchange.com (public, keyless)",
                        "refreshes_saved_answer": saved.is_some(),
                    }),
                    Operation::QaLookup {
                        question,
                        browsers: browsers
                            .iter()
                            .map(|b| (b.name.clone(), b.program.to_string_lossy().into_owned()))
                            .collect(),
                        searxng,
                        kind,
                    },
                )
            }
            "web_search" => {
                self.require_network()?;
                let query = required(args, "query")?;
                if query.chars().count() > 2_048 {
                    bail!("Web-search query exceeds 2,048 characters");
                }
                let limit = uint(args, "limit", 10, 1, 20);
                let requested_provider = args
                    .get("provider")
                    .map(|value| {
                        value
                            .as_str()
                            .context("provider must be a configured provider ID")
                    })
                    .transpose()?;
                let mode = args
                    .get("mode")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|m| !m.is_empty());
                // "general" is the ordinary web search: it used to send research
                // questions to Wikipedia and Hacker News only, skipping the real
                // web provider. The specialised modes still pick their sources.
                let provider = match mode {
                    Some(mode) if mode != "general" => self.knowledge_provider(mode)?,
                    _ => self.search_provider(requested_provider)?,
                };
                match provider {
                    SearchProvider::Knowledge { sources } => {
                        let labels: Vec<&str> = sources
                            .iter()
                            .map(|s| crate::knowledge_search::label(s))
                            .collect();
                        let hosts: Vec<&str> = sources
                            .iter()
                            .map(|s| crate::knowledge_search::host(s))
                            .collect();
                        let all_local =
                            sources.iter().all(|s| crate::knowledge_search::is_local(s));
                        (
                            format!("Search {}", labels.join(" and ")),
                            if all_local {
                                format!(
                                    "Send this query to {} on this computer's home server. It stays on this computer; nothing else is sent or executed.",
                                    labels.join(" and ")
                                )
                            } else {
                                format!(
                                    "Send this query to {} (public APIs that need no account). Nothing else is sent or executed.",
                                    labels.join(" and ")
                                )
                            },
                            json!({"query":query,"limit":limit,"provider":"knowledge","sources":labels,"hosts":hosts}),
                            Operation::KnowledgeSearch {
                                query,
                                sources,
                                limit,
                            },
                        )
                    }
                    SearchProvider::Searxng { base_url } => {
                        let fallbacks: Vec<(String, String)> = self
                            .text_browser_fallback_candidates()?
                            .into_iter()
                            .map(|b| (b.name, b.program.to_string_lossy().into_owned()))
                            .collect();
                        let names: Vec<&str> = fallbacks.iter().map(|(n, _)| n.as_str()).collect();
                        let retry = if names.is_empty() {
                            String::new()
                        } else {
                            format!(
                                " If SearXNG fails, the same query is sent to DuckDuckGo with {}.",
                                names.join(", then ")
                            )
                        };
                        let local_sites = self.local_sites_url()?;
                        let local_note = local_sites.as_ref().map_or(String::new(), |url| {
                            format!(" Your own sites are searched too, from the index at {url}; that part stays on your network.")
                        });
                        (
                            "Search your sites and the web".into(),
                            format!("Send this query to the configured SearXNG instance and the search providers it uses.{retry}{local_note}"),
                            json!({"query":query,"provider":"searxng","server":base_url,"limit":limit,"fallback_browsers":names,"local_sites":local_sites}),
                            Operation::WebSearch {
                                query,
                                base_url,
                                limit,
                                fallbacks,
                                local_sites,
                            },
                        )
                    }
                    SearchProvider::TextBrowser { browser } => {
                        let spec = integrations::text_browser_search_command(&browser, &query)?;
                        let fallbacks: Vec<(String, String)> = self
                            .text_browser_fallback_candidates()?
                            .into_iter()
                            .filter(|other| other.name != browser.name)
                            .map(|other| (other.name, other.program.to_string_lossy().into_owned()))
                            .collect();
                        let fallback_names: Vec<&str> =
                            fallbacks.iter().map(|(name, _)| name.as_str()).collect();
                        let mut retry_note = if fallback_names.is_empty() {
                            String::new()
                        } else {
                            format!(
                                " If DuckDuckGo refuses {} or it fails, the same query is retried with {}.",
                                browser.name,
                                fallback_names.join(", then ")
                            )
                        };
                        // Disclosed here, so the approval covers it: if every browser
                        // fails, the query goes to these public sources instead.
                        let knowledge_fallback: Vec<String> = if self.knowledge_enabled() {
                            crate::knowledge_search::mode_sources("general").unwrap_or_default()
                        } else {
                            Vec::new()
                        };
                        let knowledge_labels: Vec<&str> = knowledge_fallback
                            .iter()
                            .map(|s| crate::knowledge_search::label(s))
                            .collect();
                        if !knowledge_labels.is_empty() {
                            retry_note.push_str(&format!(
                                " If every browser fails, the same query is sent to {} instead.",
                                knowledge_labels.join(" and ")
                            ));
                        }
                        (
                            format!("Search the web with {}", browser.name),
                            format!("Run {} to fetch DuckDuckGo's HTML results page for exactly this query. The query is sent to DuckDuckGo; no shell is used and nothing else is executed.{retry_note}", browser.name),
                            json!({
                                "query":query,
                                "limit":limit,
                                "provider":"text_browser",
                                "browser":browser.name,
                                "fallback_browsers":fallback_names,
                                "fallback_knowledge_sources":knowledge_labels,
                                "search_engine":"DuckDuckGo (html.duckduckgo.com)",
                                "command":std::iter::once(spec.program.clone()).chain(spec.args.iter().cloned()).collect::<Vec<_>>(),
                            }),
                            Operation::TextBrowserSearch {
                                browser: browser.name,
                                program: browser.program.to_string_lossy().into_owned(),
                                query,
                                limit,
                                fallbacks,
                                knowledge_fallback,
                            },
                        )
                    }
                    SearchProvider::Mcp {
                        config,
                        mapping,
                        connected,
                    } => {
                        let arguments = mapping.arguments(&query, limit)?;
                        let config_hash = hash(&serde_json::to_vec(&config)?);
                        let cwd = config.cwd.as_deref().unwrap_or(self.root());
                        let first_launch_may_download = !connected
                            && Path::new(&config.program)
                                .file_name()
                                .and_then(|name| name.to_str())
                                .is_some_and(|name| name == "npx");
                        let details = json!({
                            "query":query,
                            "limit":limit,
                            "provider":"mcp",
                            "server_id":config.id,
                            "server_name":config.name,
                            "connected":connected,
                            "tool":mapping.tool,
                            "arguments":arguments,
                            "launch_if_needed":if connected { Value::Null } else {
                                json!({
                                    "program":config.program,
                                    "args":config.args,
                                    "cwd":cwd,
                                    "environment_keys":config.env.keys().collect::<Vec<_>>(),
                                })
                            },
                            "first_launch_may_download_package":first_launch_may_download,
                        });
                        (
                            format!("Search the web with {}", config.name),
                            "Call the configured MCP web-search tool with exactly this query. If the server is not connected, start the previewed process first. The process and search provider have ordinary network access; no environment values are shown in the preview.".into(),
                            details,
                            Operation::McpWebSearch {
                                server_id: config.id,
                                tool_name: mapping.tool,
                                query,
                                limit,
                                arguments,
                                config_hash,
                            },
                        )
                    }
                }
            }
            "mcp_connect" => {
                let id = required(args, "server_id")?;
                let config = self.mcp_config(&id)?;
                if !config.enabled {
                    bail!("Enable this MCP server in settings first");
                }
                let cwd = config.cwd.as_deref().unwrap_or(self.root());
                let details = json!({"server_id":id,"name":config.name,"program":config.program,"args":config.args,"cwd":cwd,"environment_keys":config.env.keys().collect::<Vec<_>>()});
                (format!("Start MCP: {}",config.name),"The server is an external program with your OS permissions. Starting it is approved here; each tool call requires its own approval.".into(),details,Operation::McpConnect{config})
            }
            "mcp_call" => {
                let server_id = required(args, "server_id")?;
                let tool_name = required(args, "tool_name")?;
                let arguments = args
                    .get("arguments")
                    .filter(|v| v.is_object())
                    .context("MCP arguments must be an object")?
                    .clone();
                let config = self.mcp_config(&server_id)?;
                if !config.enabled {
                    bail!("MCP server is disabled");
                }
                let client = self.mcp_client(&server_id)?;
                if !client
                    .lock()
                    .map_err(|_| anyhow::anyhow!("MCP client lock failed"))?
                    .tools
                    .iter()
                    .any(|t| t.name == tool_name)
                {
                    bail!("MCP server did not advertise this tool");
                }
                let details = json!({"server_id":server_id,"tool":tool_name,"arguments":arguments});
                let config_hash = hash(&serde_json::to_vec(&config)?);
                (format!("MCP: {tool_name}"),"Call this external tool with exactly these arguments. External tools may change files or contact services.".into(),details,Operation::McpCall{server_id,tool_name,arguments,config_hash})
            }
            "consult_agent" => {
                let id = required(args, "agent_id")?;
                let task = required(args, "task")?;
                let context = match args.get("context") {
                    None => "",
                    Some(value) => value
                        .as_str()
                        .context("External agent context must be text")?,
                };
                let files: Vec<String> = args
                    .get("files")
                    .cloned()
                    .map(serde_json::from_value)
                    .transpose()
                    .context("External agent files must be an array of workspace-relative paths")?
                    .unwrap_or_default();
                if files.len() > 8 {
                    bail!("At most eight evidence files per external consultation");
                }
                let profile = self
                    .external_agents
                    .lock()
                    .map_err(|_| anyhow::anyhow!("External agent lock failed"))?
                    .get(&id)
                    .cloned()
                    .context("Unknown external agent profile")?;
                profile.validate()?;
                if !profile.enabled {
                    bail!("External agent is not enabled");
                }
                let mut documents = Vec::new();
                for path in files {
                    let before = self.state(&path)?;
                    ensure_file(&before)?;
                    let document = self.workspace.read_text(&path, 12_000)?;
                    if self.state(&path)? != before {
                        bail!("Evidence changed while it was being read: {path}");
                    }
                    documents.push(document);
                    expected.push(before);
                }
                let request = crate::external_agents::prepare(
                    &profile,
                    &task,
                    context,
                    &documents,
                    &self.external_runs,
                )?;
                let details = json!({"agent":profile.name,"may_use_network":profile.may_use_network,
                    "command":command_preview(&request.command),"complete_prompt":request.prompt,
                    "evidence":request.evidence,"run_directory":request.command.cwd,
                    "warning":"External CLI runs with your user permissions and inherited environment. Its own login, provider, tools and billing apply. This application does not enforce a process sandbox."});
                (format!("Consult {}", profile.name),
                    "Approve this exact task, evidence and CLI launch. No conversation history is sent implicitly.".into(),
                    details, Operation::ExternalAgent { request: Box::new(request) })
            }
            "run_adapter" => {
                let adapter_id = required(args, "adapter_id")?;
                let adapter = self
                    .adapters
                    .lock()
                    .map_err(|_| anyhow::anyhow!("Adapter lock failed"))?
                    .get(&adapter_id)
                    .cloned()
                    .context("Unknown adapter")?;
                let input = args
                    .get("input")
                    .and_then(Value::as_str)
                    .map(|p| self.state(p))
                    .transpose()?;
                if let Some(state) = &input {
                    ensure_file(state)?;
                }
                let output = args
                    .get("output")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                let output_state = output.as_deref().map(|p| self.state(p)).transpose()?;
                if let Some(state) = &output_state {
                    ensure_file_or_missing(state)?;
                }
                let text = args.get("text").and_then(Value::as_str);
                let spec = adapter.command(
                    input.as_ref().map(|s| s.resolved.as_path()),
                    output_state.as_ref().map(|s| s.resolved.as_path()),
                    text,
                    self.root(),
                )?;
                let details = command_preview(&spec);
                expected.extend(input);
                expected.extend(output_state);
                let config_hash = hash(&serde_json::to_vec(&adapter)?);
                (format!("Run {}",adapter.name),"Run the configured local adapter. Its executable is not sandboxed; this exact command and input are shown for approval.".into(),details,Operation::Adapter{adapter_id,spec,output,config_hash})
            }
            _ => bail!("Unknown workspace tool: {name}"),
        };
        let action = PendingAction {
            id: uuid::Uuid::new_v4().to_string(),
            title,
            description,
            details,
            operation,
            workspace: self.root().to_path_buf(),
            expected,
        };
        let mut pending = self
            .pending
            .lock()
            .map_err(|_| anyhow::anyhow!("Approval lock failed"))?;
        if pending.len() >= 32 {
            bail!("Too many outstanding actions; approve or deny the existing proposals first");
        }
        pending.insert(action.id.clone(), action.clone());
        Ok(PreparedTool::Approval(Box::new(action)))
    }
}
