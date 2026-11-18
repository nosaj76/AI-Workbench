//! ToolEngine execution responsibilities.
use super::*;

impl ToolEngine {
    pub fn deny(&self, id: &str) {
        if let Ok(mut p) = self.pending.lock() {
            p.remove(id);
        }
    }

    pub fn approve(&self, action: &PendingAction) -> Result<Value> {
        self.approve_cancellable(action, Arc::new(AtomicBool::new(false)))
    }

    pub fn approve_cancellable(
        &self,
        action: &PendingAction,
        cancel: Arc<AtomicBool>,
    ) -> Result<Value> {
        self.approve_streaming(action, cancel, None)
    }

    /// Execute an approved action. Command and adapter output is also forwarded
    /// to `sink` while it runs so the desktop can show live progress.
    pub fn approve_streaming(
        &self,
        action: &PendingAction,
        cancel: Arc<AtomicBool>,
        sink: Option<integrations::OutputSink>,
    ) -> Result<Value> {
        if cancel.load(Ordering::Relaxed) {
            self.deny(&action.id);
            bail!("Action cancelled before execution");
        }
        {
            let mut pending = self
                .pending
                .lock()
                .map_err(|_| anyhow::anyhow!("Approval lock failed"))?;
            let issued = pending
                .get(&action.id)
                .context("This action is no longer pending; create a new preview")?;
            if issued != action || action.workspace != self.root() {
                bail!("Action differs from its issued preview");
            }
            pending.remove(&action.id);
        }
        for before in &action.expected {
            let now = self.state(&before.path)?;
            if &now != before {
                bail!(
                    "{} changed after the preview. No action was performed; request a new preview.",
                    before.path
                );
            }
        }
        if cancel.load(Ordering::Relaxed) {
            bail!("Action cancelled before execution");
        }
        let result = self.dispatch_approved(action, cancel, sink);
        self.with_post_change_check(&action.operation, result)
    }

    fn dispatch_approved(
        &self,
        action: &PendingAction,
        cancel: Arc<AtomicBool>,
        sink: Option<integrations::OutputSink>,
    ) -> Result<Value> {
        match &action.operation {
            Operation::TerminalOpen { .. }
            | Operation::TerminalWrite { .. }
            | Operation::TerminalKey { .. }
            | Operation::TerminalResize { .. }
            | Operation::TerminalClose { .. } => {
                if !self.workspace_tools_enabled() {
                    bail!("Workspace tools were disabled after the terminal preview");
                }
                self.execute_terminal(&action.operation, cancel, sink)
            }
            Operation::Write { path, content } => {
                let backup = self.backup(action, path)?;
                self.write_atomic(path, content.as_bytes())?;
                Ok(
                    json!({"ok":true,"path":path,"absolute_path":self.workspace.resolve(path, false)?,"workspace":self.root(),"validation":"File bytes saved; program syntax and execution have not been checked by write_file.","bytes":content.len(),"sha256":hash(content.as_bytes()),"backup":backup}),
                )
            }
            Operation::Replace {
                path,
                find,
                replacement,
                replace_all,
            } => {
                let target = self.workspace.resolve(path, false)?;
                let before =
                    String::from_utf8(read_limited_beneath(self.root(), &target, MAX_FILE_BYTES)?)?;
                let count = before.matches(find).count();
                if count == 0 || (!replace_all && count != 1) {
                    bail!("Replacement no longer matches the approved preview");
                }
                let after = if *replace_all {
                    before.replace(find, replacement)
                } else {
                    before.replacen(find, replacement, 1)
                };
                let backup = self.backup(action, path)?;
                self.write_atomic(path, after.as_bytes())?;
                Ok(
                    json!({"ok":true,"path":path,"replacements":if *replace_all{count}else{1},"sha256":hash(after.as_bytes()),"backup":backup}),
                )
            }
            Operation::Copy {
                source,
                destination,
            }
            | Operation::Move {
                source,
                destination,
            } => {
                let source_path = self.workspace.resolve(source, false)?;
                let bytes = read_limited_beneath(self.root(), &source_path, MAX_FILE_BYTES)?;
                let source_backup = if matches!(&action.operation, Operation::Move { .. }) {
                    self.backup(action, source)?
                } else {
                    None
                };
                let destination_backup = self.backup(action, destination)?;
                self.write_atomic(destination, &bytes)?;
                fs::set_permissions(
                    self.workspace.resolve(destination, false)?,
                    fs::metadata(&source_path)?.permissions(),
                )?;
                if matches!(&action.operation, Operation::Move { .. }) {
                    let expected = action
                        .expected
                        .iter()
                        .find(|s| s.path == *source)
                        .context("Missing source snapshot")?;
                    if self.state(source)? != *expected {
                        bail!("Source changed during the copy. Destination was written, but source was preserved; inspect both paths.");
                    }
                    self.remove_confined(source)?;
                }
                Ok(
                    json!({"ok":true,"source":source,"destination":destination,"source_backup":source_backup,"destination_backup":destination_backup}),
                )
            }
            Operation::Delete { path } => {
                let backup = self
                    .backup(action, path)?
                    .context("Could not preserve the file; nothing was removed")?;
                let expected = action
                    .expected
                    .iter()
                    .find(|s| s.path == *path)
                    .context("Missing file snapshot")?;
                if self.state(path)? != *expected {
                    bail!("File changed while backing it up. Original was preserved.");
                }
                self.remove_confined(path)?;
                Ok(json!({"ok":true,"removed":path,"recoverable_copy":backup}))
            }
            Operation::GitCommit {
                message,
                paths,
                branch,
            } => self.execute_git_commit(message, paths, branch.as_deref()),
            Operation::MakeDirectory { path } => {
                let target = self.workspace.resolve(path, true)?;
                self.create_dir_confined(&target)?;
                Ok(json!({"ok":true,"directory":path}))
            }
            Operation::Command { spec } => {
                let result = integrations::run_command_streaming(
                    spec,
                    self.root(),
                    Some(cancel.clone()),
                    sink.clone(),
                )?;
                Ok(
                    json!({"ok":result.exit_code==Some(0)&&!result.timed_out&&!result.cancelled,"result":result}),
                )
            }
            Operation::FetchUrl {
                url,
                max_chars,
                local,
            } => {
                self.require_network()?;
                Ok(serde_json::to_value(integrations::fetch_url(
                    url, *max_chars, *local,
                )?)?)
            }
            Operation::WebSearch {
                query,
                base_url,
                limit,
                fallbacks,
                local_sites,
            } => {
                self.require_network()?;
                if self
                    .search_url
                    .lock()
                    .map_err(|_| anyhow::anyhow!("Search lock failed"))?
                    .as_ref()
                    != Some(base_url)
                    || self.local_sites_url()? != *local_sites
                {
                    bail!("Search configuration changed; request a new preview");
                }
                // The user's own sites come first: they are focused on the user's
                // work. A failure here never blocks the web search.
                let local = local_sites.as_ref().map(|url| {
                    match integrations::search_local_sites(url, query, *limit) {
                        Ok(results) => json!({
                            "source":"Your own sites on the home server (listed before web results)",
                            "index":url,
                            "result_count":results.len(),
                            "results":results,
                        }),
                        Err(error) => json!({"index":url,"error":format!("{error:#}")}),
                    }
                });
                let with_local = |mut value: Value| {
                    if let (Some(local), Some(object)) = (local.clone(), value.as_object_mut()) {
                        object.insert("local_sites".into(), local);
                    }
                    value
                };
                let searxng_error = match integrations::search_searxng(base_url, query, *limit) {
                    Ok(results) => {
                        let note = results.is_empty().then_some(
                            "SearXNG answered and every engine responded, but none found a match.",
                        );
                        return Ok(with_local(json!({
                            "ok":true,
                            "query":query,
                            "provider":{"type":"searxng","server":base_url},
                            "result_count":results.len(),
                            "results":results,
                            "note":note,
                        })));
                    }
                    Err(error) => format!("{error:#}"),
                };
                // Approved in the preview: retry through DuckDuckGo with the text
                // browsers that are still installed at the previewed paths.
                let installed = self.text_browser_fallback_candidates()?;
                let mut failures = vec![format!("searxng: {searxng_error}")];
                for (name, program) in fallbacks {
                    let Some(browser) = installed.iter().find(|b| {
                        b.name == *name && b.program.to_string_lossy() == program.as_str()
                    }) else {
                        continue;
                    };
                    if cancel.load(Ordering::Relaxed) {
                        bail!("Search cancelled");
                    }
                    match integrations::search_text_browser(
                        browser,
                        query,
                        *limit,
                        self.root(),
                        Some(cancel.clone()),
                    ) {
                        Ok(results) => {
                            return Ok(with_local(json!({
                                "ok":true,
                                "query":query,
                                "provider":{"type":"text_browser","browser":browser.name,"search_engine":"duckduckgo","note":"SearXNG failed, so the approved DuckDuckGo fallback answered."},
                                "failed_providers":failures,
                                "result_count":results.len(),
                                "results":results,
                            })));
                        }
                        Err(error) => failures.push(format!("{}: {error:#}", browser.name)),
                    }
                }
                // The web is unreachable, but local matches are still useful.
                if local
                    .as_ref()
                    .and_then(|l| l.get("result_count"))
                    .and_then(Value::as_u64)
                    .is_some_and(|n| n > 0)
                {
                    return Ok(with_local(json!({
                        "ok":true,
                        "query":query,
                        "note":"Every web search provider failed; only your own sites answered.",
                        "failed_providers":failures,
                        "result_count":0,
                        "results":[],
                    })));
                }
                bail!(
                    "Web search failed with every provider: {}",
                    failures.join("; ")
                )
            }
            Operation::TextBrowserSearch {
                browser,
                program,
                query,
                limit,
                fallbacks,
                knowledge_fallback,
            } => {
                self.require_network()?;
                let configured = self.configured_text_browser()?;
                let Some(configured) = configured.filter(|configured| {
                    configured.name == *browser
                        && configured.program.to_string_lossy() == program.as_str()
                }) else {
                    bail!("Text-browser search configuration changed; request a new preview");
                };
                // Try the configured browser, then each approved fallback that
                // is still installed at the previewed path.
                let installed = self.text_browser_fallback_candidates()?;
                let candidates = std::iter::once(configured).chain(fallbacks.iter().filter_map(
                    |(name, program)| {
                        installed
                            .iter()
                            .find(|b| {
                                b.name == *name && b.program.to_string_lossy() == program.as_str()
                            })
                            .cloned()
                    },
                ));
                let mut failures = Vec::new();
                for candidate in candidates {
                    if cancel.load(Ordering::Relaxed) {
                        bail!("Search cancelled");
                    }
                    match integrations::search_text_browser(
                        &candidate,
                        query,
                        *limit,
                        self.root(),
                        Some(cancel.clone()),
                    ) {
                        Ok(results) => {
                            return Ok(json!({
                                "ok":true,
                                "query":query,
                                "limit":limit,
                                "provider":{"type":"text_browser","browser":candidate.name,"search_engine":"duckduckgo"},
                                "failed_browsers":failures,
                                "result_count":results.len(),
                                "results":results,
                            }));
                        }
                        Err(error) => failures.push(format!("{}: {error:#}", candidate.name)),
                    }
                }
                // Approved in the preview: with every browser failed, the same query
                // goes to the listed public knowledge sources.
                if !knowledge_fallback.is_empty() && self.knowledge_enabled() {
                    let endpoints = self
                        .knowledge_endpoints
                        .lock()
                        .map_err(|_| anyhow::anyhow!("Knowledge endpoint lock failed"))?
                        .clone();
                    match crate::knowledge_search::search(&endpoints, query, knowledge_fallback, *limit) {
                        Ok(found) => {
                            return Ok(json!({
                                "ok":true,
                                "query":query,
                                "limit":limit,
                                "provider":{"type":"knowledge_fallback","sources":knowledge_fallback,"note":"The text browsers failed, so the approved public knowledge sources answered instead."},
                                "failed_browsers":failures,
                                "result_count":found.results.len(),
                                "results":found.results,
                                "source_reports":found.reports,
                            }));
                        }
                        Err(error) => bail!(
                            "Every text browser failed: {}; the public knowledge sources also failed: {error:#}",
                            failures.join("; ")
                        ),
                    }
                }
                bail!("Every text browser failed: {}", failures.join("; "))
            }
            Operation::KnowledgeSearch {
                query,
                sources,
                limit,
            } => {
                self.require_network()?;
                self.check_knowledge_sources(sources)
                    .context("Search configuration changed; request a new preview")?;
                let endpoints = self
                    .knowledge_endpoints
                    .lock()
                    .map_err(|_| anyhow::anyhow!("Knowledge endpoint lock failed"))?
                    .clone();
                let found = crate::knowledge_search::search(&endpoints, query, sources, *limit)?;
                Ok(json!({
                    "ok":true,
                    "query":query,
                    "limit":limit,
                    "provider":{"type":"knowledge","sources":sources},
                    "result_count":found.results.len(),
                    "results":found.results,
                    "source_reports":found.reports,
                }))
            }
            Operation::QaLookup {
                question,
                browsers,
                searxng,
                kind,
            } => {
                self.require_network()?;
                let configured = self
                    .search_url
                    .lock()
                    .map_err(|_| anyhow::anyhow!("Search lock failed"))?
                    .clone();
                if configured != *searxng {
                    bail!("Search configuration changed; request a new preview");
                }
                // Only browsers approved in the preview and still installed at
                // the same path may run.
                let available = self.qa_browsers()?;
                let approved: Vec<integrations::TextBrowser> = browsers
                    .iter()
                    .filter_map(|(name, program)| {
                        available
                            .iter()
                            .find(|b| {
                                b.name == *name && b.program.to_string_lossy() == program.as_str()
                            })
                            .cloned()
                    })
                    .collect();
                if approved.is_empty() && searxng.is_none() {
                    bail!("Text-browser configuration changed; request a new preview");
                }
                let cache = self.qa_cache()?;
                let saved = cache.lookup(question)?;
                match crate::qa::gather(
                    question,
                    searxng.as_deref(),
                    &approved,
                    3,
                    4_000,
                    self.root(),
                    None,
                    cancel.clone(),
                ) {
                    Ok(gathered) => {
                        let kind =
                            kind.unwrap_or(if crate::qa::obviously_time_sensitive(question) {
                                crate::qa::AnswerKind::TimeSensitive
                            } else {
                                crate::qa::AnswerKind::Changing
                            });
                        let topic = gathered
                            .sources
                            .iter()
                            .flat_map(|s| s.tags.first())
                            .next()
                            .cloned()
                            .unwrap_or_else(|| "general".into());
                        let entry = cache.store(
                            crate::qa::NewAnswer {
                                question,
                                summary: "",
                                short_question: "",
                                explanation: "",
                                answer: "",
                                sources: &gathered.sources,
                                model: "",
                                kind,
                                topic: &topic,
                            },
                            crate::qa::DEFAULT_REFRESH_DAYS,
                        )?;
                        Ok(json!({
                            "ok": true,
                            "cached": false,
                            "question": question,
                            "browser": gathered.browser,
                            "kind": entry.kind.as_str(),
                            "sources": gathered.sources,
                            "notes": gathered.notes,
                            "library_file": format!("{}/{}", crate::qa::LIBRARY_DIR, entry.path),
                            "note": "Answer from these sources and cite them as [n]. They are now kept in the Q&A library.",
                        }))
                    }
                    // Answers are always kept: a failed refresh returns the saved one.
                    Err(error) if saved.is_some() && !cancel.load(Ordering::Relaxed) => {
                        let hit = saved.expect("checked");
                        Ok(Self::qa_saved_result(
                            &hit,
                            question,
                            &format!(
                                "Could not refresh ({error:#}); this is the answer saved on {}. Say that it may be out of date.",
                                crate::qa::short_date(&hit.saved_at)
                            ),
                        ))
                    }
                    Err(error) => Err(error),
                }
            }
            Operation::McpWebSearch {
                server_id,
                tool_name,
                query,
                limit,
                arguments,
                config_hash,
            } => {
                self.require_network()?;
                let config = self.mcp_config(server_id)?;
                if !config.enabled || hash(&serde_json::to_vec(&config)?) != *config_hash {
                    bail!("MCP search configuration changed or is disabled; request a new preview");
                }
                let mapping = config
                    .web_search
                    .as_ref()
                    .context("MCP server is no longer configured for web search")?;
                if mapping.tool.as_str() != tool_name.as_str()
                    || mapping.arguments(query, *limit)? != *arguments
                {
                    bail!("MCP web-search mapping changed; request a new preview");
                }
                let client = self.connected_or_start_mcp(&config, cancel.clone())?;
                let mut client = client
                    .lock()
                    .map_err(|_| anyhow::anyhow!("MCP client lock failed"))?;
                if !client
                    .tools
                    .iter()
                    .any(|tool| tool.name.as_str() == tool_name.as_str())
                {
                    client.refresh_tools()?;
                }
                if !client
                    .tools
                    .iter()
                    .any(|tool| tool.name.as_str() == tool_name.as_str())
                {
                    let available = client
                        .tools
                        .iter()
                        .map(|tool| tool.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ");
                    bail!(
                        "MCP server did not advertise configured search tool {tool_name}. Available tools: {available}"
                    );
                }
                let result =
                    client.call_tool_cancellable(tool_name, arguments.clone(), cancel.clone())?;
                let ok = result.get("isError").and_then(Value::as_bool) != Some(true);
                Ok(json!({
                    "ok":ok,
                    "query":query,
                    "limit":limit,
                    "provider":{
                        "type":"mcp",
                        "server_id":server_id,
                        "server_name":config.name,
                        "tool":tool_name,
                    },
                    "results":result,
                }))
            }
            Operation::McpConnect { config } => {
                if self.mcp_config(&config.id)? != *config || !config.enabled {
                    bail!("MCP configuration changed or is disabled");
                }
                let client =
                    McpClient::connect_cancellable(config, self.root(), Some(cancel.clone()))?;
                let result = json!({"connected":true,"server_id":config.id,"server_info":client.server_info,"tools":client.tools});
                self.mcp_clients
                    .lock()
                    .map_err(|_| anyhow::anyhow!("MCP lock failed"))?
                    .insert(config.id.clone(), Arc::new(Mutex::new(client)));
                Ok(result)
            }
            Operation::McpCall {
                server_id,
                tool_name,
                arguments,
                config_hash,
            } => {
                let config = self.mcp_config(server_id)?;
                if !config.enabled || hash(&serde_json::to_vec(&config)?) != *config_hash {
                    bail!("MCP configuration changed or is disabled");
                }
                let client = self.mcp_client(server_id)?;
                let result = client
                    .lock()
                    .map_err(|_| anyhow::anyhow!("MCP client lock failed"))?
                    .call_tool_cancellable(tool_name, arguments.clone(), cancel.clone())?;
                Ok(result)
            }
            Operation::ExternalAgent { request } => {
                let current = self
                    .external_agents
                    .lock()
                    .map_err(|_| anyhow::anyhow!("External agent lock failed"))?
                    .get(&request.profile.id)
                    .cloned()
                    .context("External agent profile was removed")?;
                if !current.enabled || current.digest()? != request.profile_hash {
                    bail!("External agent configuration changed or was disabled after preview");
                }
                crate::external_agents::execute(request, &self.external_runs, cancel)
            }
            Operation::Adapter {
                adapter_id,
                spec,
                output,
                config_hash,
            } => {
                let adapter = self
                    .adapters
                    .lock()
                    .map_err(|_| anyhow::anyhow!("Adapter lock failed"))?
                    .get(adapter_id)
                    .cloned()
                    .context("Adapter was removed")?;
                if !adapter.enabled || hash(&serde_json::to_vec(&adapter)?) != *config_hash {
                    bail!("Adapter configuration changed or is disabled");
                }
                let backup = output
                    .as_deref()
                    .map(|p| self.backup(action, p))
                    .transpose()?
                    .flatten();
                let result = integrations::run_command_streaming(
                    spec,
                    self.root(),
                    Some(cancel.clone()),
                    sink.clone(),
                )?;
                let output_result = output.as_deref().map(|path| match self.state(path) {
                    Ok(state) if state.kind == "file" => {
                        let before = action.expected.iter().find(|s| s.path == path);
                        let status = match before {
                            Some(before) if before.kind == "missing" => "created",
                            Some(before) if before.sha256 == state.sha256 => "unchanged",
                            _ => "changed",
                        };
                        json!({"path":path,"exists":true,"status":status,"bytes":state.size,"sha256":state.sha256})
                    }
                    Ok(state) => json!({"path":path,"exists":false,"status":state.kind}),
                    Err(error) => json!({"path":path,"exists":false,"status":"unavailable","error":error.to_string()}),
                });
                let output_exists = output_result
                    .as_ref()
                    .is_none_or(|value| value["exists"] == true);
                Ok(
                    json!({"ok":result.exit_code==Some(0)&&!result.timed_out&&!result.cancelled&&output_exists,"result":result,"output":output,"output_result":output_result,"backup":backup}),
                )
            }
        }
    }

    fn connected_or_start_mcp(
        &self,
        config: &McpServerConfig,
        cancel: Arc<AtomicBool>,
    ) -> Result<Arc<Mutex<McpClient>>> {
        let existing = self
            .mcp_clients
            .lock()
            .map_err(|_| anyhow::anyhow!("MCP lock failed"))?
            .get(&config.id)
            .cloned();
        if let Some(client) = existing {
            let connected = {
                let mut state = client
                    .lock()
                    .map_err(|_| anyhow::anyhow!("MCP client lock failed"))?;
                state.is_connected()
            };
            if connected {
                return Ok(client);
            }
            self.mcp_clients
                .lock()
                .map_err(|_| anyhow::anyhow!("MCP lock failed"))?
                .remove(&config.id);
        }

        let client = Arc::new(Mutex::new(McpClient::connect_cancellable(
            config,
            self.root(),
            Some(cancel),
        )?));
        self.mcp_clients
            .lock()
            .map_err(|_| anyhow::anyhow!("MCP lock failed"))?
            .insert(config.id.clone(), client.clone());
        Ok(client)
    }
}
