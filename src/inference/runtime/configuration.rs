use super::*;

pub fn configured_engine(
    settings: &Settings,
    workspace: crate::files::Workspace,
) -> Result<ToolEngine> {
    let engine = ToolEngine::new(workspace);
    engine.set_network_enabled(settings.network_tools);
    engine.set_knowledge_search(settings.knowledge_search);
    if settings.network_tools && !settings.searxng_url.trim().is_empty() {
        engine.set_search_url(Some(settings.searxng_url.clone()))?;
    }
    if settings.network_tools && !settings.local_sites_url.trim().is_empty() {
        engine.set_local_sites_url(Some(settings.local_sites_url.clone()))?;
        engine.set_knowledge_endpoints(crate::knowledge_search::Endpoints::for_local_sites(
            &settings.local_sites_url,
        )?);
    }
    if settings.network_tools {
        engine.set_text_browser(&settings.text_browser)?;
    }
    if settings.mcp_enabled {
        let servers: Vec<crate::mcp::McpServerConfig> = serde_json::from_str(&settings.mcp_json)
            .context("MCP configuration must be a JSON array of servers")?;
        for server in servers {
            engine.register_mcp(server)?;
        }
    }
    for (id, name, kind, program, args, stdin, timeout) in [
        (
            "speech",
            "Configured text to speech",
            crate::integrations::AdapterKind::TextToSpeech,
            &settings.tts_program,
            &settings.tts_args,
            true,
            120,
        ),
        (
            "transcribe",
            "Configured speech to text",
            crate::integrations::AdapterKind::SpeechToText,
            &settings.stt_program,
            &settings.stt_args,
            false,
            600,
        ),
    ] {
        if !program.trim().is_empty() {
            engine.register_adapter(crate::integrations::CommandAdapter {
                id: id.into(),
                name: name.into(),
                kind,
                enabled: true,
                program: program.clone(),
                args: serde_json::from_str(args)
                    .with_context(|| format!("Invalid argument array for {name}"))?,
                text_to_stdin: stdin,
                timeout_secs: timeout,
            })?;
        }
    }
    for profile in crate::external_agents::parse_profiles(&settings.external_agents_json)? {
        engine.register_external_agent(profile)?;
    }
    Ok(engine)
}

/// Build a restricted tool engine for chats without a selected project. The
/// private working directory exists only to give approved stdio MCP processes a
/// stable current directory. Local file, command, generic MCP, adapter, and
/// external-agent tools are hidden and rejected; mapped MCP configuration is
/// reachable only through the native `web_search` operation.
pub fn configured_network_engine(settings: &Settings, root: PathBuf) -> Result<ToolEngine> {
    std::fs::create_dir_all(&root).context("Create restricted network-tool directory")?;
    let workspace = crate::files::Workspace::new(root)?;
    let engine = ToolEngine::new(workspace);
    engine.set_workspace_tools_enabled(false);
    engine.set_network_enabled(settings.network_tools);
    engine.set_knowledge_search(settings.knowledge_search);
    if settings.network_tools && !settings.searxng_url.trim().is_empty() {
        engine.set_search_url(Some(settings.searxng_url.clone()))?;
    }
    if settings.network_tools && !settings.local_sites_url.trim().is_empty() {
        engine.set_local_sites_url(Some(settings.local_sites_url.clone()))?;
        engine.set_knowledge_endpoints(crate::knowledge_search::Endpoints::for_local_sites(
            &settings.local_sites_url,
        )?);
    }
    if settings.network_tools {
        engine.set_text_browser(&settings.text_browser)?;
    }
    if settings.mcp_enabled {
        let servers: Vec<crate::mcp::McpServerConfig> = serde_json::from_str(&settings.mcp_json)
            .context("MCP configuration must be a JSON array of servers")?;
        for server in servers {
            engine.register_mcp(server)?;
        }
    }
    Ok(engine)
}

pub fn options(s: &Settings) -> Result<ChatOptions> {
    let think = match s.thinking.trim() {
        "" | "default" => None,
        "on" | "true" => Some(Value::Bool(true)),
        "off" | "false" => Some(Value::Bool(false)),
        other => Some(Value::String(other.to_string())),
    };
    let format = match s.response_format.trim() {
        "" => None,
        "json" => Some(Value::String("json".into())),
        raw => Some(
            serde_json::from_str(raw)
                .context("Response format must be blank, json, or a valid JSON schema")?,
        ),
    };
    Ok(ChatOptions {
        temperature: s.temperature,
        num_ctx: s.num_ctx,
        num_predict: s.num_predict,
        top_p: s.top_p,
        repeat_penalty: s.repeat_penalty,
        seed: None,
        think,
        format,
        keep_alive: if crate::routing::config(s)?.single_model_mode {
            "0".into()
        } else {
            s.keep_alive.clone()
        },
    })
}
