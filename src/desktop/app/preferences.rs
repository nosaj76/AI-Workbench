//! Workbench preferences responsibilities.
use super::*;

impl Workbench {
    pub(super) fn commit_settings(
        &mut self,
        mut candidate: Settings,
        ctx: &egui::Context,
    ) -> Result<()> {
        if let Some(original) = &self.settings_draft_original {
            let mut before = original.clone();
            let mut now = self.settings.clone();
            // Session bookkeeping is not an editable preference.
            before.last_session = None;
            now.last_session = None;
            if serde_json::to_value(before)? != serde_json::to_value(now)? {
                bail!("Settings changed elsewhere while this draft was open. Discard the draft and reopen Settings before saving; no changes were applied.");
            }
        }
        candidate.last_session = self.settings.last_session.clone();
        candidate.profiles =
            serde_json::from_str(&self.profiles_json).context("Invalid profile JSON")?;
        candidate.prompts =
            serde_json::from_str(&self.prompts_json).context("Invalid prompt JSON")?;
        if candidate.profiles.is_empty() {
            bail!("Keep at least one profile");
        }
        candidate.selected_profile = candidate.selected_profile.min(candidate.profiles.len() - 1);
        candidate.validate()?;
        if self.demo {
            bail!("Demo mode does not save application settings");
        }
        // Persist the complete validated snapshot before changing running state.
        if self.settings_corrupt {
            candidate.save_after_recovery(&self.dir)?;
        } else {
            candidate.save(&self.dir)?;
        }
        let endpoint_changed = candidate.base_url != self.settings.base_url
            || candidate.allow_remote != self.settings.allow_remote;
        self.settings = candidate;
        self.settings_corrupt = false;
        self.manual_engine = None;
        self.advanced.invalidate_routing();
        self.apply_theme(ctx);
        self.status = "Settings saved".into();
        if endpoint_changed {
            self.version.clear();
            self.models.clear();
            self.running_models = Value::Null;
            self.refresh_models();
        }
        Ok(())
    }

    pub(super) fn settings_page(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Settings");
        let mut draft = self.settings_draft.take().unwrap_or_else(|| {
            self.settings_draft_original = Some(self.settings.clone());
            self.profiles_json =
                serde_json::to_string_pretty(&self.settings.profiles).unwrap_or_default();
            self.prompts_json =
                serde_json::to_string_pretty(&self.settings.prompts).unwrap_or_default();
            self.settings.clone()
        });
        let idle = !self.busy
            && !self.work_busy
            && !self.setup.is_busy()
            && !self.advanced.is_busy()
            && self.approval.is_none()
            && !self.demo;
        let mut save = false;
        let mut discard = false;
        ui.horizontal_wrapped(|ui| {
            save = ui
                .add_enabled(
                    idle,
                    egui::Button::new(RichText::new("Save changes").strong()),
                )
                .clicked();
            discard = ui.button("Discard changes").clicked();
            if ui
                .add_enabled(idle, egui::Button::new("Check saved connection"))
                .clicked()
            {
                self.refresh_models();
            }
        });
        ui.label("Connection and integration changes take effect only after Save changes. Leaving this page discards unsaved edits.");
        if self.settings_corrupt {
            ui.colored_label(ui.visuals().warn_fg_color,
                "The original settings file could not be loaded. Saving will preserve a recovery copy before replacing it.");
        }
        if !idle {
            ui.label("Finish active work before saving configuration. Appearance controls remain available in the toolbar.");
        }
        egui::ScrollArea::vertical().id_salt("settings_page").show(ui, |ui| {
            ui.heading("Tools and command approvals");
            ui.checkbox(&mut draft.tools_enabled, "Enable workspace tools");
            ui.small("Ask the AI to open Bash or use modernbasic. If no folder is selected, an explicit CLI request gets a dedicated working folder automatically.");
            ui.strong("Approve automatically in chat");
            ui.checkbox(&mut draft.auto_approve_web, "Web searches, page fetches and Q&A lookups");
            ui.checkbox(
                &mut draft.auto_approve_file_changes,
                "File changes in the project (backed up; restorable from saved versions)",
            );
            ui.checkbox(
                &mut draft.auto_approve_commands,
                "Sandboxed commands and terminals (run inside the command sandbox)",
            );
            ui.checkbox(
                &mut draft.auto_approve_unsandboxed,
                "Unsandboxed actions: commands and terminals without the sandbox, adapters, external agents and MCP calls",
            );
            ui.small("Includes opening Bash and sending text or keys to programs such as modernbasic. Reading an existing terminal needs no command approval. Chat Stop ends AI interaction; the terminal stays open until you close it.");
            ui.horizontal(|ui| {
                ui.label("Command sandbox");
                egui::ComboBox::from_id_salt("command_sandbox")
                    .selected_text(match draft.command_sandbox.as_str() {
                        "required" => "Required",
                        "off" => "Off",
                        _ => "Auto",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut draft.command_sandbox, "auto".to_string(), "Auto");
                        ui.selectable_value(&mut draft.command_sandbox, "required".to_string(), "Required");
                        ui.selectable_value(&mut draft.command_sandbox, "off".to_string(), "Off");
                    });
            });
            let sandbox_ready = crate::sandbox::available();
            ui.small(if sandbox_ready {
                "Commands and terminals the AI opens run in a bubblewrap sandbox: they can change files only in the project folder, see an empty home folder and get no network unless allowed below."
            } else {
                "Bubblewrap is not available, so Auto runs commands unsandboxed and Required refuses them. Install it with: sudo apt install bubblewrap"
            });
            ui.checkbox(&mut draft.sandbox_network, "Allow network access inside the sandbox");
            if draft.sandbox_network {
                ui.small(if crate::sandbox::network_isolation_available() {
                    "Internet only: pasta gives the sandbox its own network, and services on this computer and your LAN (such as Ollama) are blocked."
                } else {
                    "This includes services on this computer and your LAN, such as Ollama's API on 127.0.0.1:11434, which can delete or download models. Install passt (sudo apt install passt) to allow the internet only."
                });
            }
            ui.label("Folders the sandbox may read (one per line, ~/ allowed)");
            ui.add(egui::TextEdit::multiline(&mut draft.sandbox_read_only_paths).desired_rows(3).desired_width(f32::INFINITY));
            ui.horizontal(|ui| {
                ui.label("Check after file changes");
                ui.add(egui::TextEdit::singleline(&mut draft.check_after_edits).desired_width(260.0));
            });
            let check_help = match crate::tools::CheckMode::parse(&draft.check_after_edits) {
                Ok(crate::tools::CheckMode::Off) => "Off: the model sees only that files were saved.".to_string(),
                Ok(crate::tools::CheckMode::Auto) => "auto: after each approved change, runs cargo check, go build, or a syntax check of changed Python/JavaScript files in the sandbox (no network) and shows the model the errors. Needs the sandbox.".to_string(),
                Ok(crate::tools::CheckMode::Custom(argv)) => format!("Runs {} in the sandbox after each approved change.", argv.join(" ")),
                Err(e) => e.to_string(),
            };
            ui.small(check_help);
            let sandboxed = draft.command_sandbox != "off" && sandbox_ready;
            if draft.auto_approve_unsandboxed {
                ui.colored_label(
                    egui::Color32::from_rgb(200, 90, 40),
                    "Unsandboxed actions run with your user permissions without asking. The model can run anything your account can.",
                );
            }
            if draft.auto_approve_commands && !sandboxed {
                ui.small("With the sandbox unavailable or off, commands are unsandboxed and keep asking unless the unsandboxed option is also on.");
            }
            ui.small("Every automatic approval is still recorded in the action log. Background jobs and subagents keep asking or denying as before.");
            ui.separator();
            ui.heading("Connection and privacy");
            ui.label("Ollama URL");
            ui.add(egui::TextEdit::singleline(&mut draft.base_url).desired_width(f32::INFINITY));
            ui.checkbox(&mut draft.allow_remote, "Allow remote inference endpoints and cloud-backed models");
            ui.label("Remote inference sends the prompt and selected content to that service. Local-only is the default.");
            ui.checkbox(&mut draft.network_tools, "Enable explicit web fetch and search tools");
            ui.label("Optional SearXNG fallback URL (blank disables it)");
            ui.add(egui::TextEdit::singleline(&mut draft.searxng_url).desired_width(f32::INFINITY));
            ui.horizontal_wrapped(|ui| {
                ui.label("Web search depth:");
                for (value, label) in crate::settings::SEARCH_DEPTHS {
                    ui.radio_value(&mut draft.search_depth, value.to_string(), label);
                }
            });
            ui.small(match draft.search_depth.as_str() {
                "speed" => "Speed: one search, answered from result snippets. Fastest; least checked.",
                "quality" => "Quality: several searches, four or more sources read and compared. Slowest; most thorough.",
                _ => "Balanced: one or two searches, the two or three best pages read before answering.",
            });
            ui.checkbox(&mut draft.search_query_rewrite, "Turn Q&A questions into short search keywords first")
                .on_hover_text("The Q&A lookup asks the model for a keyword query before searching (falls back to removing filler words). The answer still uses your full question.");
            ui.label("Local sites address for self-hosted search services (blank turns them off)");
            ui.add(egui::TextEdit::singleline(&mut draft.local_sites_url).desired_width(f32::INFINITY));
            ui.small("Only the host is used; each site keeps its own port (8201, 8090, 8203, 8205).");
            ui.horizontal(|ui| {
                ui.label("Text-browser search");
                egui::ComboBox::from_id_salt("text_browser_search")
                    .selected_text(draft.text_browser.clone())
                    .show_ui(ui, |ui| {
                        for choice in ["auto", "lynx", "w3m", "elinks", "links", "off"] {
                            ui.selectable_value(&mut draft.text_browser, choice.to_string(), choice);
                        }
                    });
            });
            match crate::integrations::resolve_text_browser(&draft.text_browser) {
                Some(browser) => ui.label(format!(
                    "Keyless search uses {} ({}) to query DuckDuckGo when no MCP or SearXNG provider is set.",
                    browser.name, browser.program.display()
                )),
                None if draft.text_browser.trim().eq_ignore_ascii_case("off") => {
                    ui.label("Text-browser search is off.")
                }
                None => ui.colored_label(
                    ui.visuals().warn_fg_color,
                    "No matching text browser is installed. Install one, e.g.: sudo apt install lynx w3m",
                ),
            };
            ui.separator();
            ui.heading("Chat");
            ui.checkbox(&mut draft.enter_sends, "Enter sends a message")
                .on_hover_text("Shift+Enter adds a new line. When off, Ctrl+Enter sends and Enter adds a new line.");
            ui.checkbox(&mut draft.auto_titles, "Name new conversations automatically")
                .on_hover_text("After the first answer, the same model suggests a short title. A conversation you have renamed is never changed.");
            ui.separator();
            ui.heading("Appearance");
            ui.label("The Light and Dark toolbar buttons apply and save immediately. Scale below applies when settings are saved.");
            ui.horizontal(|ui| {
                ui.add(egui::Slider::new(&mut draft.font_scale, 0.75..=1.8).text("Size"));
                if ui.button("Reset to 100%").clicked() { draft.font_scale = 1.0; }
            });
            egui::CollapsingHeader::new("Generation and context").default_open(true).show(ui, |ui| {
                egui::Grid::new("generation_settings").num_columns(2).spacing([18.0,8.0]).show(ui, |ui| {
                    ui.label("Temperature"); ui.add(egui::Slider::new(&mut draft.temperature, 0.0..=2.0)); ui.end_row();
                    ui.label("Context tokens"); ui.add(egui::DragValue::new(&mut draft.num_ctx).range(2048..=1_048_576).speed(1024)); ui.end_row();
                    ui.label("Output token limit");
                    ui.horizontal(|ui| {
                        let mut unlimited = draft.num_predict <= 0;
                        if ui.checkbox(&mut unlimited, "No limit (never cut a reply off)").changed() {
                            draft.num_predict = if unlimited { -1 } else { 8192 };
                        }
                        if draft.num_predict > 0 {
                            ui.add(egui::DragValue::new(&mut draft.num_predict).range(128..=65536).speed(128));
                        }
                    });
                    ui.end_row();
                    ui.label("Top p"); ui.add(egui::Slider::new(&mut draft.top_p,0.0..=1.0)); ui.end_row();
                    ui.label("Repeat penalty"); ui.add(egui::Slider::new(&mut draft.repeat_penalty,0.0..=2.0)); ui.end_row();
                    ui.label("Keep model loaded"); ui.text_edit_singleline(&mut draft.keep_alive); ui.end_row();
                    ui.label("Thinking");
                    egui::ComboBox::from_id_salt("thinking").selected_text(&draft.thinking).show_ui(ui, |ui| {
                        for mode in ["default","on","off","low","medium","high"] {
                            ui.selectable_value(&mut draft.thinking, mode.into(), mode);
                        }
                    }); ui.end_row();
                    ui.label("Tool rounds"); ui.add(egui::DragValue::new(&mut draft.max_tool_rounds).range(1..=64)); ui.end_row();
                    ui.label("Attachment characters"); ui.add(egui::DragValue::new(&mut draft.max_context_chars).range(1000..=200000).speed(1000)); ui.end_row();
                });
                ui.label("Thinking modes depend on the model. Context and output budgets affect memory requirements.");
                ui.label("Response format: blank, json, or JSON schema");
                ui.add(egui::TextEdit::multiline(&mut draft.response_format).code_editor().desired_rows(2).desired_width(f32::INFINITY));
            });
            egui::CollapsingHeader::new("Instructions, memory and agents").show(ui, |ui| {
                ui.label("System instruction");
                ui.add(egui::TextEdit::multiline(&mut draft.system_prompt).desired_rows(5).desired_width(f32::INFINITY));
                ui.label("User-maintained preferences");
                ui.add(egui::TextEdit::multiline(&mut draft.memory).desired_rows(4).desired_width(f32::INFINITY));
                ui.checkbox(&mut draft.automatic_memory, "Automatically learn and recall sourced memories");
                ui.checkbox(&mut draft.knowledge_notes, "Build knowledge notes (one file per topic) from facts in web results and research, credited to each page");
                ui.checkbox(&mut draft.knowledge_search, "Allow public knowledge sources (Wikipedia, Hacker News, arXiv, Crossref, GitHub) for web search");
                ui.horizontal_wrapped(|ui| {
                    ui.label("Recalled memories"); ui.add(egui::DragValue::new(&mut draft.memory_limit).range(1..=64));
                    ui.label("Concurrent agents"); ui.add(egui::DragValue::new(&mut draft.agent_max_workers).range(1..=4));
                    ui.label("Tasks per team"); ui.add(egui::DragValue::new(&mut draft.agent_max_tasks).range(1..=8));
                });
                ui.checkbox(&mut draft.agents_enabled, "Allow supervised subagent delegation");
            });
            egui::CollapsingHeader::new("Knowledge and routing").show(ui, |ui| {
                ui.label("Installed embedding model");
                ui.add(egui::TextEdit::singleline(&mut draft.embedding_model).desired_width(f32::INFINITY));
                ui.checkbox(&mut draft.retrieval_enabled, "Retrieve indexed evidence for chat");
                ui.checkbox(&mut draft.model_routing, "Automatically select specialist models");
                ui.label("Role preferences and limits (JSON); a form is also available in Activities / Model routing.");
                ui.add(egui::TextEdit::multiline(&mut draft.routing_json).code_editor().desired_rows(5).desired_width(f32::INFINITY));
            });
            egui::CollapsingHeader::new("Profiles and prompt library").show(ui, |ui| {
                ui.label("Profiles (JSON array)");
                ui.add(egui::TextEdit::multiline(&mut self.profiles_json).code_editor().desired_rows(6).desired_width(f32::INFINITY));
                ui.label("Prompts (JSON array)");
                ui.add(egui::TextEdit::multiline(&mut self.prompts_json).code_editor().desired_rows(6).desired_width(f32::INFINITY));
            });
            egui::CollapsingHeader::new("Speech and transcription").show(ui, |ui| {
                ui.label("Speech executable"); ui.text_edit_singleline(&mut draft.tts_program);
                ui.label("Speech arguments (JSON array); text is passed on standard input");
                ui.add(egui::TextEdit::singleline(&mut draft.tts_args).desired_width(f32::INFINITY));
                ui.label("Transcription executable"); ui.text_edit_singleline(&mut draft.stt_program);
                ui.label("Transcription arguments (JSON array); use {input} for the audio path");
                ui.add(egui::TextEdit::multiline(&mut draft.stt_args).code_editor().desired_rows(3).desired_width(f32::INFINITY));
                ui.label("Programs, voices and model files are separate local installations. Launches require review.");
            });
            egui::CollapsingHeader::new("Local instruction skills").show(ui, |ui| {
                ui.label(if draft.skill_name.is_empty() { "No active skill" } else { &draft.skill_name });
                if ui.add_enabled(idle, egui::Button::new("Load a skill folder...")).clicked() {
                    self.work("Choose a skill folder", |tx, _| {
                        if let Some(path) = rfd::FileDialog::new().set_title("Choose a folder containing SKILL.md").pick_folder() {
                            tx.send(WorkEvent::Skill(crate::automation::load_skill(&path)?))?;
                        }
                        Ok(())
                    });
                }
                if ui.button("Clear selected skill on save").clicked() { draft.skill_name.clear(); draft.skill_prompt.clear(); }
                ui.label("Instructions are reviewed before activation. Loading a skill does not execute scripts or install dependencies.");
            });
            egui::CollapsingHeader::new("External agents: Hermes, Codex, Claude and Gemini").show(ui, |ui| {
                Self::external_agent_profiles_ui(ui, &mut draft.external_agents_json);
            });
            egui::CollapsingHeader::new("MCP servers and web search").show(ui, |ui| {
                Self::mcp_servers_ui(ui, &mut draft);
            });
            ui.separator();
            ui.label(RichText::new(format!("Local data: {}", self.dir.display())).monospace());
        });
        if discard {
            self.settings_draft = None;
            self.settings_draft_original = None;
            self.status = "Unsaved settings discarded".into();
        } else if save {
            match self.commit_settings(draft.clone(), ctx) {
                Ok(()) => {
                    self.settings_draft = None;
                    self.settings_draft_original = None;
                }
                Err(error) => {
                    self.settings_draft = Some(draft);
                    self.error = Some(format!("{error:#}"));
                }
            }
        } else {
            self.settings_draft = Some(draft);
        }
    }

    pub(super) fn mcp_servers_ui(ui: &mut egui::Ui, draft: &mut Settings) {
        ui.checkbox(
            &mut draft.mcp_enabled,
            "Enable configured stdio MCP servers",
        );
        ui.label("A server launch and every external tool call still require a concrete approval. A web-search mapping lets chat use one native web_search tool without reasoning through generic MCP calls.");

        let parsed = serde_json::from_str::<Vec<crate::mcp::McpServerConfig>>(&draft.mcp_json);
        let mut servers = match parsed {
            Ok(servers) => servers,
            Err(error) => {
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    format!("Repair the MCP JSON before using presets: {error}"),
                );
                ui.add(
                    egui::TextEdit::multiline(&mut draft.mcp_json)
                        .code_editor()
                        .desired_rows(9)
                        .desired_width(f32::INFINITY),
                );
                return;
            }
        };

        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            for preset in &crate::mcp::SEARCH_PRESETS {
                if ui
                    .button(format!("Add or repair {} preset", preset.provider))
                    .on_hover_text(format!("Official {} MCP server, pinned to {}. Keeps your key and on/off choice.", preset.provider, preset.package))
                    .clicked()
                {
                    let Some(mut fresh) = crate::mcp::McpServerConfig::search_preset(preset.id) else {
                        continue;
                    };
                    if let Some(existing) = servers.iter().find(|server| server.id == preset.id) {
                        fresh.enabled = existing.enabled;
                        if let Some(key) = existing.env.get(preset.key_env) {
                            fresh.env.insert(preset.key_env.into(), key.clone());
                        }
                    }
                    match servers.iter().position(|server| server.id == preset.id) {
                        Some(index) => servers[index] = fresh,
                        None => servers.push(fresh),
                    }
                    changed = true;
                }
            }
        });

        for server in servers.iter_mut() {
            let Some(preset) = crate::mcp::search_preset(&server.id) else {
                continue;
            };
            ui.group(|ui| {
                ui.strong(format!("{} MCP", preset.provider));
                ui.label(format!("Pinned package: {} (started with {})", preset.package, preset.launcher));
                let enabled = ui.checkbox(
                    &mut server.enabled,
                    format!("Enable {} as an MCP web-search provider", preset.provider),
                );
                if enabled.changed() {
                    changed = true;
                    if server.enabled {
                        draft.mcp_enabled = true;
                        draft.network_tools = true;
                    }
                }
                ui.horizontal_wrapped(|ui| {
                    ui.label(format!("{} API key (stored in your local settings.json and masked in this field).", preset.provider));
                    ui.hyperlink_to("Get a key", preset.key_url);
                });
                let key_is_empty = {
                    let key = server.env.entry(preset.key_env.into()).or_default();
                    if ui
                        .add(
                            egui::TextEdit::singleline(key)
                                .password(true)
                                .desired_width(f32::INFINITY),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                    key.trim().is_empty()
                };
                if server.enabled && key_is_empty {
                    ui.colored_label(
                        ui.visuals().warn_fg_color,
                        format!("Enter {} before saving an enabled {} server.", preset.key_env, preset.provider),
                    );
                }
                if server.enabled && (!draft.mcp_enabled || !draft.network_tools) {
                    ui.colored_label(
                        ui.visuals().warn_fg_color,
                        "Both MCP servers and network tools must remain enabled for native web_search to appear.",
                    );
                }
                if crate::diagnostics::find_executable(&server.program).is_none() {
                    let install = match preset.launcher {
                        "uvx" => " Install uv (it provides uvx), e.g. `pipx install uv` or your distribution's uv package.",
                        "npx" => " Install Node.js and npm (they provide npx).",
                        _ => "",
                    };
                    ui.colored_label(
                        ui.visuals().warn_fg_color,
                        format!("Executable {:?} is not currently available on PATH.{install}", server.program),
                    );
                }
                ui.label(format!(
                    "The first approved search may let {} download this pinned package. Search queries are then sent to {}'s API; local Ollama inference remains separate.",
                    preset.launcher, preset.provider
                ));
                if let Err(error) = server.validate() {
                    ui.colored_label(ui.visuals().warn_fg_color, error.to_string());
                }
            });
        }

        if changed {
            draft.mcp_json =
                serde_json::to_string_pretty(&servers).unwrap_or_else(|_| draft.mcp_json.clone());
        }
        egui::CollapsingHeader::new("Advanced raw server configuration (may reveal API keys)")
            .show(ui, |ui| {
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    "Raw MCP JSON contains stored environment values in plain text. Expand and edit it only when advanced configuration is necessary.",
                );
                ui.add(
                    egui::TextEdit::multiline(&mut draft.mcp_json)
                        .code_editor()
                        .desired_rows(9)
                        .desired_width(f32::INFINITY),
                );
            });
    }

    pub(super) fn external_agent_profiles_ui(ui: &mut egui::Ui, text: &mut String) {
        ui.label("Optional CLI agents. All presets start disabled. Install and authenticate each CLI separately; the main application never requires provider login.");
        ui.label("Each launch has an exact prompt/command preview. CLI processes have your OS permissions and may use paid remote services; a prompt is not a sandbox.");
        if ui.button("Reset this draft to disabled presets").clicked() {
            *text = serde_json::to_string_pretty(&crate::external_agents::AgentProfile::presets())
                .unwrap_or_else(|_| "[]".into());
        }
        let mut profiles: Vec<crate::external_agents::AgentProfile> =
            match serde_json::from_str(text) {
                Ok(profiles) => profiles,
                Err(error) => {
                    ui.label(format!("Repair the profile JSON before saving: {error}"));
                    ui.add(
                        egui::TextEdit::multiline(text)
                            .code_editor()
                            .desired_rows(6)
                            .desired_width(f32::INFINITY),
                    );
                    return;
                }
            };
        for profile in &mut profiles {
            ui.push_id(profile.id.clone(), |ui| {
                ui.group(|ui| {
                    ui.strong(&profile.name);
                    ui.horizontal_wrapped(|ui| {
                        ui.checkbox(&mut profile.enabled, "Enabled");
                        ui.checkbox(&mut profile.may_use_network, "This CLI may contact a provider");
                        ui.checkbox(&mut profile.allow_network, "I allow sending the reviewed task and evidence");
                    });
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Executable"); ui.text_edit_singleline(&mut profile.program);
                        ui.label("Timeout (seconds)"); ui.add(egui::DragValue::new(&mut profile.timeout_secs).range(1..=3600));
                    });
                    egui::CollapsingHeader::new("Command arguments and input").show(ui, |ui| {
                        ui.checkbox(&mut profile.prompt_on_stdin, "Send complete task on standard input");
                        ui.label("Each row is one literal argument, not a shell command. Without stdin, use one separate {prompt} argument.");
                        let mut remove = None;
                        for (index, argument) in profile.args.iter_mut().enumerate() {
                            ui.push_id(index, |ui| {
                                ui.horizontal(|ui| {
                                    ui.add(egui::TextEdit::singleline(argument).desired_width((ui.available_width() - 80.0).max(100.0)));
                                    if ui.small_button("Remove").clicked() { remove = Some(index); }
                                });
                            });
                        }
                        if let Some(index) = remove { profile.args.remove(index); }
                        if ui.button("Add argument").clicked() { profile.args.push(String::new()); }
                    });
                    if let Err(error) = profile.validate() {
                        ui.colored_label(ui.visuals().warn_fg_color, error.to_string());
                    }
                });
            });
        }
        *text = serde_json::to_string_pretty(&profiles).unwrap_or_else(|_| "[]".into());
    }
}
