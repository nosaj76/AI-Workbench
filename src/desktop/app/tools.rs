//! Workbench tools responsibilities.
use super::*;

impl Workbench {
    pub(super) fn get_engine(&mut self) -> Result<ToolEngine> {
        let workspace = self.workspace()?;
        if self
            .manual_engine
            .as_ref()
            .is_some_and(|e| e.root() != workspace.root())
        {
            self.manual_engine = None;
        }
        if self.manual_engine.is_none() {
            let terminals = self.advanced.terminal_manager_for(Some(workspace.root()));
            let mut engine = runtime::configured_engine(&self.settings, workspace)?;
            engine.set_terminal_manager(terminals);
            self.manual_engine = Some(engine);
        }
        Ok(self.manual_engine.as_ref().unwrap().clone())
    }

    pub(super) fn get_chat_engine(&mut self) -> Result<Option<ToolEngine>> {
        if !self.settings.tools_enabled {
            return Ok(None);
        }
        if self.session.project.is_some() {
            return self.get_engine().map(Some);
        }
        if !self.settings.network_tools {
            return Ok(None);
        }
        let root = self.dir.join("restricted-tools");
        if self
            .manual_engine
            .as_ref()
            .is_some_and(|engine| engine.root() != root)
        {
            self.manual_engine = None;
        }
        if self.manual_engine.is_none() {
            self.manual_engine = Some(runtime::configured_network_engine(&self.settings, root)?);
        }
        Ok(self.manual_engine.clone())
    }

    pub(super) fn manual_tool(&mut self, name: &str, args: Value) {
        if self.busy || self.work_busy || self.approval.is_some() {
            self.error =
                Some("Finish or stop the current action before starting another tool".into());
            return;
        }
        match self.get_engine() {
            Ok(engine) => {
                let name = name.to_string();
                self.work("Preparing action…", move |tx, cancel| {
                    let prepared = engine.prepare_cancellable(&name, &args, cancel)?;
                    tx.send(WorkEvent::Prepared { engine, prepared })?;
                    Ok(())
                });
            }
            Err(e) => self.error = Some(format!("{e:#}")),
        }
    }

    pub(super) fn approve(&mut self, allowed: bool) {
        if let Some(approval) = self.approval.take() {
            match approval.origin {
                ApprovalOrigin::Chat(reply) => {
                    let _ = reply.send(allowed);
                }
                ApprovalOrigin::Manual(engine) => {
                    let action = approval.action;
                    let _ = self.store.audit(
                        &self.session.id,
                        if allowed { "Approved" } else { "Denied" },
                        &format!("{}\n{}", action.title, action.details),
                    );
                    if allowed {
                        self.work("Running approved action…", move |tx, cancel| {
                            let value = engine.approve_cancellable(&action, cancel)?;
                            tx.send(WorkEvent::Text(serde_json::to_string_pretty(&value)?))?;
                            Ok(())
                        });
                    } else {
                        engine.deny(&action.id);
                        self.status = "Action denied".into();
                    }
                }
            }
        }
    }

    pub(super) fn tools_page(&mut self, ui: &mut egui::Ui) {
        ui.heading("Workspace tools & integrations");
        ui.label("Read tools operate within the chosen workspace. File changes, commands, network access, and external tool calls show a concrete approval preview.");
        ui.horizontal(|ui| {
            ui.label("The AI can keep Bash and interactive CLI programs open between tool calls.");
            if ui.button("Open terminal panel").clicked() {
                self.advanced.section = crate::advanced_ui::Section::Terminal;
                self.page = Page::Advanced;
            }
        });
        if self.session.project.is_none() {
            ui.label(
                RichText::new("Choose a workspace folder to use tools.")
                    .color(ui.visuals().warn_fg_color),
            );
        }
        egui::ScrollArea::vertical().id_salt("tools_page").show(ui,|ui|{
            self.external_agents_panel(ui);
            ui.separator();

            egui::CollapsingHeader::new("Available workspace tool definitions").show(ui,|ui|{if let Ok(ws)=self.workspace(){let engine=runtime::configured_engine(&self.settings,ws).ok();ui.add(egui::Label::new(RichText::new(serde_json::to_string_pretty(&engine.map(|e|e.definitions()).unwrap_or_default()).unwrap_or_default()).monospace().size(12.0)).selectable(true));}});
            ui.heading("Run a tool manually");ui.horizontal(|ui|{ui.label("Tool name");ui.text_edit_singleline(&mut self.tool_name);});ui.label("Arguments (JSON)");ui.add(egui::TextEdit::multiline(&mut self.tool_args).code_editor().desired_rows(5).desired_width(f32::INFINITY));
            if ui.add_enabled(!self.busy&&!self.work_busy&&self.approval.is_none(),egui::Button::new("Prepare tool action")).clicked(){match serde_json::from_str::<Value>(&self.tool_args){Ok(args)=>self.manual_tool(&self.tool_name.clone(),args),Err(e)=>self.error=Some(format!("Invalid tool arguments: {e}"))}}
            ui.separator();ui.heading("Run a local command");ui.label("Commands run with your user account and can affect files beyond the workspace. Review the executable and every argument before approval.");
            ui.horizontal(|ui|{ui.label("Executable");ui.text_edit_singleline(&mut self.command_program);});ui.label("Arguments as a JSON array (no implicit shell)");ui.add(egui::TextEdit::singleline(&mut self.command_args).desired_width(f32::INFINITY));
            if ui.add_enabled(!self.busy&&!self.work_busy&&self.approval.is_none(),egui::Button::new("Preview command…")).clicked(){match serde_json::from_str::<Vec<String>>(&self.command_args){Ok(args)=>self.manual_tool("run_command",json!({"program":self.command_program,"args":args,"timeout_secs":60})),Err(e)=>self.error=Some(format!("Arguments must be a JSON string array: {e}"))}}
            if !self.tool_result.is_empty(){ui.separator();ui.heading("Last result");ui.add(egui::Label::new(RichText::new(&self.tool_result).monospace().size(12.0)).selectable(true));if ui.button("Attach result to conversation").clicked(){self.attachments.push(Attachment{name:"Tool result".into(),source:"Explicitly attached tool result".into(),text:self.tool_result.clone(),image:None});self.page=Page::Chat;}}
            ui.separator();ui.horizontal(|ui|{ui.heading("Action log");if ui.button("Refresh log").clicked(){self.audit=self.store.recent_audit().unwrap_or_default();}});for entry in &self.audit{ui.label(RichText::new(entry).monospace().size(12.0));ui.separator();}
        });
    }

    pub(super) fn external_agents_panel(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Consult an external agent").default_open(true).show(ui, |ui| {
            ui.label("Send a concrete question and selected evidence to Hermes, Codex, Claude or Gemini. The actual prompt is reviewed before launch.");
            let profiles = match crate::external_agents::parse_profiles(&self.settings.external_agents_json) {
                Ok(profiles) => profiles,
                Err(error) => { ui.label(format!("External agent settings: {error:#}")); return; }
            };
            ui.horizontal_wrapped(|ui| {
                egui::ComboBox::from_id_salt("external_agent_picker")
                    .selected_text(&self.external_agent_id)
                    .show_ui(ui, |ui| {
                        for profile in &profiles {
                            ui.selectable_value(&mut self.external_agent_id, profile.id.clone(),
                                format!("{}{}", profile.name, if profile.enabled { "" } else { " (disabled)" }));
                        }
                    });
                if ui.button("Configure agents...").clicked() { self.page = Page::Settings; }
            });
            let selected = profiles.iter().find(|p| p.id == self.external_agent_id);
            if let Some(profile) = selected {
                if crate::diagnostics::find_executable(&profile.program).is_none() {
                    ui.label(format!("{} is not on PATH. Install/authenticate the CLI or configure its absolute path.", profile.program));
                }
                if !profile.enabled { ui.label("This provider is disabled in Settings. Nothing is launched just by opening this panel."); }
            }
            ui.label("Task / question");
            ui.add(egui::TextEdit::multiline(&mut self.external_task).desired_rows(3).desired_width(f32::INFINITY));
            ui.label("Evidence files: one workspace-relative path per line (maximum eight)");
            ui.add(egui::TextEdit::multiline(&mut self.external_files).desired_rows(2).desired_width(f32::INFINITY));
            if !self.preview_path.is_empty() && ui.button("Include currently opened file").clicked()
                && !self.external_files.lines().any(|line| line == self.preview_path) {
                if !self.external_files.is_empty() { self.external_files.push('\n'); }
                self.external_files.push_str(&self.preview_path);
            }
            egui::CollapsingHeader::new("Additional context to send").show(ui, |ui| {
                ui.add(egui::TextEdit::multiline(&mut self.external_context).desired_rows(4).desired_width(f32::INFINITY));
            });
            let ready = selected.is_some_and(|p| p.enabled) && !self.external_task.trim().is_empty()
                && self.session.project.is_some() && !self.busy && !self.work_busy && self.approval.is_none();
            if self.session.project.is_none() { ui.label("Select a workspace before preparing a consultation."); }
            if ui.add_enabled(ready, egui::Button::new("Review task and launch command...")).clicked() {
                let files: Vec<_> = self.external_files.lines().map(str::trim).filter(|s| !s.is_empty()).collect();
                let args = json!({"agent_id":self.external_agent_id,"task":self.external_task,"context":self.external_context,"files":files});
                self.manual_tool("consult_agent", args);
            }
        });
    }
}
