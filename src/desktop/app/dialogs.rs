//! Workbench dialogs responsibilities.
use super::*;

impl Workbench {
    pub(super) fn export_dialog(&mut self) {
        let session = self.session.clone();
        self.work("Choose export location", move |tx, _| {
            if let Some(path) = rfd::FileDialog::new()
                .set_file_name("conversation.json")
                .add_filter("Conversation JSON", &["json"])
                .add_filter("Markdown", &["md"])
                .save_file()
            {
                if path.extension().is_some_and(|e| e == "md") {
                    storage::export_markdown(&session, &path)?;
                } else {
                    storage::export_json(&session, &path)?;
                }
                tx.send(WorkEvent::Status(format!("Exported {}", path.display())))?;
            }
            Ok(())
        });
    }

    pub(super) fn import_dialog(&mut self) {
        self.work("Choose a conversation export", |tx, _| {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Conversation JSON", &["json"])
                .pick_file()
            {
                tx.send(WorkEvent::Import(storage::import_json(&path)?))?;
            }
            Ok(())
        });
    }

    pub(super) fn speak(&mut self, text: String) {
        if self.session.project.is_none() {
            self.error = Some("Choose a workspace before running a speech executable".into());
            return;
        }
        match serde_json::from_str::<Vec<String>>(&self.settings.tts_args){Ok(args)=>self.manual_tool("run_command",json!({"program":self.settings.tts_program,"args":args,"stdin_text":text,"timeout_secs":120})),Err(e)=>self.error=Some(format!("Speech arguments are invalid: {e}"))}
    }

    pub(super) fn transcribe(&mut self) {
        if self.session.project.is_none() {
            self.error = Some(
                "Choose a workspace and place the audio file inside it before transcription".into(),
            );
            return;
        }
        // File selection is returned to the UI, where the concrete command is prepared.
        let root = self.session.project.clone().unwrap();
        let program = self.settings.stt_program.clone();
        let args = self.settings.stt_args.clone();
        self.work("Choose local audio", move |tx, _| {
            if let Some(path) = rfd::FileDialog::new()
                .set_directory(&root)
                .add_filter("Audio", &["wav", "mp3", "flac", "ogg", "m4a"])
                .pick_file()
            {
                let ws = Workspace::new(PathBuf::from(&root))?;
                let absolute = path.canonicalize()?;
                let relative = absolute
                    .strip_prefix(ws.root())
                    .context("Audio must be inside the selected workspace")?;
                let args: Vec<String> = serde_json::from_str(&args)?;
                let args = args
                    .into_iter()
                    .map(|a| a.replace("{input}", &absolute.to_string_lossy()))
                    .collect::<Vec<_>>();
                let request = json!({"program":program,"args":args,"timeout_secs":300});
                let _ = relative;
                tx.send(WorkEvent::PrepareTool {
                    name: "run_command".into(),
                    args: request,
                })?;
            }
            Ok(())
        });
        self.page = Page::Tools;
    }

    pub(super) fn dialogs(&mut self, ctx: &egui::Context) {
        // Do not allow an old export preview to be used after a chat switch or edit.
        let valid_export = self.code_export_preview.as_ref().is_none_or(|selection| {
            selection.session_id == self.session.id
                && self
                    .session
                    .entries
                    .iter()
                    .any(|entry| entry.id == selection.entry_id)
        });
        if !valid_export {
            self.code_export_preview = None;
        }
        let mut export_file = None;
        let mut export_archive = None;
        let mut close_export = false;
        let can_export = !self.work_busy && !self.busy && self.approval.is_none();
        if let Some(selection) = self.code_export_preview.as_mut() {
            egui::Window::new("Export answer code")
                .id(egui::Id::new("code_export_dialog"))
                .default_width(460.0).resizable(true).collapsible(false)
                .show(ctx, |ui| {
                    ui.label("Save completed code blocks from this assistant answer. Existing files are never overwritten.");
                    ui.separator();
                    egui::ScrollArea::vertical().max_height(210.0).show(ui, |ui| {
                        for (index, block) in selection.blocks.iter().enumerate() {
                            let title = format!("{}  ·  {}  ·  {} bytes", block.name, block.language, block.text.len());
                            ui.selectable_value(&mut selection.selected, index, title);
                        }
                    });
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.add_enabled(can_export, egui::Button::new("Save selected file...")).clicked() {
                            export_file = selection.blocks.get(selection.selected).cloned();
                        }
                        if ui.add_enabled(can_export, egui::Button::new("Save all as TAR...")).clicked() {
                            export_archive = Some(selection.blocks.clone());
                        }
                        if ui.button("Close").clicked() { close_export = true; }
                    });
                });
        }
        if close_export {
            self.code_export_preview = None;
        }
        if let Some(block) = export_file {
            self.code_export_preview = None;
            self.work("Save selected code file", move |tx, _| {
                if let Some(path) = rfd::FileDialog::new()
                    .set_file_name(&block.name)
                    .save_file()
                {
                    code_export::save_single(&path, &block)?;
                    tx.send(WorkEvent::Status(format!("Saved {}", path.display())))?;
                }
                Ok(())
            });
        }
        if let Some(blocks) = export_archive {
            self.code_export_preview = None;
            self.work("Save code archive", move |tx, _| {
                if let Some(path) = rfd::FileDialog::new()
                    .set_file_name("answer-code.tar")
                    .add_filter("TAR archive", &["tar"])
                    .save_file()
                {
                    code_export::save_archive(&path, &blocks)?;
                    tx.send(WorkEvent::Status(format!("Saved {}", path.display())))?;
                }
                Ok(())
            });
        }
        let mut skill_decision = None;
        if let Some(skill) = &self.skill_preview {
            egui::Window::new("Review local skill")
                .collapsible(false)
                .default_width(700.0)
                .show(ctx, |ui| {
                    ui.heading(&skill.name);
                    ui.label(&skill.description);
                    ui.label(
                        RichText::new(skill.path.display().to_string())
                            .small()
                            .monospace(),
                    );
                    egui::ScrollArea::vertical()
                        .max_height(430.0)
                        .show(ui, |ui| {
                            ui.add(egui::Label::new(&skill.instructions).selectable(true));
                        });
                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() {
                            skill_decision = Some(false);
                        }
                        if ui.button("Use these instructions").clicked() {
                            skill_decision = Some(true);
                        }
                    });
                });
        }
        if let Some(activate) = skill_decision {
            if let Some(skill) = self.skill_preview.take() {
                if activate {
                    self.settings.skill_name = skill.name.clone();
                    self.settings.skill_prompt = skill.prompt_context();
                    if let Some(draft) = &mut self.settings_draft {
                        draft.skill_name = self.settings.skill_name.clone();
                        draft.skill_prompt = self.settings.skill_prompt.clone();
                    }
                    if let Some(original) = &mut self.settings_draft_original {
                        original.skill_name = self.settings.skill_name.clone();
                        original.skill_prompt = self.settings.skill_prompt.clone();
                    }
                    self.persist();
                }
            }
        }
        if let Some(error) = self.error.clone() {
            egui::Window::new("Attention")
                .collapsible(false)
                .resizable(true)
                .default_width(560.0)
                .show(ctx, |ui| {
                    ui.label(RichText::new(error).color(ui.visuals().warn_fg_color));
                    if ui.button("Dismiss").clicked() {
                        self.error = None;
                    }
                });
        }
        let mut decision = None;
        if let Some(approval) = &self.approval {
            let shown = egui::Window::new("Review action")
                .collapsible(false)
                .resizable(true)
                .default_width(720.0)
                .show(ctx, |ui| {
                    ui.heading(&approval.action.title);
                    ui.label(&approval.action.description);
                    egui::ScrollArea::both().max_height(440.0).show(ui, |ui| {
                        ui.add(
                            egui::Label::new(
                                RichText::new(
                                    serde_json::to_string_pretty(&approval.action.details)
                                        .unwrap_or_default(),
                                )
                                .monospace()
                                .size(13.0),
                            )
                            .selectable(true),
                        );
                    });
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Deny").clicked() {
                            decision = Some(false);
                        }
                        if ui
                            .button(
                                RichText::new("Approve this action")
                                    .strong()
                                    .color(ui.visuals().hyperlink_color),
                            )
                            .clicked()
                        {
                            decision = Some(true);
                        }
                    });
                });
            // Windows stack by focus, not draw order. Keep the decision above the
            // tool activity console even if the user clicks the console.
            if let Some(shown) = shown {
                ctx.move_to_top(shown.response.layer_id);
            }
        }
        if let Some(allowed) = decision {
            self.approve(allowed);
        }
        if let Some(name) = self.model_delete.clone() {
            egui::Window::new("Delete installed model?").collapsible(false).show(ctx,|ui|{ui.label(format!("Remove {name} from the connected Ollama server? It will need to be downloaded again to use it."));ui.horizontal(|ui|{if ui.button("Cancel").clicked(){self.model_delete=None;}
if ui.button("Delete model").clicked(){self.model_delete=None;self.model_action("delete",&name);}});});
        }
        if let Some(id) = self.pending_trash_id.clone() {
            let mut response = None;
            egui::Window::new("Move conversation to Trash?")
                .id(egui::Id::new("move_conversation_to_trash"))
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("The conversation will be hidden, not deleted.");
                    ui.label("Open Trash in the sidebar to restore it later.");
                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() {
                            response = Some(false);
                        }
                        if ui.button("Move to Trash").clicked() {
                            response = Some(true);
                        }
                    });
                });
            if let Some(confirm) = response {
                self.pending_trash_id = None;
                if confirm {
                    self.confirm_move_to_trash(&id);
                }
            }
        }
        if let Some(id) = self.pending_permanent_delete_id.clone() {
            let mut response = None;
            egui::Window::new("Permanently delete conversation?")
                .id(egui::Id::new("delete_trashed_conversation"))
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.colored_label(
                        ui.visuals().error_fg_color,
                        "This deletes the conversation permanently. It cannot be restored.",
                    );
                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() {
                            response = Some(false);
                        }
                        if ui.button("Delete forever").clicked() {
                            response = Some(true);
                        }
                    });
                });
            if let Some(confirm) = response {
                self.pending_permanent_delete_id = None;
                if confirm {
                    self.confirm_delete_forever(&id);
                }
            }
        }
    }
}
