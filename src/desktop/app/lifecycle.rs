//! Workbench lifecycle responsibilities.
use super::*;

impl Workbench {
    pub(super) fn render(&mut self, ctx: &egui::Context) {
        self.receive_events();
        self.advanced.poll(ctx);
        self.setup.poll(ctx, &mut self.settings);
        if ctx.input(|input| {
            input.modifiers.ctrl && input.modifiers.shift && input.key_pressed(egui::Key::T)
        }) {
            self.set_theme(ctx, !self.settings.dark_mode);
        }
        if ctx.input(|input| input.modifiers.ctrl && input.key_pressed(egui::Key::N)) {
            self.new_conversation();
        }
        self.header(ctx);
        self.sidebar(ctx);
        self.answer_inspector(ctx);
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(8.0);
            match self.page {
                Page::Chat => self.chat_page(ui, ctx),
                Page::Files => self.files_page(ui),
                Page::Models => self.models_page(ui),
                Page::Setup => {
                    let can_start = !self.busy && !self.work_busy && !self.advanced.is_busy();
                    ui.add_enabled_ui(!self.demo && !self.settings_corrupt, |ui| {
                        self.setup.show(ui, &mut self.settings, can_start);
                    });
                }
                Page::Knowledge => self.knowledge_page(ui),
                Page::Tools => self.tools_page(ui),
                Page::Tasks => {
                    if self.setup.is_busy() {
                        ui.label(
                            "Finish or stop specialist setup before changing the Ollama service.",
                        );
                    }
                    ui.add_enabled_ui(!self.setup.is_busy(), |ui| {
                        self.advanced.show(
                            ui,
                            ctx,
                            &mut self.settings,
                            self.session.project.as_deref().map(Path::new),
                        );
                    });
                    if self.advanced.section != crate::advanced_ui::Section::Services {
                        self.page = Page::Advanced;
                    }
                }
                Page::Advanced => {
                    let service_conflict = self.setup.is_busy()
                        && self.advanced.section == crate::advanced_ui::Section::Services;
                    if service_conflict {
                        ui.label(
                            "Finish or stop specialist setup before changing the Ollama service.",
                        );
                    }
                    ui.add_enabled_ui(!service_conflict, |ui| {
                        self.advanced.show(
                            ui,
                            ctx,
                            &mut self.settings,
                            self.session.project.as_deref().map(Path::new),
                        );
                    });
                }
                Page::Settings => self.settings_page(ui, ctx),
                Page::Diagnostics => self.diagnostics_page(ui),
            }
        });
        if self.page != Page::Settings {
            self.settings_draft = None;
            self.settings_draft_original = None;
        }
        // The approval window raises itself above this console (see dialogs.rs).
        self.tool_console.show(ctx, self.approval.is_some());
        self.dialogs(ctx);
        self.advanced.dialogs(ctx);
        if self.setup.take_applied() {
            self.manual_engine = None;
            self.advanced.invalidate_routing();
            self.status = "Specialists ready · automatic routing enabled".into();
            self.refresh_models();
        }
        if self.setup.take_finished() {
            self.page = Page::Chat;
            if !self.demo {
                self.refresh_models();
            }
        }
        if self.setup.take_runtime_request() {
            self.advanced.open_runtime();
            self.page = Page::Advanced;
        }
        if self.setup.take_settings_request() {
            self.page = Page::Settings;
        }
        if let Some((name, text)) = self.advanced.take_attachment() {
            self.attachments.push(Attachment {
                name,
                source: "Explicitly attached Workbench result".into(),
                text,
                image: None,
            });
            self.page = Page::Chat;
        }
        if let Some(endpoint) = self.advanced.take_endpoint_request() {
            let mut candidate = self.settings.clone();
            candidate.base_url = endpoint;
            match candidate
                .validate_endpoint()
                .and_then(|()| candidate.save(&self.dir))
            {
                Ok(()) => {
                    self.settings = candidate;
                    self.manual_engine = None;
                    self.refresh_models();
                }
                Err(e) => self.error = Some(format!("Save managed endpoint: {e}")),
            }
        }
        if let Some(id) = self.advanced.take_session_request() {
            if self.busy || self.work_busy || self.approval.is_some() {
                self.error = Some(
                    "Finish the current chat/action before opening a scheduled conversation."
                        .into(),
                );
            } else {
                self.open_conversation(&id);
            }
        }
        if !self.busy && !self.work_busy {
            let dropped = ctx.input(|i| i.raw.dropped_files.clone());
            let paths = dropped
                .into_iter()
                .filter_map(|f| f.path)
                .collect::<Vec<_>>();
            if !paths.is_empty() {
                self.attach_files(paths);
            }
        }
        if self.dirty && self.last_save.elapsed() > Duration::from_secs(3) {
            self.persist();
        }
        if self.busy || self.work_busy {
            ctx.request_repaint_after(Duration::from_millis(40));
        } else {
            ctx.request_repaint_after(Duration::from_millis(300));
        }
    }
}
