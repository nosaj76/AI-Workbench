//! Workbench diagnostics responsibilities.
use super::*;

impl Workbench {
    pub(super) fn diagnostics_page(&mut self, ui: &mut egui::Ui) {
        ui.heading("Diagnostics");
        ui.label("Readiness checks do not download weights, launch optional programs, or claim that a model or engine works.");
        ui.group(|ui| {
            ui.strong("Conversation failure log");
            ui.label("Automatic local detection runs when chats are saved. Observed failures and heuristic warnings are labelled separately. Logs stay on this computer; nothing is sent to a coder automatically.");
            ui.label(crate::conversation_diagnostics::log_path(&self.dir).display().to_string());
            if let Some(error) = self.store.diagnostic_error() {
                ui.colored_label(egui::Color32::RED, format!("Diagnostic logging failed; conversations are still saved: {error}"));
            }
            ui.horizontal_wrapped(|ui| {
                if ui.add_enabled(!self.demo, egui::Button::new("Review this conversation")).clicked() {
                    match self.store.diagnose_session(&self.session, false) {
                        Ok(n) => self.status = format!("{n} new or updated diagnostic incidents"),
                        Err(e) => self.error = Some(format!("Conversation diagnostics: {e:#}")),
                    }
                }
                if ui.add_enabled(!self.demo && !self.session.entries.is_empty(), egui::Button::new("Flag for coder review")).clicked() {
                    match self.store.flag_conversation(&self.session) {
                        Ok(_) => self.status = "Conversation flagged in the local diagnostic log".into(),
                        Err(e) => self.error = Some(format!("Conversation diagnostics: {e:#}")),
                    }
                }
                if ui.button("Copy coder report").clicked() {
                    match self.store.conversation_diagnostic_report() {
                        Ok(report) => ui.ctx().copy_text(report),
                        Err(e) => self.error = Some(format!("Conversation diagnostics: {e:#}")),
                    }
                }
            });
            ui.label("Review older chats with: ollama-workbench --diagnose-conversations. Logs contain limited conversation excerpts; review before sharing. Detection cannot establish whether every answer is correct.");
        });
        if self.diagnostics.is_empty() {
            self.diagnostics = crate::diagnostics::inspect(
                &self.settings,
                &self.dir,
                self.session.project.as_deref().map(Path::new),
            );
        }
        ui.horizontal_wrapped(|ui| {
            if ui.button("Refresh checks").clicked() {
                self.diagnostics = crate::diagnostics::inspect(
                    &self.settings,
                    &self.dir,
                    self.session.project.as_deref().map(Path::new),
                );
            }
            if ui
                .add_enabled(
                    !self.work_busy && !self.demo,
                    egui::Button::new("Check Ollama connection"),
                )
                .clicked()
            {
                self.refresh_models();
            }
            if ui.button("Copy report").clicked() {
                ui.ctx()
                    .copy_text(crate::diagnostics::report(&self.diagnostics));
            }
        });
        egui::ScrollArea::vertical()
            .id_salt("diagnostics_checks")
            .show(ui, |ui| {
                for check in &self.diagnostics {
                    ui.group(|ui| {
                        ui.strong(format!("{} - {:?}", check.feature, check.state));
                        ui.label(&check.detail);
                    });
                }
            });
    }
}
