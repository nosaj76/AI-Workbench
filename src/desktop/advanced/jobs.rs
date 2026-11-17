//! AdvancedUi jobs responsibilities.
use super::*;

impl AdvancedUi {
    pub(super) fn start<F>(&mut self, label: &str, job: F)
    where
        F: FnOnce(Sender<AgentEvent>, Arc<AtomicBool>) -> Result<(String, Vec<PathBuf>)>
            + Send
            + 'static,
    {
        if self.busy {
            return;
        }
        if self.preview_mode {
            self.error = Some("Restart without --demo to run model or background jobs".into());
            return;
        }
        self.busy = true;
        self.cancel = Arc::new(AtomicBool::new(false));
        self.error = None;
        self.tasks.clear();
        self.report.clear();
        self.files.clear();
        self.status = label.into();
        let cancel = self.cancel.clone();
        let events = self.agent_sender.clone();
        let finish = self.sender.clone();
        std::thread::spawn(move || {
            let result = crate::jobs::guarded(|| job(events, cancel)).map_err(|e| format!("{e:#}"));
            let _ = finish.send(UiEvent::Finished(result));
        });
    }

    pub(super) fn job_results(&mut self, ui: &mut egui::Ui) {
        for (id, task) in &self.tasks {
            egui::CollapsingHeader::new(format!("{} · {}", task.title, task.state))
                .id_salt(id)
                .show(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt(("task", id))
                        .max_height(180.0)
                        .show(ui, |ui| {
                            ui.add(egui::Label::new(&task.text).selectable(true));
                        });
                });
        }
        if !self.report.is_empty() {
            ui.separator();
            ui.horizontal(|ui| {
                ui.heading("Result");
                if ui.button("Attach report to chat").clicked() {
                    self.attachment = Some(("Workbench report".into(), self.report.clone()));
                }
                if ui.button("Copy").clicked() {
                    ui.ctx().copy_text(self.report.clone());
                }
            });
            for path in &self.files {
                ui.monospace(path.display().to_string());
            }
            let preview = self.report.chars().take(100_000).collect::<String>();
            if preview.len() < self.report.len() {
                ui.label("Showing the first 100,000 characters. The saved report contains the complete result.");
            }
            egui::ScrollArea::vertical()
                .id_salt("advanced_result")
                .max_height(ui.available_height().clamp(150.0, 500.0))
                .show(ui, |ui| {
                    CommonMarkViewer::new().show(ui, &mut self.markdown, &preview);
                });
        }
    }
}
