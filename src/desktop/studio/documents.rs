//! StudioUi documents responsibilities.
use super::*;

impl StudioUi {
    pub(super) fn document_form(
        &mut self,
        ui: &mut egui::Ui,
        settings: &Settings,
        workspace: Option<&Path>,
    ) {
        let Some(workspace) = workspace else {
            ui.label("Choose a workspace in Files before starting a document activity.");
            return;
        };
        ui.label(
            RichText::new(workspace.display().to_string())
                .monospace()
                .small(),
        );
        ui.horizontal_wrapped(|ui| {
            ui.label("Activity");
            egui::ComboBox::from_id_salt("document_action")
                .selected_text(self.document.action.label())
                .show_ui(ui, |ui| {
                    for action in [
                        DocumentAction::Summarize,
                        DocumentAction::Compare,
                        DocumentAction::StudyGuide,
                        DocumentAction::PodcastScript,
                        DocumentAction::Rewrite,
                        DocumentAction::Translate,
                    ] {
                        ui.selectable_value(&mut self.document.action, action, action.label());
                    }
                });
        });
        ui.label("Workspace-relative document paths, one per line");
        ui.add(
            egui::TextEdit::multiline(&mut self.document_paths)
                .code_editor()
                .desired_rows(5)
                .desired_width(f32::INFINITY)
                .hint_text("notes.md\ndocuments/manual.pdf"),
        );
        if ui.button("Choose documents…").clicked() {
            self.pick_documents(workspace);
        }
        ui.label("Instructions / target language / audience");
        ui.add(
            egui::TextEdit::multiline(&mut self.document.instruction)
                .desired_rows(3)
                .desired_width(f32::INFINITY),
        );
        ui.horizontal(|ui| {
            ui.label("Model override");
            ui.add(
                egui::TextEdit::singleline(&mut self.document.model)
                    .hint_text(format!("Use selected model: {}", settings.model))
                    .desired_width((ui.available_width() - 10.0).max(120.0)),
            );
        });
        egui::CollapsingHeader::new("Coverage and context limits").show(ui,|ui|{
            ui.horizontal_wrapped(|ui|{ui.label("Characters / chunk");ui.add(egui::DragValue::new(&mut self.document.chunk_chars).range(512..=32000).speed(256));ui.label("Maximum chunks");ui.add(egui::DragValue::new(&mut self.document.max_chunks).range(1..=512));ui.label("Output tokens");ui.add(egui::DragValue::new(&mut self.document.max_output_tokens).range(128..=8192).speed(128));});
            ui.label("Every admitted extracted-text chunk is processed. Oversized jobs fail before inference instead of silently reviewing only a subset. Extraction limits and changed sources are recorded. Large jobs can make many model requests.");
        });
        if ui
            .button(RichText::new("Review document activity…").strong())
            .clicked()
        {
            let mut request = self.document.clone();
            request.files = self
                .document_paths
                .lines()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
            let mut config = settings.clone();
            config.automatic_memory = false;
            config.agents_enabled = false;
            config.model_routing = false;
            self.prepare(StudioJob::Documents(Box::new((
                config,
                request,
                workspace.into(),
            ))));
        }
        ui.label(RichText::new("Reports include Markdown, plain text, a Word document, source snapshots and a coverage manifest. Rewrite/translation create new outputs; originals stay intact.").small());
    }
}
