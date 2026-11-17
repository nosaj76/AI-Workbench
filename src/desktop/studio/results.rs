//! StudioUi results responsibilities.
use super::*;

impl StudioUi {
    pub(super) fn results_panel(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        settings: &Settings,
    ) {
        let Some(output) = self.output.clone() else {
            return;
        };
        ui.separator();
        ui.heading("Latest completed activity");
        ui.label(
            RichText::new(output.directory().display().to_string())
                .monospace()
                .small(),
        );
        ui.horizontal_wrapped(|ui| {
            if ui.button("Open output folder").clicked() {
                open_local(output.directory().into());
            }
            if ui.button("Copy report").clicked() {
                ctx.copy_text(output.text());
            }
            if ui.button("Attach report to chat").clicked() {
                self.attachment = Some(("Studio activity report".into(), output.text()));
            }
        });
        for warning in output.warnings() {
            ui.label(
                RichText::new(warning)
                    .small()
                    .color(ui.visuals().warn_fg_color),
            );
        }
        if !self.textures.is_empty() {
            ui.horizontal_wrapped(|ui| {
                for (path, texture) in &self.textures {
                    ui.vertical(|ui| {
                        ui.add(egui::Image::new(texture).fit_to_exact_size(
                            texture.size_vec2() * (230.0 / texture.size_vec2().max_elem()).min(1.0),
                        ));
                        ui.label(
                            RichText::new(path.file_name().unwrap_or_default().to_string_lossy())
                                .small(),
                        );
                        if ui.small_button("Open image").clicked() {
                            open_local(path.clone());
                        }
                    });
                }
            });
        }
        let image_files = output
            .files()
            .into_iter()
            .filter(|path| {
                path.extension().and_then(|s| s.to_str()).is_some_and(|s| {
                    matches!(
                        s.to_ascii_lowercase().as_str(),
                        "png" | "jpg" | "jpeg" | "webp" | "gif"
                    )
                })
            })
            .collect::<Vec<_>>();
        if !image_files.is_empty() {
            egui::CollapsingHeader::new("Review output images / video frames with Ollama").show(ui,|ui|{
                ui.label(format!("{} image(s). Selected model: {}. A vision-capable model is required. The first 32 selected output images will be reviewed if more are present.",image_files.len(),settings.model));
                ui.add(egui::TextEdit::multiline(&mut self.vision_question).desired_rows(2).desired_width(f32::INFINITY));
                if ui.add_enabled(!self.busy,egui::Button::new("Review vision activity…")).clicked(){let mut config=settings.clone();config.automatic_memory=false;config.agents_enabled=false;config.model_routing=false;self.prepare(StudioJob::ImageReview(Box::new((config,image_files.into_iter().take(32).collect(),self.vision_question.clone()))));}
            });
        }
        if let StudioOutput::Activity(activity) = output.as_ref() {
            if activity.metadata["action"].as_str() == Some("podcast_script")
                && ui
                    .add_enabled(
                        !self.busy,
                        egui::Button::new("Use script to produce podcast audio"),
                    )
                    .clicked()
            {
                self.media_text = activity.text.clone();
                self.media_kind = 5;
                self.tab = Tab::Media;
                self.prepared = None;
            }
            let extracted = activity
                .files
                .iter()
                .find(|path| {
                    matches!(
                        path.file_name().and_then(|s| s.to_str()),
                        Some("transcript.txt" | "ocr.txt" | "ocr.md")
                    )
                })
                .cloned();
            if let Some(extracted) = extracted {
                if ui
                    .add_enabled(!self.busy, egui::Button::new("Summarize extracted text…"))
                    .clicked()
                {
                    let mut config = settings.clone();
                    config.automatic_memory = false;
                    config.agents_enabled = false;
                    config.model_routing = false;
                    let request=DocumentRequest{files:vec![extracted.file_name().unwrap_or_default().to_string_lossy().into_owned()],instruction:"Summarize the extracted text thoroughly, distinguish recorded/source claims from verified facts, and identify any uncertainty in recognition or transcription.".into(),..Default::default()};
                    self.prepare(StudioJob::Documents(Box::new((
                        config,
                        request,
                        activity.directory.clone(),
                    ))));
                }
            }
            egui::CollapsingHeader::new("Written result")
                .default_open(true)
                .show(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("studio_report")
                        .max_height(550.0)
                        .show(ui, |ui| {
                            CommonMarkViewer::new().show(ui, &mut self.markdown, &activity.text);
                        });
                });
        }
        egui::CollapsingHeader::new("Saved files and run details").show(ui, |ui| {
            for path in output.files() {
                ui.horizontal_wrapped(|ui| {
                    if ui.small_button("Open").clicked() {
                        open_local(path.clone());
                    }
                    ui.label(
                        RichText::new(path.display().to_string())
                            .monospace()
                            .small(),
                    );
                });
            }
            if let StudioOutput::Activity(activity) = output.as_ref() {
                ui.code(serde_json::to_string_pretty(&activity.metadata).unwrap_or_default());
            }
        });
    }
}
