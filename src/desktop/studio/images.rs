//! StudioUi images responsibilities.
use super::*;

impl StudioUi {
    pub(super) fn image_form(&mut self, ui: &mut egui::Ui, workspace: Option<&Path>) {
        let old_backend = self.config.image.backend;
        ui.horizontal_wrapped(|ui| {
            ui.label("Engine");
            egui::ComboBox::from_id_salt("image_engine")
                .selected_text(engine_name(self.config.image.backend))
                .show_ui(ui, |ui| {
                    for backend in [
                        ImageBackend::ComfyUi,
                        ImageBackend::Automatic1111,
                        ImageBackend::StableDiffusionCpp,
                    ] {
                        ui.selectable_value(
                            &mut self.config.image.backend,
                            backend,
                            engine_name(backend),
                        );
                    }
                });
        });
        if old_backend != self.config.image.backend {
            self.models.clear();
            self.prepared = None;
            if self.config.image.endpoint == "http://127.0.0.1:8188"
                || self.config.image.endpoint == "http://127.0.0.1:7860"
            {
                self.config.image.endpoint =
                    if self.config.image.backend == ImageBackend::Automatic1111 {
                        "http://127.0.0.1:7860".into()
                    } else {
                        "http://127.0.0.1:8188".into()
                    };
            }
            self.image.sampler = if self.config.image.backend == ImageBackend::Automatic1111 {
                "Euler".into()
            } else {
                "euler".into()
            };
        }
        if self.config.image.backend == ImageBackend::StableDiffusionCpp {
            text_row(ui, "Executable", &mut self.config.image.sd_program);
            let mut model = self.config.image.sd_model.display().to_string();
            ui.horizontal(|ui| {
                ui.label("Model file");
                ui.add(
                    egui::TextEdit::singleline(&mut model)
                        .desired_width((ui.available_width() - 90.0).max(120.0)),
                );
                if ui.button("Browse…").clicked() {
                    self.pick_model(workspace);
                }
            });
            self.config.image.sd_model = model.into();
            ui.label(RichText::new("Uses the installed sd-cli executable directly. Choose its local diffusion weights; no image server or web interface is needed.").small());
            egui::CollapsingHeader::new("Additional sd-cli arguments").show(ui,|ui|{ui.label("JSON argument array for local VAE/text-encoder/backend options. Output paths remain managed by Workbench.");ui.add(egui::TextEdit::multiline(&mut self.extra_args_json).code_editor().desired_rows(2).desired_width(f32::INFINITY));});
        } else {
            text_row(ui, "Engine URL", &mut self.config.image.endpoint);
            ui.horizontal(|ui| {
                ui.label("Checkpoint");
                ui.add(
                    egui::TextEdit::singleline(&mut self.config.image.checkpoint)
                        .desired_width((ui.available_width() - 160.0).max(120.0)),
                );
                if ui.button("Refresh models").clicked() {
                    let config = self.config.image.clone();
                    self.start_work("Reading image model list", move |tx, cancel| {
                        tx.send(StudioEvent::Models(image_generation::list_models(
                            &config, cancel,
                        )?))?;
                        Ok(())
                    });
                }
            });
            if !self.models.is_empty() {
                egui::ComboBox::from_id_salt("image_checkpoints")
                    .selected_text("Choose an installed checkpoint…")
                    .show_ui(ui, |ui| {
                        for model in &self.models {
                            ui.selectable_value(
                                &mut self.config.image.checkpoint,
                                model.clone(),
                                model,
                            );
                        }
                    });
            }
            ui.checkbox(
                &mut self.config.image.allow_remote,
                "Allow a remote image engine (sends prompts and selected images to that endpoint)",
            );
        }
        ui.label("Prompt");
        ui.add(
            egui::TextEdit::multiline(&mut self.image.prompt)
                .desired_rows(3)
                .desired_width(f32::INFINITY)
                .hint_text("Describe the image to generate…"),
        );
        ui.label("Negative prompt");
        ui.add(
            egui::TextEdit::multiline(&mut self.image.negative_prompt)
                .desired_rows(2)
                .desired_width(f32::INFINITY),
        );
        ui.horizontal_wrapped(|ui| {
            ui.label("Width");
            ui.add(
                egui::DragValue::new(&mut self.image.width)
                    .range(64..=2048)
                    .speed(8),
            );
            ui.label("Height");
            ui.add(
                egui::DragValue::new(&mut self.image.height)
                    .range(64..=2048)
                    .speed(8),
            );
            ui.label("Images");
            ui.add(egui::DragValue::new(&mut self.image.batch_size).range(1..=4));
            ui.label("Steps");
            ui.add(egui::DragValue::new(&mut self.image.steps).range(1..=200));
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("CFG");
            ui.add(
                egui::DragValue::new(&mut self.image.cfg_scale)
                    .range(0.0..=30.0)
                    .speed(0.1),
            );
            ui.label("Seed (-1 = new)");
            ui.add(egui::DragValue::new(&mut self.image.seed).range(-1..=i64::MAX));
            ui.label("Sampler");
            ui.add(egui::TextEdit::singleline(&mut self.image.sampler).desired_width(120.0));
        });
        if self.config.image.backend != ImageBackend::StableDiffusionCpp {
            ui.horizontal(|ui| {
                ui.label("Input image");
                ui.add(
                    egui::TextEdit::singleline(&mut self.image_input)
                        .hint_text("Optional image-to-image input")
                        .desired_width((ui.available_width() - 90.0).max(120.0)),
                );
                if ui.button("Browse…").clicked() {
                    self.pick_input(true, workspace);
                }
            });
            if !self.image_input.trim().is_empty() {
                ui.add(
                    egui::Slider::new(&mut self.image.strength, 0.0..=1.0)
                        .text("Image-to-image strength"),
                );
            }
        }
        if self.config.image.backend == ImageBackend::ComfyUi {
            text_row(ui, "Scheduler", &mut self.image.scheduler);
            egui::CollapsingHeader::new("Custom ComfyUI workflow").show(ui, |ui| {
                ui.label("Optional API-format JSON graph. Exact placeholders include {{prompt}}, {{negative_prompt}}, {{checkpoint}}, {{seed}}, {{width}}, {{height}}, {{steps}}, {{cfg_scale}}, {{batch_size}}, {{input_image}} and {{output_prefix}}.");
                ui.add(egui::TextEdit::multiline(&mut self.workflow_json)
                    .code_editor().desired_rows(5).desired_width(f32::INFINITY));
                ui.checkbox(&mut self.config.image.allow_custom_nodes,
                    "Trust custom workflow node code installed in this engine");
                ui.horizontal(|ui| {
                    if ui.button("Load text-to-image template").clicked() {
                        let mut graph = image_generation::default_workflow(
                            &self.config.image, &self.image, None, "template");
                        for (node, key, value) in [
                            ("checkpoint", "ckpt_name", "{{checkpoint}}"),
                            ("positive", "text", "{{prompt}}"),
                            ("negative", "text", "{{negative_prompt}}"),
                            ("sample", "seed", "{{seed}}"),
                            ("sample", "steps", "{{steps}}"),
                            ("sample", "cfg", "{{cfg_scale}}"),
                            ("latent", "width", "{{width}}"),
                            ("latent", "height", "{{height}}"),
                            ("latent", "batch_size", "{{batch_size}}"),
                            ("save", "filename_prefix", "{{output_prefix}}"),
                        ] {
                            graph[node]["inputs"][key] = json!(value);
                        }
                        self.workflow_json = serde_json::to_string_pretty(&graph).unwrap_or_default();
                    }
                    if ui.button("Use built-in workflow").clicked() {
                        self.workflow_json.clear();
                    }
                });
            });
        }
        ui.horizontal(|ui| {
            ui.label("Image job timeout (seconds)");
            ui.add(egui::DragValue::new(&mut self.config.image.timeout_secs).range(1..=3600));
        });
        ui.horizontal_wrapped(|ui| {
            if ui
                .button(RichText::new("Review image generation…").strong())
                .clicked()
            {
                match self.image_job(workspace) {
                    Ok(job) => self.prepare(job),
                    Err(e) => self.error = Some(format!("{e:#}")),
                }
            }
            if ui.button("Save engine settings").clicked() {
                self.save_settings();
            }
        });
        ui.label(RichText::new("Image engines and weights must already be installed. Current Ollama builds may reject image-generation models; this panel uses the selected diffusion engine's supported interface.").small());
    }

    pub(super) fn image_job(&self, workspace: Option<&Path>) -> Result<StudioJob> {
        let mut config = self.config.image.clone();
        let mut request = self.image.clone();
        request.input_image = None;
        request.workflow = None;
        if config.backend == ImageBackend::StableDiffusionCpp {
            config.sd_extra_args = serde_json::from_str(&self.extra_args_json)
                .context("Additional arguments must be a JSON array of strings")?;
        } else {
            config.sd_extra_args.clear();
            if !self.image_input.trim().is_empty() {
                request.input_image = Some(input_path(&self.image_input, workspace)?);
            }
            if config.backend == ImageBackend::ComfyUi && !self.workflow_json.trim().is_empty() {
                request.workflow = Some(
                    serde_json::from_str(&self.workflow_json)
                        .context("Custom workflow must be valid API JSON")?,
                );
            }
        }
        Ok(StudioJob::Image(Box::new((config, request))))
    }
}
