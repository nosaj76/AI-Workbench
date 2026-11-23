//! Workbench models responsibilities.
use super::*;

impl Workbench {
    pub(super) fn models_page(&mut self, ui: &mut egui::Ui) {
        ui.heading("Models");
        if self.demo {
            ui.label("Model management is available when you start Workbench without --demo.");
            ui.disable();
        }
        ui.label(
            "Use your installed Ollama models, manage their lifecycle, and inspect capabilities.",
        );
        ui.horizontal(|ui| {
            ui.label(RichText::new(&self.settings.base_url).monospace());
            if ui
                .add_enabled(!self.work_busy, egui::Button::new("Refresh"))
                .clicked()
            {
                self.refresh_models();
            }
            if !self.version.is_empty() {
                ui.label(format!("Ollama {}", self.version));
            }
        });
        ui.separator();
        egui::ScrollArea::vertical().id_salt("models_page").show(ui,|ui|{
            let mut action=None;
            let mut model_changed = false;
            egui::Grid::new("models_table").num_columns(5).striped(true).spacing([16.0,12.0]).show(ui,|ui|{
                ui.strong("Model");ui.strong("Size");ui.strong("Format / family");ui.strong("Use");ui.strong("Manage");ui.end_row();
                for model in &self.models{
                    ui.label(RichText::new(&model.name).monospace());ui.label(format!("{:.2} GB",model.size as f64/1e9));let family=model.details.get("family").and_then(Value::as_str).unwrap_or("");let quant=model.details.get("quantization_level").and_then(Value::as_str).unwrap_or("");ui.label(format!("{family} {quant}"));
                    if ui.button(if self.settings.model==model.name{"Selected"}else{"Use"}).clicked(){self.settings.model=model.name.clone();model_changed = true;}
                    ui.horizontal(|ui|{if ui.add_enabled(!self.work_busy,egui::Button::new("Info").small()).clicked(){action=Some(("show".to_string(),model.name.clone()));}
if ui.add_enabled(!self.work_busy,egui::Button::new("Unload").small()).clicked(){action=Some(("unload".to_string(),model.name.clone()));}
if ui.add_enabled(!self.work_busy,egui::Button::new("Delete").small()).clicked(){self.model_delete=Some(model.name.clone());}});ui.end_row();
                }
            });
            if model_changed { self.save_preferences(); }
            if let Some((kind,name))=action{self.model_action(&kind,&name);}
            if self.models.is_empty(){ui.label("No installed models were returned. Check that Ollama is running, then refresh.");}
            ui.add_space(16.0);ui.separator();
            ui.heading("Download a model");ui.label("This requests a download through your Ollama server. No model is downloaded automatically.");
            ui.horizontal(|ui|{ui.add(egui::TextEdit::singleline(&mut self.model_pull).hint_text("Model name, for example qwen2.5-coder:7b").desired_width(380.0));if ui.add_enabled(!self.work_busy&&!self.model_pull.trim().is_empty(),egui::Button::new("Pull model")).clicked(){self.model_action("pull",&self.model_pull.clone());}
if self.work_busy&&ui.button("Cancel operation").clicked(){self.work_cancel.store(true,Ordering::Relaxed);}});
            ui.add_space(12.0);
            egui::CollapsingHeader::new("Create a model from an installed model").show(ui,|ui|{
                ui.horizontal(|ui|{ui.label("New name");ui.text_edit_singleline(&mut self.model_new);ui.label("Base model");ui.text_edit_singleline(&mut self.model_from);});
                ui.label("System instruction");ui.add(egui::TextEdit::multiline(&mut self.model_system).desired_rows(4).desired_width(f32::INFINITY));
                if ui.add_enabled(!self.work_busy&&!self.model_new.trim().is_empty()&&!self.model_from.trim().is_empty(),egui::Button::new("Create model")).clicked(){self.model_action("create",&self.model_new.clone());}
                if ui.add_enabled(!self.work_busy&&!self.model_new.trim().is_empty()&&!self.model_from.trim().is_empty(),egui::Button::new("Copy base model to new name")).clicked(){self.model_action("copy",&self.model_new.clone());}
                if ui.add_enabled(!self.work_busy&&!self.model_new.trim().is_empty(),egui::Button::new("Import a GGUF file as new model…")).clicked(){self.model_action("import",&self.model_new.clone());}
            });
            egui::CollapsingHeader::new("Compare models on the current conversation").show(ui,|ui|{
                ui.label("Enter installed model names separated by commas. Candidates run sequentially to limit GPU memory use. Each response retains its model name.");ui.text_edit_singleline(&mut self.compare_models);
                if ui.add_enabled(!self.busy&&!self.session.entries.is_empty(),egui::Button::new("Run comparison")).clicked(){self.start_comparison();}
                if ui.add_enabled(!self.busy&&self.session.entries.len()>2,egui::Button::new("Synthesize candidate answers")).clicked(){self.input="Compare the candidate answers above. Resolve contradictions using the source evidence and produce a single improved answer. Identify remaining uncertainty.".into();self.page=Page::Chat;self.start_chat();}
            });
            if !self.model_details.is_empty(){egui::CollapsingHeader::new("Selected model details").default_open(true).show(ui,|ui|{ui.add(egui::Label::new(RichText::new(&self.model_details).monospace().size(12.0)).selectable(true));});}
            egui::CollapsingHeader::new("Currently loaded models").show(ui,|ui|{ui.add(egui::Label::new(RichText::new(serde_json::to_string_pretty(&self.running_models).unwrap_or_default()).monospace().size(12.0)).selectable(true));});
        });
    }

    pub(super) fn model_action(&mut self, action: &str, name: &str) {
        if self.demo {
            return;
        }
        if self.setup.is_busy() && action != "show" {
            self.error = Some("Finish or stop specialist setup before changing models.".into());
            return;
        }
        let action = action.to_string();
        let name = name.to_string();
        let endpoint = self.settings.base_url.clone();
        let from = self.model_from.clone();
        let system = self.model_system.clone();
        if let Err(e) = self.settings.validate_endpoint() {
            self.error = Some(e.to_string());
            return;
        }
        // A completed model mutation must belong to this endpoint and request.
        // Invalidate any older refresh before launching work, not when it returns.
        if action != "show" {
            self.connection_request_id = self.connection_request_id.wrapping_add(1);
        }
        let request_id = self.connection_request_id;
        self.work("Ollama model operation…", move |tx, cancel| {
            let import_path = if action == "import" {
                match rfd::FileDialog::new()
                    .add_filter("GGUF model", &["gguf"])
                    .pick_file()
                {
                    Some(p) => Some(p),
                    None => return Ok(()),
                }
            } else {
                None
            };
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            rt.block_on(async {
                let client = OllamaClient::new(&endpoint)?;
                let progress_tx = tx.clone();
                let progress = move |v: Value| {
                    let status = v.get("status").and_then(Value::as_str).unwrap_or("Working");
                    let suffix = match (
                        v.get("completed").and_then(Value::as_f64),
                        v.get("total").and_then(Value::as_f64),
                    ) {
                        (Some(done), Some(total)) if total > 0.0 => {
                            format!(" · {:.1}%", done / total * 100.0)
                        }
                        _ => String::new(),
                    };
                    let _ = progress_tx.send(WorkEvent::Status(format!("{status}{suffix}")));
                };
                match action.as_str() {
                    "show" => {
                        tx.send(WorkEvent::ModelInfo(client.show(&name).await?))?;
                    }
                    "unload" => client.unload(&name).await?,
                    "delete" => client.delete(&name).await?,
                    "copy" => client.copy(&from, &name).await?,
                    "pull" => client.pull(&name, cancel.clone(), progress).await?,
                    "create" => {
                        client
                            .create(
                                json!({"model":name,"from":from,"system":system}),
                                cancel.clone(),
                                progress,
                            )
                            .await?
                    }
                    "import" => {
                        let path = import_path.context("Choose a GGUF file")?;
                        let upload_tx = tx.clone();
                        let digest = client
                            .upload_blob(&path, cancel.clone(), move |done, total| {
                                let _ = upload_tx.send(WorkEvent::Status(format!(
                                    "Uploading model · {:.1}%",
                                    if total > 0 {
                                        done as f64 / total as f64 * 100.0
                                    } else {
                                        0.0
                                    }
                                )));
                            })
                            .await?;
                        let filename = path
                            .file_name()
                            .context("Invalid GGUF filename")?
                            .to_string_lossy()
                            .to_string();
                        let mut files = serde_json::Map::new();
                        files.insert(filename, Value::String(digest));
                        client
                            .create(
                                json!({"model":name,"files":files,"system":system}),
                                cancel.clone(),
                                progress,
                            )
                            .await?;
                    }
                    _ => bail!("Unknown model operation"),
                }
                if action != "show" {
                    let models = client.list_models().await?;
                    let version = client.version().await.unwrap_or_else(|_| "unknown".into());
                    let running = client.running_models().await.unwrap_or(Value::Null);
                    tx.send(WorkEvent::Models {
                        request_id,
                        endpoint: endpoint.clone(),
                        models,
                        version,
                        running,
                    })?;
                }
                tx.send(WorkEvent::Status(format!("Model {action} completed")))?;
                Ok(())
            })
        });
    }

    pub(super) fn start_comparison(&mut self) {
        if self.demo {
            return;
        }
        if self.busy || self.work_busy || self.approval.is_some() {
            return;
        }
        let models = self
            .compare_models
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>();
        if models.len() < 2 || models.len() > 6 {
            self.error = Some("Enter between two and six model names".into());
            return;
        }
        let Some(last_user) = self
            .session
            .entries
            .iter()
            .rposition(|e| e.message.role == "user")
        else {
            return;
        };
        let mut messages = vec![ChatMessage {
            role: "system".into(),
            content: self.settings.system(),
            ..Default::default()
        }];
        messages.extend(crate::storage::request_messages(
            &self.session.entries[..=last_user],
        ));
        self.compare_messages = messages.clone();
        self.compare_active = true;
        self.compare_queue = models;
        let model = self.compare_queue.remove(0);
        self.page = Page::Chat;
        self.launch_chat(messages, model);
    }
}
