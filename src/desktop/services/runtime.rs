//! ServicesUi runtime responsibilities.
use super::*;

impl ServicesUi {
    pub(super) fn runtime_page(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Managed Ollama runtime");
        ui.label("Installs official Linux binaries in this application's private runtime directory. Your system's Ollama installation and existing services remain separate. Downloads require internet access; inference remains local.");
        if let Some(status) = &self.runtime {
            ui.label(format!(
                "Selected: {} · Previous: {}",
                status.current_version.as_deref().unwrap_or("Not installed"),
                status.previous_version.as_deref().unwrap_or("None")
            ));
            if let Some(process) = &status.process {
                ui.label(format!(
                    "Running {} on 127.0.0.1:{} (PID {})",
                    process.version, process.port, process.pid
                ));
                ui.label(format!("Models: {}", process.models.display()));
                if ui.button("Use this local runtime for chat").clicked() {
                    self.endpoint_request = Some(format!("http://127.0.0.1:{}", process.port));
                }
            } else {
                ui.label("Managed server is stopped");
            }
        }
        ui.horizontal_wrapped(|ui| {
            ui.label("Version tag");
            ui.add(
                egui::TextEdit::singleline(&mut self.version)
                    .hint_text("Blank for latest stable")
                    .desired_width(220.0),
            );
            if ui
                .add_enabled(!self.busy, egui::Button::new("Check official release"))
                .clicked()
            {
                let dir = self.dir.clone();
                let version = self.version.trim().to_string();
                self.work("Checking official Ollama releases", move |tx, cancel| {
                    let manager = OllamaManager::new(&dir)?;
                    let release = if version.is_empty() {
                        manager.latest_release()?
                    } else {
                        manager.release(&version)?
                    };
                    if cancel.load(Ordering::Relaxed) {
                        bail!("Release check cancelled");
                    }
                    tx.send(Event::Release(Box::new(release)))?;
                    Ok(())
                });
            }
        });
        if let Some(release) = self.release.clone() {
            ui.group(|ui|{ui.strong(format!("Official {}",release.version));ui.label(format!("Published {}",release.published_at));ui.hyperlink_to("Release source",&release.url);match release.linux_asset(){Ok(asset)=>{ui.label(format!("{} · {} download",asset.name,size(asset.size)));ui.label(if asset.sha256.is_some(){"SHA-256 digest supplied by official release metadata"}else{"The installer will require the official checksum manifest before activation"});if ui.add_enabled(!self.busy,egui::Button::new("Download and install / update managed runtime")).clicked(){let dir=self.dir.clone();let release=release.clone();self.work("Installing official runtime",move|tx,cancel|{let manager=OllamaManager::new(&dir)?;let result=manager.install_release(&release,cancel,|progress|{let _=tx.send(Event::Progress(progress));})?;tx.send(Event::Runtime(Box::new(result)))?;Ok(())});}},Err(error)=>{ui.label(error.to_string());}}egui::CollapsingHeader::new("Release notes").show(ui,|ui|{ui.add(egui::Label::new(&release.notes).selectable(true));});});
        }
        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Local port");
            ui.add(egui::DragValue::new(&mut self.port).range(1024..=65535));
            ui.label("Use a free port if your existing Ollama already uses 11434.");
        });
        ui.label("Model directory (blank uses the managed runtime's own model store)");
        ui.add(egui::TextEdit::singleline(&mut self.models).desired_width(f32::INFINITY));
        let installed = self
            .runtime
            .as_ref()
            .is_some_and(|s| s.current_version.is_some());
        let running = self.runtime.as_ref().is_some_and(|s| s.running);
        let previous = self
            .runtime
            .as_ref()
            .is_some_and(|s| s.previous_version.is_some());
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(
                    !self.busy && installed && !running,
                    egui::Button::new("Start managed server"),
                )
                .clicked()
            {
                let dir = self.dir.clone();
                let port = self.port;
                let models = self.models.trim().to_string();
                self.work("Starting managed Ollama", move |tx, _| {
                    let manager = OllamaManager::new(&dir)?;
                    let path = if models.is_empty() {
                        None
                    } else {
                        Some(PathBuf::from(models))
                    };
                    tx.send(Event::Runtime(Box::new(
                        manager.start_with_models(port, path.as_deref())?,
                    )))?;
                    tx.send(Event::Text("Managed Ollama is ready".into()))?;
                    Ok(())
                });
            }
            if ui
                .add_enabled(
                    !self.busy && running,
                    egui::Button::new("Stop managed server"),
                )
                .clicked()
            {
                self.runtime_action("stop");
            }
            if ui
                .add_enabled(
                    !self.busy && running,
                    egui::Button::new("Restart with selected version"),
                )
                .clicked()
            {
                self.runtime_action("restart");
            }
            if ui
                .add_enabled(
                    !self.busy && previous && !running,
                    egui::Button::new("Roll back selected runtime"),
                )
                .clicked()
            {
                self.runtime_action("rollback");
            }
            if ui
                .add_enabled(!self.busy, egui::Button::new("Read server log"))
                .clicked()
            {
                self.runtime_action("logs");
            }
        });
        if running {
            ui.label(RichText::new("An update selects the new version for the next start. Restart explicitly when current inference work may be interrupted. Stop the managed server before rollback.").small());
        }
        if !self.logs.is_empty() {
            ui.separator();
            ui.horizontal(|ui| {
                ui.strong("Last service output");
                if ui.small_button("Copy").clicked() {
                    ctx.copy_text(self.logs.clone());
                }
            });
            ui.add(
                egui::Label::new(RichText::new(&self.logs).monospace().size(12.0)).selectable(true),
            );
        }
    }

    pub(super) fn runtime_action(&mut self, action: &'static str) {
        let dir = self.dir.clone();
        self.work(&format!("Managed runtime: {action}"), move |tx, _| {
            let manager = OllamaManager::new(&dir)?;
            if action == "logs" {
                tx.send(Event::Text(manager.logs(256 * 1024)?))?;
            } else {
                let status = match action {
                    "stop" => manager.stop()?,
                    "restart" => manager.restart()?,
                    "rollback" => manager.rollback()?,
                    _ => bail!("Unknown runtime action"),
                };
                tx.send(Event::Runtime(Box::new(status)))?;
                tx.send(Event::Text(format!(
                    "Managed runtime operation completed: {action}"
                )))?;
            }
            Ok(())
        });
    }
}
