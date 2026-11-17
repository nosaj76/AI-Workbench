//! Native first-run model setup. Workers retain the endpoint/configuration
//! snapshot that the user reviewed; only a fully verified pack enables routing.
use crate::{
    model_catalog::{self, CatalogEntry},
    model_setup::{self, SetupPlan, SetupProgress, SetupRequest, SetupRun},
    settings::Settings,
};
use eframe::egui::{self, RichText};
use std::{
    collections::BTreeSet,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc,
    },
    time::Duration,
};

enum Event {
    Progress(SetupProgress),
    Plan(Box<SetupPlan>),
    Finished(Result<Outcome, String>),
}
enum Outcome {
    Checked,
    Installed(Box<SetupRun>),
}

pub struct SetupUi {
    dir: PathBuf,
    entries: Vec<CatalogEntry>,
    selected_optional: BTreeSet<String>,
    context_tokens: u32,
    sender: Sender<Event>,
    receiver: Receiver<Event>,
    cancel: Arc<AtomicBool>,
    busy: bool,
    status: String,
    error: Option<String>,
    progress: Option<SetupProgress>,
    plan: Option<Box<SetupPlan>>,
    run: Option<Box<SetupRun>>,
    baseline: Option<Settings>,
    apply_context: u32,
    applied: bool,
    finished: bool,
    runtime_requested: bool,
    settings_requested: bool,
}

impl SetupUi {
    pub fn new(dir: PathBuf) -> Self {
        let (sender, receiver) = mpsc::channel();
        Self {
            dir,
            entries: model_catalog::catalog(),
            selected_optional: BTreeSet::new(),
            context_tokens: 8192,
            sender,
            receiver,
            cancel: Arc::new(AtomicBool::new(false)),
            busy: false,
            status: "Choose your specialist pack, then download and configure it in one queue."
                .into(),
            error: None,
            progress: None,
            plan: None,
            run: None,
            baseline: None,
            apply_context: 8192,
            applied: false,
            finished: false,
            runtime_requested: false,
            settings_requested: false,
        }
    }

    pub fn is_busy(&self) -> bool {
        self.busy
    }
    pub fn cancel_all(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
    pub fn take_applied(&mut self) -> bool {
        std::mem::take(&mut self.applied)
    }
    pub fn take_finished(&mut self) -> bool {
        std::mem::take(&mut self.finished)
    }
    pub fn take_runtime_request(&mut self) -> bool {
        std::mem::take(&mut self.runtime_requested)
    }
    pub fn take_settings_request(&mut self) -> bool {
        std::mem::take(&mut self.settings_requested)
    }

    pub fn poll(&mut self, ctx: &egui::Context, settings: &mut Settings) {
        for _ in 0..512 {
            let Ok(event) = self.receiver.try_recv() else {
                break;
            };
            match event {
                Event::Progress(progress) => {
                    self.status = match &progress.tag {
                        Some(tag) => format!("{tag} · {}", progress.stage),
                        None => progress.stage.clone(),
                    };
                    self.progress = Some(progress);
                }
                Event::Plan(plan) => self.plan = Some(plan),
                Event::Finished(result) => {
                    self.busy = false;
                    self.progress = None;
                    match result {
                        Ok(Outcome::Checked) => {
                            // The fresh plan is the source of current readiness.
                            // A completed historical receipt may refer to weights
                            // that have since been removed or changed.
                            self.run = None;
                            self.status = "Connection and installed models checked. The plan below shows what is already usable.".into();
                        }
                        Ok(Outcome::Installed(run)) => {
                            if run.completed() {
                                self.status = "All selected models are verified.".into();
                                let compatible = self.baseline.as_ref().is_some_and(|old| {
                                    old.base_url == settings.base_url
                                        && old.routing_json == settings.routing_json
                                        && old.model_routing == settings.model_routing
                                        && old.model == settings.model
                                        && old.num_ctx == settings.num_ctx
                                        && old.agents_enabled == settings.agents_enabled
                                        && old.agent_max_workers == settings.agent_max_workers
                                        && old.embedding_model == settings.embedding_model
                                });
                                if compatible {
                                    match model_catalog::configured_settings(settings, &run, self.apply_context)
                                        .and_then(|new| { new.save(&self.dir)?; Ok(new) })
                                    {
                                        Ok(new) => {
                                            *settings = new;
                                            self.applied = true;
                                            self.status = "Ready · automatic routing and specialist orchestration are configured.".into();
                                        }
                                        Err(error) => self.error = Some(format!("Models are installed, but saving the routing settings failed: {error:#}. Run setup again to recheck and apply.")),
                                    }
                                } else {
                                    self.error = Some("Models are installed. Your connection or routing settings changed during setup, so they were preserved. Run setup again to verify and configure the current connection.".into());
                                }
                            } else {
                                self.status = "Setup is unfinished. Completed downloads are retained; use the same setup action to resume.".into();
                                self.error = run.error.clone();
                            }
                            self.run = Some(run);
                        }
                        Err(error) => {
                            self.status = if self.cancel.load(Ordering::Relaxed) {
                                "Setup stopped. Run it again to resume retained downloads."
                            } else {
                                "Setup needs attention. Existing models and completed downloads are retained."
                            }.into();
                            self.error = Some(error);
                        }
                    }
                }
            }
        }
        if self.busy {
            ctx.request_repaint_after(Duration::from_millis(80));
        }
    }

    fn start(&mut self, settings: &Settings, install: bool) {
        if self.busy {
            return;
        }
        let candidates = self
            .entries
            .iter()
            .filter(|e| !e.optional || self.selected_optional.contains(&e.candidate.id))
            .map(|e| e.candidate.clone())
            .collect();
        let request = SetupRequest::new(settings.clone(), self.dir.clone(), candidates);
        self.baseline = Some(settings.clone());
        self.apply_context = self.context_tokens;
        self.busy = true;
        self.error = None;
        self.progress = None;
        self.plan = None;
        self.run = None;
        self.status = "Checking the selected Ollama endpoint and model metadata…".into();
        self.cancel = Arc::new(AtomicBool::new(false));
        let cancel = self.cancel.clone();
        let sender = self.sender.clone();
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || -> anyhow::Result<Outcome> {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()?;
                    runtime.block_on(async {
                        let progress_sender = sender.clone();
                        let plan =
                            model_setup::prepare(&request, cancel.clone(), move |progress| {
                                let _ = progress_sender.send(Event::Progress(progress));
                            })
                            .await?;
                        let _ = sender.send(Event::Plan(Box::new(plan.clone())));
                        if install {
                            let progress_sender = sender.clone();
                            let run = model_setup::run(&request, &plan, cancel, move |progress| {
                                let _ = progress_sender.send(Event::Progress(progress));
                            })
                            .await?;
                            Ok(Outcome::Installed(Box::new(run)))
                        } else {
                            Ok(Outcome::Checked)
                        }
                    })
                },
            ))
            .map_err(|_| {
                "Model setup worker stopped unexpectedly. Saved downloads can be resumed."
                    .to_string()
            })
            .and_then(|result| result.map_err(|e| format!("{e:#}")));
            let _ = sender.send(Event::Finished(result));
        });
    }

    pub fn show(&mut self, ui: &mut egui::Ui, settings: &mut Settings, can_start: bool) {
        ui.heading("Set up local specialists");
        ui.label("Download a small model pack, verify its capabilities, and connect each role to the router and orchestrator.");
        ui.horizontal_wrapped(|ui| {
            ui.strong("10B total parameters or less per model");
            ui.label("· one inference at a time");
        });
        let estimated: u64 = self
            .entries
            .iter()
            .filter(|e| !e.optional || self.selected_optional.contains(&e.candidate.id))
            .map(|e| e.candidate.download_bytes)
            .sum();
        let previous_context = self.context_tokens;
        ui.horizontal_wrapped(|ui| {
            ui.label(format!(
                "Selected weights ≈ {:.2} GB",
                estimated as f64 / 1_000_000_000.0
            ));
            ui.label("Starting context");
            ui.add_enabled_ui(!self.busy, |ui| {
                egui::ComboBox::from_id_salt("starter_context")
                    .selected_text(format!("{} tokens", self.context_tokens))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.context_tokens,
                            4096,
                            "4,096 · lower memory use",
                        );
                        ui.selectable_value(
                            &mut self.context_tokens,
                            8192,
                            "8,192 · more room for local files",
                        );
                    });
            });
        });
        if previous_context != self.context_tokens {
            self.status =
                "Context selection changed. Run setup to verify the pack and apply it.".into();
            self.error = None;
        }
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(
                    !self.busy && can_start,
                    egui::Button::new("Download & configure specialists"),
                )
                .clicked()
            {
                self.start(settings, true);
            }
            if ui
                .add_enabled(
                    !self.busy && can_start,
                    egui::Button::new("Check installed models"),
                )
                .clicked()
            {
                self.start(settings, false);
            }
            if self.busy && ui.button("Stop setup").clicked() {
                self.cancel_all();
                self.status = "Stopping setup…".into();
            }
            if !self.busy
                && ui
                    .button(if settings.model_setup_seen {
                        "Open chat"
                    } else {
                        "Skip setup for now"
                    })
                    .clicked()
            {
                let mut next = settings.clone();
                next.model_setup_seen = true;
                match next.save(&self.dir) {
                    Ok(()) => {
                        *settings = next;
                        self.finished = true;
                    }
                    Err(error) => self.error = Some(format!("Save setup preference: {error:#}")),
                }
            }
        });
        if !can_start && !self.busy {
            ui.small("Finish the active model operation before starting setup.");
        }
        ui.horizontal_wrapped(|ui| {
            if self.busy {
                ui.spinner();
            }
            ui.label(&self.status);
        });
        if let Some(progress) = &self.progress {
            if progress.model_index > 0 {
                ui.small(format!(
                    "Model {} of {}",
                    progress.model_index, progress.model_count
                ));
            }
            if let Some(total) = progress.total_bytes.filter(|n| *n > 0) {
                ui.add(
                    egui::ProgressBar::new(
                        (progress.completed_bytes as f64 / total as f64).clamp(0.0, 1.0) as f32,
                    )
                    .text(format!(
                        "Current layer: {:.1} / {:.1} MiB",
                        progress.completed_bytes as f64 / 1_048_576.0,
                        total as f64 / 1_048_576.0
                    )),
                );
            }
        }
        if let Some(error) = &self.error {
            ui.colored_label(ui.visuals().error_fg_color, error);
        }
        ui.separator();
        egui::ScrollArea::vertical().id_salt("model_setup_page").auto_shrink([false, false]).show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("Setup endpoint");
                let endpoint = self.plan.as_ref().map(|plan| plan.endpoint.as_str())
                    .or_else(|| self.baseline.as_ref().filter(|_| self.busy).map(|s| s.base_url.as_str()))
                    .unwrap_or(&settings.base_url);
                ui.monospace(endpoint);
                if self.baseline.as_ref().is_some_and(|s| s.base_url != settings.base_url)
                    && (self.busy || self.plan.is_some()) {
                    ui.small(format!("Current connection changed to {}", settings.base_url));
                }
                if ui.add_enabled(!self.busy, egui::Button::new("Connection settings")).clicked() {
                    self.settings_requested = true;
                }
                if ui.add_enabled(!self.busy, egui::Button::new("Install / update Ollama")).clicked() {
                    self.runtime_requested = true;
                }
            });
            ui.small("Start an existing Ollama server, or use the managed runtime controls. The Qwen3.5 pack needs Ollama 0.17.5 or newer; model metadata may require a later version.");
            ui.add_space(8.0);
            let mut selection_changed = false;
            for entry in &self.entries {
                let included = !entry.optional || self.selected_optional.contains(&entry.candidate.id);
                ui.group(|ui| {
                    ui.horizontal_wrapped(|ui| {
                        if entry.optional {
                            let mut checked = included;
                            if ui.add_enabled(!self.busy, egui::Checkbox::new(&mut checked, &entry.title)).changed() {
                                if checked { self.selected_optional.insert(entry.candidate.id.clone()); }
                                else { self.selected_optional.remove(&entry.candidate.id); }
                                selection_changed = true;
                            }
                        } else {
                            ui.strong(&entry.title);
                            ui.small("Included");
                        }
                        ui.label(format!("≈ {:.2} GB", entry.candidate.download_bytes as f64 / 1_000_000_000.0));
                    });
                    ui.monospace(&entry.candidate.tag);
                    ui.label(&entry.description);
                    ui.horizontal_wrapped(|ui| {
                        ui.small(format!("License: {}", entry.candidate.license));
                        ui.hyperlink_to("Model details", &entry.candidate.source_url);
                        if entry.candidate.id == "translategemma_4b" {
                            ui.hyperlink_to("Gemma terms", "https://ai.google.dev/gemma/terms");
                        }
                    });
                });
            }
            if selection_changed {
                self.plan = None;
                self.run = None;
                self.error = None;
                self.status = "Model selection changed. Run setup to verify and configure this pack.".into();
            }
            ui.label("Existing compatible models are reused. Actual downloads and storage may differ from the rounded catalog sizes. Model licenses apply separately to the application license.");
            ui.small("This pack is a practical starting point for 16 GB RAM and an 8 GB GPU. One model serves several roles. Larger contexts consume more memory; the app does not claim every model is best for every task.");
            if let Some(plan) = &self.plan {
                egui::CollapsingHeader::new("Download plan and results").default_open(true).show(ui, |ui| {
                    ui.label(format!("Ollama {} · original missing download estimate {:.2} GB", plan.ollama_version, plan.estimated_download_bytes as f64 / 1_000_000_000.0));
                    for item in &plan.models {
                        let result = self.run.as_ref().and_then(|run| run.models.iter().find(|m| m.candidate.id == item.candidate.id));
                        let state = result.map(|m| m.state.to_string()).unwrap_or_else(|| {
                            if item.verification.is_some() { "Installed and compatible" }
                            else if item.issue.is_some() { "Needs attention" } else { "Download required" }.into()
                        });
                        ui.label(format!("{} · {state}", item.candidate.tag));
                        if let Some(issue) = &item.issue { ui.colored_label(ui.visuals().warn_fg_color, issue); }
                        if let Some(error) = result.and_then(|m| m.error.as_deref()) { ui.colored_label(ui.visuals().warn_fg_color, error); }
                        for warning in &item.warnings { ui.small(warning); }
                    }
                    for warning in &plan.warnings { ui.small(warning); }
                });
            }
            if let Some(run) = &self.run {
                if run.completed() {
                    ui.separator();
                    ui.heading("Verified specialist roles");
                    if let Ok(roles) = run.verified_models() {
                        for (role, models) in roles {
                            ui.horizontal_wrapped(|ui| {
                                ui.label(RichText::new(format!("{role}")).strong());
                                ui.monospace(models.join(", "));
                            });
                        }
                    }
                }
            }
        });
    }
}

impl Drop for SetupUi {
    fn drop(&mut self) {
        self.cancel_all();
    }
}
