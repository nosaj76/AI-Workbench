//! AdvancedUi routing responsibilities.
use super::*;

impl AdvancedUi {
    pub(super) fn routing_page(&mut self, ui: &mut egui::Ui, settings: &mut Settings) {
        if !self.routing_loaded {
            match serde_json::from_str(&settings.routing_json) {
                Ok(config) => self.routing = config,
                Err(e) => self.error = Some(format!("Routing settings: {e}")),
            }
            self.routing_names = [
                self.routing.orchestrator.join(", "),
                self.routing.general.join(", "),
                self.routing.coding.join(", "),
                self.routing.translation.join(", "),
                self.routing.writing.join(", "),
                self.routing.reasoning.join(", "),
                self.routing.vision.join(", "),
            ];
            self.routing_loaded = true;
        }
        ui.heading("Automatic model routing");
        ui.label("Only installed models with confirmed required capabilities are eligible. Routing uses task keywords, explicit preferences, model-file size, loading state and measured task calibration when available, plus observed generation speed. Explicit model profiles still bypass automatic routing.");
        ui.checkbox(
            &mut settings.model_routing,
            "Choose a model automatically for ordinary chats",
        );
        ui.checkbox(
            &mut self.routing.use_benchmarks,
            "Use fresh, fully passing task benchmarks before heuristic preferences",
        );
        ui.small("Benchmarks use synthetic prompts only. They never execute requested tools. Vision and translation keep configured routing; this first suite measures text extraction, coding diagnosis, reasoning, writing constraints and native tool calls.");
        if ui
            .add_enabled(!self.busy, egui::Button::new("Benchmark selected model"))
            .clicked()
        {
            let settings = settings.clone();
            let models = vec![settings.active_model().to_owned()];
            let dir = self.dir.clone();
            self.start("Benchmarking selected model…", move |events, cancel| {
                let report = crate::benchmarks::run(&settings, &dir, &models, cancel, |s| {
                    let _ = events.send(AgentEvent::Status(s));
                })?;
                Ok((report, vec![dir.join("benchmarks.sqlite3")]))
            });
        }
        if let Ok(report) = crate::benchmarks::report(&self.dir) {
            egui::CollapsingHeader::new("Measured task benchmarks").show(ui, |ui| {
                ui.label(report);
            });
        }
        ui.checkbox(
            &mut self.routing.prefer_running,
            "Prefer an already loaded compatible model",
        );
        ui.checkbox(
            &mut self.routing.favor_speed,
            "Favor smaller models and observed speed",
        );
        ui.checkbox(
            &mut self.routing.single_model_mode,
            "Run one inference at a time and release our model after each request",
        );
        ui.horizontal_wrapped(|ui| {
            ui.label("Maximum total parameters, billions (0 = no cap)");
            ui.add(
                egui::DragValue::new(&mut self.routing.max_parameters_b)
                    .range(0.0..=10.0)
                    .speed(0.5),
            );
        });
        ui.small("The starter pack uses a 10B total-parameter cap, including a separate vision projector. Unknown counts are excluded when the cap is enabled. MoE active-parameter counts do not replace total counts.");
        ui.horizontal(|ui| {
            ui.label("Model-file size budget, GiB (0 = no limit)");
            ui.add(
                egui::DragValue::new(&mut self.routing.max_model_gib)
                    .range(0.0..=128.0)
                    .speed(0.5),
            );
        });
        ui.small("Model-file size is a routing limit, not a guarantee of VRAM usage. Context and runtime buffers also consume memory.");
        for (label, value) in [
            "Orchestrator",
            "General",
            "Coding",
            "Translation",
            "Writing",
            "Reasoning",
            "Vision",
        ]
        .into_iter()
        .zip(&mut self.routing_names)
        {
            ui.horizontal(|ui| {
                ui.label(label);
                ui.add(
                    egui::TextEdit::singleline(value)
                        .hint_text("Preferred installed model names, in order")
                        .desired_width((ui.available_width() - 12.0).max(200.0)),
                );
            });
        }
        if ui.button("Save routing settings").clicked() {
            for (models, text) in [
                &mut self.routing.orchestrator,
                &mut self.routing.general,
                &mut self.routing.coding,
                &mut self.routing.translation,
                &mut self.routing.writing,
                &mut self.routing.reasoning,
                &mut self.routing.vision,
            ]
            .into_iter()
            .zip(&self.routing_names)
            {
                *models = text
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .collect();
            }
            match serde_json::to_string_pretty(&self.routing)
                .map_err(anyhow::Error::from)
                .and_then(|text| {
                    settings.routing_json = text;
                    settings.save(&self.dir)
                }) {
                Ok(()) => self.status = "Routing settings saved".into(),
                Err(e) => self.error = Some(e.to_string()),
            }
        }
        if let Ok(text) = std::fs::read_to_string(self.dir.join("last-route.json")) {
            egui::CollapsingHeader::new("Last routing decision")
                .default_open(true)
                .show(ui, |ui| {
                    ui.add(egui::Label::new(RichText::new(text).monospace()).selectable(true));
                });
        }
        if let Ok(metrics) = crate::routing::observations(&self.dir) {
            ui.separator();
            for (model, tps) in metrics {
                ui.label(format!(
                    "{model}: {tps:.1} observed tokens/second (rolling average)"
                ));
            }
        }
    }
}
