//! AdvancedUi agents responsibilities.
use super::*;

impl AdvancedUi {
    pub(super) fn agents_page(
        &mut self,
        ui: &mut egui::Ui,
        settings: &mut Settings,
        workspace: Option<&Path>,
    ) {
        ui.heading("Autonomous local subagents");
        ui.label("A planner divides an objective into independent tasks. Workers inspect evidence in their own contexts; a final pass combines their results. Every run is checkpointed.");
        ui.add(egui::TextEdit::multiline(&mut self.objective).hint_text("Investigate this project, compare approaches, and produce a grounded recommendation…").desired_rows(4).desired_width(f32::INFINITY));
        ui.horizontal_wrapped(|ui| {
            ui.label("Maximum tasks");
            ui.add(egui::DragValue::new(&mut settings.agent_max_tasks).range(1..=8));
            ui.label("Concurrent workers");
            ui.add(egui::DragValue::new(&mut settings.agent_max_workers).range(1..=4));
            ui.checkbox(
                &mut self.allow_actions,
                "Allow workers to request action approvals",
            );
        });
        ui.label("Worker models, comma separated; blank routes each task automatically when Auto model is enabled.");
        ui.add(egui::TextEdit::singleline(&mut self.worker_models).desired_width(f32::INFINITY));
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(
                    !self.busy && !self.objective.trim().is_empty(),
                    egui::Button::new("Plan and run team"),
                )
                .clicked()
            {
                let mut request = AgentRequest::new(
                    settings.clone(),
                    workspace.map(Path::to_path_buf),
                    self.dir.clone(),
                    self.objective.clone(),
                    settings.active_model().into(),
                );
                request.worker_models = self
                    .worker_models
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect();
                request.allow_actions = self.allow_actions;
                self.start("Planning independent tasks…", move |tx, cancel| {
                    let run = crate::agents::run(request, tx, cancel)?;
                    Ok((run.report(), vec![]))
                });
            }
            if ui.button("Refresh saved runs").clicked() {
                match crate::agents::list_runs(&self.dir, 50) {
                    Ok(runs) => self.history = runs,
                    Err(e) => self.error = Some(e.to_string()),
                }
            }
            if ui.button("Save agent defaults").clicked() {
                if let Err(e) = settings.save(&self.dir) {
                    self.error = Some(e.to_string());
                }
            }
        });
        egui::CollapsingHeader::new("Saved teams and recovery").show(ui, |ui| {
            let mut load = None;
            for run in &self.history {
                ui.horizontal_wrapped(|ui| {
                    ui.label(format!(
                        "{} · {} · {} tasks",
                        run.created_at,
                        run.state,
                        run.tasks.len()
                    ));
                    if ui.small_button("Open").clicked() {
                        load = Some(run.id.clone());
                    }
                });
            }
            if let Some(id) = load {
                match crate::agents::load_run(&self.dir, &id) {
                    Ok(run) => {
                        self.report = run.report();
                        self.resume_id = id;
                        self.objective = run.prompt;
                    }
                    Err(e) => self.error = Some(e.to_string()),
                }
            }
            ui.horizontal(|ui| {
                ui.label("Resume run ID");
                ui.text_edit_singleline(&mut self.resume_id);
                if ui
                    .add_enabled(
                        !self.busy && !self.resume_id.is_empty(),
                        egui::Button::new("Resume unfinished workers"),
                    )
                    .clicked()
                {
                    let result = crate::agents::load_run(&self.dir, &self.resume_id);
                    match result {
                        Ok(run) => {
                            let mut request = AgentRequest::new(
                                settings.clone(),
                                run.workspace,
                                self.dir.clone(),
                                run.prompt,
                                run.planner_model,
                            );
                            request.resume_id = Some(run.id);
                            request.allow_actions = self.allow_actions;
                            self.start("Resuming saved team…", move |tx, cancel| {
                                let run = crate::agents::run(request, tx, cancel)?;
                                Ok((run.report(), vec![]))
                            });
                        }
                        Err(e) => self.error = Some(e.to_string()),
                    }
                }
            });
        });
        self.job_results(ui);
    }
}
