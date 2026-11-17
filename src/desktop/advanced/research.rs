//! AdvancedUi research responsibilities.
use super::*;

impl AdvancedUi {
    pub(super) fn research_page(
        &mut self,
        ui: &mut egui::Ui,
        settings: &Settings,
        workspace: Option<&Path>,
    ) {
        ui.heading("Deep research");
        ui.label("Plan questions, collect local and optional web sources, keep source notes, and synthesize a cited report. Web research requires enabled network tools plus SearxNG or explicit source URLs.");
        ui.add(
            egui::TextEdit::multiline(&mut self.research_question)
                .hint_text("What should this investigation establish?")
                .desired_rows(4)
                .desired_width(f32::INFINITY),
        );
        ui.horizontal_wrapped(|ui| {
            ui.label("Research rounds");
            ui.add(egui::DragValue::new(&mut self.research_rounds).range(1..=8));
            ui.label("Source budget");
            ui.add(egui::DragValue::new(&mut self.research_sources).range(1..=40));
            ui.checkbox(
                &mut self.research_web,
                "Include web search and page fetching",
            );
        });
        if self.research_web {
            egui::CollapsingHeader::new("Web sources and domain limits").show(ui, |ui| {
                ui.label("Starting URLs (one per line; optional with SearxNG)");
                ui.add(
                    egui::TextEdit::multiline(&mut self.research_seed_urls)
                        .desired_rows(3)
                        .desired_width(f32::INFINITY),
                );
                ui.label("Allowed domains (comma separated; blank allows any public source)");
                ui.text_edit_singleline(&mut self.research_domains);
                ui.small("Starting a web run authorizes its displayed source scope. Search queries and page requests leave this computer; model inference uses the selected Ollama endpoint.");
            });
        }
        if ui
            .add_enabled(
                !self.busy && !self.research_question.trim().is_empty(),
                egui::Button::new("Start research"),
            )
            .clicked()
        {
            let mut request = crate::research::ResearchRequest::new(
                settings.clone(),
                workspace.map(Path::to_path_buf),
                self.dir.clone(),
                self.research_question.clone(),
                settings.active_model().into(),
            );
            request.max_rounds = self.research_rounds;
            request.max_sources = self.research_sources;
            request.web_enabled = self.research_web;
            self.configure_research_sources(&mut request);
            self.start_research(request);
        }
        egui::CollapsingHeader::new("Saved research and resume").show(ui, |ui| {
            if ui.button("Refresh saved research").clicked() {
                match crate::research::list_runs(&self.dir, 20) {
                    Ok(runs) => self.research_history = runs,
                    Err(error) => self.error = Some(format!("{error:#}")),
                }
            }
            let mut selected = None;
            egui::ScrollArea::vertical()
                .id_salt("research-history")
                .max_height(160.0)
                .show(ui, |ui| {
                    for run in &self.research_history {
                        if ui
                            .selectable_label(
                                self.research_resume_id == run.id,
                                format!("{} · {} · {}", run.updated_at, run.state, run.question),
                            )
                            .clicked()
                        {
                            selected = Some(run.id.clone());
                        }
                    }
                });
            if let Some(id) = selected {
                self.research_resume_id = id;
            }
            ui.horizontal_wrapped(|ui| {
                ui.label("Run ID");
                ui.text_edit_singleline(&mut self.research_resume_id);
                if ui.button("Load saved report").clicked() {
                    match crate::research::load_run(&self.dir, self.research_resume_id.trim()) {
                        Ok(run) => {
                            self.report = run.report();
                            self.files = vec![self.dir.join("research_runs").join(format!("{}.json", run.id))];
                            self.research_question = run.question;
                            self.status = format!("Loaded research · {}", run.state);
                        }
                        Err(error) => self.error = Some(format!("{error:#}")),
                    }
                }
                if ui
                    .add_enabled(!self.busy && !self.research_resume_id.trim().is_empty(), egui::Button::new("Resume saved research"))
                    .clicked()
                {
                    match crate::research::load_run(&self.dir, self.research_resume_id.trim()) {
                        Ok(run) => {
                            let mut request = crate::research::ResearchRequest::new(
                                settings.clone(), run.workspace, self.dir.clone(), run.question, settings.active_model().into(),
                            );
                            request.resume_id = Some(run.id);
                            request.max_rounds = self.research_rounds.max(run.rounds_completed).clamp(1, 8);
                            request.max_sources = self.research_sources;
                            request.web_enabled = self.research_web;
                            self.configure_research_sources(&mut request);
                            self.start_research(request);
                        }
                        Err(error) => self.error = Some(format!("{error:#}")),
                    }
                }
            });
            ui.small("Resume keeps the saved question, workspace and source notes, using the selected model and current research budgets/web settings. It does not repeat completed source analysis.");
        });
        self.job_results(ui);
    }

    pub(super) fn configure_research_sources(
        &self,
        request: &mut crate::research::ResearchRequest,
    ) {
        request.seed_urls = self
            .research_seed_urls
            .lines()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect();
        request.allowed_domains = self
            .research_domains
            .split([',', '\n'])
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect();
    }

    pub(super) fn start_research(&mut self, request: crate::research::ResearchRequest) {
        let dir = self.dir.clone();
        self.start("Planning research questions…", move |tx, cancel| {
            let settings = request.settings.clone();
            let data_dir = request.data_dir.clone();
            let run = crate::research::run_research(request, tx, cancel)?;
            let mut report = run.report();
            if settings.knowledge_notes && run.state == "done" {
                // A script picks the sentences from the retrieved pages; no model is used.
                match crate::notes::capture_research(&settings, &data_dir, &run) {
                    Ok(done) if !done.is_empty() => {
                        let list = done
                            .iter()
                            .map(|a| format!("{} (+{})", a.title, a.added))
                            .collect::<Vec<_>>()
                            .join("; ");
                        report.push_str(&format!("\n\n> Saved to knowledge notes: {list}"));
                    }
                    Ok(_) => {}
                    Err(e) => report.push_str(&format!("\n\n> Knowledge notes skipped: {e}")),
                }
            }
            let path =
                dir.join("research_runs")
                    .join(format!("{}-{}.md", run.id, uuid::Uuid::new_v4()));
            std::fs::write(&path, &report)?;
            Ok((report, vec![path]))
        });
    }
}
