//! AdvancedUi review responsibilities.
use super::*;

impl AdvancedUi {
    pub(super) fn refresh_review(&mut self, workspace: &Path, settings: &Settings) {
        let root = workspace.to_path_buf();
        let db = self.dir.join("review.sqlite3");
        let mut request = ReviewRequest::new(
            settings.clone(), root.clone(), self.dir.clone(), settings.active_model().into(),
        );
        request.instruction = self.review_instruction.clone();
        request.chunk_chars = self.review_chunk;
        let results = self.sender.clone();
        self.start(
            "Scanning every eligible workspace file…",
            move |events, cancel| {
                let workspace = Workspace::new(root)?;
                let mut queue = ReviewQueue::open(db)?;
                let stats = queue.refresh_for_review(&workspace, &request, &cancel, &events)?;
                let coverage = queue.coverage(&workspace)?;
                let report = format!(
                    "# Review manifest\n\n{}\n\nScan details:\n```json\n{}\n```",
                    coverage.markdown(),
                    serde_json::to_string_pretty(&stats)?
                );
                let items = queue.items(&workspace, None, 500)?;
                results.send(UiEvent::Reviews {
                    root: workspace.root().to_path_buf(),
                    identity: request.identity(),
                    coverage,
                    items,
                })?;
                Ok((report, vec![]))
            },
        );
    }

    pub(super) fn review_page(
        &mut self,
        ui: &mut egui::Ui,
        settings: &Settings,
        workspace: Option<&Path>,
    ) {
        ui.heading("Comprehensive project review queue");
        ui.label("The queue tracks every eligible file chunk, its source hash, review result, and coverage. Completed unchanged chunks are retained when a run resumes.");
        let Some(workspace) = workspace else {
            ui.label("Choose a local workspace in Files first.");
            return;
        };
        ui.monospace(workspace.display().to_string());
        ui.add(
            egui::TextEdit::multiline(&mut self.review_instruction)
                .desired_rows(3)
                .desired_width(f32::INFINITY),
        );
        ui.horizontal_wrapped(|ui| {
            ui.label("Characters per chunk");
            ui.add(
                egui::DragValue::new(&mut self.review_chunk)
                    .range(1000..=16000)
                    .speed(250),
            );
            ui.label("Batch limit (0 = complete queue)");
            ui.add(egui::DragValue::new(&mut self.review_batch).range(0..=10000));
        });
        let current_identity = ReviewRequest::objective_identity(
            &self.review_instruction, settings.active_model(), &settings.base_url,
        );
        if self.review_result_identity != current_identity {
            self.review_items.clear();
            self.review_selected.clear();
            self.review_coverage = None;
        }
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(!self.busy, egui::Button::new("Build / refresh manifest"))
                .clicked()
            {
                self.refresh_review(workspace, settings);
            }
            if ui
                .add_enabled(!self.busy, egui::Button::new("Run / resume review"))
                .clicked()
            {
                let mut req = ReviewRequest::new(
                    settings.clone(),
                    workspace.to_path_buf(),
                    self.dir.clone(),
                    settings.active_model().into(),
                );
                req.instruction = self.review_instruction.clone();
                req.chunk_chars = self.review_chunk;
                req.max_items = self.review_batch;
                let events = self.sender.clone();
                let dir = self.dir.clone();
                let root = workspace.to_path_buf();
                self.start("Reviewing pending file chunks…", move |tx, cancel| {
                    let identity = req.identity();
                    let run = review_queue::run_review(req, tx, cancel)?;
                    let ws = Workspace::new(root)?;
                    let q = ReviewQueue::open(dir.join("review.sqlite3"))?;
                    let items = q.items(&ws, None, 500)?;
                    events.send(UiEvent::Reviews {
                        root: ws.root().to_path_buf(),
                        identity,
                        coverage: run.coverage,
                        items,
                    })?;
                    Ok((run.report, vec![run.full_report_path]))
                });
            }
            if ui
                .add_enabled(!self.busy, egui::Button::new("Retry failed items"))
                .clicked()
            {
                let result = (|| -> Result<()> {
                    let ws = Workspace::new(workspace.to_path_buf())?;
                    ReviewQueue::open(self.dir.join("review.sqlite3"))?.retry_failed(&ws)?;
                    Ok(())
                })();
                if let Err(e) = result {
                    self.error = Some(e.to_string());
                } else {
                    self.refresh_review(workspace, settings);
                }
            }
        });
        if let Some(c) = &self.review_coverage {
            ui.label(format!("Files fully reviewed: {} / {} · chunks: {} complete, {} queued, {} running, {} failed · current complete coverage: {}",c.fully_reviewed_files,c.files,c.done,c.queued,c.running,c.errors,c.complete()));
            if c.truncated_files + c.extraction_errors + c.stale_files > 0 {
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    format!(
                        "{} truncated, {} extraction errors, {} changed files require attention",
                        c.truncated_files, c.extraction_errors, c.stale_files
                    ),
                );
            }
        }
        egui::CollapsingHeader::new("Queue items (first 500)").show(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("review_items")
                .max_height(200.0)
                .show(ui, |ui| {
                    for item in &self.review_items {
                        if ui
                            .selectable_label(
                                self.review_selected == item.id,
                                format!(
                                    "{} · {}:{}–{} · attempt {}",
                                    item.state,
                                    item.path,
                                    item.start_line,
                                    item.end_line,
                                    item.attempts
                                ),
                            )
                            .clicked()
                        {
                            self.review_selected = item.id.clone();
                        }
                    }
                });
            if let Some(item) = self
                .review_items
                .iter()
                .find(|i| i.id == self.review_selected)
            {
                ui.monospace(format!(
                    "{}:{}–{} · SHA-256 {}",
                    item.path, item.start_line, item.end_line, item.source_hash
                ));
                if let Some(e) = &item.error {
                    ui.label(e);
                }
                egui::ScrollArea::vertical()
                    .max_height(180.0)
                    .show(ui, |ui| {
                        ui.label(&item.report);
                    });
            }
            ui.horizontal(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut self.review_skip_reason)
                        .hint_text("Reason for explicitly skipping this item"),
                );
                if ui
                    .add_enabled(
                        !self.busy
                            && !self.review_selected.is_empty()
                            && !self.review_skip_reason.trim().is_empty(),
                        egui::Button::new("Skip selected"),
                    )
                    .clicked()
                {
                    let result = (|| -> Result<()> {
                        let ws = Workspace::new(workspace.to_path_buf())?;
                        ReviewQueue::open(self.dir.join("review.sqlite3"))?.skip(
                            &ws,
                            &self.review_selected,
                            &self.review_skip_reason,
                        )?;
                        Ok(())
                    })();
                    if let Err(e) = result {
                        self.error = Some(e.to_string());
                    } else {
                        self.refresh_review(workspace, settings);
                    }
                }
            });
        });
        self.job_results(ui);
    }
}
