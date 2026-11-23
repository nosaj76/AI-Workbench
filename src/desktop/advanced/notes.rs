//! AdvancedUi knowledge-notes responsibilities.
use super::*;
use crate::notes::{NoteStore, NoteSummary};

pub(super) struct NoteDraft {
    slug: String,
    original: String,
    text: String,
}

impl AdvancedUi {
    pub(super) fn reload_notes(&mut self) {
        let result = NoteStore::open(&self.dir).and_then(|store| {
            if self.note_query.trim().is_empty() {
                store.list()
            } else {
                Ok(store
                    .search(&self.note_query, 100)?
                    .into_iter()
                    .map(|n| {
                        let (agreed, mixed, disputed) = n.tally();
                        NoteSummary {
                            slug: n.slug,
                            title: n.title,
                            entries: n.entries.len(),
                            updated: n.updated,
                            agreed,
                            mixed,
                            disputed,
                        }
                    })
                    .collect())
            }
        });
        match result {
            Ok(rows) => {
                self.note_rows = rows;
                self.note_loaded = true;
            }
            Err(e) => self.error = Some(e.to_string()),
        }
    }

    pub(super) fn notes_page(&mut self, ui: &mut egui::Ui, settings: &mut Settings) {
        if !self.note_loaded {
            self.reload_notes();
        }
        ui.heading("Knowledge notes");
        ui.label("Durable facts from your chats, web searches and research are saved as one Markdown file per topic and recalled in later chats. Each fact shows whether it came from the web, research, or an unverified model answer, and which sources it came from. Chats that read your project files are never saved here.");
        ui.horizontal_wrapped(|ui| {
            if ui
                .checkbox(
                    &mut settings.knowledge_notes,
                    "Build knowledge notes automatically",
                )
                .changed()
            {
                if let Err(e) = settings.save(&self.dir) {
                    self.error = Some(e.to_string());
                }
            }
            let folder = self.dir.join(crate::notes::NOTES_DIR);
            ui.small(format!("Folder: {}", folder.display()));
            if ui.small_button("Copy folder path").clicked() {
                ui.ctx().copy_text(folder.display().to_string());
            }
        });
        ui.horizontal_wrapped(|ui| {
            let mut changed = ui
                .checkbox(&mut settings.notes_review_auto, "Have several models review new facts")
                .changed();
            ui.label("Models per fact");
            changed |= ui
                .add(egui::DragValue::new(&mut settings.notes_review_count).range(2..=5))
                .changed();
            if changed {
                if let Err(e) = settings.save(&self.dir) {
                    self.error = Some(e.to_string());
                }
            }
            if ui
                .add_enabled(!self.busy, egui::Button::new("Review all notes now"))
                .on_hover_text("Each different installed model judges every fact that has not been reviewed yet")
                .clicked()
            {
                self.review_notes(settings, None);
            }
        });
        let mut models = settings.notes_review_models.join(", ");
        ui.horizontal(|ui| {
            ui.label("Reviewer models");
            if ui
                .add(
                    egui::TextEdit::singleline(&mut models)
                        .hint_text("blank = the general, coding, writing and reasoning models of your routing profile")
                        .desired_width(f32::INFINITY),
                )
                .changed()
            {
                settings.notes_review_models = models
                    .split(',')
                    .map(|m| m.trim().to_string())
                    .filter(|m| !m.is_empty())
                    .collect();
                if let Err(e) = settings.save(&self.dir) {
                    self.error = Some(e.to_string());
                }
            }
        });
        ui.small("Each model is asked separately, without seeing any source, whether a fact is true. \"All 3 models agree\" raises confidence but is not proof; facts the models call wrong are kept out of chats.");
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut self.note_query).hint_text("Search notes…"));
            if ui.button("Search / refresh").clicked() {
                self.reload_notes();
            }
        });
        if self.note_rows.is_empty() {
            ui.label(if self.note_query.trim().is_empty() {
                "No notes yet. Notes appear after a chat that teaches something general, a web search, or a finished research run."
            } else {
                "No note matches that search."
            });
        }
        let mut open = None;
        let mut delete = None;
        let mut review = None;
        egui::ScrollArea::vertical()
            .id_salt("knowledge_notes")
            .max_height(if self.note_open.is_some() { 160.0 } else { 420.0 })
            .show(ui, |ui| {
                for row in &self.note_rows {
                    ui.horizontal(|ui| {
                        ui.strong(&row.title);
                        let reviewed = row.agreed + row.mixed + row.disputed;
                        ui.small(format!(
                            "{} entries · updated {} · {} agreed, {} mixed, {} disputed, {} not reviewed",
                            row.entries,
                            row.updated,
                            row.agreed,
                            row.mixed,
                            row.disputed,
                            row.entries.saturating_sub(reviewed)
                        ));
                        if ui.small_button("Open").clicked() {
                            open = Some(row.slug.clone());
                        }
                        if ui
                            .add_enabled(!self.busy, egui::Button::new("Review").small())
                            .clicked()
                        {
                            review = Some(row.slug.clone());
                        }
                        if ui.small_button("Delete").clicked() {
                            delete = Some(row.slug.clone());
                        }
                    });
                }
            });
        if let Some(slug) = review {
            self.review_notes(settings, Some(slug));
        }
        if let Some(slug) = open {
            match NoteStore::open(&self.dir)
                .and_then(|s| std::fs::read_to_string(s.path(&slug)).map_err(Into::into))
            {
                Ok(text) => self.note_open = Some(NoteDraft { slug, original: text.clone(), text }),
                Err(e) => self.error = Some(e.to_string()),
            }
        }
        if let Some(slug) = delete {
            match NoteStore::open(&self.dir).and_then(|s| s.delete(&slug)) {
                Ok(_) => {
                    if self.note_open.as_ref().is_some_and(|draft| draft.slug == slug) {
                        self.note_open = None;
                    }
                    self.reload_notes();
                }
                Err(e) => self.error = Some(e.to_string()),
            }
            ui.small("Deleted notes are kept in the notes folder under .deleted.");
        }
        if let Some(NoteDraft { slug, original, text }) = &mut self.note_open {
            ui.separator();
            ui.strong(format!("{slug}.md (plain Markdown; edit and save)"));
            egui::ScrollArea::vertical()
                .id_salt("knowledge_note_text")
                .max_height((ui.available_height() - 40.0).max(160.0))
                .show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(text)
                            .code_editor()
                            .desired_width(f32::INFINITY)
                            .desired_rows(14),
                    );
                });
            let mut close = false;
            let mut saved = false;
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    let write = NoteStore::open(&self.dir)
                        .and_then(|store| store.save_edit(slug, original, text));
                    match write {
                        Ok(()) => {
                            original.clone_from(text);
                            saved = true;
                        }
                        Err(e) => self.error = Some(e.to_string()),
                    }
                }
                if ui.button("Close").clicked() {
                    close = true;
                }
                if ui.button("Copy").clicked() {
                    ui.ctx().copy_text(text.clone());
                }
            });
            if saved {
                self.reload_notes();
            }
            if close {
                self.note_open = None;
            }
        }
        // Counts change when a review finishes: reload once the job is done.
        if self.note_review_pending && !self.busy {
            self.note_review_pending = false;
            self.reload_notes();
        }
        self.job_results(ui);
    }

    /// Has several different models judge the facts of one note, or of all.
    fn review_notes(&mut self, settings: &Settings, only: Option<String>) {
        let settings = settings.clone();
        let dir = self.dir.clone();
        self.note_review_pending = true;
        self.start(
            "Asking several models to review the facts…",
            move |_tx, cancel| {
                let slugs: Vec<String> = match only {
                    Some(slug) => vec![slug],
                    None => NoteStore::open(&dir)?
                        .list()?
                        .into_iter()
                        .map(|n| n.slug)
                        .collect(),
                };
                let reports = crate::notes::review_notes_blocking(
                    &settings, &dir, &slugs, None, None, cancel,
                )?;
                let lines = if reports.is_empty() {
                    "No notes to review.".to_string()
                } else {
                    reports
                        .iter()
                        .map(|r| format!("- {}", r.summary()))
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                Ok((format!("# Fact review\n\n{lines}\n"), Vec::new()))
            },
        );
    }
}
