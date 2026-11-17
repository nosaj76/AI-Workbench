//! AdvancedUi Q&A responsibilities: ask community Q&A sites through the text
//! browsers, and browse the long-term Q&A library and its facts.
use super::*;
use crate::qa::{AnswerKind, CachedQa, Fact, QaAnswer, QaCache};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum QaView {
    Answers,
    Facts,
}

pub(super) enum QaLibraryAction {
    Reload,
    Rebuild,
    OpenFolder,
    Show(String),
    Delete(String),
    SetKind(String, AnswerKind),
}

impl AdvancedUi {
    pub(super) fn qa_page(&mut self, ui: &mut egui::Ui, settings: &Settings) {
        ui.heading("Q&A");
        ui.label(
            "Answers a question from community Q&A sites (Stack Overflow and the Stack Exchange \
             network, Super User, Ask Ubuntu, Reddit, GitHub discussions and more). One search runs \
             through your text browser (lynx, falling back to w3m); the best threads are read and the \
             selected model writes a cited answer.",
        );
        ui.label(
            "Every answer is kept in your Q&A library as an organised Markdown file. Factual answers \
             are kept as they are and recorded in the facts table (short question, one-sentence answer, \
             explanation). Answers that change over time are refreshed when you ask after their refresh \
             date; the previous answer stays in the file.",
        );
        if !settings.network_tools {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                "Network tools are off: saved answers still work, but new questions and refreshes need \
                 network access (Settings → Connection and privacy).",
            );
        }
        let browsers = crate::qa::search_browsers(settings);
        if browsers.is_empty() {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                "No text browser found. Install lynx or w3m to search Q&A sites.",
            );
        }
        ui.add(
            egui::TextEdit::multiline(&mut self.qa_question)
                .hint_text("Ask a question, e.g. How do I undo the last git commit?")
                .desired_rows(2)
                .desired_width(f32::INFINITY),
        );
        ui.horizontal_wrapped(|ui| {
            ui.label("Refresh answers that change every");
            ui.add(
                egui::DragValue::new(&mut self.qa_refresh_days)
                    .range(1..=365)
                    .suffix(" days"),
            )
            .on_hover_text(
                "Time-sensitive answers (latest versions, prices, news) refresh after at most 3 days. \
                 Factual answers are never refreshed.",
            );
            ui.label("Threads to read");
            ui.add(egui::DragValue::new(&mut self.qa_sources).range(1..=6));
            if !browsers.is_empty() {
                let names: Vec<&str> = browsers.iter().map(|b| b.name.as_str()).collect();
                ui.small(format!("Browser order: {}", names.join(" → ")));
            }
        });
        let can_ask = !self.busy && !self.qa_question.trim().is_empty();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(can_ask, crate::theme::primary_button(ui.visuals(), "Ask"))
                .on_hover_text("Uses the saved answer unless it is due for a refresh")
                .clicked()
            {
                self.start_qa(settings, false);
            }
            if ui
                .add_enabled(
                    can_ask && settings.network_tools,
                    egui::Button::new("Ask again online"),
                )
                .on_hover_text(
                    "Refresh now; the current answer moves to the file's previous answers",
                )
                .clicked()
            {
                self.start_qa(settings, true);
            }
        });
        ui.small(
            "Asking sends the question to DuckDuckGo and reads the matching threads (Stack Exchange \
             threads through its public API). The answer is written by your local model.",
        );
        if self.qa_refresh_list && !self.busy {
            self.qa_refresh_list = false;
            self.load_qa_library();
        }
        self.qa_library_section(ui);
        self.job_results(ui);
    }

    fn start_qa(&mut self, settings: &Settings, refresh: bool) {
        let mut request = crate::qa::QaRequest::new(
            settings.clone(),
            self.dir.clone(),
            self.qa_question.trim().to_owned(),
            settings.active_model().to_owned(),
        );
        request.refresh_days = self.qa_refresh_days;
        request.max_sources = self.qa_sources;
        request.refresh = refresh;
        self.qa_refresh_list = true;
        self.start("Checking the Q&A library…", move |tx, cancel| {
            let answer = crate::qa::run_qa(request, &tx, cancel)?;
            Ok((answer.report(), Vec::new()))
        });
    }

    fn load_qa_library(&mut self) {
        let search = self.qa_search.trim().to_owned();
        let result = QaCache::open(&self.dir).and_then(|cache| {
            let answers = if search.is_empty() {
                cache.list(500)?
            } else {
                cache.search(&search, 500)?
            };
            let facts = if search.is_empty() {
                cache.facts()?
            } else {
                cache.search_facts(&search, 500)?
            };
            Ok((answers, facts))
        });
        match result {
            Ok((answers, facts)) => {
                self.qa_cache = answers;
                self.qa_facts = facts;
            }
            Err(error) => self.error = Some(format!("Q&A library: {error:#}")),
        }
    }

    fn qa_library_section(&mut self, ui: &mut egui::Ui) {
        let mut action: Option<QaLibraryAction> = None;
        egui::CollapsingHeader::new("Q&A library")
            .id_salt("qa_library")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    let search = ui.add(
                        egui::TextEdit::singleline(&mut self.qa_search)
                            .hint_text("🔍 Search questions, answers and facts")
                            .desired_width(260.0),
                    );
                    if search.changed() {
                        action = Some(QaLibraryAction::Reload);
                    }
                    ui.selectable_value(
                        &mut self.qa_view,
                        QaView::Answers,
                        format!("Answers ({})", self.qa_cache.len()),
                    );
                    ui.selectable_value(
                        &mut self.qa_view,
                        QaView::Facts,
                        format!("Facts ({})", self.qa_facts.len()),
                    );
                });
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .button("Open library folder")
                        .on_hover_text("Markdown files by topic, with INDEX.md and FACTS.md")
                        .clicked()
                    {
                        action = Some(QaLibraryAction::OpenFolder);
                    }
                    if ui
                        .add_enabled(!self.busy, egui::Button::new("Rebuild index from files"))
                        .on_hover_text(
                            "Re-read every Markdown file into the search index and facts table",
                        )
                        .clicked()
                    {
                        action = Some(QaLibraryAction::Rebuild);
                    }
                    ui.checkbox(&mut self.qa_allow_delete, "Allow deleting");
                    ui.small(format!(
                        "{}",
                        self.dir.join(crate::qa::LIBRARY_DIR).display()
                    ));
                });
                ui.separator();
                match self.qa_view {
                    QaView::Answers => self.qa_answers_list(ui, &mut action),
                    QaView::Facts => self.qa_facts_list(ui, &mut action),
                }
            });
        if let Some(action) = action {
            self.qa_library_action(action);
        }
    }

    fn qa_answers_list(&mut self, ui: &mut egui::Ui, action: &mut Option<QaLibraryAction>) {
        if self.qa_cache.is_empty() {
            ui.label(
                RichText::new(if self.qa_search.trim().is_empty() {
                    "No saved answers yet."
                } else {
                    "No saved answers match the search."
                })
                .color(ui.visuals().weak_text_color()),
            );
            return;
        }
        egui::ScrollArea::vertical()
            .id_salt("qa_answers_list")
            .max_height(260.0)
            .show(ui, |ui| {
                for entry in &self.qa_cache {
                    ui.push_id(&entry.id, |ui| {
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(!self.busy, egui::Button::new("Show"))
                                .clicked()
                            {
                                *action = Some(QaLibraryAction::Show(entry.id.clone()));
                            }
                            if self.qa_allow_delete
                                && ui
                                    .add_enabled(!self.busy, egui::Button::new("Delete"))
                                    .clicked()
                            {
                                *action = Some(QaLibraryAction::Delete(entry.id.clone()));
                            }
                            kind_selector(ui, entry, action);
                            let due = entry.needs_refresh();
                            ui.add(
                                egui::Label::new(RichText::new(&entry.question).strong())
                                    .truncate(),
                            )
                            .on_hover_text(entry_details(entry));
                            ui.label(RichText::new(entry.refresh_label()).small().color(if due {
                                ui.visuals().warn_fg_color
                            } else {
                                ui.visuals().weak_text_color()
                            }));
                        });
                    });
                }
            });
    }

    fn qa_facts_list(&mut self, ui: &mut egui::Ui, action: &mut Option<QaLibraryAction>) {
        if self.qa_facts.is_empty() {
            ui.label(
                RichText::new(if self.qa_search.trim().is_empty() {
                    "No facts yet. Answers classified as factual are recorded here."
                } else {
                    "No facts match the search."
                })
                .color(ui.visuals().weak_text_color()),
            );
            return;
        }
        egui::ScrollArea::vertical()
            .id_salt("qa_facts_list")
            .max_height(260.0)
            .show(ui, |ui| {
                let mut topic = "";
                for fact in &self.qa_facts {
                    if self.qa_search.trim().is_empty() && fact.topic != topic {
                        topic = &fact.topic;
                        ui.add_space(4.0);
                        ui.label(RichText::new(topic.to_uppercase()).small().strong());
                    }
                    ui.push_id(&fact.id, |ui| {
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(!self.busy, egui::Button::new("Show"))
                                .on_hover_text("Open the full answer and sources")
                                .clicked()
                            {
                                *action = Some(QaLibraryAction::Show(fact.id.clone()));
                            }
                            ui.add(
                                egui::Label::new(RichText::new(format!(
                                    "{}  {}",
                                    fact.question, fact.answer
                                )))
                                .truncate(),
                            )
                            .on_hover_text(fact_details(fact));
                        });
                        ui.label(
                            RichText::new(&fact.explanation)
                                .small()
                                .color(ui.visuals().weak_text_color()),
                        );
                    });
                }
            });
    }

    fn qa_library_action(&mut self, action: QaLibraryAction) {
        let refresh_days = self.qa_refresh_days;
        let result = QaCache::open(&self.dir).and_then(|cache| match action {
            QaLibraryAction::Reload => Ok(()),
            QaLibraryAction::Rebuild => cache.import_library().map(|n| {
                self.status = format!("Rebuilt the Q&A index from {n} file(s)");
            }),
            QaLibraryAction::OpenFolder => {
                let folder = cache.library_dir().to_path_buf();
                std::process::Command::new("xdg-open")
                    .arg(&folder)
                    .spawn()
                    .map(|_| ())
                    .map_err(|e| anyhow::anyhow!("Open {}: {e}", folder.display()))
            }
            QaLibraryAction::Delete(id) => cache.delete(&id).map(|_| ()),
            QaLibraryAction::SetKind(id, kind) => cache.set_kind(&id, kind, refresh_days).map(|e| {
                self.status = format!("“{}” is now {}", e.question, e.refresh_label());
            }),
            QaLibraryAction::Show(id) => {
                if let Some(entry) = cache.get(&id)? {
                    self.qa_question = entry.question.clone();
                    let question = entry.question.clone();
                    let mut shown = QaAnswer::from_entry(&question, entry, true, Vec::new());
                    if shown.answer.trim().is_empty() {
                        shown.answer = "_Only sources were saved for this question; press Ask to write an answer from them (no website is contacted)._".into();
                    }
                    self.report = shown.report();
                    self.files = vec![cache.library_dir().join(&shown.path)];
                }
                Ok(())
            }
        });
        if let Err(error) = result {
            self.error = Some(format!("Q&A library: {error:#}"));
        }
        self.load_qa_library();
    }
}

/// A small selector to reclassify how an answer is kept.
fn kind_selector(ui: &mut egui::Ui, entry: &CachedQa, action: &mut Option<QaLibraryAction>) {
    let mut kind = entry.kind;
    egui::ComboBox::from_id_salt("kind")
        .width(130.0)
        .selected_text(kind.label())
        .show_ui(ui, |ui| {
            for option in AnswerKind::ALL {
                ui.selectable_value(&mut kind, option, option.label());
            }
        })
        .response
        .on_hover_text("Factual: kept forever, in the facts table. Changes over time: refreshed after the period above. Time-sensitive: refreshed after 3 days.");
    if kind != entry.kind {
        *action = Some(QaLibraryAction::SetKind(entry.id.clone(), kind));
    }
}

fn entry_details(entry: &CachedQa) -> String {
    let mut text = format!(
        "Saved {} · topic {} · used {}×\n{}/{}",
        crate::qa::short_date(&entry.saved_at),
        entry.topic,
        entry.hits,
        crate::qa::LIBRARY_DIR,
        entry.path
    );
    if !entry.summary.is_empty() {
        text.push_str(&format!("\n\n{}", entry.summary));
    }
    if !entry.history.is_empty() {
        text.push_str(&format!(
            "\n\n{} previous answer(s) kept",
            entry.history.len()
        ));
    }
    text
}

fn fact_details(fact: &Fact) -> String {
    format!(
        "{}\n\n{}\n\nSaved {} · {}\n{}",
        fact.answer,
        fact.explanation,
        crate::qa::short_date(&fact.saved_at),
        fact.library_path,
        fact.sources.join("\n")
    )
}
