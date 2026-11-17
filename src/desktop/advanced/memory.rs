//! AdvancedUi memory responsibilities.
use super::*;

impl AdvancedUi {
    pub(super) fn reload_memories(&mut self, workspace: Option<&Path>) {
        let scope = crate::memory::scope(workspace);
        match MemoryStore::open(&self.dir)
            .and_then(|s| s.list(Some(&scope), &self.memory_query, 500))
        {
            Ok(rows) => self.memory_rows = rows,
            Err(e) => self.error = Some(e.to_string()),
        }
    }

    pub(super) fn memory_page(
        &mut self,
        ui: &mut egui::Ui,
        settings: &mut Settings,
        workspace: Option<&Path>,
    ) {
        ui.heading("Automatic long-term memory");
        ui.label("Useful user-stated facts are extracted after completed chats and recalled in later conversations. Every automatic fact retains its exact supporting quote; file contents and assistant guesses are excluded. Project memories stay scoped to that project.");
        ui.horizontal_wrapped(|ui| {
            let changed = ui
                .checkbox(
                    &mut settings.automatic_memory,
                    "Automatically learn and recall",
                )
                .changed();
            ui.label("Maximum recalled facts");
            let changed = ui
                .add(egui::DragValue::new(&mut settings.memory_limit).range(1..=24))
                .changed()
                || changed;
            if changed {
                if let Err(e) = settings.save(&self.dir) {
                    self.error = Some(e.to_string());
                }
            }
        });
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.memory_query)
                    .hint_text("Find a remembered fact…"),
            );
            if ui.button("Refresh / search").clicked() {
                self.reload_memories(workspace);
            }
            if ui.button("Copy memory JSON").clicked() {
                match MemoryStore::open(&self.dir).and_then(|s| s.export()) {
                    Ok(text) => ui.ctx().copy_text(text),
                    Err(e) => self.error = Some(e.to_string()),
                }
            }
        });
        if self.memory_rows.is_empty() {
            ui.label("Refresh to inspect saved memories. You can edit, pin, or forget individual facts below.");
        }
        let mut action = None;
        egui::ScrollArea::vertical()
            .id_salt("memory_records")
            .max_height((ui.available_height() - 180.0).clamp(150.0, 350.0))
            .show(ui, |ui| {
                for memory in &self.memory_rows {
                    ui.group(|ui| {
                        ui.strong(&memory.fact);
                        ui.small(format!(
                            "{} · {} · confidence {:.2} · recalled {} time(s)",
                            memory.key, memory.scope, memory.confidence, memory.used
                        ));
                        ui.label(format!("Source: {}", memory.source));
                        ui.add(
                            egui::Label::new(format!("User quote: {}", memory.quote))
                                .selectable(true),
                        );
                        ui.horizontal(|ui| {
                            if ui.small_button("Edit").clicked() {
                                self.memory_edit = Some((memory.id.clone(), memory.fact.clone()));
                            }
                            if ui
                                .small_button(if memory.pinned { "Unpin" } else { "Pin" })
                                .clicked()
                            {
                                action = Some((memory.id.clone(), Some(!memory.pinned)));
                            }
                            if ui.small_button("Forget").clicked() {
                                action = Some((memory.id.clone(), None));
                            }
                        });
                    });
                }
            });
        if let Some((id, pin)) = action {
            let result = MemoryStore::open(&self.dir).and_then(|s| {
                if let Some(pin) = pin {
                    s.pin(&id, pin)
                } else {
                    s.forget(&id)
                }
            });
            if let Err(e) = result {
                self.error = Some(e.to_string());
            }
            self.reload_memories(workspace);
        }
        if let Some((id, text)) = &mut self.memory_edit {
            ui.separator();
            ui.label("Edit remembered fact");
            ui.add(
                egui::TextEdit::multiline(text)
                    .desired_rows(2)
                    .desired_width(f32::INFINITY),
            );
            let mut done = false;
            ui.horizontal(|ui| {
                if ui.button("Save correction").clicked() {
                    match MemoryStore::open(&self.dir).and_then(|s| s.edit(id, text)) {
                        Ok(()) => done = true,
                        Err(e) => self.error = Some(e.to_string()),
                    }
                }
                if ui.button("Cancel edit").clicked() {
                    done = true;
                }
            });
            if done {
                self.memory_edit = None;
                self.reload_memories(workspace);
            }
        }
        egui::CollapsingHeader::new("Remember an explicit fact").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Topic");
                ui.text_edit_singleline(&mut self.memory_new_key);
                ui.checkbox(&mut self.memory_global, "Global across projects");
            });
            ui.add(
                egui::TextEdit::multiline(&mut self.memory_new_fact)
                    .desired_rows(2)
                    .desired_width(f32::INFINITY),
            );
            if ui.button("Remember").clicked() {
                let scope = if self.memory_global {
                    "global".into()
                } else {
                    crate::memory::scope(workspace)
                };
                let c = MemoryCandidate {
                    key: self.memory_new_key.clone(),
                    fact: self.memory_new_fact.clone(),
                    quote: self.memory_new_fact.clone(),
                    confidence: 1.0,
                };
                match MemoryStore::open(&self.dir)
                    .and_then(|s| s.upsert(&scope, &c, "User entered", true))
                {
                    Ok(_) => {
                        self.memory_new_key.clear();
                        self.memory_new_fact.clear();
                        self.reload_memories(workspace);
                    }
                    Err(e) => self.error = Some(e.to_string()),
                }
            }
        });
    }
}
