//! A contextual, opt-in inspector for *saved evidence*, not fabricated model metadata.
use super::*;

impl Workbench {
    fn inspected_entry_index(&self) -> Option<usize> {
        self.selected_entry_id
            .as_deref()
            .and_then(|id| self.session.entries.iter().position(|entry| entry.id == id))
            .or_else(|| {
                self.session
                    .entries
                    .iter()
                    .rposition(|entry| entry.message.role == "assistant")
            })
    }

    /// Show a normal resizable right pane on a desktop; use a floating window
    /// when the viewport is too small for three permanent columns.
    pub(super) fn answer_inspector(&mut self, ctx: &egui::Context) {
        if self.page != Page::Chat || !self.show_inspector {
            return;
        }
        let mut close = false;
        if ctx.screen_rect().width() < 1080.0 {
            egui::Window::new("Answer details")
                .id(egui::Id::new("answer_inspector_compact"))
                .default_size(egui::vec2(340.0, 470.0))
                .resizable(true)
                .collapsible(false)
                .show(ctx, |ui| {
                    close = self.inspector_contents(ui, ctx);
                });
        } else {
            egui::SidePanel::right("answer_inspector")
                .default_width(320.0)
                .width_range(270.0..=460.0)
                .resizable(true)
                .show(ctx, |ui| {
                    close = self.inspector_contents(ui, ctx);
                });
        }
        if close {
            self.show_inspector = false;
        }
    }

    fn inspector_contents(&self, ui: &mut egui::Ui, ctx: &egui::Context) -> bool {
        let mut close = false;
        ui.horizontal(|ui| {
            ui.heading("Answer details");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("Close").clicked() {
                    close = true;
                }
            });
        });
        ui.label(
            RichText::new(
                "Inspect actual sources, model metadata, and tool activity for a selected message.",
            )
            .small()
            .color(ui.visuals().weak_text_color()),
        );
        ui.separator();
        let Some(index) = self.inspected_entry_index() else {
            ui.label("Select Details on an assistant response to inspect it.");
            return close;
        };
        let entry = &self.session.entries[index];
        // Tool evidence can be stored in separate tool-role entries. Bound the
        // display to this turn; never misattribute unrelated older tool results.
        let turn_start = self.session.entries[..=index]
            .iter()
            .rposition(|item| item.message.role == "user")
            .unwrap_or(index);
        egui::ScrollArea::vertical().id_salt("answer_details_scroll")
            .auto_shrink([false, false]).show(ui, |ui| {
                ui.strong(if entry.message.role == "assistant" { "Assistant answer" }
                    else if entry.message.role == "tool" { "Tool result" } else { "User message" });
                ui.label(format!("Message {} of {}", index + 1, self.session.entries.len()));
                if entry.model.is_empty() {
                    ui.label("Model: not recorded for this message");
                } else {
                    ui.horizontal(|ui| {
                        ui.label("Model:");
                        ui.add(egui::Label::new(&entry.model).selectable(true));
                    });
                }
                if entry.stats.get("incomplete").and_then(Value::as_bool) == Some(true) {
                    ui.colored_label(ui.visuals().warn_fg_color, "Incomplete response");
                }
                ui.add_space(10.0);
                ui.heading("Sources and evidence");
                // Input attachments/retrieval can belong to the originating user
                // turn; include them with their exact owner clearly indicated.
                let mut found_sources = false;
                for source_entry in &self.session.entries[turn_start..=index] {
                    if source_entry.sources.is_empty() && source_entry.context.is_empty() { continue; }
                    found_sources = true;
                    ui.collapsing(format!("{} evidence · {} source(s)",
                        source_entry.message.role, source_entry.sources.len()), |ui| {
                        for source in &source_entry.sources {
                            ui.horizontal_wrapped(|ui| {
                                ui.add(egui::Label::new(RichText::new(source).monospace().small()).selectable(true));
                                if ui.small_button("Copy").clicked() { ctx.copy_text(source.clone()); }
                            });
                        }
                        if !source_entry.context.is_empty() {
                            ui.collapsing("Retrieved text", |ui| {
                                ui.add(egui::Label::new(RichText::new(&source_entry.context).monospace().small()).selectable(true));
                            });
                        }
                    });
                }
                if !found_sources { ui.label("No indexed or attached source evidence was recorded for this turn."); }
                ui.add_space(12.0);
                ui.heading("Tool activity");
                let trace = self.session.entries[turn_start..=index].iter()
                    .filter(|item| item.message.role == "tool" || !item.message.tool_calls.is_empty())
                    .take(40).collect::<Vec<_>>();
                if trace.is_empty() { ui.label("No tool requests or results recorded for this turn."); }
                for (step, item) in trace.iter().enumerate() {
                    if item.message.role == "tool" {
                        let label = item.message.tool_name.as_deref().unwrap_or("Tool result");
                        ui.collapsing(format!("{} · {}", step + 1, label), |ui| {
                            // A UI-only bound on extremely large diagnostic output.
                            let shortened: String = item.message.content.chars().take(12_000).collect();
                            ui.add(egui::Label::new(RichText::new(shortened).monospace().small()).selectable(true));
                            if item.message.content.chars().count() > 12_000 { ui.label("Display truncated; complete result is retained in the conversation."); }
                        });
                    } else {
                        for call in &item.message.tool_calls {
                            ui.collapsing(format!("{} · requested {}", step + 1, call.function.name), |ui| {
                                ui.label(RichText::new(call.function.arguments.to_string()).monospace().small());
                            });
                        }
                    }
                }
                if entry.stats.get("unexecuted_tool_text").and_then(Value::as_bool) == Some(true)
                    || (entry.message.role == "assistant" && entry.message.tool_calls.is_empty()
                        && crate::runtime::inspect_tool_text(&entry.message.content).is_some()) {
                    ui.add_space(10.0);
                    ui.heading("Unexecuted model text");
                    ui.label("These tool-shaped JSON objects were ordinary model output. No tool or file action was performed from this text.");
                    ui.collapsing("Show raw response", |ui| {
                        let raw: String = entry.message.content.chars().take(16_000).collect();
                        ui.add(egui::Label::new(RichText::new(raw).monospace().small()).selectable(true));
                        if entry.message.content.chars().count() > 16_000 {
                            ui.label("Preview shortened; the saved conversation retains the complete response.");
                        }
                        if ui.small_button("Copy raw response").clicked() {
                            ctx.copy_text(entry.message.content.clone());
                        }
                    });
                }
                ui.add_space(12.0);
                ui.heading("Generation statistics");
                if entry.stats.is_null() || entry.stats == json!({}) {
                    ui.label("No generation statistics recorded.");
                } else {
                    let displayed = serde_json::to_string_pretty(&entry.stats).unwrap_or_else(|_| "Unavailable".into());
                    ui.add(egui::Label::new(RichText::new(displayed).monospace().small()).selectable(true));
                }
                ui.label(RichText::new("Routing decisions are inspectable in Activities → Model routing; an older decision is not attributed to this answer.")
                    .small().color(ui.visuals().weak_text_color()));
            });
        close
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inspector_selects_exact_message_and_falls_back_only_to_current_conversation() {
        let temp = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        let mut app = Workbench::from_context(&ctx, temp.path().to_path_buf(), true).unwrap();
        // Demo mode seeds a user/assistant exchange. Start a new conversation
        // and leave an old selection to prove it cannot select a prior entry.
        let old_message_id = app.session.entries.last().unwrap().id.clone();
        app.session = Session::default();
        app.selected_entry_id = Some(old_message_id);
        app.session.entries.push(Entry::new(ChatMessage {
            role: "assistant".into(),
            content: "One".into(),
            ..Default::default()
        }));
        app.session.entries.push(Entry::new(ChatMessage {
            role: "assistant".into(),
            content: "Two".into(),
            ..Default::default()
        }));
        assert_eq!(app.inspected_entry_index(), Some(1));
        app.selected_entry_id = Some(app.session.entries[0].id.clone());
        assert_eq!(app.inspected_entry_index(), Some(0));
        app.session.entries.clear();
        assert_eq!(app.inspected_entry_index(), None);
    }
}
