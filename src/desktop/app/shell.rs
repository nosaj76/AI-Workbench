//! Conventional compact desktop shell. Keep common actions in one toolbar;
//! put advanced work in menus instead of permanently occupying the chat area.
use super::*;

impl Workbench {
    pub(super) fn apply_theme(&self, ctx: &egui::Context) {
        crate::theme::apply(ctx, self.settings.dark_mode, self.settings.font_scale);
    }

    pub(super) fn set_theme(&mut self, ctx: &egui::Context, dark: bool) {
        if self.settings.dark_mode == dark {
            return;
        }
        self.settings.dark_mode = dark;
        if let Some(draft) = &mut self.settings_draft {
            draft.dark_mode = dark;
        }
        if let Some(original) = &mut self.settings_draft_original {
            original.dark_mode = dark;
        }
        self.apply_theme(ctx);
        self.save_preferences();
    }

    pub(super) fn reset_interface_size(&mut self, ctx: &egui::Context) {
        self.settings.font_scale = 1.0;
        if let Some(draft) = &mut self.settings_draft {
            draft.font_scale = 1.0;
        }
        if let Some(original) = &mut self.settings_draft_original {
            original.font_scale = 1.0;
        }
        self.apply_theme(ctx);
        self.save_preferences();
    }

    pub(super) fn sidebar(&mut self, ctx: &egui::Context) {
        if !self.show_sidebar {
            return;
        }
        egui::SidePanel::left("conversation_sidebar")
            .default_width(220.0)
            .width_range(180.0..=320.0)
            .resizable(true)
            .frame(
                egui::Frame::default()
                    .fill(ctx.style().visuals.faint_bg_color)
                    .inner_margin(egui::Margin::symmetric(10, 9)),
            )
            .show(ctx, |ui| {
                let idle = !self.busy && !self.work_busy && self.approval.is_none();
                let new_chat = crate::theme::primary_button(ui.visuals(), "+  New chat")
                    .min_size(egui::vec2(ui.available_width(), 28.0));
                if ui
                    .add_enabled(idle, new_chat)
                    .on_hover_text("Start a new conversation (Ctrl+N)")
                    .clicked()
                {
                    self.new_conversation();
                    self.show_trash = false;
                    self.refresh_conversation_list();
                }
                ui.add_space(6.0);
                let mut change_filter = ui
                    .add(
                        egui::TextEdit::singleline(&mut self.session_search)
                            .hint_text("🔍 Search conversations")
                            .desired_width(f32::INFINITY),
                    )
                    .changed();
                ui.add_space(2.0);
                ui.horizontal(|ui| {
                    if ui.selectable_label(!self.show_trash, "Chats").clicked() {
                        self.show_trash = false;
                        change_filter = true;
                    }
                    if ui.selectable_label(self.show_trash, "🗑 Trash").clicked() {
                        self.show_trash = true;
                        change_filter = true;
                    }
                    if !self.show_trash {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .add(egui::Checkbox::new(&mut self.include_archived, "Archived"))
                                .changed()
                            {
                                change_filter = true;
                            }
                        });
                    }
                });
                if change_filter {
                    self.refresh_conversation_list();
                }
                ui.add_space(2.0);
                let list_height = (ui.available_height() - 92.0).max(45.0);
                let mut action = None;
                let now = chrono::Local::now();
                egui::ScrollArea::vertical()
                    .id_salt("conversation_list")
                    .max_height(list_height)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        // The open conversation is not listed until it has been
                        // saved; show it so the list never claims nothing is open.
                        let current_unlisted = !self.show_trash
                            && self.session_search.trim().is_empty()
                            && !self.session.entries.is_empty()
                            && !self.summaries.iter().any(|s| s.id == self.session.id);
                        if current_unlisted {
                            sidebar_group_label(ui, "Current");
                            conversation_row(ui, &self.session.title, "now", true, false, false);
                        }
                        let mut group = "";
                        for summary in &self.summaries {
                            let selected = !self.show_trash && summary.id == self.session.id;
                            let label = if summary.pinned && !self.show_trash {
                                "Pinned"
                            } else {
                                date_group(&summary.updated_at, now)
                            };
                            if label != group {
                                group = label;
                                sidebar_group_label(ui, label);
                            }
                            let row = conversation_row(
                                ui,
                                &summary.title,
                                &relative_time(&summary.updated_at, now),
                                selected,
                                summary.archived && !self.show_trash,
                                !idle || self.show_trash,
                            );
                            if row.clicked {
                                action = Some(ConversationAction::Open(summary.id.clone()));
                            }
                            if let Some(menu) = row.menu {
                                let ui = &mut ui.new_child(egui::UiBuilder::new().max_rect(menu));
                                ui.menu_button("...", |ui| {
                                    conversation_menu(
                                        ui,
                                        summary,
                                        self.show_trash,
                                        idle,
                                        &mut action,
                                    );
                                });
                            }
                        }
                        if self.summaries.is_empty() && !current_unlisted {
                            ui.add_space(16.0);
                            ui.vertical_centered(|ui| {
                                let text = if self.show_trash {
                                    "Trash is empty."
                                } else if !self.session_search.trim().is_empty() {
                                    "No matching conversations."
                                } else {
                                    "No saved conversations yet."
                                };
                                ui.label(RichText::new(text).color(ui.visuals().weak_text_color()));
                            });
                        }
                    });
                if let Some(action) = action {
                    self.conversation_action(action);
                }
                ui.add_space(4.0);
                self.workspace_card(ui, idle);
            });
    }

    fn workspace_card(&mut self, ui: &mut egui::Ui, idle: bool) {
        let visuals = ui.visuals().clone();
        egui::Frame::default()
            .fill(visuals.window_fill)
            .stroke(visuals.widgets.noninteractive.bg_stroke)
            .corner_radius(egui::CornerRadius::same(8))
            .inner_margin(egui::Margin::symmetric(9, 7))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.label(
                    RichText::new("WORKSPACE")
                        .small()
                        .strong()
                        .color(visuals.weak_text_color()),
                );
                let name = self
                    .session
                    .project
                    .as_deref()
                    .and_then(|path| Path::new(path).file_name())
                    .map(|name| name.to_string_lossy().into_owned());
                ui.horizontal(|ui| {
                    let text = match &name {
                        Some(name) => RichText::new(format!("📁 {name}")).strong(),
                        None => {
                            RichText::new("No folder selected").color(visuals.weak_text_color())
                        }
                    };
                    ui.add(egui::Label::new(text).truncate()).on_hover_text(
                        self.session
                            .project
                            .as_deref()
                            .unwrap_or("No workspace selected"),
                    );
                });
                let label = if name.is_some() {
                    "Change folder..."
                } else {
                    "Open folder..."
                };
                if ui
                    .add_enabled(
                        idle,
                        egui::Button::new(label).min_size(egui::vec2(ui.available_width(), 0.0)),
                    )
                    .clicked()
                {
                    self.browse_workspace();
                }
            });
    }

    /// Toolbar status pill: Ollama connection state at a glance.
    fn connection_pill(&self, ui: &mut egui::Ui) {
        let visuals = ui.visuals().clone();
        let connected = !self.version.is_empty() && !self.demo;
        let (text, color) = if self.demo {
            ("Preview".to_owned(), visuals.warn_fg_color)
        } else if connected {
            (
                format!("Ollama {}", self.version),
                status_ok_color(&visuals),
            )
        } else {
            ("Offline".to_owned(), visuals.error_fg_color)
        };
        // Painted at its exact size: a Frame in the right-to-left toolbar
        // would stretch across all remaining width.
        let galley = ui.painter().layout_no_wrap(
            text,
            egui::TextStyle::Small.resolve(ui.style()),
            visuals.text_color(),
        );
        let size = egui::vec2(galley.size().x + 28.0, 20.0);
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::hover());
        let painter = ui.painter();
        painter.rect_stroke(
            rect,
            egui::CornerRadius::same(10),
            visuals.widgets.noninteractive.bg_stroke,
            egui::StrokeKind::Inside,
        );
        painter.circle_filled(egui::pos2(rect.left() + 11.0, rect.center().y), 4.0, color);
        painter.galley(
            egui::pos2(rect.left() + 20.0, rect.center().y - galley.size().y / 2.0),
            galley,
            visuals.text_color(),
        );
        response.on_hover_text(if self.demo {
            "Preview mode: sample data, no model requests"
        } else if connected {
            "Connected to the configured Ollama endpoint"
        } else {
            "Ollama is not reachable at the configured endpoint"
        });
    }

    /// Compact model selector for the chat header; connection state is shown
    /// by the toolbar pill.
    pub(super) fn connection_picker(&mut self, ui: &mut egui::Ui) {
        let mut chosen = self.settings.active_model().to_owned();
        let previous = chosen.clone();
        egui::ComboBox::from_id_salt("model_selector")
            .selected_text(if chosen.is_empty() {
                "Select a model".to_owned()
            } else {
                chosen.clone()
            })
            .width(ui.available_width().clamp(140.0, 240.0))
            .show_ui(ui, |ui| {
                for model in &self.models {
                    ui.selectable_value(&mut chosen, model.name.clone(), &model.name);
                }
            });
        if chosen != previous {
            if let Some(profile) = self
                .settings
                .profiles
                .get_mut(self.settings.selected_profile)
            {
                if !profile.model.is_empty() {
                    profile.model = chosen.clone();
                }
            }
            self.settings.model = chosen;
            self.save_preferences();
        }
        if ui
            .add_enabled(!self.work_busy && !self.demo, egui::Button::new("⟳"))
            .on_hover_text("Refresh the installed model list")
            .clicked()
        {
            self.refresh_models();
        }
    }

    pub(super) fn header(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("application_toolbar")
            .frame(
                egui::Frame::default()
                    .fill(ctx.style().visuals.panel_fill)
                    .inner_margin(egui::Margin::symmetric(10, 6)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(self.show_sidebar, RichText::new("☰").size(15.0))
                        .on_hover_text("Show or hide conversations")
                        .clicked()
                    {
                        self.show_sidebar = !self.show_sidebar;
                    }
                    brand_mark(ui);
                    ui.label(RichText::new("Workbench").strong().size(14.0));
                    ui.add_space(10.0);
                    for (page, title) in [
                        (Page::Chat, "Chat"),
                        (Page::Files, "Files"),
                        (Page::Models, "Models"),
                        (Page::Knowledge, "Knowledge"),
                    ] {
                        if ui.selectable_label(self.page == page, title).clicked() {
                            self.page = page;
                        }
                    }
                    ui.menu_button("More", |ui| {
                        for (section, title) in [
                            (crate::advanced_ui::Section::Agents, "Agents"),
                            (crate::advanced_ui::Section::Review, "Project review"),
                            (crate::advanced_ui::Section::Research, "Research"),
                            (crate::advanced_ui::Section::Qa, "Q&A"),
                            (crate::advanced_ui::Section::Terminal, "Terminal"),
                            (crate::advanced_ui::Section::Memory, "Memory"),
                            (crate::advanced_ui::Section::Notes, "Knowledge notes"),
                            (crate::advanced_ui::Section::Routing, "Model routing"),
                            (crate::advanced_ui::Section::Studio, "Studio"),
                        ] {
                            if ui.button(title).clicked() {
                                self.advanced.section = section;
                                self.page = Page::Advanced;
                                ui.close_menu();
                            }
                        }
                        ui.separator();
                        if ui.button("Tools and external agents").clicked() {
                            self.page = Page::Tools;
                            ui.close_menu();
                        }
                        if ui.button("Schedules").clicked() {
                            self.advanced.open_schedules();
                            self.page = Page::Tasks;
                            ui.close_menu();
                        }
                        if ui.button("Ollama service").clicked() {
                            self.advanced.open_runtime();
                            self.page = Page::Advanced;
                            ui.close_menu();
                        }
                        if ui.button("Model setup").clicked() {
                            self.page = Page::Setup;
                            ui.close_menu();
                        }
                        if ui.button("Diagnostics").clicked() {
                            self.page = Page::Diagnostics;
                            ui.close_menu();
                        }
                        ui.separator();
                        if ui.button("Reset interface size").clicked() {
                            self.reset_interface_size(ctx);
                            ui.close_menu();
                        }
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("⚙ Settings").clicked() {
                            self.page = Page::Settings;
                        }
                        // Segmented Light/Dark control, kept visible in the toolbar.
                        if ui
                            .selectable_label(self.settings.dark_mode, "Dark")
                            .clicked()
                        {
                            self.set_theme(ctx, true);
                        }
                        if ui
                            .selectable_label(!self.settings.dark_mode, "Light")
                            .clicked()
                        {
                            self.set_theme(ctx, false);
                        }
                        ui.add_space(6.0);
                        self.connection_pill(ui);
                    });
                });
            });
        egui::TopBottomPanel::bottom("application_status")
            .frame(
                egui::Frame::default()
                    .fill(ctx.style().visuals.panel_fill)
                    .inner_margin(egui::Margin::symmetric(8, 3)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.add(egui::Label::new(RichText::new(&self.status).small()).truncate());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new(concat!("v", env!("CARGO_PKG_VERSION"))).small())
                            .on_hover_text(concat!("Source build: ", env!("WORKBENCH_SOURCE_ID")));
                        if self.busy
                            || self.work_busy
                            || self.setup.is_busy()
                            || self.advanced.is_busy()
                        {
                            ui.spinner();
                        }
                    });
                });
            });
    }
}

fn brand_mark(ui: &mut egui::Ui) {
    let (fill, text) = (
        crate::theme::accent(ui.visuals()),
        crate::theme::on_accent(ui.visuals()),
    );
    // The "AI" badge; with the label beside it the toolbar reads "AI Workbench".
    let (rect, _) = ui.allocate_exact_size(egui::vec2(24.0, 20.0), egui::Sense::hover());
    ui.painter()
        .rect_filled(rect, egui::CornerRadius::same(5), fill);
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        "AI",
        egui::FontId::proportional(11.0),
        text,
    );
}

fn status_ok_color(visuals: &egui::Visuals) -> egui::Color32 {
    crate::theme::success(visuals)
}

fn sidebar_group_label(ui: &mut egui::Ui, label: &str) {
    ui.add_space(6.0);
    ui.label(
        RichText::new(label.to_uppercase())
            .small()
            .strong()
            .color(ui.visuals().weak_text_color()),
    );
}

struct RowResponse {
    clicked: bool,
    /// Where to place the row's `⋯` menu button, when it has one.
    menu: Option<egui::Rect>,
}

/// One full-width conversation row: title, relative time, and a reserved
/// slot for the `⋯` menu. The selected row gets an accent bar.
fn conversation_row(
    ui: &mut egui::Ui,
    title: &str,
    time: &str,
    selected: bool,
    archived: bool,
    disabled: bool,
) -> RowResponse {
    let width = ui.available_width();
    let sense = if disabled {
        egui::Sense::hover()
    } else {
        egui::Sense::click()
    };
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, 30.0), sense);
    let visuals = ui.visuals().clone();
    let fill = if selected {
        visuals.selection.bg_fill
    } else if response.hovered() && !disabled {
        visuals.widgets.hovered.weak_bg_fill
    } else {
        egui::Color32::TRANSPARENT
    };
    let painter = ui.painter();
    painter.rect_filled(rect, egui::CornerRadius::same(6), fill);
    if selected {
        painter.rect_filled(
            egui::Rect::from_min_size(
                rect.min + egui::vec2(0.0, 7.0),
                egui::vec2(3.0, rect.height() - 14.0),
            ),
            egui::CornerRadius::same(2),
            crate::theme::accent(&visuals),
        );
    }
    let menu = egui::Rect::from_min_size(
        egui::pos2(rect.right() - 28.0, rect.top() + 3.0),
        egui::vec2(26.0, 24.0),
    );
    let weak = visuals.weak_text_color();
    let time_galley =
        painter.layout_no_wrap(time.to_owned(), egui::FontId::proportional(11.0), weak);
    let time_pos = egui::pos2(
        menu.left() - 2.0 - time_galley.size().x,
        rect.center().y - time_galley.size().y / 2.0,
    );
    let title_max = (time_pos.x - rect.left() - 18.0).max(20.0);
    let text = if archived {
        format!("{title} · archived")
    } else {
        title.to_owned()
    };
    let color = if disabled && !selected {
        weak
    } else {
        visuals.text_color()
    };
    let mut job =
        egui::text::LayoutJob::simple_singleline(text, egui::FontId::proportional(12.5), color);
    job.wrap = egui::text::TextWrapping::truncate_at_width(title_max);
    let title_galley = ui.fonts(|fonts| fonts.layout_job(job));
    let painter = ui.painter();
    painter.galley(
        egui::pos2(
            rect.left() + 10.0,
            rect.center().y - title_galley.size().y / 2.0,
        ),
        title_galley,
        color,
    );
    painter.galley(time_pos, time_galley, weak);
    let clicked = response.clicked();
    response.on_hover_text(title);
    RowResponse {
        clicked,
        menu: Some(menu),
    }
}

fn conversation_menu(
    ui: &mut egui::Ui,
    summary: &crate::storage::SessionSummary,
    trash: bool,
    idle: bool,
    action: &mut Option<ConversationAction>,
) {
    let mut item = |ui: &mut egui::Ui, label: &str, chosen: ConversationAction| {
        if ui.add_enabled(idle, egui::Button::new(label)).clicked() {
            *action = Some(chosen);
            ui.close_menu();
        }
    };
    let id = summary.id.clone();
    if trash {
        item(ui, "Restore", ConversationAction::Restore(id.clone()));
        item(
            ui,
            "Delete permanently...",
            ConversationAction::DeleteForever(id),
        );
    } else {
        item(
            ui,
            if summary.pinned { "Unpin" } else { "Pin" },
            ConversationAction::TogglePin(id.clone()),
        );
        item(
            ui,
            if summary.archived {
                "Unarchive"
            } else {
                "Archive"
            },
            ConversationAction::ToggleArchive(id.clone()),
        );
        ui.separator();
        item(ui, "Move to Trash...", ConversationAction::Trash(id));
    }
}

fn parse_local(updated_at: &str) -> Option<chrono::DateTime<chrono::Local>> {
    chrono::DateTime::parse_from_rfc3339(updated_at)
        .ok()
        .map(|t| t.with_timezone(&chrono::Local))
}

/// Sidebar section for a conversation's last update.
pub(super) fn date_group(updated_at: &str, now: chrono::DateTime<chrono::Local>) -> &'static str {
    let Some(time) = parse_local(updated_at) else {
        return "Older";
    };
    let days = (now.date_naive() - time.date_naive()).num_days();
    match days {
        ..=0 => "Today",
        1 => "Yesterday",
        2..=6 => "Previous 7 days",
        7..=29 => "Previous 30 days",
        _ => "Older",
    }
}

/// Compact relative time for a conversation row.
pub(super) fn relative_time(updated_at: &str, now: chrono::DateTime<chrono::Local>) -> String {
    let Some(time) = parse_local(updated_at) else {
        return String::new();
    };
    let minutes = (now - time).num_minutes();
    let days = (now.date_naive() - time.date_naive()).num_days();
    if minutes < 1 {
        "now".into()
    } else if minutes < 60 {
        format!("{minutes}m")
    } else if days <= 0 {
        time.format("%H:%M").to_string()
    } else if days < 7 {
        time.format("%a").to_string()
    } else if time.format("%Y").to_string() == now.format("%Y").to_string() {
        time.format("%b %-d").to_string()
    } else {
        time.format("%b %Y").to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn conversations_group_by_local_day_and_show_compact_times() {
        let now = chrono::Local
            .with_ymd_and_hms(2026, 11, 8, 15, 30, 0)
            .unwrap();
        let at = |days: i64, hours: i64| {
            (now - chrono::Duration::days(days) - chrono::Duration::hours(hours)).to_rfc3339()
        };
        assert_eq!(date_group(&at(0, 1), now), "Today");
        assert_eq!(date_group(&at(1, 0), now), "Yesterday");
        assert_eq!(date_group(&at(4, 0), now), "Previous 7 days");
        assert_eq!(date_group(&at(20, 0), now), "Previous 30 days");
        assert_eq!(date_group(&at(90, 0), now), "Older");
        assert_eq!(date_group("not a date", now), "Older");
        assert_eq!(relative_time(&now.to_rfc3339(), now), "now");
        assert_eq!(
            relative_time(&(now - chrono::Duration::minutes(5)).to_rfc3339(), now),
            "5m"
        );
        assert_eq!(relative_time(&at(0, 3), now), "12:30");
        assert_eq!(relative_time(&at(400, 0), now), "Oct 2025");
        assert_eq!(relative_time("garbage", now), "");
    }
}
