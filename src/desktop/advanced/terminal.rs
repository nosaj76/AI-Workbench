//! AdvancedUi terminal responsibilities.
use super::*;

impl AdvancedUi {
    pub(super) fn terminal_page(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        workspace: Option<&Path>,
    ) {
        self.terminal_manager_for(workspace);
        ui.heading("Persistent interactive terminals");
        ui.label("Each session is a real PTY. Shell variables, working directory and interactive programs survive command calls and page changes. Processes run with your account permissions.");
        ui.label("The AI uses these same sessions through its terminal tools. You can watch its work here or take over the keyboard. Sessions are kept separately for each workspace.");
        let sessions = self.terminals.list();
        if !sessions
            .iter()
            .any(|session| session.id == self.terminal_id)
        {
            self.terminal_id = sessions
                .first()
                .map(|session| session.id.clone())
                .unwrap_or_default();
            self.terminal_scrollback = 0;
        }
        ui.horizontal_wrapped(|ui| {
            if ui.button("New terminal").clicked() {
                let mut config = TerminalConfig {
                    program: "bash".into(),
                    ..Default::default()
                };
                if let Some(root) = workspace {
                    config.cwd = root.to_path_buf();
                }
                config.title = format!("Terminal {}", self.terminals.list().len() + 1);
                match self.terminals.open(config) {
                    Ok(id) => {
                        self.terminal_id = id;
                        self.terminal_scrollback = 0;
                    }
                    Err(e) => self.error = Some(e.to_string()),
                }
            }
            for terminal in self.terminals.list() {
                if ui
                    .selectable_label(
                        self.terminal_id == terminal.id,
                        format!(
                            "{}{}",
                            terminal.title,
                            if terminal.alive { "" } else { " [exited]" }
                        ),
                    )
                    .clicked()
                {
                    self.terminal_id = terminal.id;
                    self.terminal_scrollback = 0;
                }
            }
        });
        if self.terminal_id.is_empty() {
            return;
        }
        ui.horizontal(|ui| {
            if ui.button("Interrupt · Ctrl+C").clicked() {
                let _ = self
                    .terminals
                    .key(&self.terminal_id, TerminalKey::Interrupt);
            }
            if ui.button("Close session").clicked() {
                if let Err(e) = self.terminals.close(&self.terminal_id) {
                    self.error = Some(e.to_string());
                }
                self.terminal_id = self
                    .terminals
                    .list()
                    .first()
                    .map(|t| t.id.clone())
                    .unwrap_or_default();
            }
            if ui.button("Copy screen").clicked() {
                if let Ok(screen) = self.terminals.snapshot(&self.terminal_id) {
                    ctx.copy_text(screen.text);
                }
            }
            if let Ok(summary) = self.terminals.summary(&self.terminal_id) {
                let mut shared = summary.ai_access;
                let label = if summary.sandboxed {
                    "AI may use (sandboxed)"
                } else {
                    "AI may use (not sandboxed)"
                };
                if ui
                    .checkbox(&mut shared, label)
                    .on_hover_text("Lets the assistant read this terminal and, with command approval, type into it. Terminals you open are private by default; with automatic command approval the assistant could run anything your account can in an unsandboxed one.")
                    .changed()
                {
                    if let Err(e) = self.terminals.set_ai_access(&self.terminal_id, shared) {
                        self.error = Some(e.to_string());
                    }
                }
            }
            ui.label("Scrollback");
            if ui
                .add(egui::DragValue::new(&mut self.terminal_scrollback).range(0..=10_000))
                .changed()
            {
                let _ = self
                    .terminals
                    .scrollback(&self.terminal_id, self.terminal_scrollback);
            }
        });
        if self.terminal_id.is_empty() {
            return;
        }
        let font = egui::FontId::monospace(13.0);
        let cell_width = ui
            .painter()
            .layout_no_wrap("W".into(), font.clone(), Color32::WHITE)
            .size()
            .x
            .max(5.0);
        let cell_height = 17.0;
        let size = egui::vec2(
            ui.available_width(),
            (ui.available_height() - 92.0).clamp(170.0, 700.0),
        );
        let cols = (size.x / cell_width).floor().clamp(20.0, 240.0) as u16;
        let rows = (size.y / cell_height).floor().clamp(8.0, 80.0) as u16;
        if self.terminal_sizes.get(&self.terminal_id) != Some(&(rows, cols))
            && self.terminals.resize(&self.terminal_id, rows, cols).is_ok()
        {
            self.terminal_sizes
                .insert(self.terminal_id.clone(), (rows, cols));
        }
        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(cols as f32 * cell_width, rows as f32 * cell_height),
            egui::Sense::click(),
        );
        if response.clicked() {
            response.request_focus();
        }
        if response.has_focus() {
            ui.memory_mut(|memory| {
                memory.set_focus_lock_filter(
                    response.id,
                    egui::EventFilter {
                        tab: true,
                        horizontal_arrows: true,
                        vertical_arrows: true,
                        escape: true,
                    },
                )
            });
        }
        ui.painter()
            .rect_filled(rect, 3.0, Color32::from_rgb(12, 16, 22));
        match self.terminals.snapshot(&self.terminal_id) {
            Ok(screen) => {
                for (row, cells) in screen.cells.iter().enumerate() {
                    for (col, cell) in cells.iter().enumerate() {
                        if cell.wide_continuation {
                            continue;
                        }
                        let point = rect.min
                            + egui::vec2(col as f32 * cell_width, row as f32 * cell_height);
                        let mut foreground = cell
                            .foreground
                            .map(|c| Color32::from_rgb(c[0], c[1], c[2]))
                            .unwrap_or(Color32::from_rgb(217, 226, 235));
                        let mut background = cell
                            .background
                            .map(|c| Color32::from_rgb(c[0], c[1], c[2]))
                            .unwrap_or(Color32::from_rgb(12, 16, 22));
                        if cell.inverse {
                            std::mem::swap(&mut foreground, &mut background);
                        }
                        let width = cell_width * if cell.wide { 2.0 } else { 1.0 };
                        ui.painter().rect_filled(
                            egui::Rect::from_min_size(point, egui::vec2(width, cell_height)),
                            0.0,
                            background,
                        );
                        ui.painter().text(
                            point,
                            egui::Align2::LEFT_TOP,
                            &cell.text,
                            font.clone(),
                            foreground,
                        );
                        if cell.bold {
                            ui.painter().text(
                                point + egui::vec2(0.35, 0.0),
                                egui::Align2::LEFT_TOP,
                                &cell.text,
                                font.clone(),
                                foreground,
                            );
                        }
                        if cell.underline {
                            ui.painter().line_segment(
                                [
                                    point + egui::vec2(0.0, cell_height - 2.0),
                                    point + egui::vec2(width, cell_height - 2.0),
                                ],
                                egui::Stroke::new(1.0_f32, foreground),
                            );
                        }
                    }
                }
                if response.has_focus() && !screen.cursor_hidden {
                    let point = rect.min
                        + egui::vec2(
                            screen.cursor.1 as f32 * cell_width,
                            (screen.cursor.0 as f32 + 1.0) * cell_height - 2.0,
                        );
                    ui.painter().line_segment(
                        [point, point + egui::vec2(cell_width, 0.0)],
                        egui::Stroke::new(2.0_f32, Color32::from_rgb(116, 216, 204)),
                    );
                }
                if !screen.summary.alive {
                    ui.label(format!("Session exited: {:?}", screen.summary.exit_code));
                }
                if let Some(error) = screen.error {
                    ui.label(error);
                }
            }
            Err(e) => {
                self.error = Some(e.to_string());
            }
        }
        if response.has_focus() {
            let events = ctx.input(|i| i.events.clone());
            for event in events {
                let result = match event {
                    egui::Event::Text(text) => {
                        self.terminals.input(&self.terminal_id, text.as_bytes())
                    }
                    egui::Event::Paste(text) => self.terminals.paste(&self.terminal_id, &text),
                    egui::Event::Key {
                        key,
                        pressed: true,
                        modifiers,
                        ..
                    } => {
                        if modifiers.ctrl {
                            let byte = match key {
                                egui::Key::A => Some(1),
                                egui::Key::B => Some(2),
                                egui::Key::C => Some(3),
                                egui::Key::D => Some(4),
                                egui::Key::E => Some(5),
                                egui::Key::F => Some(6),
                                egui::Key::H => Some(8),
                                egui::Key::K => Some(11),
                                egui::Key::L => Some(12),
                                egui::Key::N => Some(14),
                                egui::Key::P => Some(16),
                                egui::Key::R => Some(18),
                                egui::Key::U => Some(21),
                                egui::Key::W => Some(23),
                                egui::Key::Z => Some(26),
                                _ => None,
                            };
                            if let Some(byte) = byte {
                                self.terminals.input(&self.terminal_id, &[byte])
                            } else {
                                Ok(())
                            }
                        } else {
                            let key = match key {
                                egui::Key::Enter => Some(TerminalKey::Enter),
                                egui::Key::Backspace => Some(TerminalKey::Backspace),
                                egui::Key::Tab => Some(TerminalKey::Tab),
                                egui::Key::Escape => Some(TerminalKey::Escape),
                                egui::Key::ArrowUp => Some(TerminalKey::Up),
                                egui::Key::ArrowDown => Some(TerminalKey::Down),
                                egui::Key::ArrowLeft => Some(TerminalKey::Left),
                                egui::Key::ArrowRight => Some(TerminalKey::Right),
                                egui::Key::Home => Some(TerminalKey::Home),
                                egui::Key::End => Some(TerminalKey::End),
                                egui::Key::Delete => Some(TerminalKey::Delete),
                                egui::Key::PageUp => Some(TerminalKey::PageUp),
                                egui::Key::PageDown => Some(TerminalKey::PageDown),
                                egui::Key::F1 => Some(TerminalKey::Function(1)),
                                egui::Key::F2 => Some(TerminalKey::Function(2)),
                                egui::Key::F3 => Some(TerminalKey::Function(3)),
                                egui::Key::F4 => Some(TerminalKey::Function(4)),
                                egui::Key::F5 => Some(TerminalKey::Function(5)),
                                egui::Key::F6 => Some(TerminalKey::Function(6)),
                                egui::Key::F7 => Some(TerminalKey::Function(7)),
                                egui::Key::F8 => Some(TerminalKey::Function(8)),
                                egui::Key::F9 => Some(TerminalKey::Function(9)),
                                egui::Key::F10 => Some(TerminalKey::Function(10)),
                                egui::Key::F11 => Some(TerminalKey::Function(11)),
                                egui::Key::F12 => Some(TerminalKey::Function(12)),
                                _ => None,
                            };
                            if let Some(key) = key {
                                self.terminals.key(&self.terminal_id, key)
                            } else {
                                Ok(())
                            }
                        }
                    }
                    _ => Ok(()),
                };
                if let Err(e) = result {
                    self.error = Some(e.to_string());
                }
            }
        }
        ui.small("Click the terminal screen to send keyboard input. Ctrl+C interrupts the foreground program; Ctrl+D sends EOF. Chat Stop leaves the session running. Close session or exit Workbench to end it.");
        ui.horizontal(|ui| {
            let input = ui.add(
                egui::TextEdit::singleline(&mut self.terminal_input)
                    .hint_text("Optional command entry")
                    .desired_width((ui.available_width() - 90.0).max(100.0)),
            );
            let enter = input.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if ui.button("Send line").clicked() || enter {
                let command = std::mem::take(&mut self.terminal_input);
                if let Err(e) = self
                    .terminals
                    .input(&self.terminal_id, format!("{command}\r").as_bytes())
                {
                    self.error = Some(e.to_string());
                }
            }
        });
    }
}
