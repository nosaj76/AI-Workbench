//! Live "Tool activity" console: pops open while chat tools run, streams command
//! output, and closes itself shortly after the last tool finishes. It is a view
//! only; complete results are still saved in the transcript as tool entries.
use eframe::egui::{self, Color32, RichText};
use std::time::{Duration, Instant};

const MAX_RUNS: usize = 30;
const MAX_OUTPUT_BYTES: usize = 64 * 1024;
const CLOSE_DELAY: Duration = Duration::from_millis(2500);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RunState {
    Running,
    Succeeded,
    Failed,
    Stopped,
}

#[derive(Debug, Clone)]
pub(super) struct ToolRun {
    pub id: String,
    pub name: String,
    pub summary: String,
    pub output: String,
    pub preview: String,
    pub state: RunState,
    pub started: Instant,
    pub finished: Option<Instant>,
}

impl ToolRun {
    fn elapsed(&self) -> Duration {
        self.finished.unwrap_or_else(Instant::now) - self.started
    }
}

#[derive(Debug, Default)]
pub(super) struct ToolConsole {
    pub runs: Vec<ToolRun>,
    pub open: bool,
    pub keep_open: bool,
    close_at: Option<Instant>,
}

impl ToolConsole {
    pub fn started(&mut self, id: String, name: String, summary: String) {
        if !self.open {
            // A new burst of activity starts with a clean console.
            self.runs.clear();
        }
        self.runs.push(ToolRun {
            id,
            name,
            summary,
            output: String::new(),
            preview: String::new(),
            state: RunState::Running,
            started: Instant::now(),
            finished: None,
        });
        if self.runs.len() > MAX_RUNS {
            self.runs.drain(..self.runs.len() - MAX_RUNS);
        }
        self.open = true;
        self.close_at = None;
    }

    pub fn output(&mut self, id: &str, chunk: &str) {
        if let Some(run) = self.runs.iter_mut().find(|r| r.id == id) {
            append_terminal_text(&mut run.output, chunk);
        }
    }

    pub fn finished(&mut self, id: &str, ok: bool, preview: String) {
        if let Some(run) = self.runs.iter_mut().find(|r| r.id == id) {
            run.state = if ok {
                RunState::Succeeded
            } else {
                RunState::Failed
            };
            run.preview = preview;
            run.finished = Some(Instant::now());
        }
        self.schedule_close_if_idle();
    }

    /// The chat request ended; anything still marked running was interrupted.
    pub fn chat_finished(&mut self) {
        for run in self
            .runs
            .iter_mut()
            .filter(|r| r.state == RunState::Running)
        {
            run.state = RunState::Stopped;
            run.finished = Some(Instant::now());
        }
        self.schedule_close_if_idle();
    }

    pub fn running(&self) -> usize {
        self.runs
            .iter()
            .filter(|r| r.state == RunState::Running)
            .count()
    }

    fn schedule_close_if_idle(&mut self) {
        if self.open && self.running() == 0 {
            self.close_at = Some(Instant::now() + CLOSE_DELAY);
        }
    }

    pub fn show(&mut self, ctx: &egui::Context, awaiting_approval: bool) {
        if !self.open {
            return;
        }
        let mut open = true;
        let screen = ctx.screen_rect();
        let window = egui::Window::new("Tool activity")
            .id(egui::Id::new("tool_activity_console"))
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .pivot(egui::Align2::RIGHT_BOTTOM)
            .default_pos(screen.right_bottom() - egui::vec2(16.0, 150.0))
            .default_size(egui::vec2(560.0, 320.0))
            .show(ctx, |ui| self.contents(ui, awaiting_approval));
        let hovered = window
            .as_ref()
            .is_some_and(|w| w.response.contains_pointer());
        if !open {
            self.open = false;
            self.close_at = None;
            return;
        }
        if let Some(close_at) = self.close_at {
            if self.keep_open || hovered {
                // Postpone while the user is reading or has pinned the console.
                self.close_at = Some(Instant::now() + CLOSE_DELAY);
            } else if Instant::now() >= close_at {
                self.open = false;
                self.close_at = None;
                return;
            }
            ctx.request_repaint_after(Duration::from_millis(250));
        } else if self.running() > 0 {
            ctx.request_repaint_after(Duration::from_millis(250));
        }
    }

    fn contents(&mut self, ui: &mut egui::Ui, awaiting_approval: bool) {
        let running = self.running();
        ui.horizontal(|ui| {
            let status = if running > 0 && awaiting_approval {
                "Waiting for your approval".to_owned()
            } else if running > 0 {
                format!("{running} running")
            } else if self.keep_open {
                "All tools finished".to_owned()
            } else {
                "All tools finished · closing".to_owned()
            };
            ui.label(RichText::new(status).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.checkbox(&mut self.keep_open, "Keep open");
            });
        });
        ui.separator();
        let last = self.runs.len().saturating_sub(1);
        egui::ScrollArea::vertical()
            .id_salt("tool_activity_scroll")
            .stick_to_bottom(true)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for (index, run) in self.runs.iter().enumerate() {
                    ui.push_id(&run.id, |ui| {
                        run_header(ui, run, awaiting_approval);
                        // Running tools and the most recent one show their output;
                        // earlier finished tools stay as one compact line.
                        if run.state == RunState::Running || index == last {
                            terminal(ui, run);
                        }
                    });
                    ui.add_space(4.0);
                }
            });
    }
}

fn status_color(visuals: &egui::Visuals, state: RunState) -> Color32 {
    match state {
        RunState::Running => crate::theme::accent(visuals),
        RunState::Succeeded => crate::theme::success(visuals),
        RunState::Failed => visuals.error_fg_color,
        RunState::Stopped => visuals.warn_fg_color,
    }
}

fn run_header(ui: &mut egui::Ui, run: &ToolRun, awaiting_approval: bool) {
    ui.horizontal(|ui| {
        let color = status_color(ui.visuals(), run.state);
        match run.state {
            RunState::Running => {
                ui.add(egui::Spinner::new().size(12.0).color(color));
            }
            RunState::Succeeded => {
                ui.label(RichText::new("✔").color(color));
            }
            RunState::Failed => {
                ui.label(RichText::new("✖").color(color));
            }
            RunState::Stopped => {
                ui.label(RichText::new("■").color(color));
            }
        }
        ui.label(RichText::new(&run.name).monospace().strong());
        let elapsed = run.elapsed().as_secs_f32();
        let detail = match run.state {
            RunState::Running if awaiting_approval => "awaiting approval".to_owned(),
            RunState::Stopped => "stopped".to_owned(),
            _ => format!("{elapsed:.1}s"),
        };
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                RichText::new(detail)
                    .small()
                    .color(ui.visuals().weak_text_color()),
            );
            if !run.summary.is_empty() {
                ui.add(
                    egui::Label::new(
                        RichText::new(&run.summary)
                            .monospace()
                            .small()
                            .color(ui.visuals().weak_text_color()),
                    )
                    .truncate(),
                );
            }
        });
    });
}

/// Terminal-style output pane for a console run.
fn terminal(ui: &mut egui::Ui, run: &ToolRun) {
    let text = if !run.output.is_empty() {
        run.output.as_str()
    } else if !run.preview.is_empty() {
        run.preview.as_str()
    } else {
        return;
    };
    terminal_text(ui, text, 180.0);
}

/// Monospace output pane, dark in both themes like a real console.
pub(super) fn terminal_text(ui: &mut egui::Ui, text: &str, max_height: f32) {
    egui::Frame::default()
        .fill(Color32::from_rgb(20, 22, 27))
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::Margin::symmetric(8, 6))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            egui::ScrollArea::vertical()
                .id_salt("terminal_output")
                .max_height(max_height)
                .stick_to_bottom(true)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    ui.add(
                        egui::Label::new(
                            RichText::new(text)
                                .monospace()
                                .size(11.5)
                                .color(Color32::from_rgb(214, 219, 226)),
                        )
                        .selectable(true)
                        .wrap(),
                    );
                });
        });
}

/// Glyph and color for a tool outcome label shared by the console and transcript.
pub(super) fn outcome_style(visuals: &egui::Visuals, outcome: &str) -> (&'static str, Color32) {
    match outcome {
        "completed" => ("✔", status_color(visuals, RunState::Succeeded)),
        "partial result" => ("◐", crate::theme::accent(visuals)),
        "failed" => ("✖", visuals.error_fg_color),
        _ => ("⊘", visuals.warn_fg_color),
    }
}

/// Append raw process output for display: strips ANSI escape sequences, applies
/// carriage returns (progress bars redraw one line), and keeps a bounded tail.
pub(super) fn append_terminal_text(buffer: &mut String, chunk: &str) {
    let mut chars = chunk.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\u{1b}' => {
                // CSI: ESC [ params... final byte in @..~ ; other escapes: skip one char.
                if chars.peek() == Some(&'[') {
                    chars.next();
                    for next in chars.by_ref() {
                        if ('@'..='~').contains(&next) {
                            break;
                        }
                    }
                } else {
                    chars.next();
                }
            }
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    continue;
                }
                let line_start = buffer.rfind('\n').map_or(0, |i| i + 1);
                buffer.truncate(line_start);
            }
            c if c.is_control() && c != '\n' && c != '\t' => {}
            c => buffer.push(c),
        }
    }
    if buffer.len() > MAX_OUTPUT_BYTES {
        let mut cut = buffer.len() - MAX_OUTPUT_BYTES;
        while !buffer.is_char_boundary(cut) {
            cut += 1;
        }
        let cut = buffer[cut..].find('\n').map_or(cut, |i| cut + i + 1);
        buffer.drain(..cut);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_text_strips_ansi_and_applies_carriage_returns() {
        let mut out = String::new();
        append_terminal_text(&mut out, "\u{1b}[1;32mok\u{1b}[0m done\r\n");
        append_terminal_text(&mut out, "progress 10%\rprogress 100%\n");
        assert_eq!(out, "ok done\nprogress 100%\n");
    }

    #[test]
    fn terminal_text_keeps_a_bounded_tail() {
        let mut out = String::new();
        for i in 0..20_000 {
            append_terminal_text(&mut out, &format!("line {i}\n"));
        }
        assert!(out.len() <= MAX_OUTPUT_BYTES);
        assert!(out.ends_with("line 19999\n"));
        assert!(out.starts_with("line "));
    }

    #[test]
    fn console_opens_on_start_and_schedules_close_when_idle() {
        let mut console = ToolConsole::default();
        console.started("a".into(), "run_command".into(), "$ ls".into());
        console.started("b".into(), "read_file".into(), "src/main.rs".into());
        assert!(console.open);
        console.output("a", "hello\n");
        console.finished("a", true, String::new());
        assert!(console.close_at.is_none(), "b is still running");
        console.finished("b", false, "error: missing".into());
        assert!(console.close_at.is_some());
        assert_eq!(console.runs[0].output, "hello\n");
        assert_eq!(console.runs[1].state, RunState::Failed);
    }

    #[test]
    fn chat_end_marks_unfinished_runs_stopped_and_new_burst_resets() {
        let mut console = ToolConsole::default();
        console.started("a".into(), "run_command".into(), String::new());
        console.chat_finished();
        assert_eq!(console.runs[0].state, RunState::Stopped);
        console.open = false;
        console.started("b".into(), "list_files".into(), String::new());
        assert_eq!(console.runs.len(), 1);
        assert_eq!(console.runs[0].id, "b");
    }
}
