//! The body of an approval dialog: file changes as a coloured diff, code as
//! source, and the remaining details as JSON.

use crate::diff::{self, LineKind};
use eframe::egui::{self, Color32, RichText};
use serde_json::Value;

/// Lines of context around each change in a preview.
const CONTEXT_LINES: usize = 3;
/// Diff lines drawn before the rest is summarised (keeps huge rewrites responsive).
const MAX_DRAWN_LINES: usize = 4_000;

fn colours(ui: &egui::Ui) -> (Color32, Color32, Color32, Color32) {
    if ui.visuals().dark_mode {
        (
            Color32::from_rgb(28, 64, 40),
            Color32::from_rgb(170, 230, 180),
            Color32::from_rgb(78, 32, 36),
            Color32::from_rgb(245, 175, 180),
        )
    } else {
        (
            Color32::from_rgb(220, 245, 226),
            Color32::from_rgb(20, 90, 40),
            Color32::from_rgb(252, 225, 228),
            Color32::from_rgb(140, 25, 35),
        )
    }
}

/// Draw `before` → `after` as unified hunks with line numbers.
pub fn show_diff(ui: &mut egui::Ui, before: &str, after: &str) {
    let hunks = diff::hunks(before, after, CONTEXT_LINES);
    let stats = diff::stats(&hunks);
    let (add_bg, add_fg, del_bg, del_fg) = colours(ui);
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(format!("+{}", stats.added))
                .color(add_fg)
                .strong(),
        );
        ui.label(
            RichText::new(format!("−{}", stats.removed))
                .color(del_fg)
                .strong(),
        );
        if hunks.is_empty() {
            ui.label("No changes to the file's text.");
        }
    });
    let mut drawn = 0usize;
    for hunk in &hunks {
        if drawn >= MAX_DRAWN_LINES {
            break;
        }
        ui.label(
            RichText::new(format!(
                "@@ −{},{} +{},{} @@",
                hunk.old_start, hunk.old_len, hunk.new_start, hunk.new_len
            ))
            .monospace()
            .weak(),
        );
        let (mut old_line, mut new_line) = (hunk.old_start, hunk.new_start);
        for (kind, text) in &hunk.lines {
            if drawn >= MAX_DRAWN_LINES {
                break;
            }
            drawn += 1;
            let (marker, number, background, foreground) = match kind {
                LineKind::Context => {
                    let n = new_line;
                    old_line += 1;
                    new_line += 1;
                    (' ', n, None, None)
                }
                LineKind::Removed => {
                    let n = old_line;
                    old_line += 1;
                    ('-', n, Some(del_bg), Some(del_fg))
                }
                LineKind::Added => {
                    let n = new_line;
                    new_line += 1;
                    ('+', n, Some(add_bg), Some(add_fg))
                }
            };
            let mut line = RichText::new(format!("{number:>5} {marker} {text}"))
                .monospace()
                .size(13.0);
            if let Some(colour) = foreground {
                line = line.color(colour);
            }
            if let Some(colour) = background {
                line = line.background_color(colour);
            }
            ui.add(
                egui::Label::new(line)
                    .selectable(true)
                    .wrap_mode(egui::TextWrapMode::Extend),
            );
        }
    }
    let total: usize = hunks.iter().map(|h| h.lines.len()).sum();
    if total > drawn {
        ui.label(
            RichText::new(format!(
                "… {} more diff lines not shown (+{} −{} in total).",
                total - drawn,
                stats.added,
                stats.removed
            ))
            .italics(),
        );
    }
}

/// The scrollable body of an approval: diff or code first, then other details.
pub fn show_action_details(ui: &mut egui::Ui, details: &Value) {
    let mut rest = details.clone();
    let map = rest.as_object_mut();
    let mut shown_special = false;
    if let Some(map) = map {
        let before = map.get("before").and_then(Value::as_str).map(str::to_owned);
        let after = map.get("after").and_then(Value::as_str).map(str::to_owned);
        if let (Some(before), Some(after)) = (before, after) {
            map.remove("before");
            map.remove("after");
            show_diff(ui, &before, &after);
            shown_special = true;
        } else if let Some(code) = map.remove("code") {
            if let Some(code) = code.as_str() {
                ui.add(
                    egui::Label::new(RichText::new(code).monospace().size(13.0)).selectable(true),
                );
                shown_special = true;
            }
        }
    }
    if shown_special {
        ui.separator();
    }
    ui.add(
        egui::Label::new(
            RichText::new(serde_json::to_string_pretty(&rest).unwrap_or_default())
                .monospace()
                .size(13.0),
        )
        .selectable(true),
    );
}
