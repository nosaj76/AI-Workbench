//! Application appearance, restricted to APIs in the pinned egui 0.31.1.
use eframe::egui::{self, Color32, FontId, Stroke, TextStyle};

pub fn apply(ctx: &egui::Context, dark: bool, scale: f32) {
    ctx.set_theme(if dark {
        egui::Theme::Dark
    } else {
        egui::Theme::Light
    });
    let mut visuals = if dark {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    };
    let (background, surface, input, text, border, accent, selection) = if dark {
        (
            Color32::from_rgb(31, 32, 35),
            Color32::from_rgb(37, 38, 42),
            Color32::from_rgb(27, 28, 31),
            Color32::from_rgb(224, 227, 231),
            Color32::from_rgb(65, 68, 74),
            Color32::from_rgb(143, 181, 224),
            Color32::from_rgb(46, 64, 86),
        )
    } else {
        (
            Color32::from_rgb(245, 246, 248),
            Color32::WHITE,
            Color32::WHITE,
            Color32::from_rgb(34, 38, 44),
            Color32::from_rgb(210, 214, 220),
            Color32::from_rgb(31, 85, 145),
            Color32::from_rgb(224, 236, 249),
        )
    };
    visuals.panel_fill = background;
    visuals.window_fill = surface;
    visuals.extreme_bg_color = input;
    visuals.faint_bg_color = if dark {
        Color32::from_rgb(43, 47, 54)
    } else {
        Color32::from_rgb(236, 241, 248)
    };
    visuals.code_bg_color = if dark {
        Color32::from_rgb(25, 28, 33)
    } else {
        Color32::from_rgb(237, 240, 245)
    };
    visuals.hyperlink_color = accent;
    visuals.selection.bg_fill = selection;
    visuals.selection.stroke = Stroke::new(1.0_f32, text);
    visuals.window_stroke = Stroke::new(1.0_f32, border);
    visuals.widgets.noninteractive.fg_stroke.color = text;
    visuals.widgets.noninteractive.bg_stroke.color = border;
    visuals.widgets.inactive.fg_stroke.color = text;
    visuals.widgets.hovered.fg_stroke.color = text;
    visuals.widgets.active.fg_stroke.color = text;
    visuals.widgets.inactive.bg_fill = surface;
    visuals.widgets.inactive.weak_bg_fill = surface;
    visuals.widgets.inactive.bg_stroke.color = border;
    visuals.widgets.hovered.bg_fill = selection;
    visuals.widgets.hovered.weak_bg_fill = selection;
    visuals.widgets.active.bg_fill = selection;
    visuals.widgets.active.weak_bg_fill = selection;
    visuals.warn_fg_color = if dark {
        Color32::from_rgb(255, 200, 116)
    } else {
        Color32::from_rgb(137, 75, 0)
    };
    visuals.error_fg_color = if dark {
        Color32::from_rgb(255, 158, 158)
    } else {
        Color32::from_rgb(170, 28, 39)
    };
    // Small enough that checkboxes stay rounded squares rather than circles.
    let radius = egui::CornerRadius::same(4);
    for widget in [
        &mut visuals.widgets.noninteractive,
        &mut visuals.widgets.inactive,
        &mut visuals.widgets.hovered,
        &mut visuals.widgets.active,
        &mut visuals.widgets.open,
    ] {
        widget.corner_radius = radius;
    }
    visuals.window_corner_radius = egui::CornerRadius::same(10);
    visuals.menu_corner_radius = egui::CornerRadius::same(8);
    ctx.set_visuals(visuals);
    ctx.set_zoom_factor(if scale.is_finite() {
        scale.clamp(0.75, 1.8)
    } else {
        1.0
    });
    ctx.style_mut(|style| {
        style.spacing.item_spacing = egui::vec2(7.0, 6.0);
        style.spacing.button_padding = egui::vec2(8.0, 3.0);
        style.spacing.interact_size.y = 25.0;
        style.spacing.window_margin = egui::Margin::same(8);
        style.spacing.scroll = egui::style::ScrollStyle::solid();
        style.url_in_tooltip = true;
        style
            .text_styles
            .insert(TextStyle::Body, FontId::proportional(13.0));
        style
            .text_styles
            .insert(TextStyle::Button, FontId::proportional(12.5));
        style
            .text_styles
            .insert(TextStyle::Small, FontId::proportional(11.5));
        style
            .text_styles
            .insert(TextStyle::Monospace, FontId::monospace(12.5));
        style
            .text_styles
            .insert(TextStyle::Heading, FontId::proportional(17.0));
    });
    ctx.request_repaint();
}

/// The theme accent. `apply` stores it as the hyperlink color, so every
/// caller sees the value matching the active Light/Dark theme.
pub fn accent(visuals: &egui::Visuals) -> Color32 {
    visuals.hyperlink_color
}

/// Color for successful or completed states.
pub fn success(visuals: &egui::Visuals) -> Color32 {
    if visuals.dark_mode {
        Color32::from_rgb(96, 186, 160)
    } else {
        Color32::from_rgb(22, 122, 98)
    }
}

/// Readable text color on top of an accent fill.
pub fn on_accent(visuals: &egui::Visuals) -> Color32 {
    if visuals.dark_mode {
        Color32::from_rgb(18, 22, 28)
    } else {
        Color32::WHITE
    }
}

/// An accent-filled button for the single primary action in a region.
pub fn primary_button(visuals: &egui::Visuals, text: &str) -> egui::Button<'static> {
    egui::Button::new(
        egui::RichText::new(text.to_owned())
            .strong()
            .color(on_accent(visuals)),
    )
    .fill(accent(visuals))
    .min_size(egui::vec2(64.0, 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn both_themes_are_explicit_and_have_different_surfaces() {
        let ctx = egui::Context::default();
        apply(&ctx, false, 1.0);
        let light = ctx.style().visuals.clone();
        assert!(!light.dark_mode);
        apply(&ctx, true, 1.0);
        let dark = ctx.style().visuals.clone();
        assert!(dark.dark_mode);
        assert_ne!(light.panel_fill, dark.panel_fill);
        assert_ne!(light.text_color(), dark.text_color());
        assert_ne!(accent(&light), accent(&dark));
        assert_ne!(on_accent(&light), on_accent(&dark));
        apply(&ctx, false, 1.0);
        assert_eq!(ctx.style().visuals.panel_fill, light.panel_fill);
    }
}
