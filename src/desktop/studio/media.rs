//! StudioUi media responsibilities.
use super::*;

impl StudioUi {
    pub(super) fn media_form(&mut self, ui: &mut egui::Ui, workspace: Option<&Path>) {
        const NAMES: [&str; 8] = [
            "PDF page OCR",
            "Image OCR",
            "Audio transcription",
            "Video frame extraction",
            "Speech to audio file",
            "Podcast audio",
            "Remove image background",
            "LaTeX to PDF",
        ];
        let old = self.media_kind;
        egui::ComboBox::from_id_salt("media_activity")
            .selected_text(NAMES[self.media_kind])
            .show_ui(ui, |ui| {
                for (index, name) in NAMES.iter().enumerate() {
                    ui.selectable_value(&mut self.media_kind, index, *name);
                }
            });
        if self.media_kind != old {
            self.prepared = None;
            if self.media_kind == 2 {
                self.language = "auto".into();
            } else if matches!(self.media_kind, 0 | 1) && self.language == "auto" {
                self.language = "eng".into();
            }
            if self.media_kind == 7 && self.media_text.is_empty() {
                self.media_text="\\documentclass{article}\n\\usepackage[utf8]{inputenc}\n\\begin{document}\n\\section{Local document}\nWrite your LaTeX here.\n\\end{document}\n".into();
            }
        }
        if matches!(self.media_kind, 0 | 1 | 2 | 3 | 6) {
            ui.horizontal(|ui| {
                ui.label("Local input");
                ui.add(
                    egui::TextEdit::singleline(&mut self.media_input)
                        .desired_width((ui.available_width() - 90.0).max(120.0))
                        .hint_text("Absolute path or path inside the selected workspace"),
                );
                if ui.button("Browse…").clicked() {
                    self.pick_input(false, workspace);
                }
            });
        }
        match self.media_kind {
            0 => {
                ui.horizontal_wrapped(|ui| {
                    ui.label("First page");
                    ui.add(egui::DragValue::new(&mut self.first_page).range(1..=100000));
                    ui.label("Last page");
                    ui.add(egui::DragValue::new(&mut self.last_page).range(1..=100000));
                    ui.label("DPI");
                    ui.add(egui::DragValue::new(&mut self.dpi).range(50..=300));
                });
                text_row(ui, "OCR language", &mut self.language);
                ui.label("Renders up to 100 requested pages with Poppler and recognizes each with Tesseract. Saves page images and page-numbered text.");
            }
            1 => {
                text_row(ui, "OCR language", &mut self.language);
                ui.label("Runs installed Tesseract and saves its actual recognized text.");
            }
            2 => {
                text_row(ui, "Language / auto", &mut self.language);
                ui.checkbox(&mut self.translate_audio, "Translate speech to English");
                ui.horizontal(|ui| {
                    ui.label("Maximum seconds");
                    ui.add(
                        egui::DragValue::new(&mut self.max_audio_seconds)
                            .range(1..=21600)
                            .speed(60),
                    );
                });
                ui.label("Converts the selected audio/video track using FFmpeg, then produces text and subtitles with the configured Whisper model.");
            }
            3 => {
                ui.horizontal_wrapped(|ui| {
                    ui.label("Interval seconds");
                    ui.add(
                        egui::DragValue::new(&mut self.frame_interval)
                            .range(0.1..=3600.0)
                            .speed(0.5),
                    );
                    ui.label("Maximum frames");
                    ui.add(egui::DragValue::new(&mut self.frame_count).range(1..=120));
                });
                ui.label("Saves timed still images with a manifest. Use the vision-review control below after extraction; sampled frames are not a complete video/audio analysis.");
            }
            4 | 5 => {
                ui.horizontal_wrapped(|ui| {
                    ui.label("Speech engine");
                    egui::ComboBox::from_id_salt("speech_engine")
                        .selected_text(if self.speech_engine == SpeechEngine::Espeak {
                            "eSpeak"
                        } else {
                            "Piper"
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.speech_engine,
                                SpeechEngine::Espeak,
                                "eSpeak",
                            );
                            ui.selectable_value(
                                &mut self.speech_engine,
                                SpeechEngine::Piper,
                                "Piper",
                            );
                        });
                    ui.label("Words / minute");
                    ui.add(egui::DragValue::new(&mut self.speech_rate).range(80..=450));
                });
                text_row(
                    ui,
                    if self.media_kind == 5 {
                        "Host voice"
                    } else {
                        "Voice"
                    },
                    &mut self.host_voice,
                );
                if self.media_kind == 5 {
                    text_row(ui, "Guest voice", &mut self.guest_voice);
                    ui.label("Use HOST: and GUEST: speech lines. A document podcast script can be sent here from its completed result.");
                }
                ui.add(
                    egui::TextEdit::multiline(&mut self.media_text)
                        .desired_rows(8)
                        .desired_width(f32::INFINITY)
                        .hint_text(if self.media_kind == 5 {
                            "HOST: Welcome to our discussion.\nGUEST: Here is the first point…"
                        } else {
                            "Text to synthesize…"
                        }),
                );
            }
            6 => {
                ui.checkbox(
                    &mut self.cutout_semantic,
                    "Semantic cutout with installed rembg + local u2net.onnx",
                );
                if !self.cutout_semantic {
                    ui.horizontal_wrapped(|ui| {
                        ui.add(
                            egui::Slider::new(&mut self.cutout_tolerance, 0..=150)
                                .text("Color tolerance"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.cutout_feather, 0..=100)
                                .text("Edge feather"),
                        );
                    });
                    ui.label("Native mode estimates the plain background from image corners and removes connected border color. It preserves existing transparency; complex backgrounds need semantic mode.");
                }
            }
            7 => {
                ui.label("LaTeX source");
                ui.add(
                    egui::TextEdit::multiline(&mut self.media_text)
                        .code_editor()
                        .desired_rows(12)
                        .desired_width(f32::INFINITY),
                );
                ui.label("Compiles twice with installed pdflatex, shell escape disabled. Saves both source and PDF in a new activity directory.");
            }
            _ => {}
        }
        egui::CollapsingHeader::new("Local program and model settings").show(ui, |ui| {
            text_row(ui, "FFmpeg", &mut self.config.media.ffmpeg);
            text_row(ui, "ffprobe", &mut self.config.media.ffprobe);
            text_row(ui, "PDF page inspector", &mut self.config.media.pdfinfo);
            text_row(ui, "PDF renderer", &mut self.config.media.pdftoppm);
            text_row(ui, "Tesseract", &mut self.config.media.tesseract);
            text_row(ui, "Whisper executable", &mut self.config.media.whisper);
            path_row(ui, "Whisper model", &mut self.config.media.whisper_model);
            text_row(ui, "eSpeak", &mut self.config.media.espeak);
            text_row(ui, "Piper", &mut self.config.media.piper);
            path_row(ui, "Piper voice model", &mut self.config.media.piper_model);
            text_row(ui, "pdflatex", &mut self.config.media.pdflatex);
            text_row(ui, "rembg", &mut self.config.media.rembg);
            path_row(
                ui,
                "rembg model folder",
                &mut self.config.media.rembg_model_directory,
            );
            ui.horizontal(|ui| {
                ui.label("Seconds per command");
                ui.add(
                    egui::DragValue::new(&mut self.config.media.timeout_secs)
                        .range(1..=3600)
                        .speed(30),
                );
            });
            if ui.button("Save local program settings").clicked() {
                self.save_settings();
            }
        });
        if ui
            .button(RichText::new("Review media activity…").strong())
            .clicked()
        {
            match self.media_request(workspace) {
                Ok(request) => self.prepare(StudioJob::Media(Box::new((
                    self.config.media.clone(),
                    request,
                )))),
                Err(e) => self.error = Some(format!("{e:#}")),
            }
        }
    }

    pub(super) fn media_request(&self, workspace: Option<&Path>) -> Result<MediaRequest> {
        let input = || input_path(&self.media_input, workspace);
        Ok(match self.media_kind {
            0 => MediaRequest::PdfOcr {
                input: input()?,
                first_page: self.first_page,
                last_page: self.last_page,
                dpi: self.dpi,
                language: self.language.clone(),
            },
            1 => MediaRequest::ImageOcr {
                input: input()?,
                language: self.language.clone(),
            },
            2 => MediaRequest::Transcribe {
                input: input()?,
                language: self.language.clone(),
                translate_to_english: self.translate_audio,
                max_seconds: self.max_audio_seconds,
            },
            3 => MediaRequest::VideoFrames {
                input: input()?,
                every_seconds: self.frame_interval,
                max_frames: self.frame_count,
            },
            4 => MediaRequest::Speech {
                text: self.media_text.clone(),
                engine: self.speech_engine,
                voice: self.host_voice.clone(),
                words_per_minute: self.speech_rate,
            },
            5 => MediaRequest::Podcast {
                script: self.media_text.clone(),
                engine: self.speech_engine,
                host_voice: self.host_voice.clone(),
                guest_voice: self.guest_voice.clone(),
                words_per_minute: self.speech_rate,
            },
            6 => MediaRequest::RemoveBackground {
                input: input()?,
                color: None,
                tolerance: self.cutout_tolerance,
                feather: self.cutout_feather,
                semantic_model: self.cutout_semantic,
            },
            7 => MediaRequest::LatexPdf {
                source: self.media_text.clone(),
            },
            _ => bail!("Unknown media activity"),
        })
    }
}
