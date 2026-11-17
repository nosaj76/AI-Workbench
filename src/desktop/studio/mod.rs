//! Native Studio panels and off-thread activity workers.

mod documents;
mod execution;
mod files;
mod images;
mod media;
mod results;

use crate::{
    document_activities::{self, DocumentAction, DocumentRequest},
    files::Workspace,
    image_generation::{self, ImageBackend, ImageConfig, ImageOutput, ImageRequest},
    media::{ActivityOutput, MediaRequest, MediaTools, SpeechEngine},
    settings::Settings,
};
use anyhow::{bail, Context, Result};
use eframe::egui::{self, RichText};
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
struct StudioConfig {
    image: ImageConfig,
    media: MediaTools,
}

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Images,
    Documents,
    Media,
}

#[derive(Clone)]
enum StudioJob {
    Image(Box<(ImageConfig, ImageRequest)>),
    Documents(Box<(Settings, DocumentRequest, PathBuf)>),
    Media(Box<(MediaTools, MediaRequest)>),
    ImageReview(Box<(Settings, Vec<PathBuf>, String)>),
}

#[derive(Clone)]
enum StudioOutput {
    Images(Box<ImageOutput>),
    Activity(Box<ActivityOutput>),
}
impl StudioOutput {
    fn directory(&self) -> &Path {
        match self {
            Self::Images(o) => &o.directory,
            Self::Activity(o) => &o.directory,
        }
    }
    fn files(&self) -> Vec<PathBuf> {
        match self {
            Self::Images(o) => o
                .images
                .iter()
                .map(|i| i.path.clone())
                .chain(std::iter::once(o.metadata.clone()))
                .collect(),
            Self::Activity(o) => o.files.clone(),
        }
    }
    fn text(&self) -> String {
        match self {
            Self::Images(o) => format!(
                "Generated {} image(s), seed {}.\n{}",
                o.images.len(),
                o.seed,
                o.images
                    .iter()
                    .map(|i| format!("{} ({} × {})", i.path.display(), i.width, i.height))
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
            Self::Activity(o) => o.text.clone(),
        }
    }
    fn warnings(&self) -> &[String] {
        match self {
            Self::Images(o) => &o.warnings,
            Self::Activity(o) => &o.warnings,
        }
    }
}

struct Thumbnail {
    path: PathBuf,
    size: [usize; 2],
    rgba: Vec<u8>,
}
enum StudioEvent {
    Status(String),
    Error(String),
    Done,
    Prepared(Box<(StudioJob, Value)>),
    Finished(Box<StudioOutput>, Vec<Thumbnail>),
    Models(Vec<String>),
    PickedInput(String, bool),
    PickedModel(String),
    PickedDocuments(Vec<String>),
}

pub struct StudioUi {
    data_dir: PathBuf,
    config: StudioConfig,
    tab: Tab,
    image: ImageRequest,
    image_input: String,
    workflow_json: String,
    extra_args_json: String,
    models: Vec<String>,
    document: DocumentRequest,
    document_paths: String,
    media_kind: usize,
    media_input: String,
    media_text: String,
    language: String,
    first_page: u32,
    last_page: u32,
    dpi: u32,
    max_audio_seconds: u32,
    translate_audio: bool,
    frame_interval: f64,
    frame_count: u32,
    speech_engine: SpeechEngine,
    host_voice: String,
    guest_voice: String,
    speech_rate: u32,
    cutout_tolerance: u8,
    cutout_feather: u8,
    cutout_semantic: bool,
    vision_question: String,
    prepared: Option<(StudioJob, Value)>,
    output: Option<Arc<StudioOutput>>,
    textures: Vec<(PathBuf, egui::TextureHandle)>,
    markdown: CommonMarkCache,
    attachment: Option<(String, String)>,
    status: String,
    error: Option<String>,
    busy: bool,
    started: Option<Instant>,
    cancel: Arc<AtomicBool>,
    tx: Sender<StudioEvent>,
    rx: Receiver<StudioEvent>,
    worker: Option<thread::JoinHandle<()>>,
}

impl StudioUi {
    pub fn new(data_dir: PathBuf) -> Self {
        let (config, error) = match load_config(&data_dir) {
            Ok(c) => (c, None),
            Err(e) => (StudioConfig::default(), Some(format!("{e:#}"))),
        };
        let extra_args_json =
            serde_json::to_string(&config.image.sd_extra_args).unwrap_or_else(|_| "[]".into());
        let (tx, rx) = mpsc::channel();
        Self{data_dir,config,tab:Tab::Images,image:ImageRequest::default(),image_input:String::new(),workflow_json:String::new(),extra_args_json,models:vec![],document:DocumentRequest::default(),document_paths:String::new(),media_kind:0,media_input:String::new(),media_text:String::new(),language:"eng".into(),first_page:1,last_page:10,dpi:150,max_audio_seconds:3600,translate_audio:false,frame_interval:10.0,frame_count:12,speech_engine:SpeechEngine::Espeak,host_voice:"en-us".into(),guest_voice:"en-sc".into(),speech_rate:175,cutout_tolerance:24,cutout_feather:12,cutout_semantic:false,vision_question:"Describe the visible content and cite the supplied image label. State uncertainty and what this still image cannot establish.".into(),prepared:None,output:None,textures:vec![],markdown:CommonMarkCache::default(),attachment:None,status:"Choose an image, document or media activity".into(),error,busy:false,started:None,cancel:Arc::new(AtomicBool::new(false)),tx,rx,worker:None}
    }

    pub fn is_busy(&self) -> bool {
        self.busy
    }
    pub fn take_attachment(&mut self) -> Option<(String, String)> {
        self.attachment.take()
    }
    pub fn cancel_all(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        // Gives local process-group cleanup time to finish during normal GUI exit.
        let start = Instant::now();
        while self.worker.as_ref().is_some_and(|w| !w.is_finished())
            && start.elapsed() < Duration::from_millis(300)
        {
            thread::sleep(Duration::from_millis(10));
        }
        if self.worker.as_ref().is_some_and(|w| w.is_finished()) {
            if let Some(worker) = self.worker.take() {
                let _ = worker.join();
            }
        }
    }

    pub fn poll(&mut self, ctx: &egui::Context) {
        while let Ok(event) = self.rx.try_recv() {
            match event {
                StudioEvent::Status(status) => self.status = status,
                StudioEvent::Error(error) => self.error = Some(error),
                StudioEvent::Done => {
                    self.busy = false;
                    self.started = None;
                }
                StudioEvent::Prepared(prepared) => {
                    self.prepared = Some(*prepared);
                    self.status = "Review the activity below, then run it".into();
                }
                StudioEvent::Finished(output, thumbnails) => {
                    // Reports can be several MiB. Share the immutable result
                    // across a frame instead of copying it on every repaint.
                    self.output = Some(Arc::new(*output));
                    self.textures.clear();
                    for thumbnail in thumbnails {
                        let image = egui::ColorImage::from_rgba_unmultiplied(
                            thumbnail.size,
                            &thumbnail.rgba,
                        );
                        let texture = ctx.load_texture(
                            format!("studio:{}", thumbnail.path.display()),
                            image,
                            egui::TextureOptions::LINEAR,
                        );
                        self.textures.push((thumbnail.path, texture));
                    }
                    self.status = "Completed; outputs are saved locally".into();
                }
                StudioEvent::Models(models) => {
                    if self.config.image.checkpoint.is_empty() {
                        if let Some(first) = models.first() {
                            self.config.image.checkpoint = first.clone();
                        }
                    }
                    self.status = format!("Found {} image checkpoint(s)", models.len());
                    self.models = models;
                }
                StudioEvent::PickedInput(path, image) => {
                    if image {
                        self.image_input = path;
                    } else {
                        self.media_input = path;
                    }
                    self.status = "File selected".into();
                }
                StudioEvent::PickedModel(path) => {
                    self.config.image.sd_model = path.into();
                    self.status = "Diffusion model selected".into();
                }
                StudioEvent::PickedDocuments(paths) => {
                    let mut existing = self
                        .document_paths
                        .lines()
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(str::to_string)
                        .collect::<Vec<_>>();
                    for path in paths {
                        if !existing.contains(&path) {
                            existing.push(path);
                        }
                    }
                    self.document_paths = existing.join("\n");
                    self.status = "Documents selected".into();
                }
            }
        }
        if self.worker.as_ref().is_some_and(|w| w.is_finished()) {
            if let Some(worker) = self.worker.take() {
                let _ = worker.join();
            }
        }
        if self.busy {
            ctx.request_repaint_after(Duration::from_millis(60));
        }
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        settings: &Settings,
        workspace: Option<&Path>,
    ) {
        self.poll(ctx);
        ui.heading("Studio");
        ui.label("Create images, work through local documents, and process media with installed local engines.");
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.tab, Tab::Images, "Images");
            ui.selectable_value(&mut self.tab, Tab::Documents, "Documents");
            ui.selectable_value(&mut self.tab, Tab::Media, "Media & LaTeX");
            if self.busy {
                ui.spinner();
                if ui.button("Cancel activity").clicked() {
                    self.cancel.store(true, Ordering::Relaxed);
                    self.status = "Cancelling the activity…".into();
                }
            }
            if let Some(start) = self.started {
                ui.label(format!("{} s", start.elapsed().as_secs()));
            }
        });
        ui.label(
            RichText::new(&self.status)
                .small()
                .color(ui.visuals().hyperlink_color),
        );
        if let Some(error) = self.error.clone() {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.label(RichText::new(error).color(ui.visuals().warn_fg_color));
                if ui.small_button("Dismiss").clicked() {
                    self.error = None;
                }
            });
        }
        ui.separator();
        egui::ScrollArea::vertical()
            .id_salt("studio_content")
            .show(ui, |ui| {
                ui.add_enabled_ui(!self.busy, |ui| match self.tab {
                    Tab::Images => self.image_form(ui, workspace),
                    Tab::Documents => self.document_form(ui, settings, workspace),
                    Tab::Media => self.media_form(ui, workspace),
                });
                self.preview_panel(ui);
                self.results_panel(ui, ctx, settings);
                ui.add_space(16.0);
            });
    }
}

impl Drop for StudioUi {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}
fn engine_name(engine: ImageBackend) -> &'static str {
    match engine {
        ImageBackend::ComfyUi => "ComfyUI (local API)",
        ImageBackend::Automatic1111 => "AUTOMATIC1111 (local API)",
        ImageBackend::StableDiffusionCpp => "stable-diffusion.cpp (local executable)",
    }
}
fn text_row(ui: &mut egui::Ui, label: &str, value: &mut String) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::TextEdit::singleline(value).desired_width(ui.available_width().max(120.0)));
    });
}
fn path_row(ui: &mut egui::Ui, label: &str, path: &mut PathBuf) {
    let mut text = path.display().to_string();
    text_row(ui, label, &mut text);
    *path = text.into();
}
fn input_path(text: &str, workspace: Option<&Path>) -> Result<PathBuf> {
    let path = PathBuf::from(text.trim());
    if text.trim().is_empty() {
        bail!("Choose a local input file");
    }
    if path.is_absolute() {
        Ok(path)
    } else {
        Workspace::new(
            workspace
                .context("Enter an absolute path or select a workspace")?
                .into(),
        )?
        .resolve(text.trim(), false)
    }
}

fn thumbnails(paths: &[PathBuf]) -> Vec<Thumbnail> {
    paths
        .iter()
        .filter(|path| {
            path.extension().and_then(|e| e.to_str()).is_some_and(|e| {
                matches!(
                    e.to_ascii_lowercase().as_str(),
                    "png" | "jpg" | "jpeg" | "gif" | "webp"
                )
            })
        })
        .take(12)
        .filter_map(|path| {
            let image = crate::media::load_image(path)
                .ok()?
                .thumbnail(512, 512)
                .into_rgba8();
            Some(Thumbnail {
                path: path.clone(),
                size: [image.width() as usize, image.height() as usize],
                rgba: image.into_raw(),
            })
        })
        .collect()
}
fn open_local(path: PathBuf) {
    thread::spawn(move || {
        #[cfg(target_os = "linux")]
        let _ = std::process::Command::new("xdg-open")
            .arg(path)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    });
}

fn load_config(directory: &Path) -> Result<StudioConfig> {
    let path = directory.join("studio-settings.json");
    if !path.exists() {
        return Ok(StudioConfig::default());
    }
    let bytes = crate::media::read_bytes_bounded(&path, 1024 * 1024)?;
    serde_json::from_slice(&bytes)
        .context("Could not read studio-settings.json; the original file was preserved")
}
fn save_config(directory: &Path, config: &StudioConfig) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(config)?;
    if bytes.len() > 1024 * 1024 {
        bail!("Studio settings exceed 1 MiB");
    }
    // Reuse the directory validator before writing a sibling settings file.
    let validation = crate::media::create_run_dir(directory, "media")?;
    let _ = fs::remove_dir(validation);
    let path = directory.join("studio-settings.json");
    if fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink()) {
        bail!("Refusing to replace symlinked Studio settings");
    }
    crate::persistence::write_atomic(&path, &bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn studio_settings_roundtrip_and_invalid_original_is_preserved() {
        let dir = tempfile::tempdir().unwrap();
        let config = StudioConfig {
            image: ImageConfig {
                checkpoint: "local.safetensors".into(),
                ..Default::default()
            },
            ..Default::default()
        };
        save_config(dir.path(), &config).unwrap();
        assert_eq!(
            load_config(dir.path()).unwrap().image.checkpoint,
            "local.safetensors"
        );
        fs::write(dir.path().join("studio-settings.json"), "invalid JSON").unwrap();
        let studio = StudioUi::new(dir.path().into());
        assert!(studio.error.is_some());
        assert_eq!(
            fs::read_to_string(dir.path().join("studio-settings.json")).unwrap(),
            "invalid JSON"
        );
    }
    #[test]
    fn relative_media_inputs_stay_in_the_selected_workspace() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("audio.wav"), b"data").unwrap();
        assert_eq!(
            input_path("audio.wav", Some(dir.path())).unwrap(),
            dir.path().join("audio.wav")
        );
        assert!(input_path("../secret", Some(dir.path())).is_err());
        assert!(input_path("audio.wav", None).is_err());
    }
}
