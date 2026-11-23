//! Native controls for durable schedules and the user-managed Ollama runtime.
//! Potentially slow work is dispatched away from the egui thread.

mod jobs;
mod runtime;
mod schedules;

use crate::ollama_manager::{ManagerProgress, ManagerStatus, OllamaManager, ReleaseInfo};
use crate::scheduler::{
    self, DaemonStatus, RunRecord, ScheduleTiming, ScheduledJob, SchedulerStore, ServiceAction,
};
use crate::settings::Settings;
use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use eframe::egui::{self, RichText};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

#[derive(PartialEq)]
enum Tab {
    Schedules,
    Runtime,
}
struct Overview {
    jobs: Vec<ScheduledJob>,
    runs: Vec<RunRecord>,
    daemon: Option<DaemonStatus>,
    runtime: ManagerStatus,
    migrated: usize,
}
enum Event {
    Overview(Box<Overview>),
    Release(Box<ReleaseInfo>),
    Runtime(Box<ManagerStatus>),
    Progress(ManagerProgress),
    Text(String),
    Workspace(PathBuf),
    Error(String),
    Done,
}

pub struct ServicesUi {
    dir: PathBuf,
    tab: Tab,
    tx: mpsc::Sender<Event>,
    rx: mpsc::Receiver<Event>,
    busy: bool,
    cancel: Arc<AtomicBool>,
    status: String,
    error: Option<String>,
    jobs: Vec<ScheduledJob>,
    runs: Vec<RunRecord>,
    daemon: Option<DaemonStatus>,
    runtime: Option<ManagerStatus>,
    release: Option<ReleaseInfo>,
    progress: Option<ManagerProgress>,
    last_refresh: Instant,
    edit_id: Option<String>,
    name: String,
    prompt: String,
    model: String,
    workspace: String,
    timing_mode: usize,
    cron: String,
    cron_preview: Option<schedules::CronPreview>,
    timezone: String,
    interval: u64,
    once: String,
    enabled: bool,
    timeout: u64,
    version: String,
    port: u16,
    models: String,
    logs: String,
    show_unit: bool,
    endpoint_request: Option<String>,
    session_request: Option<String>,
}

impl ServicesUi {
    pub fn new(data_dir: PathBuf) -> Self {
        let (tx, rx) = mpsc::channel();
        let mut ui=Self{dir:data_dir,tab:Tab::Schedules,tx,rx,busy:false,cancel:Arc::new(AtomicBool::new(false)),status:"Loading local services".into(),error:None,jobs:vec![],runs:vec![],daemon:None,runtime:None,release:None,progress:None,last_refresh:Instant::now(),edit_id:None,name:"Project review".into(),prompt:"Review the selected workspace for concrete defects. Cite the files and lines actually inspected, identify incomplete coverage, and report findings without changing files.".into(),model:String::new(),workspace:String::new(),timing_mode:0,cron:"0 9 * * *".into(),cron_preview:None,timezone:system_timezone(),interval:60,once:(Utc::now()+chrono::Duration::hours(1)).to_rfc3339(),enabled:true,timeout:1800,version:String::new(),port:11434,models:String::new(),logs:String::new(),show_unit:false,endpoint_request:None,session_request:None};
        ui.refresh();
        ui
    }
    pub fn is_busy(&self) -> bool {
        self.busy
    }
    /// Cancels foreground downloads/jobs only. Explicitly started background
    /// services remain running when the desktop window is closed.
    pub fn cancel_all(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
    pub fn take_endpoint_request(&mut self) -> Option<String> {
        self.endpoint_request.take()
    }
    pub fn take_session_request(&mut self) -> Option<String> {
        self.session_request.take()
    }
    pub fn show_schedules(&mut self) {
        self.tab = Tab::Schedules;
    }
    pub fn show_runtime(&mut self) {
        self.tab = Tab::Runtime;
    }

    pub fn poll(&mut self, ctx: &egui::Context) {
        while let Ok(event) = self.rx.try_recv() {
            match event {
                Event::Overview(value) => {
                    let value = *value;
                    self.jobs = value.jobs;
                    self.runs = value.runs;
                    self.daemon = value.daemon;
                    self.runtime = Some(value.runtime);
                    if value.migrated > 0 {
                        self.status = format!(
                            "Migrated {} existing schedules; original JSON preserved",
                            value.migrated
                        );
                    } else if self.status == "Refreshing local service state" {
                        self.status = "Local service state is up to date".into();
                    }
                    self.last_refresh = Instant::now();
                }
                Event::Release(value) => {
                    self.status = format!("Official release {} is available", value.version);
                    self.release = Some(*value);
                }
                Event::Runtime(value) => self.runtime = Some(*value),
                Event::Progress(value) => {
                    self.status = value.stage.clone();
                    self.progress = Some(value);
                }
                Event::Text(text) => {
                    self.status = if text.len() > 240 || text.contains('\n') {
                        "Service output updated; details are shown below".into()
                    } else if text.trim().is_empty() {
                        "Operation completed with no additional output".into()
                    } else {
                        text.clone()
                    };
                    self.logs = text;
                }
                Event::Workspace(path) => self.workspace = path.display().to_string(),
                Event::Error(error) => {
                    self.error = Some(error);
                    self.status = "Operation did not complete".into();
                }
                Event::Done => {
                    self.busy = false;
                    self.progress = None;
                }
            }
            ctx.request_repaint();
        }
        if self.busy {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
        if !self.busy && self.last_refresh.elapsed() > Duration::from_secs(15) {
            self.refresh();
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
        if self.model.is_empty() {
            self.model = settings.model.clone();
        }
        if self.workspace.is_empty() {
            if let Some(path) = workspace {
                self.workspace = path.display().to_string();
            }
        }
        ui.heading("Local services");
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.tab, Tab::Schedules, "Schedules & background jobs");
            ui.selectable_value(&mut self.tab, Tab::Runtime, "Ollama installation & updates");
            if ui
                .add_enabled(!self.busy, egui::Button::new("Refresh"))
                .clicked()
            {
                self.refresh();
            }
            if self.busy {
                ui.spinner();
                if ui.button("Cancel foreground operation").clicked() {
                    self.cancel_all();
                }
            }
        });
        ui.label(&self.status);
        if let Some(progress) = &self.progress {
            if let Some(total) = progress.total.filter(|n| *n > 0) {
                ui.add(
                    egui::ProgressBar::new(progress.completed as f32 / total as f32).text(format!(
                        "{} / {}",
                        size(progress.completed),
                        size(total)
                    )),
                );
            }
        }
        if let Some(error) = self.error.clone() {
            ui.group(|ui| {
                ui.colored_label(ui.visuals().error_fg_color, error);
                if ui.small_button("Dismiss").clicked() {
                    self.error = None;
                }
            });
        }
        ui.separator();
        egui::ScrollArea::vertical()
            .id_salt("local_services_content")
            .show(ui, |ui| match self.tab {
                Tab::Schedules => self.schedules_page(ui, workspace),
                Tab::Runtime => self.runtime_page(ui, ctx),
            });
    }
}

fn load_overview(dir: &Path) -> Result<Overview> {
    let mut store = SchedulerStore::open(dir)?;
    let migrated = store.migrate_legacy(dir)?;
    Ok(Overview {
        jobs: store.list()?,
        runs: store.history(20)?,
        daemon: scheduler::daemon_status(dir)?,
        runtime: OllamaManager::new(dir)?.status()?,
        migrated,
    })
}
fn display_time(time: DateTime<Utc>, timezone: &str) -> String {
    timezone
        .parse::<chrono_tz::Tz>()
        .map(|tz| {
            time.with_timezone(&tz)
                .format("%Y-%m-%d %H:%M %Z")
                .to_string()
        })
        .unwrap_or_else(|_| time.to_rfc3339())
}
fn system_timezone() -> String {
    for candidate in [
        std::env::var("TZ").ok(),
        std::fs::read_to_string("/etc/timezone").ok(),
        std::fs::read_link("/etc/localtime").ok().and_then(|p| {
            p.to_string_lossy()
                .split("zoneinfo/")
                .nth(1)
                .map(String::from)
        }),
    ]
    .into_iter()
    .flatten()
    {
        let candidate = candidate.trim();
        if candidate.parse::<chrono_tz::Tz>().is_ok() {
            return candidate.into();
        }
    }
    "UTC".into()
}
fn size(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{bytes} bytes")
    }
}
