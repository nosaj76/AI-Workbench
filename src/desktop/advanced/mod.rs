//! Native controls for independent agent jobs, complete review queues and local services.
mod agents;
mod jobs;
mod memory;
mod notes;
mod qa;
mod research;
mod review;
mod routing;
mod terminal;

use crate::{
    agents::{AgentEvent, AgentRequest},
    files::Workspace,
    memory::{MemoryCandidate, MemoryRecord, MemoryStore},
    review_queue::{self, ReviewCoverage, ReviewItem, ReviewQueue, ReviewRequest},
    routing::RoutingConfig,
    settings::Settings,
    terminal::{TerminalConfig, TerminalKey, TerminalManager},
    tools::PendingAction,
};
use anyhow::Result;
use eframe::egui::{self, Color32, RichText};
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use std::{
    collections::{BTreeMap, VecDeque},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc,
    },
    time::Duration,
};

#[derive(Clone, Copy, PartialEq)]
pub enum Section {
    Agents,
    Review,
    Research,
    Qa,
    Terminal,
    Memory,
    Notes,
    Routing,
    Studio,
    Services,
}
struct TaskView {
    title: String,
    state: String,
    text: String,
}
struct PendingApproval {
    task: String,
    action: Box<PendingAction>,
    reply: Sender<bool>,
}
enum UiEvent {
    Finished(Result<(String, Vec<PathBuf>), String>),
    Reviews {
        root: PathBuf,
        coverage: ReviewCoverage,
        items: Vec<ReviewItem>,
    },
}

pub struct AdvancedUi {
    dir: PathBuf,
    workspace_key: String,
    pub section: Section,
    sender: Sender<UiEvent>,
    receiver: Receiver<UiEvent>,
    agent_sender: Sender<AgentEvent>,
    agent_receiver: Receiver<AgentEvent>,
    busy: bool,
    preview_mode: bool,
    cancel: Arc<AtomicBool>,
    status: String,
    error: Option<String>,
    tasks: BTreeMap<String, TaskView>,
    approvals: VecDeque<PendingApproval>,
    report: String,
    files: Vec<PathBuf>,
    markdown: CommonMarkCache,
    attachment: Option<(String, String)>,
    objective: String,
    worker_models: String,
    allow_actions: bool,
    resume_id: String,
    self_review_count: usize,
    self_review_code: bool,
    history: Vec<crate::agents::AgentRun>,
    review_instruction: String,
    review_chunk: usize,
    review_batch: usize,
    review_coverage: Option<ReviewCoverage>,
    review_items: Vec<ReviewItem>,
    review_selected: String,
    review_skip_reason: String,
    research_question: String,
    qa_question: String,
    qa_refresh_days: u32,
    qa_sources: usize,
    qa_cache: Vec<crate::qa::CachedQa>,
    qa_facts: Vec<crate::qa::Fact>,
    qa_search: String,
    qa_view: qa::QaView,
    qa_refresh_list: bool,
    qa_allow_delete: bool,
    research_rounds: usize,
    research_sources: usize,
    research_web: bool,
    research_seed_urls: String,
    research_domains: String,
    research_resume_id: String,
    research_history: Vec<crate::research::ResearchRun>,
    memory_query: String,
    memory_rows: Vec<MemoryRecord>,
    note_query: String,
    note_rows: Vec<crate::notes::NoteSummary>,
    note_loaded: bool,
    note_open: Option<(String, String)>,
    note_review_pending: bool,
    memory_edit: Option<(String, String)>,
    memory_new_key: String,
    memory_new_fact: String,
    memory_global: bool,
    routing: RoutingConfig,
    routing_names: [String; 7],
    routing_loaded: bool,
    terminals: TerminalManager,
    /// Terminal processes belong to a workspace, not a temporary tool engine.
    /// This keeps them alive across turns/settings changes without exposing
    /// another workspace's sessions to its model.
    terminal_workspaces: BTreeMap<Option<PathBuf>, TerminalManager>,
    terminal_workspace: Option<PathBuf>,
    terminal_id: String,
    terminal_input: String,
    terminal_scrollback: usize,
    terminal_sizes: BTreeMap<String, (u16, u16)>,
    studio: crate::studio_ui::StudioUi,
    services: crate::services_ui::ServicesUi,
}

impl AdvancedUi {
    pub fn new(dir: PathBuf) -> Self {
        let (sender, receiver) = mpsc::channel();
        let (agent_sender, agent_receiver) = mpsc::channel();
        Self {
            studio: crate::studio_ui::StudioUi::new(dir.clone()),
            services: crate::services_ui::ServicesUi::new(dir.clone()),
            dir,
            workspace_key: String::new(),
            section: Section::Agents,
            sender,
            receiver,
            agent_sender,
            agent_receiver,
            busy: false,
            preview_mode: false,
            cancel: Arc::new(AtomicBool::new(false)),
            status: "Ready".into(),
            error: None,
            tasks: BTreeMap::new(),
            approvals: VecDeque::new(),
            report: String::new(),
            files: vec![],
            markdown: CommonMarkCache::default(),
            attachment: None,
            objective: String::new(),
            worker_models: String::new(),
            allow_actions: false,
            self_review_count: 3,
            self_review_code: crate::self_review::default_source_root().is_some(),
            resume_id: String::new(),
            history: vec![],
            review_instruction: "Review correctness, error handling, security boundaries, maintainability and missing behavior. Cite exact file and source-line ranges. Distinguish observed bugs from hypotheses. Do not claim tests were run.".into(),
            review_chunk: 6000,
            review_batch: 0,
            review_coverage: None,
            review_items: vec![],
            review_selected: String::new(),
            review_skip_reason: String::new(),
            research_question: String::new(),
            qa_question: String::new(),
            qa_refresh_days: crate::qa::DEFAULT_REFRESH_DAYS,
            qa_sources: 3,
            qa_cache: vec![],
            qa_facts: vec![],
            qa_search: String::new(),
            qa_view: qa::QaView::Answers,
            qa_refresh_list: true,
            qa_allow_delete: false,
            research_rounds: 3,
            research_sources: 12,
            research_web: false,
            research_seed_urls: String::new(),
            research_domains: String::new(),
            research_resume_id: String::new(),
            research_history: vec![],
            memory_query: String::new(),
            memory_rows: vec![],
            note_query: String::new(),
            note_rows: vec![],
            note_loaded: false,
            note_open: None,
            note_review_pending: false,
            memory_edit: None,
            memory_new_key: String::new(),
            memory_new_fact: String::new(),
            memory_global: false,
            routing: RoutingConfig::default(),
            routing_names: std::array::from_fn(|_| String::new()),
            routing_loaded: false,
            terminals: TerminalManager::default(),
            terminal_workspaces: BTreeMap::new(),
            terminal_workspace: None,
            terminal_id: String::new(),
            terminal_input: String::new(),
            terminal_scrollback: 0,
            terminal_sizes: BTreeMap::new(),
        }
    }
    pub fn set_preview_mode(&mut self, preview: bool) {
        self.preview_mode = preview;
    }
    pub fn open_schedules(&mut self) {
        self.section = Section::Services;
        self.services.show_schedules();
    }
    pub fn open_runtime(&mut self) {
        self.section = Section::Services;
        self.services.show_runtime();
    }
    pub fn invalidate_routing(&mut self) {
        self.routing_loaded = false;
    }
    /// Return the same manager used by the visible terminal page. Canonical
    /// paths make an engine and a symlink-selected UI workspace agree.
    pub fn terminal_manager_for(&mut self, workspace: Option<&Path>) -> TerminalManager {
        let key = workspace.map(|path| path.canonicalize().unwrap_or_else(|_| path.to_path_buf()));
        let manager = self
            .terminal_workspaces
            .entry(key.clone())
            .or_default()
            .clone();
        if self.terminal_workspace != key {
            self.terminal_workspace = key;
            self.terminal_id.clear();
            self.terminal_input.clear();
            self.terminal_scrollback = 0;
            self.terminal_sizes.clear();
        }
        self.terminals = manager.clone();
        manager
    }
    pub fn is_busy(&self) -> bool {
        self.busy || self.studio.is_busy() || self.services.is_busy()
    }
    pub fn take_attachment(&mut self) -> Option<(String, String)> {
        self.attachment
            .take()
            .or_else(|| self.studio.take_attachment())
    }
    pub fn take_endpoint_request(&mut self) -> Option<String> {
        self.services.take_endpoint_request()
    }
    pub fn take_session_request(&mut self) -> Option<String> {
        self.services.take_session_request()
    }
    pub fn cancel_all(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        for a in self.approvals.drain(..) {
            let _ = a.reply.send(false);
        }
        self.studio.cancel_all();
        self.services.cancel_all();
        for manager in self.terminal_workspaces.values_mut() {
            manager.close_all();
        }
        self.terminals.close_all();
    }
    pub fn poll(&mut self, ctx: &egui::Context) {
        for _ in 0..512 {
            let Ok(event) = self.agent_receiver.try_recv() else {
                break;
            };
            match event {
                AgentEvent::Status(text) => self.status = text,
                AgentEvent::Task {
                    id,
                    title,
                    state,
                    message,
                } => {
                    let t = self.tasks.entry(id).or_insert_with(|| TaskView {
                        title: title.clone(),
                        state: String::new(),
                        text: String::new(),
                    });
                    t.title = title;
                    t.state = state;
                    if !message.is_empty() {
                        t.text = message.chars().take(64_000).collect();
                    }
                }
                AgentEvent::Token { task_id, text } => {
                    let t = self
                        .tasks
                        .entry(task_id.clone())
                        .or_insert_with(|| TaskView {
                            title: task_id,
                            state: "running".into(),
                            text: String::new(),
                        });
                    if t.text.len() < 256_000 {
                        t.text.push_str(&text);
                    }
                }
                AgentEvent::Approval {
                    task_id,
                    action,
                    reply,
                } => {
                    if self.cancel.load(Ordering::Relaxed) {
                        let _ = reply.send(false);
                    } else {
                        self.approvals.push_back(PendingApproval {
                            task: task_id,
                            action,
                            reply,
                        });
                    }
                }
                AgentEvent::Finished { report } => self.report = report,
                AgentEvent::Error(error) => self.error = Some(error),
            }
        }
        while let Ok(event) = self.receiver.try_recv() {
            match event {
                UiEvent::Finished(result) => {
                    self.busy = false;
                    match result {
                        Ok((report, files)) => {
                            self.report = report;
                            self.files = files;
                            self.status = if self.cancel.load(Ordering::Relaxed) {
                                "Stopped; completed work was retained".into()
                            } else {
                                "Job finished; inspect its result and coverage".into()
                            };
                        }
                        Err(error) => {
                            self.error = Some(error);
                            self.status = "Job ended with an error".into();
                        }
                    }
                    for a in self.approvals.drain(..) {
                        let _ = a.reply.send(false);
                    }
                }
                UiEvent::Reviews {
                    root,
                    coverage,
                    items,
                } => {
                    if crate::memory::scope(Some(&root)) == self.workspace_key {
                        self.review_coverage = Some(coverage);
                        self.review_items = items;
                    }
                }
            }
        }
        self.studio.poll(ctx);
        self.services.poll(ctx);
        if self.is_busy() || !self.terminals.list().is_empty() {
            ctx.request_repaint_after(Duration::from_millis(80));
        }
    }

    pub fn dialogs(&mut self, ctx: &egui::Context) {
        let mut decision = None;
        if let Some(a) = self.approvals.front() {
            egui::Window::new("Agent action approval")
                .id(egui::Id::new("advanced_action_approval"))
                .collapsible(false)
                .resizable(true)
                .default_width(680.0)
                .show(ctx, |ui| {
                    ui.label(format!(
                        "Worker: {} · {} pending action(s)",
                        a.task,
                        self.approvals.len()
                    ));
                    ui.heading(&a.action.title);
                    ui.label(&a.action.description);
                    egui::ScrollArea::both().max_height(410.0).show(ui, |ui| {
                        crate::action_preview::show_action_details(ui, &a.action.details);
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Deny").clicked() {
                            decision = Some(false);
                        }
                        if ui.button("Approve this exact action").clicked() {
                            decision = Some(true);
                        }
                    });
                });
        }
        if let Some(allowed) = decision {
            if let Some(a) = self.approvals.pop_front() {
                let _ = a.reply.send(allowed);
            }
        }
    }
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        settings: &mut Settings,
        workspace: Option<&Path>,
    ) {
        let current = crate::memory::scope(workspace);
        if current != self.workspace_key {
            self.workspace_key = current;
            self.memory_rows.clear();
            self.review_coverage = None;
            self.review_items.clear();
            self.review_selected.clear();
        }
        ui.horizontal_wrapped(|ui| {
            for (section, label) in [
                (Section::Agents, "Agents"),
                (Section::Review, "Review queue"),
                (Section::Research, "Research"),
                (Section::Qa, "Q&A"),
                (Section::Terminal, "Terminal"),
                (Section::Memory, "Memory"),
                (Section::Notes, "Knowledge notes"),
                (Section::Routing, "Routing"),
                (Section::Studio, "Studio"),
                (Section::Services, "Services"),
            ] {
                if ui
                    .selectable_label(self.section == section, label)
                    .clicked()
                {
                    self.section = section;
                }
            }
        });
        ui.separator();
        if self.busy {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(&self.status);
                if ui.button("Stop current job").clicked() {
                    self.cancel.store(true, Ordering::Relaxed);
                    for a in self.approvals.drain(..) {
                        let _ = a.reply.send(false);
                    }
                }
            });
        }
        if let Some(error) = self.error.clone() {
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(ui.visuals().error_fg_color, error);
                if ui.small_button("Dismiss").clicked() {
                    self.error = None;
                }
            });
        }
        match self.section {
            Section::Terminal => self.terminal_page(ui, ctx, workspace),
            Section::Studio => {
                ui.add_enabled_ui(!self.preview_mode, |ui| {
                    self.studio.show(ui, ctx, settings, workspace)
                });
            }
            Section::Services => {
                ui.add_enabled_ui(!self.preview_mode, |ui| {
                    self.services.show(ui, ctx, settings, workspace)
                });
            }
            section => {
                egui::ScrollArea::vertical()
                    .id_salt(("advanced_page", section as u8))
                    .auto_shrink([false, false])
                    .show(ui, |ui| match section {
                        Section::Agents => self.agents_page(ui, settings, workspace),
                        Section::Review => self.review_page(ui, settings, workspace),
                        Section::Research => self.research_page(ui, settings, workspace),
                        Section::Qa => self.qa_page(ui, settings),
                        Section::Memory => self.memory_page(ui, settings, workspace),
                        Section::Notes => self.notes_page(ui, settings),
                        Section::Routing => self.routing_page(ui, settings),
                        _ => {}
                    });
            }
        }
    }
}
