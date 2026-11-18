//! Workbench workspace responsibilities.
use super::*;

impl Workbench {
    pub(super) fn set_workspace(&mut self, path: PathBuf) {
        if let Err(error) = self.set_workspace_checked(path) {
            self.error = Some(format!("Workspace: {error:#}"));
        }
    }

    fn set_workspace_checked(&mut self, path: PathBuf) -> Result<()> {
        let workspace = Workspace::new(path)?;
        let root = workspace.root().display().to_string();
        self.preview_request_id = self.preview_request_id.wrapping_add(1);
        self.preview_original.clear();
        self.preview_note.clear();
        self.session.project = Some(root.clone());
        self.workspace_input = root;
        self.manual_engine = None;
        self.tree.clear();
        self.preview.clear();
        self.preview_path.clear();
        self.file_results.clear();
        self.knowledge_results.clear();
        self.knowledge_hits.clear();
        self.checkpoints = None;
        self.persist();
        self.refresh_tree();
        Ok(())
    }

    /// An explicit CLI action can start without a manually chosen project.
    /// Select a dedicated working folder before constructing the tool engine,
    /// so the AI and terminal panel use the same persistent session registry.
    /// Ordinary chat, disabled tools, and model comparisons do not gain access.
    pub(super) fn prepare_chat_workspace(&mut self, messages: &[ChatMessage]) -> Result<bool> {
        let library = std::env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join("ModernBASIC/programs"));
        self.prepare_chat_workspace_with_library(messages, library)
    }

    pub(super) fn prepare_chat_workspace_with_library(
        &mut self,
        messages: &[ChatMessage],
        library: Option<PathBuf>,
    ) -> Result<bool> {
        if !self.settings.tools_enabled || self.compare_active {
            return Ok(false);
        }
        if !runtime::requests_terminal_action_for_turn(messages) {
            return Ok(false);
        }
        if runtime::requests_modernbasic_for_turn(messages) && !runtime::modernbasic_installed() {
            return Ok(false);
        }

        let fallback = self.dir.join("workspaces/terminal");
        if runtime::requests_modernbasic_for_turn(messages) {
            if let Some(library) = library.filter(|p| p.is_dir()) {
                if self.session.project.is_none()
                    || self.session.project.as_deref().map(Path::new) == Some(fallback.as_path())
                {
                    self.set_workspace_checked(library)?;
                    return Ok(true);
                }
            }
        }
        if self.session.project.is_some() {
            return Ok(false);
        }
        let data_root = self
            .dir
            .canonicalize()
            .context("Locate the Workbench directory for terminal work")?;
        let workspaces = data_root.join("workspaces");
        let root = workspaces.join("terminal");
        for path in [&workspaces, &root] {
            match std::fs::symlink_metadata(path) {
                Ok(metadata) => {
                    if metadata.file_type().is_symlink() || !metadata.is_dir() {
                        bail!(
                            "Default terminal workspace needs a regular directory at {}. Choose a working folder with Open folder…",
                            path.display()
                        );
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    std::fs::create_dir(path).with_context(|| {
                        format!("Create default terminal workspace at {}", path.display())
                    })?;
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("Inspect default terminal workspace at {}", path.display())
                    });
                }
            }
        }
        let workspace =
            Workspace::new(root.clone()).context("Validate the default terminal workspace")?;
        if workspace.root() != root {
            bail!("The default terminal workspace location changed. Choose a working folder with Open folder…");
        }
        self.set_workspace_checked(root)?;
        Ok(true)
    }

    /// Drops everything the UI derived from the connected folder.
    pub(super) fn clear_workspace_view(&mut self) {
        self.preview_request_id = self.preview_request_id.wrapping_add(1);
        self.preview_original.clear();
        self.preview_note.clear();
        self.workspace_input.clear();
        self.manual_engine = None;
        self.tree.clear();
        self.preview.clear();
        self.preview_path.clear();
        self.file_results.clear();
        self.knowledge_results.clear();
        self.knowledge_hits.clear();
        self.checkpoints = None;
    }

    /// Detaches the project from this conversation only; other conversations keep theirs.
    pub(super) fn disconnect_workspace(&mut self) {
        self.session.project = None;
        self.clear_workspace_view();
        self.persist();
    }

    /// Reloads the list of saved versions shown on the Local files page.
    pub(super) fn refresh_checkpoints(&mut self) {
        let listed = self.get_engine().and_then(|engine| engine.checkpoints());
        match listed {
            Ok(list) => self.checkpoints = Some(list),
            Err(error) => {
                self.checkpoints = Some(Default::default());
                self.error = Some(format!("Saved versions: {error:#}"));
            }
        }
    }

    /// Puts a saved version of a file back and refreshes what depends on it.
    pub(super) fn restore_saved_version(&mut self, id: &str) {
        match self
            .get_engine()
            .and_then(|engine| engine.restore_checkpoint(id))
        {
            Ok(result) => {
                let path = result["path"].as_str().unwrap_or("the file").to_string();
                self.status = format!("Restored {path}. The version it replaced is the newest entry in Saved versions.");
                if self.preview_path == path {
                    self.read_file(path);
                }
                self.refresh_tree();
            }
            Err(error) => self.error = Some(format!("Restore failed: {error:#}")),
        }
        self.refresh_checkpoints();
    }

    pub(super) fn workspace(&self) -> Result<Workspace> {
        Workspace::new(PathBuf::from(
            self.session
                .project
                .as_deref()
                .context("Choose a workspace folder first")?,
        ))
    }

    pub(super) fn refresh_tree(&mut self) {
        let Some(root) = self.session.project.clone() else {
            return;
        };
        let tx = self.work_tx.clone();
        std::thread::spawn(move || {
            let result = (|| {
                let workspace = Workspace::new(PathBuf::from(&root))?;
                let entries = workspace.tree(5000)?;
                tx.send(WorkEvent::Tree { root, entries })?;
                Ok::<_, anyhow::Error>(())
            })();
            if let Err(e) = result {
                let _ = tx.send(WorkEvent::Error(format!("Read workspace: {e:#}")));
            }
        });
    }

    pub(super) fn browse_workspace(&mut self) {
        self.work("Choose a workspace folder", |tx, _| {
            if let Some(path) = rfd::FileDialog::new()
                .set_title("Open local workspace")
                .pick_folder()
            {
                tx.send(WorkEvent::PickedWorkspace(path))?;
            }
            Ok(())
        });
    }

    pub(super) fn browse_files(&mut self) {
        self.work("Choose files to attach", |tx, _| {
            if let Some(paths) = rfd::FileDialog::new()
                .set_title("Attach local files or images")
                .pick_files()
            {
                tx.send(WorkEvent::PickedFiles(paths))?;
            }
            Ok(())
        });
    }

    pub(super) fn attach_files(&mut self, paths: Vec<PathBuf>) {
        let cap = self.settings.max_context_chars;
        self.work("Reading selected files…", move |tx, _| {
            if paths.len() > 16 {
                bail!("Attach at most 16 files at a time");
            }
            let mut attachments = vec![];
            let mut image_bytes = 0usize;
            for path in paths {
                if path.is_dir() {
                    tx.send(WorkEvent::PickedWorkspace(path))?;
                    continue;
                }
                let name = path
                    .file_name()
                    .context("Invalid filename")?
                    .to_string_lossy()
                    .to_string();
                let workspace =
                    Workspace::new(path.parent().context("File has no parent")?.to_path_buf())?;
                let ext = path
                    .extension()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_lowercase();
                let absolute = workspace.resolve(&name, false)?;
                if matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "webp" | "gif") {
                    if std::fs::metadata(&absolute)?.len() > 16 * 1024 * 1024 {
                        bail!("Image {name} exceeds 16 MiB");
                    }
                    let bytes = std::fs::read(&absolute)?;
                    image_bytes = image_bytes.saturating_add(bytes.len());
                    if image_bytes > 32 * 1024 * 1024 {
                        bail!("Image attachments exceed 32 MiB in total");
                    }
                    let dimensions = image::ImageReader::new(std::io::Cursor::new(&bytes))
                        .with_guessed_format()?
                        .into_dimensions()
                        .context("Unsupported or damaged image")?;
                    if dimensions.0 as u64 * dimensions.1 as u64 > 32_000_000 {
                        bail!("Image {name} exceeds 32 megapixels");
                    }
                    attachments.push(Attachment {
                        name,
                        source: absolute.display().to_string(),
                        text: String::new(),
                        image: Some(base64::engine::general_purpose::STANDARD.encode(bytes)),
                    });
                } else {
                    let document = workspace.read_text(&name, cap)?;
                    let numbered = document
                        .text
                        .lines()
                        .enumerate()
                        .map(|(i, line)| format!("{:>5} | {line}", i + 1))
                        .collect::<Vec<_>>()
                        .join("\n");
                    let source = format!(
                        "{} · {}{}",
                        absolute.display(),
                        document.kind,
                        if document.truncated {
                            " · TRUNCATED"
                        } else {
                            ""
                        }
                    );
                    attachments.push(Attachment {
                        name,
                        source,
                        text: numbered,
                        image: None,
                    });
                }
            }
            tx.send(WorkEvent::Attached(attachments))?;
            Ok(())
        });
    }

    pub(super) fn read_file(&mut self, path: String) {
        let Some(root) = self.session.project.clone() else {
            return;
        };
        self.preview_request_id = self.preview_request_id.wrapping_add(1);
        let request_id = self.preview_request_id;
        self.work("Reading file...", move |tx, _| {
            let result = (|| {
                let workspace = Workspace::new(PathBuf::from(&root))?;
                workspace.read_text(&path, 200_000)
            })();
            match result {
                Ok(document) => {
                    let note = format!(
                        "{} - {} characters{}",
                        document.kind,
                        document.text.chars().count(),
                        if document.truncated {
                            " - TRUNCATED preview"
                        } else {
                            ""
                        }
                    );
                    tx.send(WorkEvent::Preview {
                        source_text: document.source_text,
                        truncated: document.truncated,
                        request_id,
                        root,
                        path,
                        text: document.text,
                        note,
                    })?;
                }
                Err(error) => {
                    tx.send(WorkEvent::PreviewError {
                        request_id,
                        root,
                        message: format!("Read {path}: {error:#}"),
                    })?;
                }
            }
            Ok(())
        });
    }

    pub(super) fn can_save_preview_edit(&self) -> bool {
        self.preview_source_text && !self.preview_truncated && self.preview != self.preview_original
    }

    pub(super) fn review_preview_edit(&mut self) {
        if !self.can_save_preview_edit() {
            return;
        }
        match self.workspace().and_then(|ws| ws.read_text(&self.preview_path, 200_000)) {
            Ok(current) if current.source_text && !current.truncated && current.text == self.preview_original => {
                self.manual_tool("write_file", json!({"path": self.preview_path, "content": self.preview}));
            }
            Ok(_) => self.error = Some("The file changed since you opened it. Reload before saving this edit.".into()),
            Err(error) => self.error = Some(error.to_string()),
        }
    }

    pub(super) fn files_page(&mut self, ui: &mut egui::Ui) {
        ui.heading("Local files");
        ui.label("Choose the folder the assistant may inspect. File references stay attached to the conversation.");
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.workspace_input)
                    .hint_text(example_project_path())
                    .desired_width((ui.available_width() - 295.0).max(160.0)),
            );
            if ui
                .add_enabled(
                    !self.busy && !self.work_busy && self.approval.is_none(),
                    egui::Button::new("Open path"),
                )
                .clicked()
            {
                self.set_workspace(PathBuf::from(&self.workspace_input));
            }
            if ui
                .add_enabled(
                    !self.busy && !self.work_busy && self.approval.is_none(),
                    egui::Button::new("Browse…"),
                )
                .clicked()
            {
                self.browse_workspace();
            }
            if ui.small_button("Refresh").clicked() {
                self.refresh_tree();
            }
            if ui
                .add_enabled(
                    self.session.project.is_some()
                        && !self.busy
                        && !self.work_busy
                        && self.approval.is_none(),
                    egui::Button::new("Disconnect"),
                )
                .on_hover_text("Detach the project from this conversation only")
                .clicked()
            {
                self.disconnect_workspace();
            }
        });
        ui.separator();
        if self.session.project.is_none() {
            ui.add_space(35.0);
            ui.label("Open a local project or document folder to begin.");
            return;
        }
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.file_query)
                    .hint_text("Search inside workspace files…")
                    .desired_width(350.0),
            );
            if ui
                .add_enabled(!self.work_busy, egui::Button::new("Search contents"))
                .clicked()
            {
                let query = self.file_query.clone();
                let root = self.session.project.clone().unwrap();
                self.work("Searching local files…", move |tx, _| {
                    let ws = Workspace::new(PathBuf::from(&root))?;
                    let hits = ws.search(&query, 100)?;
                    let text = hits
                        .iter()
                        .map(|h| format!("{}:{}\n{}", h.path, h.line, h.text))
                        .collect::<Vec<_>>()
                        .join("\n\n");
                    tx.send(WorkEvent::FileSearch {
                        root,
                        text: if text.is_empty() {
                            "No matches".into()
                        } else {
                            text
                        },
                    })?;
                    Ok(())
                });
            }
        });
        if !self.file_results.is_empty() {
            egui::CollapsingHeader::new("Search results")
                .default_open(true)
                .show(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("file_search_results")
                        .max_height(180.0)
                        .show(ui, |ui| {
                            ui.add(
                                egui::Label::new(
                                    RichText::new(&self.file_results).monospace().size(12.0),
                                )
                                .selectable(true),
                            );
                        });
                });
        }
        let mut restore: Option<String> = None;
        let mut refresh_versions = self.checkpoints.is_none();
        egui::CollapsingHeader::new("Saved versions · undo changes")
            .default_open(false)
            .show(ui, |ui| {
                ui.label("Before every approved edit, move or delete, Workbench saves the file as it was. Restore puts a saved version back. The version it replaces is saved too, so a restore can itself be undone.");
                if ui.small_button("Refresh list").clicked() {
                    refresh_versions = true;
                }
                match &self.checkpoints {
                    None => {}
                    Some(list) if list.checkpoints.is_empty() => {
                        ui.label("Nothing has been changed in this project yet.");
                    }
                    Some(list) => {
                        egui::ScrollArea::vertical()
                            .id_salt("saved_versions")
                            .max_height(180.0)
                            .show(ui, |ui| {
                                for item in &list.checkpoints {
                                    ui.horizontal(|ui| {
                                        ui.monospace(&item.relative_path);
                                        ui.small(format!(
                                            "saved {} · {}",
                                            item.saved_at,
                                            match item.state {
                                                crate::tools::CheckpointState::SameAsSaved => "file is the same as this version",
                                                crate::tools::CheckpointState::ChangedSince => "file changed since",
                                                crate::tools::CheckpointState::Missing => "file is gone",
                                            }
                                        ));
                                        let restorable = item.state != crate::tools::CheckpointState::SameAsSaved
                                            && !self.busy
                                            && !self.work_busy
                                            && self.approval.is_none();
                                        if ui
                                            .add_enabled(restorable, egui::Button::new("Restore").small())
                                            .clicked()
                                        {
                                            restore = Some(item.id.clone());
                                        }
                                    });
                                }
                                if list.older_not_listed > 0 {
                                    ui.small(format!("{} older saved versions are not listed; they remain in the backup folder.", list.older_not_listed));
                                }
                            });
                    }
                }
            });
        if let Some(id) = restore {
            self.restore_saved_version(&id);
        } else if refresh_versions {
            self.refresh_checkpoints();
        }
        ui.columns(2,|columns|{
            columns[0].horizontal(|ui|{ui.label(RichText::new(format!("{} entries",self.tree.len())).small());ui.add(egui::TextEdit::singleline(&mut self.file_filter).hint_text("Filter paths…"));});
            let filter=self.file_filter.to_lowercase();let mut selected=None;
            egui::ScrollArea::vertical().id_salt("file_tree").max_height((columns[0].available_height()-100.0).max(150.0)).show(&mut columns[0],|ui|{
                for entry in &self.tree{if !filter.is_empty()&&!entry.path.to_lowercase().contains(&filter){continue;}let icon=if entry.is_dir{"▸"}else{"·"};let label=format!("{icon} {}",entry.path);if ui.add_enabled(!entry.is_dir&&!self.work_busy,egui::Button::new(RichText::new(label).monospace().size(12.0)).selected(self.preview_path==entry.path).wrap()).on_hover_text(format!("{} bytes",entry.size)).clicked(){selected=Some(entry.path.clone());}}
            });
            if let Some(path)=selected{self.read_file(path);}
            columns[0].separator();columns[0].label(RichText::new("Ignored build folders, secrets, binaries, and paths outside this workspace are excluded.").small().color(columns[0].visuals().weak_text_color()));
            let ui=&mut columns[1];
            if self.preview_path.is_empty(){ui.label("Select a file to preview it.");}else{
                ui.label(RichText::new(&self.preview_path).strong());ui.label(RichText::new(&self.preview_note).small());if !self.preview_source_text{ui.label(RichText::new("Extracted document text: attach for review; editing cannot overwrite this document format.").small().color(ui.visuals().warn_fg_color));}
                ui.horizontal_wrapped(|ui|{if ui.add_enabled(!self.work_busy,egui::Button::new("Attach to chat")).clicked(){if let Ok(ws)=self.workspace(){if let Ok(path)=ws.resolve(&self.preview_path,false){self.attach_files(vec![path]);}}}
if ui.button("Review file").clicked(){if let Ok(ws)=self.workspace(){if let Ok(path)=ws.resolve(&self.preview_path,false){self.input=format!("Review {} thoroughly. Explain its purpose, find concrete problems, and cite source lines.",self.preview_path);self.attach_files(vec![path]);}}}
                    if ui.add_enabled(self.can_save_preview_edit(), egui::Button::new("Review & save edit")).clicked() {
                        self.review_preview_edit();
                    }
                });
                if !self.preview_source_text && ui.button("Export preview as text…").clicked() {
                    let destination = format!("{}.txt", self.preview_path);
                    self.manual_tool("write_file", json!({"path":destination,"content":self.preview}));
                }
                egui::ScrollArea::both().id_salt("file_preview").max_height((ui.available_height()-110.0).max(140.0)).show(ui,|ui|{ui.add(egui::TextEdit::multiline(&mut self.preview).code_editor().desired_width(f32::INFINITY).desired_rows(24));});
                ui.horizontal(|ui|{if ui.button("Discard editor changes").clicked(){self.preview=self.preview_original.clone();}
if ui.button("Move to recovery…").clicked(){self.manual_tool("delete_file",json!({"path":self.preview_path}));}});
            }
            ui.separator();ui.horizontal(|ui|{ui.add(egui::TextEdit::singleline(&mut self.new_path).hint_text("new/file.rs").desired_width(170.0));if ui.button("Create file…").clicked(){self.manual_tool("write_file",json!({"path":self.new_path,"content":""}));}
if ui.button("New folder…").clicked(){self.manual_tool("make_directory",json!({"path":self.new_path}));}});
        });
    }
}

/// Placeholder for the workspace path field, based on the current user's
/// home directory at runtime rather than any fixed account name.
fn example_project_path() -> String {
    directories::BaseDirs::new()
        .map(|dirs| dirs.home_dir().join("projects").join("your-project"))
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_else(|| "/path/to/your-project".into())
}
