//! Workbench events responsibilities.
use super::*;

impl Workbench {
    pub(super) fn work<F>(&mut self, label: &str, operation: F)
    where
        F: FnOnce(Sender<WorkEvent>, Arc<AtomicBool>) -> Result<()> + Send + 'static,
    {
        self.active_work += 1;
        self.work_busy = true;
        self.status = label.into();
        self.work_cancel = Arc::new(AtomicBool::new(false));
        let tx = self.work_tx.clone();
        let cancel = self.work_cancel.clone();
        self.work_cancels.push(cancel.clone());
        std::thread::spawn(move || {
            if let Err(e) = crate::jobs::guarded(|| operation(tx.clone(), cancel.clone())) {
                let _ = tx.send(WorkEvent::Error(format!("{e:#}")));
            }
            cancel.store(true, Ordering::Relaxed);
            let _ = tx.send(WorkEvent::Done);
        });
    }

    pub(super) fn refresh_models(&mut self) {
        self.connection_request_id = self.connection_request_id.wrapping_add(1);
        self.version.clear();
        self.models.clear();
        self.running_models = Value::Null;
        if let Err(error) = self.settings.validate_endpoint() {
            self.error = Some(error.to_string());
            return;
        }
        let endpoint = self.settings.base_url.clone();
        let request_id = self.connection_request_id;
        self.work("Connecting to Ollama...", move |tx, _| {
            let result = (|| {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?;
                rt.block_on(async {
                    let client = OllamaClient::new(&endpoint)?;
                    let models = client.list_models().await?;
                    let version = client.version().await?;
                    let running = client.running_models().await.unwrap_or(Value::Null);
                    Ok::<_, anyhow::Error>((models, version, running))
                })
            })();
            match result {
                Ok((models, version, running)) => tx.send(WorkEvent::Models {
                    request_id,
                    endpoint,
                    models,
                    version,
                    running,
                })?,
                Err(error) => tx.send(WorkEvent::ConnectionError {
                    request_id,
                    endpoint,
                    message: format!("{error:#}"),
                })?,
            }
            Ok(())
        });
    }

    /// Persist the actual failure even when it happened before the first token.
    /// Do not relabel a transport/configuration error as a user cancellation.
    pub(super) fn record_chat_interruption(&mut self, kind: &str, reason: &str) {
        let index = self.streaming.take().unwrap_or_else(|| {
            self.session.entries.push(Entry::new(ChatMessage {
                role: "assistant".into(),
                ..Default::default()
            }));
            self.session.entries.len() - 1
        });
        let entry = &mut self.session.entries[index];
        if !entry.stats.is_object() {
            entry.stats = json!({});
        }
        entry.stats["incomplete"] = json!(true);
        entry.stats["interruption_kind"] = json!(kind);
        entry.stats["failure_reason"] = json!(reason);
        if entry.message.content.trim().is_empty() {
            entry.message.content = reason.into();
        } else {
            entry.message.content.push_str(&format!("\n\n{reason}"));
        }
        self.dirty = true;
    }

    pub(super) fn receive_events(&mut self) {
        while let Ok(event) = self.chat_rx.try_recv() {
            match event {
                ChatEvent::Begin(model) => {
                    let mut entry = Entry::new(ChatMessage {
                        role: "assistant".into(),
                        ..Default::default()
                    });
                    entry.model = model;
                    self.session.entries.push(entry);
                    self.streaming = Some(self.session.entries.len() - 1);
                    self.status = "Generating locally…".into();
                    self.dirty = true;
                }
                ChatEvent::Content(s) => {
                    if let Some(i) = self.streaming {
                        self.session.entries[i].message.content.push_str(&s);
                        self.dirty = true;
                    }
                }
                ChatEvent::Thinking(s) => {
                    if let Some(i) = self.streaming {
                        self.session.entries[i].message.thinking.push_str(&s);
                        self.dirty = true;
                    }
                }
                ChatEvent::Stats(v) => {
                    if let Some(i) = self.streaming {
                        self.session.entries[i].stats = v;
                    }
                }
                ChatEvent::Complete(message) => {
                    if let Some(i) = self.streaming.take() {
                        let entry = &mut self.session.entries[i];
                        // Text written next to a tool call is a draft; show its start
                        // so the final answer is not seen to begin several times.
                        if !message.tool_calls.is_empty()
                            && message.content.chars().count() > crate::storage::TOOL_DRAFT_CHARS
                        {
                            let start: String = message
                                .content
                                .chars()
                                .take(crate::storage::TOOL_DRAFT_CHARS)
                                .collect();
                            entry.display = Some(format!(
                                "{start}…\n\n_(Draft written while still using tools; the full text is in Details. The answer follows after the tool results.)_"
                            ));
                        }
                        entry.message = message;
                    }
                    self.dirty = true;
                }
                ChatEvent::Superseded(reason) => {
                    // Keep the discarded draft for Details, show why it was
                    // dropped, and never send it back to the model.
                    if let Some(i) = self.streaming.take() {
                        let entry = &mut self.session.entries[i];
                        if !entry.stats.is_object() {
                            entry.stats = json!({});
                        }
                        entry.stats["superseded"] = json!(true);
                        entry.display = Some(reason);
                    }
                    self.dirty = true;
                }
                ChatEvent::UnexecutedToolText {
                    message,
                    names,
                    retrying,
                } => {
                    // The original unexecuted model text is saved for the
                    // inspector, never shown as a completed file-review result.
                    if let Some(i) = self.streaming.take() {
                        let entry = &mut self.session.entries[i];
                        entry.message = message;
                        if !entry.stats.is_object() {
                            entry.stats = json!({});
                        }
                        entry.stats["unexecuted_tool_text"] = json!(true);
                        entry.stats["unexecuted_tool_names"] = json!(&names);
                        let request_label = if names.len() == 1 {
                            "tool request"
                        } else {
                            "tool requests"
                        };
                        entry.display = Some(if retrying {
                            format!(
                                "The model printed {} {request_label} as text. That text was not executed. Retrying with native tool calls…",
                                names.len()
                            )
                        } else {
                            format!(
                                "The model printed {} {request_label} as text. That text was not executed; the request will end with an evidence-only explanation.",
                                names.len()
                            )
                        });
                        self.dirty = true;
                    }
                }
                ChatEvent::Append(entry) => {
                    self.session.entries.push(entry);
                    self.dirty = true;
                }
                ChatEvent::Status(s) => self.status = s,
                ChatEvent::BackgroundFailure { task, reason } => {
                    if let Some(entry) = self
                        .session
                        .entries
                        .iter_mut()
                        .rev()
                        .take_while(|e| e.message.role != "user")
                        .find(|e| e.message.role == "assistant")
                    {
                        if !entry.stats.is_object() {
                            entry.stats = serde_json::json!({});
                        }
                        let failures = entry
                            .stats
                            .as_object_mut()
                            .unwrap()
                            .entry("background_failures")
                            .or_insert_with(|| serde_json::json!([]));
                        if let Some(failures) = failures.as_array_mut() {
                            if failures.len() < 8 {
                                failures.push(serde_json::json!({"task": task, "reason": reason}));
                            }
                        }
                        self.dirty = true;
                    }
                }
                ChatEvent::ToolStarted { id, name, summary } => {
                    self.tool_console.started(id, name, summary);
                }
                ChatEvent::ToolOutput { id, chunk } => self.tool_console.output(&id, &chunk),
                ChatEvent::ToolFinished { id, ok, preview } => {
                    self.tool_console.finished(&id, ok, preview);
                }
                ChatEvent::Approval { action, reply } => {
                    // Only the user's own chat reaches this handler: sub-agents and
                    // scheduled jobs keep their own approval rules.
                    let automatic = match action.operation.approval_kind() {
                        crate::tools::ApprovalKind::Web => self.settings.auto_approve_web,
                        crate::tools::ApprovalKind::FileChange => {
                            self.settings.auto_approve_file_changes
                        }
                        crate::tools::ApprovalKind::Command => self.settings.auto_approve_commands,
                        crate::tools::ApprovalKind::UnsandboxedCommand => {
                            self.settings.auto_approve_unsandboxed
                        }
                        crate::tools::ApprovalKind::LocalNetwork => false,
                    };
                    if automatic {
                        let _ = self.store.audit(
                            &self.session.id,
                            "Auto-approved",
                            &format!("{}\n{}", action.title, action.details),
                        );
                        self.status = format!("Auto-approved: {}", action.title);
                        let _ = reply.send(true);
                        continue;
                    }
                    self.status = "Waiting for your decision".into();
                    self.approval = Some(Approval {
                        action,
                        origin: ApprovalOrigin::Chat(reply),
                    });
                }
                ChatEvent::Audit { action, details } => {
                    if let Err(e) = self.store.audit(&self.session.id, &action, &details) {
                        self.error = Some(format!("Could not save action log: {e}"));
                    }
                }
                ChatEvent::Evidence { context, sources } => {
                    if let Some(entry) = self
                        .session
                        .entries
                        .iter_mut()
                        .rev()
                        .find(|e| e.message.role == "user")
                    {
                        entry.context = context;
                        entry.sources.extend(sources);
                        self.dirty = true;
                    }
                }
                ChatEvent::Error(s) => {
                    self.record_chat_interruption("error", &format!("Request failed: {s}"));
                    self.error = Some(s);
                    self.compare_queue.clear();
                    self.compare_active = false;
                }
                ChatEvent::Finished => {
                    self.busy = false;
                    self.tool_console.chat_finished();
                    let unfinished_tail = self.session.entries.last().is_some_and(|entry| {
                        matches!(entry.message.role.as_str(), "user" | "tool")
                            || !entry.message.tool_calls.is_empty()
                            || entry
                                .stats
                                .get("unexecuted_tool_text")
                                .and_then(Value::as_bool)
                                == Some(true)
                    });
                    if self.streaming.is_some() || unfinished_tail {
                        if self.cancel.load(Ordering::Relaxed) {
                            self.record_chat_interruption(
                                "cancelled",
                                "Cancellation was requested before the answer completed. Any completed tool results above are retained; they are not a completed review.",
                            );
                        } else {
                            self.record_chat_interruption(
                                "unexpected_end",
                                "Inference ended without a completed answer. No cancellation was requested. Any tool results above are retained; inspect Details for the last model output.",
                            );
                        }
                    }
                    self.approval = None;
                    self.persist();
                    if !self.cancel.load(Ordering::Relaxed) && !self.compare_queue.is_empty() {
                        let model = self.compare_queue.remove(0);
                        self.launch_chat(self.compare_messages.clone(), model);
                    } else {
                        self.status = if self.cancel.load(Ordering::Relaxed) {
                            "Stopped".into()
                        } else if self.error.is_some() {
                            "Request failed".into()
                        } else {
                            "Ready".into()
                        };
                        self.compare_active = false;
                        // Send the next prompt typed while this reply was written.
                        // After Stop or a failure the queue waits for the user.
                        if !self.cancel.load(Ordering::Relaxed)
                            && self.error.is_none()
                            && !self.queued_prompts.is_empty()
                        {
                            self.send_next_queued();
                        } else if !self.queued_prompts.is_empty() {
                            self.status = format!(
                                "{} · {} queued prompt(s) kept; send them when ready",
                                self.status,
                                self.queued_prompts.len()
                            );
                        }
                    }
                }
            }
        }
        while let Ok(event) = self.work_rx.try_recv() {
            match event {
                WorkEvent::Models {
                    request_id,
                    endpoint,
                    models,
                    version,
                    running,
                } => {
                    if request_id != self.connection_request_id
                        || endpoint != self.settings.base_url
                    {
                        continue;
                    }
                    self.models = models;
                    self.version = version;
                    self.running_models = running;
                    if self.settings.model.is_empty() {
                        if let Some(model) = self
                            .models
                            .iter()
                            .find(|m| !m.name.contains("embed") && !m.name.contains("-cloud"))
                        {
                            self.settings.model = model.name.clone();
                        }
                    }
                    self.status = format!(
                        "Connected · Ollama {} · {} installed model(s)",
                        self.version,
                        self.models.len()
                    );
                }
                WorkEvent::ConnectionError {
                    request_id,
                    endpoint,
                    message,
                } => {
                    if request_id == self.connection_request_id
                        && endpoint == self.settings.base_url
                    {
                        self.version.clear();
                        self.models.clear();
                        self.running_models = Value::Null;
                        self.status = "Ollama is not connected".into();
                        self.error = Some(format!("Could not connect to {endpoint}: {message}"));
                    }
                }
                WorkEvent::PreviewError {
                    request_id,
                    root,
                    message,
                } => {
                    if request_id == self.preview_request_id
                        && self.session.project.as_deref() == Some(root.as_str())
                    {
                        self.error = Some(message);
                    }
                }
                WorkEvent::Knowledge { root, hits } => {
                    if self.session.project.as_deref() == Some(root.as_str()) {
                        self.knowledge_results = crate::retrieval::format_context(&hits, 60000);
                        self.knowledge_hits = hits;
                    }
                }
                WorkEvent::FileSearch { root, text } => {
                    if self.session.project.as_deref() == Some(root.as_str()) {
                        self.file_results = text;
                    }
                }
                WorkEvent::Skill(skill) => self.skill_preview = Some(skill),
                WorkEvent::ModelInfo(value) => {
                    self.model_details = serde_json::to_string_pretty(&value).unwrap_or_default()
                }
                WorkEvent::PrepareTool { name, args } => self.queued_tool = Some((name, args)),
                WorkEvent::Prepared { engine, prepared } => match prepared {
                    PreparedTool::Immediate(v) => {
                        self.tool_result = serde_json::to_string_pretty(&v).unwrap_or_default()
                    }
                    PreparedTool::Approval(action) => {
                        let action = *action;
                        self.approval = Some(Approval {
                            action,
                            origin: ApprovalOrigin::Manual(engine),
                        })
                    }
                },
                WorkEvent::PickedWorkspace(path) => self.set_workspace(path),
                WorkEvent::PickedFiles(paths) => self.attach_files(paths),
                WorkEvent::Tree { root, entries } => {
                    if self.session.project.as_deref() == Some(root.as_str()) {
                        self.tree = entries;
                    }
                }
                WorkEvent::Preview {
                    request_id,
                    root,
                    path,
                    text,
                    note,
                } => {
                    if request_id == self.preview_request_id
                        && self.session.project.as_deref() == Some(root.as_str())
                    {
                        self.preview_path = path;
                        self.preview_original = text.clone();
                        self.preview = text;
                        self.preview_note = note;
                    }
                }
                WorkEvent::Attached(items) => {
                    self.attachments.extend(items);
                    self.page = Page::Chat;
                }
                WorkEvent::Text(text) => {
                    self.tool_result = text;
                }
                WorkEvent::Status(s) => self.status = s,
                WorkEvent::Import(session) => {
                    self.persist();
                    self.input.clear();
                    self.queued_prompts.clear();
                    self.attachments.clear();
                    self.session = session;
                    self.workspace_input = self.session.project.clone().unwrap_or_default();
                    self.tree.clear();
                    self.preview.clear();
                    self.preview_path.clear();
                    self.manual_engine = None;
                    self.file_results.clear();
                    self.knowledge_results.clear();
                    self.knowledge_hits.clear();
                    self.persist();
                    self.refresh_tree();
                    self.page = Page::Chat;
                }
                WorkEvent::Error(s) => {
                    self.status = "Operation failed · see error details".into();
                    self.error = Some(s);
                }
                WorkEvent::Done => {
                    self.active_work = self.active_work.saturating_sub(1);
                    self.work_busy = self.active_work > 0;
                    self.work_cancels.retain(|c| !c.load(Ordering::Relaxed));
                }
            }
        }
        if !self.work_busy {
            if let Some((name, args)) = self.queued_tool.take() {
                self.manual_tool(&name, args);
            }
        }
    }
}
