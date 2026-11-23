//! Workbench conversations responsibilities.
use super::*;

impl Workbench {
    pub(super) fn save_preferences(&mut self) {
        if self.demo {
            return;
        }
        if self.settings_corrupt {
            self.error = Some("The existing settings file is unreadable and has not been overwritten. Review and explicitly save Settings to replace it.".into());
            return;
        }
        if let Err(error) = self.settings.save(&self.dir) {
            self.error = Some(format!("Save preferences: {error:#}"));
        }
    }

    pub(super) fn new_conversation(&mut self) {
        if self.busy || self.work_busy || self.approval.is_some() {
            return;
        }
        if !self.persist() {
            return;
        }
        // A new conversation starts without a project; connect one explicitly.
        self.session = Session::default();
        self.clear_workspace_view();
        self.input.clear();
        self.queued_prompts.clear();
        self.attachments.clear();
        self.selected_entry_id = None;
        self.code_export_preview = None;
        self.visible_message_limit = 80;
        self.show_trash = false;
        self.page = Page::Chat;
        self.persist();
    }

    pub(super) fn open_conversation(&mut self, id: &str) {
        if self.busy || self.work_busy || self.approval.is_some() {
            return;
        }
        if !self.persist() {
            return;
        }
        match self.store.load(id) {
            Ok(session) => {
                if session.deleted_at.is_some() {
                    self.error =
                        Some("This conversation is in Trash. Restore it before opening.".into());
                    return;
                }
                self.session = session;
                self.workspace_input = self.session.project.clone().unwrap_or_default();
                self.tree.clear();
                self.preview.clear();
                self.preview_original.clear();
                self.preview_path.clear();
                self.preview_note.clear();
                self.manual_engine = None;
                self.file_results.clear();
                self.knowledge_results.clear();
                self.knowledge_hits.clear();
                self.input.clear();
                self.queued_prompts.clear();
                self.attachments.clear();
                self.selected_entry_id = None;
                self.code_export_preview = None;
                self.visible_message_limit = 80;
                self.preview_request_id = self.preview_request_id.wrapping_add(1);
                self.refresh_tree();
                self.page = Page::Chat;
                self.persist();
            }
            Err(error) => self.error = Some(format!("Open conversation: {error:#}")),
        }
    }

    /// A caller replacing this session must keep it in memory if saving fails.
    pub(super) fn persist(&mut self) -> bool {
        if self.demo {
            return true;
        }
        if let Err(e) = self.store.save(&mut self.session) {
            if !e.is::<storage::SessionConflict>() {
                self.error = Some(format!("Save conversation: {e:#}"));
                self.dirty = true;
                self.last_save = Instant::now();
                return false;
            }
            let mut copy = self.session.clone();
            copy.id = uuid::Uuid::new_v4().to_string();
            if !copy.title.ends_with(" · recovered edits") {
                copy.title = format!("{} · recovered edits", copy.title);
            }
            if let Err(error) = self.store.save(&mut copy) {
                self.error = Some(format!("Save recovered conversation: {error:#}"));
                self.dirty = true;
                self.last_save = Instant::now();
                return false;
            }
            self.session = copy;
            self.error = Some("Another window changed this conversation. Your edits were saved in a separate conversation.".into());
        }
        self.settings.last_session = Some(self.session.id.clone());
        if let Err(e) =
            crate::persistence::write_json(&self.dir.join("last-session.json"), &self.session.id)
        {
            self.error = Some(format!("Save last-opened conversation: {e:#}"));
        }
        self.refresh_conversation_list();
        self.dirty = false;
        self.last_save = Instant::now();
        true
    }

    pub(super) fn refresh_conversation_list(&mut self) {
        let result = if self.show_trash {
            self.store.list_trash(&self.session_search)
        } else {
            self.store.list(&self.session_search, self.include_archived)
        };
        match result {
            Ok(summaries) => self.summaries = summaries,
            Err(error) => self.error = Some(format!("Load conversations: {error:#}")),
        }
    }

    pub(super) fn conversation_action(&mut self, action: ConversationAction) {
        if self.busy || self.work_busy || self.approval.is_some() {
            return;
        }
        match action {
            ConversationAction::Open(id) => self.open_conversation(&id),
            ConversationAction::Trash(id) => self.pending_trash_id = Some(id),
            ConversationAction::DeleteForever(id) => self.pending_permanent_delete_id = Some(id),
            ConversationAction::Restore(id) => match self.store.restore(&id) {
                Ok(()) => {
                    self.show_trash = false;
                    self.refresh_conversation_list();
                    self.open_conversation(&id);
                    self.status = "Conversation restored from Trash".into();
                }
                Err(error) => self.error = Some(format!("Restore conversation: {error:#}")),
            },
            ConversationAction::TogglePin(id) => self.toggle_conversation_flag(&id, true),
            ConversationAction::ToggleArchive(id) => self.toggle_conversation_flag(&id, false),
        }
    }

    fn toggle_conversation_flag(&mut self, id: &str, pin: bool) {
        if id == self.session.id && !self.persist() {
            return;
        }
        match self.store.load(id) {
            Ok(mut session) => {
                if session.deleted_at.is_some() {
                    self.error = Some("Restore the conversation before editing it".into());
                    return;
                }
                if pin {
                    session.pinned = !session.pinned;
                } else {
                    session.archived = !session.archived;
                }
                match self.store.save(&mut session) {
                    Ok(()) => {
                        if id == self.session.id {
                            self.session = session;
                        }
                        self.refresh_conversation_list();
                    }
                    Err(error) => self.error = Some(format!("Update conversation: {error:#}")),
                }
            }
            Err(error) => self.error = Some(format!("Load conversation: {error:#}")),
        }
    }

    pub(super) fn confirm_move_to_trash(&mut self, id: &str) {
        if self.busy || self.work_busy || self.approval.is_some() {
            return;
        }
        if id == self.session.id && !self.persist() {
            return;
        }
        match self.store.move_to_trash(id) {
            Ok(()) => {
                if id == self.session.id {
                    self.session = Session::default();
                    self.clear_workspace_view();
                    self.input.clear();
                    self.queued_prompts.clear();
                    self.attachments.clear();
                    self.selected_entry_id = None;
                    self.code_export_preview = None;
                    self.page = Page::Chat;
                    self.persist();
                } else {
                    self.refresh_conversation_list();
                }
                self.status =
                    "Conversation moved to Trash. Restore it from the sidebar if needed.".into();
            }
            Err(error) => self.error = Some(format!("Move to Trash: {error:#}")),
        }
    }

    pub(super) fn confirm_delete_forever(&mut self, id: &str) {
        if self.busy || self.work_busy || self.approval.is_some() {
            return;
        }
        match self.store.delete_trashed_permanently(id) {
            Ok(()) => {
                self.refresh_conversation_list();
                self.status = "Conversation permanently deleted".into();
            }
            Err(error) => self.error = Some(format!("Delete conversation: {error:#}")),
        }
    }

    /// After the first answer, ask the model that wrote it for a short title.
    /// Only a title still equal to the automatic one (the start of the first
    /// message) is replaced, so a rename always wins.
    pub(super) fn generate_title(&mut self) {
        if self.demo || !self.settings.auto_titles {
            return;
        }
        let mut users = self.session.entries.iter().filter(|e| e.message.role == "user");
        let (Some(first), None) = (users.next(), users.next()) else {
            return;
        };
        let question = first.display.clone().unwrap_or_else(|| first.message.content.clone());
        let expected: String = question.chars().take(52).collect();
        if self.session.title != expected {
            return;
        }
        let Some(answer) = self.session.entries.iter().rev().find(|e| {
            e.message.role == "assistant"
                && !e.message.content.trim().is_empty()
                && e.stats.get("incomplete").and_then(Value::as_bool) != Some(true)
        }) else {
            return;
        };
        let model = if answer.model.trim().is_empty() {
            self.settings.active_model().to_string()
        } else {
            answer.model.clone()
        };
        if model.is_empty()
            || (!self.settings.allow_remote
                && self.models.iter().any(|m| m.name == model && m.is_remote()))
        {
            return;
        }
        let excerpt = |text: &str| text.chars().take(1_500).collect::<String>();
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: "You name conversations. Reply with a title of at most six words and nothing else: no quotes, no punctuation at the end, no explanation.".into(),
                ..Default::default()
            },
            ChatMessage {
                role: "user".into(),
                content: format!(
                    "Name this conversation.\n\n<user>\n{}\n</user>\n\n<assistant>\n{}\n</assistant>",
                    excerpt(&question),
                    excerpt(&answer.message.content)
                ),
                ..Default::default()
            },
        ];
        let options = crate::ollama::ChatOptions {
            temperature: 0.2,
            num_ctx: 4096,
            num_predict: 48,
            think: Some(json!(false)),
            keep_alive: self.settings.keep_alive.clone(),
            ..Default::default()
        };
        let base_url = self.settings.base_url.clone();
        let session_id = self.session.id.clone();
        let tx = self.work_tx.clone();
        std::thread::spawn(move || {
            // A title is a convenience: any failure just keeps the automatic one.
            let _ = crate::jobs::guarded(|| -> Result<()> {
                let client = crate::ollama::OllamaClient::new(&base_url)?;
                let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
                let reply = runtime.block_on(client.chat_stream(
                    &model,
                    &messages,
                    &options,
                    &[],
                    Arc::new(AtomicBool::new(false)),
                    |_| {},
                ))?;
                if let Some(title) = clean_title(&reply.content) {
                    let _ = tx.send(WorkEvent::Title { session_id, expected, title });
                }
                Ok(())
            });
        });
    }

    pub(super) fn apply_generated_title(&mut self, session_id: &str, expected: &str, title: String) {
        if self.session.id == session_id {
            if self.session.title == expected {
                self.session.title = title;
                self.persist();
            }
        } else if let Ok(mut session) = self.store.load(session_id) {
            // The user moved on to another conversation; update the saved one.
            if session.title == expected {
                session.title = title;
                if self.store.save(&mut session).is_ok() {
                    self.refresh_conversation_list();
                }
            }
        }
    }
}

/// A model reply reduced to a clean one-line title, or None if nothing usable is left.
pub(super) fn clean_title(raw: &str) -> Option<String> {
    let mut text = raw.to_string();
    // Drop any reasoning a thinking model wrote into its content.
    while let (Some(start), Some(end)) = (text.find("<think>"), text.find("</think>")) {
        if end < start {
            break;
        }
        text.replace_range(start..end + "</think>".len(), "");
    }
    let line = text.lines().map(str::trim).find(|l| !l.is_empty())?;
    let mut title = line.trim_start_matches(['#', '*', '_', '`', '-', ' ']).to_string();
    for prefix in ["title:", "conversation title:", "suggested title:"] {
        if title.to_lowercase().starts_with(prefix) {
            title = title[prefix.len()..].to_string();
        }
    }
    let title = title
        .trim()
        .trim_matches(|c: char| matches!(c, '"' | '\'' | '“' | '”' | '‘' | '’' | '*' | '_' | '`'))
        .trim_end_matches(['.', ':', ';', ','])
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let mut short = String::new();
    for word in title.split(' ') {
        if !short.is_empty() && short.chars().count() + 1 + word.chars().count() > 60 {
            break;
        }
        if !short.is_empty() {
            short.push(' ');
        }
        short.push_str(word);
    }
    let short: String = short.chars().take(60).collect();
    (!short.trim().is_empty()).then_some(short)
}
