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
        self.persist();
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
        self.persist();
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

    pub(super) fn persist(&mut self) {
        if self.demo {
            return;
        }
        if let Err(e) = self.store.save(&mut self.session) {
            if !e.is::<storage::SessionConflict>() {
                self.error = Some(format!("Save conversation: {e:#}"));
                return;
            }
            let mut copy = self.session.clone();
            copy.id = uuid::Uuid::new_v4().to_string();
            copy.title = format!("{} · recovered edits", copy.title);
            if let Err(error) = self.store.save(&mut copy) {
                self.error = Some(format!("Save recovered conversation: {error:#}"));
                return;
            }
            self.session = copy;
            self.error = Some("Another window changed this conversation. Your edits were saved in a separate conversation.".into());
        }
        self.settings.last_session = Some(self.session.id.clone());
        if !self.settings_corrupt {
            if let Err(e) = self.settings.save(&self.dir) {
                self.error = Some(format!("Save settings: {e:#}"));
            }
        }
        self.refresh_conversation_list();
        self.dirty = false;
        self.last_save = Instant::now();
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
        if id == self.session.id {
            self.persist();
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
        if id == self.session.id {
            self.persist();
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
}
