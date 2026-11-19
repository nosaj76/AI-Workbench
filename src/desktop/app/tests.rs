//! Headless egui input/layout regressions. These do not qualify a graphics driver.
use super::*;

fn fixture() -> (tempfile::TempDir, egui::Context, Workbench) {
    let dir = tempfile::tempdir().unwrap();
    let ctx = egui::Context::default();
    let app = Workbench::from_context(&ctx, dir.path().to_path_buf(), true).unwrap();
    (dir, ctx, app)
}
fn frame(
    app: &mut Workbench,
    ctx: &egui::Context,
    size: [f32; 2],
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    ctx.run(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(size[0], size[1]),
            )),
            events,
            ..Default::default()
        },
        |ctx| app.render(ctx),
    )
}
fn text_rect(shape: &egui::epaint::Shape, text: &str) -> Option<egui::Rect> {
    match shape {
        egui::epaint::Shape::Text(label) if label.galley.job.text == text => {
            Some(egui::Rect::from_min_size(label.pos, label.galley.size()))
        }
        egui::epaint::Shape::Vec(shapes) => shapes.iter().find_map(|shape| text_rect(shape, text)),
        _ => None,
    }
}
fn visible_label(output: &egui::FullOutput, text: &str, size: [f32; 2]) -> egui::Rect {
    let rect = output
        .shapes
        .iter()
        .find_map(|shape| text_rect(&shape.shape, text))
        .unwrap_or_else(|| panic!("Missing GUI control: {text}"));
    assert!(
        rect.min.x >= 0.0 && rect.min.y >= 0.0 && rect.max.x <= size[0] && rect.max.y <= size[1],
        "{text} is outside the viewport: {rect:?}"
    );
    rect
}

#[test]
fn theme_and_send_controls_remain_in_compact_and_desktop_viewports() {
    for size in [[1280.0, 720.0], [900.0, 600.0], [760.0, 560.0]] {
        let (_dir, ctx, mut app) = fixture();
        app.input = "A test message".into();
        // Settle the egui panel geometry before checking final paint output.
        let _ = frame(&mut app, &ctx, size, vec![]);
        let output = frame(&mut app, &ctx, size, vec![]);
        for label in ["Light", "Dark", "Send"] {
            visible_label(&output, label, size);
        }
    }
}

#[test]
fn clicking_the_toolbar_switch_changes_the_theme_and_persists_it() {
    let (dir, ctx, mut app) = fixture();
    app.demo = false;
    let size = [1000.0, 700.0];
    let _ = frame(&mut app, &ctx, size, vec![]);
    let output = frame(&mut app, &ctx, size, vec![]);
    let pos = visible_label(&output, "Dark", size).center();
    let _ = frame(
        &mut app,
        &ctx,
        size,
        vec![
            egui::Event::PointerMoved(pos),
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::default(),
            },
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            },
        ],
    );
    assert!(app.settings.dark_mode);
    assert!(ctx.style().visuals.dark_mode);
    assert!(Settings::load(dir.path()).unwrap().dark_mode);
    app.set_theme(&ctx, false);
    assert!(!Settings::load(dir.path()).unwrap().dark_mode);
}

#[test]
fn invalid_settings_draft_does_not_mutate_live_or_persisted_preferences() {
    let (dir, ctx, mut app) = fixture();
    app.demo = false;
    app.settings.save(dir.path()).unwrap();
    let previous = std::fs::read(dir.path().join("settings.json")).unwrap();
    let original = serde_json::to_value(&app.settings).unwrap();
    app.profiles_json = r#"[{"name":"Uncommitted","model":"changed","instruction":""}]"#.into();
    app.prompts_json = "not valid json".into();
    assert!(app.commit_settings(app.settings.clone(), &ctx).is_err());
    assert_eq!(serde_json::to_value(&app.settings).unwrap(), original);
    assert_eq!(
        std::fs::read(dir.path().join("settings.json")).unwrap(),
        previous
    );
}

#[test]
fn stale_connection_events_cannot_replace_the_current_server_state() {
    let (_dir, _ctx, mut app) = fixture();
    app.connection_request_id = 3;
    app.version = "current".into();
    let endpoint = app.settings.base_url.clone();
    app.work_tx
        .send(WorkEvent::Models {
            request_id: 2,
            endpoint: endpoint.clone(),
            models: vec![],
            version: "stale".into(),
            running: Value::Null,
        })
        .unwrap();
    app.receive_events();
    assert_eq!(app.version, "current");
    app.work_tx
        .send(WorkEvent::ConnectionError {
            request_id: 2,
            endpoint: endpoint.clone(),
            message: "old failure".into(),
        })
        .unwrap();
    app.receive_events();
    assert_eq!(app.version, "current");
    app.work_tx
        .send(WorkEvent::ConnectionError {
            request_id: 3,
            endpoint,
            message: "current failure".into(),
        })
        .unwrap();
    app.receive_events();
    assert!(app.version.is_empty());
    assert!(app.error.as_deref().unwrap().contains("current failure"));
}

#[test]
fn an_older_file_read_cannot_replace_the_last_selected_preview() {
    let (_dir, _ctx, mut app) = fixture();
    app.session.project = Some("fixture-project".into());
    app.preview_request_id = 2;
    app.preview = "new preview".into();
    app.work_tx
        .send(WorkEvent::Preview {
            source_text: true,
            truncated: false,
            request_id: 1,
            root: "fixture-project".into(),
            path: "old.txt".into(),
            text: "old preview".into(),
            note: String::new(),
        })
        .unwrap();
    app.receive_events();
    assert_eq!(app.preview, "new preview");
    app.work_tx
        .send(WorkEvent::Preview {
            source_text: true,
            truncated: false,
            request_id: 2,
            root: "fixture-project".into(),
            path: "new.txt".into(),
            text: "selected preview".into(),
            note: String::new(),
        })
        .unwrap();
    app.receive_events();
    assert_eq!(app.preview, "selected preview");
    assert_eq!(app.preview_path, "new.txt");
}

#[test]
fn an_extracted_or_partial_preview_cannot_request_an_overwrite() {
    let (dir, _ctx, mut app) = fixture();
    app.session.project = Some(dir.path().display().to_string());
    app.preview_request_id = 1;
    for (path, source_text, truncated) in [
        ("scan.bmp", false, false),
        ("scan.tif", false, false),
        ("scan.tiff", false, false),
        ("large.txt", true, true),
    ] {
        let original = b"original file bytes";
        std::fs::write(dir.path().join(path), original).unwrap();
        app.work_tx.send(WorkEvent::Preview {
            request_id: 1,
            root: dir.path().display().to_string(),
            path: path.into(),
            source_text,
            truncated,
            text: "extracted preview".into(),
            note: String::new(),
        }).unwrap();
        app.receive_events();
        app.preview = "edited preview".into();
        assert!(!app.can_save_preview_edit());
        app.review_preview_edit();
        assert!(!app.work_busy && app.approval.is_none());
        assert_eq!(std::fs::read(dir.path().join(path)).unwrap().as_slice(), original);
    }
}

#[test]
fn saving_divergent_window_edits_keeps_both_conversations() {
    let (dir, _ctx, mut app) = fixture();
    app.demo = false;
    app.session = Session::default();
    app.persist();
    let original_id = app.session.id.clone();
    let other = Store::open(dir.path()).unwrap();
    let mut newer = other.load(&original_id).unwrap();
    newer.title = "Changed in another window".into();
    other.save(&mut newer).unwrap();

    app.session.title = "Edited here".into();
    app.persist();
    assert_ne!(app.session.id, original_id);
    assert_eq!(other.load(&original_id).unwrap().title, newer.title);
    assert_eq!(other.load(&app.session.id).unwrap().title, "Edited here · recovered edits");
    assert_eq!(Settings::load(dir.path()).unwrap().last_session.as_deref(), Some(app.session.id.as_str()));
}

#[test]
fn conversation_switch_clears_previous_draft_and_attachments() {
    let (_dir, _ctx, mut app) = fixture();
    app.demo = false;
    app.session = Session::default();
    let mut next = Session {
        title: "Second conversation".into(),
        ..Default::default()
    };
    app.store.save(&mut next).unwrap();
    app.input = "Private unsent draft".into();
    app.attachments.push(Attachment {
        name: "Old evidence".into(),
        source: "Old project".into(),
        text: "private evidence".into(),
        image: None,
    });
    app.open_conversation(&next.id);
    assert_eq!(app.session.id, next.id);
    assert!(app.input.is_empty());
    assert!(app.attachments.is_empty());
}

#[test]
fn a_stale_settings_draft_cannot_overwrite_new_service_preferences() {
    let (dir, ctx, mut app) = fixture();
    app.demo = false;
    let old = app.settings.clone();
    app.settings_draft_original = Some(old.clone());
    app.profiles_json = serde_json::to_string(&old.profiles).unwrap();
    app.prompts_json = serde_json::to_string(&old.prompts).unwrap();
    app.settings.base_url = "http://127.0.0.1:11435".into();
    app.settings.save(dir.path()).unwrap();
    let failure = app.commit_settings(old, &ctx).unwrap_err().to_string();
    assert!(failure.contains("changed elsewhere"));
    assert_eq!(
        Settings::load(dir.path()).unwrap().base_url,
        "http://127.0.0.1:11435"
    );
}

#[test]
fn sidebar_trash_actions_preserve_saved_content_until_explicit_purge() {
    let (dir, _ctx, mut app) = fixture();
    app.demo = false;
    app.session = Session::default();
    app.session.title = "Review to retain".into();
    app.session.entries.push(Entry::new(ChatMessage {
        role: "assistant".into(),
        content: "Saved answer".into(),
        ..Default::default()
    }));
    app.persist();
    let id = app.session.id.clone();
    app.conversation_action(ConversationAction::Trash(id.clone()));
    assert_eq!(app.pending_trash_id.as_deref(), Some(id.as_str()));
    // The first step only opens a confirmation; no deletion has occurred.
    assert_eq!(app.store.list("", false).unwrap().len(), 1);
    app.confirm_move_to_trash(&id);
    app.pending_trash_id = None;
    assert!(app.store.list("Review to retain", true).unwrap().is_empty());
    assert_eq!(app.store.list_trash("Review to retain").unwrap().len(), 1);
    assert_eq!(
        app.store.load(&id).unwrap().entries[0].message.content,
        "Saved answer"
    );
    app.conversation_action(ConversationAction::Restore(id.clone()));
    assert_eq!(app.session.id, id);
    assert_eq!(app.session.entries[0].message.content, "Saved answer");
    app.conversation_action(ConversationAction::Trash(id.clone()));
    app.confirm_move_to_trash(&id);
    app.conversation_action(ConversationAction::DeleteForever(id.clone()));
    assert_eq!(
        app.pending_permanent_delete_id.as_deref(),
        Some(id.as_str())
    );
    assert!(app.store.load(&id).is_ok());
    app.confirm_delete_forever(&id);
    assert!(app.store.load(&id).is_err());
    assert!(dir.path().join("workbench.sqlite3").exists());
}

#[test]
fn compact_theme_uses_normal_desktop_text_sizes() {
    let ctx = egui::Context::default();
    crate::theme::apply(&ctx, false, 1.0);
    let style = ctx.style();
    let body = style.text_styles.get(&egui::TextStyle::Body).unwrap();
    let button = style.text_styles.get(&egui::TextStyle::Button).unwrap();
    assert!(body.size <= 14.0);
    assert!(button.size <= 13.0);
    assert!(style.spacing.interact_size.y <= 26.0);
}

#[test]
fn code_export_preview_tracks_the_selected_answer_and_invalidates_on_switch() {
    let (_dir, _ctx, mut app) = fixture();
    app.session.entries.push(Entry::new(ChatMessage {
        role: "assistant".into(),
        content: "Example\n```rust src/main.rs\nfn main() {}\n```".into(),
        ..Default::default()
    }));
    let answer_id = app.session.entries.last().unwrap().id.clone();
    app.prepare_code_export(&answer_id);
    let preview = app.code_export_preview.as_ref().unwrap();
    assert_eq!(preview.session_id, app.session.id);
    assert_eq!(preview.entry_id, answer_id);
    assert_eq!(preview.blocks[0].name, "main.rs");
    app.new_conversation();
    assert!(app.code_export_preview.is_none());
}

#[test]
fn incomplete_assistant_output_is_not_code_exportable() {
    let (_dir, _ctx, mut app) = fixture();
    let mut entry = Entry::new(ChatMessage {
        role: "assistant".into(),
        content: "```rust\nfn main() {}\n```".into(),
        ..Default::default()
    });
    entry.stats = json!({"incomplete": true});
    let id = entry.id.clone();
    app.session.entries.push(entry);
    app.prepare_code_export(&id);
    assert!(app.code_export_preview.is_none());
    assert!(app.error.as_deref().unwrap().contains("completed"));
}

#[test]
fn unexecuted_json_tool_text_shows_warning_but_preserves_raw_evidence() {
    let (_dir, _ctx, mut app) = fixture();
    let raw = r#"{"name":"project_tree","arguments":{"path":"."}}{"name":"read_file","arguments":{"path":"main.rs"}}"#;
    app.chat_tx
        .send(ChatEvent::Begin("fixture:7b".into()))
        .unwrap();
    app.chat_tx.send(ChatEvent::Content(raw.into())).unwrap();
    app.chat_tx
        .send(ChatEvent::UnexecutedToolText {
            message: ChatMessage {
                role: "assistant".into(),
                content: raw.into(),
                ..Default::default()
            },
            names: vec!["project_tree".into(), "read_file".into()],
            retrying: true,
        })
        .unwrap();
    app.receive_events();
    let entry = app.session.entries.last().unwrap();
    assert_eq!(
        entry.message.content, raw,
        "original raw bytes remain inspectable"
    );
    let display = entry.display.as_deref().unwrap();
    assert!(display.contains("2 tool requests as text"));
    assert!(display.contains("not executed"));
    assert_eq!(entry.stats["unexecuted_tool_text"], true);
    assert_eq!(entry.stats["unexecuted_tool_names"][0], "project_tree");
    assert!(
        !entry
            .request_message()
            .content
            .contains(r#"{"name":"project_tree""#),
        "future model turns should not repeat invalid tool JSON"
    );
    assert_eq!(
        entry.request_message().content,
        entry.display.as_deref().unwrap()
    );
}

#[test]
fn singular_unexecuted_tool_warning_uses_correct_grammar() {
    let (_dir, _ctx, mut app) = fixture();
    let raw = r#"{"name":"read_file","arguments":{"path":"main.rs"}}"#;
    app.chat_tx
        .send(ChatEvent::Begin("fixture:7b".into()))
        .unwrap();
    app.chat_tx.send(ChatEvent::Content(raw.into())).unwrap();
    app.chat_tx
        .send(ChatEvent::UnexecutedToolText {
            message: ChatMessage {
                role: "assistant".into(),
                content: raw.into(),
                ..Default::default()
            },
            names: vec!["read_file".into()],
            retrying: false,
        })
        .unwrap();
    app.receive_events();

    let display = app
        .session
        .entries
        .last()
        .unwrap()
        .display
        .as_deref()
        .unwrap();
    assert!(display.contains("1 tool request as text"));
    assert!(!display.contains("1 tool requests"));
    assert!(display.contains("evidence-only explanation"));
}

#[test]
fn chat_preflight_error_is_persisted_in_transcript_not_relabelled_stopped() {
    let (_dir, _ctx, mut app) = fixture();
    app.chat_tx
        .send(ChatEvent::Error("No search provider configured".into()))
        .unwrap();
    app.chat_tx.send(ChatEvent::Finished).unwrap();
    app.receive_events();
    let entry = app.session.entries.last().unwrap();
    assert!(entry
        .message
        .content
        .contains("No search provider configured"));
    assert_eq!(entry.stats["interruption_kind"], "error");
    assert_eq!(entry.stats["incomplete"], true);
    assert_eq!(app.status, "Request failed");
}

#[test]
fn chat_stream_error_preserves_partial_output_and_actual_failure() {
    let (_dir, _ctx, mut app) = fixture();
    app.chat_tx.send(ChatEvent::Begin("test".into())).unwrap();
    app.chat_tx
        .send(ChatEvent::Content("Partial observation.".into()))
        .unwrap();
    app.chat_tx
        .send(ChatEvent::Error("Ollama connection reset".into()))
        .unwrap();
    app.chat_tx.send(ChatEvent::Finished).unwrap();
    app.receive_events();
    let entry = app.session.entries.last().unwrap();
    assert!(entry.message.content.contains("Partial observation"));
    assert!(entry.message.content.contains("Ollama connection reset"));
    assert!(!entry.message.content.contains("Request stopped before"));
    assert_eq!(entry.stats["interruption_kind"], "error");
    assert!(app.streaming.is_none());
}

#[test]
fn chat_cancel_and_unexpected_end_have_distinct_saved_reasons() {
    for cancelled in [false, true] {
        let (_dir, _ctx, mut app) = fixture();
        app.chat_tx.send(ChatEvent::Begin("test".into())).unwrap();
        app.cancel.store(cancelled, Ordering::Relaxed);
        app.chat_tx.send(ChatEvent::Finished).unwrap();
        app.receive_events();
        let entry = app.session.entries.last().unwrap();
        assert_eq!(
            entry.stats["interruption_kind"],
            if cancelled {
                "cancelled"
            } else {
                "unexpected_end"
            }
        );
        assert!(entry.message.content.contains(if cancelled {
            "Cancellation was requested"
        } else {
            "No cancellation was requested"
        }));
    }
}

#[test]
fn tool_rows_distinguish_failed_denied_skipped_partial_and_completed_results() {
    use super::chat::tool_outcome;
    assert_eq!(tool_outcome(r#"{"error":"No such file"}"#).0, "failed");
    assert!(tool_outcome(r#"{"error":"No such file"}"#)
        .1
        .unwrap()
        .contains("No such file"));
    assert_eq!(tool_outcome(r#"{"ok":false}"#).0, "failed");
    assert_eq!(
        tool_outcome(r#"{"ok":true,"error":null,"completion_known":false}"#),
        ("completed", None)
    );
    assert_eq!(tool_outcome(r#"{"ok":false,"error":null}"#).0, "failed");
    assert_eq!(
        tool_outcome(r#"{"denied":true,"executed":false}"#).0,
        "denied"
    );
    assert_eq!(
        tool_outcome(r#"{"duplicate":true,"executed":false}"#).0,
        "skipped duplicate"
    );
    assert_eq!(tool_outcome(r#"{"cancelled":true}"#).0, "cancelled");
    assert_eq!(tool_outcome(r#"{"truncated":true}"#).0, "partial result");
    assert_eq!(tool_outcome(r#"{"entries":[]}"#).0, "completed");
}

#[test]
fn cancellation_between_tool_result_and_followup_gets_a_saved_explanation() {
    let (_dir, _ctx, mut app) = fixture();
    app.session.entries.push(Entry::new(ChatMessage {
        role: "tool".into(),
        tool_name: Some("list_files".into()),
        content: r#"{"entries":[]}"#.into(),
        ..Default::default()
    }));
    app.cancel.store(true, Ordering::Relaxed);
    app.chat_tx.send(ChatEvent::Finished).unwrap();
    app.receive_events();
    let last = app.session.entries.last().unwrap();
    assert_eq!(last.message.role, "assistant");
    assert_eq!(last.stats["interruption_kind"], "cancelled");
    assert!(app
        .session
        .entries
        .iter()
        .any(|entry| entry.message.role == "tool"));
}

#[test]
fn chat_tool_engine_setup_failure_is_saved_before_worker_launch() {
    let (dir, _ctx, mut app) = fixture();
    app.demo = false;
    app.session = Session::default();
    app.session.project = Some(dir.path().display().to_string());
    app.settings.tools_enabled = true;
    app.settings.mcp_enabled = true;
    app.settings.mcp_json = "not a JSON server array".into();
    app.launch_chat(vec![], "fixture".into());
    assert!(!app.busy);
    let entry = app.session.entries.last().unwrap();
    assert_eq!(entry.stats["interruption_kind"], "configuration");
    assert!(entry.message.content.contains("Tool setup failed"));
    assert_eq!(app.status, "Request failed");
}

#[test]
fn approval_window_stays_above_the_tool_console_after_the_console_is_clicked() {
    let (dir, ctx, mut app) = fixture();
    let project = dir.path().join("project");
    std::fs::create_dir(&project).unwrap();
    let engine = ToolEngine::new(crate::files::Workspace::new(project).unwrap());
    let action = match engine
        .prepare("write_file", &json!({"path":"note.txt","content":"x"}))
        .unwrap()
    {
        crate::tools::PreparedTool::Approval(action) => *action,
        crate::tools::PreparedTool::Immediate(_) => panic!("write_file requires approval"),
    };
    let (reply, _rx) = std::sync::mpsc::channel();
    app.tool_console
        .started("1".into(), "write_file".into(), "note.txt".into());
    app.approval = Some(Approval {
        action,
        origin: ApprovalOrigin::Chat(reply),
    });
    let size = [1280.0, 800.0];
    let _ = frame(&mut app, &ctx, size, vec![]);
    let _ = frame(&mut app, &ctx, size, vec![]);
    let console = ctx
        .memory(|m| m.area_rect(egui::Id::new("tool_activity_console")))
        .expect("console is shown");
    let approval_layer = ctx
        .memory(|m| m.areas().top_layer_id(egui::Order::Middle))
        .expect("a window is shown");
    assert_eq!(approval_layer.id, egui::Id::new("Review action"));
    let approval_rect = ctx
        .memory(|m| m.area_rect(approval_layer.id))
        .expect("approval rect");
    // Click a point of the console that the approval window does not cover.
    let pos = [
        console.left_bottom() + egui::vec2(20.0, -20.0),
        console.right_bottom() + egui::vec2(-20.0, -20.0),
        console.left_top() + egui::vec2(20.0, 12.0),
    ]
    .into_iter()
    .find(|p| !approval_rect.contains(*p))
    .expect("part of the console is uncovered");
    let click = |pressed| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::default(),
    };
    let _ = frame(
        &mut app,
        &ctx,
        size,
        vec![egui::Event::PointerMoved(pos), click(true), click(false)],
    );
    let _ = frame(&mut app, &ctx, size, vec![]);
    assert_eq!(
        ctx.memory(|m| m.areas().top_layer_id(egui::Order::Middle)),
        Some(approval_layer),
        "the approval decision must not be hidden behind the console"
    );
}

#[test]
fn qa_page_renders_its_controls_and_cache() {
    let (_dir, ctx, mut app) = fixture();
    app.page = Page::Advanced;
    app.advanced.section = crate::advanced_ui::Section::Qa;
    let size = [1280.0, 800.0];
    let _ = frame(&mut app, &ctx, size, vec![]);
    let output = frame(&mut app, &ctx, size, vec![]);
    for label in ["Q&A", "Ask", "Ask again online", "Answers (0)", "Facts (0)"] {
        visible_label(&output, label, size);
    }
}

#[test]
fn project_can_be_disconnected_and_new_conversations_start_without_one() {
    let (dir, _ctx, mut app) = fixture();
    app.set_workspace(dir.path().to_path_buf());
    assert!(app.session.project.is_some());
    app.disconnect_workspace();
    assert!(app.session.project.is_none());
    assert!(app.workspace_input.is_empty());
    app.set_workspace(dir.path().to_path_buf());
    app.new_conversation();
    assert!(app.session.project.is_none());
    assert!(app.workspace_input.is_empty());
}

#[test]
fn auto_approval_answers_only_the_enabled_kinds_and_records_each_one() {
    let (_dir, _ctx, mut app) = fixture();
    let action = |operation: crate::tools::Operation| crate::tools::PendingAction {
        id: "a".into(),
        title: "Search the web with SearXNG".into(),
        description: String::new(),
        details: serde_json::json!({"query": "ginseng"}),
        operation,
        workspace: std::path::PathBuf::new(),
        expected: vec![],
    };
    let search = crate::tools::Operation::WebSearch {
        query: "ginseng".into(),
        base_url: "http://127.0.0.1:18080".into(),
        limit: 5,
        fallbacks: vec![],
        local_sites: None,
        options: Default::default(),
    };
    let delete = crate::tools::Operation::Delete {
        path: "a.txt".into(),
    };

    // Off by default: the user is asked.
    let (reply, answer) = std::sync::mpsc::channel();
    app.chat_tx
        .send(ChatEvent::Approval {
            action: action(search.clone()),
            reply,
        })
        .unwrap();
    app.receive_events();
    assert!(app.approval.is_some());
    assert!(answer.try_recv().is_err());
    app.approval = None;

    // Web searches enabled: approved at once and logged; a file deletion still asks.
    app.settings.auto_approve_web = true;
    let (reply, answer) = std::sync::mpsc::channel();
    app.chat_tx
        .send(ChatEvent::Approval {
            action: action(search),
            reply,
        })
        .unwrap();
    app.receive_events();
    assert_eq!(answer.try_recv(), Ok(true));
    assert!(app.approval.is_none());
    assert!(app.store.recent_audit().unwrap()[0].contains("Auto-approved"));

    let (reply, answer) = std::sync::mpsc::channel();
    app.chat_tx
        .send(ChatEvent::Approval {
            action: action(delete),
            reply,
        })
        .unwrap();
    app.receive_events();
    assert!(app.approval.is_some(), "file changes were not enabled");
    assert!(answer.try_recv().is_err());
}

#[test]
fn prompts_typed_while_a_reply_is_written_are_queued_and_sent_in_order() {
    let (_dir, ctx, mut app) = fixture();
    app.busy = true;
    // The composer stays editable while busy; queued prompts wait.
    app.input = "first follow-up".into();
    app.queued_prompts
        .push(std::mem::take(&mut app.input).trim().to_string());
    app.input = "second follow-up".into();
    app.queued_prompts
        .push(std::mem::take(&mut app.input).trim().to_string());
    app.input = "still typing".into();
    let _ = frame(&mut app, &ctx, [1000.0, 700.0], vec![]);
    assert_eq!(app.queued_prompts.len(), 2);
    // The reply finishes and the oldest queued prompt is sent. This fixture runs
    // in demo mode, where sending is refused: the prompt must stay queued, in
    // order, and the draft being typed must be untouched.
    app.chat_tx.send(ChatEvent::Finished).unwrap();
    app.receive_events();
    assert_eq!(app.queued_prompts, ["first follow-up", "second follow-up"]);
    assert_eq!(app.input, "still typing");
    assert!(
        app.error.as_deref().unwrap_or("").contains("demo"),
        "{:?}",
        app.error
    );
}

#[test]
fn after_stop_the_queue_is_kept_and_not_sent() {
    let (_dir, _ctx, mut app) = fixture();
    app.busy = true;
    app.queued_prompts.push("later".into());
    app.cancel.store(true, Ordering::Relaxed);
    app.chat_tx.send(ChatEvent::Finished).unwrap();
    app.receive_events();
    assert_eq!(app.queued_prompts, ["later"]);
    assert!(app.status.contains("queued prompt"), "{}", app.status);
}

#[cfg(unix)]
#[test]
fn ai_and_visible_terminal_share_sessions_across_engine_rebuilds_but_not_workspaces() {
    let (dir, _ctx, mut app) = fixture();
    let first = dir.path().join("first-project");
    let second = dir.path().join("second-project");
    std::fs::create_dir(&first).unwrap();
    std::fs::create_dir(&second).unwrap();
    app.session.project = Some(first.display().to_string());
    let engine = app.get_engine().unwrap();
    let manager = engine.terminal_manager();
    let id = manager
        .open(crate::terminal::TerminalConfig {
            cwd: first.clone(),
            program: "bash".into(),
            args: vec!["--noprofile".into(), "--norc".into(), "-i".into()],
            ..Default::default()
        })
        .unwrap();
    assert!(
        app.advanced
            .terminal_manager_for(Some(&first))
            .summary(&id)
            .unwrap()
            .alive
    );

    // Settings changes discard configured engines. They must not kill the
    // ongoing interpreter or create an unrelated terminal registry.
    app.manual_engine = None;
    drop(engine);
    drop(manager);
    let rebuilt = app.get_engine().unwrap();
    assert!(rebuilt.terminal_manager().summary(&id).unwrap().alive);
    app.session.project = Some(second.display().to_string());
    let other = app.get_engine().unwrap();
    assert!(other.terminal_manager().list().is_empty());
    assert!(other.terminal_manager().summary(&id).is_err());

    // Returning to the original workspace resumes the same process.
    app.session.project = Some(first.display().to_string());
    let resumed = app.get_engine().unwrap();
    assert!(resumed.terminal_manager().summary(&id).unwrap().alive);
    resumed.terminal_manager().close(&id).unwrap();
    assert!(rebuilt.terminal_manager().summary(&id).is_err());
}

#[test]
fn bash_request_selects_a_workspace_and_exposes_shared_terminal_tools() {
    let (dir, _ctx, mut app) = fixture();
    app.demo = false;
    app.session = Session::default();
    app.settings.tools_enabled = true;
    app.settings.network_tools = false;
    app.settings.auto_approve_commands = false;
    let request = ChatMessage {
        role: "user".into(),
        content: "open bash".into(),
        ..Default::default()
    };
    app.session.entries.push(Entry::new(request.clone()));
    assert!(app
        .prepare_chat_workspace(std::slice::from_ref(&request))
        .unwrap());
    let expected = dir.path().join("workspaces").join("terminal");
    assert_eq!(
        app.session.project.as_deref().map(Path::new),
        Some(expected.as_path())
    );
    assert_eq!(app.workspace_input, expected.display().to_string());
    assert_eq!(
        app.store.load(&app.session.id).unwrap().project,
        app.session.project
    );
    assert!(!app.settings.auto_approve_commands);

    let engine = app.get_chat_engine().unwrap().unwrap();
    assert_eq!(engine.root(), expected);
    let definitions = engine.definitions();
    for name in [
        "terminal_open",
        "terminal_list",
        "terminal_read",
        "terminal_write",
    ] {
        assert!(
            definitions.iter().any(|definition| {
                definition.pointer("/function/name").and_then(Value::as_str) == Some(name)
            }),
            "missing {name}"
        );
    }
    assert!(
        engine.terminal_manager().list().is_empty(),
        "Selecting a folder must not launch a shell"
    );
    assert!(!app.prepare_chat_workspace(&[request]).unwrap());

    #[cfg(unix)]
    {
        let action = match engine
            .prepare(
                "terminal_open",
                &json!({"args":["--noprofile","--norc","-i"],"wait_ms":20}),
            )
            .unwrap()
        {
            crate::tools::PreparedTool::Approval(action) => *action,
            crate::tools::PreparedTool::Immediate(_) => {
                panic!("Opening Bash still requires command approval")
            }
        };
        assert!(engine.terminal_manager().list().is_empty());
        let result = engine.approve(&action).unwrap();
        let id = result["session_id"].as_str().unwrap();
        assert!(
            app.advanced
                .terminal_manager_for(Some(&expected))
                .summary(id)
                .unwrap()
                .alive
        );
        app.manual_engine = None;
        let rebuilt = app.get_chat_engine().unwrap().unwrap();
        assert!(rebuilt.terminal_manager().summary(id).unwrap().alive);
        rebuilt.terminal_manager().close(id).unwrap();
    }
}

#[test]
fn terminal_request_keeps_the_chosen_workspace() {
    let (dir, _ctx, mut app) = fixture();
    let selected = dir.path().join("my-basic-programs");
    std::fs::create_dir(&selected).unwrap();
    app.set_workspace(selected.clone());
    let request = ChatMessage {
        role: "user".into(),
        content: "use modern basic".into(),
        ..Default::default()
    };
    assert!(!app.prepare_chat_workspace(&[request]).unwrap());
    assert_eq!(
        app.session.project.as_deref().map(Path::new),
        Some(selected.as_path())
    );
    assert_eq!(app.get_chat_engine().unwrap().unwrap().root(), selected);
    assert!(!dir.path().join("workspaces").exists());
}

#[test]
fn non_actions_disabled_tools_and_comparisons_do_not_create_a_terminal_workspace() {
    for (prompt, tools_enabled, comparison) in [
        ("Explain how BASIC works", true, false),
        ("How do I open modernbasic?", true, false),
        ("Do not open modernbasic", true, false),
        (
            "Review this text\n<local_file>use modern basic</local_file>",
            true,
            false,
        ),
        ("use modern basic", false, false),
        ("use modern basic", true, true),
    ] {
        let (dir, _ctx, mut app) = fixture();
        app.session = Session::default();
        app.settings.tools_enabled = tools_enabled;
        app.compare_active = comparison;
        let messages = [
            ChatMessage {
                role: "user".into(),
                content: "use modern basic".into(),
                ..Default::default()
            },
            ChatMessage {
                role: "user".into(),
                content: prompt.into(),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".into(),
                content: "Open Bash now".into(),
                ..Default::default()
            },
        ];
        assert!(!app.prepare_chat_workspace(&messages).unwrap(), "{prompt}");
        assert!(app.session.project.is_none(), "{prompt}");
        assert!(!dir.path().join("workspaces").exists(), "{prompt}");
        assert_eq!(app.settings.tools_enabled, tools_enabled);
    }
}

#[test]
fn failed_default_terminal_workspace_stops_before_model_launch_and_keeps_the_turn() {
    let (dir, _ctx, mut app) = fixture();
    app.demo = false;
    app.session = Session::default();
    app.settings.tools_enabled = true;
    std::fs::write(dir.path().join("workspaces"), "keep this file").unwrap();
    let request = ChatMessage {
        role: "user".into(),
        content: "open bash".into(),
        ..Default::default()
    };
    app.session.entries.push(Entry::new(request.clone()));
    app.launch_chat(vec![request], "fixture".into());
    assert!(!app.busy);
    assert!(app.session.project.is_none());
    assert!(app.manual_engine.is_none());
    assert_eq!(app.session.entries[0].message.content, "open bash");
    let failure = app.session.entries.last().unwrap();
    assert_eq!(failure.stats["interruption_kind"], "configuration");
    assert!(failure
        .message
        .content
        .contains("Default terminal workspace"));
    assert_eq!(app.status, "Request failed");
    assert_eq!(
        std::fs::read_to_string(dir.path().join("workspaces")).unwrap(),
        "keep this file"
    );
    let saved = app.store.load(&app.session.id).unwrap();
    assert_eq!(saved.entries.len(), app.session.entries.len());
}

#[cfg(unix)]
#[test]
fn default_terminal_workspace_does_not_follow_a_redirected_directory() {
    let (dir, _ctx, mut app) = fixture();
    app.session = Session::default();
    let outside = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(outside.path(), dir.path().join("workspaces")).unwrap();
    let request = ChatMessage {
        role: "user".into(),
        content: "open bash".into(),
        ..Default::default()
    };
    let error = app.prepare_chat_workspace(&[request]).unwrap_err();
    assert!(error.to_string().contains("regular directory"));
    assert!(app.session.project.is_none());
    assert!(!outside.path().join("terminal").exists());
}

#[test]
fn terminal_and_command_approval_controls_are_visible_in_chat_and_settings() {
    for size in [[1366.0, 768.0], [900.0, 600.0], [760.0, 560.0]] {
        let (_dir, ctx, mut app) = fixture();
        app.session = Session::default();
        app.page = Page::Chat;
        let _ = frame(&mut app, &ctx, size, vec![]);
        let output = frame(&mut app, &ctx, size, vec![]);
        visible_label(&output, "Terminal", size);
        visible_label(
            &output,
            "CLI requests select a default working folder automatically.",
            size,
        );
        let pos = visible_label(&output, "Command approvals…", size).center();
        let _ = frame(
            &mut app,
            &ctx,
            size,
            vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::default(),
                },
            ],
        );
        assert!(app.page == Page::Settings);
        let _ = frame(&mut app, &ctx, size, vec![]);
        let settings = frame(&mut app, &ctx, size, vec![]);
        for text in [
            "Tools and command approvals",
            "Enable workspace tools",
            "Approve automatically in chat",
            "Save changes",
        ] {
            visible_label(&settings, text, size);
        }
    }
}

#[test]
fn basic_library_replaces_only_missing_or_fallback_workspace() {
    let (dir, _ctx, mut app) = fixture();
    app.session = Session::default();
    app.settings.tools_enabled = true;
    let library = dir.path().join("ModernBASIC/programs");
    std::fs::create_dir_all(&library).unwrap();
    let request = ChatMessage {
        role: "user".into(),
        content: "write basic code for teh game frogger".into(),
        ..Default::default()
    };
    for current in [
        None,
        Some(dir.path().join("workspaces/terminal").display().to_string()),
    ] {
        app.session.project = current;
        assert!(app
            .prepare_chat_workspace_with_library(
                std::slice::from_ref(&request),
                Some(library.clone())
            )
            .unwrap());
        assert_eq!(
            app.session.project.as_deref().map(Path::new),
            Some(library.as_path())
        );
    }
    let chosen = dir.path().join("chosen");
    std::fs::create_dir(&chosen).unwrap();
    app.set_workspace(chosen.clone());
    assert!(!app
        .prepare_chat_workspace_with_library(&[request], Some(library))
        .unwrap());
    assert_eq!(
        app.session.project.as_deref().map(Path::new),
        Some(chosen.as_path())
    );
}

#[test]
fn background_failure_is_logged_without_marking_the_answer_incomplete() {
    let (dir, _ctx, mut app) = fixture();
    for (role, content) in [
        ("user", "Remember my preference"),
        ("assistant", "Answer complete"),
    ] {
        app.session.entries.push(Entry::new(ChatMessage {
            role: role.into(),
            content: content.into(),
            ..Default::default()
        }));
    }
    app.chat_tx
        .send(ChatEvent::BackgroundFailure {
            task: "memory".into(),
            reason: "deadline exceeded".into(),
        })
        .unwrap();
    app.receive_events();
    let entry = app.session.entries.last().unwrap();
    assert_eq!(entry.message.content, "Answer complete");
    assert!(entry.stats.get("interruption_kind").is_none());
    assert_eq!(entry.stats["background_failures"][0]["task"], "memory");
    app.store.save(&mut app.session).unwrap();
    let report =
        std::fs::read_to_string(crate::conversation_diagnostics::log_path(dir.path())).unwrap();
    assert!(report.contains("background_task_failed"));
    assert!(report.contains("deadline exceeded"));
}

#[test]
fn background_failure_does_not_attach_to_a_previous_turn() {
    let (_dir, _ctx, mut app) = fixture();
    for role in ["assistant", "user"] {
        app.session.entries.push(Entry::new(ChatMessage {
            role: role.into(),
            content: "text".into(),
            ..Default::default()
        }));
    }
    app.chat_tx
        .send(ChatEvent::BackgroundFailure {
            task: "memory".into(),
            reason: "deadline exceeded".into(),
        })
        .unwrap();
    app.receive_events();
    assert!(app
        .session
        .entries
        .iter()
        .all(|e| e.stats.get("background_failures").is_none()));
}

#[test]
fn demo_model_actions_never_start_a_worker_or_contact_ollama() {
    let (_dir, _ctx, mut app) = fixture();
    let mock = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    mock.set_nonblocking(true).unwrap();
    app.settings.base_url = format!("http://{}", mock.local_addr().unwrap());
    let original = app.session.id.clone();
    app.refresh_models();
    for action in ["show", "pull", "unload", "delete", "copy", "create", "import"] {
        app.model_action(action, "demo-model");
    }
    app.compare_models = "first,second".into();
    app.start_comparison();
    app.regenerate();
    app.launch_chat(Vec::new(), "demo-model".into());
    assert!(!app.busy && !app.work_busy);
    assert_eq!(app.active_work, 0);
    assert_eq!(app.session.id, original);
    assert_eq!(mock.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock);
}

#[test]
fn importing_a_stored_conversation_keeps_both_copies() {
    let (dir, _ctx, mut app) = fixture();
    app.demo = false;
    app.session = Session::default();
    app.session.title = "Original".into();
    app.persist();
    let original_id = app.session.id.clone();
    let imported = Session {
        id: original_id.clone(),
        title: "Imported".into(),
        ..Default::default()
    };
    app.work_tx.send(WorkEvent::Import(imported)).unwrap();
    app.receive_events();
    assert!(app.error.is_none(), "{:?}", app.error);
    assert_ne!(app.session.id, original_id);
    let other = Store::open(dir.path()).unwrap();
    assert_eq!(other.load(&original_id).unwrap().title, "Original");
    assert_eq!(other.load(&app.session.id).unwrap().title, "Imported");
}

#[test]
fn repeated_conflicts_do_not_stack_the_recovered_suffix() {
    let (dir, _ctx, mut app) = fixture();
    app.demo = false;
    app.session = Session::default();
    app.session.title = "Notes".into();
    app.persist();
    let other = Store::open(dir.path()).unwrap();
    for round in 0..2 {
        let mut newer = other.load(&app.session.id).unwrap();
        newer.title = format!("Other window {round}");
        other.save(&mut newer).unwrap();
        app.session.entries.push(Entry::new(ChatMessage {
            role: "user".into(),
            content: format!("edit {round}"),
            ..Default::default()
        }));
        app.persist();
    }
    assert_eq!(app.session.title, "Notes · recovered edits");
}

fn click(pos: egui::Pos2, pressed: bool) -> egui::Event {
    egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    }
}

fn enter(modifiers: egui::Modifiers) -> egui::Event {
    egui::Event::Key {
        key: egui::Key::Enter,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers,
    }
}

/// Focus the composer by clicking it, type `text`, then press Enter with `modifiers`.
fn type_and_press_enter(enter_sends: bool, modifiers: egui::Modifiers) -> Workbench {
    let (_dir, ctx, mut app) = fixture();
    let size = [1366.0, 768.0];
    app.page = Page::Chat;
    app.settings.enter_sends = enter_sends;
    let output = frame(&mut app, &ctx, size, vec![]);
    let field = visible_label(&output, "Type a message...", size).center();
    frame(&mut app, &ctx, size, vec![egui::Event::PointerMoved(field), click(field, true)]);
    frame(&mut app, &ctx, size, vec![click(field, false)]);
    frame(&mut app, &ctx, size, vec![egui::Event::Text("hello".into())]);
    // Like a real window: the held modifiers are set globally and on the key event.
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(size[0], size[1]))),
            modifiers,
            events: vec![enter(modifiers)],
            ..Default::default()
        },
        |ctx| app.render(ctx),
    );
    frame(&mut app, &ctx, size, vec![]);
    app
}

#[test]
fn enter_sends_only_when_chosen_and_shift_enter_adds_a_line() {
    // Demo mode refuses to contact Ollama with this error, which shows a send was attempted.
    let sent = |app: &Workbench| app.error.as_deref().is_some_and(|e| e.contains("--demo"));

    let app = type_and_press_enter(true, egui::Modifiers::NONE);
    assert!(sent(&app), "Enter should send: {:?}", app.error);
    assert_eq!(app.input, "hello");

    let app = type_and_press_enter(true, egui::Modifiers::SHIFT);
    assert!(!sent(&app));
    assert_eq!(app.input, "hello\n");

    let app = type_and_press_enter(false, egui::Modifiers::NONE);
    assert!(!sent(&app));
    assert_eq!(app.input, "hello\n");

    let app = type_and_press_enter(false, egui::Modifiers::CTRL);
    assert!(sent(&app), "Ctrl+Enter always sends: {:?}", app.error);
}

#[test]
fn answer_feedback_toggles_and_is_saved() {
    let (dir, _ctx, mut app) = fixture();
    app.demo = false;
    app.session = Session::default();
    app.session.entries.push(Entry::new(ChatMessage { role: "user".into(), content: "Q".into(), ..Default::default() }));
    app.session.entries.push(Entry::new(ChatMessage { role: "assistant".into(), content: "A".into(), ..Default::default() }));
    let user_id = app.session.entries[0].id.clone();
    let answer_id = app.session.entries[1].id.clone();
    app.set_feedback(&answer_id, "up");
    assert_eq!(chat::entry_feedback(&app.session.entries[1]), Some("up"));
    app.set_feedback(&answer_id, "down");
    let saved = Store::open(dir.path()).unwrap().load(&app.session.id).unwrap();
    assert_eq!(chat::entry_feedback(&saved.entries[1]), Some("down"));
    app.set_feedback(&answer_id, "down");
    assert_eq!(chat::entry_feedback(&app.session.entries[1]), None);
    // Only answers can be rated.
    app.set_feedback(&user_id, "up");
    assert_eq!(chat::entry_feedback(&app.session.entries[0]), None);
}

#[test]
fn model_replies_are_cleaned_into_short_titles() {
    use conversations::clean_title;
    assert_eq!(clean_title("\"Rust Error Handling Basics.\"").as_deref(), Some("Rust Error Handling Basics"));
    assert_eq!(clean_title("Title: Baking Sourdough Bread").as_deref(), Some("Baking Sourdough Bread"));
    assert_eq!(clean_title("<think>The user asks about...</think>\n\n**Linux Disk Cleanup**\nMore text").as_deref(), Some("Linux Disk Cleanup"));
    assert_eq!(clean_title("# Planning a Garden").as_deref(), Some("Planning a Garden"));
    assert!(clean_title("   \n  ").is_none());
    assert!(clean_title("<think>only thinking</think>").is_none());
    let long = clean_title(&"word ".repeat(40)).unwrap();
    assert!(long.chars().count() <= 60 && !long.ends_with(' '));
}

#[test]
fn a_generated_title_never_replaces_a_rename() {
    let (dir, _ctx, mut app) = fixture();
    app.demo = false;
    app.session = Session { title: "How do I fix this borrow checker error".into(), ..Default::default() };
    app.persist();
    let id = app.session.id.clone();
    app.session.title = "My rename".into();
    app.apply_generated_title(&id, "How do I fix this borrow checker error", "Borrow Checker Fix".into());
    assert_eq!(app.session.title, "My rename");
    app.session.title = "How do I fix this borrow checker error".into();
    app.apply_generated_title(&id, "How do I fix this borrow checker error", "Borrow Checker Fix".into());
    assert_eq!(app.session.title, "Borrow Checker Fix");
    // A conversation the user has left is updated in storage.
    let mut other = Session { title: "Old automatic title".into(), ..Default::default() };
    app.store.save(&mut other).unwrap();
    app.apply_generated_title(&other.id, "Old automatic title", "New Title".into());
    assert_eq!(Store::open(dir.path()).unwrap().load(&other.id).unwrap().title, "New Title");
}

#[test]
fn the_first_answer_gets_a_title_from_the_model_that_wrote_it() {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = Vec::new();
        let mut buffer = [0u8; 4096];
        loop {
            let n = stream.read(&mut buffer).unwrap();
            request.extend_from_slice(&buffer[..n]);
            let text = String::from_utf8_lossy(&request);
            if let Some(head_end) = text.find("\r\n\r\n") {
                let length = text[..head_end]
                    .lines()
                    .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap()))
                    .unwrap_or(0);
                if request.len() >= head_end + 4 + length {
                    break;
                }
            }
        }
        let body = "{\"message\":{\"role\":\"assistant\",\"content\":\"\\\"Sourdough Starter Care.\\\"\"},\"done\":false}\n{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true}\n";
        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
        String::from_utf8_lossy(&request).to_string()
    });
    let (_dir, _ctx, mut app) = fixture();
    app.demo = false;
    app.settings.base_url = format!("http://{address}");
    let question = "How often should I feed a sourdough starter that lives in the fridge?";
    app.session = Session { title: question.chars().take(52).collect(), ..Default::default() };
    let mut user = Entry::new(ChatMessage { role: "user".into(), content: question.into(), ..Default::default() });
    user.display = Some(question.into());
    let mut answer = Entry::new(ChatMessage { role: "assistant".into(), content: "Once a week.".into(), ..Default::default() });
    answer.model = "writer-model".into();
    app.session.entries = vec![user, answer];
    app.generate_title();
    let request = server.join().unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while app.session.title != "Sourdough Starter Care" && std::time::Instant::now() < deadline {
        app.receive_events();
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert_eq!(app.session.title, "Sourdough Starter Care");
    assert!(request.contains("\"model\":\"writer-model\""), "{request}");
    assert!(request.contains("\"think\":false"), "{request}");
    assert!(!request.contains("\"tools\""), "{request}");
}

#[test]
fn titles_are_not_generated_when_off_renamed_or_after_later_turns() {
    let (_dir, _ctx, mut app) = fixture();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    app.demo = false;
    app.settings.base_url = format!("http://{}", listener.local_addr().unwrap());
    let user = |text: &str| Entry::new(ChatMessage { role: "user".into(), content: text.into(), ..Default::default() });
    let answer = || { let mut e = Entry::new(ChatMessage { role: "assistant".into(), content: "A".into(), ..Default::default() }); e.model = "m".into(); e };
    app.session = Session { title: "Question".into(), entries: vec![user("Question"), answer()], ..Default::default() };
    app.settings.auto_titles = false;
    app.generate_title();
    app.settings.auto_titles = true;
    app.session.title = "Renamed".into();
    app.generate_title();
    app.session.title = "Question".into();
    app.session.entries.extend([user("Second"), answer()]);
    app.generate_title();
    std::thread::sleep(std::time::Duration::from_millis(200));
    assert_eq!(listener.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock);
}

#[test]
fn formulas_in_answers_are_shown_as_readable_math() {
    let (_dir, ctx, mut app) = fixture();
    let size = [1366.0, 768.0];
    app.page = Page::Chat;
    app.session.entries.push(Entry::new(ChatMessage { role: "user".into(), content: "Area?".into(), ..Default::default() }));
    app.session.entries.push(Entry::new(ChatMessage {
        role: "assistant".into(),
        content: r"The area is $\pi r^2$ and it costs $5 and $10.".into(),
        ..Default::default()
    }));
    frame(&mut app, &ctx, size, vec![]);
    let output = frame(&mut app, &ctx, size, vec![]);
    visible_label(&output, "The area is π r² and it costs $5 and $10.", size);
}
