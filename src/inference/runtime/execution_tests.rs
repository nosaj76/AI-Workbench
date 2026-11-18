//! Tests for the chat turn loop and its helpers.

use super::*;

#[cfg(test)]
mod detector_tests {
    use super::*;

    #[test]
    fn a_reply_stuck_repeating_lines_is_cut_after_the_first_copy() {
        let mut text = String::from("Here is the code:\nfunction update() {\n");
        let block = "    const targetVelocity = Math.sqrt(\n        this.aircraft.position.x ** 2 +\n        this.aircraft.position.z ** 2\n    );\n";
        let first_copy_end = text.len() + block.len();
        for _ in 0..12 {
            text.push_str(block);
        }
        let cut = repetition_loop(&text).expect("loop");
        assert_eq!(cut, first_copy_end);
        assert_eq!(text[..cut].matches("targetVelocity").count(), 1);
        // Ordinary code with a few similar lines and closing braces is not a loop.
        let normal = "fn a() {\n    x();\n}\n".repeat(10);
        assert!(repetition_loop(&normal).is_none());
        let braces = "}\n".repeat(40);
        assert!(repetition_loop(&braces).is_none());
    }

    #[test]
    fn github_source_files_of_listed_repositories_may_be_fetched() {
        let listed: HashSet<String> = ["github.com/phuang17/flightsimulator".to_string()].into();
        assert!(github_file_of_listed_repo(
            "https://raw.githubusercontent.com/phuang17/FlightSimulator/master/index.html",
            &listed
        ));
        assert!(github_file_of_listed_repo(
            "https://github.com/phuang17/FlightSimulator/blob/master/airplane.js",
            &listed
        ));
        assert!(!github_file_of_listed_repo(
            "https://raw.githubusercontent.com/someone/else/main/x.js",
            &listed
        ));
        assert!(!github_file_of_listed_repo(
            "https://github.com/phuang17/FlightSimulator",
            &HashSet::new()
        ));
    }

    #[test]
    fn requests_to_build_code_are_recognised_but_questions_are_not() {
        assert!(requests_code_creation(
            "write an html game flight simulator"
        ));
        assert!(requests_code_creation(
            "Please build a python script that renames photos"
        ));
        assert!(requests_code_creation(
            "create a rust cli tool for todo lists"
        ));
        assert!(!requests_code_creation("how do I write a python script?"));
        assert!(!requests_code_creation("fix the bug in main.rs"));
        assert!(!requests_code_creation("tell me about growing ginseng"));
    }
}

#[cfg(test)]
mod notes_turn_tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
            ..Default::default()
        }
    }

    #[test]
    fn answers_from_the_model_alone_are_never_saved_as_notes() {
        let long = "Dandelions tolerate drought well because of their deep taproot. ".repeat(8);
        let first = vec![
            msg("user", "tell me about growing dandelions"),
            msg("assistant", &long),
        ];
        assert!(notes_turn(&first, "tell me about growing dandelions").is_none());
        let follow_up = vec![
            msg("user", "tell me about growing dandelions"),
            msg("assistant", &long),
            msg("user", "5 dandelions 7 day no water"),
            msg("assistant", &long),
        ];
        assert!(notes_turn(&follow_up, "5 dandelions 7 day no water").is_none());
    }
}

#[cfg(test)]
mod continuation_tests {
    use super::*;

    fn message(role: &str, content: impl Into<String>) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
            ..Default::default()
        }
    }

    // Recreated contract from the user's installation log. The unuploaded
    // tests/agent_flow.rs fixture is not reproduced or modified here.
    #[test]
    fn long_continuations_preserve_the_real_request_and_latest_unicode_tail() {
        let messages = vec![
            message("system", crate::settings::BASE_SYSTEM),
            message("user", "Earlier unrelated question"),
            message("assistant", "old context ".repeat(1000)),
            message("user", "Write the whole program"),
        ];
        let original = messages.clone();
        let mut continuation = AnswerContinuation::default();
        let mut generated = 0;
        for part in 1..=2 {
            let text = (0..450)
                .map(|line| format!("{line}: PRINT \"part {part}, value λ{line}\"\n"))
                .collect::<String>()
                + &format!("PART_{part}_TAIL_Ω_終");
            let saved_full_response = text.clone();
            generated += text.chars().count();
            continuation.observe(&text, &messages);
            let (request, _, _, _) = continuation.fit_request(&messages, 4096, -1, 0).unwrap();
            assert!(request
                .iter()
                .any(|message| message.role == "user"
                    && message.content == "Write the whole program"));
            assert_eq!(
                request
                    .iter()
                    .filter(|message| message.role == "user")
                    .count(),
                1
            );
            let tail = request
                .iter()
                .rev()
                .find(|message| message.role == "assistant")
                .unwrap();
            assert!(tail.content.ends_with(&format!("PART_{part}_TAIL_Ω_終")));
            assert!(
                tail.content.chars().count() < CONTINUE_TAIL_CHARS,
                "The fixed 6,000-character tail must shrink to fit this context"
            );
            assert!(tail.content.starts_with("[The first "));
            assert!(tail
                .content
                .contains("saved but left out here to fit the context. It ends with:]\n"));
            assert!(continuation.tail.chars().count() <= CONTINUE_TAIL_CHARS);
            assert_eq!(continuation.generated_characters, generated);
            assert_eq!(text, saved_full_response);
            assert_eq!(
                messages, original,
                "Continuation must not mutate the original messages"
            );
            assert_eq!(
                request
                    .iter()
                    .filter(|message| message.content == CONTINUE_INSTRUCTION)
                    .count(),
                1
            );
            assert!(!messages
                .iter()
                .any(|message| message.content == CONTINUE_INSTRUCTION));
        }
    }

    #[test]
    fn supplied_2000_line_continuation_contract_keeps_compatible_notice_and_end() {
        let messages = vec![
            message("system", crate::settings::BASE_SYSTEM),
            message("user", "Write the whole program"),
        ];
        let original_messages = messages.clone();
        let partial: String = (0..2000)
            .map(|n| format!("line {n} of the program\n"))
            .collect();
        let full_saved_partial = partial.clone();
        let mut continuation = AnswerContinuation::default();
        continuation.observe(&partial, &messages);
        let (request, _, _, _) = continuation.fit_request(&messages, 4096, -1, 0).unwrap();
        assert!(request
            .iter()
            .any(|message| message.role == "user" && message.content == "Write the whole program"));
        let text = &request
            .iter()
            .rev()
            .find(|message| message.role == "assistant")
            .unwrap()
            .content;
        // These are the exact compatibility conditions supplied from the
        // original integration test, including its last program line.
        assert!(text.starts_with("[The first "), "{}", &text[..80]);
        assert!(text.ends_with("line 1999 of the program\n"));
        assert!(text.chars().count() < CONTINUE_TAIL_CHARS);
        assert_eq!(partial, full_saved_partial);
        assert_eq!(messages, original_messages);
        assert_eq!(
            request
                .iter()
                .filter(|message| message.role == "user")
                .count(),
            1
        );
    }

    #[test]
    fn continuation_stays_before_subsequent_native_tool_exchanges() {
        let mut messages = vec![
            message("system", crate::settings::BASE_SYSTEM),
            message("user", "Write the whole program"),
        ];
        let mut continuation = AnswerContinuation::default();
        continuation.observe("10 PRINT \"unfinished", &messages);
        messages.push(ChatMessage {
            role: "assistant".into(),
            tool_calls: vec![crate::ollama::ToolCall {
                function: crate::ollama::FunctionCall {
                    name: "read_file".into(),
                    arguments: serde_json::json!({"path":"notes.txt"}),
                },
            }],
            ..Default::default()
        });
        let evidence = ChatMessage { role:"tool".into(), tool_name:Some("read_file".into()), content:serde_json::json!({"path":"notes.txt","content":"actual reference text","complete":true}).to_string(), ..Default::default() };
        messages.push(evidence.clone());
        let original = messages.clone();
        let (request, _, _, _) = continuation.fit_request(&messages, 8192, -1, 0).unwrap();
        let tail_index = request
            .iter()
            .position(|message| message.content == "10 PRINT \"unfinished")
            .unwrap();
        let tool_call_index = request
            .iter()
            .position(|message| !message.tool_calls.is_empty())
            .unwrap();
        let evidence_index = request
            .iter()
            .position(|message| message == &evidence)
            .unwrap();
        assert!(tail_index < tool_call_index && tool_call_index < evidence_index);
        assert_eq!(messages, original);
        assert_eq!(
            request
                .iter()
                .filter(|message| message.role == "user")
                .count(),
            1
        );
    }

    #[test]
    fn continuation_never_discards_an_oversized_real_user_to_fit_its_own_prompt() {
        let messages = vec![
            message("system", crate::settings::BASE_SYSTEM),
            message("user", "Required program details. ".repeat(1000)),
        ];
        let original = messages.clone();
        let mut continuation = AnswerContinuation::default();
        continuation.observe("Some partial answer", &messages);
        assert!(continuation.fit_request(&messages, 4096, -1, 0).is_err());
        assert_eq!(messages, original);
    }
}

#[cfg(test)]
mod terminal_guidance_tests {
    use super::*;
    use serde_json::json;

    fn message(role: &str, content: impl Into<String>) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
            ..Default::default()
        }
    }

    fn default_context(user: &str) -> (Settings, Vec<Value>, Vec<ChatMessage>) {
        let settings = Settings::default();
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("README.txt"), "BASIC exercises").unwrap();
        let engine = configured_engine(
            &settings,
            crate::files::Workspace::new(directory.path().to_path_buf()).unwrap(),
        )
        .unwrap();
        let mut definitions = engine.definitions();
        if settings.agents_enabled {
            definitions.push(delegation_tool_definition());
        }
        let history = vec![
            message("system", settings.system_prompt.clone()),
            message(
                "system",
                context::project_orientation(directory.path(), settings.num_ctx).unwrap(),
            ),
            message("user", user),
        ];
        (settings, definitions, history)
    }

    #[test]
    fn ordinary_file_work_has_no_terminal_guidance_overhead() {
        let user = "Change note.txt";
        let (settings, definitions, mut history) = default_context(user);
        let original = history.clone();
        assert!(add_terminal_guidance(
            &mut history,
            &definitions,
            terminal_action_kind(user),
            settings.num_ctx,
            settings.num_predict,
        )
        .is_none());
        assert_eq!(history, original);
    }

    #[test]
    fn bash_only_countermand_and_how_to_question_do_not_get_launch_instructions() {
        let user = "Open Bash but do not run modernbasic";
        let (settings, definitions, mut history) = default_context(user);
        let guide = add_terminal_guidance(
            &mut history,
            &definitions,
            terminal_action_kind(user),
            settings.num_ctx,
            settings.num_predict,
        );
        assert!(matches!(
            guide,
            Some(GENERAL_TERMINAL_GUIDANCE | COMPACT_GENERAL_TERMINAL_GUIDANCE)
        ));
        assert!(!guide.unwrap().contains("modernbasic"));
        let user = "How do I use modern basic?";
        let (settings, definitions, mut history) = default_context(user);
        let original = history.clone();
        assert!(add_terminal_guidance(
            &mut history,
            &definitions,
            terminal_action_kind(user),
            settings.num_ctx,
            settings.num_predict
        )
        .is_none());
        assert_eq!(history, original);
    }

    #[test]
    fn default_context_keeps_basic_workflow_evidence_ahead_of_optional_guidance() {
        let user = "Open Bash, run modernbasic, type help, then write, run and save a small BASIC program.";
        let (settings, definitions, base_history) = default_context(user);
        let schema_chars = serde_json::to_string(&definitions).unwrap().chars().count();
        let id = "07f20b39-341d-4608-89e0-35dc61638026";
        let source = "10 PRINT \"HELLO\"\n20 END\n";
        let steps = [
            ("terminal_open", json!({}), "bash$ ".to_owned()),
            ("terminal_write", json!({"session_id":id,"text":"modernbasic"}), "ModernBASIC\nREADY> ".to_owned()),
            ("terminal_write", json!({"session_id":id,"text":"help"}), format!("HELP\n{}\nREADY> ", "PRINT displays values. LIST shows program lines. RUN executes the program. SAVE writes a BASIC source file. LOAD reads a saved program. BYE returns to the shell.\n".repeat(8))),
            ("terminal_write", json!({"session_id":id,"text":source}), format!("{source}READY> ")),
            ("terminal_write", json!({"session_id":id,"text":"LIST\nRUN"}), format!("{source}HELLO\nREADY> ")),
            ("terminal_write", json!({"session_id":id,"text":"SAVE \"hello.bas\""}), "Saved hello.bas\nREADY> ".to_owned()),
            ("read_file", json!({"path":"hello.bas"}), source.to_owned()),
        ];
        for bounded_ingestion in [true, false] {
            let mut history = base_history.clone();
            let mut cursor = 0u64;
            for round in 0..=steps.len() {
                // This is the production order: fit actual conversation evidence,
                // then add optional coaching only to the outgoing request copy.
                let (mut request, _, _, _) = context::fit_context_for_tool_round(
                &history,
                settings.num_ctx,
                settings.num_predict,
                schema_chars,
            )
            .unwrap_or_else(|error| panic!("BASIC workflow round {round}, bounded_ingestion={bounded_ingestion}: {error}"));
                let selected_evidence = request.clone();
                let guide = add_terminal_guidance(
                    &mut request,
                    &definitions,
                    terminal_action_kind(user),
                    settings.num_ctx,
                    settings.num_predict,
                );
                if round == 0 {
                    assert!(
                        guide.is_some(),
                        "Default context must retain basic interactive guidance"
                    );
                }
                let (_, removed) = context::fit_context_reserved(
                    &request,
                    settings.num_ctx,
                    settings.num_predict,
                    schema_chars,
                )
                .unwrap();
                assert_eq!(removed, 0, "Guidance must not evict selected evidence");
                request.retain(|message| {
                    !matches!(
                        message.content.as_str(),
                        INTERACTIVE_TERMINAL_GUIDANCE | COMPACT_TERMINAL_GUIDANCE
                    )
                });
                assert_eq!(request, selected_evidence);
                assert!(!history.iter().any(|message| matches!(
                    message.content.as_str(),
                    INTERACTIVE_TERMINAL_GUIDANCE | COMPACT_TERMINAL_GUIDANCE
                )));
                if round == steps.len() {
                    break;
                }
                let (name, arguments, output) = &steps[round];
                history.push(ChatMessage {
                    role: "assistant".into(),
                    tool_calls: vec![crate::ollama::ToolCall {
                        function: crate::ollama::FunctionCall {
                            name: (*name).into(),
                            arguments: arguments.clone(),
                        },
                    }],
                    ..Default::default()
                });
                let result = if *name == "read_file" {
                    json!({"path":"hello.bas","content":output,"complete":true})
                } else {
                    let start = cursor;
                    cursor += output.len() as u64;
                    // Representative real PTY fields and help volume. Native
                    // process/tool transport is covered by its separate fixture.
                    json!({"ok":true,"session_id":id,"summary":{"id":id,"title":"AI Bash","initial_directory":"/workspace","pid":1234,"alive":true,"exit_code":null},"output":output,"screen":output,"start_cursor":start,"next_cursor":cursor,"bytes_received":cursor,"has_more":false,"output_lossy":false,"output_closed":false,"screen_truncated":false,"dropped_transcript_bytes":0,"lost_bytes":0,"truncated":false,"cancelled":false,"timed_out":false,"rows":28,"cols":100,"cursor":[1,0],"cursor_hidden":false,"alternate_screen":false,"error":null,"input_sent":*name=="terminal_write","action_applied":true,"completion_known":false})
                };
                // Exercise both complete saved transcripts and the production
                // in-flight projection, which keeps the full result in storage
                // while showing only its bounded model view in this turn.
                let result = if bounded_ingestion {
                    context::bounded_tool_result(
                        &result,
                        context::tool_result_room(
                            &history,
                            settings.num_ctx,
                            settings.num_predict,
                            schema_chars,
                        ),
                    )
                    .0
                } else {
                    result
                };
                let mut tool = message("tool", result.to_string());
                tool.tool_name = Some((*name).into());
                history.push(tool);
            }
        }
    }

    #[test]
    fn optional_terminal_guidance_shrinks_then_yields_to_current_evidence() {
        let user = "Continue in modernbasic";
        let (settings, definitions, base) = default_context(user);
        let schemas = serde_json::to_string(&definitions).unwrap().chars().count();
        let mut seen = HashSet::new();
        for bytes in (0..12_000).step_by(100) {
            let mut history = base.clone();
            history.push(message("tool", "HELP evidence ".repeat(bytes / 14)));
            if context::fit_context_reserved(
                &history,
                settings.num_ctx,
                settings.num_predict,
                schemas,
            )
            .is_err()
            {
                break;
            }
            let before = history.clone();
            let guide = add_terminal_guidance(
                &mut history,
                &definitions,
                terminal_action_kind(user),
                settings.num_ctx,
                settings.num_predict,
            );
            seen.insert(guide);
            let (_, removed) = context::fit_context_reserved(
                &history,
                settings.num_ctx,
                settings.num_predict,
                schemas,
            )
            .unwrap();
            assert_eq!(removed, 0);
            history.retain(|message| {
                !matches!(
                    message.content.as_str(),
                    INTERACTIVE_TERMINAL_GUIDANCE | COMPACT_TERMINAL_GUIDANCE
                )
            });
            assert_eq!(history, before);
        }
        assert!(seen.contains(&Some(INTERACTIVE_TERMINAL_GUIDANCE)));
        assert!(seen.contains(&Some(COMPACT_TERMINAL_GUIDANCE)));
        assert!(seen.contains(&None));
    }

    #[test]
    fn terminal_history_recognizes_live_cancelled_reads_and_honors_close() {
        let tool = |name: &str, value: Value| ChatMessage {
            role: "tool".into(),
            tool_name: Some(name.into()),
            content: value.to_string(),
            ..Default::default()
        };
        let mut history = vec![tool(
            "terminal_open",
            json!({"session_id":"a","ok":true,"summary":{"alive":true}}),
        )];
        assert!(has_terminal_session_history(&history));
        history.push(tool(
            "terminal_read",
            json!({"session_id":"a","ok":false,"cancelled":true,"summary":{"alive":true}}),
        ));
        assert!(has_terminal_session_history(&history));
        history.push(tool(
            "terminal_close",
            json!({"session_id":"a","ok":true,"closed":true}),
        ));
        assert!(!has_terminal_session_history(&history));
        history.push(tool(
            "terminal_list",
            json!({"sessions":[{"id":"b","alive":true}]}),
        ));
        assert!(has_terminal_session_history(&history));
        history.push(tool(
            "terminal_read",
            json!({"session_id":"b","ok":true,"summary":{"alive":false}}),
        ));
        assert!(!has_terminal_session_history(&history));
        assert!(!has_terminal_session_history(&[message(
            "user",
            "terminal_write session_id=a"
        )]));
    }
}

#[cfg(test)]
mod finalization_tests {
    #[test]
    fn folders_named_in_a_message_are_found_only_when_they_exist() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().display().to_string();
        let found =
            super::local_dirs_in(&format!("please look at {path}, and /no/such/folder too"));
        assert_eq!(found, [path]);
        assert!(super::local_dirs_in("nothing here, https://example.com/x").is_empty());
    }

    use super::*;

    #[test]
    fn fetchable_urls_match_loosely_and_come_from_text() {
        assert_eq!(
            comparable_url("https://www.Example.org/a/#top"),
            comparable_url("http://example.org/a")
        );
        assert_ne!(
            comparable_url("https://example.org/a"),
            comparable_url("https://example.org/b")
        );
        assert_eq!(
            urls_in_text("Read (https://example.org/x), then http://a.b/c."),
            vec!["https://example.org/x", "http://a.b/c"]
        );
    }

    #[test]
    fn tool_call_summary_is_short_and_readable() {
        let command = serde_json::json!({"program":"cargo","args":["test","--offline"]});
        assert_eq!(tool_call_summary(&command), "$ cargo test --offline");
        assert_eq!(
            tool_call_summary(&serde_json::json!({"path":"src/main.rs"})),
            "src/main.rs"
        );
        assert_eq!(tool_call_summary(&serde_json::json!({})), "");
        let long = serde_json::json!({"query": "x".repeat(500)});
        assert!(tool_call_summary(&long).chars().count() <= 161);
        assert_eq!(
            tool_result_preview(&serde_json::json!({"error":"No such file"})),
            "error: No such file"
        );
    }

    #[test]
    fn repeat_scope_separates_observations_actions_and_expensive_requests() {
        assert_eq!(repeat_scope("read_file"), RepeatScope::Observation);
        assert_eq!(repeat_scope("project_tree"), RepeatScope::Observation);
        assert_eq!(repeat_scope("write_file"), RepeatScope::StateChanging);
        assert_eq!(repeat_scope("run_command"), RepeatScope::StateChanging);
        assert_eq!(repeat_scope("fetch_url"), RepeatScope::Persistent);
        assert_eq!(repeat_scope("web_search"), RepeatScope::Persistent);
        assert_eq!(repeat_scope("delegate_tasks"), RepeatScope::Persistent);
        assert_eq!(repeat_scope("terminal_read"), RepeatScope::Observation);
        assert_eq!(repeat_scope("terminal_write"), RepeatScope::StateChanging);
        assert_eq!(repeat_scope("terminal_open"), RepeatScope::StateChanging);
    }

    #[test]
    fn terminal_polling_and_successive_identical_input_are_not_duplicates() {
        let mut seen = HashSet::new();
        let args = serde_json::json!({"session_id":"test", "text":"1"});
        for name in [
            "terminal_read",
            "terminal_list",
            "terminal_write",
            "terminal_key",
            "terminal_resize",
        ] {
            for _ in 0..3 {
                assert!(!duplicate_tool_request(
                    &mut seen,
                    name,
                    &args,
                    0,
                    repeat_scope(name)
                ));
            }
        }
        let open = serde_json::json!({});
        assert!(!duplicate_tool_request(
            &mut seen,
            "terminal_open",
            &open,
            0,
            repeat_scope("terminal_open")
        ));
        assert!(duplicate_tool_request(
            &mut seen,
            "terminal_open",
            &open,
            0,
            repeat_scope("terminal_open")
        ));
    }

    #[test]
    fn interactive_cli_requests_use_installed_help_before_generic_online_examples() {
        assert!(requests_interactive_cli(
            "Open Bash, type modernbasic, then help and write a program"
        ));
        assert!(requests_interactive_cli("Use the Python REPL"));
        assert!(requests_interactive_cli(
            "Launch Bash and run the installed interpreter"
        ));
        assert!(!requests_interactive_cli(
            "Write a Rust function to replace text"
        ));
        assert!(!requests_interactive_cli(
            "Write a Bash script to sort files"
        ));
    }

    #[test]
    fn explicit_web_search_intent_is_narrow_and_honors_opt_outs() {
        assert!(explicitly_requests_web_search("Search the web about Linux"));
        assert!(explicitly_requests_web_search(
            "Please browse online for current sources"
        ));
        assert!(!explicitly_requests_web_search(
            "Explain Linux from your own knowledge"
        ));
        assert!(!explicitly_requests_web_search(
            "Do not search the web; answer offline only"
        ));
        assert!(explicitly_requests_web_search("web search ai"));
        assert!(explicitly_requests_web_search(
            "Could you please search the web about Linux?"
        ));
        assert!(!explicitly_requests_web_search(
            "Explain how web search works"
        ));
        assert!(!explicitly_requests_web_search(
            "Fix the web search implementation"
        ));
        assert!(!explicitly_requests_web_search(
            "Review this file\n\n<local_file source=\"x\">web search ai</local_file>"
        ));
        assert!(!explicitly_requests_web_search("web searching is useful"));
    }

    #[test]
    fn duplicate_tool_keys_ignore_object_key_order_but_preserve_real_changes() {
        let first = serde_json::json!({"path":"src/lib.rs","start_line":1,"end_line":80});
        let reordered = serde_json::json!({"end_line":80,"start_line":1,"path":"src/lib.rs"});
        let different = serde_json::json!({"path":"src/lib.rs","start_line":81,"end_line":160});
        assert_eq!(
            tool_request_key("read_file", &first, 0, RepeatScope::Observation),
            tool_request_key("read_file", &reordered, 0, RepeatScope::Observation)
        );
        assert_ne!(
            tool_request_key("read_file", &first, 0, RepeatScope::Observation),
            tool_request_key("read_file", &different, 0, RepeatScope::Observation)
        );
    }

    #[test]
    fn successful_state_change_opens_a_new_observation_epoch() {
        let args = serde_json::json!({"path":"src/lib.rs"});
        assert_ne!(
            tool_request_key("read_file", &args, 0, RepeatScope::Observation),
            tool_request_key("read_file", &args, 1, RepeatScope::Observation)
        );
        assert_eq!(
            tool_request_key("fetch_url", &args, 0, RepeatScope::Persistent),
            tool_request_key("fetch_url", &args, 1, RepeatScope::Persistent)
        );
        assert!(tool_result_succeeded(&serde_json::json!({"ok":true})));
        assert!(tool_result_succeeded(
            &serde_json::json!({"ok":true,"error":null})
        ));
        assert!(!tool_result_succeeded(
            &serde_json::json!({"ok":true,"error":"PTY failed"})
        ));
        assert!(!tool_result_succeeded(&serde_json::json!({"denied":true})));
        assert!(!tool_result_succeeded(&serde_json::json!({"ok":false})));
    }

    #[test]
    fn unfinished_report_never_counts_requested_but_unread_files() {
        let messages = vec![
            ChatMessage {
                role: "tool".into(),
                tool_name: Some("read_file".into()),
                content: serde_json::json!({"path":"previous.rs", "content":"old source"})
                    .to_string(),
                ..Default::default()
            },
            ChatMessage {
                role: "user".into(),
                content: "Review my new project".into(),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".into(),
                tool_calls: vec![crate::ollama::ToolCall {
                    function: crate::ollama::FunctionCall {
                        name: "read_file".into(),
                        arguments: serde_json::json!({"path":"unread.rs"}),
                    },
                }],
                ..Default::default()
            },
            ChatMessage {
                role: "tool".into(),
                tool_name: Some("read_file".into()),
                content: serde_json::json!({"error":"Missing file", "path":"missing.rs"})
                    .to_string(),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".into(),
                tool_name: Some("read_file".into()),
                content: serde_json::json!({"path":"main.rs", "requested_start_line":1,
                    "requested_end_line":20, "content":"fn main() {}", "truncated":true})
                .to_string(),
                ..Default::default()
            },
        ];
        let report = incomplete_review_report(&messages, 12, "no final answer");
        assert!(report.contains("Review incomplete"));
        assert!(report.contains("`main.rs`"));
        assert!(report.contains("partial result"));
        assert!(!report.contains("unread.rs"));
        assert!(!report.contains("missing.rs"));
        assert!(!report.contains("previous.rs"));
        assert!(report.contains("1 successful tool result"));
    }

    #[test]
    fn unfinished_report_does_not_count_duplicate_requests_as_successes() {
        let messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "Review this".into(),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".into(),
                tool_name: Some("read_file".into()),
                content: serde_json::json!({
                    "duplicate": true,
                    "executed": false,
                    "tool": "read_file"
                })
                .to_string(),
                ..Default::default()
            },
        ];
        let report = incomplete_review_report(&messages, 12, "repeated request loop");
        assert!(report.contains("0 successful tool result"));
        assert!(report.contains("1 duplicate/skipped request"));
        assert!(report.contains("None recorded"));
    }
}

#[cfg(test)]
mod basic_validation_tests {
    use super::*;
    #[test]
    fn check_evidence_requires_process_output_and_success() {
        let good = serde_json::json!({"ok":true,"output":"PROGRAM STRUCTURE OK.\nREADY."});
        assert!(basic_structure_checked("terminal_read", &good));
        assert!(!basic_structure_checked("read_file", &good));
        assert!(!basic_structure_checked(
            "terminal_read",
            &serde_json::json!({"ok":true,"screen":"PROGRAM STRUCTURE OK."})
        ));
        assert!(!basic_structure_checked(
            "terminal_read",
            &serde_json::json!({"ok":false,"output":"PROGRAM STRUCTURE OK."})
        ));
        assert!(!basic_structure_checked(
            "terminal_write",
            &serde_json::json!({"ok":true,"output":"PRINT \"PROGRAM STRUCTURE OK.\""})
        ));
    }
}

#[cfg(test)]
mod boundary_failure_tests {
    use super::*;
    #[test]
    fn only_actionable_boundary_failures_are_grouped() {
        assert_eq!(
            boundary_failure_class(
                &serde_json::json!({"error":"That absolute path is outside the connected project folder (a)"})
            ),
            Some("workspace")
        );
        assert_eq!(
            boundary_failure_class(
                &serde_json::json!({"error":"The interpreter is repeated in args"})
            ),
            Some("invocation")
        );
        assert_eq!(
            boundary_failure_class(
                &serde_json::json!({"ok":false,"result":{"exit_code":1,"stderr":"test failed"}})
            ),
            None
        );
        assert_eq!(
            boundary_failure_class(&serde_json::json!({"denied":true})),
            None
        );
    }
}

#[cfg(test)]
mod tool_scope_tests {
    use super::*;

    fn names(definitions: &[Value]) -> Vec<String> {
        definitions
            .iter()
            .map(|d| d["function"]["name"].as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn ordinary_turns_drop_terminal_adapter_and_git_schemas() {
        let settings = Settings::default();
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir(directory.path().join(".git")).unwrap();
        let engine = configured_engine(
            &settings,
            crate::files::Workspace::new(directory.path().to_path_buf()).unwrap(),
        )
        .unwrap();
        let all = engine.definitions();
        let size = |d: &[Value]| serde_json::to_string(d).unwrap().chars().count();

        let mut plain = all.clone();
        scope_tools_to_turn(&mut plain, "Fix the off-by-one bug in src/lib.rs", false);
        let kept = names(&plain);
        assert!(kept.contains(&"read_file".to_string()));
        assert!(kept.contains(&"replace_text".to_string()));
        assert!(!kept
            .iter()
            .any(|n| n.starts_with("terminal_") || n.starts_with("git_")));
        assert!(
            size(&all) - size(&plain) > 3_000,
            "saved {}",
            size(&all) - size(&plain)
        );

        let mut terminal = all.clone();
        scope_tools_to_turn(&mut terminal, "open bash and run the tests", false);
        assert!(names(&terminal).contains(&"terminal_write".to_string()));
        let mut resumed = all.clone();
        scope_tools_to_turn(&mut resumed, "now try again", true);
        assert!(names(&resumed).contains(&"terminal_read".to_string()));

        let mut git = all.clone();
        scope_tools_to_turn(&mut git, "Commit my changes on a new branch", false);
        assert!(names(&git).contains(&"git_commit".to_string()));
        assert!(!names(&git).contains(&"terminal_open".to_string()));
    }
}
