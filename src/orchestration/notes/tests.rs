//! Tests for the notes store.

use super::*;

fn fact(text: &str, basis: &str, ix: &[usize]) -> IncomingFact {
    IncomingFact {
        text: text.into(),
        basis: basis.into(),
        source_ix: ix.to_vec(),
    }
}
fn web(url: &str) -> IncomingSource {
    IncomingSource {
        kind: "web".into(),
        label: "A page".into(),
        url: url.into(),
    }
}

#[test]
fn a_topic_gets_one_readable_file_that_round_trips_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let store = NoteStore::open(dir.path()).unwrap();
    let out = store
        .absorb(
            "Linux Kernel Configuration",
            &[
                fact(
                    "make menuconfig opens an interactive kernel configuration menu.",
                    BASIS_WEB,
                    &[0],
                ),
                fact(
                    "The .config file stores the selected kernel options.",
                    BASIS_MODEL,
                    &[],
                ),
            ],
            &[web("https://kernel.org/doc")],
        )
        .unwrap();
    assert!(out.created);
    assert_eq!((out.added, out.rejected), (2, 0));
    let text = fs::read_to_string(
        dir.path()
            .join("knowledge-notes/linux-kernel-configuration.md"),
    )
    .unwrap();
    assert!(
        text.contains("## Notes") && text.contains("## Sources") && text.contains("[S1]"),
        "{text}"
    );
    let note = store.get("linux-kernel-configuration").unwrap().unwrap();
    assert_eq!(parse_markdown(&render_markdown(&note), &note.slug), note);
    assert_eq!(note.entries[0].sources, ["S1"]);
    assert_eq!(note.entries[1].basis, BASIS_MODEL);
    assert!(dir.path().join("knowledge-notes/INDEX.md").is_file());
}

#[test]
fn repeated_facts_are_corroborated_not_duplicated_and_web_upgrades_unverified() {
    let dir = tempfile::tempdir().unwrap();
    let store = NoteStore::open(dir.path()).unwrap();
    let text = "The kernel build system reads options from the .config file.";
    store
        .absorb("Kernel Build", &[fact(text, BASIS_MODEL, &[])], &[])
        .unwrap();
    let again = store
        .absorb(
            "Kernel Build",
            &[fact(
                "The kernel build system reads options from the .config file",
                BASIS_WEB,
                &[0],
            )],
            &[web("https://example.org/a")],
        )
        .unwrap();
    assert_eq!((again.added, again.corroborated, again.upgraded), (0, 1, 1));
    let note = store.get("kernel-build").unwrap().unwrap();
    assert_eq!(note.entries.len(), 1);
    assert_eq!(note.entries[0].basis, BASIS_WEB);
    assert_eq!(note.entries[0].sources, ["S1"]);
    // A different source for the same fact adds a second citation.
    store
        .absorb(
            "Kernel Build",
            &[fact(text, BASIS_WEB, &[0])],
            &[web("https://example.org/b")],
        )
        .unwrap();
    assert_eq!(
        store.get("kernel-build").unwrap().unwrap().entries[0].sources,
        ["S1", "S2"]
    );
}

#[test]
fn near_identical_topic_titles_share_one_note_but_unrelated_ones_do_not() {
    let dir = tempfile::tempdir().unwrap();
    let store = NoteStore::open(dir.path()).unwrap();
    store
        .absorb(
            "Linux Kernel Configuration",
            &[fact(
                "Kernel options live in the .config file.",
                BASIS_WEB,
                &[],
            )],
            &[],
        )
        .unwrap();
    let same = store
        .absorb(
            "Configuring the Linux Kernel",
            &[fact(
                "Menuconfig is a text based configuration interface.",
                BASIS_WEB,
                &[],
            )],
            &[],
        )
        .unwrap();
    assert!(!same.created);
    assert_eq!(same.slug, "linux-kernel-configuration");
    let other = store
        .absorb(
            "Linux Distribution Building",
            &[fact(
                "A distribution needs a package manager and a base system.",
                BASIS_WEB,
                &[],
            )],
            &[],
        )
        .unwrap();
    assert!(other.created);
    assert_eq!(store.list().unwrap().len(), 2);
}

#[test]
fn unusable_facts_are_counted_and_an_empty_topic_leaves_no_file() {
    let dir = tempfile::tempdir().unwrap();
    let store = NoteStore::open(dir.path()).unwrap();
    let out = store
        .absorb("Nothing Useful", &[fact("too short", BASIS_WEB, &[])], &[])
        .unwrap();
    assert_eq!((out.added, out.rejected, out.created), (0, 1, false));
    assert!(store.list().unwrap().is_empty());
    assert!(!store.path("nothing-useful").exists());
    assert!(store.absorb("   ", &[], &[]).is_err());
}

#[test]
fn notes_are_searchable_and_edits_made_outside_workbench_are_picked_up() {
    let dir = tempfile::tempdir().unwrap();
    let store = NoteStore::open(dir.path()).unwrap();
    store
        .absorb(
            "Rust Ownership",
            &[fact(
                "Each value in Rust has a single owner at a time.",
                BASIS_WEB,
                &[],
            )],
            &[],
        )
        .unwrap();
    assert_eq!(
        store.search("who owns a value in rust", 5).unwrap()[0].slug,
        "rust-ownership"
    );
    assert!(store
        .search("quantum chromodynamics", 5)
        .unwrap()
        .is_empty());
    // A hand-written file with no header or tags still becomes a searchable note.
    fs::write(
        store.dir().join("zig-comptime.md"),
        "# Zig Comptime\n\n## Notes\n\n- Zig evaluates comptime blocks while compiling.\n",
    )
    .unwrap();
    assert_eq!(store.sync().unwrap(), 1);
    let hit = &store.search("zig comptime compiling", 5).unwrap()[0];
    assert_eq!((hit.title.as_str(), hit.entries.len()), ("Zig Comptime", 1));
    // Reopening in a fresh process sees the same notes without re-reading all of them.
    let reopened = NoteStore::open(dir.path()).unwrap();
    assert_eq!(reopened.list().unwrap().len(), 2);
    assert_eq!(reopened.sync().unwrap(), 0);
}

#[test]
fn deleting_moves_the_file_aside_and_removes_it_from_search() {
    let dir = tempfile::tempdir().unwrap();
    let store = NoteStore::open(dir.path()).unwrap();
    store
        .absorb(
            "Temporary Topic",
            &[fact(
                "This topic exists only for the deletion test.",
                BASIS_WEB,
                &[],
            )],
            &[],
        )
        .unwrap();
    assert!(store.delete("temporary-topic").unwrap());
    assert!(store.list().unwrap().is_empty());
    assert!(store.search("deletion test", 5).unwrap().is_empty());
    let kept = fs::read_dir(store.dir().join(".deleted")).unwrap().count();
    assert_eq!(kept, 1);
    assert!(!store.delete("temporary-topic").unwrap());
}

#[test]
fn search_listing_noise_is_not_saved_and_unused_pages_are_not_sources() {
    let dir = tempfile::tempdir().unwrap();
    let store = NoteStore::open(dir.path()).unwrap();
    let parts = [
        Part {
            text: "Jun 22, 2026 · Ginseng roots usually need five to seven years before they are harvested.",
            source: Some(0),
        },
        Part {
            text: "Discover how long ginseng takes to grow, from seed germination to root maturity.",
            source: Some(1),
        },
        Part {
            text: "The method described below requires seven years of growth before the harvesting, but produces...",
            source: Some(2),
        },
    ];
    let sources = [
        web("https://garden.example/ginseng"),
        web("https://pitch.example/guide"),
        web("https://cut.example/snippet"),
        web("https://grow.example/unrelated"),
    ];
    let facts = extract_facts(&Material {
        topic: None,
        question: "how long does ginseng take to grow",
        parts: &parts,
        basis: BASIS_WEB,
        sources: &sources,
    });
    let texts: Vec<&str> = facts.iter().map(|f| f.text.as_str()).collect();
    assert_eq!(
        texts,
        ["Ginseng roots usually need five to seven years before they are harvested."]
    );
    store.absorb("Growing Ginseng", &facts, &sources).unwrap();
    let note = store.get("growing-ginseng").unwrap().unwrap();
    let urls: Vec<&str> = note.sources.iter().map(|s| s.url.as_str()).collect();
    assert_eq!(urls, ["https://garden.example/ginseng"]);
}

#[test]
fn a_topic_title_does_not_end_on_half_a_phrase() {
    let title = topic_for(
        "research the lisp programming language, open source versions",
        &["Lisp is a programming language. Open source Lisp language versions exist."],
    )
    .unwrap();
    assert!(!title.ends_with("Open"), "{title}");
    assert!(title.starts_with("Lisp"), "{title}");
}

#[test]
fn a_follow_up_is_filed_under_the_conversation_subject() {
    let dir = tempfile::tempdir().unwrap();
    let settings = crate::settings::Settings {
        knowledge_notes: true,
        ..Default::default()
    };
    let parts = [Part {
        text: "Ginseng is grown best in well drained, slightly acidic soil under deep shade.",
        source: Some(0),
    }];
    let sources = [web("https://garden.example/soil")];
    let done = capture(
        dir.path(),
        &settings,
        &Material {
            question: "soil requirements and environment conditions",
            topic: Some("tell me about growing ginseng"),
            parts: &parts,
            basis: BASIS_WEB,
            sources: &sources,
        },
    )
    .unwrap();
    assert_eq!(done.len(), 1);
    assert!(
        done[0].title.to_lowercase().contains("ginseng"),
        "{}",
        done[0].title
    );
}

#[test]
fn a_conversation_is_not_given_its_own_answers_back_as_evidence() {
    let dir = tempfile::tempdir().unwrap();
    let store = NoteStore::open(dir.path()).unwrap();
    let own = IncomingSource {
        kind: "conversation".into(),
        label: "Conversation: tell me about dandelion watering".into(),
        url: String::new(),
    };
    store
        .absorb(
            "Dandelion Watering",
            &[
                fact(
                    "Dandelion watering needs are moderate and they tolerate drought.",
                    BASIS_MODEL,
                    &[0],
                ),
                fact(
                    "Dandelion watering guides say a deep taproot survives dry spells.",
                    BASIS_WEB,
                    &[1],
                ),
            ],
            &[own, web("https://extension.example/dandelion")],
        )
        .unwrap();
    let mine = vec!["Conversation: tell me about dandelion watering".to_string()];
    let (text, _) = store
        .context_for_excluding("dandelion watering", 4000, &mine)
        .unwrap()
        .unwrap();
    assert!(!text.contains("tolerate drought"), "{text}");
    assert!(text.contains("deep taproot"), "{text}");
    // Another conversation still gets both facts.
    let (other, _) = store
        .context_for("dandelion watering", 4000)
        .unwrap()
        .unwrap();
    assert!(other.contains("tolerate drought"), "{other}");
}

#[test]
fn chat_context_labels_basis_cites_sources_and_says_what_it_omits() {
    let dir = tempfile::tempdir().unwrap();
    let store = NoteStore::open(dir.path()).unwrap();
    let facts: Vec<IncomingFact> = (0..30)
        .map(|n| {
            fact(
                &format!("Kernel configuration fact number {n} about menuconfig options."),
                if n == 0 { BASIS_WEB } else { BASIS_MODEL },
                if n == 0 { &[0][..] } else { &[] },
            )
        })
        .collect();
    store
        .absorb(
            "Kernel Configuration",
            &facts,
            &[web("https://kernel.org/x")],
        )
        .unwrap();
    let (text, used) = store
        .context_for("kernel configuration menuconfig", 600)
        .unwrap()
        .unwrap();
    assert_eq!(used, ["Kernel Configuration"]);
    assert!(text.starts_with("<knowledge_notes>") && text.ends_with("</knowledge_notes>"));
    assert!(text.contains("model answer, unverified"));
    assert!(
        text.contains("more entries in this note are not shown"),
        "{text}"
    );
    assert!(text.len() < 1500, "{}", text.len());
    let (roomy, _) = store
        .context_for("kernel configuration menuconfig", 8000)
        .unwrap()
        .unwrap();
    assert!(roomy.contains("<https://kernel.org/x>"));
    assert!(!roomy.contains("not shown"), "{roomy}");
    assert!(store
        .context_for("unrelated gardening question", 600)
        .unwrap()
        .is_none());
}

const ANSWER: &str = "Running 32-bit code emulation on modern 64-bit CPUs is generally **not worth it** for most use cases, though there are specific scenarios where the trade-off makes sense. Here's what you should consider:

## Performance Overhead (Significant)

| Aspect | Native Execution | Emulated Execution |
|--------|------------------|--------------------|
| Speed | Baseline (100%) | 5-30x slower |

**Why the slowdown?** Emulators such as QEMU translate 32-bit instructions dynamically, which adds interpretation layers that slow compute-intensive tasks.

Most 64-bit x86 processors already run 32-bit code natively in compatibility mode, so emulation is rarely needed for ordinary programs.

## Recommendations

- Use `qemu-user` only for architectures that your CPU cannot execute natively.
- Let me know if you have a specific application in mind.

```bash
sudo apt install qemu-user
```

It also increases memory use. Would you like more detail?";

fn question() -> &'static str {
    "is it worth running 32 bit cpu emulator on 64 bit cpu?"
}

#[test]
fn the_script_keeps_essential_sentences_word_for_word_and_drops_chatter() {
    let parts = [Part {
        text: ANSWER,
        source: Some(0),
    }];
    let sources = [web("https://example.org")];
    let facts = extract_facts(&Material {
        topic: None,
        question: question(),
        parts: &parts,
        basis: BASIS_MODEL,
        sources: &sources,
    });
    let texts: Vec<&str> = facts.iter().map(|f| f.text.as_str()).collect();
    let all = texts.join("\n");
    assert!(
        texts.iter().any(|t| t.starts_with(
            "Running 32-bit code emulation on modern 64-bit CPUs is generally not worth it"
        )),
        "{all}"
    );
    assert!(
        all.contains("Emulators such as QEMU translate 32-bit instructions dynamically"),
        "{all}"
    );
    assert!(
        all.contains("Most 64-bit x86 processors already run 32-bit code natively"),
        "{all}"
    );
    // A how-to step that states something is kept; the code block under it is not.
    assert!(
        all.contains("Use `qemu-user` only for architectures"),
        "{all}"
    );
    for junk in [
        "Let me know",
        "sudo",
        "Speed",
        "Would you like",
        "It also",
        "Here's",
        "slowdown?",
    ] {
        assert!(!all.contains(junk), "{junk} leaked into: {all}");
    }
    assert!(texts.len() <= MAX_FACTS_ALONE);
    // Every fact is a sentence that really appears in the answer, with emphasis marks only removed.
    let plain = ANSWER.replace('*', "");
    assert!(texts.iter().all(|t| plain.contains(t)), "{all}");
}

#[test]
fn each_fact_is_credited_to_its_own_page_and_the_same_fact_from_two_pages_cites_both() {
    let one = "The kernel reads its build options from the .config file in the source tree.";
    let two = "Menuconfig is a text menu that edits the kernel .config file interactively.";
    let sources = [web("https://a.example/x"), web("https://b.example/y")];
    let a = format!("{one} {two}");
    let parts = [
        Part {
            text: &a,
            source: Some(0),
        },
        Part {
            text: one,
            source: Some(1),
        },
    ];
    let facts = extract_facts(&Material {
        topic: None,
        question: "how is the kernel .config file edited",
        parts: &parts,
        basis: BASIS_WEB,
        sources: &sources,
    });
    assert!(
        facts.iter().any(|f| f.text == two && f.source_ix == [0]),
        "{facts:?}"
    );
    let dir = tempfile::tempdir().unwrap();
    let store = NoteStore::open(dir.path()).unwrap();
    store.absorb("Kernel Config", &facts, &sources).unwrap();
    let note = store.get("kernel-config").unwrap().unwrap();
    let shared = note.entries.iter().find(|e| e.text == one).unwrap();
    assert_eq!(shared.sources, ["S1", "S2"], "{note:?}");
}

#[test]
fn sentences_that_cannot_stand_alone_or_are_not_knowledge_are_refused() {
    assert!(usable_sentence(
        "The borrow checker rejects programs that could alias a mutable reference.",
        false
    ));
    for bad in [
        "It prevents data races at compile time in every case.",
        "However, the compiler still checks every reference carefully.",
        "Would you like more detail about this particular setup today?",
        "What does the borrow checker actually verify in a program?",
        "Here is the list of things the compiler checks for you:",
        "You should always profile before choosing an emulator for this.",
        "Short sentence.",
        "Void Linux: Simple, no systemd by default.",
        "Interactive text-based GUI with tabs and menus",
        "Kernel Land → isolated process trees per environment today.",
        "Below is a structured guide covering the most common configurations.",
        "I\u{2019}ll focus on Debian based systems since they are widely used everywhere.",
        "| Aspect | Native Execution | Emulated Execution | Notes here |",
        "I don't have filesystem access to inspect that folder in this session.",
    ] {
        assert!(!usable_sentence(bad, false), "{bad}");
    }
}

#[test]
fn a_bullet_without_a_full_stop_counts_only_when_it_states_something() {
    assert!(usable_sentence(
        "Can save configuration to `.config` file automatically",
        true
    ));
    assert!(!usable_sentence(
        "Can save configuration to `.config` file automatically",
        false
    ));
    assert!(!usable_sentence(
        "Interactive text-based GUI with tabs and menus and shortcuts",
        true
    ));
    assert!(!usable_sentence(
        "Dependency errors: run olddefconfig after major changes",
        true
    ));
}

#[test]
fn chatter_phrases_match_whole_words_only() {
    assert!(usable_sentence(
        "Most modern CPUs have specific instructions where there are no safe equivalents.",
        false
    ));
    assert!(usable_sentence(
        "AI can classify images faster than a hand written rule set could manage.",
        false
    ));
    assert!(!usable_sentence(
        "Here are the main things that the compiler checks for each reference.",
        false
    ));
}

#[test]
fn sentence_splitting_respects_abbreviations_and_version_numbers() {
    let parts = split_sentences("Python 3.12 adds faster startup, e.g. for short scripts. The GIL is unchanged in this release! Is it?");
    assert_eq!(parts.len(), 3, "{parts:?}");
    assert!(parts[0].ends_with("short scripts."), "{parts:?}");
}

#[test]
fn topics_come_from_the_subject_words_of_the_question() {
    let title = topic_for(question(), &[ANSWER]).unwrap();
    assert!(
        title.contains("Emulator") && title.contains("Bit"),
        "{title}"
    );
    assert!(
        !title.to_lowercase().contains("worth") && title.split(' ').count() <= MAX_TOPIC_WORDS,
        "{title}"
    );
    assert!(topic_from_question("tell me how to configure linux")
        .unwrap()
        .contains("Linux"));
    // Words the text is not about are not title words, however early they come.
    let answer = "Userland is everything outside the kernel. Each user land has its own binaries and libraries, while the kernel stays shared between every user land.";
    let title = topic_for(
        "if I create three user lands from the linux kernel, which directories move?",
        &[answer],
    )
    .unwrap();
    assert!(
        title.contains("User")
            && title.contains("Kernel")
            && !title.contains("If")
            && !title.contains("Three"),
        "{title}"
    );
    let rust = "Rust's borrow checker prevents data races. The borrow checker enforces ownership, so each value has one owner. Shared references allow reads, mutable references allow writes, and shared data is never mutated through a shared reference. Data crossing threads must be Send, and data shared between threads must be Sync.";
    let title = topic_for("Explain how the Rust borrow checker prevents data races, including ownership, shared and mutable references, and the Send and Sync traits.", &[rust]).unwrap();
    assert!(
        title.contains("Rust") && title.contains("Borrow") && title.contains("Checker"),
        "{title}"
    );
    // Nothing to be about: numbers only, or only words that ask for something.
    assert_eq!(topic_from_question("?!"), None);
    assert_eq!(topic_from_question("1 and 3"), None);
    assert_eq!(topic_from_question("please explain more"), None);
}

#[test]
fn capture_builds_a_note_without_any_model_and_repeating_it_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let settings = Settings {
        knowledge_notes: true,
        ..Default::default()
    };
    let parts = [Part {
        text: ANSWER,
        source: Some(0),
    }];
    let sources = [IncomingSource {
        kind: "conversation".into(),
        label: "Conversation: emulation".into(),
        url: String::new(),
    }];
    let material = Material {
        topic: None,
        question: question(),
        parts: &parts,
        basis: BASIS_MODEL,
        sources: &sources,
    };
    let done = capture(dir.path(), &settings, &material).unwrap();
    assert_eq!(done.len(), 1);
    assert!(done[0].created && done[0].added >= 2, "{done:?}");
    let store = NoteStore::open(dir.path()).unwrap();
    let note = store.get(&done[0].slug).unwrap().unwrap();
    assert!(note
        .entries
        .iter()
        .all(|e| e.basis == BASIS_MODEL && e.sources == ["S1"]));
    assert!(capture(dir.path(), &settings, &material)
        .unwrap()
        .is_empty());
    let off = Settings {
        knowledge_notes: false,
        ..Default::default()
    };
    assert!(capture(dir.path(), &off, &material).unwrap().is_empty());
    let secret = Material {
        topic: None,
        question: "my password is hunter2 what now",
        parts: &parts,
        basis: BASIS_MODEL,
        sources: &sources,
    };
    assert!(capture(dir.path(), &settings, &secret).unwrap().is_empty());
}

fn source(kind: &str, url: &str, content: &str) -> crate::research::ResearchSource {
    crate::research::ResearchSource {
        id: "R1".into(),
        kind: kind.into(),
        title: "A page".into(),
        url: url.into(),
        path: None,
        start_line: None,
        end_line: None,
        content: content.into(),
        content_sha256: String::new(),
        truncated: false,
        fetched_at: String::new(),
        notes: "an analysis the model wrote".into(),
        analysis_error: None,
    }
}

fn run_with(sources: Vec<crate::research::ResearchSource>) -> crate::research::ResearchRun {
    crate::research::ResearchRun {
        id: "r".into(),
        question: "How are kernels configured?".into(),
        workspace: None,
        model: "m".into(),
        state: "complete".into(),
        rounds_completed: 1,
        queries: vec![],
        sources,
        answer: "Kernels are configured with menuconfig.".into(),
        warnings: vec![],
        stop_reason: String::new(),
        created_at: String::new(),
        updated_at: String::new(),
    }
}

#[test]
fn research_notes_use_the_retrieved_pages_not_the_models_analysis_and_never_local_content() {
    let page = "Menuconfig is a text menu that edits the kernel .config file interactively.";
    let web_only = run_with(vec![source("web", "https://kernel.org/doc", page)]);
    let (parts, sources) = research_material(&web_only).unwrap();
    assert_eq!((parts.len(), sources.len()), (1, 1));
    assert_eq!(parts[0], (page.to_string(), Some(0)));
    assert!(!parts[0].0.contains("an analysis the model wrote"));

    let mixed = run_with(vec![
        source("web", "https://kernel.org/doc", page),
        source("local", "", "private internal design notes"),
    ]);
    let (parts, sources) = research_material(&mixed).unwrap();
    assert_eq!(sources.len(), 1);
    assert!(parts.iter().all(|(t, _)| !t.contains("private internal")));

    let local_only = run_with(vec![source("local", "", "private")]);
    assert!(research_material(&local_only).is_none());
    let empty_page = run_with(vec![source("web", "https://x.org", "  ")]);
    assert!(research_material(&empty_page).is_none());
}

#[test]
fn a_research_run_becomes_a_note_credited_to_its_pages() {
    let dir = tempfile::tempdir().unwrap();
    let settings = Settings {
        knowledge_notes: true,
        ..Default::default()
    };
    let run = run_with(vec![source(
        "web",
        "https://kernel.org/doc",
        "Menuconfig is a text menu that edits the kernel configuration file interactively. The selected options are saved to the .config file in the source tree.",
    )]);
    let done = capture_research(&settings, dir.path(), &run).unwrap();
    assert_eq!(done.len(), 1, "{done:?}");
    let note = NoteStore::open(dir.path())
        .unwrap()
        .get(&done[0].slug)
        .unwrap()
        .unwrap();
    assert!(
        note.entries
            .iter()
            .all(|e| e.basis == BASIS_RESEARCH && e.sources == ["S1"]),
        "{note:?}"
    );
    assert_eq!(note.sources[0].url, "https://kernel.org/doc");
}

#[test]
fn statements_about_the_assistant_or_its_access_are_not_saved_as_knowledge() {
    let dir = tempfile::tempdir().unwrap();
    let store = NoteStore::open(dir.path()).unwrap();
    let out = store
        .absorb(
            "32-Bit Emulation",
            &[
                fact("The material indicates limited code analysis capabilities without direct file access.", BASIS_MODEL, &[]),
                fact("I don't have filesystem access in this session unless a folder is connected.", BASIS_MODEL, &[]),
                fact("Is emulation worth the overhead on a modern CPU?", BASIS_MODEL, &[]),
                fact("Emulating 32-bit code on a 64-bit CPU adds translation overhead compared with native execution.", BASIS_MODEL, &[]),
            ],
            &[],
        )
        .unwrap();
    assert_eq!((out.added, out.rejected), (1, 3));
    assert_eq!(
        store
            .get("32-bit-emulation")
            .unwrap()
            .unwrap()
            .entries
            .len(),
        1
    );
}

#[test]
fn votes_add_up_to_agreed_mixed_or_disputed() {
    let c = |agree, disagree, unsure| Consensus {
        agree,
        disagree,
        unsure,
    };
    assert_eq!(c(3, 0, 0).status(), ReviewStatus::Agreed);
    assert_eq!(c(2, 0, 0).status(), ReviewStatus::Agreed);
    // One model alone cannot agree with anyone.
    assert_eq!(c(1, 0, 0).status(), ReviewStatus::Mixed);
    assert_eq!(c(2, 0, 1).status(), ReviewStatus::Mixed);
    assert_eq!(c(2, 1, 0).status(), ReviewStatus::Mixed);
    assert_eq!(c(1, 1, 1).status(), ReviewStatus::Disputed);
    assert_eq!(c(0, 2, 1).status(), ReviewStatus::Disputed);
    assert_eq!(c(3, 0, 0).label(), "all 3 models agree");
    assert_eq!(c(2, 1, 0).label(), "2 of 3 models agree, 1 disagree");
    assert!(c(1, 2, 0).label().starts_with("disputed: 2 of 3"));
}

#[test]
fn verdicts_are_parsed_strictly_and_a_skipped_statement_is_unsure_not_agreement() {
    let v = parse_verdicts(
        r#"{"verdicts":[{"id":1,"verdict":"true"},{"id":3,"verdict":"FALSE"}]}"#,
        3,
    )
    .unwrap();
    assert_eq!(v, [Verdict::True, Verdict::Unsure, Verdict::False]);
    assert!(parse_verdicts("I think they are all true", 2).is_err());
    assert!(parse_verdicts(r#"{"answer":"true"}"#, 2).is_err());
    let out_of_range = parse_verdicts(r#"{"verdicts":[{"id":9,"verdict":"true"}]}"#, 2).unwrap();
    assert_eq!(out_of_range, [Verdict::Unsure, Verdict::Unsure]);
    assert_eq!(
        tally(&[
            Verdict::True,
            Verdict::True,
            Verdict::False,
            Verdict::Unsure
        ]),
        Consensus {
            agree: 2,
            disagree: 1,
            unsure: 1
        }
    );
}

#[test]
fn the_review_tally_and_log_survive_the_markdown_file_and_the_index() {
    let dir = tempfile::tempdir().unwrap();
    let store = NoteStore::open(dir.path()).unwrap();
    store
        .absorb(
            "Rust Ownership",
            &[
                fact(
                    "Each value in Rust has a single owner at a time.",
                    BASIS_MODEL,
                    &[],
                ),
                fact(
                    "Rust frees memory with a tracing garbage collector at runtime.",
                    BASIS_MODEL,
                    &[],
                ),
                fact(
                    "Moving a value transfers ownership to the new binding.",
                    BASIS_MODEL,
                    &[],
                ),
            ],
            &[],
        )
        .unwrap();
    let results = vec![
        (
            "Each value in Rust has a single owner at a time.".to_string(),
            Consensus {
                agree: 3,
                disagree: 0,
                unsure: 0,
            },
        ),
        (
            "Rust frees memory with a tracing garbage collector at runtime.".to_string(),
            Consensus {
                agree: 0,
                disagree: 3,
                unsure: 0,
            },
        ),
        (
            "Moving a value transfers ownership to the new binding.".to_string(),
            Consensus {
                agree: 2,
                disagree: 0,
                unsure: 1,
            },
        ),
    ];
    assert_eq!(
        store
            .apply_reviews(
                "rust-ownership",
                &results,
                "2026-11-09 · models: a, b, c · 3 fact(s)"
            )
            .unwrap(),
        3
    );
    let text = fs::read_to_string(store.path("rust-ownership")).unwrap();
    assert!(
        text.contains("{review: 3 agree, 0 disagree, 0 unsure} · "),
        "{text}"
    );
    assert!(
        text.contains("## Reviews") && text.contains("models: a, b, c"),
        "{text}"
    );
    let note = store.get("rust-ownership").unwrap().unwrap();
    assert_eq!(parse_markdown(&render_markdown(&note), &note.slug), note);
    assert_eq!(note.tally(), (1, 1, 1));
    let summary = &NoteStore::open(dir.path()).unwrap().list().unwrap()[0];
    assert_eq!((summary.agreed, summary.mixed, summary.disputed), (1, 1, 1));
    // A fact that was edited since the models judged it is not given their verdict.
    assert_eq!(
        store
            .apply_reviews(
                "rust-ownership",
                &[("some other text".into(), Consensus::default())],
                "x"
            )
            .unwrap(),
        0
    );
}

#[test]
fn reviewers_are_different_installed_models_and_leave_out_the_author() {
    let mut settings = Settings::default();
    let installed: Vec<String> = [
        "qwen3.5:4b",
        "qwen3.5:9b",
        "ministral-3:8b",
        "gemma4:e4b",
        "deepseek-ocr:latest",
    ]
    .map(String::from)
    .to_vec();
    settings.notes_review_models = vec![
        "qwen3.5:9b".into(),
        "qwen3.5:4b".into(),
        "ministral-3:8b".into(),
        "gemma4:e4b".into(),
        "not-installed:1b".into(),
    ];
    settings.notes_review_count = 3;
    // Different families first, even though both qwen sizes come earlier in the list.
    assert_eq!(
        pick_reviewers(&settings, &installed, None),
        ["qwen3.5:9b", "ministral-3:8b", "gemma4:e4b"]
    );
    // The author never reviews its own answer while enough others remain.
    assert_eq!(
        pick_reviewers(&settings, &installed, Some("qwen3.5:9b")),
        ["qwen3.5:4b", "ministral-3:8b", "gemma4:e4b"]
    );
    // With too few others, the author is kept so a comparison is still possible.
    settings.notes_review_models = vec!["qwen3.5:9b".into(), "ministral-3:8b".into()];
    assert_eq!(
        pick_reviewers(&settings, &installed, Some("qwen3.5:9b")),
        ["qwen3.5:9b", "ministral-3:8b"]
    );
    settings.notes_review_count = 99;
    settings.notes_review_models = installed.clone();
    assert_eq!(pick_reviewers(&settings, &installed, None).len(), 5);
    // Nothing configured and nothing installed in the routing profile: no reviewers.
    settings.notes_review_models.clear();
    assert!(pick_reviewers(&settings, &["x:1b".to_string()], None).is_empty());
}

#[test]
fn chat_context_shows_consensus_and_withholds_what_the_models_called_wrong() {
    let dir = tempfile::tempdir().unwrap();
    let store = NoteStore::open(dir.path()).unwrap();
    store
        .absorb(
            "Kernel Build",
            &[
                fact(
                    "The kernel build reads its options from the .config file.",
                    BASIS_MODEL,
                    &[],
                ),
                fact(
                    "The kernel build reads its options from the registry hive.",
                    BASIS_MODEL,
                    &[],
                ),
                fact(
                    "The kernel build compiles object files with the system compiler.",
                    BASIS_MODEL,
                    &[],
                ),
            ],
            &[],
        )
        .unwrap();
    store
        .apply_reviews(
            "kernel-build",
            &[
                (
                    "The kernel build reads its options from the .config file.".into(),
                    Consensus {
                        agree: 3,
                        disagree: 0,
                        unsure: 0,
                    },
                ),
                (
                    "The kernel build reads its options from the registry hive.".into(),
                    Consensus {
                        agree: 0,
                        disagree: 3,
                        unsure: 0,
                    },
                ),
            ],
            "log",
        )
        .unwrap();
    let (text, _) = store
        .context_for("how does the kernel build read options", 4000)
        .unwrap()
        .unwrap();
    assert!(
        text.contains(".config file. (model answer, unverified; all 3 models agree)"),
        "{text}"
    );
    assert!(
        !text.contains("registry hive"),
        "a disputed fact reached the chat model: {text}"
    );
    assert!(
        text.contains("1 entries that the reviewing models judged wrong are not shown"),
        "{text}"
    );
    assert!(
        text.contains("system compiler. (model answer, unverified)"),
        "{text}"
    );
}

#[test]
fn stale_editor_cannot_erase_a_background_fact_or_review() {
    let dir = tempfile::tempdir().unwrap();
    let editor = NoteStore::open(dir.path()).unwrap();
    let background = NoteStore::open(dir.path()).unwrap();
    let first = "The pollen grain contains the male gametophyte.";
    let slug = editor
        .absorb("Flowering plants", &[fact(first, BASIS_WEB, &[])], &[])
        .unwrap()
        .slug;
    let path = editor.path(&slug);
    let original = fs::read_to_string(&path).unwrap();
    let second = "Flower petals can attract pollinators with bright colours.";
    background
        .absorb("Flowering plants", &[fact(second, BASIS_WEB, &[])], &[])
        .unwrap();
    background
        .apply_reviews(
            &slug,
            &[(
                first.into(),
                Consensus {
                    agree: 2,
                    disagree: 0,
                    unsure: 0,
                },
            )],
            "Two reviewers agreed",
        )
        .unwrap();
    let updated = fs::read_to_string(&path).unwrap();
    assert!(updated.contains(second));
    assert!(updated.contains("Two reviewers agreed"));
    assert!(editor.save_edit(&slug, &original, "my draft").is_err());
    assert_eq!(fs::read_to_string(&path).unwrap(), updated);
    let edited = updated.replace("Flowering plants", "Flowering plants in spring");
    editor.save_edit(&slug, &updated, &edited).unwrap();
    assert_eq!(fs::read_to_string(path).unwrap(), edited);
}
