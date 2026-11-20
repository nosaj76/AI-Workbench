use anyhow::Result;
use ollama_workbench::{app::Workbench, settings};

fn main() {
    if let Err(error) = run() {
        eprintln!("AI Workbench: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("AI Workbench — native Rust desktop AI workspace for local models (works with Ollama)\n\nUsage: ollama-workbench [--demo] [--data-dir PATH] [--check] [--doctor] [--daemon] [--diagnose-conversations] [--self-review [N]]\n\n  --demo           Labeled UI preview with sample content; no model requests\n  --data-dir PATH  Use an explicit local data directory\n  --check          Check configuration and local Ollama connectivity, then exit\n  --doctor         Inspect configuration and optional program availability (no network)\n  --daemon         Run the persistent scheduler without a desktop window\n  --diagnose-conversations  Scan saved chats and write local failure diagnostics\n  --self-review [N]  Agent team reviews the N newest chats (default 3) for workflow problems and suggests improvements; reads this build's source when present\n  --benchmark-models  Calibrate installed local text models (no tools execute)\n  --benchmark-model NAME  Calibrate one model; repeat for a selection\n  --version        Show application version\n\nThe application opens a native desktop window. It does not start an HTTP server.\nDefault endpoint: http://127.0.0.1:11434");
        return Ok(());
    }
    if args.iter().any(|a| a == "--version") {
        println!(
            "AI Workbench {} (source {})",
            env!("CARGO_PKG_VERSION"),
            env!("WORKBENCH_SOURCE_ID")
        );
        return Ok(());
    }
    let mut demo = false;
    let mut check = false;
    let mut daemon = false;
    let mut doctor = false;
    let mut diagnose = false;
    let mut self_review: Option<usize> = None;
    let mut benchmark = false;
    let mut benchmark_models = Vec::new();
    let mut dir = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--benchmark-models" => benchmark = true,
            "--benchmark-model" => {
                benchmark = true;
                i += 1;
                benchmark_models.push(
                    args.get(i)
                        .ok_or_else(|| {
                            anyhow::anyhow!("--benchmark-model requires an installed model name")
                        })?
                        .clone(),
                );
            }
            "--demo" => demo = true,
            "--check" => check = true,
            "--daemon" => daemon = true,
            "--doctor" => doctor = true,
            "--diagnose-conversations" => diagnose = true,
            "--self-review" => {
                let count = match args.get(i + 1).and_then(|n| n.parse::<usize>().ok()) {
                    Some(n) => {
                        i += 1;
                        n
                    }
                    None => 3,
                };
                if !(1..=ollama_workbench::self_review::MAX_CONVERSATIONS).contains(&count) {
                    anyhow::bail!(
                        "--self-review takes 1–{} conversations",
                        ollama_workbench::self_review::MAX_CONVERSATIONS
                    );
                }
                self_review = Some(count);
            }
            "--data-dir" => {
                i += 1;
                dir =
                    Some(std::path::PathBuf::from(args.get(i).ok_or_else(|| {
                        anyhow::anyhow!("--data-dir requires a path")
                    })?));
            }
            other => anyhow::bail!("Unknown argument: {other}. Use --help."),
        };
        i += 1;
    }
    if [demo, check, daemon, doctor, diagnose, benchmark, self_review.is_some()]
        .into_iter()
        .filter(|enabled| *enabled)
        .count()
        > 1
    {
        anyhow::bail!(
            "Choose only one of --demo, --check, --doctor, --daemon, --diagnose-conversations, --self-review or --benchmark-model(s)"
        );
    }
    let dir = match dir {
        Some(dir) => dir,
        None if demo => {
            std::env::temp_dir().join(format!("ollama-workbench-preview-{}", uuid::Uuid::new_v4()))
        }
        None => settings::data_dir()?,
    };
    if demo {
        eprintln!("Preview uses isolated storage: {}", dir.display());
    } else {
        ollama_workbench::crash_reports::install_panic_hook(&dir);
    }
    if benchmark {
        let config = settings::Settings::load(&dir)?;
        let report = ollama_workbench::benchmarks::run(
            &config,
            &dir,
            &benchmark_models,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            |s| eprintln!("{s}"),
        )?;
        println!("{report}");
        return Ok(());
    }
    if diagnose {
        let store = ollama_workbench::storage::Store::open(&dir)?;
        let (sessions, incidents) = store.scan_conversation_diagnostics()?;
        println!("Reviewed {sessions} saved conversations; {incidents} new or updated incidents.\nLog: {}\n\n{}",
            ollama_workbench::conversation_diagnostics::log_path(&dir).display(), store.conversation_diagnostic_report()?);
        return Ok(());
    }
    if let Some(count) = self_review {
        let config = settings::Settings::load(&dir)?;
        let store = ollama_workbench::storage::Store::open(&dir)?;
        let sessions = ollama_workbench::self_review::recent_sessions(&store, count)?;
        drop(store);
        let source = ollama_workbench::self_review::default_source_root();
        eprintln!(
            "Reviewing {} conversation(s){}",
            sessions.len(),
            source.as_ref().map_or(String::new(), |root| format!(
                " and source code at {}",
                root.display()
            ))
        );
        let request = ollama_workbench::self_review::request(
            &config,
            &dir,
            &sessions,
            source,
            config.active_model().into(),
        )?;
        let (tx, rx) = std::sync::mpsc::channel();
        let printer = std::thread::spawn(move || {
            for event in rx {
                match event {
                    ollama_workbench::agents::AgentEvent::Status(text) => eprintln!("{text}"),
                    ollama_workbench::agents::AgentEvent::Task { title, state, message, .. } => {
                        eprintln!("[{state}] {title}: {message}")
                    }
                    // Read-only teams never request approvals; refuse if one arrives.
                    ollama_workbench::agents::AgentEvent::Approval { reply, .. } => {
                        let _ = reply.send(false);
                    }
                    _ => {}
                }
            }
        });
        let run = ollama_workbench::agents::run(
            request,
            tx,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        )?;
        let _ = printer.join();
        println!("{}", run.report());
        return Ok(());
    }
    if doctor {
        let config = settings::Settings::load(&dir)?;
        println!(
            "{}",
            ollama_workbench::diagnostics::report(&ollama_workbench::diagnostics::inspect(
                &config, &dir, None
            ))
        );
        let crashes = ollama_workbench::crash_reports::recent_reports(&dir, 5);
        if !crashes.is_empty() {
            println!("\nRecent panic reports (newest first):");
            for path in crashes {
                println!("  {}", path.display());
            }
        }
        return Ok(());
    }
    // A single data-root environment setting keeps backup and plugin storage
    // aligned with --data-dir before any worker threads are created.
    std::env::set_var("OLLAMA_WORKBENCH_DATA_DIR", &dir);
    if daemon {
        if demo || check {
            anyhow::bail!("--daemon cannot be combined with --demo or --check");
        }
        return ollama_workbench::scheduler::run_daemon_cli(&dir);
    }
    if check {
        let config = settings::Settings::load(&dir)?;
        config.validate_endpoint()?;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        return rt.block_on(async {
            let client = ollama_workbench::ollama::OllamaClient::new(&config.base_url)?;
            println!("Ollama version: {}", client.version().await?);
            for model in client.list_models().await? {
                println!("{}\t{} bytes", model.name, model.size);
            }
            Ok(())
        });
    }
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title(format!(
                "{} {}",
                settings::APP_NAME,
                env!("CARGO_PKG_VERSION")
            ))
            // Matches the installed launcher, so the desktop shows its icon.
            .with_app_id("org.localtools.AIWorkbench")
            .with_inner_size([1180.0, 680.0])
            .with_min_inner_size([760.0, 560.0]),
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };
    // Probe the GPU (nvidia-smi) off the UI thread so the first prompt doesn't wait on it.
    std::thread::spawn(|| {
        settings::hardware_summary();
    });
    // Only a clean return from the window removes the marker; see crash_reports.
    let (session, crash_notice) = if demo {
        (None, None)
    } else {
        let (session, notice) = ollama_workbench::crash_reports::begin_session(&dir);
        (Some(session), notice)
    };
    eframe::run_native(
        settings::APP_NAME,
        options,
        Box::new(move |cc| {
            let mut app = Workbench::new(cc, dir, demo)?;
            if let Some(notice) = crash_notice {
                app.show_startup_notice(notice);
            }
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| {
        anyhow::anyhow!(
            "Native window could not start: {e}. Run from a graphical X11 or Wayland desktop."
        )
    })?;
    if let Some(session) = session {
        session.finish();
    }
    Ok(())
}
