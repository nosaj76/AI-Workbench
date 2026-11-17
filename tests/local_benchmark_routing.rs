//! Read-only inference inventory and saved calibration routing acceptance.
#[test]
#[ignore = "requires local Ollama and completed model benchmarks"]
fn saved_benchmarks_drive_real_inventory_routing() {
    use ollama_workbench::{
        benchmarks,
        ollama::OllamaClient,
        routing::{self, TaskKind},
        settings::{self, Settings},
    };
    use std::sync::{atomic::AtomicBool, Arc};
    let dir = settings::data_dir().unwrap();
    let settings = Settings::load(&dir).unwrap();
    assert!(settings.model_routing);
    assert!(settings.profiles[settings.selected_profile]
        .model
        .is_empty());
    assert!(benchmarks::results(&dir).unwrap().len() >= 2);
    let client = OllamaClient::new(&settings.base_url).unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        for task in [
            TaskKind::General,
            TaskKind::Coding,
            TaskKind::Writing,
            TaskKind::Reasoning,
            TaskKind::Orchestrator,
        ] {
            let decision = routing::route_for_task(
                &settings,
                &client,
                task,
                true,
                &settings.model,
                &dir,
                Arc::new(AtomicBool::new(false)),
            )
            .await
            .unwrap();
            eprintln!("{task:?}: {} — {}", decision.model, decision.reason);
            if task != TaskKind::Reasoning {
                assert!(decision.reason.contains("Local calibration suite"));
            }
            assert!(decision.use_tools);
        }
    });
}
