//! CLI protocol fixtures, not calls to real providers or paid inference.
use ollama_workbench::{
    external_agents::AgentProfile,
    files::Workspace,
    tools::{PendingAction, PreparedTool, ToolEngine},
};
use serde_json::json;
use std::fs;

fn profile() -> AgentProfile {
    AgentProfile {
        id: "fixture".into(),
        name: "Local echo fixture".into(),
        program: "/bin/cat".into(),
        args: vec![],
        enabled: true,
        prompt_on_stdin: true,
        may_use_network: false,
        allow_network: false,
        timeout_secs: 5,
    }
}
fn setup() -> (tempfile::TempDir, ToolEngine) {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().join("project");
    fs::create_dir(&workspace).unwrap();
    fs::write(workspace.join("source.txt"), "local evidence\n").unwrap();
    let engine =
        ToolEngine::with_backup_dir(Workspace::new(workspace).unwrap(), dir.path().join("data"));
    engine.register_external_agent(profile()).unwrap();
    (dir, engine)
}
fn pending(engine: &ToolEngine) -> PendingAction {
    match engine.prepare("consult_agent", &json!({"agent_id":"fixture","task":"Review the evidence", "context":"Do not execute $(touch NEVER_CREATED)","files":["source.txt"]})).unwrap() {
        PreparedTool::Approval(action) => *action,
        _ => panic!("External consultation must require approval"),
    }
}
#[test]
fn external_consultation_requires_exact_single_use_approval_and_saves_actual_output() {
    let (dir, engine) = setup();
    let action = pending(&engine);
    assert!(!dir.path().join("data").exists());
    assert!(action.details["complete_prompt"]
        .as_str()
        .unwrap()
        .contains("local evidence"));
    let result = engine.approve(&action).unwrap();
    assert_eq!(result["ok"], true);
    let output = result["result"]["stdout"].as_str().unwrap();
    assert!(output.contains("Review the evidence"));
    assert!(output.contains("local evidence"));
    let run = std::path::PathBuf::from(result["run_directory"].as_str().unwrap());
    assert!(run.join("request.json").is_file());
    assert!(run.join("result.json").is_file());
    assert!(!run.join("NEVER_CREATED").exists());
    assert!(engine.approve(&action).is_err());
}
#[test]
fn denied_external_consultation_launches_nothing() {
    let (dir, engine) = setup();
    let action = pending(&engine);
    engine.deny(&action.id);
    assert!(engine.approve(&action).is_err());
    assert!(!dir.path().join("data").exists());
}
#[test]
fn changed_external_evidence_is_rejected_before_launch() {
    let (dir, engine) = setup();
    let action = pending(&engine);
    fs::write(
        dir.path().join("project/source.txt"),
        "changed after preview",
    )
    .unwrap();
    assert!(engine
        .approve(&action)
        .unwrap_err()
        .to_string()
        .contains("changed after the preview"));
    assert!(!dir.path().join("data").exists());
}
#[test]
fn disabling_external_profile_invalidates_pending_launch() {
    let (dir, engine) = setup();
    let action = pending(&engine);
    let mut changed = profile();
    changed.enabled = false;
    engine.register_external_agent(changed).unwrap();
    assert!(engine.approve(&action).is_err());
    assert!(!dir.path().join("data").exists());
}
#[test]
fn empty_successful_process_is_not_a_completed_consultation() {
    let (_dir, engine) = setup();
    let mut empty = profile();
    empty.program = "/bin/sh".into();
    empty.args = vec!["-c".into(), "cat >/dev/null".into()];
    engine.register_external_agent(empty).unwrap();
    let action = pending(&engine);
    let result = engine.approve(&action).unwrap();
    assert_eq!(result["ok"], false);
    assert_eq!(result["state"], "failed");
}
