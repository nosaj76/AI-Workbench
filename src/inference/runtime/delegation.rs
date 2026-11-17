use super::*;

pub(super) fn delegate_run(
    request: crate::agents::AgentRequest,
    parent: Sender<ChatEvent>,
    cancel: Arc<AtomicBool>,
) -> Result<Value> {
    let (tx, rx) = mpsc::channel();
    let forward = std::thread::spawn(move || {
        while let Ok(event) = rx.recv() {
            match event {
                crate::agents::AgentEvent::Status(s) => {
                    let _ = parent.send(ChatEvent::Status(format!("Subagents: {s}")));
                }
                crate::agents::AgentEvent::Task { title, state, .. } => {
                    let _ = parent.send(ChatEvent::Status(format!("Subagent {title}: {state}")));
                }
                crate::agents::AgentEvent::Approval { reply, .. } => {
                    let _ = reply.send(false);
                }
                crate::agents::AgentEvent::Error(s) => {
                    let _ = parent.send(ChatEvent::Status(format!("Subagent reported: {s}")));
                }
                _ => {}
            }
        }
    });
    let result = crate::agents::run(request, tx, cancel);
    let _ = forward.join();
    let run = result?;
    let report = run.report();
    // The whole report; the model's view of tool results is bounded (and says
    // so) only if it cannot fit in the context.
    let excerpt = report.clone();
    Ok(
        serde_json::json!({"run_id":run.id,"state":run.state,"report_excerpt":excerpt,"report_truncated":false,"task_count":run.tasks.len(),"note":"Full worker traces and report are saved under agent_runs and can be opened in Workbench → Agents."}),
    )
}
