//! Small local task calibration. Synthetic prompts only; proposed tools never execute.
use crate::{
    ollama::{ChatMessage, ChatOptions, OllamaClient},
    routing::{self, ModelCandidate, RoutingConfig, TaskKind},
    settings::Settings,
};
use anyhow::{bail, Result};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

const SUITE: u32 = 1;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CaseResult {
    pub task: TaskKind,
    pub case: String,
    pub passed: bool,
    pub milliseconds: u64,
    pub response: String,
    pub error: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelResult {
    pub suite: u32,
    pub model: String,
    pub digest: String,
    pub endpoint: String,
    pub measured_at: DateTime<Utc>,
    pub cases: Vec<CaseResult>,
}
struct Case {
    task: TaskKind,
    id: &'static str,
    prompt: &'static str,
    expected: &'static str,
    words: Option<usize>,
    tool: bool,
}
fn cases() -> Vec<Case> {
    use TaskKind::*;
    vec![
        Case{task:General,id:"extract",prompt:"Record: owner=Nora; ticket=Q47; priority=high. Return only the ticket identifier.",expected:"q47",words:None,tool:false},
        Case{task:General,id:"unknown",prompt:"Record: name=Omar; city=Lima. What is Omar's age? Return only UNKNOWN if it is not stated.",expected:"unknown",words:None,tool:false},
        Case{task:Coding,id:"bounds",prompt:"Python: xs=[2,4,6]; print(xs[len(xs)]). Return only the exception class name.",expected:"indexerror",words:None,tool:false},
        Case{task:Coding,id:"loop",prompt:"Python: n=0\nfor x in range(1,5):\n n += x*x\nprint(n)\nReturn only the printed number.",expected:"30",words:None,tool:false},
        Case{task:Reasoning,id:"arithmetic",prompt:"A box has 7 rows of 8 tokens. Remove 19 tokens. How many remain? Return only the number.",expected:"37",words:None,tool:false},
        Case{task:Reasoning,id:"logic",prompt:"All larks are birds. Some birds swim. Does it follow that all larks swim? Return only YES or NO.",expected:"no",words:None,tool:false},
        Case{task:Writing,id:"sentence",prompt:"Write one short sentence containing both release and Friday. Use at most 12 words. Return only the sentence.",expected:"release|friday",words:Some(12),tool:false},
        Case{task:Writing,id:"heading",prompt:"Turn 'The launch has been delayed' into a heading of at most 5 words. Include launch and delayed. Return only the heading.",expected:"launch|delayed",words:Some(5),tool:false},
        Case{task:Orchestrator,id:"tool_a",prompt:"Call lookup_order with order_id Q47 and include_history false. Do not write an answer.",expected:"Q47",words:None,tool:true},
        Case{task:Orchestrator,id:"tool_b",prompt:"Call lookup_order with order_id R82 and include_history true. Do not write an answer.",expected:"R82",words:None,tool:true},
    ]
}
fn grade(case: &Case, reply: &ChatMessage) -> bool {
    if case.tool {
        return reply.tool_calls.len() == 1
            && reply.tool_calls[0].function.name == "lookup_order"
            && reply.tool_calls[0].function.arguments
                == json!({"order_id":case.expected,"include_history":case.id=="tool_b"});
    }
    if !reply.tool_calls.is_empty() {
        return false;
    }
    let text = reply.content.trim().to_lowercase();
    if let Some(max) = case.words {
        let words: Vec<_> = text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .collect();
        !words.is_empty()
            && words.len() <= max
            && case.expected.split('|').all(|w| words.contains(&w))
            && text.matches(['.', '!', '?']).count() <= 1
    } else {
        text.trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace()) == case.expected
    }
}
fn db(dir: &Path) -> Result<Connection> {
    std::fs::create_dir_all(dir)?;
    let c = Connection::open(dir.join("benchmarks.sqlite3"))?;
    c.busy_timeout(Duration::from_secs(5))?;
    c.execute_batch("CREATE TABLE IF NOT EXISTS model_benchmarks (model TEXT NOT NULL, endpoint TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(model,endpoint));")?;
    Ok(c)
}
fn save(dir: &Path, record: &ModelResult) -> Result<()> {
    db(dir)?.execute("INSERT INTO model_benchmarks VALUES(?1,?2,?3) ON CONFLICT(model,endpoint) DO UPDATE SET payload=excluded.payload", params![routing::canonical_model_name(&record.model),record.endpoint,serde_json::to_string(record)?])?;
    Ok(())
}
pub fn results(dir: &Path) -> Result<Vec<ModelResult>> {
    let c = db(dir)?;
    let mut statement = c.prepare("SELECT payload FROM model_benchmarks ORDER BY model")?;
    let rows = statement
        .query_map([], |r| r.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    rows.into_iter()
        .map(|s| Ok(serde_json::from_str(&s)?))
        .collect()
}
fn score(record: &ModelResult, task: TaskKind) -> Option<(usize, u64)> {
    let rows: Vec<_> = record.cases.iter().filter(|c| c.task == task).collect();
    if rows.len() != 2 {
        return None;
    }
    Some((
        rows.iter().filter(|c| c.passed).count(),
        rows.iter().map(|c| c.milliseconds).sum::<u64>() / 2,
    ))
}
fn fresh(record: &ModelResult, candidate: &ModelCandidate, endpoint: &str) -> bool {
    let age = Utc::now().signed_duration_since(record.measured_at);
    record.suite == SUITE
        && record.endpoint == endpoint
        && age.num_seconds() >= 0
        && age.num_days() < 30
        && !record.digest.is_empty()
        && candidate.model.extra.get("digest").and_then(Value::as_str)
            == Some(record.digest.as_str())
        && routing::canonical_model_name(&record.model)
            == routing::canonical_model_name(&candidate.model.name)
}
/// Capability, privacy, resource and retry exclusions still precede measured ranking.
pub fn choose_measured(
    settings: &Settings,
    config: &RoutingConfig,
    task: TaskKind,
    need_tools: bool,
    excluded: &[String],
    candidates: &[ModelCandidate],
    dir: &Path,
) -> Result<Option<routing::RouteDecision>> {
    if !config.use_benchmarks {
        return Ok(None);
    }
    let records = results(dir)?;
    let mut ranked = Vec::new();
    for candidate in candidates {
        let Ok(decision) = routing::choose_excluding(
            config,
            task,
            need_tools,
            settings.allow_remote,
            &settings.model,
            excluded,
            std::slice::from_ref(candidate),
        ) else {
            continue;
        };
        for record in &records {
            if !fresh(record, candidate, &settings.base_url) {
                continue;
            }
            let Some((passed, ms)) = score(record, task) else {
                continue;
            };
            // Sparse checks are evidence only when every task case passed.
            if passed != 2
                || (need_tools && score(record, TaskKind::Orchestrator).is_none_or(|s| s.0 != 2))
            {
                continue;
            }
            ranked.push((ms, decision.clone()));
        }
    }
    ranked.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.model.cmp(&b.1.model)));
    Ok(ranked.into_iter().next().map(|(ms,mut d)|{
        d.reason.push_str(&format!(" Local calibration suite {SUITE}: 2/2 {task:?} checks passed; mean request {ms} ms; fastest fully passing eligible measured model. Small synthetic suite, not a guarantee for this prompt."));d
    }))
}
pub fn report(dir: &Path) -> Result<String> {
    let mut lines=vec!["Local calibration: two checks per measured task; fixed 4096 context, 160 output tokens, temperature 0, thinking off. Scores measure narrow skills, not overall model quality. Records expire for routing after 30 days or model digest/suite changes. Explicit model profiles still override automatic routing.".into()];
    for r in results(dir)? {
        let scores = [
            TaskKind::General,
            TaskKind::Coding,
            TaskKind::Reasoning,
            TaskKind::Writing,
            TaskKind::Orchestrator,
        ]
        .into_iter()
        .filter_map(|task| score(&r, task).map(|(n, ms)| format!("{task:?}: {n}/2, {ms} ms")))
        .collect::<Vec<_>>()
        .join("; ");
        lines.push(format!(
            "{} — {} — {}",
            r.model,
            r.measured_at.to_rfc3339(),
            scores
        ));
    }
    Ok(lines.join("\n\n"))
}
pub fn run(
    settings: &Settings,
    dir: &Path,
    models: &[String],
    cancel: Arc<AtomicBool>,
    mut progress: impl FnMut(String),
) -> Result<String> {
    settings.validate_endpoint()?;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(async {
        let client=OllamaClient::new(&settings.base_url)?;
        let cfg=routing::config(settings)?;
        let inventory=routing::inventory(settings,&client,dir,cancel.clone()).await?;
        let chosen:Vec<_>=inventory.into_iter().filter(|c|models.is_empty() || models.iter().any(|m|routing::canonical_model_name(m)==routing::canonical_model_name(&c.model.name))).collect();
        if chosen.is_empty(){bail!("No requested installed model was found");}
        for candidate in chosen {
            if cancel.load(Ordering::Relaxed){break;}
            if candidate.model.is_remote() || !candidate.capabilities.iter().any(|c|c=="completion")
                || crate::translation::is_dedicated_model(&candidate.model.name,&candidate.metadata)
                || routing::validate_parameter_cap(&cfg,&candidate.model.name,&candidate.metadata,&candidate.model.details).is_err()
                || (cfg.max_model_gib>0.0 && candidate.model.size as f64>cfg.max_model_gib*1_073_741_824.0) {
                progress(format!("Skipped {}: outside this local text suite's capability/resource scope",candidate.model.name));continue;
            }
            let mut record=ModelResult{suite:SUITE,model:candidate.model.name.clone(),digest:candidate.model.extra.get("digest").and_then(Value::as_str).unwrap_or_default().into(),endpoint:settings.base_url.clone(),measured_at:Utc::now(),cases:vec![]};
            // Hold the endpoint permit for this short model batch; cancellation releases it.
            let _permit=routing::inference_permit(settings,cancel.clone()).await?;
            for case in cases() {
                if cancel.load(Ordering::Relaxed){break;}
                if case.tool && !candidate.capabilities.iter().any(|c|c=="tools"){continue;}
                progress(format!("Benchmark {} / {:?} / {}",candidate.model.name,case.task,case.id));
                let tools=if case.tool {vec![json!({"type":"function","function":{"name":"lookup_order","description":"A simulated benchmark tool. It will not execute.","parameters":{"type":"object","properties":{"order_id":{"type":"string"},"include_history":{"type":"boolean"}},"required":["order_id","include_history"]}}})]}else{vec![]};
                let options=ChatOptions{temperature:0.0,num_ctx:4096,num_predict:160,think:Some(json!(false)),seed:Some(42),keep_alive:"2m".into(),..Default::default()};
                let started=Instant::now();
                let response=tokio::time::timeout(Duration::from_secs(60),client.chat_stream(&candidate.model.name,&[ChatMessage{role:"user".into(),content:case.prompt.into(),..Default::default()}],&options,&tools,cancel.clone(),|_|{})).await;
                let (passed,text,error)=match response {
                    Ok(Ok(reply))=>(grade(&case,&reply),serde_json::to_string(&reply)?.chars().take(2000).collect(),None),
                    Ok(Err(e))=>(false,String::new(),Some(e.to_string())),
                    Err(_)=>(false,String::new(),Some("60-second request deadline exceeded".into())),
                };
                record.cases.push(CaseResult{task:case.task,case:case.id.into(),passed,milliseconds:started.elapsed().as_millis() as u64,response:text,error});
            }
            // Only unload the model this benchmark just used, never unrelated models.
            let _=tokio::time::timeout(Duration::from_secs(15),client.unload(&candidate.model.name)).await;
            if !cancel.load(Ordering::Relaxed){save(dir,&record)?;}
        }
        report(dir)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn printed_tool_json_is_not_a_native_tool_call() {
        let case = cases().into_iter().find(|c| c.tool).unwrap();
        assert!(!grade(&case,&ChatMessage{role:"assistant".into(),content:r#"{"name":"lookup_order","arguments":{"order_id":"Q47","include_history":false}}"#.into(),..Default::default()}));
        let mut reply = ChatMessage { tool_calls: vec![serde_json::from_value(json!({"function":{"name":"lookup_order","arguments":{"order_id":"Q47","include_history":true}}})).unwrap()], ..Default::default() };
        assert!(!grade(&case, &reply));
        reply.tool_calls[0].function.arguments["include_history"] = json!(false);
        assert!(grade(&case, &reply));
    }
    #[test]
    fn task_grades_require_correct_answers_and_constraints() {
        for case in cases().into_iter().filter(|c| !c.tool) {
            assert!(!grade(
                &case,
                &ChatMessage {
                    content: "I completed the task".into(),
                    ..Default::default()
                }
            ));
        }
        let c = cases().remove(0);
        assert!(grade(
            &c,
            &ChatMessage {
                content: "Q47".into(),
                ..Default::default()
            }
        ));
    }
    fn candidate(name: &str) -> ModelCandidate {
        let mut model = crate::ollama::ModelInfo {
            name: name.into(),
            size: 1024,
            ..Default::default()
        };
        model.extra.insert("digest".into(), json!("digest-v1"));
        ModelCandidate {
            model,
            capabilities: vec!["completion".into(), "tools".into()],
            metadata: json!({}),
            running: false,
            observed_tokens_per_second: None,
        }
    }
    fn record(name: &str, settings: &Settings, passed: bool, ms: u64) -> ModelResult {
        ModelResult {
            suite: SUITE,
            model: name.into(),
            digest: "digest-v1".into(),
            endpoint: settings.base_url.clone(),
            measured_at: Utc::now(),
            cases: cases()
                .into_iter()
                .filter(|c| matches!(c.task, TaskKind::General | TaskKind::Orchestrator))
                .map(|c| CaseResult {
                    task: c.task,
                    case: c.id.into(),
                    passed,
                    milliseconds: ms,
                    response: String::new(),
                    error: None,
                })
                .collect(),
        }
    }
    #[test]
    fn quality_and_capabilities_precede_latency_and_stale_evidence_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let settings = Settings::default();
        let cfg = RoutingConfig::default();
        let slow = candidate("slow");
        let fast = candidate("fast");
        save(dir.path(), &record("slow", &settings, true, 100)).unwrap();
        save(dir.path(), &record("fast", &settings, false, 1)).unwrap();
        assert_eq!(
            choose_measured(
                &settings,
                &cfg,
                TaskKind::General,
                true,
                &[],
                &[fast, slow.clone()],
                dir.path()
            )
            .unwrap()
            .unwrap()
            .model,
            "slow"
        );
        let mut stale = record("slow", &settings, true, 100);
        stale.measured_at = Utc::now() - chrono::Duration::days(31);
        save(dir.path(), &stale).unwrap();
        assert!(choose_measured(
            &settings,
            &cfg,
            TaskKind::General,
            true,
            &[],
            std::slice::from_ref(&slow),
            dir.path()
        )
        .unwrap()
        .is_none());
        save(dir.path(), &record("slow", &settings, true, 100)).unwrap();
        let mut changed = slow.clone();
        changed
            .model
            .extra
            .insert("digest".into(), json!("digest-v2"));
        assert!(choose_measured(
            &settings,
            &cfg,
            TaskKind::General,
            true,
            &[],
            &[changed],
            dir.path()
        )
        .unwrap()
        .is_none());
        let mut no_tools = slow.clone();
        no_tools.capabilities = vec!["completion".into()];
        assert!(choose_measured(
            &settings,
            &cfg,
            TaskKind::General,
            true,
            &[],
            &[no_tools],
            dir.path()
        )
        .unwrap()
        .is_none());
        assert!(choose_measured(
            &settings,
            &cfg,
            TaskKind::General,
            true,
            &["slow".into()],
            &[slow],
            dir.path()
        )
        .unwrap()
        .is_none());
    }
    #[test]
    fn records_roundtrip_without_losing_other_models() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = ModelResult {
            suite: SUITE,
            model: "one".into(),
            digest: "abc".into(),
            endpoint: "local".into(),
            measured_at: Utc::now(),
            cases: vec![],
        };
        save(dir.path(), &r).unwrap();
        r.model = "two".into();
        save(dir.path(), &r).unwrap();
        assert_eq!(results(dir.path()).unwrap().len(), 2);
    }
}
