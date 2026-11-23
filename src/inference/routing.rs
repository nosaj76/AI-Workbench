//! Explainable local model routing: capability gates precede preference scoring.
use crate::{
    ollama::{ChatMessage, ModelInfo, OllamaClient},
    settings::Settings,
};
use anyhow::{bail, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, OnceLock, Weak,
    },
    time::Duration,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct RoutingConfig {
    pub general: Vec<String>,
    pub coding: Vec<String>,
    pub writing: Vec<String>,
    pub reasoning: Vec<String>,
    pub vision: Vec<String>,
    pub translation: Vec<String>,
    pub orchestrator: Vec<String>,
    /// Total parameters, in decimal billions. Zero preserves legacy unlimited routing.
    pub max_parameters_b: f64,
    pub max_model_gib: f64,
    pub prefer_running: bool,
    pub favor_speed: bool,
    pub use_benchmarks: bool,
    /// Serialize this application's inference requests to an endpoint and release
    /// the model after each request. No unrelated model is explicitly unloaded.
    pub single_model_mode: bool,
}
impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            general: vec![],
            coding: vec![],
            writing: vec![],
            reasoning: vec![],
            vision: vec![],
            translation: vec![],
            orchestrator: vec![],
            max_parameters_b: 0.0,
            max_model_gib: 8.0,
            prefer_running: true,
            favor_speed: false,
            use_benchmarks: true,
            single_model_mode: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskKind {
    General,
    Coding,
    Writing,
    Reasoning,
    Vision,
    Translation,
    Orchestrator,
}
pub fn classify(messages: &[ChatMessage]) -> TaskKind {
    if messages.iter().any(|m| !m.images.is_empty()) {
        return TaskKind::Vision;
    }
    let prompt = messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| crate::memory::user_text(&m.content).to_lowercase())
        .unwrap_or_default();
    let words: Vec<_> = prompt
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    let has = |phrases: &[&str]| {
        phrases.iter().any(|phrase| {
            let phrase: Vec<_> = phrase.split_whitespace().collect();
            words.windows(phrase.len()).any(|window| window == phrase)
        })
    };
    if has(&[
        "translate",
        "translation",
        "translations",
        "translating",
        "translator",
    ]) && !has(&[
        "implement",
        "implementation",
        "debug",
        "refactor",
        "compiler",
    ]) {
        TaskKind::Translation
    } else if has(&[
        "code",
        "rust",
        "cargo",
        "compiler",
        "function",
        "debug",
        "repository",
        "refactor",
        "programming",
        "source files",
        "unit test",
        "unit tests",
        "tests",
        "bug",
        "bugs",
        "coding",
        "sql",
        "python",
        "typescript",
    ]) {
        TaskKind::Coding
    } else if has(&[
        "prove",
        "reason",
        "reasoning",
        "mathematics",
        "mathematical",
        "math",
        "theorem",
        "research",
        "analyze",
        "analyse",
        "compare evidence",
        "design architecture",
        "analysis",
        "analyzing",
        "analysing",
    ]) {
        TaskKind::Reasoning
    } else if has(&[
        "write",
        "writing",
        "rewrite",
        "story",
        "chapter",
        "editor",
        "proofread",
        "paragraph",
        "document",
        "poem",
        "draft",
        "summarize",
        "summarise",
        "summary",
        "summaries",
        "editing",
    ]) {
        TaskKind::Writing
    } else {
        TaskKind::General
    }
}

/// Canonical Ollama spelling, with an exact tag (an omitted tag means latest).
/// Namespace and tag differences are never reduced to substring matches.
pub fn canonical_model_name(name: &str) -> String {
    let lower = name.trim().to_ascii_lowercase();
    let name = lower
        .strip_prefix("registry.ollama.ai/")
        .or_else(|| lower.strip_prefix("registry.ollama.com/"))
        .unwrap_or(&lower);
    let name = name.strip_prefix("library/").unwrap_or(name);
    if name
        .rsplit('/')
        .next()
        .is_some_and(|part| !part.contains(':') && !part.contains('@'))
    {
        format!("{name}:latest")
    } else {
        name.to_string()
    }
}

fn exact_count(value: &Value) -> Option<u64> {
    let count = value
        .as_u64()
        .or_else(|| {
            value.as_f64().and_then(|n| {
                (n.is_finite() && (1.0..=9_007_199_254_740_991.0).contains(&n) && n.fract() == 0.0)
                    .then_some(n as u64)
            })
        })
        .or_else(|| {
            value.as_str().and_then(|s| {
                let s = s.trim();
                (!s.is_empty() && s.bytes().all(|c| c.is_ascii_digit()))
                    .then(|| s.parse().ok())
                    .flatten()
            })
        })?;
    (count > 0).then_some(count)
}

fn labelled_count(value: &Value) -> Option<u64> {
    let raw = value.as_str()?.trim().to_ascii_uppercase();
    let (number, multiplier) = match raw.chars().last()? {
        'K' => (&raw[..raw.len() - 1], 1_000.0),
        'M' => (&raw[..raw.len() - 1], 1_000_000.0),
        'B' => (&raw[..raw.len() - 1], 1_000_000_000.0),
        'T' => (&raw[..raw.len() - 1], 1_000_000_000_000.0),
        _ => return exact_count(value),
    };
    let number = number.trim();
    if number.is_empty()
        || number.chars().filter(|c| *c == '.').count() > 1
        || !number.bytes().all(|b| b.is_ascii_digit() || b == b'.')
    {
        return None;
    }
    let count = number.parse::<f64>().ok()? * multiplier;
    (count.is_finite() && count >= 1.0 && count < u64::MAX as f64).then(|| count.ceil() as u64)
}

/// Read total parameters from server metadata, never from a model's tag or its
/// quantized file size. MoE active-expert counts are not total parameter counts.
/// A separately reported projector is added to the language-model count.
pub fn total_parameters(metadata: &Value, listed_details: &Value) -> Option<u64> {
    let info = metadata.get("model_info");
    let declared = info.and_then(|v| v.get("general.parameter_count"));
    let projector = metadata
        .get("projector_info")
        .filter(|v| !v.is_null() && !v.as_object().is_some_and(|o| o.is_empty()));
    let count = if let Some(value) = declared {
        exact_count(value)?
    } else {
        // Labels such as "3B" may describe active parameters for MoE models.
        let experts = info.and_then(Value::as_object).is_some_and(|map| {
            map.iter().any(|(key, value)| {
                (key.ends_with(".expert_count") || key.ends_with(".expert_used_count"))
                    && exact_count(value).is_none_or(|n| n > 1)
            })
        }) || info
            .and_then(|v| v.get("general.architecture"))
            .and_then(Value::as_str)
            .is_some_and(moe_family)
            || [metadata.get("details"), Some(listed_details)]
                .into_iter()
                .flatten()
                .any(|details| {
                    details
                        .get("family")
                        .and_then(Value::as_str)
                        .is_some_and(moe_family)
                        || details
                            .get("families")
                            .and_then(Value::as_array)
                            .is_some_and(|families| {
                                families.iter().filter_map(Value::as_str).any(moe_family)
                            })
                });
        if experts || projector.is_some() {
            return None;
        }
        let label = metadata
            .get("details")
            .and_then(|v| v.get("parameter_size"))
            .or_else(|| listed_details.get("parameter_size"))?;
        labelled_count(label)?
    };
    if let Some(projector) = projector {
        count.checked_add(exact_count(projector.get("general.parameter_count")?)?)
    } else {
        Some(count)
    }
}

fn moe_family(family: &str) -> bool {
    let lower = family.to_ascii_lowercase();
    lower.contains("moe") || lower.contains("mixtral")
}

pub fn config(settings: &Settings) -> Result<RoutingConfig> {
    let config: RoutingConfig = serde_json::from_str(&settings.routing_json)
        .context("Invalid routing configuration JSON")?;
    if !config.max_model_gib.is_finite() || config.max_model_gib < 0.0 {
        bail!("Routing model-file size budget must be a finite nonnegative GiB value");
    }
    if !config.max_parameters_b.is_finite() || config.max_parameters_b < 0.0 {
        bail!("Routing parameter cap must be a finite nonnegative number of billions");
    }
    Ok(config)
}

/// Enforce the configured total-parameter cap for manual as well as automatic
/// inference. Unknown or malformed metadata cannot bypass an active cap.
pub fn validate_parameter_cap(
    config: &RoutingConfig,
    model: &str,
    metadata: &Value,
    details: &Value,
) -> Result<Option<u64>> {
    let count = total_parameters(metadata, details);
    if config.max_parameters_b > 0.0 {
        if metadata
            .get("model_info")
            .and_then(|info| info.get("general.parameter_count"))
            .and_then(exact_count)
            .is_none()
        {
            bail!("Cannot verify exact total parameters for {model}; the {:.2}B routing cap rejects rounded parameter-size labels and unknown totals", config.max_parameters_b);
        }
        let count = count.with_context(|| format!("Cannot verify total parameters for {model}; the {:.2}B routing cap rejects unknown or ambiguous sizes", config.max_parameters_b))?;
        if count as f64 > config.max_parameters_b * 1_000_000_000.0 {
            bail!(
                "{model} has {:.3}B total parameters and exceeds the {:.2}B routing cap",
                count as f64 / 1e9,
                config.max_parameters_b
            );
        }
    }
    Ok(count)
}

/// Apply the same local-only, total-parameter and single-model policy to
/// embedding inference used by indexing and retrieval.
pub async fn embeddings_with_policy(
    settings: &Settings,
    client: &OllamaClient,
    model: &str,
    inputs: &[String],
    cancel: Arc<AtomicBool>,
) -> Result<Vec<Vec<f32>>> {
    settings.validate_endpoint()?;
    if inputs.is_empty() {
        return Ok(vec![]);
    }
    let config = config(settings)?;
    if !settings.allow_remote {
        client.ensure_local_model(model, cancel.clone()).await?;
    }
    if config.max_parameters_b > 0.0 {
        let metadata = tokio::select! {
            result = client.show(model) => result?,
            _ = cancelled(cancel.clone()) => bail!("Embedding model verification cancelled"),
        };
        validate_parameter_cap(&config, model, &metadata, &Value::Null)?;
        if !metadata
            .get("capabilities")
            .and_then(Value::as_array)
            .is_some_and(|caps| caps.iter().any(|cap| cap == "embedding"))
        {
            bail!("{model} has no verified embedding capability");
        }
    }
    let _permit = inference_permit(settings, cancel.clone()).await?;
    client
        .embeddings_with_keep_alive(
            model,
            inputs,
            config.single_model_mode.then_some("0"),
            cancel,
        )
        .await
}

/// This process shares a request permit across all runtimes using an endpoint.
/// Callers MUST release it before tools, delegation or approval waits.
pub(crate) async fn inference_permit(
    settings: &Settings,
    cancel: Arc<AtomicBool>,
) -> Result<Option<tokio::sync::OwnedSemaphorePermit>> {
    if !config(settings)?.single_model_mode {
        return Ok(None);
    }
    type Gates = Mutex<BTreeMap<String, Weak<tokio::sync::Semaphore>>>;
    static GATES: OnceLock<Gates> = OnceLock::new();
    let url = crate::endpoint::normalize(&settings.base_url)?;
    let gate = {
        let mut gates = GATES
            .get_or_init(|| Mutex::new(BTreeMap::new()))
            .lock()
            .map_err(|_| anyhow::anyhow!("Inference scheduler is unavailable"))?;
        gates.retain(|_, gate| gate.strong_count() > 0);
        let gate = gates
            .get(url.as_str())
            .and_then(Weak::upgrade)
            .unwrap_or_else(|| Arc::new(tokio::sync::Semaphore::new(1)));
        gates.insert(url.to_string(), Arc::downgrade(&gate));
        gate
    };
    tokio::select! {
        permit = gate.acquire_owned() => Ok(Some(permit.context("Inference scheduler closed")?)),
        _ = cancelled(cancel) => bail!("Inference cancelled while waiting for the local model"),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelCandidate {
    pub model: ModelInfo,
    pub capabilities: Vec<String>,
    pub metadata: Value,
    pub running: bool,
    pub observed_tokens_per_second: Option<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RouteDecision {
    pub model: String,
    pub task: TaskKind,
    pub reason: String,
    pub candidates_considered: usize,
    pub rejected: Vec<String>,
    pub capabilities: Vec<String>,
    pub use_tools: bool,
    pub total_parameters: Option<u64>,
    pub dedicated_translation: bool,
}

pub fn choose(
    config: &RoutingConfig,
    task: TaskKind,
    need_tools: bool,
    allow_remote: bool,
    selected: &str,
    candidates: &[ModelCandidate],
) -> Result<RouteDecision> {
    if !config.max_model_gib.is_finite() || config.max_model_gib < 0.0 {
        bail!("Routing memory budget must be a finite nonnegative GiB value");
    }
    if !config.max_parameters_b.is_finite() || config.max_parameters_b < 0.0 {
        bail!("Routing parameter cap must be a finite nonnegative number of billions");
    }
    let preferences = match task {
        TaskKind::Coding => &config.coding,
        TaskKind::Writing => &config.writing,
        TaskKind::Reasoning => &config.reasoning,
        TaskKind::Vision => &config.vision,
        TaskKind::General => &config.general,
        TaskKind::Translation => &config.translation,
        TaskKind::Orchestrator => &config.orchestrator,
    };
    let mut scored = vec![];
    let mut rejected = vec![];
    for c in candidates {
        let has = |cap: &str| c.capabilities.iter().any(|v| v == cap);
        let remote = c.model.is_remote()
            || ["remote_host", "remote_model"].iter().any(|k| {
                c.metadata.get(k).is_some_and(|value| {
                    !value.is_null() && !value.as_str().is_some_and(|s| s.trim().is_empty())
                })
            });
        let reason = if !allow_remote && remote {
            Some("cloud routing is disabled")
        } else if !has("completion") {
            Some("no confirmed completion capability")
        } else if task == TaskKind::Vision && !has("vision") {
            Some("vision required")
        } else if need_tools && !has("tools") {
            Some("tool calling required")
        } else if crate::translation::is_dedicated_model(&c.model.name, &c.metadata)
            && task != TaskKind::Translation
        {
            Some("dedicated translation model cannot serve this role")
        } else if config.max_model_gib > 0.0
            && c.model.size as f64 > config.max_model_gib * 1_073_741_824.0
        {
            Some("exceeds configured model-file size budget")
        } else {
            None
        };
        if let Some(reason) = reason {
            rejected.push(format!("{}: {reason}", c.model.name));
            continue;
        }
        if let Err(error) =
            validate_parameter_cap(config, &c.model.name, &c.metadata, &c.model.details)
        {
            rejected.push(format!("{}: {error}", c.model.name));
            continue;
        }
        let name = c.model.name.to_lowercase();
        let canonical = canonical_model_name(&c.model.name);
        let gib = c.model.size as f64 / 1_073_741_824.0;
        let mut score = if config.favor_speed {
            -gib * 3.0
        } else {
            gib.min(12.0)
        };
        if let Some(index) = preferences
            .iter()
            .position(|n| canonical_model_name(n) == canonical)
        {
            score += 1000.0 - index as f64 * 20.0;
        } else if let Some(index) = config
            .general
            .iter()
            .position(|n| canonical_model_name(n) == canonical)
        {
            score += 200.0 - index as f64 * 10.0;
        }
        if canonical == canonical_model_name(selected) {
            score += 8.0;
        }
        if c.running && config.prefer_running {
            score += 12.0;
        }
        if task == TaskKind::Coding
            && ["coder", "codegemma", "starcoder", "codestral"]
                .iter()
                .any(|s| name.contains(s))
        {
            score += 40.0;
        }
        if task == TaskKind::Reasoning && has("thinking") {
            score += 25.0;
        }
        if task == TaskKind::Writing && !name.contains("coder") {
            score += 15.0;
        }
        if config.favor_speed {
            score += c
                .observed_tokens_per_second
                .unwrap_or(0.0)
                .clamp(0.0, 100.0)
                * 0.15;
        }
        scored.push((score, c));
    }
    scored.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.1.model.name.cmp(&b.1.model.name))
    });
    let (_,chosen)=scored.first().context(format!("No installed model satisfies routing requirements ({task:?}; tools={need_tools}). Rejections: {}. Adjust routing preferences/budget or install a compatible model.",rejected.join("; ")))?;
    let parameters = total_parameters(&chosen.metadata, &chosen.model.details);
    Ok(RouteDecision {
        model: chosen.model.name.clone(), task,
        reason: format!("Task {task:?}; verified capabilities [{}]; {:.2} GiB model file; {} parameters; {}. Selected using configured role preferences, current loading state and {} scoring.", chosen.capabilities.join(", "), chosen.model.size as f64 / 1_073_741_824.0, parameters.map(|n| format!("{:.3}B reported total", n as f64 / 1e9)).unwrap_or_else(|| "unknown total".into()), if chosen.running { "already loaded" } else { "will load on demand" }, if config.favor_speed { "size/speed" } else { "task/size" }),
        candidates_considered: candidates.len(), rejected,
        capabilities: chosen.capabilities.clone(), use_tools: need_tools && chosen.capabilities.iter().any(|cap| cap == "tools"),
        total_parameters: parameters,
        dedicated_translation: crate::translation::is_dedicated_model(&chosen.model.name, &chosen.metadata),
    })
}

/// Choose a model while excluding tags that have already failed a required
/// runtime protocol in the current request. Capability metadata is necessary,
/// but it is not proof that a particular model/template will actually emit a
/// native tool call for every prompt. Exclusions are deliberately scoped to
/// the caller's request; they do not permanently blacklist an installed model.
#[allow(clippy::too_many_arguments)]
pub fn choose_excluding(
    config: &RoutingConfig,
    task: TaskKind,
    need_tools: bool,
    allow_remote: bool,
    selected: &str,
    excluded: &[String],
    candidates: &[ModelCandidate],
) -> Result<RouteDecision> {
    if excluded.is_empty() {
        return choose(config, task, need_tools, allow_remote, selected, candidates);
    }
    let excluded: BTreeSet<String> = excluded
        .iter()
        .map(|model| canonical_model_name(model))
        .collect();
    let filtered: Vec<ModelCandidate> = candidates
        .iter()
        .filter(|candidate| !excluded.contains(&canonical_model_name(&candidate.model.name)))
        .cloned()
        .collect();
    if filtered.is_empty() {
        bail!(
            "No alternate installed model remains after excluding the current request's failed model(s): {}",
            excluded.into_iter().collect::<Vec<_>>().join(", ")
        );
    }
    let mut decision = choose(config, task, need_tools, allow_remote, selected, &filtered)?;
    let mut runtime_rejections = candidates
        .iter()
        .filter(|candidate| excluded.contains(&canonical_model_name(&candidate.model.name)))
        .map(|candidate| {
            format!(
                "{}: excluded after a runtime protocol failure in this request",
                candidate.model.name
            )
        })
        .collect::<Vec<_>>();
    runtime_rejections.append(&mut decision.rejected);
    decision.rejected = runtime_rejections;
    decision.candidates_considered = candidates.len();
    Ok(decision)
}
pub(crate) async fn cancelled(cancel: Arc<AtomicBool>) {
    while !cancel.load(Ordering::Relaxed) {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

pub async fn route(
    settings: &Settings,
    client: &OllamaClient,
    messages: &[ChatMessage],
    tools_available: bool,
    selected: &str,
    dir: &Path,
    cancel: Arc<AtomicBool>,
) -> Result<RouteDecision> {
    let task = classify(messages);
    // Translation specialists are first assessed without tool schemas. The
    // runtime validates their isolated input, or hands the full task back to a
    // tool-capable coordinator before any specialist request is sent.
    let need_tools = settings.tools_enabled && tools_available && task != TaskKind::Translation;
    // Skip models that recently failed native tool calling: choosing them
    // again costs a full wasted inference before the coordinator retry.
    let failed = if need_tools {
        recent_native_tool_failures(dir).unwrap_or_default()
    } else {
        Vec::new()
    };
    let candidates = inventory(settings, client, dir, cancel).await?;
    let config = config(settings)?;
    if let Some(previous) = settings.conversation_model.as_deref() {
        if let Some(candidate) = candidates.iter().find(|candidate| {
            canonical_model_name(&candidate.model.name) == canonical_model_name(previous)
        }) {
            // Continuity is a preference, subject to today's capability, size,
            // parameter, privacy and verified native-tool requirements.
            if let Ok(mut decision) = choose_excluding(
                &config,
                task,
                need_tools,
                settings.allow_remote,
                selected,
                &failed,
                std::slice::from_ref(candidate),
            ) {
                decision.reason = format!(
                    "Keeping {} for this conversation. {}",
                    decision.model, decision.reason
                );
                record_decision(dir, &decision)?;
                return Ok(decision);
            }
        }
    }
    let select = |excluded: &[String]| {
        route_from_candidates(
            settings,
            &config,
            task,
            need_tools,
            selected,
            excluded,
            dir,
            &candidates,
        )
    };
    if !failed.is_empty() {
        if let Ok(decision) = select(&failed) {
            return Ok(decision);
        }
    }
    select(&[])
}

#[allow(clippy::too_many_arguments)]
pub async fn route_for_task(
    settings: &Settings,
    client: &OllamaClient,
    task: TaskKind,
    need_tools: bool,
    selected: &str,
    dir: &Path,
    cancel: Arc<AtomicBool>,
) -> Result<RouteDecision> {
    route_for_task_excluding(
        settings,
        client,
        task,
        need_tools,
        selected,
        &[],
        dir,
        cancel,
    )
    .await
}

/// Route to a verified candidate other than models that already failed a
/// required runtime protocol in this request. This is used for bounded
/// failover; normal routing continues to call `route_for_task`.
#[allow(clippy::too_many_arguments)]
pub async fn route_for_task_excluding(
    settings: &Settings,
    client: &OllamaClient,
    task: TaskKind,
    need_tools: bool,
    selected: &str,
    excluded: &[String],
    dir: &Path,
    cancel: Arc<AtomicBool>,
) -> Result<RouteDecision> {
    let config = config(settings)?;
    let candidates = inventory(settings, client, dir, cancel).await?;
    route_from_candidates(
        settings,
        &config,
        task,
        need_tools,
        selected,
        excluded,
        dir,
        &candidates,
    )
}

#[allow(clippy::too_many_arguments)]
fn route_from_candidates(
    settings: &Settings,
    config: &RoutingConfig,
    task: TaskKind,
    need_tools: bool,
    selected: &str,
    excluded: &[String],
    dir: &Path,
    candidates: &[ModelCandidate],
) -> Result<RouteDecision> {
    let mut decision = choose_excluding(
        config,
        task,
        need_tools,
        settings.allow_remote,
        selected,
        excluded,
        candidates,
    )?;
    match crate::benchmarks::choose_measured(
        settings, config, task, need_tools, excluded, candidates, dir,
    ) {
        Ok(Some(measured)) => decision = measured,
        Ok(None) => {
            if config.use_benchmarks {
                decision.reason.push_str(" No fresh fully passing benchmark covers this task and tool requirement; configured fallback used.");
            }
        }
        Err(error) => decision.reason.push_str(&format!(
            " Benchmark evidence unavailable; using configured fallback: {error}"
        )),
    }
    record_decision(dir, &decision)?;
    Ok(decision)
}

/// Inspect only installed tags; routing never silently downloads a model.
pub async fn inventory(
    settings: &Settings,
    client: &OllamaClient,
    dir: &Path,
    cancel: Arc<AtomicBool>,
) -> Result<Vec<ModelCandidate>> {
    let models = tokio::select! {r=client.list_models()=>r?,_=cancelled(cancel.clone())=>bail!("Model routing cancelled")};
    let running = tokio::select! {r=client.running_models()=>r.unwrap_or(Value::Null),_=cancelled(cancel.clone())=>bail!("Model routing cancelled")};
    let loaded = running
        .get("models")
        .and_then(Value::as_array)
        .map(|v| {
            v.iter()
                .filter_map(|m| {
                    m.get("name")
                        .or_else(|| m.get("model"))
                        .and_then(Value::as_str)
                        .map(canonical_model_name)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let metrics = observations(dir).unwrap_or_default();
    let mut candidates = vec![];
    for model in models.into_iter().take(128) {
        if !settings.allow_remote && model.is_remote() {
            continue;
        }
        let metadata = tokio::select! {r=client.show(&model.name)=>r,_=cancelled(cancel.clone())=>bail!("Model routing cancelled")};
        if let Ok(metadata) = metadata {
            let capabilities = metadata
                .get("capabilities")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            let running = loaded.contains(&canonical_model_name(&model.name));
            let speed = metrics.get(&model.name).copied();
            candidates.push(ModelCandidate {
                model,
                capabilities,
                metadata,
                running,
                observed_tokens_per_second: speed,
            });
        }
    }
    Ok(candidates)
}

pub fn record_decision(dir: &Path, decision: &RouteDecision) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let temp = dir.join(format!("route-{}.tmp", uuid::Uuid::new_v4()));
    std::fs::write(&temp, serde_json::to_vec_pretty(&decision)?)?;
    std::fs::rename(temp, dir.join("last-route.json"))?;
    Ok(())
}
fn metric_db(dir: &Path) -> Result<Connection> {
    std::fs::create_dir_all(dir)?;
    let c = Connection::open(dir.join("routing.sqlite3"))?;
    c.busy_timeout(Duration::from_secs(5))?;
    c.execute_batch("PRAGMA journal_mode=WAL; CREATE TABLE IF NOT EXISTS observations(model TEXT PRIMARY KEY, tps REAL NOT NULL, samples INTEGER NOT NULL, updated_at TEXT NOT NULL); CREATE TABLE IF NOT EXISTS native_tool_failures(model TEXT PRIMARY KEY, failures INTEGER NOT NULL, updated_at TEXT NOT NULL); CREATE TABLE IF NOT EXISTS native_tool_successes(model TEXT PRIMARY KEY, successes INTEGER NOT NULL, updated_at TEXT NOT NULL);")?;
    Ok(c)
}
/// How long a native tool-calling failure keeps a model out of tool routing.
/// It expires so an updated model or template gets another chance.
const NATIVE_TOOL_FAILURE_DAYS: i64 = 14;

/// Remember that `model` answered a tool turn with text-form tool JSON instead
/// of native tool calls, so routing stops choosing it for tool turns.
pub fn record_native_tool_failure(dir: &Path, model: &str) -> Result<()> {
    metric_db(dir)?.execute(
        "INSERT INTO native_tool_failures(model, failures, updated_at) VALUES(?1, 1, ?2)
         ON CONFLICT(model) DO UPDATE SET failures = failures + 1, updated_at = excluded.updated_at",
        params![model, chrono::Utc::now().to_rfc3339()],
    )?;
    Ok(())
}

/// Remember that `model` made a real native tool call that executed, so its tool
/// support is verified by use rather than assumed from metadata.
pub fn record_native_tool_success(dir: &Path, model: &str) -> Result<()> {
    metric_db(dir)?.execute(
        "INSERT INTO native_tool_successes(model, successes, updated_at) VALUES(?1, 1, ?2)
         ON CONFLICT(model) DO UPDATE SET successes = successes + 1, updated_at = excluded.updated_at",
        params![model, chrono::Utc::now().to_rfc3339()],
    )?;
    Ok(())
}

/// What Workbench has observed about a model's native tool calling, as opposed to
/// what its metadata claims.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeToolRecord {
    /// Never seen making (or failing) a native tool call.
    Unverified,
    /// Its most recent observation was a working native tool call.
    Verified { successes: i64 },
    /// Its most recent observation was tool JSON printed as text.
    Failed { failures: i64 },
}

pub fn native_tool_record(dir: &Path, model: &str) -> Result<NativeToolRecord> {
    let c = metric_db(dir)?;
    let row = |table: &str, column: &str| -> Result<Option<(i64, String)>> {
        let mut statement = c.prepare(&format!(
            "SELECT {column}, updated_at FROM {table} WHERE model = ?1"
        ))?;
        let mut rows = statement.query(params![model])?;
        Ok(match rows.next()? {
            Some(row) => Some((row.get(0)?, row.get(1)?)),
            None => None,
        })
    };
    let success = row("native_tool_successes", "successes")?;
    let failure = row("native_tool_failures", "failures")?;
    Ok(match (success, failure) {
        (None, None) => NativeToolRecord::Unverified,
        (Some((successes, _)), None) => NativeToolRecord::Verified { successes },
        (None, Some((failures, _))) => NativeToolRecord::Failed { failures },
        (Some((successes, won)), Some((failures, lost))) => {
            if won >= lost {
                NativeToolRecord::Verified { successes }
            } else {
                NativeToolRecord::Failed { failures }
            }
        }
    })
}

/// Send `model` one tiny request with a single tool and see whether it answers
/// with a native tool call. Some(true): it did. Some(false): it wrote the call as
/// text instead. None: inconclusive (it answered in prose), nothing is recorded.
pub async fn probe_native_tools(
    client: &crate::ollama::OllamaClient,
    model: &str,
    keep_alive: &str,
    cancel: Arc<AtomicBool>,
) -> Result<Option<bool>> {
    let tool = serde_json::json!({"type":"function","function":{
        "name":"report_ready",
        "description":"Report that you are ready.",
        "parameters":{"type":"object","properties":{"ready":{"type":"boolean"}},"required":["ready"]}
    }});
    let messages = [
        crate::ollama::ChatMessage {
            role: "system".into(),
            content: "This is a tool-calling check. Respond only by calling the provided tool."
                .into(),
            ..Default::default()
        },
        crate::ollama::ChatMessage {
            role: "user".into(),
            content: "Call the report_ready tool with ready set to true.".into(),
            ..Default::default()
        },
    ];
    let options = crate::ollama::ChatOptions {
        temperature: 0.0,
        num_ctx: 4096,
        num_predict: 512,
        keep_alive: keep_alive.into(),
        ..Default::default()
    };
    let reply = client
        .chat_stream(model, &messages, &options, &[tool], cancel, |_| {})
        .await?;
    if reply
        .tool_calls
        .iter()
        .any(|call| call.function.name == "report_ready")
    {
        Ok(Some(true))
    } else if reply.content.contains("report_ready") {
        Ok(Some(false))
    } else {
        Ok(None)
    }
}

/// Models with a recent native tool-calling failure that has not been followed by
/// a working native tool call.
pub fn recent_native_tool_failures(dir: &Path) -> Result<Vec<String>> {
    let cutoff =
        (chrono::Utc::now() - chrono::Duration::days(NATIVE_TOOL_FAILURE_DAYS)).to_rfc3339();
    let c = metric_db(dir)?;
    let mut statement = c.prepare(
        "SELECT f.model FROM native_tool_failures f
         LEFT JOIN native_tool_successes s ON s.model = f.model
         WHERE f.updated_at >= ?1 AND (s.updated_at IS NULL OR s.updated_at < f.updated_at)
         ORDER BY f.model",
    )?;
    let models = statement
        .query_map(params![cutoff], |row| row.get::<_, String>(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(models)
}

pub fn observe(dir: &Path, model: &str, stats: &Value) -> Result<()> {
    let count = stats
        .get("eval_count")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let duration = stats
        .get("eval_duration")
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
        / 1e9;
    if count <= 0.0 || duration <= 0.0 {
        return Ok(());
    }
    let tps = count / duration;
    if !tps.is_finite() {
        return Ok(());
    }
    let c = metric_db(dir)?;
    let previous: Option<f64> = c
        .query_row(
            "SELECT tps FROM observations WHERE model=?1",
            [model],
            |r| r.get(0),
        )
        .optional()?;
    let value = previous.map(|v| v * 0.75 + tps * 0.25).unwrap_or(tps);
    c.execute("INSERT INTO observations VALUES(?1,?2,1,?3) ON CONFLICT(model) DO UPDATE SET tps=excluded.tps,samples=observations.samples+1,updated_at=excluded.updated_at",params![model,value,chrono::Utc::now().to_rfc3339()])?;
    Ok(())
}
pub fn observations(dir: &Path) -> Result<BTreeMap<String, f64>> {
    let c = metric_db(dir)?;
    let mut s = c.prepare("SELECT model,tps FROM observations")?;
    let rows = s
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_later_native_tool_success_clears_an_earlier_failure() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            native_tool_record(dir.path(), "m").unwrap(),
            NativeToolRecord::Unverified
        );
        record_native_tool_failure(dir.path(), "m").unwrap();
        assert_eq!(
            native_tool_record(dir.path(), "m").unwrap(),
            NativeToolRecord::Failed { failures: 1 }
        );
        assert_eq!(recent_native_tool_failures(dir.path()).unwrap(), ["m"]);
        std::thread::sleep(std::time::Duration::from_millis(5));
        record_native_tool_success(dir.path(), "m").unwrap();
        assert_eq!(
            native_tool_record(dir.path(), "m").unwrap(),
            NativeToolRecord::Verified { successes: 1 }
        );
        assert!(recent_native_tool_failures(dir.path()).unwrap().is_empty());
    }

    #[test]
    fn native_tool_failures_are_remembered_and_expire() {
        let dir = tempfile::tempdir().unwrap();
        assert!(recent_native_tool_failures(dir.path()).unwrap().is_empty());
        record_native_tool_failure(dir.path(), "coder:7b").unwrap();
        record_native_tool_failure(dir.path(), "coder:7b").unwrap();
        assert_eq!(
            recent_native_tool_failures(dir.path()).unwrap(),
            vec!["coder:7b"]
        );
        let old = (chrono::Utc::now() - chrono::Duration::days(NATIVE_TOOL_FAILURE_DAYS + 1))
            .to_rfc3339();
        metric_db(dir.path())
            .unwrap()
            .execute(
                "UPDATE native_tool_failures SET updated_at = ?1",
                params![old],
            )
            .unwrap();
        assert!(recent_native_tool_failures(dir.path()).unwrap().is_empty());
    }
    fn c(name: &str, caps: &[&str], gib: f64) -> ModelCandidate {
        ModelCandidate {
            model: ModelInfo {
                name: name.into(),
                size: (gib * 1_073_741_824.0) as u64,
                ..Default::default()
            },
            capabilities: caps.iter().map(|s| s.to_string()).collect(),
            metadata: json!({}),
            running: false,
            observed_tokens_per_second: None,
        }
    }
    #[test]
    fn capability_requirements_override_preference() {
        let cfg = RoutingConfig {
            vision: vec!["fast".into()],
            ..Default::default()
        };
        let models = vec![
            c("fast", &["completion", "tools"], 2.0),
            c("vision", &["completion", "tools", "vision"], 4.0),
        ];
        assert_eq!(
            choose(&cfg, TaskKind::Vision, true, false, "fast", &models)
                .unwrap()
                .model,
            "vision"
        );
        assert!(choose(&cfg, TaskKind::Vision, true, false, "", &models[..1]).is_err());
    }
    #[test]
    fn cloud_alias_and_budget_are_enforced() {
        let mut remote = c("innocent", &["completion"], 1.0);
        remote.metadata = json!({"remote_host":"https://ollama.com"});
        let candidates = vec![
            remote,
            c("huge", &["completion"], 20.0),
            c("local", &["completion"], 4.0),
        ];
        let result = choose(
            &RoutingConfig::default(),
            TaskKind::General,
            false,
            false,
            "innocent",
            &candidates,
        )
        .unwrap();
        assert_eq!(result.model, "local");
        assert_eq!(result.rejected.len(), 2);
    }
    #[test]
    fn coding_preferences_and_observations_work() {
        let models = vec![
            c("generic", &["completion", "tools"], 4.0),
            c("qwen-coder", &["completion", "tools"], 4.0),
        ];
        assert_eq!(
            choose(
                &RoutingConfig::default(),
                TaskKind::Coding,
                true,
                false,
                "generic",
                &models
            )
            .unwrap()
            .model,
            "qwen-coder"
        );
        let d = tempfile::tempdir().unwrap();
        observe(
            d.path(),
            "qwen-coder",
            &json!({"eval_count":40,"eval_duration":2_000_000_000_u64}),
        )
        .unwrap();
        assert_eq!(observations(d.path()).unwrap()["qwen-coder"], 20.0);
    }

    #[test]
    fn runtime_protocol_failure_excludes_the_failed_model_for_bounded_failover() {
        let cfg = RoutingConfig {
            orchestrator: vec!["preferred".into(), "backup".into()],
            ..Default::default()
        };
        let candidates = vec![
            c("preferred", &["completion", "tools"], 4.0),
            c("backup", &["completion", "tools"], 3.0),
        ];
        let decision = choose_excluding(
            &cfg,
            TaskKind::Orchestrator,
            true,
            false,
            "preferred",
            &["preferred:latest".into()],
            &candidates,
        )
        .unwrap();
        assert_eq!(decision.model, "backup");
        assert_eq!(decision.candidates_considered, 2);
        assert!(decision.rejected[0].contains("runtime protocol failure"));
        assert!(choose_excluding(
            &cfg,
            TaskKind::Orchestrator,
            true,
            false,
            "preferred",
            &["preferred".into(), "backup".into()],
            &candidates,
        )
        .is_err());
    }

    #[test]
    fn parameter_cap_counts_total_moe_and_projector_not_file_bytes() {
        let cfg = RoutingConfig {
            max_parameters_b: 10.0,
            general: vec!["tiny-file".into()],
            ..Default::default()
        };
        let mut huge = c("tiny-file", &["completion", "tools"], 0.1);
        huge.metadata = json!({"model_info":{"general.parameter_count":30_000_000_000_u64,"qwen3moe.expert_used_count":8},"details":{"parameter_size":"3B"}});
        let mut good = c("small", &["completion", "tools"], 4.0);
        good.metadata = json!({"model_info":{"general.parameter_count":4_210_000_000_u64},"projector_info":{"general.parameter_count":334_000_000}});
        assert_eq!(
            total_parameters(&good.metadata, &Value::Null),
            Some(4_544_000_000)
        );
        let result = choose(
            &cfg,
            TaskKind::General,
            true,
            false,
            "tiny-file",
            &[huge, good],
        )
        .unwrap();
        assert_eq!(result.model, "small");
        assert!(result.rejected[0].contains("30.000B"));
        assert!(result.use_tools);
    }

    #[test]
    fn parameter_metadata_fails_closed_and_legacy_config_is_compatible() {
        assert_eq!(
            total_parameters(&json!({}), &json!({"parameter_size":"7.6 B"})),
            Some(7_600_000_000)
        );
        for metadata in [
            json!({"details":{"parameter_size":"8x7B"}}),
            json!({"details":{"parameter_size":"3B active"}}),
            json!({"model_info":{"qwen3moe.expert_count":128},"details":{"parameter_size":"3B"}}),
            json!({"model_info":{"general.architecture":"qwen3moe"},"details":{"parameter_size":"3B"}}),
            json!({"model_info":{"general.parameter_count":"bad"},"details":{"parameter_size":"7B"}}),
            json!({"model_info":{"general.parameter_count":4_000_000_000_u64},"projector_info":{"general.name":"unknown projector"}}),
        ] {
            assert_eq!(total_parameters(&metadata, &Value::Null), None);
        }
        let unknown = c("unknown", &["completion"], 1.0);
        let legacy: RoutingConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(legacy.max_parameters_b, 0.0);
        assert!(choose(
            &legacy,
            TaskKind::General,
            false,
            false,
            "",
            std::slice::from_ref(&unknown)
        )
        .is_ok());
        let capped = RoutingConfig {
            max_parameters_b: 10.0,
            ..legacy
        };
        assert!(choose(&capped, TaskKind::General, false, false, "", &[unknown]).is_err());
        assert!(validate_parameter_cap(
            &capped,
            "rounded",
            &json!({"details":{"parameter_size":"10B"}}),
            &Value::Null
        )
        .is_err());
    }

    #[test]
    fn classifier_uses_words_and_has_a_translation_role() {
        let task = |text: &str| {
            classify(&[ChatMessage {
                role: "user".into(),
                content: text.into(),
                ..Default::default()
            }])
        };
        assert_eq!(
            task("Translate this paragraph from English to French: Hello"),
            TaskKind::Translation
        );
        assert_eq!(
            task("Implement a translation function in Rust"),
            TaskKind::Coding
        );
        assert_eq!(
            task("A rustic unreasonable discordant greeting"),
            TaskKind::General
        );
        assert_eq!(task("Analyze the evidence"), TaskKind::Reasoning);
        assert_eq!(task("Summarize this chapter"), TaskKind::Writing);
    }

    #[test]
    fn roles_use_exact_canonical_tags_and_capability_fallbacks() {
        assert_eq!(
            canonical_model_name("registry.ollama.ai/library/Qwen3.5:4b-Q4_K_M"),
            "qwen3.5:4b-q4_k_m"
        );
        assert_eq!(
            canonical_model_name("custom/team-model"),
            "custom/team-model:latest"
        );
        let config = RoutingConfig {
            orchestrator: vec!["registry.ollama.ai/library/coordinator".into()],
            translation: vec!["translategemma:4b".into()],
            general: vec!["coordinator:latest".into()],
            coding: vec!["coder:7b".into()],
            ..Default::default()
        };
        let models = vec![
            c("coordinator:latest", &["completion", "tools"], 2.0),
            c("translategemma:4b", &["completion"], 3.0),
            c("coder:14b", &["completion", "tools"], 7.0),
        ];
        let translator = choose(&config, TaskKind::Translation, false, false, "", &models).unwrap();
        assert_eq!(translator.model, "translategemma:4b");
        assert!(!translator.use_tools);
        assert!(translator.dedicated_translation);
        assert_eq!(
            choose(&config, TaskKind::Translation, true, false, "", &models)
                .unwrap()
                .model,
            "coordinator:latest"
        );
        assert_eq!(
            choose(
                &config,
                TaskKind::Orchestrator,
                false,
                false,
                "translategemma:4b",
                &models
            )
            .unwrap()
            .model,
            "coordinator:latest"
        );
        assert_eq!(
            choose(&config, TaskKind::Coding, true, false, "", &models)
                .unwrap()
                .model,
            "coordinator:latest"
        );
    }
}
