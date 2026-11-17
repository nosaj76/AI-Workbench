//! Reviewable, resumable installation of a bounded local specialist-model pack.
//! The selected Ollama server owns weights and partial downloads. This module
//! never deletes models, edits application settings, or activates an unverified
//! model. Successful pull EOF is followed by independent metadata validation.

use crate::{
    agents::{self, RunLease},
    ollama::{ModelInfo, OllamaClient},
    routing::{canonical_model_name, validate_parameter_cap, RoutingConfig},
    settings::Settings,
};
use anyhow::{bail, Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    future::Future,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

pub const MAX_SETUP_PARAMETERS_B: f64 = 10.0;
const MAX_MODELS: usize = 12;
const STATE_DIRECTORY: &str = "model-setup";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelRole {
    Orchestrator,
    General,
    Coding,
    Reasoning,
    Translation,
    Writing,
    Vision,
    Embedding,
}

impl fmt::Display for ModelRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Orchestrator => "Orchestrator",
            Self::General => "General",
            Self::Coding => "Code",
            Self::Reasoning => "Analysis / reasoning",
            Self::Translation => "Translation",
            Self::Writing => "Writing",
            Self::Vision => "Vision",
            Self::Embedding => "Retrieval embeddings",
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Candidate {
    pub id: String,
    pub tag: String,
    pub roles: Vec<ModelRole>,
    /// Documented total size, including a separate vision projector when known.
    /// It is a catalog estimate; actual metadata is checked independently.
    pub total_parameters_b: f64,
    /// Rounded per-model download/storage estimate, not a free-space guarantee.
    pub download_bytes: u64,
    pub license: String,
    pub source_url: String,
    pub required_capabilities: Vec<String>,
    #[serde(default)]
    pub minimum_ollama: Option<String>,
    /// Explicit, catalog-reviewed compatible alternatives. Metadata and
    /// quantization are verified before one is reused instead of the pinned tag.
    #[serde(default)]
    pub installed_aliases: Vec<String>,
    #[serde(default)]
    pub required_quantization: Option<String>,
}

#[derive(Clone)]
pub struct SetupRequest {
    pub settings: Settings,
    pub data_dir: PathBuf,
    pub candidates: Vec<Candidate>,
    pub max_parameters_b: f64,
    pub required_roles: Vec<ModelRole>,
}

impl SetupRequest {
    pub fn new(settings: Settings, data_dir: PathBuf, candidates: Vec<Candidate>) -> Self {
        Self {
            settings,
            data_dir,
            candidates,
            max_parameters_b: MAX_SETUP_PARAMETERS_B,
            required_roles: vec![
                ModelRole::Orchestrator,
                ModelRole::General,
                ModelRole::Coding,
                ModelRole::Reasoning,
                ModelRole::Translation,
                ModelRole::Writing,
                ModelRole::Vision,
            ],
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.settings.validate_endpoint()?;
        normalize_endpoint(&self.settings.base_url)?;
        validate_selection(
            &self.candidates,
            self.max_parameters_b,
            &self.required_roles,
        )
    }

    pub fn request_hash(&self) -> Result<String> {
        self.validate()?;
        identity(
            &normalize_endpoint(&self.settings.base_url)?,
            &self.candidates,
            self.max_parameters_b,
            &self.required_roles,
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelVerification {
    pub model_name: String,
    pub total_parameters: u64,
    pub capabilities: Vec<String>,
    pub size_bytes: u64,
    pub digest: String,
    pub quantization: Option<String>,
    pub details: Value,
    /// Retains the raw fields used to establish the total parameter count.
    pub parameter_evidence: Value,
    pub minimum_ollama: Option<String>,
    pub verified_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlannedModel {
    pub candidate: Candidate,
    pub installed_name: Option<String>,
    pub verification: Option<ModelVerification>,
    /// An incompatible exact tag must be corrected deliberately; setup will
    /// not overwrite it. An incompatible alternative is only a warning.
    pub issue: Option<String>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SetupPlan {
    pub id: String,
    pub endpoint: String,
    pub request_hash: String,
    pub ollama_version: String,
    pub max_parameters_b: f64,
    pub required_roles: Vec<ModelRole>,
    pub models: Vec<PlannedModel>,
    pub estimated_download_bytes: u64,
    pub total_disk_bytes: u64,
    pub warnings: Vec<String>,
    pub created_at: String,
}

impl SetupPlan {
    pub fn can_run(&self) -> bool {
        !self.models.is_empty() && self.models.iter().all(|m| m.issue.is_none())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SetupState {
    Running,
    Complete,
    Failed,
    Cancelled,
    Interrupted,
}

impl fmt::Display for SetupState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Running => "Running",
            Self::Complete => "Complete",
            Self::Failed => "Failed",
            Self::Cancelled => "Cancelled",
            Self::Interrupted => "Interrupted",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSetupStatus {
    Queued,
    Checking,
    Downloading,
    Verifying,
    Ready,
    Failed,
    Cancelled,
}

impl fmt::Display for ModelSetupStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Queued => "Queued",
            Self::Checking => "Checking",
            Self::Downloading => "Downloading",
            Self::Verifying => "Verifying",
            Self::Ready => "Verified",
            Self::Failed => "Failed",
            Self::Cancelled => "Cancelled",
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SetupProgress {
    pub tag: Option<String>,
    /// One-based current model number; zero denotes a pack-level operation.
    pub model_index: usize,
    pub model_count: usize,
    pub stage: String,
    /// Ollama reports these counts for the current content layer, not the
    /// entire pack. The digest identifies that layer when it is available.
    pub completed_bytes: u64,
    pub total_bytes: Option<u64>,
    pub digest: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelSetupState {
    pub candidate: Candidate,
    pub state: ModelSetupStatus,
    pub reused: bool,
    pub verification: Option<ModelVerification>,
    pub attempts: u32,
    pub error: Option<String>,
    pub progress: Option<SetupProgress>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SetupRun {
    pub id: String,
    pub endpoint: String,
    pub request_hash: String,
    pub ollama_version: String,
    pub max_parameters_b: f64,
    pub required_roles: Vec<ModelRole>,
    pub state: SetupState,
    pub models: Vec<ModelSetupState>,
    pub error: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

impl SetupRun {
    /// A report state alone is insufficient: each selected model, its source
    /// digest, parameter cap, capabilities, and required role must also agree.
    pub fn completed(&self) -> bool {
        self.state == SetupState::Complete && self.verified_roles().is_ok()
    }

    pub fn verified_models(&self) -> Result<BTreeMap<ModelRole, Vec<String>>> {
        if self.state != SetupState::Complete {
            bail!("The model pack is not complete; routing settings must not be activated");
        }
        self.verified_roles()
    }

    fn verified_roles(&self) -> Result<BTreeMap<ModelRole, Vec<String>>> {
        validate_record(self)?;
        check_catalog_versions(
            &self
                .models
                .iter()
                .map(|m| m.candidate.clone())
                .collect::<Vec<_>>(),
            &self.ollama_version,
        )?;
        let mut roles: BTreeMap<ModelRole, Vec<String>> = BTreeMap::new();
        for model in &self.models {
            if model.state != ModelSetupStatus::Ready {
                bail!("Model {} is not verified", model.candidate.tag);
            }
            let verification = model
                .verification
                .as_ref()
                .context("Ready model is missing its metadata verification")?;
            validate_verification(&model.candidate, verification, self.max_parameters_b)?;
            if let Some(minimum) = &verification.minimum_ollama {
                require_version(&self.ollama_version, minimum)?;
            }
            for role in &model.candidate.roles {
                let names = roles.entry(*role).or_default();
                if !names.contains(&verification.model_name) {
                    names.push(verification.model_name.clone());
                }
            }
        }
        if self
            .required_roles
            .iter()
            .any(|role| !roles.contains_key(role))
        {
            bail!("The verified pack does not cover every required response role");
        }
        Ok(roles)
    }
}

/// Inspect the selected endpoint and already-installed models. No pull, model
/// load, model deletion, or settings mutation occurs while preparing a plan.
pub async fn prepare(
    request: &SetupRequest,
    cancel: Arc<AtomicBool>,
    mut on_progress: impl FnMut(SetupProgress),
) -> Result<SetupPlan> {
    request.validate()?;
    check_cancel(&cancel)?;
    let endpoint = normalize_endpoint(&request.settings.base_url)?;
    let client = OllamaClient::new(&endpoint)?;
    emit(
        &mut on_progress,
        None,
        0,
        request.candidates.len(),
        "Checking Ollama version",
    );
    let version = cancellable(&cancel, client.version()).await?;
    check_catalog_versions(&request.candidates, &version)?;
    let listed = cancellable(&cancel, client.list_models()).await?;
    let mut models = Vec::new();
    let mut missing = 0_u64;
    let mut total = 0_u64;
    for (index, candidate) in request.candidates.iter().enumerate() {
        emit(
            &mut on_progress,
            Some(&candidate.tag),
            index + 1,
            request.candidates.len(),
            "Inspecting installed models",
        );
        let inspected = inspect_installed(
            &client,
            &listed,
            candidate,
            request.max_parameters_b,
            &version,
            &cancel,
        )
        .await;
        check_cancel(&cancel)?;
        let (verification, issue, warnings) = match inspected {
            Ok((v, warnings)) => (v, None, warnings),
            Err(error) => (None, Some(format!("{error:#}")), Vec::new()),
        };
        if verification.is_none() && issue.is_none() {
            missing = missing
                .checked_add(candidate.download_bytes)
                .context("Download estimate overflow")?;
        }
        total = total
            .checked_add(
                verification
                    .as_ref()
                    .map_or(candidate.download_bytes, |v| v.size_bytes),
            )
            .context("Disk estimate overflow")?;
        models.push(PlannedModel {
            candidate: candidate.clone(),
            installed_name: verification.as_ref().map(|v| v.model_name.clone()),
            verification,
            issue,
            warnings,
        });
    }
    check_cancel(&cancel)?;
    let hash = request.request_hash()?;
    Ok(SetupPlan {
        id: hash.clone(),
        endpoint,
        request_hash: hash,
        ollama_version: version,
        max_parameters_b: request.max_parameters_b,
        required_roles: request.required_roles.clone(),
        models,
        estimated_download_bytes: missing,
        total_disk_bytes: total,
        warnings: vec!["Sizes are per-model estimates. Shared layers may reduce downloads and disk use; free space belongs to the selected Ollama server and is not measured by this client.".into()],
        created_at: now(),
    })
}

/// Execute the reviewed endpoint/catalog snapshot, one model at a time.
/// Operational failures return a saved Failed/Cancelled report; validation,
/// locking, or checkpoint-storage failures return an error. Call completed()
/// before applying any role assignments or treating installation as successful.
pub async fn run(
    request: &SetupRequest,
    plan: &SetupPlan,
    cancel: Arc<AtomicBool>,
    mut on_progress: impl FnMut(SetupProgress),
) -> Result<SetupRun> {
    request.validate()?;
    check_cancel(&cancel)?;
    let endpoint = normalize_endpoint(&request.settings.base_url)?;
    let hash = request.request_hash()?;
    if plan.request_hash != hash
        || plan.id != hash
        || plan.endpoint != endpoint
        || plan.models.len() != request.candidates.len()
        || plan
            .models
            .iter()
            .zip(&request.candidates)
            .any(|(m, c)| &m.candidate != c)
        || plan.max_parameters_b != request.max_parameters_b
        || plan.required_roles != request.required_roles
    {
        bail!("The model selection or endpoint changed after review. Prepare a new setup plan");
    }
    if !plan.can_run() {
        bail!("The plan contains incompatible installed models. Correct the reported issues and prepare the plan again; setup never overwrites or deletes those models automatically");
    }
    let directory = request.data_dir.join(STATE_DIRECTORY);
    let _lease = RunLease::try_acquire(&directory, &endpoint_lock_id(&endpoint))?
        .context("Another model setup is already running for this Ollama endpoint")?;
    let mut state = match read_record(&request.data_dir, &hash)? {
        Some(mut previous) => {
            if previous.request_hash != hash || previous.endpoint != endpoint {
                bail!("Saved model setup does not match this endpoint and catalog");
            }
            previous.state = SetupState::Running;
            previous.error = None;
            for model in &mut previous.models {
                // Previous metadata remains inspectable until revalidation,
                // but it is not counted as current readiness across a resume.
                model.state = ModelSetupStatus::Queued;
                model.error = None;
                model.progress = None;
            }
            previous
        }
        None => SetupRun {
            id: hash.clone(),
            request_hash: hash,
            endpoint,
            ollama_version: plan.ollama_version.clone(),
            max_parameters_b: request.max_parameters_b,
            required_roles: request.required_roles.clone(),
            state: SetupState::Running,
            models: request
                .candidates
                .iter()
                .map(|candidate| ModelSetupState {
                    candidate: candidate.clone(),
                    state: ModelSetupStatus::Queued,
                    reused: false,
                    verification: None,
                    attempts: 0,
                    error: None,
                    progress: None,
                })
                .collect(),
            error: None,
            created_at: now(),
            updated_at: now(),
        },
    };
    persist(&request.data_dir, &mut state)?;
    let client = OllamaClient::new(&state.endpoint)?;
    let outcome = execute(request, &client, &cancel, &mut state, &mut on_progress).await;
    match outcome {
        Ok(()) => {
            state.state = SetupState::Complete;
            state.verified_models()?;
        }
        Err(error) => {
            let cancelled = cancel.load(Ordering::Relaxed);
            state.state = if cancelled {
                SetupState::Cancelled
            } else {
                SetupState::Failed
            };
            state.error = Some(format!("{error:#}"));
            for model in &mut state.models {
                if matches!(
                    model.state,
                    ModelSetupStatus::Checking
                        | ModelSetupStatus::Downloading
                        | ModelSetupStatus::Verifying
                ) {
                    model.state = if cancelled {
                        ModelSetupStatus::Cancelled
                    } else {
                        ModelSetupStatus::Failed
                    };
                    model.error = state.error.clone();
                }
            }
        }
    }
    persist(&request.data_dir, &mut state)?;
    emit(
        &mut on_progress,
        None,
        0,
        state.models.len(),
        &format!("Model setup {}", state.state),
    );
    Ok(state)
}

async fn execute(
    request: &SetupRequest,
    client: &OllamaClient,
    cancel: &Arc<AtomicBool>,
    state: &mut SetupRun,
    on_progress: &mut impl FnMut(SetupProgress),
) -> Result<()> {
    let version = cancellable(cancel, client.version()).await?;
    check_catalog_versions(&request.candidates, &version)?;
    state.ollama_version = version.clone();
    for index in 0..state.models.len() {
        check_cancel(cancel)?;
        let candidate = state.models[index].candidate.clone();
        state.models[index].state = ModelSetupStatus::Checking;
        state.models[index].attempts = state.models[index].attempts.saturating_add(1);
        state.models[index].verification = None;
        persist(&request.data_dir, state)?;
        emit(
            on_progress,
            Some(&candidate.tag),
            index + 1,
            state.models.len(),
            "Rechecking installed models",
        );
        let listed = cancellable(cancel, client.list_models()).await?;
        let (installed, _) = inspect_installed(
            client,
            &listed,
            &candidate,
            request.max_parameters_b,
            &version,
            cancel,
        )
        .await?;
        let (verification, reused) = if let Some(verification) = installed {
            (verification, true)
        } else {
            state.models[index].state = ModelSetupStatus::Downloading;
            persist(&request.data_dir, state)?;
            emit(
                on_progress,
                Some(&candidate.tag),
                index + 1,
                state.models.len(),
                "Downloading missing model",
            );
            let pull_cancel = Arc::new(AtomicBool::new(false));
            let mut checkpoint_error = None;
            let mut last_checkpoint = Instant::now();
            let mut last_stage = String::new();
            let count = state.models.len();
            let pulled = cancellable(
                cancel,
                client.pull(&candidate.tag, pull_cancel.clone(), |record| {
                    let progress = SetupProgress {
                        tag: Some(candidate.tag.clone()),
                        model_index: index + 1,
                        model_count: count,
                        stage: record
                            .get("status")
                            .and_then(Value::as_str)
                            .unwrap_or("Downloading")
                            .chars()
                            .take(2048)
                            .collect(),
                        completed_bytes: record
                            .get("completed")
                            .and_then(Value::as_u64)
                            .unwrap_or(0),
                        total_bytes: record
                            .get("total")
                            .and_then(Value::as_u64)
                            .filter(|n| *n > 0),
                        digest: record
                            .get("digest")
                            .and_then(Value::as_str)
                            .map(|s| s.chars().take(160).collect()),
                    };
                    state.models[index].progress = Some(progress.clone());
                    let changed = progress.stage != last_stage;
                    last_stage.clone_from(&progress.stage);
                    on_progress(progress);
                    if checkpoint_error.is_none()
                        && (changed || last_checkpoint.elapsed() >= Duration::from_secs(1))
                    {
                        last_checkpoint = Instant::now();
                        if let Err(error) = persist(&request.data_dir, state) {
                            checkpoint_error = Some(error);
                            pull_cancel.store(true, Ordering::Relaxed);
                        }
                    }
                }),
            )
            .await;
            if let Some(error) = checkpoint_error {
                return Err(error).context("Save model download checkpoint");
            }
            pulled.with_context(|| format!("Download {}", candidate.tag))?;
            check_cancel(cancel)?;
            state.models[index].state = ModelSetupStatus::Verifying;
            persist(&request.data_dir, state)?;
            emit(
                on_progress,
                Some(&candidate.tag),
                index + 1,
                state.models.len(),
                "Verifying downloaded model metadata",
            );
            let listed = cancellable(cancel, client.list_models()).await?;
            let model = find_model(&listed, &candidate.tag)?.context("Ollama reported pull success but the model is absent from its installed-model list")?;
            let verified = verify_model(
                client,
                model,
                &candidate,
                request.max_parameters_b,
                &version,
                cancel,
            )
            .await?;
            (verified, false)
        };
        state.models[index].state = ModelSetupStatus::Ready;
        state.models[index].verification = Some(verification);
        state.models[index].reused = reused;
        state.models[index].error = None;
        persist(&request.data_dir, state)?;
        emit(
            on_progress,
            Some(&candidate.tag),
            index + 1,
            state.models.len(),
            if reused {
                "Verified installed model; no download needed"
            } else {
                "Downloaded model verified"
            },
        );
    }
    // A different client can change a model while later members download.
    // Recheck the complete pack before granting the single ready-to-apply state.
    emit(
        on_progress,
        None,
        0,
        state.models.len(),
        "Checking the complete pack before activation",
    );
    let listed = cancellable(cancel, client.list_models()).await?;
    for index in 0..state.models.len() {
        let expected = state.models[index]
            .verification
            .clone()
            .context("Verified model is missing metadata")?;
        let result = async {
            let model = find_model(&listed, &expected.model_name)?
                .context("A verified model was removed before setup completed")?;
            verify_model(
                client,
                model,
                &state.models[index].candidate,
                request.max_parameters_b,
                &version,
                cancel,
            )
            .await
        }
        .await;
        match result {
            Ok(verification) if verification.digest == expected.digest => {
                state.models[index].verification = Some(verification);
            }
            Ok(_) => {
                state.models[index].state = ModelSetupStatus::Failed;
                state.models[index].error = Some("Model content changed while the pack was being installed; review the selection again".into());
                bail!("{} changed before setup completed", expected.model_name);
            }
            Err(error) => {
                state.models[index].state = if cancel.load(Ordering::Relaxed) {
                    ModelSetupStatus::Cancelled
                } else {
                    ModelSetupStatus::Failed
                };
                state.models[index].error = Some(format!("{error:#}"));
                return Err(error);
            }
        }
    }
    check_cancel(cancel)?;
    Ok(())
}

async fn inspect_installed(
    client: &OllamaClient,
    listed: &[ModelInfo],
    candidate: &Candidate,
    cap: f64,
    version: &str,
    cancel: &Arc<AtomicBool>,
) -> Result<(Option<ModelVerification>, Vec<String>)> {
    if let Some(model) = find_model(listed, &candidate.tag)? {
        return verify_model(client, model, candidate, cap, version, cancel)
            .await
            .map(|v| (Some(v), Vec::new()))
            .with_context(|| {
                format!(
                    "Installed exact tag {} is incompatible; it has not been changed",
                    candidate.tag
                )
            });
    }
    let mut warnings = Vec::new();
    for alias in &candidate.installed_aliases {
        if let Some(model) = find_model(listed, alias)? {
            match verify_model(client, model, candidate, cap, version, cancel).await {
                Ok(verification) => return Ok((Some(verification), warnings)),
                Err(error) => {
                    check_cancel(cancel)?;
                    warnings.push(format!(
                        "Installed alternative {alias} was not reused: {error:#}"
                    ));
                }
            }
        }
    }
    Ok((None, warnings))
}

async fn verify_model(
    client: &OllamaClient,
    model: &ModelInfo,
    candidate: &Candidate,
    cap: f64,
    version: &str,
    cancel: &Arc<AtomicBool>,
) -> Result<ModelVerification> {
    if model.is_remote() {
        bail!(
            "{} is a cloud/remote model alias; this setup only installs local weights",
            model.name
        );
    }
    let initial_digest = model_digest(model)?;
    let metadata = cancellable(cancel, client.show(&model.name))
        .await
        .with_context(|| format!("Inspect model {}", model.name))?;
    for field in ["remote_host", "remote_model"] {
        if metadata
            .get(field)
            .is_some_and(|v| !v.is_null() && v.as_str() != Some(""))
        {
            bail!("{} resolves to remote model metadata", model.name);
        }
    }
    let cap_policy = RoutingConfig {
        max_parameters_b: cap,
        ..Default::default()
    };
    let parameters = validate_parameter_cap(&cap_policy, &model.name, &metadata, &model.details)?
        .with_context(|| format!("{} does not expose an unambiguous total parameter count; a tag name or MoE active count is insufficient", model.name))?;
    let capabilities = metadata
        .get("capabilities")
        .and_then(Value::as_array)
        .context("Ollama model metadata is missing capabilities; update Ollama before setup")?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .context("Ollama returned an invalid capability value")
        })
        .collect::<Result<Vec<_>>>()?;
    let required = metadata
        .get("requires")
        .filter(|v| !v.is_null())
        .map(|v| v.as_str().context("Invalid model minimum-version metadata"))
        .transpose()?
        .filter(|s| !s.trim().is_empty())
        .map(str::to_owned);
    if let Some(required) = &required {
        require_version(version, required)
            .with_context(|| format!("{} requires a newer Ollama runtime", model.name))?;
    }
    let details = metadata
        .get("details")
        .cloned()
        .unwrap_or_else(|| model.details.clone());
    if capabilities.len() > 32
        || capabilities.iter().any(|c| c.len() > 64)
        || serde_json::to_vec(&details)?.len() > 32_768
    {
        bail!("Ollama returned oversized model capability/details metadata");
    }
    let quantization = details
        .get("quantization_level")
        .and_then(Value::as_str)
        .or_else(|| {
            model
                .details
                .get("quantization_level")
                .and_then(Value::as_str)
        })
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    let evidence = json!({
        "model_info": parameter_fields(metadata.get("model_info")),
        "projector_info": parameter_fields(metadata.get("projector_info")),
        "details": size_evidence(metadata.get("details")),
        "listed_details": size_evidence(Some(&model.details)),
    });
    if serde_json::to_vec(&evidence)?.len() > 32_768 {
        bail!("Ollama returned oversized parameter-count metadata");
    }
    let verification = ModelVerification {
        model_name: model.name.clone(),
        total_parameters: parameters,
        capabilities,
        size_bytes: model.size,
        digest: initial_digest.clone(),
        quantization,
        parameter_evidence: evidence,
        details,
        minimum_ollama: required,
        verified_at: now(),
    };
    validate_verification(candidate, &verification, cap)?;
    let after = cancellable(cancel, client.list_models()).await?;
    let after = find_model(&after, &model.name)?
        .context("Model disappeared during metadata verification")?;
    if model_digest(after)? != initial_digest || after.size != model.size || after.is_remote() {
        bail!(
            "{} changed during metadata verification; refresh the setup plan",
            model.name
        );
    }
    check_cancel(cancel)?;
    Ok(verification)
}

fn validate_verification(candidate: &Candidate, v: &ModelVerification, cap: f64) -> Result<()> {
    let cap_policy = RoutingConfig {
        max_parameters_b: cap,
        ..Default::default()
    };
    if validate_parameter_cap(
        &cap_policy,
        &v.model_name,
        &v.parameter_evidence,
        &v.parameter_evidence["listed_details"],
    )? != Some(v.total_parameters)
    {
        bail!(
            "{} has inconsistent saved parameter-count evidence; inspect the model again",
            v.model_name
        );
    }
    if v.total_parameters == 0 || v.total_parameters as f64 > cap * 1_000_000_000.0 {
        bail!(
            "{} has {:.3}B total parameters, above the {:.3}B setup cap",
            v.model_name,
            v.total_parameters as f64 / 1_000_000_000.0,
            cap
        );
    }
    let canonical = canonical_model_name(&v.model_name);
    if canonical != canonical_model_name(&candidate.tag)
        && !candidate
            .installed_aliases
            .iter()
            .any(|a| canonical_model_name(a) == canonical)
    {
        bail!("Verified model name is outside the reviewed catalog selection");
    }
    validate_tag(&v.model_name)?;
    if v.size_bytes == 0 || !valid_digest(&v.digest) {
        bail!(
            "{} has no valid installed size or SHA-256 content digest",
            v.model_name
        );
    }
    let mut caps: BTreeSet<&str> = candidate
        .required_capabilities
        .iter()
        .map(String::as_str)
        .collect();
    for role in &candidate.roles {
        match role {
            ModelRole::Embedding => {
                caps.insert("embedding");
            }
            ModelRole::Vision => {
                caps.insert("completion");
                caps.insert("vision");
            }
            ModelRole::Orchestrator => {
                caps.insert("completion");
                caps.insert("tools");
            }
            _ => {
                caps.insert("completion");
            }
        }
    }
    for capability in caps {
        if !v.capabilities.iter().any(|c| c == capability) {
            bail!(
                "{} does not confirm the required {capability} capability",
                v.model_name
            );
        }
    }
    if let Some(required) = &candidate.required_quantization {
        if !v
            .quantization
            .as_ref()
            .is_some_and(|q| q.eq_ignore_ascii_case(required))
        {
            bail!(
                "{} uses {:?} quantization; the reviewed pack requires {required}",
                v.model_name,
                v.quantization
            );
        }
    }
    Ok(())
}

fn parameter_fields(value: Option<&Value>) -> Value {
    let Some(fields) = value.and_then(Value::as_object) else {
        return Value::Null;
    };
    Value::Object(
        fields
            .iter()
            .filter(|(key, _)| key.contains("parameter_count") || key.contains("expert"))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    )
}

fn size_evidence(details: Option<&Value>) -> Value {
    match details.and_then(|v| v.get("parameter_size")) {
        Some(size) => json!({"parameter_size":size}),
        None => Value::Null,
    }
}

fn find_model<'a>(listed: &'a [ModelInfo], name: &str) -> Result<Option<&'a ModelInfo>> {
    let key = canonical_model_name(name);
    let mut matching = listed
        .iter()
        .filter(|m| canonical_model_name(&m.name) == key);
    let first = matching.next();
    if matching.next().is_some() {
        bail!(
            "Installed model names are ambiguous for {name}; refresh or correct the model catalog"
        );
    }
    Ok(first)
}

fn model_digest(model: &ModelInfo) -> Result<String> {
    let digest = model
        .extra
        .get("digest")
        .and_then(Value::as_str)
        .filter(|s| valid_digest(s))
        .with_context(|| format!("{} has no verifiable SHA-256 model digest", model.name))?;
    Ok(digest
        .strip_prefix("sha256:")
        .unwrap_or(digest)
        .to_ascii_lowercase())
}

fn valid_digest(digest: &str) -> bool {
    let digest = digest.strip_prefix("sha256:").unwrap_or(digest);
    digest.len() == 64 && digest.bytes().all(|c| c.is_ascii_hexdigit())
}

fn validate_selection(candidates: &[Candidate], cap: f64, required: &[ModelRole]) -> Result<()> {
    if !cap.is_finite() || cap <= 0.0 || cap > MAX_SETUP_PARAMETERS_B {
        bail!("Starter setup requires a positive total-parameter cap no larger than 10B");
    }
    if candidates.is_empty() || candidates.len() > MAX_MODELS {
        bail!("Select between one and twelve unique specialist models");
    }
    if required.is_empty() || required.len() > 8 {
        bail!("Select between one and eight required response roles");
    }
    let mut ids = BTreeSet::new();
    let mut names = BTreeSet::new();
    let mut provided = BTreeSet::new();
    for candidate in candidates {
        agents::validate_id(&candidate.id)?;
        if !ids.insert(&candidate.id) {
            bail!("Duplicate specialist catalog ID {}", candidate.id);
        }
        if !candidate.total_parameters_b.is_finite()
            || candidate.total_parameters_b <= 0.0
            || candidate.total_parameters_b > cap
        {
            bail!("Catalog model {} exceeds the total-parameter cap or lacks a valid documented count", candidate.tag);
        }
        if candidate.download_bytes == 0 || candidate.download_bytes > 64 * 1_073_741_824 {
            bail!(
                "Catalog model {} has an invalid download estimate",
                candidate.tag
            );
        }
        if candidate.roles.is_empty()
            || candidate.roles.len() > 8
            || candidate.installed_aliases.len() > 12
        {
            bail!("Catalog model role/alternative list is empty or over its bound");
        }
        if candidate.license.trim().is_empty() || candidate.license.len() > 2048 {
            bail!("Catalog models require a bounded, explicit license description");
        }
        let source =
            url::Url::parse(&candidate.source_url).context("Invalid catalog source URL")?;
        if candidate.source_url.len() > 2048
            || source.scheme() != "https"
            || source.host_str().is_none()
            || !source.username().is_empty()
            || source.password().is_some()
        {
            bail!("Catalog sources must be credential-free HTTPS documentation URLs");
        }
        if candidate.required_capabilities.len() > 8
            || candidate.required_capabilities.iter().any(|c| {
                !matches!(
                    c.as_str(),
                    "completion" | "tools" | "thinking" | "vision" | "embedding"
                )
            })
        {
            bail!("Catalog model capabilities contain an unsupported value");
        }
        if candidate.required_quantization.as_ref().is_some_and(|q| {
            q.is_empty()
                || q.len() > 64
                || !q.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
        }) {
            bail!("Invalid catalog quantization requirement");
        }
        if let Some(minimum) = &candidate.minimum_ollama {
            parse_version(minimum)?;
        }
        let mut candidate_names = BTreeSet::new();
        for name in std::iter::once(&candidate.tag).chain(&candidate.installed_aliases) {
            validate_tag(name)?;
            let canonical = canonical_model_name(name);
            if candidate_names.insert(canonical.clone()) && !names.insert(canonical) {
                bail!("Two specialist catalog entries share the same model tag or alternative");
            }
        }
        provided.extend(candidate.roles.iter().copied());
    }
    if required.iter().any(|r| !provided.contains(r)) {
        bail!("The selected pack does not cover all required response roles");
    }
    Ok(())
}

fn validate_tag(tag: &str) -> Result<()> {
    let lower = tag.to_ascii_lowercase();
    if tag.is_empty()
        || tag.len() > 256
        || !tag.is_ascii()
        || tag.contains("//")
        || tag
            .split('/')
            .any(|p| p.is_empty() || p == "." || p == "..")
        || !tag
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.' | b'/' | b':'))
        || !tag.rsplit('/').next().is_some_and(|part| {
            part.split_once(':').is_some_and(|(name, variant)| {
                !name.is_empty() && !variant.is_empty() && !variant.contains(':')
            })
        })
        || lower.ends_with("-cloud")
        || lower.contains(":cloud")
    {
        bail!(
            "Starter models require a bounded explicit local model:tag, not a cloud alias or URL"
        );
    }
    Ok(())
}

pub(crate) fn normalize_endpoint(endpoint: &str) -> Result<String> {
    Ok(crate::endpoint::normalize(endpoint)?.to_string())
}

fn identity(
    endpoint: &str,
    candidates: &[Candidate],
    cap: f64,
    required: &[ModelRole],
) -> Result<String> {
    let bytes = serde_json::to_vec(
        &json!({"version":1,"endpoint":endpoint,"candidates":candidates,"max_parameters_b":cap,"required_roles":required}),
    )?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn endpoint_lock_id(endpoint: &str) -> String {
    format!("endpoint-{:x}", Sha256::digest(endpoint))
}

fn check_catalog_versions(candidates: &[Candidate], version: &str) -> Result<()> {
    for candidate in candidates {
        if let Some(minimum) = &candidate.minimum_ollama {
            require_version(version, minimum).with_context(|| {
                format!(
                    "{} needs a newer Ollama runtime before model download",
                    candidate.tag
                )
            })?;
        }
    }
    Ok(())
}

fn parse_version(value: &str) -> Result<((u64, u64, u64), bool)> {
    if value.len() > 128 {
        bail!("Invalid Ollama version");
    }
    let value = value.trim().strip_prefix('v').unwrap_or(value.trim());
    let value = value.split('+').next().unwrap_or(value);
    let (core, prerelease) = value
        .split_once('-')
        .map_or((value, false), |(c, _)| (c, true));
    let parts = core.split('.').collect::<Vec<_>>();
    if parts.len() != 3
        || parts
            .iter()
            .any(|p| p.is_empty() || !p.bytes().all(|c| c.is_ascii_digit()))
    {
        bail!("Cannot verify Ollama version {value:?}; a major.minor.patch version is required");
    }
    Ok((
        (parts[0].parse()?, parts[1].parse()?, parts[2].parse()?),
        prerelease,
    ))
}

fn require_version(actual: &str, required: &str) -> Result<()> {
    let (actual_version, prerelease) = parse_version(actual)?;
    let (required_version, _) = parse_version(required)?;
    if actual_version < required_version || (actual_version == required_version && prerelease) {
        bail!("Ollama {actual} is too old for this pack; install stable Ollama {required} or newer through Services or update the selected server");
    }
    Ok(())
}

fn validate_record(run: &SetupRun) -> Result<()> {
    let candidates = run
        .models
        .iter()
        .map(|m| m.candidate.clone())
        .collect::<Vec<_>>();
    validate_selection(&candidates, run.max_parameters_b, &run.required_roles)?;
    if run.id.len() != 64
        || !run.id.bytes().all(|c| c.is_ascii_hexdigit())
        || run.request_hash != run.id
        || normalize_endpoint(&run.endpoint)? != run.endpoint
        || identity(
            &run.endpoint,
            &candidates,
            run.max_parameters_b,
            &run.required_roles,
        )? != run.id
    {
        bail!("Saved setup identity does not match its endpoint, policy and catalog; the original record was preserved");
    }
    Ok(())
}

fn read_record(data_dir: &Path, id: &str) -> Result<Option<SetupRun>> {
    if id.len() != 64 || !id.bytes().all(|c| c.is_ascii_hexdigit()) {
        bail!("Invalid model setup checkpoint ID");
    }
    let path = data_dir.join(STATE_DIRECTORY).join(format!("{id}.json"));
    if !path.try_exists()? {
        return Ok(None);
    }
    let run: SetupRun = agents::read_json(&path)?;
    validate_record(&run)?;
    if run.id != id {
        bail!("Model setup checkpoint ID does not match its filename");
    }
    Ok(Some(run))
}

/// Loads one endpoint/catalog checkpoint. An orphaned Running record is marked
/// Interrupted only after acquiring its endpoint lease and re-reading it.
pub fn load_run(data_dir: &Path, id: &str) -> Result<Option<SetupRun>> {
    let Some(mut state) = read_record(data_dir, id)? else {
        return Ok(None);
    };
    if state.state == SetupState::Running {
        if let Some(_lease) = RunLease::try_acquire(
            &data_dir.join(STATE_DIRECTORY),
            &endpoint_lock_id(&state.endpoint),
        )? {
            state =
                read_record(data_dir, id)?.context("Model setup disappeared during recovery")?;
            if state.state == SetupState::Running {
                state.state = SetupState::Interrupted;
                state.error = Some("The setup worker ended before its final checkpoint. Completed models and Ollama's partial downloads are retained; review and resume this pack.".into());
                for model in &mut state.models {
                    if matches!(
                        model.state,
                        ModelSetupStatus::Checking
                            | ModelSetupStatus::Downloading
                            | ModelSetupStatus::Verifying
                    ) {
                        model.state = ModelSetupStatus::Queued;
                        model.error =
                            Some("Interrupted before metadata verification completed".into());
                    }
                }
                persist(data_dir, &mut state)?;
            }
        }
    }
    Ok(Some(state))
}

fn persist(data_dir: &Path, run: &mut SetupRun) -> Result<()> {
    run.updated_at = now();
    agents::write_json(
        &data_dir.join(STATE_DIRECTORY),
        &format!("{}.json", run.id),
        run,
    )
}

fn now() -> String {
    Utc::now().to_rfc3339()
}

fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        bail!(
            "Model setup cancelled; completed models and resumable partial downloads were retained"
        );
    }
    Ok(())
}

async fn wait_cancel(cancel: &AtomicBool) {
    while !cancel.load(Ordering::Relaxed) {
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
}

async fn cancellable<T>(
    cancel: &Arc<AtomicBool>,
    future: impl Future<Output = Result<T>>,
) -> Result<T> {
    check_cancel(cancel)?;
    tokio::select! {
        biased;
        _ = wait_cancel(cancel) => bail!("Model setup cancelled; completed models and resumable partial downloads were retained"),
        result = future => result,
    }
}

fn emit(
    callback: &mut impl FnMut(SetupProgress),
    tag: Option<&str>,
    index: usize,
    count: usize,
    stage: &str,
) {
    callback(SetupProgress {
        tag: tag.map(str::to_owned),
        model_index: index,
        model_count: count,
        stage: stage.into(),
        completed_bytes: 0,
        total_bytes: None,
        digest: None,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(tag: &str, roles: Vec<ModelRole>) -> Candidate {
        Candidate {
            id: tag.replace(':', "-"),
            tag: tag.into(),
            roles,
            total_parameters_b: 4.6,
            download_bytes: 3_300_000_000,
            license: "Apache-2.0".into(),
            source_url: "https://ollama.com/library/fixture".into(),
            required_capabilities: vec!["completion".into()],
            minimum_ollama: Some("0.17.5".into()),
            installed_aliases: vec![],
            required_quantization: Some("Q4_K_M".into()),
        }
    }

    #[test]
    fn selection_rejects_oversized_cloud_duplicate_and_uncovered_models() {
        let candidate = model("local:4b", vec![ModelRole::General]);
        validate_selection(
            std::slice::from_ref(&candidate),
            10.0,
            &[ModelRole::General],
        )
        .unwrap();
        let mut large = candidate.clone();
        large.total_parameters_b = 10.01;
        assert!(validate_selection(&[large], 10.0, &[ModelRole::General]).is_err());
        let mut cloud = candidate.clone();
        cloud.tag = "local:4b-cloud".into();
        assert!(validate_selection(&[cloud], 10.0, &[ModelRole::General]).is_err());
        assert!(validate_selection(
            &[candidate.clone(), candidate.clone()],
            10.0,
            &[ModelRole::General]
        )
        .is_err());
        assert!(validate_selection(&[candidate], 10.0, &[ModelRole::Translation]).is_err());
    }

    #[test]
    fn setup_identity_tracks_endpoint_catalog_and_policy_but_not_chat_settings() {
        let mut request = SetupRequest::new(
            Settings::default(),
            PathBuf::from("/unused"),
            vec![model("local:4b", vec![ModelRole::General])],
        );
        request.required_roles = vec![ModelRole::General];
        let original = request.request_hash().unwrap();
        request.settings.temperature = 0.25;
        request.settings.system_prompt = "different private prompt".into();
        request.settings.base_url = "http://127.0.0.1:11434/api/".into();
        assert_eq!(request.request_hash().unwrap(), original);
        request.settings.base_url = "http://127.0.0.1:11435".into();
        assert_ne!(request.request_hash().unwrap(), original);
        request.settings.base_url = "http://127.0.0.1:11434".into();
        request.candidates[0].required_quantization = Some("Q8_0".into());
        assert_ne!(request.request_hash().unwrap(), original);
    }

    #[test]
    fn version_comparison_uses_numeric_components_and_rejects_prerelease_at_minimum() {
        require_version("0.18.0", "0.17.5").unwrap();
        require_version("v0.17.5+vendor", "0.17.5").unwrap();
        assert!(require_version("0.9.9", "0.17.5").is_err());
        assert!(require_version("0.17.5-rc1", "0.17.5").is_err());
        assert!(require_version("latest", "0.17.5").is_err());
    }

    #[test]
    fn running_checkpoints_are_recovered_only_after_the_endpoint_lease_is_released() {
        let temp = tempfile::tempdir().unwrap();
        let candidate = model("local:4b", vec![ModelRole::General]);
        let endpoint = normalize_endpoint("http://127.0.0.1:11434").unwrap();
        let required = vec![ModelRole::General];
        let id = identity(&endpoint, std::slice::from_ref(&candidate), 10.0, &required).unwrap();
        let mut record = SetupRun {
            id: id.clone(),
            endpoint: endpoint.clone(),
            request_hash: id.clone(),
            ollama_version: "0.18.0".into(),
            max_parameters_b: 10.0,
            required_roles: required,
            state: SetupState::Running,
            models: vec![ModelSetupState {
                candidate,
                state: ModelSetupStatus::Downloading,
                reused: false,
                verification: None,
                attempts: 1,
                error: None,
                progress: None,
            }],
            error: None,
            created_at: now(),
            updated_at: now(),
        };
        let lease = RunLease::try_acquire(
            &temp.path().join(STATE_DIRECTORY),
            &endpoint_lock_id(&endpoint),
        )
        .unwrap()
        .unwrap();
        persist(temp.path(), &mut record).unwrap();
        assert_eq!(
            load_run(temp.path(), &id).unwrap().unwrap().state,
            SetupState::Running
        );
        drop(lease);
        let recovered = load_run(temp.path(), &id).unwrap().unwrap();
        assert_eq!(recovered.state, SetupState::Interrupted);
        assert_eq!(recovered.models[0].state, ModelSetupStatus::Queued);
        assert!(!recovered.completed());
        assert!(recovered.verified_models().is_err());
    }
}
