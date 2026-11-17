//! Reviewed starter models and atomic application of a verified setup result.
//!
//! Sizes and parameter totals are conservative catalog estimates. The setup
//! backend verifies the installed model before any of these role assignments
//! become settings. See `docs/MODEL_SETUP_ROUTING.md` for source evidence.

use crate::{
    model_setup::{normalize_endpoint, Candidate, ModelRole, SetupRun},
    routing::{canonical_model_name, RoutingConfig},
    settings::Settings,
};
use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;

const GENERAL_TAG: &str = "qwen3.5:4b-q4_K_M";
const CODING_TAG: &str = "qwen2.5-coder:7b-instruct-q4_K_M";
const TRANSLATION_TAG: &str = "translategemma:4b-it-q4_K_M";
const REASONING_TAG: &str = "qwen3.5:9b-q4_K_M";
const REASONING_ALIAS: &str = "qwen3.5:9b";
const EMBEDDING_TAG: &str = "qwen3-embedding:0.6b-q8_0";

#[derive(Clone, Debug)]
pub struct CatalogEntry {
    pub candidate: Candidate,
    pub title: String,
    pub description: String,
    pub optional: bool,
}

/// The first three entries form the default pack. Optional entries are not
/// selected implicitly. Aliases were checked against the official Linux
/// llama.cpp manifests on 2026-11-07 and still require local verification.
pub fn catalog() -> Vec<CatalogEntry> {
    vec![
        CatalogEntry {
            candidate: Candidate {
                id: "qwen35_4b".into(),
                tag: GENERAL_TAG.into(),
                roles: vec![
                    ModelRole::Orchestrator,
                    ModelRole::General,
                    ModelRole::Reasoning,
                    ModelRole::Writing,
                    ModelRole::Vision,
                ],
                // 4.21B model + 334M separate vision projector, rounded up.
                total_parameters_b: 4.6,
                download_bytes: 3_300_000_000,
                license: "Apache-2.0".into(),
                source_url: format!("https://ollama.com/library/{GENERAL_TAG}"),
                required_capabilities: strings(&["completion", "tools", "thinking", "vision"]),
                minimum_ollama: Some("0.17.5".into()),
                installed_aliases: strings(&["qwen3.5:4b"]),
                required_quantization: Some("Q4_K_M".into()),
            },
            title: "Qwen3.5 4B — coordinator and general assistant".into(),
            description: "Plans work and handles general answers, analysis, writing, and images. Shared across five roles to keep the initial download compact.".into(),
            optional: false,
        },
        CatalogEntry {
            candidate: Candidate {
                id: "qwen25_coder_7b".into(),
                tag: CODING_TAG.into(),
                roles: vec![ModelRole::Coding],
                total_parameters_b: 7.7,
                download_bytes: 4_700_000_000,
                license: "Apache-2.0".into(),
                source_url: format!("https://ollama.com/library/{CODING_TAG}"),
                required_capabilities: strings(&["completion", "tools"]),
                minimum_ollama: None,
                installed_aliases: strings(&["qwen2.5-coder:7b"]),
                required_quantization: Some("Q4_K_M".into()),
            },
            title: "Qwen2.5-Coder 7B — coding specialist".into(),
            description: "Handles code questions, implementation, and project review with tool support. An installed compatible 7B alias can be reused.".into(),
            optional: false,
        },
        CatalogEntry {
            candidate: Candidate {
                id: "translategemma_4b".into(),
                tag: TRANSLATION_TAG.into(),
                roles: vec![ModelRole::Translation],
                total_parameters_b: 4.4,
                download_bytes: 3_300_000_000,
                license: "Gemma Terms of Use".into(),
                source_url: format!("https://ollama.com/library/{TRANSLATION_TAG}"),
                required_capabilities: strings(&["completion", "vision"]),
                minimum_ollama: None,
                installed_aliases: strings(&["translategemma:4b"]),
                required_quantization: Some("Q4_K_M".into()),
            },
            title: "TranslateGemma 4B — translation specialist".into(),
            description: "Translates short passages through a dedicated language-aware adapter. Review the Gemma Terms of Use before downloading.".into(),
            optional: false,
        },
        CatalogEntry {
            candidate: Candidate {
                id: "qwen35_9b".into(),
                tag: REASONING_TAG.into(),
                roles: vec![ModelRole::Reasoning],
                // 8.95B model + 456M separate vision projector, rounded up.
                total_parameters_b: 9.5,
                download_bytes: 6_600_000_000,
                license: "Apache-2.0".into(),
                source_url: format!("https://ollama.com/library/{REASONING_TAG}"),
                required_capabilities: strings(&["completion", "tools", "thinking", "vision"]),
                minimum_ollama: Some("0.17.5".into()),
                installed_aliases: strings(&[REASONING_ALIAS]),
                required_quantization: Some("Q4_K_M".into()),
            },
            title: "Qwen3.5 9B — optional larger analysis model".into(),
            description: "Becomes the first analysis preference when verified. Uses more memory and leaves less room for context on an 8 GB GPU; speed and quality depend on the task.".into(),
            optional: true,
        },
        CatalogEntry {
            candidate: Candidate {
                id: "qwen3_embedding_06b".into(),
                tag: EMBEDDING_TAG.into(),
                roles: vec![ModelRole::Embedding],
                total_parameters_b: 0.6,
                download_bytes: 639_000_000,
                license: "Apache-2.0".into(),
                source_url: format!("https://ollama.com/library/{EMBEDDING_TAG}"),
                required_capabilities: strings(&["embedding"]),
                minimum_ollama: None,
                installed_aliases: strings(&["qwen3-embedding:0.6b"]),
                required_quantization: Some("Q8_0".into()),
            },
            title: "Qwen3-Embedding 0.6B — optional semantic retrieval".into(),
            description: "Creates vectors for multilingual document and code retrieval. It is not a chat model; configuring it does not index files or enable retrieval automatically.".into(),
            optional: true,
        },
    ]
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

/// Build new settings only from a completed, fully verified setup. The caller
/// persists the returned settings; this function never writes or downloads.
/// An incomplete run cannot leave partially applied role assignments behind.
pub fn configured_settings(
    existing: &Settings,
    run: &SetupRun,
    context_tokens: u32,
) -> Result<Settings> {
    let verified = run
        .verified_models()
        .context("Complete and verify every selected setup model before applying its routes")?;
    existing.validate_endpoint()?;
    ensure_matching_endpoint(&existing.base_url, &run.endpoint)?;
    settings_from_verified(existing, &verified, context_tokens)
}

fn ensure_matching_endpoint(current: &str, verified: &str) -> Result<()> {
    if normalize_endpoint(current)? != verified {
        bail!("The Ollama endpoint changed after setup; verify the pack on the selected server before applying its routes");
    }
    Ok(())
}

fn settings_from_verified(
    existing: &Settings,
    verified: &BTreeMap<ModelRole, Vec<String>>,
    context_tokens: u32,
) -> Result<Settings> {
    if !matches!(context_tokens, 4096 | 8192) {
        bail!("Starter setup supports a context of 4096 or 8192 tokens");
    }

    let role_models = |role: ModelRole| -> Result<Vec<String>> {
        let models = verified
            .get(&role)
            .filter(|models| !models.is_empty())
            .with_context(|| format!("Setup is missing a verified {role} model"))?;
        if models.iter().any(|model| model.trim().is_empty()) {
            bail!("Setup contains an empty verified model name for {role}");
        }
        Ok(models.clone())
    };

    // Resolve every required role before changing even the cloned settings.
    let general = role_models(ModelRole::General)?;
    let coding = role_models(ModelRole::Coding)?;
    let mut reasoning = role_models(ModelRole::Reasoning)?;
    let translation = role_models(ModelRole::Translation)?;
    let writing = role_models(ModelRole::Writing)?;
    let vision = role_models(ModelRole::Vision)?;
    let orchestrator = role_models(ModelRole::Orchestrator)?;

    // A successfully verified optional model outranks the shared 4B fallback.
    // Keep the actual installed alias returned by the setup backend.
    let larger_reasoner_names = [
        canonical_model_name(REASONING_TAG),
        canonical_model_name(REASONING_ALIAS),
    ];
    reasoning.sort_by_key(|model| !larger_reasoner_names.contains(&canonical_model_name(model)));

    let mut routing: RoutingConfig =
        serde_json::from_str(&existing.routing_json).unwrap_or_default();
    routing.general = general;
    routing.coding = coding;
    routing.reasoning = reasoning;
    routing.translation = translation;
    routing.writing = writing;
    routing.vision = vision;
    routing.orchestrator = orchestrator.clone();
    routing.max_parameters_b = 10.0;
    routing.max_model_gib = 8.0;
    routing.prefer_running = true;
    routing.single_model_mode = true;

    let mut configured = existing.clone();
    configured.model = orchestrator[0].clone();
    configured.routing_json = serde_json::to_string_pretty(&routing)?;
    configured.model_routing = true;
    configured.agents_enabled = true;
    configured.agent_max_workers = 1;
    configured.num_ctx = context_tokens;
    configured.model_setup_seen = true;

    if let Some(models) = verified.get(&ModelRole::Embedding) {
        if let Some(model) = models.first() {
            if model.trim().is_empty() {
                bail!("Setup contains an empty verified embedding model name");
            }
            configured.embedding_model = model.clone();
        }
    }
    Ok(configured)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_setup::SetupState;
    use crate::settings::Profile;

    fn verified_aliases() -> BTreeMap<ModelRole, Vec<String>> {
        let mut verified = BTreeMap::new();
        for role in [
            ModelRole::Orchestrator,
            ModelRole::General,
            ModelRole::Reasoning,
            ModelRole::Writing,
            ModelRole::Vision,
        ] {
            verified.insert(role, strings(&["qwen3.5:4b"]));
        }
        verified.insert(ModelRole::Coding, strings(&["qwen2.5-coder:7b"]));
        verified.insert(ModelRole::Translation, strings(&["translategemma:4b"]));
        verified
    }

    #[test]
    fn missing_role_cannot_apply_a_partial_setup() {
        let settings = Settings {
            model: "existing:user-choice".into(),
            embedding_model: "existing:embedding".into(),
            ..Default::default()
        };
        let before = serde_json::to_value(&settings).unwrap();
        let mut verified = verified_aliases();
        verified.remove(&ModelRole::Translation);

        let error = settings_from_verified(&settings, &verified, 4096).unwrap_err();
        assert!(error.to_string().to_lowercase().contains("translation"));
        assert_eq!(serde_json::to_value(&settings).unwrap(), before);
    }

    #[test]
    fn a_completion_flag_without_a_verified_receipt_cannot_apply_settings() {
        let settings = Settings::default();
        let mut run = SetupRun {
            id: "unfinished-setup".into(),
            endpoint: "http://127.0.0.1:11434/".into(),
            request_hash: String::new(),
            ollama_version: "0.17.5".into(),
            max_parameters_b: 10.0,
            required_roles: vec![ModelRole::Orchestrator],
            state: SetupState::Running,
            models: vec![],
            error: None,
            created_at: String::new(),
            updated_at: String::new(),
        };
        assert!(configured_settings(&settings, &run, 4096).is_err());
        run.state = SetupState::Complete;
        assert!(configured_settings(&settings, &run, 4096).is_err());
    }

    #[test]
    fn a_setup_receipt_cannot_configure_a_different_server() {
        let verified = "http://127.0.0.1:11434/";
        ensure_matching_endpoint("http://127.0.0.1:11434", verified).unwrap();
        ensure_matching_endpoint("http://127.0.0.1:11434/api/", verified).unwrap();
        assert!(ensure_matching_endpoint("http://127.0.0.1:11435", verified).is_err());
        assert!(ensure_matching_endpoint("http://127.0.0.1:11434/proxy", verified).is_err());
    }

    #[test]
    fn verified_aliases_and_optional_reasoner_preserve_unrelated_preferences() {
        let settings = Settings {
            embedding_model: "existing:embedding".into(),
            temperature: 0.35,
            memory: "User-maintained notes".into(),
            network_tools: true,
            agent_max_tasks: 7,
            routing_json: r#"{"favor_speed":true}"#.into(),
            profiles: vec![Profile {
                name: "My writing profile".into(),
                model: "existing:profile-model".into(),
                instruction: "Preserve my voice".into(),
            }],
            ..Default::default()
        };
        let mut verified = verified_aliases();
        let installed_reasoner = "registry.ollama.ai/library/qwen3.5:9b";
        verified
            .get_mut(&ModelRole::Reasoning)
            .unwrap()
            .push(installed_reasoner.into());

        let configured = settings_from_verified(&settings, &verified, 4096).unwrap();
        let routing: RoutingConfig = serde_json::from_str(&configured.routing_json).unwrap();
        assert_eq!(configured.model, "qwen3.5:4b");
        assert_eq!(routing.coding, ["qwen2.5-coder:7b"]);
        assert_eq!(routing.translation, ["translategemma:4b"]);
        assert_eq!(routing.reasoning, [installed_reasoner, "qwen3.5:4b"]);
        assert_eq!(routing.orchestrator, ["qwen3.5:4b"]);
        assert!(routing.favor_speed && routing.prefer_running && routing.single_model_mode);
        assert_eq!(routing.max_parameters_b, 10.0);
        assert_eq!(configured.num_ctx, 4096);
        assert_eq!(configured.agent_max_workers, 1);
        assert!(
            configured.model_routing && configured.agents_enabled && configured.model_setup_seen
        );

        let mut expected = settings.clone();
        expected.model = configured.model.clone();
        expected.routing_json = configured.routing_json.clone();
        expected.model_routing = true;
        expected.agents_enabled = true;
        expected.agent_max_workers = 1;
        expected.num_ctx = 4096;
        expected.model_setup_seen = true;
        assert_eq!(
            serde_json::to_value(&configured).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );

        verified.insert(ModelRole::Embedding, strings(&["qwen3-embedding:0.6b"]));
        let with_embeddings = settings_from_verified(&settings, &verified, 8192).unwrap();
        assert_eq!(with_embeddings.embedding_model, "qwen3-embedding:0.6b");
        assert_eq!(
            with_embeddings.retrieval_enabled,
            settings.retrieval_enabled
        );
    }

    #[test]
    fn setup_rejects_an_unsupported_context_budget() {
        let settings = Settings::default();
        let verified = verified_aliases();
        assert!(settings_from_verified(&settings, &verified, 0).is_err());
        assert!(settings_from_verified(&settings, &verified, 32768).is_err());
    }
}
