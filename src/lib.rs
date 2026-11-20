//! Native desktop, inference, workspace and service modules.
//! Public module names are preserved so existing integrations keep their imports.

#[path = "desktop/advanced/mod.rs"]
pub mod advanced_ui;

#[path = "orchestration/agents.rs"]
pub mod agents;

#[path = "desktop/app/mod.rs"]
pub mod app;

#[path = "services/automation.rs"]
pub mod automation;

#[path = "workspace/calc.rs"]
pub mod calc;

#[path = "studio/document_activities.rs"]
pub mod document_activities;

#[path = "workspace/confined.rs"]
pub mod confined;

#[path = "workspace/diff.rs"]
pub mod diff;
#[path = "workspace/files.rs"]
pub mod files;

#[path = "studio/image_generation.rs"]
pub mod image_generation;

#[path = "services/integrations.rs"]
pub mod integrations;
/// Public, keyless knowledge sources (Wikipedia, arXiv, ...) as a search provider.
#[path = "services/knowledge_search.rs"]
pub mod knowledge_search;

#[path = "services/mcp.rs"]
pub mod mcp;

#[path = "studio/media.rs"]
pub mod media;

#[path = "orchestration/memory.rs"]
pub mod memory;

#[path = "inference/model_catalog.rs"]
pub mod model_catalog;

#[path = "inference/model_setup.rs"]
pub mod model_setup;

#[path = "inference/ollama.rs"]
pub mod ollama;

#[path = "services/ollama_manager.rs"]
pub mod ollama_manager;

/// Domain knowledge notes built from conversations, web results and research.
#[path = "orchestration/notes.rs"]
pub mod notes;
#[path = "orchestration/qa.rs"]
pub mod qa;
#[path = "orchestration/qa_library.rs"]
pub mod qa_library;
#[path = "orchestration/research.rs"]
pub mod research;

/// Ranked outline of a project's code, sized to a budget.
#[path = "workspace/repomap.rs"]
pub mod repomap;
#[path = "workspace/retrieval.rs"]
pub mod retrieval;

#[path = "workspace/review_queue.rs"]
pub mod review_queue;

/// Workbench reviews its own recent conversations (and optionally its code).
#[path = "orchestration/self_review.rs"]
pub mod self_review;

#[path = "inference/routing.rs"]
pub mod routing;

#[path = "inference/runtime/mod.rs"]
pub mod runtime;

#[path = "services/scheduler.rs"]
pub mod scheduler;

#[path = "desktop/services/mod.rs"]
pub mod services_ui;

#[path = "core/settings.rs"]
pub mod settings;

#[path = "desktop/setup_ui.rs"]
pub mod setup_ui;

#[path = "core/storage.rs"]
pub mod storage;

#[path = "desktop/studio/mod.rs"]
pub mod studio_ui;

#[path = "services/terminal.rs"]
pub mod terminal;

/// Bubblewrap isolation for model-proposed commands and terminals.
#[path = "services/sandbox.rs"]
pub mod sandbox;

#[path = "workspace/tools/mod.rs"]
pub mod tools;

#[path = "inference/translation.rs"]
pub mod translation;

#[path = "core/endpoint.rs"]
pub mod endpoint;

#[path = "core/persistence.rs"]
pub mod persistence;

#[path = "core/jobs.rs"]
pub mod jobs;

#[path = "desktop/theme.rs"]
pub mod theme;

#[path = "desktop/action_preview.rs"]
pub mod action_preview;

#[path = "core/diagnostics.rs"]
pub mod diagnostics;

#[path = "orchestration/external_agents.rs"]
pub mod external_agents;

#[path = "core/conversation_diagnostics.rs"]
pub mod conversation_diagnostics;

/// Panic reports and unclean-shutdown detection in the data folder.
#[path = "core/crash_reports.rs"]
pub mod crash_reports;

#[path = "inference/benchmarks.rs"]
pub mod benchmarks;
