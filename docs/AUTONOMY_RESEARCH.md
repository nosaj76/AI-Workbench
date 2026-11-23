# Autonomous teams, review queues and research

These are implemented worker services in the native Rust application. They call the configured Ollama API, perform actual workspace reads, and retain their work locally. Sample text and a model's unsupported claim are never used as substitutes for a successful tool result or source fetch.

## Autonomous teams

A team run has three stages:

1. The supervisor model converts the user's objective into a bounded JSON plan. With automatic routing enabled and the worker model list empty, the orchestrator role selects the supervisor and each child's actual task determines its specialist route. Explicit worker model lists retain manual selection. Invalid plans receive one correction attempt; an invalid or over-budget plan does not silently create extra workers.
2. Children execute with independent conversation histories. They can inspect the selected workspace through the existing file tools. Results and tool traces are retained separately, so one child's speculation does not become another child's hidden premise.
3. The supervisor synthesizes the actual completed child answers, preserving source citations and identifying failures and disagreements. If the answers do not fit one context, a bounded hierarchy of synthesis calls processes every input answer.

The normal concurrency is one child at a time, suitable for a single local GPU. The implementation permits up to four simultaneous workers and eight planned tasks. The 0.3 starter setup enables a shared inference permit for the selected endpoint, so even concurrent child workflows serialize their individual model requests while that mode is enabled. The permit is released before tools, delegation or approval waits. Disabling that mode and increasing worker concurrency can increase memory use or cause Ollama to swap models; it is a user-selected resource tradeoff, not an automatic speed guarantee.

Children are read-only by default. A model-requested write, command, network action or external adapter invocation is denied and the denial is returned to the model. When the caller explicitly enables actions, the actual pending action and its reply channel are forwarded to the native approval UI. This does not authorize the model to approve itself. Multiple workers can independently wait for approvals, and cancelling the team stops those waiting requests.

Worker settings disable further delegation and automatic memory updates. This is a supervised team with bounded children, not an unbounded recursive agent tree. The assigned model is retained throughout a child run, except for the dedicated translation adapter's validated coordinator fallback when automatic routing applies. Each worker has a thirty-minute wall-clock bound and the existing bounded tool loop. See [Specialist setup and routing](MODEL_SETUP_ROUTING.md) for the starter models, capability gates, 10B cap and translation behavior.

### Saved work and resume

`agent_runs/<run-id>.json` stores the objective, workspace, plan, per-child status, answer, tool trace, errors and final synthesis. A trace that exceeds its bound is explicitly marked truncated. Updates use temporary files and an atomic rename.

Resuming a run requires the same objective and workspace. Completed children are preserved; incomplete children are queued again. This avoids rerunning already finished work. A failed planning stage can be retried without pretending a plan existed. A run is marked `done`, `partial`, `cancelled` or `error` according to the actual recorded results.

An OS-held lease prevents two workers or application processes from resuming the same run concurrently. If the application exits before its final checkpoint, the next saved-run load detects that no worker owns the lease and records `interrupted`; completed children are retained and previously running children become queued. A live run remains marked running. Lease files stay in place to avoid races between different file handles; their existence alone does not mean a worker is active. This cross-process protection currently requires Unix, matching the shipped Linux build.

Public entry points are `agents::AgentRequest::new`, `agents::run`, `agents::spawn`, `agents::load_run` and `agents::list_runs`. `AgentRun::report()` produces the combined report.

## Comprehensive project-review queues

The review queue serves a different purpose from retrieval. Retrieval selects a few relevant excerpts. A comprehensive queue enumerates every file eligible under the workspace policy and creates work for every character successfully extracted from those files.

### Manifest and coverage

The manifest traversal is not limited by the file browser's displayed-tree cap. Files are split into bounded chunks, including long individual lines. Unicode is preserved, and each chunk records its exact source line range and starting column. Chunking never silently drops the remainder of a long line.

The queue records:

- Workspace and relative source path.
- SHA-256 of the source file and the extraction/chunk policy generation.
- Chunk ordinal, line range, starting column and complete chunk text.
- Queued, running, done, error, skipped or stale state.
- Model, attempts, result, error and update time.

Protected secret paths, ignored dependency/build folders, binaries, symlinks and files above the workspace's 16 MiB bound are outside the eligible policy. Native extraction has an 8 MiB character bound. A document that reaches the extraction limit is marked truncated; finishing its available chunks does not count as full-file coverage. Extraction errors are retained as errors, rather than making the file disappear from the coverage report.

The application, not the model, computes completed-chunk and completed-file counts. Full eligible-file coverage requires a completed traversal, every current chunk completed, no extraction failure, no truncated document, no stale file and no skipped work. A model reporting “everything looks complete” cannot change those counters.

### Running and resuming

Each chunk is sent to the selected model with its path, hash, exact location, review instruction and source text. This per-chunk stage has no action tools. The review is an inspection of supplied evidence; it does not claim that tests, builds or commands were run.

Completed results are committed to SQLite one item at a time. A stop returns the current unfinished item to the queue. A later process can recover an interrupted `running` item after acquiring the project lock. That lock prevents two application/service processes from concurrently claiming the same project review.

With the item limit set to zero, the runner continues through all currently queued work until complete or stopped. A nonzero item limit is an explicit batch size, useful for reviewing a large repository over several runs. Three consecutive inference errors stop the current run to avoid hammering an unavailable server; unattempted items stay queued.

Refreshing preserves completed work for unchanged sources when the review instruction, model and endpoint match. Changing the question or model starts a separate review of the same extracted text. Switching back resumes the matching review. Findings from older versions that did not record the review objective remain available in their exported run reports.

Changed files invalidate their old chunks and generate new queued work. Removed or newly ignored files are retired after a successful full traversal. Cancellation or a traversal error prevents deletion reconciliation, so a partial scan cannot erase the previous manifest.

Source hashes are checked before and after each model review and again before the final coverage record. A file changed during inference is marked stale, and the old verdict is not counted as a current completed review.

### Reports

The database is `review.sqlite3`. Each run writes a complete per-chunk Markdown record and a JSON run record under `review_runs/`. The complete record includes pending work, failed chunks, skipped chunks and extraction failures. The concise synthesis processes all completed current-item reports in bounded groups. The detailed report remains available even if the final model synthesis fails or is stopped.

Public entry points are `ReviewQueue::open`, `refresh`, `refresh_with_events`, `coverage`, `items`, `retry_failed` and `skip`; `ReviewRequest::new` and `run_review` execute the queue.

## Iterative deep research

Research runs combine a real model-driven search plan, actual local or web source acquisition, per-source reading notes, follow-up planning and a final cited synthesis. They use a fixed round and source budget so the user can bound model time and network traffic.

The default limits are three research rounds and twelve source snapshots. The configurable limits are one to eight rounds, one to one hundred sources, and bounded excerpts per source. A planner can stop early when the retained source notes answer the question. Repeated queries and duplicate page URLs are not treated as new evidence.

### Local research

Local mode builds or refreshes the workspace text index, searches it, validates source hashes, and records the exact excerpt, path and line range. A source ID identifies an actually read excerpt. It does not imply that the entire file, book or repository was reviewed. Extraction/indexing failures are recorded as limitations.

### Optional web research

Web research requires two explicit settings: the application's network tools must be enabled and the research request must allow web access. Search queries go to the configured SearxNG service. Explicit seed URLs can be used without a search service.

Search snippets are only discovery hints. A web page receives a source ID after an actual successful fetch of readable content. Unsupported binary responses and failed requests are logged and not counted as reviewed sources. HTML scripts, style data and hidden template bodies are excluded from the extracted text.

Pages have a 3 MiB download bound, a thirty-second request deadline and a three-redirect limit. Domain restrictions, when supplied, apply before following redirects as well as before accepting the resulting evidence. URLs with embedded credentials and non-HTTP schemes are rejected. A blocking page request can take up to its request deadline to return after cancellation; additional searches and model work stop at the next cancellation check.

### Notes, citations and retained evidence

Each retrieved source receives an `S1`, `S2`, ... ID, a content hash, acquisition time, excerpt text, truncation flag and model reading notes. Follow-up planning asks which evidence gaps remain. The final synthesis is based on the retained source notes, and unknown model-generated citation IDs are marked unverified rather than silently treated as evidence.

This validates source provenance and citation identity. It does not independently prove every model interpretation of those sources. The report explicitly records source counts, model-analysed counts, failed queries, unsupported citations and why the research stopped.

`research_runs/<run-id>.json` retains all snapshots, notes, query history and final output. A cancelled run can resume unfinished source analyses using the saved snapshots. Public entry points are `ResearchRequest::new`, `run_research`, `load_run` and `list_runs`; `ResearchRun::report()` produces the human-readable report.

Research uses the same OS-held per-run lease as teams. Concurrent resume of a live run is rejected. A saved run left running after its process exits is marked `interrupted` when loaded, with retrieved snapshots and completed source notes preserved for resume.

## Verification

Module tests exercise plan budgets, saved-run confinement, Unicode chunk coverage, incremental review state, cancellation recovery, cross-process review locks, live-run resume exclusion and orphaned agent/research checkpoint recovery, citation validation and domain matching.

`tests/autonomy_flow.rs` exercises the complete worker flows through real loopback HTTP fixtures and temporary files: independent child tool contexts, denied side effects, approval cancellation, parent-chat cancellation while a delegated child's HTTP response is pending, persisted review resume, mid-review source changes, local research, actual web source fetching, duplicate URLs, failed/binary pages, and redirect restrictions. These protocol fixtures verify implementation behavior; they do not measure real-model reasoning quality or GPU performance.
