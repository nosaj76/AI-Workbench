# Local model benchmarks

Workbench can calibrate installed local text models and use the results for automatic task routing. In Advanced → Routing, enable automatic model selection and fresh task benchmarks. “Benchmark selected model” measures the current model; “Measured task benchmarks” shows saved results. Explicit model profiles still override automatic routing.

From the installed executable:

```sh
ollama-workbench --benchmark-models
ollama-workbench --benchmark-model qwen3.5:4b-q4_K_M --benchmark-model granite4.2:latest
```

The first command tests all installed models eligible for this local text suite; the second selects particular models. No models are downloaded. Remote models, dedicated translators and candidates outside configured resource limits are skipped. Proposed tool calls are graded but never executed. Cancelled partial model batches are not saved; completed model records are retained.

Suite 1 has two cases per task: extraction/unknown facts, Python diagnosis/evaluation, arithmetic/logic, constrained writing, and native tool arguments. It uses context 4096, output limit 160, temperature 0, seed 42, thinking disabled, and a 60-second deadline per request. Latency includes request overhead and any model load; it is not a warm-throughput benchmark. These few synthetic checks do not establish overall model quality, long-context performance, thinking-mode reasoning, vision or translation quality.

Routing first applies capability, privacy, resource and retry exclusions. A benchmark candidate must pass both cases for the task, plus both native-tool cases when tools are required. The fastest fully passing eligible candidate wins. Evidence must match the endpoint, model digest and suite, and be less than 30 days old. Missing, stale or failed evidence falls back to configured routing. The routing explanation identifies measured selection or fallback.

Results and response evidence stay in `benchmarks.sqlite3` in the Workbench data directory (normally `~/.local/share/ollama-workbench`). Benchmarking is manually started, not automatically scheduled. Re-run after model upgrades or significant hardware/runtime changes.

## Local qualification, 2026-11-13

Five installed models were measured: Qwen 3.5 4B and 9B, Granite 4.2, Ministral 3 8B, and OpenCoder 8B. Actual inventory routing selected:

| Task | Model | Evidence |
| --- | --- | --- |
| General | qwen3.5:4b-q4_K_M | 2/2 task and 2/2 tool cases |
| Coding | qwen3.5:4b-q4_K_M | 2/2 task and 2/2 tool cases |
| Writing | granite4.2:latest | 2/2 task and 2/2 tool cases |
| Native tool orchestration | ministral-3:8b | 2/2 tool cases |
| Reasoning | qwen3.5:4b-q4_K_M | Configured fallback; no model passed both checks |

The General profile is selected for ordinary chats, with automatic routing enabled. Code review remains an explicit Granite specialist profile. Other profiles, models, conversations and optional-engine settings were preserved.

Evidence: `reports/local-build/live-model-benchmarks.log`, `benchmark-live-routing.log`, `benchmark-routing-install.log`, and `work/benchmark-and-cleanup-2026-11-13/results.json`. Reproduce the saved-configuration routing check with `cargo test --offline --test local_benchmark_routing -- --ignored --nocapture` while Ollama is available.
