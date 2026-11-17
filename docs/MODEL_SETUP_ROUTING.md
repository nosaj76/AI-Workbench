# Starter models, orchestration, and routing

Workbench provides a native setup flow for downloading a compact collection of Ollama models and assigning them to specialist roles. The application is written in Rust and does not require a browser or a hosted chat service. The default endpoint is a local Ollama server. An existing remote Ollama endpoint can be used only when the user has explicitly enabled remote access in Settings; setup preserves that preference.

The starter pack is intended for machines such as an Ubuntu desktop with 16 GB RAM and an NVIDIA GPU with 8 GB VRAM. It uses three models for seven chat roles, with optional larger analysis and embedding models. This is a practical initial selection, not a claim that one model is best for every language, programming task, or benchmark. Review answers and adjust role preferences for your own work.

## The initial downloads

| Selection | Exact download tag | Assigned roles | Estimated download | Conservative catalog parameter total | License |
|---|---|---|---:|---:|---|
| Default | [`qwen3.5:4b-q4_K_M`](https://ollama.com/library/qwen3.5:4b-q4_K_M) | Orchestrator, general, analysis, writing, vision | 3.3 GB | 4.6B | Apache 2.0 |
| Default | [`qwen2.5-coder:7b-instruct-q4_K_M`](https://ollama.com/library/qwen2.5-coder:7b-instruct-q4_K_M) | Coding | 4.7 GB | 7.7B | Apache 2.0 |
| Default | [`translategemma:4b-it-q4_K_M`](https://ollama.com/library/translategemma:4b-it-q4_K_M) | Translation through its dedicated adapter | 3.3 GB | 4.4B | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |
| Optional | [`qwen3.5:9b-q4_K_M`](https://ollama.com/library/qwen3.5:9b-q4_K_M) | First analysis preference, with the 4B model retained as an alternative | 6.6 GB | 9.5B | Apache 2.0 |
| Optional | [`qwen3-embedding:0.6b-q8_0`](https://ollama.com/library/qwen3-embedding:0.6b-q8_0) | Semantic retrieval embeddings | 639 MB | 0.6B | [Apache 2.0](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |

The three default downloads total approximately **11.3 GB**. Adding only embeddings makes that approximately **11.94 GB**; adding both optional models makes it approximately **18.54 GB**. These are rounded decimal download estimates, not exact free-space requirements or VRAM usage. Existing layers can reduce the bytes that Ollama downloads. Updates, temporary data, context state, and other applications can increase the storage or memory needed in practice.

The catalog deliberately rounds parameter totals upward. The official Qwen3.5 4B entry lists a 4.21B language model and a separate 334M vision projector, approximately 4.544B combined. The 9B entry lists 8.95B plus a 456M projector, approximately 9.406B combined. The code model is listed as 7.62B, TranslateGemma as 4.3B, and the embedding model as 596M. The larger catalog values in the table are conservative estimates for setup, not new measurements or changes to the models' names. All selected models stay below the user's **10B total parameter limit**.

The chat models use Q4_K_M quantization. Qwen3.5's separate vision projector uses BF16. The embedding model uses Q8_0. The application verifies installed metadata; a similarly named model with a different quantization is not automatically treated as the requested download.

## Run setup

On first launch, including an upgrade from 0.2, the native **Setup** tab is shown. Existing conversations and settings remain available. **Skip setup for now** keeps the existing manual configuration; the Setup tab remains available later. Applying a verified pack changes the selected default model, role preferences, context preset and agent/inference limits described below. Edits to those preferences during an active setup are preserved and require a fresh setup pass before applying the pack.

1. Start a local Ollama server and select its loopback URL in Settings. An existing installation can be used. The optional managed installation controls are described in [LOCAL_SERVICES.md](LOCAL_SERVICES.md). If you intentionally use a remote Ollama server, enable remote access explicitly first; downloads and inference then occur on that server.
2. Open starter model setup and review the default and optional selections, sizes, roles, and licenses. Choose a 4,096-token or 8,192-token context preset. The smaller preset leaves more memory headroom.
3. Review the linked [Gemma Terms of Use](https://ai.google.dev/gemma/terms) for the translation model. The other catalog models use Apache 2.0. Model licenses remain separate from Workbench's application license.
4. Select **Download & configure specialists**. Workbench checks the selected server, looks for suitable installed names, and asks Ollama to pull missing models with visible progress. Opening the application or viewing the catalog does not itself download weights.
5. After every selected model is verified, that same action applies the role configuration automatically. Failed or interrupted setup does not apply a partial collection of routes. If relevant settings changed while setup ran, review the new configuration and run setup again instead of applying a stale result.

The default pack needs **Ollama 0.17.5 or later** because that release introduced the small Qwen3.5 variants. This is a historical minimum; use a currently supported official release and respect any newer model requirement returned by Ollama. A model's tag can be updated independently of Workbench. The application surfaces version and pull failures instead of claiming that an old runtime can run every future revision. Source: [Ollama v0.17.5 release notes](https://github.com/ollama/ollama/releases/tag/v0.17.5).

Ollama downloads the weights into its own model store. They are not bundled in the Workbench application archive. The initial pulls need network access; installed local models can subsequently run without downloading their weights again. Setup does not enable network tools, change the user's remote-access preferences, or install GPU drivers.

### Existing models and interrupted downloads

The following aliases were checked against the official Linux llama.cpp manifests on **2026-11-07**. They matched the corresponding explicit quantized tags at the time of that review.

| Installed alias considered for reuse | Explicit download tag | Observed manifest prefix |
|---|---|---|
| [`qwen3.5:4b`](https://ollama.com/library/qwen3.5:4b) | `qwen3.5:4b-q4_K_M` | `d8b0f5e9760c` |
| [`qwen2.5-coder:7b`](https://ollama.com/library/qwen2.5-coder:7b) | `qwen2.5-coder:7b-instruct-q4_K_M` | `dae161e27b0e` |
| [`translategemma:4b`](https://ollama.com/library/translategemma:4b) | `translategemma:4b-it-q4_K_M` | `c49d986b0764` |
| [`qwen3.5:9b`](https://ollama.com/library/qwen3.5:9b) | `qwen3.5:9b-q4_K_M` | `56671c2ab938` |
| [`qwen3-embedding:0.6b`](https://ollama.com/library/qwen3-embedding:0.6b) | `qwen3-embedding:0.6b-q8_0` | `ac6da0dfba84` |

These observations are not permanent digest pins. Aliases and tags are mutable, and the general Qwen3.5 alias pages can also expose a separate MLX manifest for other platforms. Before reuse, setup checks the installed model's capabilities, parameter metadata, quantization, and whether its weights are present on the selected server rather than exposed through a cloud-model alias. Successful setup records and uses the actual installed name, including a verified alias, instead of writing a tag that is absent from the server.

Setup progress is checkpointed in Workbench's data directory. Resume or retry reconciles the saved run with the selected Ollama server. Model weights and partial download layers remain under Ollama's control; Workbench does not pretend that its checkpoint itself contains the weights. A cancelled pull or interrupted connection can require another pull request, while Ollama can reuse layers that are already available. A selected optional model must also finish verification before that setup run can be applied.

## What the orchestrator and router do

The **orchestrator** coordinates the work: it interprets a task, can delegate bounded subagent work, and combines the resulting evidence. It uses the verified orchestrator preference, normally the 4B general model. The **router** chooses an installed model for the kind of response being requested. Seven chat roles are configured: general, coding, reasoning, translation, writing, vision, and orchestration.

Routing first applies capability and resource requirements, then considers the configured role order, whether a model is already loaded, and the existing speed preference. Tool-based work needs confirmed tool support; image input needs confirmed vision support. An embedding model is not eligible for a chat response. A selected role does not give a model abilities it does not possess, and a failed capability check produces an explanation or a compatible fallback.

The optional 9B model is placed first in the analysis preference list only after verification. The shared 4B model remains in that list. This is an explicit model preference, not an automatic benchmark or a guarantee that the larger model will answer more accurately. The 9B download leaves less room for context on an 8 GB GPU and may need CPU offloading depending on the available memory and runtime.

Setup turns on routing and subagents, selects one subagent worker, and enables the mode that serializes this application's inference requests to the same endpoint. It sets a 10B total parameter ceiling and an 8 GiB model-file budget. **The file-size budget is not a VRAM reservation.** The application releases its requested model after inference so a different specialist can be loaded without deliberately keeping the whole pack resident. Another Ollama client or a separate background process can still use the same server; the desktop process's inference lock is not a machine-wide GPU scheduler.

The context preset is limited to 4,096 or 8,192 tokens. Advertised architectural context sizes are not safe defaults for every GPU. Other user preferences, saved profiles, instructions, memory, and retrieval settings are preserved. The selected profile may still have its own explicit model preference. Choosing embeddings changes the embedding model only after verification; it does not index files or turn retrieval on automatically.

### The 10B rule includes the whole model

Total parameters are checked from installed metadata, including a separate vision projector when present. With an active cap, the application requires an exact `general.parameter_count`; a rounded size label is insufficient. A mixture-of-experts model with only a few billion active parameters still exceeds this rule if its total weights exceed 10B. Labels such as “A3B” do not make a 30B model eligible. Model names and quantized file sizes alone do not prove the total parameter count.

Ollama's model detail response exposes `model_info`, `projector_info`, capabilities, and, in supported versions, a minimum runtime requirement. See the official [Ollama API types](https://github.com/ollama/ollama/blob/main/api/types.go) and [CLI model-detail handling](https://github.com/ollama/ollama/blob/main/cmd/cmd.go). Catalog estimates are used conservatively; unknown or incompatible metadata must not silently establish that a model satisfies the cap.

## Translation uses a dedicated adapter

TranslateGemma is a specialized translation model with a particular prompt contract. It is not used as an ordinary tool-calling chat assistant. The [official Ollama prompt guide](https://ollama.com/library/translategemma:4b) specifies one user message containing the source language and code, target language and code, translation instructions, two empty lines, and then the text to translate.

Workbench validates the language codes, derives their names from its supported mapping, and requires the complete source passage to match the current input or a single supplied attachment exactly. Recognized boundaries include a colon, a newline, quoted text or a complete local-file attachment. For example, `Translate from English to French: Hello, friend!` supplies both a direction and an unambiguous source. The target language is checked in the instruction outside the source passage; a language mentioned inside the passage cannot change the requested direction. The adapter sends only that translation request. Normal conversation history, system instructions, memory retrieval, and tool definitions are not inserted into the specialist prompt. It does not ask the specialist to return JSON or produce a thinking trace.

Google's [TranslateGemma model card](https://huggingface.co/google/translategemma-4b-it) specifies **2K tokens of total input**, even though the Ollama catalog displays the larger context capacity of its underlying architecture. Workbench therefore bounds the complete constructed prompt to **1,800 UTF-8 bytes**, including the translation instructions. This is a deliberately conservative limit without requiring a separate tokenizer; it is not a claim that 1,800 bytes equals 2,000 tokens.

If the language direction is uncertain, a code is unsupported, the exact source text is unavailable, the prompt is too large, or a requested local file has not been read, the coordinator handles the request through the general route. It can clarify the language direction or inspect the requested file using the normal tools. Workbench does not silently truncate a long document to squeeze it into TranslateGemma. Google's model is designed for 55 languages; the extensive locale-code table is not evidence of equal quality across every region or language pair.

Qwen3.5 can also act as the multilingual general fallback. For routing and structured coordination, the application disables thinking through the API when supported. The owner's [Qwen3.5 model card](https://huggingface.co/Qwen/Qwen3.5-4B) states that `/think` and `/nothink` text switches are unsupported, so they are not a substitute for the API option. Analysis tasks can use different thinking settings and need enough output budget to produce a final answer.

## Optional embeddings and memory

The 0.6B embedding model converts text into vectors for retrieval. It does not answer prompts or supervise subagents. Its owner documents multilingual and code retrieval, up to 1,024 embedding dimensions, and query-side task instructions; see [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B). The 4B and 8B family members have different output dimensions, so the larger-family example in a shared catalog readme should not be used as the 0.6B model's specification.

Automatic memory and a file index are separate application features. Selecting an embedding model does not mean existing documents have been embedded, old vectors are compatible, or every future answer searches all local files. Keep the retrieval model consistent with the index and use the actual indexing and retrieval controls for the workspace.

Indexing, knowledge search, and chat retrieval use the same endpoint-access, parameter-cap, and inference-permit policy as chat. Under the starter configuration, the embedding model must report an exact eligible parameter count and embedding capability. Its request also asks Ollama to release the model afterward, so the optional embedder is not intentionally retained alongside the next chat specialist.

## Validation boundaries

Configuration is built from a completed setup run's verified role map. All seven core roles must be present and the normalized server endpoint must still match before any new settings are returned. Tests exercise rejection of missing roles, unverified receipts, changed endpoints, and unsupported context budgets, use of actual installed aliases, optional analysis ordering, and preservation of unrelated settings. Backend and routing tests cover their own download, metadata, checkpoint, and inference behavior.

Mock-service and unit tests do not establish real model quality, GPU speed, or memory fit on the user's computer. No model weights are downloaded as part of those tests. Check a few representative coding, translation, analysis, and local-file tasks after setup, then adjust model preferences using observed results.
