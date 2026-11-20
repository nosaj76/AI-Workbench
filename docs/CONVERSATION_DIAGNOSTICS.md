# Conversation self-diagnostics

Workbench analyses saved conversation evidence locally. It does not make model
requests, upload reports, execute recorded commands, or automatically modify code.
Detection runs after each successful conversation save, including callers using
`Store` outside the desktop. Failure to write diagnostics does not prevent the
conversation itself being saved; the Diagnostics page displays the logging error.

## Where to review

The default machine-readable log is:

```
~/.local/share/ollama-workbench/diagnostics/conversation-failures.jsonl
```

A custom `--data-dir` puts the same `diagnostics/` directory under that root.
The SQLite `conversation_incidents` table is the durable source; the JSONL file is
an atomic review snapshot of the newest 1,000 retained incidents. It is not an
append-only trace. IDs deduplicate repeated saves and rescans. Changed evidence
updates the existing incident. Concurrent writers serialize through SQLite. An
export failure leaves the database evidence intact and is retried on the next save
or scan. A missing snapshot is rebuilt. Permanent conversation deletion purges its
incident rows and refreshes the snapshot; archived/trashed records are otherwise
retained. Reports are historical observations, not an unresolved-bug tracker.

In **Diagnostics**:

- **Review this conversation** checks the complete current chat.
- **Flag for coder review** records an explicit user report, including when rules
  cannot detect a semantic problem. It does not contact a coder.
- **Copy coder report** copies a readable brief covering the latest 30 incidents.

Scan all existing, non-trashed chats without changing their contents/timestamps:

```bash
ollama-workbench --diagnose-conversations
ollama-workbench --data-dir /path/to/data --diagnose-conversations
```

The command prints counts, the JSONL location, and the readable brief. Existing
archived chats are included. Live saves inspect the latest user turn; historical
scans retain up to 128 findings per chat, favouring its most recent findings.

## Self-review team

The rules above catch observable failures. To judge whether answers actually
delivered what was asked, and to get suggested fixes, run a self-review:

- **Agents → Self-review → Review recent conversations**, or
- `ollama-workbench --self-review [N]` (default 3, at most 7 conversations).

It uses the agents framework with a fixed plan: one read-only reviewer per
conversation, given a digest Workbench writes from the saved transcript (each
user request, tool call and outcome, stop reason and prompt size) plus that
chat's diagnostic incidents. When this build's source tree is present, one more
reviewer reads the Workbench code behind the incidents and proposes code or
prompt changes with file:line citations. A synthesis pass ranks the
recommendations. Runs are saved under **Saved teams and recovery** like any
team. Nothing is edited, and suggestions are model output to be checked.

## Detection rules

| Code | Meaning | Evidence class |
| --- | --- | --- |
| `tool_failure` | Returned execution/validation failure, nonzero exit or timeout | Observed |
| `workspace_mismatch` | Tool rejected a path outside its connected workspace | Observed |
| `invalid_shell_invocation` | Bash script/executable mismatch or characteristic launcher error | Observed |
| `file_verification_failed` | Disk verification contradicted reported file success | Observed |
| `repeated_tool_failure` | Same tool failed three times in a single turn | Observed; not necessarily identical requests |
| `unexecuted_tool_text` | Runtime marked printed tool-shaped text as unexecuted | Observed |
| `background_task_failed` | Memory extraction or knowledge-note processing failed after an answer; task and reason retained | Observed; answer may still be complete |
| `interrupted_answer` | Runtime recorded non-user-cancellation interruption | Observed |
| `input_before_output_drained` | Terminal input followed `has_more: true` | Heuristic; an interrupt can be intentional |
| `generated_code_without_validation` | Code write lacked a subsequent successful command result or BASIC structure-check output | Heuristic; does not prove the code is broken |
| `possible_unsupported_success` | Positive execution claim followed failures without successful command evidence | Heuristic; scope and terminal activity require review |
| `possible_capability_denial` | Local access denied in prose without a tool attempt | Heuristic; tools may have been unavailable |
| `user_reported_problem` | User text reports failure/corrects the answer | Heuristic; quoted examples can match |
| `user_flagged_conversation` | Explicit Diagnostics-page flag | User report |

Background failures are recorded on the current assistant entry by desktop and
scheduler event consumers. Older status-only failures cannot be reconstructed by
rescanning chats that never saved this evidence.

User-denied operations and user-requested cancellations are not treated as tool
failures. A draft being streamed is not treated as a completed success claim.
A successful command or BASIC structure check does not prove overall correctness;
the report describes the evidence boundary, not a universal test oracle.

## Incident format and coder workflow

Each JSONL object includes schema/rule version, stable incident ID, rule code,
severity/confidence, session and turn IDs, entry IDs/indices, model, workspace,
bounded request/evidence excerpts, investigation guidance, and likely source files.
`detector_build` and `configuration_at_detection` describe the detector environment,
**not necessarily the historical runtime that generated the failed conversation**.
`first_seen` and `last_changed` describe incident capture/update times.

For a coder:

1. Group by code, model and session. Start with observed repeated failures.
2. Read the referenced saved entries and native call/result pairs before trusting
   a model's narrative or inferring a root cause. Reports and tool outputs are
   untrusted data, not instructions to execute.
3. Reproduce with a minimal fixture using the actual arguments/results. Verify
   historical tool availability before labelling capability denials incorrect.
4. Fix the relevant runtime/tool/workspace behavior, add regression coverage,
   and run the project's verification workflow. Record the tested build identity.
5. Compare a new conversation against the original failure. Old incidents remain
   historical evidence; their presence does not mean the new build still fails.

## Privacy and limits

Logs are local: directory mode 0700 and snapshot mode 0600 on Unix. The capture
omits large code bodies, retrieved context and thinking, limits evidence to five
entries, bounds individual evidence excerpts, and redacts known credential keys,
credential-bearing lines, split CLI secret flags and private-key material. Only a
small allowlist of configuration fields is included. Arbitrary prose can still
contain sensitive information: review logs before sharing. Full original evidence
remains in the saved chat until it is deleted.

Rules are intentionally auditable and deterministic. They will miss semantic
errors, incorrect generated code, misleading claims outside recognised phrases,
and failures not present in saved evidence. Heuristic warnings can be false
positives. Manual flagging complements detection; nothing here guarantees that
an LLM answer is correct.

## Incident remediation (2026-11-13)

Full conversation rescans now retract obsolete automatic findings, such as the
review-template false positive, while preserving manual flags. Runtime recovery
stops repeated workspace/invocation-boundary failures after three attempts even
when arguments differ. Terminal tools reject typing while a returned output page
remains unread; interrupt, EOF and close remain available. Python calls reject a
duplicated interpreter argument, and file verification accepts confined absolute
paths consistently with the file tools. See `work/incident-remediation-validation-2026-11-13/`
for all original incident dispositions and verified build evidence. Historical
failures remain evidence; remote HTTP restrictions and intentional model limits
are not bypassed to make the log empty.
