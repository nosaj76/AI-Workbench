# Reviewed external-agent consultations

AI Workbench can consult optional Hermes, Codex, Claude and Gemini command-line
agents. It does not require any provider login to start or use local Ollama.
All four presets start **disabled** and require a separate installed/authenticated
CLI. Preset flags are editable because installed CLI versions and account behavior
can differ.

## Interactive workflow

In Settings, open External agents, choose the executable/arguments, enable the
profile and explicitly consent to sending the reviewed task/evidence when the CLI
may use a provider. Save the full valid settings snapshot.

Under Activities → Tools, select the agent, enter a concrete task, optionally list
up to eight workspace-relative evidence files and add context. Choose Review. The
existing approval dialog presents the full task/evidence prompt and the literal
command arguments. Approve to execute that snapshot, or deny to launch nothing.

Enabled profiles are also exposed through `list_external_agents` and
`consult_agent` to a tool-enabled normal-chat orchestrator. These consultations are
not silently inserted into the Ollama model list or automatic specialist routing.
Autonomous children in default read-only mode cannot approve or launch them.

## What is sent

Only the explicitly assembled task, context and file evidence are passed as the
consultation prompt. The application does not implicitly append all conversation
history. Each file uses the existing workspace reader; paths, extraction kind and
truncation are visible. Source hashes are checked while preparing and before
executing the approved action. A modified source or revoked profile invalidates
the prepared launch.

Tasks are capped at 16 KiB; added context at 32 KiB; at most eight documents are
accepted; the complete encoded prompt must fit 64 KiB. Oversized requests fail
rather than silently shedding evidence. Input goes on stdin or into one explicit
whole `{prompt}` argument according to the profile. There is no implicit shell
expansion. Arguments remain literal values.

## Results, failures and cancellation

Each approved run receives an exclusive directory under the tool engine's backup
root at `external-agent-runs/<run-id>/`. `request.json` stores the snapshot;
`result.json` stores the bounded actual command result. A pre-result launch failure
is recorded in `failed.json` when possible. Unix run directories are private and
JSON files use the shared private atomic-write helper.

A successful consultation requires exit code zero, nonempty stdout and no timeout,
cancellation or output truncation. Nonzero/empty/truncated results are not labeled
completed reasoning. Processes retain the existing timeout, output limit and Unix
process-group cancellation behavior. Up to 256 KiB of bounded output is recorded;
return values expose up to 12,000 stdout characters and 2,000 stderr characters,
with an explicit excerpt-truncation flag and full-result path. This helps prevent
verbose CLI logs from immediately exhausting a small coordinator context.

## Security and cost boundary

The external process has the desktop user's ordinary OS permissions and inherited
environment. A private cwd, disabled built-in tools in one preset, a provider's
read-only mode, or a prompt saying 'do not edit' is **not** a Workbench-enforced OS
sandbox. The CLI may load its own user configuration, extensions or credentials.
Only enable a trusted installation. No network packet filter is installed by a
profile's consent flag; a falsely labeled executable can still use its permissions.

The Codex preset requests its documented read-only sandbox; the Claude preset
requests no built-in tools and disallows MCP tool names. Other presets provide
one-shot task instructions, not an independently enforced tool restriction.
Provider limits, cloud use and billing remain those of the installed CLI/account.
For stronger isolation, arrange and verify an OS sandbox outside this application.

## Primary interface references

- Codex non-interactive execution: https://developers.openai.com/codex/noninteractive/
- Claude CLI reference: https://code.claude.com/docs/en/cli-reference
- Gemini headless mode: https://geminicli.com/docs/cli/headless/
- Hermes CLI reference: https://hermes-agent.nousresearch.com/docs/reference/cli-commands

These references informed the adapter presets. The automated tests use local
process fixtures rather than the real vendor CLIs.
