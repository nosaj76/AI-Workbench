# Architecture and boundaries

## Public compatibility, internal organization

`src/lib.rs` retains all 30 public modules from 0.3.0 through explicit path
attributes. Six shared modules have been added: endpoints, persistence, jobs,
theme, diagnostics and external agents. The project remains one native Rust crate;
this refactor does not add a web frontend, Node process or Python application.
Python is used only by the optional packaging regression test.

The 76 application Rust files are grouped into core, inference, orchestration,
workspace, services, studio and desktop domains. Large `impl` blocks were split
into child modules with deliberately limited visibility. Existing domain algorithms
and public entry points remain in place. `reports/source-map.json` maps the previous
flat paths and moved methods to their new homes.

## Request flow

The desktop owns view state, draft preferences, selected workspace and message
channels. Workers receive explicit request snapshots. The runtime validates model
policy and context, obtains the shared endpoint inference permit, requests the
actual Ollama stream, then releases that permit before waiting for tools,
delegated work or a user's action decision.

Workspace tools keep canonical path restrictions, expected source hashes,
one-use approval identity and backups. A proposed command or edit is not an
executed result. A model's textual JSON tool envelope triggers at most one native
protocol correction; assistant text is never parsed directly into authority.

External consultations are ordinary approved tools: configuration → prepared
prompt/evidence snapshot → approval → process result. A private run directory
separates output artifacts from source files but is not a process sandbox. Model
quality and third-party CLI behavior remain outside the approval mechanism's proof.

## Settings and persistence

The Settings page owns a cloned draft and a baseline snapshot. Save first parses
and validates the entire candidate, refuses conflicting background preference
changes, writes the full snapshot atomically, then replaces the live configuration.
An invalid field leaves live and saved preferences unchanged. Session bookkeeping
is preserved rather than overwritten by an old form. Leaving Settings discards
unsaved edits; explicit theme/skill controls update their corresponding draft and
baseline fields.

Shared JSON writes use a unique, exclusively created same-directory temporary file,
private Unix mode, flush/sync and rename. This avoids shared temporary-name races.
Failures before rename leave the destination unchanged; a directory-sync failure
after rename is a reported durability uncertainty, not an assertion of rollback.
An explicitly replaced corrupt settings file first receives a recovery copy.

Settings, Ollama transport, setup identity and inference permits share the endpoint
normalizer. It rejects credentials/query/fragment and uses the parsed host for
loopback policy; the word 'localhost' inside a remote domain is insufficient.

## UI lifecycle and asynchronous results

A full-width wrapping toolbar is laid out before side panels. The chat composer
reserves its own lower panel before the transcript. Models are selected consistently
with the currently active profile. Theme styling uses the pinned egui 0.31.1 API,
not a field from a different release.

Connection and file-preview tasks carry monotonically changing request identities.
Responses for a superseded endpoint or preview cannot replace the current UI.
Conversation changes clear unsent content and attachments from the previous
conversation. Connection errors clear obsolete online state.

Shared worker guards convert panics into error paths for the desktop work queue,
advanced activity queue, service queue and inference worker; their completion
notifications still run. This is not a claim that every possible process/thread
failure across all external engines can be recovered.

## Build and installation integrity

Builds use the checked-in lockfile (`--locked`), so dependency versions stay
pinned. `build.rs` calculates a deterministic diagnostic source
identity. Launch scripts use Cargo against the checkout instead of choosing an
old bundled binary. Installation requires the real compile/test/lint/release gate,
stages the new executable and preserves the previous installed executable.

## Retained limitations

This is not a hard execution sandbox, provider API SDK suite, virtual machine,
complete office-layout editor, video understanding engine or automatic engine/model
installer. Existing scope limits remain explicit. A discovered executable does not
prove its model files or credentials work. The new source remains uncompiled until
the real Rust gate runs successfully; structural refactoring is not end-to-end
qualification.
