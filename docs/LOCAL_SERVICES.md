# Terminals, scheduled work, and managed Ollama

Workbench 0.2 provides native Linux controls for interactive terminals, durable scheduled model jobs, and an application-managed Ollama runtime. Open **Workbench → Terminal** for shells, **Schedules** for background work, or **Workbench → Services** for scheduling and runtime management. These features are implemented in [terminal.rs](../src/services/terminal.rs), [scheduler.rs](../src/services/scheduler.rs), [ollama_manager.rs](../src/services/ollama_manager.rs), and [services_ui.rs](../src/desktop/services/mod.rs).

## Interactive terminals

An open terminal is a real shell attached to a pseudoterminal. Changing directories, exporting variables, activating a development environment, and starting an interactive program affect the same shell until that terminal closes. Switching Workbench pages or sending the next command does not replace the process.

The native display handles ANSI colors, styled cells, cursor position, alternate screens, scrollback, bracketed paste, terminal device/status queries, keyboard input, and window resizing. Resizing changes both the terminal screen and the kernel PTY dimensions, so programs can observe the new size. Multiple independent shells can be open at once. The implementation supports up to 12 sessions, 10,000 scrollback lines per session, and a bounded one-megabyte raw transcript per session.

Open a terminal explicitly and choose its working directory. Normal shell commands run with the user's account permissions. A shell can read or change anything that account can access; Workbench's workspace file-tool restrictions do not turn a shell into an operating-system sandbox. OSC clipboard requests from terminal programs are ignored.

Persistence here means the shell retains its state across commands and page changes while Workbench is running. On Linux, closing a terminal stops its shell and the foreground and background jobs in that terminal’s POSIX session. Workbench also cleans up remaining session jobs when the shell exits on its own. A program that deliberately creates a separate session has its own lifetime. The desktop shows a terminal as closing while its supervisor finishes cleanup. Quitting Workbench waits for terminal cleanup. Terminal sessions are not restored after application exit or reboot. Use an external terminal multiplexer when a shell needs to outlive the desktop application. The separately launched scheduler and managed Ollama server have their own lifecycles and can remain running after the window closes.

## Create a scheduled review

1. Open **Schedules** and enter a name, installed model, workspace, and prompt.
2. Select **Cron**, **Once**, or **Interval**. For a weekday morning review, use `0 9 * * MON-FRI` and the explicit timezone `America/New_York`.
3. Check the displayed next occurrence, choose a maximum run time, and save the job.
4. Start the background process, or install and enable the user service as described below.
5. Inspect **Run history** for results. **Open saved conversation** opens the actual model/tool exchange for that run.

For example:

> Review the selected workspace for concrete defects introduced by recent changes. Read the relevant source and tests. Cite the files and line ranges actually inspected, explain each finding, and identify any unreviewed areas. Do not change files or execute commands.

The scheduler uses the model explicitly saved with each job. It loads the current connection and generation settings when that job starts. Automatic routing, subagent delegation, automatic memory extraction, external network tools, and MCP are disabled in scheduled runs. This keeps the job's model and background action policy predictable. Built-in workspace read tools remain available.

### Timing syntax and timezone behavior

Cron uses five fields in this order:

```text
minute hour day-of-month month day-of-week
```

| Expression | Meaning in the saved timezone |
|---|---|
| `0 9 * * MON-FRI` | At 09:00 Monday through Friday |
| `*/15 * * * *` | Every quarter hour |
| `0 8,17 * * *` | At 08:00 and 17:00 |
| `0 9 1 * *` | At 09:00 on the first day of each month |
| `0 9 * JAN,MAR,SEP MON` | At 09:00 on Mondays in January, March, and September |
| `@reboot` | Once for a new operating-system boot observed by the daemon |

Lists, ascending ranges, steps, month/day names, and Sunday as either `0` or `7` are supported. Aliases include `@hourly`, `@daily`, `@midnight`, `@weekly`, `@monthly`, `@yearly`, `@annually`, and `@reboot`. Six-field/seconds syntax and Quartz extensions such as `L`, `W`, and `#` are not supported.

The parser follows traditional cron day semantics: if both day-of-month and day-of-week are restricted, either matching field selects the date. If either field starts with `*`, both fields must match. Consequently, `0 9 1 * MON` selects the first day of the month and Mondays, rather than only months whose first day is a Monday.

The timezone is stored with the job as an IANA name. The new-job form checks `TZ`, `/etc/timezone`, and the `/etc/localtime` zoneinfo symlink before falling back to `UTC`; always verify the displayed zone. Existing jobs keep their saved timezone if the computer's local timezone later changes. Calendar candidates use the bundled timezone database. A time skipped by the spring daylight-saving transition does not run; a wall time repeated in autumn runs at both corresponding UTC instants. For example, `30 1 * * *` in `America/New_York` has two occurrences on the autumn fallback date.

One-time jobs accept an RFC3339 time with an explicit UTC offset, such as `2026-11-09T09:00:00-05:00`. They disable themselves after claiming their run. Interval jobs repeat in elapsed minutes, independently of civil-time daylight-saving changes. The current limits are 500 saved jobs, a prompt of up to 64,000 bytes, and a per-job timeout between 30 seconds and 24 hours.

### Background lifecycle

**Start background process** launches the installed Workbench executable with `--daemon` and the same data directory. It has no desktop window and continues after the GUI closes. Its log is `scheduler.log` in that data directory. **Stop background process** requests a graceful stop and cancels an active job. A fallback process does not register itself for future logins or reboots.

For login startup, open **Startup service details**. Preview the unit, select **Install / update user service**, then **Enable at login & start**. The unit is written to the user systemd directory, normally `~/.config/systemd/user/ollama-workbench-scheduler.service`. Installation saves a previous unit copy and reloads the user's service manager. The unit runs the installed executable with the exact data-directory path, restarts on failure, and starts with the user's normal systemd session.

Install Workbench to its final location before registering this service: moving the executable later requires updating the unit. A user service starting before login depends on that account's existing systemd lingering configuration; Workbench does not enable lingering or request administrator access. Use **Stop service** or **Disable at login & stop** for a systemd-managed scheduler. When switching from the fallback process to systemd, stop the fallback first. The shared lock prevents two daemons from doing the same work.

The equivalent headless command is:

```sh
ollama-workbench --daemon --data-dir "$HOME/.local/share/ollama-workbench"
```

`SIGINT` and `SIGTERM` request cancellation and shutdown. The GUI does not automatically stop an explicitly started scheduler when it closes. Only enabled jobs run. Saving or editing a job does not itself install or start a service.

### Durable results and recovery

Jobs and run records live in `scheduler.sqlite3`. A SQLite immediate transaction claims a due occurrence, advances its schedule, and records a unique job/occurrence pair before inference starts. A heartbeat renews the worker lease, and a second consumer cannot claim the same active job. The daemon also holds a data-directory lock. Job changes from the GUI are visible to the running daemon without restarting it.

The daemon runs one model job at a time. When a machine sleeps or the scheduler is stopped, due recurring work is coalesced into one overdue run per job when scheduling resumes; the application does not replay every missed minute. Time already spent waiting behind another job also counts as a missed occurrence for this purpose. Future times are then calculated from the schedule. A one-time job that became due while the daemon was stopped is still due when it resumes.

Run states include `running`, `completed`, `needs_approval`, `failed`, `cancelled`, and `interrupted`. A daemon crash or lost lease leaves an `interrupted` record and does not silently repeat that occurrence. Saved conversation IDs, completed assistant messages, and denied-action previews are checkpointed during a run so they survive a later interruption. A stale worker cannot overwrite a run already recovered by another worker. The native history shows the 20 most recent runs; older records remain in the database. Removing a schedule preserves its history and conversations.

Legacy `schedules.json` jobs are imported once, in a single SQLite transaction, with their stable IDs, due times, intervals, enabled state, and previous-run timestamps. The original JSON is preserved. Version 0.2 no longer runs the former in-window scheduler. Migration errors are shown rather than overwriting the source file.

### Actions requiring a person

Background execution supports file inspection and model responses. File mutation, arbitrary commands, or other tool actions that require approval are denied in a headless run. Their concrete action previews are saved, and the run is marked `needs_approval` even if the model subsequently produces a normal answer. This prevents a denied edit from appearing as completed work.

Open that saved conversation and review the requested work interactively if it is still wanted. A historical preview is evidence of what was requested; it is not a pending authorization that the daemon will automatically replay. There is no per-job preauthorization for arbitrary shell commands or unattended file edits in this version.

## Managed Ollama installation and updating

Existing Ollama installations remain usable directly through the configured endpoint. Managed installation is optional and keeps its runtime under `managed-ollama` inside Workbench's data directory. It does not replace `/usr/bin/ollama`, alter an administrator's system service, install a GPU driver, or download a model automatically.

1. Open **Workbench → Services → Ollama installation & updates**.
2. Leave the version field blank for the latest stable official release, or enter an exact release tag such as `v0.12.0` when you intentionally want a specific published version.
3. Select **Check official release** and review the tag, release source, package size, checksum availability, and release notes.
4. Select **Download and install / update managed runtime** to perform that explicit installation.
5. Choose a free local port and optionally an existing model directory, then select **Start managed server**.
6. Select **Use this local runtime for chat** to set Workbench's endpoint to that managed server.

If an existing Ollama already occupies port `11434`, managed startup reports that the port is busy and leaves the existing service alone. Choose a free port such as `11435` to run the managed instance separately. A blank model-directory field uses the managed instance's own model store. Selecting an existing store reuses its files; the user must have access to that directory.

Release metadata comes from the [official Ollama GitHub releases API](https://api.github.com/repos/ollama/ollama/releases/latest). Linux packages follow [Ollama's Linux installation documentation](https://docs.ollama.com/linux): the manager selects the official `ollama-linux-amd64.tar.zst` or `ollama-linux-arm64.tar.zst` package, with `.tgz` support for older releases. The main amd64 archive includes CUDA runtime libraries in the upstream [Linux build script](https://github.com/ollama/ollama/blob/main/scripts/build_linux.sh) and [Dockerfile assembly](https://github.com/ollama/ollama/blob/main/Dockerfile). The machine still needs a compatible installed NVIDIA driver. Separate ROCm and JetPack additions are not automatically installed by this manager.

### Verification, activation, and rollback

The installer accepts official release-asset URLs and requires a SHA-256 digest from release metadata or an official checksum manifest. It streams the archive to a private download file, checks its expected size and digest, and extracts it into a new staging directory. It does not pipe an installation script into a shell. If a release provides neither digest nor a recognized checksum manifest, installation stops without activating an unverified executable.

Extraction rejects absolute paths, parent traversal, duplicates, special device files, and links that escape the package root. Package-internal symbolic and hard links are resolved only after their targets are validated. File count and compressed/expanded sizes are bounded. The resulting executable is checked for the host's Linux ELF architecture before activation. An interrupted or rejected extraction leaves the selected runtime unchanged.

Each installed version has its own directory and installation metadata. An atomic state file records the selected and previous versions. Installing an update selects it for the next start while an existing managed server continues running its current executable. Use **Restart with selected version** explicitly when interruption is acceptable. Stop the managed server before selecting **Roll back selected runtime**; rollback swaps the selected and previous versions, preserving both installed directories.

An operation lock prevents simultaneous app processes from racing installation, rollback, startup, or shutdown. Managed-server records include the executable identity and Linux process start time, so stale PID records do not authorize stopping a different process. The server binds to loopback, writes `server.log`, and becomes ready only after its version endpoint responds with the selected version. Read the server log for startup and GPU diagnostics. A managed server started from Workbench continues after the GUI closes; it is not automatically registered as a systemd service for reboot startup.

The data layout is:

| Path below the Workbench data directory | Purpose |
|---|---|
| `scheduler.sqlite3` | Jobs, unique run claims, leases, results, and migration marker |
| `scheduler-status.json`, `scheduler.log` | Scheduler identity/status and fallback daemon output |
| `managed-ollama/releases/<tag>/` | Extracted, verified runtime and installation metadata |
| `managed-ollama/active.json` | Selected and rollback version |
| `managed-ollama/process.json`, `server.log` | Managed server identity and output |
| `managed-ollama/models/` | Default managed model store |

## Validation scope

Automated tests exercise a real shell through a PTY, retained environment/working directory, actual kernel terminal resizing, ANSI screen changes, wide-character resize handling, and device/status query replies. Scheduler tests cover cron syntax, daylight-saving gaps/repetitions, competing SQLite consumers, lease recovery, persisted interruption evidence, legacy migration, reboot identity, and escaped service paths. A local mock Ollama service requests a file edit and verifies that the scheduled run preserves the file, saves the tool exchange and preview, and ends as `needs_approval`.

Managed-runtime fixture tests exercise safe extraction, internal/external links, cancellation, official-source/digest checks, version activation and rollback, concurrent-operation exclusion, invalid packages, and PID identity. They do not download a multigigabyte runtime, replace the host's Ollama, register a real service, or demonstrate GPU inference on the user's hardware.
