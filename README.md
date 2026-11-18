# AI Workbench

A native Linux desktop app for working with local AI models through
[Ollama](https://ollama.com): chat, review your files and projects, search the
web, run approved commands and terminals, schedule jobs, and generate documents
and media — with nothing leaving your computer unless you allow it.

It is a Rust/egui desktop program, not a website, and needs no browser.
Version 0.4.0-rc.8 (release candidate). MIT licensed.

> The executable and data folder are still called `ollama-workbench`, so
> existing installs keep their conversations and settings.

## Quick install

On Debian/Ubuntu/Mint, Fedora, Arch/Manjaro or openSUSE:

```bash
git clone https://github.com/nosaj76/AI-Workbench.git
cd AI-Workbench
./setup.sh
```

`setup.sh` installs everything the app needs, asking before optional parts:

1. build tools and desktop libraries from your distribution (uses `sudo`);
2. optional helper programs — PDF tools, OCR, ffmpeg, speech, text browsers
   for key-free web search, LaTeX;
3. Rust 1.89 or newer through [rustup](https://rustup.rs), with clippy and rustfmt;
4. Ollama, with its official installer, if it is not already installed;
5. a release build that must pass the full test suite before it is installed to
   `~/.local/bin/ollama-workbench` with an applications-menu entry.

Useful options:

```bash
./setup.sh --yes                    # unattended: accept every default
./setup.sh --model qwen3:8b         # also download a model
./setup.sh --no-ollama              # Ollama runs on another machine
./setup.sh --no-optional --jobs 2   # minimal install, limit compiler memory use
./setup.sh --dry-run                # show every command without running it
```

Run it as your normal user, not with `sudo`. It is safe to run again to update
dependencies. Compilation takes several minutes the first time.

## Manual install

If you prefer to install dependencies yourself, you need:

- Rust 1.89+ with Cargo and Clippy, a C compiler and `pkg-config`
- `flock` (util-linux), Bash, curl
- development headers for xkbcommon, OpenGL (Mesa) and Wayland
  (Ubuntu: `build-essential pkg-config libxkbcommon-dev libgl1-mesa-dev libwayland-dev`)
- Ollama, locally or on another machine

Then:

```bash
./install.sh          # check, test, lint, build release, install for this user
./run.sh              # or: build and launch from the checkout without installing
```

See [BUILDING.md](BUILDING.md) for every build command and option.

## First run

1. Start **AI Workbench** from your applications menu, or run `ollama-workbench`.
2. **Setup** opens on first launch. It checks the connection to Ollama and can
   download a recommended set of models for chat, coding, writing and embeddings.
3. Open **Chat**, pick a model and start typing. Use **Files → Open folder…**
   to connect a project the assistant can read.

To look around without a model, run `ollama-workbench --demo`; it uses
throwaway storage and sends no requests.

## Configuration

Everything is in **Settings** (toolbar). Edits apply only when you press **Save**;
invalid values are rejected without changing anything.

| Section | What to set |
| --- | --- |
| **Connection and privacy** | Ollama URL (default `http://127.0.0.1:11434`). Remote or cloud model endpoints are blocked unless you allow them. Turn on web fetch and search tools here, and optionally a SearXNG URL, a text browser for search, and the local sites address. |
| **Tools and command approvals** | Enable workspace tools (files, commands, terminals). Choose what may run without asking: web actions, file changes made by file tools (backed up and restorable), sandboxed commands, and — separately — unsandboxed actions (commands without the sandbox, adapters, external agents, MCP calls). Fetching from this computer or the local network always asks. Set the command sandbox (auto, required or off), its network access and the folders it may read. Everything else shows a preview to approve. |
| **Appearance** | Light/Dark theme and interface scale (`Ctrl+Shift+T` toggles the theme). |
| **Generation and context** | Temperature, context size, output limit, thinking mode, tool rounds. Larger contexts need more memory. |
| **Instructions, memory and agents** | System instruction, your own standing preferences, automatic memory, and supervised subagents. |
| **Knowledge and routing** | Embedding model for file search, retrieval in chat, and automatic choice of specialist models. |
| **Profiles and prompt library** | Named model/instruction profiles and reusable prompts. |
| **Speech and transcription** | Programs used to read replies aloud and transcribe audio (e.g. `espeak-ng`, `piper`, `whisper-cli`). |
| **External agents** | Optional consultation of installed Hermes, Codex, Claude or Gemini CLIs. All start disabled. |
| **MCP servers and web search** | Brave Search (needs an API key and Node's `npx`) or any MCP server. Starting a server and each tool call need approval; once a server is running, the assistant can read the resources and prompt templates it advertises. |

Settings are stored in `settings.json` in the data directory.

### Web search

Turn on **Enable explicit web fetch and search tools**, then use any one of:

- **No setup:** install `lynx` or `w3m` (`setup.sh` offers them). Searches go to
  DuckDuckGo through the text browser, with Wikipedia, arXiv, Crossref, Hacker
  News and GitHub as additional sources.
- **SearXNG:** enter your instance's URL.
- **Brave Search:** enable the preset under **MCP servers and web search** and
  enter an API key.

Every search is shown for approval unless web actions are set to approve
automatically. Details: [docs/WEB_SEARCH.md](docs/WEB_SEARCH.md).

### Terminals, commands and the sandbox

With workspace tools on, the assistant can run one-off commands or open a
persistent Bash terminal and type into it. Each command, keystroke batch or
terminal action needs your approval unless you enable automatic approval for
commands.

Commands and terminals the assistant opens run in a
[bubblewrap](https://github.com/containers/bubblewrap) sandbox
(**Settings → Tools and command approvals → Command sandbox**):

- they can change files only in the project folder;
- they see an empty home folder, except read-only tool folders you list
  (by default `~/.cargo/bin`, `~/.rustup`, `~/.local/bin`);
- they have no network unless you allow it, and a cleared environment, so API
  keys in your environment are not passed in. With
  [passt](https://passt.top) installed (`setup.sh` installs it), allowed
  network means the internet only: the sandbox gets its own network stack, and
  this computer's services (such as Ollama on `127.0.0.1:11434`), Docker
  bridges and the LAN are blocked. Without passt, allowed network also reaches
  those local services;
- the project's `.git` folder is read-only (so hooks and git config cannot be
  planted to run outside the sandbox later), and secret files in the project
  (`.env`, `*.pem`, `*.key`, `secrets/`, `credentials.json` and the other names
  file tools refuse) appear empty. Installed packages such as Python
  `site-packages` are left alone.

**Auto** (the default) sandboxes whenever bubblewrap is installed and runs
commands unsandboxed otherwise, saying so in the approval preview. **Required**
refuses to run commands without the sandbox. **Off** runs them with your normal
permissions. Terminals you open yourself, and adapters, MCP servers and external
agents you configure, are not sandboxed. `setup.sh` installs bubblewrap.

After each approved file change, **Check after file changes** (default
`auto`) runs `cargo check`, `go build`, or a syntax check of the changed Python
or JavaScript files in the sandbox, without network, and shows the model the
errors. Set it to `off`, or to your own command as a JSON array such as
`["make","check"]`. It never runs outside the sandbox.

Terminals you open yourself are private: the assistant cannot list, read or
type into them unless you tick **AI may use** for that terminal in the
Terminals panel. It can always use the terminals it opened.

The assistant can also propose a Python 3 snippet (`run_python`) for
calculations or checking project files; the approval preview shows the code.
It runs only in the sandbox: if bubblewrap is missing it is refused (even in
Auto), unless the sandbox is set to Off.

### Data, backups and crash reports

Data lives in `~/.local/share/ollama-workbench` (or `$XDG_DATA_HOME`). Override it
with `--data-dir PATH` or `OLLAMA_WORKBENCH_DATA_DIR`. Back it up with the app
closed. It holds conversations (SQLite), settings, saved file versions, knowledge
notes, the Q&A library and `crash-reports/`.

If the app crashes, the next start says so and points to the report. Nothing is
sent anywhere.

## Command-line options

```text
ollama-workbench                        open the desktop app
  --demo                                preview with sample content, no model requests
  --data-dir PATH                       use another data directory
  --check                               test the Ollama connection and list models
  --doctor                              report configuration and which optional programs are installed
  --daemon                              run scheduled jobs without a window
  --diagnose-conversations              scan saved chats for failures, write a local report
  --benchmark-models                    measure installed models for automatic routing
  --benchmark-model NAME                measure one model (repeatable)
  --version                             print the version
```

## Troubleshooting

- **Run `ollama-workbench --doctor` first.** It lists what is configured,
  missing or invalid, without changing anything.
- **"Cannot connect to Ollama":** check `ollama list` works, or set the right URL
  in Settings → Connection and privacy. `ollama-workbench --check` tests it.
- **Window does not open:** run from a graphical X11 or Wayland session; over SSH
  there is no display.
- **Build runs out of memory:** `./install.sh --jobs 2`.
- **A feature says a program is missing:** install it (`./setup.sh --optional`)
  or set its path in Settings.

## Guides

- [Building and installing](BUILDING.md)
- [Web search](docs/WEB_SEARCH.md)
- [Models, starter packs and routing](docs/MODEL_SETUP_ROUTING.md)
- [Model benchmarks](docs/MODEL_BENCHMARKS.md)
- [Terminals, scheduled jobs and managed Ollama](docs/LOCAL_SERVICES.md)
- [Studio: images, documents and media](docs/STUDIO.md)
- [Agent teams, review queues and research](docs/AUTONOMY_RESEARCH.md)
- [External agents](docs/EXTERNAL_AGENTS.md)
- [Conversation diagnostics](docs/CONVERSATION_DIAGNOSTICS.md)
- [Source architecture](docs/ARCHITECTURE.md)

## License

MIT — see [LICENSE](LICENSE). Model weights, Ollama and optional programs are
separate projects under their own licenses and are not included.
