# Building and installing

The quickest route is `./setup.sh`, which installs every dependency and then runs
`./install.sh` for you (see the [README](README.md#quick-install)). This page
describes the individual build commands.

Run these commands in the project directory as the normal desktop user, not with
`sudo`. The wrappers resolve their own checkout path, so they also work when
invoked by absolute path from another directory.

## Everyday build commands

```bash
./build.sh                    # Incremental optimized release build
./clean-build.sh              # Clean checkout artifacts, then build release
./clean.sh                    # Clean checkout artifacts only; no rebuild
```

Both build commands write `target/release/ollama-workbench`. They do **not**
install the executable, run inference, format source, or claim that verification
passed. A clean build is useful after toolchain or dependency changes; normal
source edits should use the incremental `build.sh` path.

All compilation uses the checked-in lockfile through `--locked`. None of these
scripts runs `cargo update`, deletes `Cargo.lock`, clears the Cargo registry,
removes model weights, or modifies Workbench application data.

Limit compiler parallelism on machines with 16 GB of memory or less:

```bash
./build.sh --jobs 2
./clean-build.sh --jobs 2
```

Other supported forms:

```bash
./build.sh --debug
./build.sh --offline
./clean-build.sh --debug --offline --jobs 2
./clean.sh --dry-run
```

`--offline` only forbids Cargo network access; it cannot supply missing cached
crates. The supplied Brave Search MCP preset is a **runtime integration**, not a
Rust build dependency. Ordinary Workbench builds do not require Node, npm, an API
key, SearXNG, Ollama model downloads, or a running MCP server.

## Check, test, verify, run, and install

| Command | Operation |
| --- | --- |
| `./check.sh` | `cargo check --all-targets --locked` |
| `./test.sh` | All Rust test targets, then documentation tests |
| `./verify.sh` | Check, tests, documentation tests, strict Clippy, release build |
| `./run.sh` | Incremental release build, then launch that exact executable |
| `./run.sh --demo` | Build, then launch the labelled no-inference preview |
| `./run.sh --version` | Build, then print version and source identity |
| `./run.sh --doctor` | Build, then print non-destructive readiness diagnostics |
| `./install.sh` | Run verification first; then install for the current user |
| `./install.sh --clean --jobs 2` | Clean artifacts before verification and installation |
| `./rebuild-and-install.sh` | Compatibility alias for `install.sh` |
| `./rebuild-and-run.sh` | Verify, then launch; remaining arguments go to Workbench |

`check.sh`, `test.sh`, `verify.sh`, and `install.sh` accept `--offline` and
`--jobs N`. `verify.sh` and `install.sh` also accept `--clean` and
`--format-check`. `run.sh` and `rebuild-and-run.sh` preserve all application
arguments, including paths containing spaces. Cargo's standard
`CARGO_BUILD_JOBS` and `CARGO_NET_OFFLINE` environment settings remain usable.

The installer never bypasses a failed compiler, test, documentation, lint, or
release-build gate. It installs under `~/.local/bin`, writes the current user's
desktop entry, and preserves an existing executable as
`~/.local/bin/ollama-workbench.previous`. It does not alter conversations,
settings, models, workspaces, or an existing Ollama installation. Close an old
Workbench window before reopening the installed build.

## Formatting is explicit

```bash
./format.sh                   # Apply rustfmt changes
./format.sh --check           # Check only; do not write
./verify.sh --format-check    # Add formatting to the verification gates
```

Build and install do not rewrite source automatically. `format.sh` requires the
rustfmt component that matches the selected Rust toolchain.

## Output safety and logs

The wrappers force final and intermediate artifacts into this checkout's
`target/`, overriding global Cargo output paths. They refuse symlinked
`target/`, `target/debug/`, or `target/release/` paths. `clean.sh` asks Cargo to
clean that explicit target directory; it never recursively removes a guessed
path, the project root, Workbench data, Ollama data, or Cargo's shared cache.

A shared `.workbench-build.lock` serializes the supplied build, clean, verify,
and install wrappers. The GUI launches only after the lock is released. Cargo
commands run manually outside these wrappers do not participate in that lock.

Every operation writes a timestamped log beneath `reports/local-build/` and
keeps the real failing exit status even though terminal output is also piped to
a log. Optional `make` aliases are available for build, clean-build, clean,
check, test, verify, format, format-check, install, and run.

## Requirements

The manifest requires Rust **1.89 or newer**, Cargo, a native C compiler/linker,
`pkg-config`, Bash, `flock` from util-linux, and the xkbcommon, OpenGL (Mesa) and
Wayland development libraries used by the desktop window. Verification and
installation also need Clippy; `format.sh` needs rustfmt.

The build scripts report missing tools but do not install them; `./setup.sh` does.
`./setup.sh --no-install` installs the dependencies without building.
