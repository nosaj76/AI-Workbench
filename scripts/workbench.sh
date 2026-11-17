#!/usr/bin/env bash
# Shared implementation for the root build scripts. No eval; arguments stay arrays.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
mode=${1-}
[[ $# -gt 0 ]] && shift
case "$mode" in
    build|clean|clean-build|check|test|verify|format|install|run|rebuild-run) ;;
    *) printf 'Use a root script such as ./build.sh or ./clean-build.sh.\n' >&2; exit 2 ;;
esac

usage() {
    case "$mode" in
        build|clean-build)
            printf 'Usage: ./%s.sh [--release|--debug] [--offline] [--jobs N]\n' "$mode"
            printf 'Default: optimized release build. clean-build first removes local target artifacts.\n' ;;
        clean)
            printf 'Usage: ./clean.sh [--dry-run]\nRemoves only this checkout\x27s target artifacts. Keeps source, Cargo.lock, models and application data.\n' ;;
        check|test)
            printf 'Usage: ./%s.sh [--offline] [--jobs N]\n' "$mode" ;;
        verify|install)
            printf 'Usage: ./%s.sh [--clean] [--offline] [--jobs N] [--format-check]\n' "$mode"
            printf 'Checks, tests, documentation tests, strict Clippy and release build.\n'
            printf 'Formatting is only checked when --format-check is explicitly selected.\n' ;;
        format)
            printf 'Usage: ./format.sh [--check]\nDefault: apply rustfmt. --check: report formatting differences without writing.\n' ;;
    esac
}

profile=release
clean_first=false
dry_run=false
format_check=false
offline=false
jobs=()
app_args=()
if [[ "$mode" == run || "$mode" == rebuild-run ]]; then
    # Every argument belongs to Workbench, including --help and --data-dir.
    app_args=("$@")
else
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help) usage; exit 0 ;;
            --release|--debug)
                [[ "$mode" == build || "$mode" == clean-build ]] || { usage >&2; exit 2; }
                profile=${1#--}; shift ;;
            --clean)
                [[ "$mode" == verify || "$mode" == install ]] || { usage >&2; exit 2; }
                clean_first=true; shift ;;
            --dry-run)
                [[ "$mode" == clean ]] || { usage >&2; exit 2; }
                dry_run=true; shift ;;
            --check)
                [[ "$mode" == format ]] || { usage >&2; exit 2; }
                format_check=true; shift ;;
            --format-check)
                [[ "$mode" == verify || "$mode" == install ]] || { usage >&2; exit 2; }
                format_check=true; shift ;;
            --offline)
                [[ "$mode" != clean && "$mode" != format ]] || { usage >&2; exit 2; }
                offline=true; shift ;;
            -j|--jobs|--jobs=*)
                [[ "$mode" != clean && "$mode" != format ]] || { usage >&2; exit 2; }
                if [[ "$1" == --jobs=* ]]; then
                    value=${1#--jobs=}; shift
                else
                    [[ $# -ge 2 ]] || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
                    value=$2; shift 2
                fi
                [[ "$value" =~ ^[1-9][0-9]*$ ]] || { printf 'Jobs must be a positive integer.\n' >&2; exit 2; }
                jobs=(--jobs "$value") ;;
            *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
        esac
    done
fi
[[ "$mode" == clean-build ]] && clean_first=true
cd -- "$root"
[[ -f Cargo.toml && -f Cargo.lock ]] || { printf 'Incomplete source checkout: Cargo.toml and Cargo.lock are required.\n' >&2; exit 2; }
# Keep all compiler output here, even when a global Cargo config uses shared caches.
# Older Cargo versions that do not support build-dir simply ignore that env key.
export CARGO_TARGET_DIR="$root/target"
export CARGO_BUILD_TARGET_DIR="$root/target"
export CARGO_BUILD_BUILD_DIR="$root/target"
common=(--locked --target-dir "$CARGO_TARGET_DIR")
if $offline; then common+=(--offline); fi
profile_args=()
if [[ "$profile" == release ]]; then profile_args=(--release); fi
binary="$CARGO_TARGET_DIR/$profile/ollama-workbench"

require_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'Required command is missing: %s\n' "$1" >&2
        printf 'No compiler, test or installation success is claimed.\n' >&2
        return 127
    fi
}

guard_target() {
    # Refuse redirected output roots, including dangling symlinks.
    local path
    for path in "$CARGO_TARGET_DIR" "$CARGO_TARGET_DIR/debug" "$CARGO_TARGET_DIR/release"; do
        if [[ -L "$path" ]]; then
            printf 'Refusing a symlinked build directory: %s\n' "$path" >&2
            printf 'Use a normal target directory inside this checkout.\n' >&2
            return 2
        fi
    done
}

clean_target() {
    printf '\n== Clean checkout build artifacts ==\n'
    local options=(--target-dir "$CARGO_TARGET_DIR")
    if $dry_run; then options+=(--dry-run); fi
    cargo clean "${options[@]}" || return "$?"
    printf 'Source, lockfile, Cargo registry cache, conversations and Ollama models were not removed.\n'
}

format_source() {
    printf '\n== Rust formatting ==\n'
    if $format_check; then
        cargo fmt --all -- --check || return "$?"
    else
        cargo fmt --all || return "$?"
    fi
}

check_source() {
    printf '\n== Check all Rust targets ==\n'
    cargo check --all-targets "${common[@]}" "${jobs[@]}" || return "$?"
}

test_source() {
    printf '\n== Module and integration tests ==\n'
    cargo test --all-targets "${common[@]}" "${jobs[@]}" || return "$?"
    printf '\n== Documentation tests ==\n'
    cargo test --doc "${common[@]}" "${jobs[@]}" || return "$?"
}

verify_identity() {
    if [[ ! -f "$binary" || ! -x "$binary" || -L "$binary" ]]; then
        printf 'Cargo did not create the expected executable: %s\n' "$binary" >&2
        printf 'These scripts build the native host; a custom build.target configuration is not supported.\n' >&2
        return 1
    fi
    local version identity
    version=$(awk '/^version = / {gsub(/"/, "", $3); print $3; exit}' "$root/Cargo.toml")
    identity=$("$binary" --version) || return "$?"
    if [[ "$identity" != "AI Workbench $version (source "*')' ]]; then
        printf 'Refusing unexpected build identity: %s\n' "$identity" >&2
        return 1
    fi
    printf '%s\n' "$identity"
}

build_source() {
    printf '\n== Build %s executable ==\n' "$profile"
    cargo build --bin ollama-workbench "${profile_args[@]}" "${common[@]}" "${jobs[@]}" || return "$?"
    verify_identity || return "$?"
    printf 'Executable: %s\n' "$binary"
}

verify_source() {
    if $format_check; then format_source || return "$?"; fi
    check_source || return "$?"
    test_source || return "$?"
    printf '\n== Strict Clippy ==\n'
    cargo clippy --all-targets "${common[@]}" "${jobs[@]}" -- -D warnings || return "$?"
    build_source || return "$?"
    printf '\nAll requested compiler, test and lint gates passed.\n'
    printf 'Actual Ollama inference, graphics drivers and optional engines require separate runtime checks.\n'
}

install_binary() {
    # This function is reached only after verify_source succeeds in this process.
    local bindir="$HOME/.local/bin"
    local datadir="${XDG_DATA_HOME:-$HOME/.local/share}"
    local appdir="$datadir/applications"
    local icondir="$datadir/icons/hicolor/scalable/apps"
    local stage='' desktop_stage='' icon_stage='' backup_stage='' exec_path
    case "$bindir$datadir" in
        *$'\n'*|*$'\r'*) printf 'Installation paths must not contain newline characters.\n' >&2; return 2 ;;
    esac
    mkdir -p -- "$bindir" "$appdir" "$icondir"
    # EXIT traps run in perform()'s pipeline subshell, not the caller's shell.
    # Expand filenames now because these locals leave scope before that trap runs.
    stage=$(mktemp "$bindir/.ollama-workbench.install.XXXXXXXX")
    desktop_stage=$(mktemp "$appdir/.ollama-workbench.desktop.XXXXXXXX")
    icon_stage=$(mktemp "$icondir/.ollama-workbench.icon.XXXXXXXX")
    backup_stage=$(mktemp "$bindir/.ollama-workbench.backup.XXXXXXXX")
    printf -v cleanup_command 'rm -f -- %q %q %q %q' "$stage" "$desktop_stage" "$icon_stage" "$backup_stage"
    trap "$cleanup_command" EXIT
    install -m 755 "$binary" "$stage"
    install -m 644 "$root/assets/ollama-workbench.svg" "$icon_stage"
    exec_path=$(printf '%s' "$bindir/ollama-workbench" | sed 's/\\/\\\\/g; s/"/\\"/g; s/%/%%/g; s/\$/\\$/g; s/`/\\`/g')
    cat > "$desktop_stage" <<EOF
[Desktop Entry]
Type=Application
Name=AI Workbench
Comment=Native local AI workspace for local models
Exec="$exec_path"
Icon=org.localtools.AIWorkbench
StartupWMClass=org.localtools.AIWorkbench
Terminal=false
Categories=Development;Utility;
Keywords=Ollama;AI;Chat;Rust;Local;
StartupNotify=true
EOF
    chmod 644 "$desktop_stage"
    if [[ -e "$bindir/ollama-workbench" ]]; then
        cp -p -- "$bindir/ollama-workbench" "$backup_stage"
        mv -f -- "$backup_stage" "$bindir/ollama-workbench.previous"
    fi
    mv -f -- "$icon_stage" "$icondir/org.localtools.AIWorkbench.svg"
    mv -f -- "$desktop_stage" "$appdir/org.localtools.AIWorkbench.desktop"
    # The launcher and icon from before the rename to AI Workbench.
    rm -f -- "$appdir/org.localtools.OllamaWorkbench.desktop" "$icondir/org.localtools.OllamaWorkbench.svg"
    mv -f -- "$stage" "$bindir/ollama-workbench"
    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "$appdir" >/dev/null 2>&1 || true
    fi
    printf '\nInstalled: %s\n' "$bindir/ollama-workbench"
    printf 'Close the old window and reopen Workbench. Existing data and models were not removed.\n'
}

perform() {
    printf 'AI Workbench: %s\nUTC: %s\nCheckout: %s\n' "$mode" "$(date -u +%FT%TZ)" "$root"
    require_tool cargo || return "$?"
    require_tool flock || return "$?"
    if [[ "$mode" != clean && "$mode" != format ]]; then
        require_tool rustc || return "$?"
        rustc --version || return "$?"
    fi
    cargo --version || return "$?"
    [[ ! -L "$root/.workbench-build.lock" ]] || { printf 'Refusing symlinked build lock.\n' >&2; return 2; }
    exec 9> "$root/.workbench-build.lock"
    flock 9 || return "$?"
    guard_target || return "$?"
    if $clean_first; then clean_target || return "$?"; fi
    case "$mode" in
        build|clean-build|run) build_source || return "$?" ;;
        clean) clean_target || return "$?" ;;
        check) check_source || return "$?" ;;
        test) test_source || return "$?" ;;
        verify|rebuild-run) verify_source || return "$?" ;;
        install) verify_source || return "$?"; install_binary ;;
        format) format_source || return "$?" ;;
    esac
}

require_tool tee
mkdir -p "$root/reports/local-build"
log="$root/reports/local-build/$mode-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
set +e
( set -e; perform ) 2>&1 | tee "$log"
statuses=("${PIPESTATUS[@]}")
set -e
status=${statuses[0]}
if [[ "$status" -eq 0 && ${statuses[1]} -ne 0 ]]; then status=${statuses[1]}; fi
printf '\nLog: %s\n' "$log"
if [[ "$status" -ne 0 ]]; then
    printf 'FAILED: %s (exit %s).\n' "$mode" "$status" >&2
    exit "$status"
fi
if [[ "$mode" == run || "$mode" == rebuild-run ]]; then
    # The build lock is no longer held when the GUI is launched.
    exec "$binary" "${app_args[@]}"
fi
