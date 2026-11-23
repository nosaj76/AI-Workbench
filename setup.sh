#!/usr/bin/env bash
# One-command setup: system build libraries, Rust, optional Ollama and helper
# programs, then a verified build and per-user install via ./install.sh.
# Run as your normal user; sudo is used only for the system package manager.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd "$root"

MIN_RUST=1.89
assume_yes=false
dry_run=false
optional=ask
ollama=ask
model=""
install_app=true
jobs=()

usage() {
    cat <<'EOF'
Usage: ./setup.sh [options]

Installs everything AI Workbench needs, then builds, tests and installs it for
the current user (~/.local/bin/ollama-workbench and a desktop entry).

  -y, --yes          Answer yes to every question (unattended)
      --optional     Install optional helper programs without asking
      --no-optional  Skip optional helper programs
      --ollama       Install Ollama without asking if it is missing
      --no-ollama    Do not install Ollama
      --model NAME   Pull an Ollama model afterwards (e.g. qwen3:8b)
      --no-install   Set up dependencies only; do not build or install
      --jobs N       Limit compiler parallelism (useful on 16 GB machines)
      --dry-run      Print every command instead of running it
  -h, --help         Show this help

Supported package managers: apt (Debian/Ubuntu/Mint), dnf (Fedora/RHEL),
pacman (Arch/Manjaro) and zypper (openSUSE).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -y|--yes) assume_yes=true ;;
        --optional) optional=yes ;;
        --no-optional) optional=no ;;
        --ollama) ollama=yes ;;
        --no-ollama) ollama=no ;;
        --model) shift; model=${1:?--model needs a model name} ;;
        --no-install) install_app=false ;;
        --jobs) shift; [[ ${1-} =~ ^[1-9][0-9]*$ ]] || { echo "--jobs needs a positive number" >&2; exit 2; }; jobs=(--jobs "$1") ;;
        --dry-run) dry_run=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1 (see ./setup.sh --help)" >&2; exit 2 ;;
    esac
    shift
done

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

run() {
    if $dry_run; then printf '    [dry-run] %s\n' "$*"; else "$@"; fi
}
# Fetch the complete installer before allowing it to change the system.
run_installer() (
    local url=$1 installer
    shift
    if $dry_run; then
        note "[dry-run] Download $url, then run it with sh $*"
        return
    fi
    installer=$(mktemp)
    trap 'rm -f -- "$installer"' EXIT
    curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL -o "$installer" "$url" || return "$?"
    sh "$installer" "$@"
)

# yes/no question; unattended runs (no terminal) take the default.
confirm() {
    local question=$1 default=${2:-y} reply
    $assume_yes && return 0
    if [[ ! -t 0 ]]; then [[ $default == y ]]; return; fi
    local hint="[Y/n]"; [[ $default == n ]] && hint="[y/N]"
    read -r -p "    $question $hint " reply || reply=""
    reply=${reply:-$default}
    [[ $reply =~ ^[Yy] ]]
}

[[ $(id -u) -ne 0 ]] || die "Run ./setup.sh as your normal user, not root. It asks for sudo only when installing system packages."
[[ $(uname -s) == Linux ]] || die "This setup script supports Linux. AI Workbench's terminals and installer are Linux-only in this release."
[[ -f Cargo.toml && -f Cargo.lock ]] || die "Run this from a complete AI Workbench checkout (Cargo.toml and Cargo.lock are missing)."

$dry_run || command -v sudo >/dev/null 2>&1 || die "sudo is required to install system packages."
sudo_cmd=(sudo)

# ---------------------------------------------------------------- packages
if command -v apt-get >/dev/null 2>&1; then
    pm=apt
    build_pkgs=(build-essential pkg-config curl git util-linux libxkbcommon-dev libgl1-mesa-dev libwayland-dev bubblewrap passt)
    optional_pkgs=(poppler-utils tesseract-ocr tesseract-ocr-eng ffmpeg espeak-ng lynx w3m texlive-latex-base xdg-utils)
    pm_update() { run "${sudo_cmd[@]}" apt-get update; }
    pm_install() { run "${sudo_cmd[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"; }
elif command -v dnf >/dev/null 2>&1; then
    pm=dnf
    build_pkgs=(gcc gcc-c++ make pkgconf-pkg-config curl git util-linux libxkbcommon-devel mesa-libGL-devel wayland-devel bubblewrap passt)
    optional_pkgs=(poppler-utils tesseract tesseract-langpack-eng ffmpeg-free espeak-ng lynx w3m texlive-scheme-basic xdg-utils)
    pm_update() { :; }
    pm_install() { run "${sudo_cmd[@]}" dnf install -y "$@"; }
elif command -v pacman >/dev/null 2>&1; then
    pm=pacman
    build_pkgs=(base-devel pkgconf curl git util-linux libxkbcommon mesa wayland bubblewrap passt)
    optional_pkgs=(poppler tesseract tesseract-data-eng ffmpeg espeak-ng lynx w3m texlive-basic xdg-utils)
    pm_update() { :; }  # pacman -S without -u would risk a partial upgrade
    pm_install() { run "${sudo_cmd[@]}" pacman -S --needed --noconfirm "$@"; }
elif command -v zypper >/dev/null 2>&1; then
    pm=zypper
    build_pkgs=(gcc gcc-c++ make pkg-config curl git util-linux libxkbcommon-devel Mesa-libGL-devel wayland-devel bubblewrap passt)
    optional_pkgs=(poppler-tools tesseract-ocr tesseract-ocr-traineddata-english ffmpeg espeak-ng lynx w3m texlive-latex xdg-utils)
    pm_update() { run "${sudo_cmd[@]}" zypper --non-interactive refresh; }
    pm_install() { run "${sudo_cmd[@]}" zypper --non-interactive install --no-recommends "$@"; }
else
    die "No supported package manager found (apt, dnf, pacman or zypper). Install a C compiler, pkg-config, curl and the X11/Wayland/OpenGL development libraries yourself, then rerun with --no-optional."
fi

step "Installing build tools and desktop libraries ($pm)"
note "${build_pkgs[*]}"
pm_update
pm_install "${build_pkgs[@]}"

if [[ $optional == ask ]]; then
    step "Optional helper programs"
    note "PDF text and page images (poppler), OCR (tesseract), audio/video (ffmpeg),"
    note "speech (espeak-ng), web search without an API key (lynx, w3m), LaTeX reports."
    note "The app works without them; the matching features stay unavailable."
    confirm "Install them?" y && optional=yes || optional=no
fi
if [[ $optional == yes ]]; then
    step "Installing optional helper programs"
    # One at a time, so a package missing from this distribution's repositories
    # (ffmpeg on some Fedora setups, for example) does not stop the rest.
    for pkg in "${optional_pkgs[@]}"; do
        pm_install "$pkg" || warn "could not install optional package $pkg; skipping it"
    done
fi

# -------------------------------------------------------------------- Rust
version_ge() { [[ $(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1) == "$2" ]]; }

step "Rust toolchain (needs $MIN_RUST or newer, with clippy and rustfmt)"
[[ -f "$HOME/.cargo/env" ]] && . "$HOME/.cargo/env"
if command -v rustup >/dev/null 2>&1; then
    current=$(rustc --version 2>/dev/null | awk '{print $2}' || true)
    if [[ -z $current ]] || ! version_ge "$current" "$MIN_RUST"; then
        note "Updating Rust ${current:-(none)} to the latest stable release"
        run rustup toolchain install stable --profile minimal
        run rustup default stable
    else
        note "Found Rust $current"
    fi
    run rustup component add clippy rustfmt
elif command -v cargo >/dev/null 2>&1; then
    current=$(rustc --version | awk '{print $2}')
    version_ge "$current" "$MIN_RUST" || die "Rust $current is installed without rustup and is older than $MIN_RUST. Remove the distribution's rust/cargo packages and rerun, so rustup can install a current toolchain."
    command -v cargo-clippy >/dev/null 2>&1 || warn "clippy is missing; ./install.sh needs it. Install your distribution's clippy package."
    note "Found Rust $current (not managed by rustup)"
else
    note "Installing Rust with rustup (into ~/.cargo, no sudo)"
    run_installer https://sh.rustup.rs -y --profile minimal --component clippy,rustfmt
    $dry_run || . "$HOME/.cargo/env"
fi

# ------------------------------------------------------------------ Ollama
step "Ollama (runs the local models)"
if command -v ollama >/dev/null 2>&1; then
    note "Found $(ollama --version 2>/dev/null | head -1 || echo ollama)"
else
    if [[ $ollama == ask ]]; then
        note "AI Workbench talks to Ollama at http://127.0.0.1:11434. Skip this if Ollama"
        note "runs on another machine; set its address in Settings afterwards."
        confirm "Install Ollama with its official installer (uses sudo)?" y && ollama=yes || ollama=no
    fi
    if [[ $ollama == yes ]]; then
        # Current official Linux archives use zstd, including minimal installs.
        pm_install zstd
        run_installer https://ollama.com/install.sh
    else
        note "Skipped. Install it later from https://ollama.com/download"
    fi
fi
if [[ -n $model ]]; then
    if command -v ollama >/dev/null 2>&1 || $dry_run; then
        step "Pulling model $model"
        run ollama pull "$model"
    else
        warn "Ollama is not installed, so model $model was not pulled."
    fi
fi

# ----------------------------------------------------------- build/install
if $install_app; then
    step "Building, testing and installing AI Workbench"
    note "This compiles in release mode and runs the full test suite first; it can take"
    note "several minutes. Nothing is installed unless every check passes."
    run ./install.sh "${jobs[@]}"
    if ! $dry_run; then
        step "Readiness report"
        "$HOME/.local/bin/ollama-workbench" --doctor || {
            doctor_status=$?
            warn "AI Workbench was installed, but its readiness check failed (exit $doctor_status)."
            exit "$doctor_status"
        }
    fi
fi

step "Done"
if $install_app; then
    note "Start AI Workbench from your applications menu, or run: ollama-workbench"
    case ":$PATH:" in
        *":$HOME/.local/bin:"*) ;;
        *) warn "$HOME/.local/bin is not on your PATH; add it, or start the app from the menu." ;;
    esac
else
    note "Dependencies are ready. Build and install with: ./install.sh"
fi
if ! command -v ollama >/dev/null 2>&1 && ! $dry_run; then
    note "Remember to install or point the app at an Ollama server before chatting."
fi
note "The first start opens Setup, which can download recommended models for you."
