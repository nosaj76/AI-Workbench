#!/usr/bin/env bash
# Run on an isolated Linux machine with Xvfb, Mesa, Python Xlib, Pillow and Tesseract.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
binary="$root/target/release/ollama-workbench"
report="$root/reports/local-build/native-chat"
mkdir -p "$report"
scratch=$(mktemp -d)
server=
cleanup() {
    if [[ -n "$server" ]]; then
        kill "$server" 2>/dev/null || true
        wait "$server" 2>/dev/null || true
    fi
    rm -rf -- "$scratch"
}
trap cleanup EXIT
mkdir -p "$scratch/ollama" "$scratch/data" "$scratch/runtime"
chmod 700 "$scratch/runtime"
archive="$scratch/ollama.tar.zst"
curl --fail --location --silent --show-error --retry 2 --max-time 450 \
    --output "$archive" https://github.com/ollama/ollama/releases/download/v0.40.17/ollama-linux-amd64.tar.zst
printf '%s  %s\n' 1df9d6228bac5a22e420b242708c0d10e5b886db97bab99d446a41760f4e79af "$archive" | sha256sum --check
tar --zstd -xf "$archive" -C "$scratch/ollama" --exclude='*cuda*' --exclude='*rocm*' --exclude='*vulkan*'
rm -- "$archive"
export XDG_CONFIG_HOME="$scratch/config" XDG_DATA_HOME="$scratch/share"
export XDG_RUNTIME_DIR="$scratch/runtime" OLLAMA_MODELS="$scratch/models"
export OLLAMA_HOST=127.0.0.1:11439 OLLAMA_NO_CLOUD=1 OLLAMA_NUM_PARALLEL=1
ollama="$scratch/ollama/bin/ollama"
"$ollama" serve > "$report/ollama.log" 2>&1 &
server=$!
for ((i=0; i<80; i++)); do
    if curl --fail --silent "http://$OLLAMA_HOST/api/version" > "$report/ollama-version.json"; then break; fi
    kill -0 "$server" || { cat "$report/ollama.log"; exit 1; }
    sleep .25
done
curl --fail --silent "http://$OLLAMA_HOST/api/version"
model=qwen2.5:0.5b
if ! timeout 240 "$ollama" pull "$model" > "$scratch/pull.log" 2>&1; then
    tail -c 2000 "$scratch/pull.log" >&2
    exit 1
fi
curl --fail --silent "http://$OLLAMA_HOST/api/tags" > "$report/models.json"
export LIBGL_ALWAYS_SOFTWARE=1
unset WAYLAND_DISPLAY
"$binary" --version
xvfb-run -a -s '-screen 0 1440x1000x24 -nolisten tcp' \
    /usr/bin/python3 "$root/scripts/verify_native_chat.py" "$binary" "$scratch/data" "$report" "$model"
