#!/usr/bin/env bash
# OttoSplatto launcher — installs deps if needed, then starts TUI or CLI
set -euo pipefail
cd "$(dirname "$0")"

# Ensure pip deps are installed
if ! python3 -c "import textual" 2>/dev/null; then
    echo "Installing OttoSplatto dependencies..."
    pip install -q textual rich
fi

# Check system deps
for cmd in ffmpeg colmap; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "WARNING: $cmd not found — install it before running the pipeline"
    fi
done

if ! command -v nvidia-smi &>/dev/null; then
    echo "WARNING: nvidia-smi not found — CUDA GPU required for training"
fi

# Show device on first run
if [ "${1:-}" = "" ] || [ "${1:-}" = "tui" ]; then
    python3 -c "from pipeline.device import detect; d=detect(); g=d.primary_gpu; print(f'OttoSplatto — {g.name} ({d.arch})' if g else 'OttoSplatto — no GPU detected')" 2>/dev/null || true
fi

exec python3 main.py "$@"
