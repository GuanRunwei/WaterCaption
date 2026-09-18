#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
PACKAGE_DIR="${PACKAGE_DIR:-internvl_chat_llava}"
MODULE_NAME="${MODULE_NAME:-llava.serve.gradio_web_server}"
MODEL_PATH="${MODEL_PATH:-./checkpoints/llava-v1.5-13b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "$PACKAGE_DIR"
python -m "$MODULE_NAME" --model-path "$MODEL_PATH" --host "$HOST" --port "$PORT"
