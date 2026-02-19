#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"

PROFILE="${1:-fast}"
PROMPT="${2:-A cinematic shot of ocean waves at sunset}"

"$VENV_PY" -m ltx_pipelines.mac_native \
  --profile "$PROFILE" \
  --prompt "$PROMPT" \
  --models-root "$ROOT_DIR/models/LTX-2" \
  --output-dir "$ROOT_DIR/output/mac-native"
