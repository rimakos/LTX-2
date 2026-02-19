#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"

PROFILE="${1:-fast}" # fast | quality
PROMPT="${2:-A cinematic shot of ocean waves at sunset}"
OUTPUT_PATH="${3:-$ROOT_DIR/output/ltx2-mac-${PROFILE}.mp4}"

if [[ "$PROFILE" != "fast" && "$PROFILE" != "quality" ]]; then
  echo "Invalid profile '$PROFILE'. Use: fast | quality"
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

PYTORCH_ENABLE_MPS_FALLBACK=1 "$VENV_PY" -m ltx_pipelines.distilled \
  --checkpoint-path "$ROOT_DIR/models/LTX-2/ltx-2-19b-distilled.safetensors" \
  --spatial-upsampler-path "$ROOT_DIR/models/LTX-2/ltx-2-spatial-upscaler-x2-1.0.safetensors" \
  --gemma-root "$ROOT_DIR/models/LTX-2/gemma-3-12b-it-qat-q4_0-unquantized" \
  --prompt "$PROMPT" \
  --output-path "$OUTPUT_PATH" \
  --mac-optimize \
  --mac-profile "$PROFILE"

echo "Done: $OUTPUT_PATH"
