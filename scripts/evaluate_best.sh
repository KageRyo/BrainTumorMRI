#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-brain-tumor-mri}"
RUN_DIR="${RUN_DIR:-outputs/convnext_tiny_mtl}"
CHECKPOINT="${CHECKPOINT:-$RUN_DIR/best.pt}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$CONDA_BIN" run -n "$ENV_NAME" python -m brain_tumor_mri.evaluate \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE" \
  "$@"
