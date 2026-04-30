#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-brain-tumor-mri}"
CONFIG="${CONFIG:-configs/convnext_base_mtl.yaml}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/smoke_1epoch}"

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$CONDA_BIN" run -n "$ENV_NAME" python -m brain_tumor_mri.train \
  --config "$CONFIG" \
  --device cuda \
  --epochs 1 \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
