#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/mnt/8tb_hdd/ryo/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-dl-class-ryo}"
RUN_DIR="${RUN_DIR:-outputs/convnext_tiny_mtl}"
CHECKPOINT="${CHECKPOINT:-$RUN_DIR/best.pt}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$CONDA_BIN" run -n "$ENV_NAME" python -m brisc_mtl.evaluate \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE" \
  "$@"
