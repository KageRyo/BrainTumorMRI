#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-brain-tumor-mri}"
RUN_DIR="${RUN_DIR:-outputs/convnext_tiny_mtl}"
CHECKPOINT="${CHECKPOINT:-$RUN_DIR/best.pt}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-2}"
REPORT_FIGURES_DIR="${REPORT_FIGURES_DIR:-reports/figures}"

cd "$ROOT_DIR"
mkdir -p "$REPORT_FIGURES_DIR"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$CONDA_BIN" run -n "$ENV_NAME" python -m brain_tumor_mri.evaluate \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE"

"$CONDA_BIN" run -n "$ENV_NAME" python scripts/plot_training_curves.py \
  "$RUN_DIR" \
  --out "$REPORT_FIGURES_DIR/$(basename "$RUN_DIR")_history.png"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$CONDA_BIN" run -n "$ENV_NAME" python scripts/make_prediction_grid.py \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE" \
  --samples-per-class "$SAMPLES_PER_CLASS" \
  --out "$REPORT_FIGURES_DIR/qualitative_$(basename "$RUN_DIR").png"

if [[ -f "$RUN_DIR/test_eval/confusion_matrix.png" ]]; then
  cp "$RUN_DIR/test_eval/confusion_matrix.png" "$REPORT_FIGURES_DIR/$(basename "$RUN_DIR")_confusion_matrix.png"
fi
