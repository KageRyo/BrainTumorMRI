#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-brain-tumor-mri-demo:gpu}"
DOCKER_GPUS="${DOCKER_GPUS:-device=0}"
CONFIG="${CONFIG:-configs/convnext_tiny_mtl.yaml}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/convnext_tiny_mtl}"
DEVICE="${DEVICE:-cuda}"

cd "$ROOT_DIR"

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not available for this user." >&2
  echo "Run in a refreshed docker group shell, for example: sg docker -c '$0'" >&2
  exit 1
fi

docker run --rm --gpus "$DOCKER_GPUS" \
  -v "$ROOT_DIR/data:/app/data:ro" \
  -v "$ROOT_DIR/outputs:/app/outputs" \
  "$IMAGE_NAME" \
  python -m brisc_mtl.train \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
