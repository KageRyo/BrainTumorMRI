#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-brain-tumor-mri-demo:gpu}"
DOCKER_GPUS="${DOCKER_GPUS:-device=0}"
CHECKPOINT="${CHECKPOINT:-outputs/convnext_tiny_mtl/best.pt}"
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
  python -m brain_tumor_mri.evaluate \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE" \
    "$@"
