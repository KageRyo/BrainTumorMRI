#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-brain-tumor-mri-demo}"
PORT="${PORT:-7860}"
DEVICE="${DEVICE:-cpu}"
CHECKPOINT="${CHECKPOINT:-outputs/convnext_tiny_mtl/best.pt}"
THRESHOLD="${THRESHOLD:-0.5}"

cd "$ROOT_DIR"

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not available for this user." >&2
  echo "Run with Docker permissions, for example:" >&2
  echo "  sudo IMAGE_NAME=$IMAGE_NAME CHECKPOINT=$CHECKPOINT $0" >&2
  echo "or add the user to the docker group and start a new login session." >&2
  exit 1
fi

if [[ "$CHECKPOINT" = /* ]]; then
  HOST_CHECKPOINT="$CHECKPOINT"
else
  HOST_CHECKPOINT="$ROOT_DIR/$CHECKPOINT"
fi

if [[ ! -f "$HOST_CHECKPOINT" ]]; then
  echo "Checkpoint not found: $HOST_CHECKPOINT" >&2
  echo "Set CHECKPOINT=path/to/best.pt or mount the expected file under outputs/." >&2
  exit 1
fi

DOCKER_ARGS=(--rm -p "$PORT:7860")
if [[ -n "${DOCKER_GPUS:-}" ]]; then
  DOCKER_ARGS+=(--gpus "$DOCKER_GPUS")
fi

if [[ "$HOST_CHECKPOINT" == "$ROOT_DIR"/outputs/* ]]; then
  CONTAINER_CHECKPOINT="${HOST_CHECKPOINT#"$ROOT_DIR"/}"
  DOCKER_ARGS+=(-v "$ROOT_DIR/outputs:/app/outputs:ro")
else
  CHECKPOINT_DIR="$(dirname "$HOST_CHECKPOINT")"
  CHECKPOINT_FILE="$(basename "$HOST_CHECKPOINT")"
  CONTAINER_CHECKPOINT="/checkpoints/$CHECKPOINT_FILE"
  DOCKER_ARGS+=(-v "$CHECKPOINT_DIR:/checkpoints:ro")
fi

docker run "${DOCKER_ARGS[@]}" "$IMAGE_NAME" \
  python app/gradio_app.py \
    --checkpoint "$CONTAINER_CHECKPOINT" \
    --device "$DEVICE" \
    --threshold "$THRESHOLD" \
    --host 0.0.0.0 \
    --port 7860
