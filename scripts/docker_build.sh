#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-brain-tumor-mri-demo}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"

cd "$ROOT_DIR"
if ! docker info >/dev/null 2>&1; then
  echo "Docker is not available for this user." >&2
  echo "Run with Docker permissions, for example:" >&2
  echo "  sudo IMAGE_NAME=$IMAGE_NAME $0" >&2
  echo "or add the user to the docker group and start a new login session." >&2
  exit 1
fi

docker build -f "$DOCKERFILE" -t "$IMAGE_NAME" "$ROOT_DIR" "$@"
