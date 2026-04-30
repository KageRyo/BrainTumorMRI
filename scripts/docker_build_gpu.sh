#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-brain-tumor-mri-demo:gpu}" \
DOCKERFILE="${DOCKERFILE:-Dockerfile.gpu}" \
"$(dirname "${BASH_SOURCE[0]}")/docker_build.sh" "$@"
