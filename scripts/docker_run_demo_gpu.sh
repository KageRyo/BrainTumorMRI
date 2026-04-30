#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-brain-tumor-mri-demo:gpu}" \
DEVICE="${DEVICE:-cuda}" \
DOCKER_GPUS="${DOCKER_GPUS:-device=0}" \
"$(dirname "${BASH_SOURCE[0]}")/docker_run_demo.sh" "$@"
