#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-brain-tumor-mri}"
CONFIG="${CONFIG:-configs/convnext_base_mtl.yaml}"

cd "$ROOT_DIR"
"$CONDA_BIN" run -n "$ENV_NAME" python scripts/preflight.py --config "$CONFIG" "$@"
