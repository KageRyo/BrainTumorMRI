#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/mnt/8tb_hdd/ryo/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-dl-class-ryo}"
CONFIG="${CONFIG:-configs/convnext_base_mtl.yaml}"

cd "$ROOT_DIR"
"$CONDA_BIN" run -n "$ENV_NAME" python scripts/preflight.py --config "$CONFIG" "$@"
