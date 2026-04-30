.PHONY: lint test preflight evaluate report-assets docker-build docker-run docker-build-gpu docker-run-gpu

CONDA_BIN ?= conda
ENV_NAME ?= brain-tumor-mri
GPU_ID ?= 0
RUN_DIR ?= outputs/convnext_tiny_mtl

lint:
	$(CONDA_BIN) run -n $(ENV_NAME) ruff check src app scripts tests

test:
	$(CONDA_BIN) run -n $(ENV_NAME) pytest

preflight:
	GPU_ID=$(GPU_ID) CONDA_BIN=$(CONDA_BIN) ENV_NAME=$(ENV_NAME) scripts/preflight.sh

evaluate:
	GPU_ID=$(GPU_ID) CONDA_BIN=$(CONDA_BIN) ENV_NAME=$(ENV_NAME) RUN_DIR=$(RUN_DIR) scripts/evaluate_best.sh

report-assets:
	GPU_ID=$(GPU_ID) CONDA_BIN=$(CONDA_BIN) ENV_NAME=$(ENV_NAME) RUN_DIR=$(RUN_DIR) scripts/make_report_assets.sh

docker-build:
	scripts/docker_build.sh

docker-run:
	scripts/docker_run_demo.sh

docker-build-gpu:
	scripts/docker_build_gpu.sh

docker-run-gpu:
	scripts/docker_run_demo_gpu.sh
