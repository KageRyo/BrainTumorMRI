# Repository Guidelines

## Project Structure & Module Organization

This repository implements a PyTorch + MONAI multitask brain tumor project for BRISC 2025.

- `src/brisc_mtl/`: package source code.
  - `data.py`: BRISC sample discovery, transforms, dataset, and dataloaders.
  - `model.py`: ConvNeXt shared encoder with classification and U-Net segmentation heads.
  - `train.py`, `evaluate.py`, `predict.py`: CLI entry points.
  - `config.py`, `utils.py`: configuration and shared helpers.
- `configs/`: experiment configuration files, currently `convnext_base_mtl.yaml`.
- `scripts/`: utility scripts such as Kaggle dataset download.
- `data/`, `outputs/`, checkpoints, and local environments are ignored and must not be committed.
- Tests should live under `tests/` when added.

## Build, Test, and Development Commands

Create and activate the environment:

```bash
/mnt/8tb_hdd/ryo/miniconda3/bin/conda env create -f environment.yml
conda activate dl-class-ryo
```

Download BRISC 2025:

```bash
python scripts/download_dataset.py --copy --out data/brisc2025
```

Train, evaluate, and run single-image inference:

```bash
python -m brisc_mtl.train --config configs/convnext_base_mtl.yaml
python -m brisc_mtl.evaluate --checkpoint outputs/convnext_base_mtl/best.pt
python -m brisc_mtl.predict --checkpoint outputs/convnext_base_mtl/best.pt --image PATH_TO_IMAGE
```

Run linting:

```bash
ruff check src scripts
```

## Coding Style & Naming Conventions

Use Python 3.10+, 4-space indentation, type hints, and concise functions. Prefer clear module-level helpers over ad hoc inline logic. File and function names use `snake_case`; classes use `PascalCase`. Keep code ASCII unless the existing file requires otherwise. Ruff is configured in `pyproject.toml` with a 120-character line length.

## Testing Guidelines

Use `pytest` for tests. Name test files `tests/test_*.py` and test functions `test_*`. Prioritize fast unit tests for dataset path parsing, config loading, model output shapes, and metric helpers. Avoid tests that require the full Kaggle dataset unless marked as integration tests.

## Commit & Pull Request Guidelines

Existing history uses Conventional Commit style:

- `init`
- `chore(env): add conda and project configuration`
- `feat(data): add BRISC multitask data pipeline`
- `feat(model): add ConvNeXt multitask U-Net`
- `docs(readme): document BRISC multitask workflow`

Use short scopes such as `data`, `model`, `train`, `env`, or `docs`. Pull requests should describe the change, include verification commands, note dataset/checkpoint assumptions, and avoid committing generated artifacts.

## Security & Configuration Tips

Do not commit Kaggle credentials, downloaded datasets, model checkpoints, or `outputs/`. Keep secrets in local environment variables or Kaggle’s standard credential locations.
