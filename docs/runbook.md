# Project Runbook

This runbook covers the standard workflow for reproducing the BRISC multitask MRI project.

## 1. Environment

Create or update the conda environment:

```bash
conda env create -f environment.yml
conda activate brain-tumor-mri
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
```

## 2. Dataset

Download BRISC 2025:

```bash
python scripts/download_dataset.py --copy --out data/brisc2025
```

Expected local layout:

```text
data/brisc2025/
  classification_task/
  segmentation_task/
```

## 3. Preflight

Check dependencies, dataset layout, CUDA visibility, dataloader behavior, and model shape:

```bash
scripts/preflight.sh
```

If running inside a restricted sandbox that cannot see CUDA, use:

```bash
scripts/preflight.sh --allow-cpu
```

## 4. Training

Smoke run:

```bash
GPU_ID=0 BATCH_SIZE=8 scripts/train_smoke_1epoch.sh
```

Full default run:

```bash
GPU_ID=0 BATCH_SIZE=16 scripts/train_full.sh
```

Train ConvNeXt-Tiny explicitly:

```bash
GPU_ID=0 \
CONFIG=configs/convnext_tiny_mtl.yaml \
OUTPUT_DIR=outputs/convnext_tiny_mtl \
scripts/train_full.sh
```

## 5. Evaluation

Evaluate the current headline checkpoint:

```bash
GPU_ID=0 RUN_DIR=outputs/convnext_tiny_mtl scripts/evaluate_best.sh
```

Outputs:

```text
outputs/convnext_tiny_mtl/test_eval/
  metrics.json
  confusion_matrix.png
```

## 6. Report Assets

Regenerate report figures and metrics:

```bash
GPU_ID=0 RUN_DIR=outputs/convnext_tiny_mtl scripts/make_report_assets.sh
```

This updates:

```text
reports/figures/<run>_history.png
reports/figures/qualitative_<run>.png
reports/figures/<run>_confusion_matrix.png
outputs/<run>/test_eval/metrics.json
```

Update [reports/report.md](../reports/report.md) if headline metrics change.

## 7. Demo

Local Gradio:

```bash
python app/gradio_app.py \
  --checkpoint outputs/convnext_tiny_mtl/best.pt \
  --device cuda
```

Docker Gradio:

```bash
scripts/docker_build.sh
scripts/docker_run_demo.sh
```

GPU Docker Gradio:

```bash
scripts/docker_build_gpu.sh
scripts/docker_run_demo_gpu.sh
```

See [docker.md](docker.md) for Docker permissions, checkpoint mounting, and port options.

## 8. CI Checks

Run the same checks as GitHub Actions:

```bash
make lint
make test
```

Equivalent direct commands:

```bash
ruff check src app scripts tests
pytest
```

## 9. Current Headline Result

Current checkpoint:

```text
outputs/convnext_tiny_mtl/best.pt
```

Current test metrics:

| metric | value |
| --- | ---: |
| classification accuracy | 0.9940 |
| binary detection accuracy | 1.0000 |
| Dice | 0.8387 |
| IoU | 0.7814 |
| ECE | 0.0055 |
