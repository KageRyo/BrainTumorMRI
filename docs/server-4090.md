# BrainTumorMRI On The 2x RTX 4090 Server

This repository is hosted on a server with two NVIDIA GeForce RTX 4090 GPUs. Use the GPU Docker image for local
inference, evaluation, and training when Docker has NVIDIA Container Toolkit access.

## Refresh Docker Group Access

The `ryo` user has been added to the `docker` group. Existing shells may not see the new group until a new login
session starts. Without logging out, run Docker commands through:

```bash
sg docker -c "docker info"
```

After a fresh login, plain `docker info` should work.

## Check GPU Occupancy

```bash
nvidia-smi
```

Use one GPU per experiment unless you explicitly need both. This keeps the second GPU available for another run or
for existing services.

## Build Images

CPU image:

```bash
sg docker -c "scripts/docker_build.sh"
```

GPU image:

```bash
sg docker -c "scripts/docker_build_gpu.sh"
```

Image names:

| image | purpose |
| --- | --- |
| `brain-tumor-mri-demo` | portable CPU Gradio demo |
| `brain-tumor-mri-demo:gpu` | CUDA Gradio, evaluation, and training |

## Run GPU Demo

Default: expose only host GPU 0 into the container.

```bash
sg docker -c "scripts/docker_run_demo_gpu.sh"
```

Use host GPU 1 instead:

```bash
sg docker -c "DOCKER_GPUS=device=1 scripts/docker_run_demo_gpu.sh"
```

Open <http://127.0.0.1:7860>.

## Evaluate In GPU Docker

```bash
sg docker -c "DOCKER_GPUS=device=0 scripts/docker_evaluate_gpu.sh"
```

This mounts:

```text
data/    -> /app/data:ro
outputs/ -> /app/outputs
```

The default checkpoint is:

```text
outputs/convnext_tiny_mtl/best.pt
```

Evaluate a different checkpoint:

```bash
sg docker -c "DOCKER_GPUS=device=1 CHECKPOINT=outputs/convnext_base_mtl/best.pt scripts/docker_evaluate_gpu.sh"
```

## Train In GPU Docker

```bash
sg docker -c "DOCKER_GPUS=device=0 scripts/docker_train_gpu.sh"
```

Train on GPU 1 with a custom output directory:

```bash
sg docker -c "DOCKER_GPUS=device=1 OUTPUT_DIR=outputs/convnext_tiny_gpu1 scripts/docker_train_gpu.sh"
```

Recommended host-side non-Docker training remains:

```bash
GPU_ID=0 BATCH_SIZE=16 scripts/train_full.sh
```

Use Docker training mainly when you want a containerized reproduction environment.
