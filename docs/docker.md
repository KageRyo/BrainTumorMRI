# Docker Demo

This project ships separate Docker images for CPU and GPU inference with the Gradio demo. Checkpoints and datasets
are not copied into either image; mount them at runtime.

## CPU Build

```bash
scripts/docker_build.sh
```

If Docker reports a socket permission error, either run the script with `sudo` or add the user to the `docker` group:

```bash
sudo usermod -aG docker "$USER"
```

After changing group membership, log out and back in before running Docker again.

Optional variables:

```bash
IMAGE_NAME=brain-tumor-mri-demo:local scripts/docker_build.sh
DOCKERFILE=Dockerfile scripts/docker_build.sh
```

The default `Dockerfile` installs CPU-only PyTorch wheels. This keeps the image portable and avoids pulling CUDA
runtime packages into the CPU demo image.

## CPU Run With Default Checkpoint

The default checkpoint path is `outputs/convnext_tiny_mtl/best.pt`.

```bash
scripts/docker_run_demo.sh
```

Open <http://127.0.0.1:7860>.

## GPU Build

Use the GPU image when the host has NVIDIA Container Toolkit configured:

```bash
scripts/docker_build_gpu.sh
```

Equivalent direct command:

```bash
DOCKERFILE=Dockerfile.gpu IMAGE_NAME=brain-tumor-mri-demo:gpu scripts/docker_build.sh
```

## GPU Run

```bash
scripts/docker_run_demo_gpu.sh
```

Equivalent explicit variables:

```bash
IMAGE_NAME=brain-tumor-mri-demo:gpu \
DEVICE=cuda \
DOCKER_GPUS=device=0 \
scripts/docker_run_demo.sh
```

## Run With A Different Checkpoint

```bash
CHECKPOINT=outputs/convnext_base_mtl/best.pt scripts/docker_run_demo.sh
```

For a checkpoint outside `outputs/`, pass an absolute path:

```bash
CHECKPOINT=/path/to/best.pt scripts/docker_run_demo.sh
```

The script mounts only the checkpoint directory read-only.

## Ports And Options

```bash
PORT=7861 THRESHOLD=0.45 scripts/docker_run_demo.sh
```

The CPU Docker image is intended for portable demo serving:

```bash
DEVICE=cpu scripts/docker_run_demo.sh
```

GPU serving uses [../Dockerfile.gpu](../Dockerfile.gpu), which is based on a CUDA-enabled PyTorch runtime image.

## Direct Docker Commands

Equivalent manual build:

```bash
docker build -t brain-tumor-mri-demo .
```

Equivalent manual GPU build:

```bash
docker build -f Dockerfile.gpu -t brain-tumor-mri-demo:gpu .
```

Equivalent manual run:

```bash
docker run --rm -p 7860:7860 \
  -v "$PWD/outputs:/app/outputs:ro" \
  brain-tumor-mri-demo
```

Equivalent manual GPU run:

```bash
docker run --rm --gpus all -p 7860:7860 \
  -v "$PWD/outputs:/app/outputs:ro" \
  brain-tumor-mri-demo:gpu
```

## Report Assets

After training, regenerate evaluation outputs and figures with:

```bash
RUN_DIR=outputs/convnext_tiny_mtl \
CHECKPOINT=outputs/convnext_tiny_mtl/best.pt \
GPU_ID=0 \
scripts/make_report_assets.sh
```

This updates `metrics.json`, training curves, qualitative prediction grids, and the report copy of the confusion
matrix.

## Troubleshooting

- Checkpoint not found: set `CHECKPOINT=outputs/convnext_tiny_mtl/best.pt` or mount a directory that contains the
  requested `best.pt`. Checkpoints are intentionally not copied into the Docker image.
- Port already in use: run with another host port, for example `PORT=7861 scripts/docker_run_demo.sh`, then open
  <http://127.0.0.1:7861>.
- Docker permission denied: add the user to the `docker` group and start a new login session, or run through a shell
  that has Docker group access.
- GPU container cannot see CUDA: install and configure NVIDIA Container Toolkit on the host, then verify
  `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`.
