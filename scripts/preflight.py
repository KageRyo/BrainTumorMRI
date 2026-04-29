from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch

from brisc_mtl.config import load_config
from brisc_mtl.data import build_samples, class_counts, make_loader
from brisc_mtl.model import ConvNeXtUNetMultiTask


def status(name: str, ok: bool, detail: str) -> bool:
    marker = "OK" if ok else "FAIL"
    print(f"[{marker}] {name}: {detail}")
    return ok


def check_imports() -> bool:
    missing: list[str] = []
    for module in ("monai", "timm", "kagglehub", "sklearn", "PIL", "yaml"):
        try:
            __import__(module)
        except Exception as exc:  # pragma: no cover - diagnostic script
            missing.append(f"{module} ({exc})")
    return status("imports", not missing, "all required packages imported" if not missing else ", ".join(missing))


def check_cuda(require_cuda: bool) -> bool:
    available = torch.cuda.is_available()
    if available:
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        return status("cuda", True, f"torch={torch.__version__}, cuda={torch.version.cuda}, devices={names}")
    detail = f"torch={torch.__version__}, cuda_build={torch.version.cuda}, device_count={torch.cuda.device_count()}"
    return status("cuda", not require_cuda, detail)


def check_dataset(cfg: dict[str, Any]) -> bool:
    try:
        train_samples = build_samples(cfg["data_root"], split="train")
        test_samples = build_samples(cfg["data_root"], split="test")
    except Exception as exc:
        return status("dataset", False, str(exc))

    detail = (
        f"train={len(train_samples)} {class_counts(train_samples)}, "
        f"test={len(test_samples)} {class_counts(test_samples)}"
    )
    return status("dataset", True, detail)


def check_loader(cfg: dict[str, Any]) -> bool:
    try:
        samples = build_samples(cfg["data_root"], split="train")
        loader = make_loader(
            samples[:8],
            image_size=cfg["data"]["image_size"],
            batch_size=4,
            num_workers=0,
            training=True,
        )
        batch = next(iter(loader))
    except Exception as exc:
        return status("loader", False, str(exc))

    detail = (
        f"image={tuple(batch['image'].shape)} {batch['image'].dtype}, "
        f"mask={tuple(batch['mask'].shape)} range=({batch['mask'].min().item():.0f}, {batch['mask'].max().item():.0f})"
    )
    return status("loader", True, detail)


def check_model(cfg: dict[str, Any]) -> bool:
    model_cfg = dict(cfg["model"])
    model_cfg["pretrained"] = False
    try:
        model = ConvNeXtUNetMultiTask(**model_cfg)
        image_size = int(cfg["data"]["image_size"])
        with torch.no_grad():
            outputs = model(torch.zeros(1, model_cfg["in_channels"], image_size, image_size))
    except Exception as exc:
        return status("model", False, str(exc))

    cls_shape = tuple(outputs["class_logits"].shape)
    mask_shape = tuple(outputs["mask_logits"].shape)
    return status("model", True, f"class_logits={cls_shape}, mask_logits={mask_shape}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the BRISC project before starting training.")
    parser.add_argument("--config", default="configs/convnext_base_mtl.yaml")
    parser.add_argument("--allow-cpu", action="store_true", help="Do not fail when CUDA is unavailable.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    print(f"python={sys.version.split()[0]} executable={sys.executable}")
    print(f"platform={platform.platform()}")
    print(f"config={cfg_path}")
    print(f"data_root={cfg['data_root']}")
    print(f"output_dir={cfg['output_dir']}")

    checks = [
        status("python", sys.version_info >= (3, 10), f"requires >=3.10, got {sys.version.split()[0]}"),
        check_imports(),
        check_cuda(require_cuda=not args.allow_cpu),
        check_dataset(cfg),
        check_loader(cfg),
        check_model(cfg),
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
