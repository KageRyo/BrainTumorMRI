from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def device(preference: str = "cuda") -> torch.device:
    if preference not in {"cuda", "cpu", "auto"}:
        raise ValueError(f"Unsupported device preference: {preference}")
    if preference == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "auto":
        return torch.device("cpu")
    raise RuntimeError(
        "CUDA was requested, but PyTorch cannot see a CUDA device. "
        "Check the NVIDIA driver, container/WSL GPU passthrough, and the installed PyTorch CUDA build."
    )
