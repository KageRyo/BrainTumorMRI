from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from brisc_mtl.model import ConvNeXtUNetMultiTask


def build_model(cfg: dict[str, Any]) -> ConvNeXtUNetMultiTask:
    return ConvNeXtUNetMultiTask(**cfg["model"])


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def load_model_from_checkpoint(path: str | Path, dev: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    ckpt = load_checkpoint(path)
    cfg = ckpt["config"]
    model = build_model(cfg).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg
