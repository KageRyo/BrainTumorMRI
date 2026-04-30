from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn

from brain_tumor_mri.model import ConvNeXtUNetMultiTask


def build_model(cfg: dict[str, Any]) -> ConvNeXtUNetMultiTask:
    return ConvNeXtUNetMultiTask(**cfg["model"])


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model_from_checkpoint(path: str | Path, dev: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    ckpt = load_checkpoint(path)
    cfg = ckpt["config"]
    model_cfg = deepcopy(cfg)
    model_cfg["model"]["pretrained"] = False
    model = build_model(model_cfg).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg
