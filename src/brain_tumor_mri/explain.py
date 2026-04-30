from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def classification_gradcam(model: nn.Module, image: torch.Tensor, class_index: int | None = None) -> torch.Tensor:
    """Compute Grad-CAM for the final ConvNeXt encoder feature map.

    Args:
        model: BrainTumorMRI multitask model in eval mode.
        image: Input tensor shaped ``1 x C x H x W``.
        class_index: Optional target class. Defaults to the predicted class.

    Returns:
        A normalized heatmap tensor shaped ``H x W`` on CPU.
    """
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Grad-CAM expects a single image batch shaped 1 x C x H x W")

    activation: torch.Tensor | None = None

    def capture_activation(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: list[torch.Tensor]) -> None:
        nonlocal activation
        activation = output[-1]
        activation.retain_grad()

    handle = model.encoder.register_forward_hook(capture_activation)
    try:
        model.zero_grad(set_to_none=True)
        out = model(image)
        logits = out["class_logits"]
        target = int(logits.argmax(dim=1).item()) if class_index is None else class_index
        logits[0, target].backward()
    finally:
        handle.remove()

    if activation is None or activation.grad is None:
        raise RuntimeError("Could not capture encoder activations and gradients for Grad-CAM")

    gradients = activation.grad
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu()
    cam_min = cam.min()
    cam_max = cam.max()
    if float(cam_max - cam_min) > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)
    return cam
