from __future__ import annotations

import torch


def binary_detection_accuracy(class_logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred_has_tumor = class_logits.argmax(dim=1) != 0
    true_has_tumor = labels != 0
    return (pred_has_tumor == true_has_tumor).float().mean().item()


def binary_detection_labels(labels: list[int]) -> list[bool]:
    return [label != 0 for label in labels]
