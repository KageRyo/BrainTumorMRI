from __future__ import annotations

from collections.abc import Sequence

import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def binary_detection_accuracy(class_logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred_has_tumor = class_logits.argmax(dim=1) != 0
    true_has_tumor = labels != 0
    return (pred_has_tumor == true_has_tumor).float().mean().item()


def binary_detection_labels(labels: list[int]) -> list[bool]:
    return [label != 0 for label in labels]


def binary_detection_scores(class_probabilities: Sequence[Sequence[float]]) -> list[float]:
    return [1.0 - float(probs[0]) for probs in class_probabilities]


def binary_detection_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    tumor_scores: Sequence[float] | None = None,
) -> dict[str, float | None]:
    true_has_tumor = [label != 0 for label in y_true]
    pred_has_tumor = [label != 0 for label in y_pred]

    tp = sum(true and pred for true, pred in zip(true_has_tumor, pred_has_tumor, strict=True))
    tn = sum((not true) and (not pred) for true, pred in zip(true_has_tumor, pred_has_tumor, strict=True))
    fp = sum((not true) and pred for true, pred in zip(true_has_tumor, pred_has_tumor, strict=True))
    fn = sum(true and (not pred) for true, pred in zip(true_has_tumor, pred_has_tumor, strict=True))

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_has_tumor,
        pred_has_tumor,
        average="binary",
        zero_division=0,
    )
    has_both_binary_classes = len(set(true_has_tumor)) == 2
    metrics: dict[str, float | None] = {
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy_score(true_has_tumor, pred_has_tumor))
        if has_both_binary_classes
        else None,
    }
    if tumor_scores is not None and has_both_binary_classes:
        metrics["roc_auc"] = float(roc_auc_score(true_has_tumor, tumor_scores))
        metrics["pr_auc"] = float(average_precision_score(true_has_tumor, tumor_scores))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def expected_calibration_error(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    num_bins: int = 15,
) -> float:
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    if not probabilities:
        return 0.0

    probs = torch.tensor(probabilities, dtype=torch.float32)
    true = torch.tensor(labels, dtype=torch.long)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(true)

    ece = torch.zeros((), dtype=torch.float32)
    boundaries = torch.linspace(0.0, 1.0, steps=num_bins + 1)
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        if lower == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)
        if in_bin.any():
            bin_accuracy = accuracies[in_bin].float().mean()
            bin_confidence = confidences[in_bin].mean()
            ece += in_bin.float().mean() * (bin_confidence - bin_accuracy).abs()
    return float(ece.item())


class SegmentationConfusion:
    def __init__(self) -> None:
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        pred = y_pred.bool()
        true = y_true.bool()
        self.true_positive += int((pred & true).sum().item())
        self.false_positive += int((pred & ~true).sum().item())
        self.false_negative += int((~pred & true).sum().item())

    def compute(self) -> dict[str, float | None]:
        tp = self.true_positive
        fp = self.false_positive
        fn = self.false_negative
        return {
            "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }
