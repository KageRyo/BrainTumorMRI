from __future__ import annotations

import torch
from pytest import approx

from brisc_mtl.metrics import (
    SegmentationConfusion,
    binary_detection_accuracy,
    binary_detection_labels,
    binary_detection_metrics,
    binary_detection_scores,
    expected_calibration_error,
)


def test_binary_detection_accuracy_collapses_classes_to_tumor_presence() -> None:
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 2, 0])

    assert binary_detection_accuracy(logits, labels) == approx(2 / 3)


def test_binary_detection_labels() -> None:
    assert binary_detection_labels([0, 1, 2, 3, 0]) == [False, True, True, True, False]


def test_binary_detection_metrics_include_medical_operating_points() -> None:
    y_true = [0, 1, 2, 0]
    y_pred = [0, 1, 0, 3]
    probabilities = [
        [0.90, 0.05, 0.03, 0.02],
        [0.10, 0.80, 0.05, 0.05],
        [0.60, 0.20, 0.10, 0.10],
        [0.20, 0.30, 0.25, 0.25],
    ]

    metrics = binary_detection_metrics(y_true, y_pred, binary_detection_scores(probabilities))

    assert metrics["sensitivity"] == approx(0.5)
    assert metrics["specificity"] == approx(0.5)
    assert metrics["precision"] == approx(0.5)
    assert metrics["recall"] == approx(0.5)
    assert metrics["roc_auc"] == approx(0.75)
    assert metrics["pr_auc"] == approx(0.8333333333)


def test_expected_calibration_error() -> None:
    probabilities = [
        [0.8, 0.2],
        [0.6, 0.4],
        [0.3, 0.7],
        [0.1, 0.9],
    ]
    labels = [0, 1, 0, 1]

    assert expected_calibration_error(probabilities, labels, num_bins=2) == approx(0.25, abs=1e-6)


def test_segmentation_confusion_metrics() -> None:
    confusion = SegmentationConfusion()
    pred = torch.tensor([[[[1, 1], [0, 0]]]], dtype=torch.bool)
    true = torch.tensor([[[[1, 0], [1, 0]]]], dtype=torch.bool)

    confusion.update(pred, true)
    metrics = confusion.compute()

    assert metrics["iou"] == approx(1 / 3)
    assert metrics["precision"] == approx(0.5)
    assert metrics["recall"] == approx(0.5)
