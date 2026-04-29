from __future__ import annotations

import torch
from pytest import approx

from brisc_mtl.metrics import binary_detection_accuracy, binary_detection_labels


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
