from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch
from monai.metrics import DiceMetric
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from brain_tumor_mri.data import INDEX_TO_CLASS, build_samples, make_loader
from brain_tumor_mri.metrics import (
    SegmentationConfusion,
    binary_detection_labels,
    binary_detection_metrics,
    binary_detection_scores,
    expected_calibration_error,
)
from brain_tumor_mri.runtime import load_model_from_checkpoint
from brain_tumor_mri.utils import device, ensure_dir, save_json


def plot_confusion_matrix(matrix: list[list[int]], names: list[str], out_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(names)), labels=names, rotation=35, ha="right")
    axis.set_yticks(range(len(names)), labels=names)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("BRISC Test Confusion Matrix")

    max_value = max(max(row) for row in matrix)
    threshold = max_value / 2
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            color = "white" if value > threshold else "black"
            axis.text(col_index, row_index, str(value), ha="center", va="center", color=color)

    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a BRISC multitask checkpoint on the test split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="cuda",
        help="Evaluation device. Default is 'cuda' so evaluation will not silently fall back to CPU.",
    )
    args = parser.parse_args()

    dev = device(args.device)
    model, cfg = load_model_from_checkpoint(args.checkpoint, dev)
    if args.data_root:
        cfg["data_root"] = args.data_root
    out_dir = ensure_dir(args.out or Path(cfg["output_dir"]) / "test_eval")

    samples = build_samples(cfg["data_root"], split="test")
    loader = make_loader(
        samples,
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["eval"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        training=False,
    )

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    seg_confusion = SegmentationConfusion()
    y_true, y_pred = [], []
    y_prob = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="test"):
            images = batch["image"].to(dev, non_blocking=True)
            masks = batch["mask"].to(dev, non_blocking=True)
            labels = batch["label"].to(dev, non_blocking=True)
            out = model(images)
            probabilities = torch.softmax(out["class_logits"], dim=1)
            preds = probabilities.argmax(dim=1)
            pred_masks = torch.sigmoid(out["mask_logits"]) > 0.5
            true_masks = masks > 0.5
            dice_metric(y_pred=pred_masks, y=true_masks)
            seg_confusion.update(pred_masks, true_masks)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probabilities.cpu().tolist())

    names = [INDEX_TO_CLASS[i] for i in range(len(INDEX_TO_CLASS))]
    report = classification_report(y_true, y_pred, target_names=names, output_dict=True, zero_division=0)
    det_true = binary_detection_labels(y_true)
    det_pred = binary_detection_labels(y_pred)
    det_scores = binary_detection_scores(y_prob)
    cm = confusion_matrix(y_true, y_pred).tolist()
    confusion_matrix_path = out_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, names, confusion_matrix_path)
    metrics = {
        "classification_accuracy": accuracy_score(y_true, y_pred),
        "classification_macro_f1": report["macro avg"]["f1-score"],
        "classification_weighted_f1": report["weighted avg"]["f1-score"],
        "classification_ece": expected_calibration_error(y_prob, y_true),
        "binary_detection_accuracy": accuracy_score(det_true, det_pred),
        "binary_detection": binary_detection_metrics(y_true, y_pred, det_scores),
        "dice": dice_metric.aggregate().item(),
        "segmentation": seg_confusion.compute(),
        "classification_report": report,
        "confusion_matrix": cm,
        "confusion_matrix_png": str(confusion_matrix_path),
    }
    save_json(metrics, out_dir / "metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
