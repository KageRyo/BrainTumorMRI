from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from monai.metrics import DiceMetric
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from brisc_mtl.data import INDEX_TO_CLASS, build_samples, make_loader
from brisc_mtl.metrics import binary_detection_labels
from brisc_mtl.runtime import load_model_from_checkpoint
from brisc_mtl.utils import device, ensure_dir, save_json


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
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="test"):
            images = batch["image"].to(dev, non_blocking=True)
            masks = batch["mask"].to(dev, non_blocking=True)
            labels = batch["label"].to(dev, non_blocking=True)
            out = model(images)
            preds = out["class_logits"].argmax(dim=1)
            dice_metric(y_pred=torch.sigmoid(out["mask_logits"]) > 0.5, y=masks > 0.5)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    names = [INDEX_TO_CLASS[i] for i in range(len(INDEX_TO_CLASS))]
    report = classification_report(y_true, y_pred, target_names=names, output_dict=True, zero_division=0)
    det_true = binary_detection_labels(y_true)
    det_pred = binary_detection_labels(y_pred)
    metrics = {
        "classification_accuracy": accuracy_score(y_true, y_pred),
        "binary_detection_accuracy": accuracy_score(det_true, det_pred),
        "dice": dice_metric.aggregate().item(),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    save_json(metrics, out_dir / "metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
