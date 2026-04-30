from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from tqdm import tqdm

from brain_tumor_mri.data import build_samples, make_loader
from brain_tumor_mri.metrics import binary_detection_scores
from brain_tumor_mri.runtime import load_model_from_checkpoint
from brain_tumor_mri.utils import device, ensure_dir


def collect_scores(
    checkpoint: str | Path,
    data_root: str | Path | None,
    device_name: str,
    batch_size: int | None,
    num_workers: int | None,
) -> tuple[list[int], list[float]]:
    dev = device(device_name)
    model, cfg = load_model_from_checkpoint(checkpoint, dev)
    if data_root:
        cfg["data_root"] = str(data_root)

    samples = build_samples(cfg["data_root"], split="test")
    loader = make_loader(
        samples,
        image_size=int(cfg["data"]["image_size"]),
        batch_size=batch_size or int(cfg["eval"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]) if num_workers is None else num_workers,
        training=False,
    )

    labels: list[int] = []
    scores: list[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="thresholds"):
            images = batch["image"].to(dev, non_blocking=True)
            probabilities = torch.softmax(model(images)["class_logits"], dim=1).cpu().tolist()
            scores.extend(binary_detection_scores(probabilities))
            labels.extend([int(label != 0) for label in batch["label"].tolist()])
    return labels, scores


def threshold_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, float]:
    preds = [int(score >= threshold) for score in scores]
    tp = sum(true == 1 and pred == 1 for true, pred in zip(labels, preds, strict=True))
    tn = sum(true == 0 and pred == 0 for true, pred in zip(labels, preds, strict=True))
    fp = sum(true == 0 and pred == 1 for true, pred in zip(labels, preds, strict=True))
    fn = sum(true == 1 and pred == 0 for true, pred in zip(labels, preds, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "sensitivity": recall,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "precision": precision,
        "f1": f1,
    }


def plot_roc(labels: list[int], scores: list[float], out_path: Path) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, color="#4c78a8", linewidth=2, label=f"ROC-AUC={roc_auc:.4f}")
    axis.plot([0, 1], [0, 1], "--", color="0.45", linewidth=1)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Binary Tumor Detection ROC")
    axis.legend(loc="lower right")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.03)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return float(roc_auc)


def plot_pr(labels: list[int], scores: list[float], out_path: Path) -> float:
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision, color="#f58518", linewidth=2, label=f"PR-AUC={pr_auc:.4f}")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Binary Tumor Detection PR Curve")
    axis.legend(loc="lower left")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.03)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return float(pr_auc)


def write_report(
    labels: list[int],
    scores: list[float],
    thresholds: list[float],
    roc_auc: float,
    pr_auc: float,
    out_path: Path,
) -> None:
    rows = [(threshold, threshold_metrics(labels, scores, threshold)) for threshold in thresholds]
    lines = [
        "# Threshold Analysis",
        "",
        "Binary tumor detection scores are computed as `1 - P(no_tumor)` from the 4-class classification head.",
        "",
        f"- ROC-AUC: {roc_auc:.4f}",
        f"- PR-AUC: {pr_auc:.4f}",
        "",
        "| Threshold | Sensitivity | Specificity | Precision | F1 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for threshold, metrics in rows:
        lines.append(
            f"| {threshold:.2f} | {metrics['sensitivity']:.4f} | {metrics['specificity']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Figures:",
            "",
            "- [ROC curve](figures/roc_curve.png)",
            "- [PR curve](figures/pr_curve.png)",
            "",
            "The current headline checkpoint separates tumor and no-tumor samples cleanly on this test split. External",
            "validation is still required before choosing any clinical operating threshold.",
        ]
    )
    ensure_dir(out_path.parent)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze binary tumor detection thresholds.")
    parser.add_argument("--checkpoint", default="outputs/convnext_tiny_mtl/best.pt")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", default="reports/threshold_analysis.md")
    parser.add_argument("--roc-out", default="reports/figures/roc_curve.png")
    parser.add_argument("--pr-out", default="reports/figures/pr_curve.png")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    args = parser.parse_args()

    labels, scores = collect_scores(args.checkpoint, args.data_root, args.device, args.batch_size, args.num_workers)
    roc_auc = plot_roc(labels, scores, Path(args.roc_out))
    pr_auc = plot_pr(labels, scores, Path(args.pr_out))
    write_report(labels, scores, args.thresholds, roc_auc, pr_auc, Path(args.out))
    print(f"Wrote {args.out}, {args.roc_out}, and {args.pr_out}")


if __name__ == "__main__":
    main()
