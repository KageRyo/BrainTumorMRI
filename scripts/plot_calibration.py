from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from brain_tumor_mri.data import build_samples, make_loader
from brain_tumor_mri.metrics import expected_calibration_error
from brain_tumor_mri.runtime import load_model_from_checkpoint
from brain_tumor_mri.utils import device, ensure_dir


def collect_probabilities(
    checkpoint: str | Path,
    data_root: str | Path | None,
    device_name: str,
    batch_size: int | None,
    num_workers: int | None,
) -> tuple[list[list[float]], list[int]]:
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
    probabilities: list[list[float]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="calibration"):
            images = batch["image"].to(dev, non_blocking=True)
            out = model(images)
            probs = torch.softmax(out["class_logits"], dim=1)
            probabilities.extend(probs.cpu().tolist())
            labels.extend(batch["label"].tolist())
    return probabilities, labels


def plot_calibration(probabilities: list[list[float]], labels: list[int], out_path: Path, num_bins: int) -> None:
    probs = torch.tensor(probabilities, dtype=torch.float32)
    true = torch.tensor(labels, dtype=torch.long)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(true).float()
    ece = expected_calibration_error(probabilities, labels, num_bins=num_bins)

    boundaries = torch.linspace(0.0, 1.0, steps=num_bins + 1)
    bin_centers: list[float] = []
    bin_confidences: list[float] = []
    bin_accuracies: list[float] = []
    bin_counts: list[int] = []
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        if lower == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)
        bin_centers.append(float((lower + upper) / 2))
        bin_counts.append(int(in_bin.sum().item()))
        if in_bin.any():
            bin_confidences.append(float(confidences[in_bin].mean().item()))
            bin_accuracies.append(float(accuracies[in_bin].mean().item()))
        else:
            bin_confidences.append(float((lower + upper) / 2))
            bin_accuracies.append(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot([0, 1], [0, 1], "--", color="0.45", linewidth=1.2, label="Perfect calibration")
    axes[0].bar(
        bin_confidences,
        bin_accuracies,
        width=1.0 / num_bins * 0.85,
        color="#4c78a8",
        edgecolor="white",
        label="Observed accuracy",
    )
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.03)
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"Reliability Diagram (ECE={ece:.4f})")
    axes[0].legend(loc="upper left")

    axes[1].bar(bin_centers, bin_counts, width=1.0 / num_bins * 0.85, color="#f58518", edgecolor="white")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("Sample count")
    axes[1].set_title("Confidence Histogram")

    fig.suptitle("BrainTumorMRI Classification Calibration", y=1.03)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a classification calibration curve for a checkpoint.")
    parser.add_argument("--checkpoint", default="outputs/convnext_tiny_mtl/best.pt")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", default="reports/figures/calibration_curve.png")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-bins", type=int, default=15)
    args = parser.parse_args()

    probabilities, labels = collect_probabilities(
        args.checkpoint,
        args.data_root,
        args.device,
        args.batch_size,
        args.num_workers,
    )
    plot_calibration(probabilities, labels, Path(args.out), args.num_bins)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
