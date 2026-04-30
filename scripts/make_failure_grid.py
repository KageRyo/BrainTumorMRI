from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from brain_tumor_mri.data import INDEX_TO_CLASS, BriscMultiTaskDataset, build_samples
from brain_tumor_mri.runtime import load_model_from_checkpoint
from brain_tumor_mri.utils import device, ensure_dir


@dataclass
class Case:
    image: np.ndarray
    true_mask: np.ndarray
    pred_mask: np.ndarray
    true_label: int
    pred_label: int
    confidence: float
    dice: float


def sample_dice(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> float:
    pred = pred_mask.bool()
    true = true_mask.bool()
    intersection = (pred & true).sum().item()
    denominator = pred.sum().item() + true.sum().item()
    return float(2.0 * intersection / denominator) if denominator else 1.0


def overlay(image: np.ndarray, true_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    rgb = np.repeat(image[..., None], 3, axis=2)
    rgb = np.clip(rgb, 0.0, 1.0)
    true = true_mask.astype(bool)
    pred = pred_mask.astype(bool)
    rgb[true, 1] = 1.0
    rgb[pred, 0] = 1.0
    rgb[true & pred] = np.array([1.0, 1.0, 0.0])
    return rgb


def collect_cases(
    checkpoint: str | Path,
    data_root: str | Path | None,
    device_name: str,
    max_cases: int,
    threshold: float,
) -> tuple[list[Case], list[Case]]:
    dev = device(device_name)
    model, cfg = load_model_from_checkpoint(checkpoint, dev)
    if data_root:
        cfg["data_root"] = str(data_root)

    samples = build_samples(cfg["data_root"], split="test")
    dataset = BriscMultiTaskDataset(samples, image_size=int(cfg["data"]["image_size"]), training=False)
    classification_failures: list[Case] = []
    low_dice_cases: list[Case] = []

    with torch.no_grad():
        for index in tqdm(range(len(dataset)), desc="failure cases"):
            item = dataset[index]
            image = item["image"].unsqueeze(0).to(dev)
            true_mask = item["mask"]
            true_label = int(item["label"].item())
            out = model(image)
            probabilities = torch.softmax(out["class_logits"], dim=1)[0].cpu()
            pred_label = int(probabilities.argmax().item())
            pred_mask = (torch.sigmoid(out["mask_logits"])[0, 0].cpu() > threshold).float()
            dice = sample_dice(pred_mask, true_mask[0])
            case = Case(
                image=item["image"][0].numpy(),
                true_mask=true_mask[0].numpy(),
                pred_mask=pred_mask.numpy(),
                true_label=true_label,
                pred_label=pred_label,
                confidence=float(probabilities[pred_label].item()),
                dice=dice,
            )
            if pred_label != true_label:
                classification_failures.append(case)
            if true_label != 0:
                low_dice_cases.append(case)

    classification_failures.sort(key=lambda case: case.confidence, reverse=True)
    low_dice_cases.sort(key=lambda case: case.dice)
    return classification_failures[:max_cases], low_dice_cases[:max_cases]


def draw_grid(cases: list[Case], out_path: Path, title: str) -> None:
    ensure_dir(out_path.parent)
    if not cases:
        fig, axis = plt.subplots(figsize=(8, 2.5))
        axis.axis("off")
        axis.text(0.5, 0.5, "No cases found.", ha="center", va="center", fontsize=14)
        fig.suptitle(title)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    columns = ["Image", "GT mask", "Pred mask", "Overlay"]
    fig, axes = plt.subplots(len(cases), len(columns), figsize=(12, 3.0 * len(cases)), squeeze=False)
    for row, case in enumerate(cases):
        axes[row][0].imshow(case.image, cmap="gray")
        axes[row][0].set_title(
            f"GT: {INDEX_TO_CLASS[case.true_label]}\n"
            f"Pred: {INDEX_TO_CLASS[case.pred_label]}\n"
            f"conf={case.confidence:.3f}"
        )
        axes[row][1].imshow(case.true_mask, cmap="gray", vmin=0, vmax=1)
        axes[row][1].set_title("GT mask")
        axes[row][2].imshow(case.pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[row][2].set_title("Pred mask")
        axes[row][3].imshow(overlay(case.image, case.true_mask, case.pred_mask))
        axes[row][3].set_title(f"Dice={case.dice:.3f}")
        for axis in axes[row]:
            axis.axis("off")

    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create classification failure and low-Dice visualization grids.")
    parser.add_argument("--checkpoint", default="outputs/convnext_tiny_mtl/best.pt")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--classification-out", default="reports/figures/classification_failures.png")
    parser.add_argument("--low-dice-out", default="reports/figures/low_dice_cases.png")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--max-cases", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    classification_failures, low_dice_cases = collect_cases(
        args.checkpoint,
        args.data_root,
        args.device,
        args.max_cases,
        args.threshold,
    )
    draw_grid(classification_failures, Path(args.classification_out), "Classification Failure Cases")
    draw_grid(low_dice_cases, Path(args.low_dice_out), "Lowest Dice Segmentation Cases")
    print(f"Wrote {args.classification_out} and {args.low_dice_out}")


if __name__ == "__main__":
    main()
