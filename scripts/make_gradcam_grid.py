from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch

from brain_tumor_mri.data import CLASS_TO_INDEX, INDEX_TO_CLASS, BriscMultiTaskDataset, build_samples
from brain_tumor_mri.explain import classification_gradcam
from brain_tumor_mri.runtime import load_model_from_checkpoint
from brain_tumor_mri.utils import device, ensure_dir


@dataclass
class GradCamExample:
    image: np.ndarray
    heatmap: np.ndarray
    pred_mask: np.ndarray
    overlay: np.ndarray
    true_label: int
    pred_label: int
    confidence: float


def heatmap_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    cmap = plt.get_cmap("magma")(heatmap)[..., :3]
    base = np.repeat(image[..., None], 3, axis=2)
    return np.clip((1.0 - alpha) * base + alpha * cmap, 0.0, 1.0)


def mask_overlay(image: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    rgb = np.repeat(image[..., None], 3, axis=2)
    rgb[pred_mask.astype(bool), 0] = 1.0
    return np.clip(rgb, 0.0, 1.0)


def select_indices(dataset: BriscMultiTaskDataset, examples_per_class: int) -> list[int]:
    selected: list[int] = []
    counts = {label: 0 for label in CLASS_TO_INDEX.values()}
    for index, sample in enumerate(dataset.samples):
        if counts[sample.label] < examples_per_class:
            selected.append(index)
            counts[sample.label] += 1
        if all(count >= examples_per_class for count in counts.values()):
            break
    return selected


def build_examples(
    checkpoint: str | Path,
    data_root: str | Path | None,
    device_name: str,
    examples_per_class: int,
    threshold: float,
) -> list[GradCamExample]:
    dev = device(device_name)
    model, cfg = load_model_from_checkpoint(checkpoint, dev)
    if data_root:
        cfg["data_root"] = str(data_root)

    samples = build_samples(cfg["data_root"], split="test")
    dataset = BriscMultiTaskDataset(samples, image_size=int(cfg["data"]["image_size"]), training=False)
    examples: list[GradCamExample] = []
    for index in select_indices(dataset, examples_per_class):
        item = dataset[index]
        image = item["image"].unsqueeze(0).to(dev)
        heatmap = classification_gradcam(model, image)
        with torch.no_grad():
            out = model(image)
            probabilities = torch.softmax(out["class_logits"], dim=1)[0].cpu()
            pred_label = int(probabilities.argmax().item())
            pred_mask = (torch.sigmoid(out["mask_logits"])[0, 0].cpu() > threshold).numpy()
        image_np = item["image"][0].numpy()
        heatmap_np = heatmap.numpy()
        examples.append(
            GradCamExample(
                image=image_np,
                heatmap=heatmap_np,
                pred_mask=pred_mask,
                overlay=heatmap_overlay(image_np, heatmap_np),
                true_label=int(item["label"].item()),
                pred_label=pred_label,
                confidence=float(probabilities[pred_label].item()),
            )
        )
    return examples


def draw_grid(examples: list[GradCamExample], out_path: Path) -> None:
    columns = ["Original MRI", "Grad-CAM", "Predicted mask", "Overlay"]
    fig, axes = plt.subplots(len(examples), len(columns), figsize=(12, 3.0 * len(examples)), squeeze=False)
    for row, example in enumerate(examples):
        axes[row][0].imshow(example.image, cmap="gray")
        axes[row][0].set_title(
            f"GT: {INDEX_TO_CLASS[example.true_label]}\n"
            f"Pred: {INDEX_TO_CLASS[example.pred_label]}\n"
            f"conf={example.confidence:.3f}"
        )
        axes[row][1].imshow(example.heatmap, cmap="magma", vmin=0, vmax=1)
        axes[row][1].set_title("Grad-CAM")
        axes[row][2].imshow(example.pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[row][2].set_title("Predicted mask")
        axes[row][3].imshow(mask_overlay(example.image, example.pred_mask))
        axes[row][3].imshow(example.overlay, alpha=0.55)
        axes[row][3].set_title("Mask + Grad-CAM")
        for axis in axes[row]:
            axis.axis("off")

    fig.suptitle("BrainTumorMRI Grad-CAM Examples", y=1.0)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Grad-CAM example grid for classification explainability.")
    parser.add_argument("--checkpoint", default="outputs/convnext_tiny_mtl/best.pt")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", default="reports/figures/gradcam_examples.png")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--examples-per-class", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    examples = build_examples(
        args.checkpoint,
        args.data_root,
        args.device,
        args.examples_per_class,
        args.threshold,
    )
    draw_grid(examples, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
