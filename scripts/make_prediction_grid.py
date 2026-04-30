from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from brisc_mtl.data import INDEX_TO_CLASS, BriscSample, build_samples
from brisc_mtl.preprocessing import image_to_tensor, load_grayscale, mask_to_tensor, resize_pair
from brisc_mtl.runtime import load_model_from_checkpoint
from brisc_mtl.utils import device


def pick_samples(samples: list[BriscSample], samples_per_class: int, seed: int) -> list[BriscSample]:
    rng = random.Random(seed)
    selected: list[BriscSample] = []
    for class_index in sorted(INDEX_TO_CLASS):
        class_samples = [sample for sample in samples if sample.label == class_index]
        rng.shuffle(class_samples)
        selected.extend(class_samples[:samples_per_class])
    return selected


def load_pair(sample: BriscSample, image_size: int) -> tuple[Image.Image, Image.Image]:
    image = load_grayscale(sample.image)
    if sample.mask is None:
        mask = Image.new("L", image.size, 0)
    else:
        mask = load_grayscale(sample.mask)
    return resize_pair(image, mask, image_size)


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred_bool = pred > 0
    target_bool = target > 0
    if not pred_bool.any() and not target_bool.any():
        return 1.0
    intersection = np.logical_and(pred_bool, target_bool).sum()
    denominator = pred_bool.sum() + target_bool.sum()
    return float(2 * intersection / max(denominator, 1))


def overlay_mask(image: np.ndarray, target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    rgb = np.repeat(image[..., None], 3, axis=2)
    target_bool = target > 0
    pred_bool = pred > 0
    rgb[target_bool, 1] = 1.0
    rgb[pred_bool, 0] = 1.0
    rgb[np.logical_and(target_bool, pred_bool)] = [1.0, 1.0, 0.0]
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Create qualitative prediction grids for BRISC checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--out", default="reports/figures/qualitative_predictions.png")
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="cuda",
        help="Inference device. Default is 'cuda' so inference will not silently fall back to CPU.",
    )
    args = parser.parse_args()

    dev = device(args.device)
    model, cfg = load_model_from_checkpoint(args.checkpoint, dev)
    if args.data_root:
        cfg["data_root"] = args.data_root

    samples = pick_samples(build_samples(cfg["data_root"], split=args.split), args.samples_per_class, args.seed)
    rows = len(samples)
    fig, axes = plt.subplots(rows, 4, figsize=(12, 2.7 * rows), squeeze=False)

    for row, sample in enumerate(samples):
        image, gt_mask = load_pair(sample, cfg["data"]["image_size"])
        image_tensor = image_to_tensor(image).unsqueeze(0).to(dev)
        gt_tensor = mask_to_tensor(gt_mask)

        with torch.no_grad():
            out = model(image_tensor)
            probs = torch.softmax(out["class_logits"], dim=1)[0].cpu()
            pred_label = int(probs.argmax().item())
            pred_mask = (torch.sigmoid(out["mask_logits"])[0, 0].cpu() > args.threshold).numpy().astype(np.uint8)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        target_np = gt_tensor[0].numpy().astype(np.uint8)
        sample_dice = dice_score(pred_mask, target_np)
        title = (
            f"true={sample.class_name}\n"
            f"pred={INDEX_TO_CLASS[pred_label]} p={probs[pred_label]:.2f}\n"
            f"dice={sample_dice:.3f}"
        )

        axes[row][0].imshow(image_np, cmap="gray")
        axes[row][0].set_title(title, fontsize=9)
        axes[row][1].imshow(target_np, cmap="gray")
        axes[row][1].set_title("GT mask", fontsize=9)
        axes[row][2].imshow(pred_mask, cmap="gray")
        axes[row][2].set_title("Pred mask", fontsize=9)
        axes[row][3].imshow(overlay_mask(image_np, target_np, pred_mask))
        axes[row][3].set_title("Overlay", fontsize=9)

    for axis in axes.ravel():
        axis.axis("off")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
