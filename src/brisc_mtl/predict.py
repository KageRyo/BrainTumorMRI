from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch

from brisc_mtl.data import INDEX_TO_CLASS
from brisc_mtl.preprocessing import load_image_tensor
from brisc_mtl.runtime import load_model_from_checkpoint
from brisc_mtl.utils import device, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-image BRISC multitask inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", default="outputs/predictions")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="cuda",
        help="Inference device. Default is 'cuda' so inference will not silently fall back to CPU.",
    )
    args = parser.parse_args()

    dev = device(args.device)
    model, cfg = load_model_from_checkpoint(args.checkpoint, dev)
    out_dir = ensure_dir(args.out_dir)

    image = load_image_tensor(args.image, cfg["data"]["image_size"]).unsqueeze(0).to(dev)

    with torch.no_grad():
        out = model(image)
        probs = torch.softmax(out["class_logits"], dim=1)[0].cpu()
        label = int(probs.argmax().item())
        mask = (torch.sigmoid(out["mask_logits"])[0, 0].cpu() > args.threshold).float()

    print(f"class={INDEX_TO_CLASS[label]} probability={probs[label]:.4f} has_tumor={label != 0}")
    for idx, prob in enumerate(probs.tolist()):
        print(f"{INDEX_TO_CLASS[idx]}: {prob:.4f}")

    plt.imsave(out_dir / f"{Path(args.image).stem}_mask.png", mask.numpy(), cmap="gray")


if __name__ == "__main__":
    main()
