from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from brisc_mtl.data import INDEX_TO_CLASS
from brisc_mtl.model import ConvNeXtUNetMultiTask
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

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    dev = device(args.device)
    out_dir = ensure_dir(args.out_dir)

    image_pil = Image.open(args.image).convert("L")
    image_pil = TF.resize(
        image_pil,
        size=[cfg["data"]["image_size"], cfg["data"]["image_size"]],
        interpolation=InterpolationMode.BILINEAR,
    )
    image = TF.to_tensor(image_pil).float().unsqueeze(0).to(dev)

    model = ConvNeXtUNetMultiTask(**cfg["model"]).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

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
