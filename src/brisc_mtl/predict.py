from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Resized, ScaleIntensityd, ToTensord

from brisc_mtl.data import INDEX_TO_CLASS
from brisc_mtl.model import ConvNeXtUNetMultiTask
from brisc_mtl.utils import device, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-image BRISC multitask inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", default="outputs/predictions")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    dev = device()
    out_dir = ensure_dir(args.out_dir)

    transform = Compose(
        [
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=(cfg["data"]["image_size"], cfg["data"]["image_size"]), mode="bilinear"),
            ToTensord(keys=["image"]),
        ]
    )
    image = transform({"image": Path(args.image)})["image"].float().unsqueeze(0).to(dev)

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
