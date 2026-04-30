from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from brisc_mtl.data import INDEX_TO_CLASS
from brisc_mtl.preprocessing import image_to_tensor
from brisc_mtl.runtime import load_model_from_checkpoint
from brisc_mtl.utils import device

DISCLAIMER = "Research demo only. Not for clinical diagnosis."


def prepare_image(image: Image.Image, image_size: int) -> tuple[torch.Tensor, Image.Image]:
    grayscale = image.convert("L")
    resized = TF.resize(grayscale, size=[image_size, image_size], interpolation=InterpolationMode.BILINEAR)
    return image_to_tensor(resized).unsqueeze(0), resized


def mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def overlay_mask(image: Image.Image, mask: np.ndarray, alpha: float = 0.42) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    color = np.zeros_like(base)
    color[..., 0] = 255.0
    blended = np.where(mask[..., None], (1.0 - alpha) * base + alpha * color, base)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def build_predict_fn(checkpoint: str | Path, device_name: str, threshold: float):
    dev = device(device_name)
    model, cfg = load_model_from_checkpoint(checkpoint, dev)
    image_size = int(cfg["data"]["image_size"])

    def predict(image: Image.Image | None):
        if image is None:
            return "Upload an MRI image.", {}, None, None, DISCLAIMER

        image_tensor, display_image = prepare_image(image, image_size)
        image_tensor = image_tensor.to(dev)
        with torch.no_grad():
            out = model(image_tensor)
            probs = torch.softmax(out["class_logits"], dim=1)[0].cpu().numpy()
            mask = torch.sigmoid(out["mask_logits"])[0, 0].cpu().numpy() > threshold

        label_idx = int(probs.argmax())
        label = INDEX_TO_CLASS[label_idx]
        probabilities = {INDEX_TO_CLASS[idx]: float(prob) for idx, prob in enumerate(probs)}
        summary = f"Prediction: {label} | confidence={probabilities[label]:.4f} | has_tumor={label_idx != 0}"
        return summary, probabilities, mask_to_image(mask), overlay_mask(display_image, mask), DISCLAIMER

    return predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a Gradio BRISC MRI multitask inference demo.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="cuda",
        help="Inference device. Default is 'cuda' so inference will not silently fall back to CPU.",
    )
    args = parser.parse_args()

    predict = build_predict_fn(args.checkpoint, args.device, args.threshold)
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", image_mode="L", label="MRI image"),
        outputs=[
            gr.Textbox(label="Result"),
            gr.Label(label="Class probabilities", num_top_classes=4),
            gr.Image(type="pil", image_mode="L", label="Predicted mask"),
            gr.Image(type="pil", image_mode="RGB", label="Mask overlay"),
            gr.Textbox(label="Warning"),
        ],
        title="BrainTumorMRI Demo",
        description=DISCLAIMER,
    )
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
