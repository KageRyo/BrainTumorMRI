from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def load_grayscale(path: str | Path) -> Image.Image:
    return Image.open(path).convert("L")


def resize_pair(image: Image.Image, mask: Image.Image, image_size: int) -> tuple[Image.Image, Image.Image]:
    size = [image_size, image_size]
    image = TF.resize(image, size=size, interpolation=InterpolationMode.BILINEAR)
    mask = TF.resize(mask, size=size, interpolation=InterpolationMode.NEAREST)
    return image, mask


def augment_pair(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    if random.random() < 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    if random.random() < 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)
    if random.random() < 0.25:
        angle = random.uniform(-12.0, 12.0)
        scale = random.uniform(0.9, 1.1)
        image = TF.affine(
            image,
            angle=angle,
            translate=[0, 0],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        mask = TF.affine(
            mask,
            angle=angle,
            translate=[0, 0],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
    return image, mask


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    return TF.to_tensor(image).float()


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    return (TF.to_tensor(mask) > 0).float()


def load_image_tensor(path: str | Path, image_size: int) -> torch.Tensor:
    image = load_grayscale(path)
    image = TF.resize(image, size=[image_size, image_size], interpolation=InterpolationMode.BILINEAR)
    return image_to_tensor(image)
