from __future__ import annotations

import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

CLASS_TO_INDEX = {
    "no_tumor": 0,
    "glioma": 1,
    "meningioma": 2,
    "pituitary": 3,
}
INDEX_TO_CLASS = {idx: name for name, idx in CLASS_TO_INDEX.items()}
SHORT_TO_CLASS = {"nt": "no_tumor", "no": "no_tumor", "gl": "glioma", "me": "meningioma", "pi": "pituitary"}


@dataclass(frozen=True)
class BriscSample:
    image: Path
    mask: Path | None
    label: int
    class_name: str
    split: str


def find_dataset_root(root: str | Path) -> Path:
    root = Path(root)
    if (root / "classification_task").exists() and (root / "segmentation_task").exists():
        return root
    candidates = [p for p in root.rglob("classification_task") if (p.parent / "segmentation_task").exists()]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find BRISC dataset under {root}. Expected classification_task/ and segmentation_task/."
        )
    return candidates[0].parent


def _class_from_filename(path: Path) -> str:
    match = re.search(r"_(gl|me|pi|nt|no)_", path.name)
    if match:
        return SHORT_TO_CLASS[match.group(1)]
    lowered = path.name.lower()
    for class_name in CLASS_TO_INDEX:
        if class_name in lowered:
            return class_name
    raise ValueError(f"Could not infer tumor class from filename: {path.name}")


def build_samples(data_root: str | Path, split: str) -> list[BriscSample]:
    root = find_dataset_root(data_root)
    image_dir = root / "segmentation_task" / split / "images"
    mask_dir = root / "segmentation_task" / split / "masks"
    no_tumor_dir = root / "classification_task" / split / "no_tumor"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Missing segmentation images/masks for split '{split}' under {root}")
    if not no_tumor_dir.exists():
        raise FileNotFoundError(f"Missing no_tumor classification images for split '{split}' under {root}")

    samples: list[BriscSample] = []
    for image in sorted(image_dir.glob("*.jpg")):
        class_name = _class_from_filename(image)
        mask = mask_dir / f"{image.stem}.png"
        if not mask.exists():
            raise FileNotFoundError(f"Missing mask for {image}: expected {mask}")
        samples.append(
            BriscSample(
                image=image,
                mask=mask,
                label=CLASS_TO_INDEX[class_name],
                class_name=class_name,
                split=split,
            )
        )
    for image in sorted(no_tumor_dir.glob("*.jpg")):
        samples.append(
            BriscSample(
                image=image,
                mask=None,
                label=CLASS_TO_INDEX["no_tumor"],
                class_name="no_tumor",
                split=split,
            )
        )
    if not samples:
        raise FileNotFoundError(f"No BRISC JPG images found in {image_dir}")
    return samples


def split_train_val(
    samples: Sequence[BriscSample],
    val_fraction: float,
    seed: int,
) -> tuple[list[BriscSample], list[BriscSample]]:
    labels = [sample.label for sample in samples]
    train, val = train_test_split(
        list(samples),
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )
    return train, val


def _load_grayscale(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def _resize_pair(image: Image.Image, mask: Image.Image, image_size: int) -> tuple[Image.Image, Image.Image]:
    size = [image_size, image_size]
    image = TF.resize(image, size=size, interpolation=InterpolationMode.BILINEAR)
    mask = TF.resize(mask, size=size, interpolation=InterpolationMode.NEAREST)
    return image, mask


def _augment_pair(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
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


class BriscMultiTaskDataset(Dataset):
    def __init__(self, samples: Sequence[BriscSample], image_size: int, training: bool) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        image = _load_grayscale(sample.image)
        if sample.mask is None:
            mask = Image.new("L", image.size, 0)
        else:
            mask = _load_grayscale(sample.mask)
        image, mask = _resize_pair(image, mask, self.image_size)
        if self.training:
            image, mask = _augment_pair(image, mask)

        image_tensor = TF.to_tensor(image).float()
        mask_tensor = (TF.to_tensor(mask) > 0).float()
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(sample.label, dtype=torch.long),
        }


def make_loader(
    samples: Sequence[BriscSample],
    image_size: int,
    batch_size: int,
    num_workers: int,
    training: bool,
) -> DataLoader:
    dataset = BriscMultiTaskDataset(samples, image_size=image_size, training=training)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
    )


def class_counts(samples: Iterable[BriscSample]) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_TO_INDEX}
    for sample in samples:
        counts[sample.class_name] += 1
    return counts
