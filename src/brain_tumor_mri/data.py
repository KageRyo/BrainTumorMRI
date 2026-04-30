from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from brain_tumor_mri.preprocessing import augment_pair, image_to_tensor, load_grayscale, mask_to_tensor, resize_pair

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


class BriscMultiTaskDataset(Dataset):
    def __init__(self, samples: Sequence[BriscSample], image_size: int, training: bool) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        image = load_grayscale(sample.image)
        if sample.mask is None:
            mask = Image.new("L", image.size, 0)
        else:
            mask = load_grayscale(sample.mask)
        image, mask = resize_pair(image, mask, self.image_size)
        if self.training:
            image, mask = augment_pair(image, mask)

        image_tensor = image_to_tensor(image)
        mask_tensor = mask_to_tensor(mask)
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
