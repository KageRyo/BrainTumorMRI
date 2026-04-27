from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    RandFlipd,
    RandRotated,
    RandZoomd,
    Resized,
    ScaleIntensityd,
    ToTensord,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

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
    mask: Path
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


def build_samples(data_root: str | Path, split: str) -> List[BriscSample]:
    root = find_dataset_root(data_root)
    image_dir = root / "segmentation_task" / split / "images"
    mask_dir = root / "segmentation_task" / split / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Missing segmentation images/masks for split '{split}' under {root}")

    samples: List[BriscSample] = []
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
    if not samples:
        raise FileNotFoundError(f"No BRISC JPG images found in {image_dir}")
    return samples


def split_train_val(samples: Sequence[BriscSample], val_fraction: float, seed: int) -> tuple[List[BriscSample], List[BriscSample]]:
    labels = [sample.label for sample in samples]
    train, val = train_test_split(
        list(samples),
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )
    return train, val


def make_transforms(image_size: int, training: bool) -> Compose:
    transforms = [
        LoadImaged(keys=["image", "mask"], image_only=True),
        EnsureChannelFirstd(keys=["image", "mask"], channel_dim="no_channel"),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image", "mask"], spatial_size=(image_size, image_size), mode=("bilinear", "nearest")),
    ]
    if training:
        transforms.extend(
            [
                RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
                RandRotated(keys=["image", "mask"], prob=0.25, range_x=0.2, mode=("bilinear", "nearest")),
                RandZoomd(keys=["image", "mask"], prob=0.25, min_zoom=0.9, max_zoom=1.1, mode=("bilinear", "nearest")),
            ]
        )
    transforms.append(ToTensord(keys=["image", "mask"]))
    return Compose(transforms)


class BriscMultiTaskDataset(Dataset):
    def __init__(self, samples: Sequence[BriscSample], transform: Compose) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        item = self.transform({"image": sample.image, "mask": sample.mask})
        mask = (item["mask"] > 0).float()
        return {
            "image": item["image"].float(),
            "mask": mask,
            "label": torch.tensor(sample.label, dtype=torch.long),
        }


def make_loader(
    samples: Sequence[BriscSample],
    image_size: int,
    batch_size: int,
    num_workers: int,
    training: bool,
) -> DataLoader:
    dataset = BriscMultiTaskDataset(samples, make_transforms(image_size=image_size, training=training))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
    )


def class_counts(samples: Iterable[BriscSample]) -> Dict[str, int]:
    counts = {name: 0 for name in CLASS_TO_INDEX}
    for sample in samples:
        counts[sample.class_name] += 1
    return counts
