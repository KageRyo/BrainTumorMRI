from __future__ import annotations

from pathlib import Path

from PIL import Image

from brisc_mtl.data import BriscMultiTaskDataset, build_samples, class_counts


def _write_image(path: Path, value: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (value, value, value)).save(path)


def _write_mask(path: Path, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), value).save(path)


def _make_dataset(root: Path) -> Path:
    dataset = root / "brisc2025"
    seg_image = dataset / "segmentation_task" / "train" / "images" / "brisc2025_train_00001_gl_ax_t1.jpg"
    seg_mask = dataset / "segmentation_task" / "train" / "masks" / "brisc2025_train_00001_gl_ax_t1.png"
    no_tumor = dataset / "classification_task" / "train" / "no_tumor" / "brisc2025_train_00002_no_ax_t1.jpg"
    _write_image(seg_image)
    _write_mask(seg_mask)
    _write_image(no_tumor)
    return dataset


def test_build_samples_includes_segmentation_and_no_tumor(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)

    samples = build_samples(dataset, split="train")

    assert len(samples) == 2
    assert class_counts(samples) == {"no_tumor": 1, "glioma": 1, "meningioma": 0, "pituitary": 0}


def test_no_tumor_sample_uses_empty_mask(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    samples = build_samples(dataset, split="train")
    no_tumor_sample = [sample for sample in samples if sample.class_name == "no_tumor"]

    item = BriscMultiTaskDataset(no_tumor_sample, image_size=8, training=False)[0]

    assert tuple(item["image"].shape) == (1, 8, 8)
    assert tuple(item["mask"].shape) == (1, 8, 8)
    assert item["mask"].sum().item() == 0
    assert item["label"].item() == 0
