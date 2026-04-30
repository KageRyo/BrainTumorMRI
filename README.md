# BRISC 2025 Brain Tumor Multitask Detection

This project trains a PyTorch + MONAI multitask model on the BRISC 2025 brain MRI dataset:

- Classification: 4 classes, `no_tumor`, `glioma`, `meningioma`, `pituitary`
- Detection: derived binary output from the classification head, tumor vs no tumor
- Segmentation: binary tumor mask from the segmentation decoder

Dataset source: <https://www.kaggle.com/datasets/briscdataset/brisc2025>

## Model Choice

Recommended model: **Shared ConvNeXt-Base encoder + classification head + U-Net decoder**.

Why this is the best default for this hardware and dataset:

- ConvNeXt-Base is a strong modern CNN backbone for 2D MRI slices and should run comfortably on an RTX 4090.
- The shared encoder learns tumor morphology once, then serves both class prediction and mask prediction.
- U-Net style skip connections preserve spatial detail for masks, which is important for medical segmentation boundaries.
- It is easier to train and debug than Swin-UNETR while still being strong enough for BRISC scale.

Expected target range after tuning:

- 4-class classification accuracy: about 97.5-98.8%
- Binary tumor detection accuracy: usually at or above classification accuracy
- Segmentation Dice: about 0.89-0.93

These are realistic goals, not guaranteed results. Final numbers depend on augmentation, image size, seed, train/validation split, and checkpoint selection.

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate dl-class-ryo
```

If you already created the environment and changed dependencies:

```bash
conda env update -f environment.yml --prune
conda activate dl-class-ryo
```

The project uses `pyproject.toml` for Python package metadata and dependencies. PyTorch and CUDA are installed through conda because that is more reliable than resolving GPU wheels through pip.

## Download Data

Download the Kaggle dataset:

```bash
python scripts/download_dataset.py --copy --out data/brisc2025
```

Or use the Kaggle cache path printed by:

```python
import kagglehub

path = kagglehub.dataset_download("briscdataset/brisc2025")
print("Path to dataset files:", path)
```

Then set `data_root` in [configs/convnext_base_mtl.yaml](/mnt/8tb_hdd/ryo/BrainTumorMRI/configs/convnext_base_mtl.yaml) if needed.

Expected structure:

```text
data/brisc2025/
  classification_task/
    train/
    test/
  segmentation_task/
    train/images/
    train/masks/
    test/images/
    test/masks/
```

## Preflight

Before training, verify the environment, CUDA visibility, dataset layout, dependencies, and a CPU model shape smoke test:

```bash
scripts/preflight.sh
```

This project is configured to train with PyTorch on GPU. The training entry point defaults to `--device cuda` and will fail instead of silently falling back to CPU if CUDA is not visible.

## Train

```bash
/mnt/8tb_hdd/ryo/miniconda3/bin/conda run -n dl-class-ryo python -m brisc_mtl.train \
  --config configs/convnext_base_mtl.yaml \
  --device cuda
```

Outputs are written to `outputs/convnext_base_mtl/`:

- `best.pt`: best validation checkpoint
- `last.pt`: last epoch checkpoint
- `history.json`: epoch metrics
- `data_summary.json`: split and class counts

For a one-epoch smoke run without changing the config:

```bash
GPU_ID=0 BATCH_SIZE=8 scripts/train_smoke_1epoch.sh
```

For the full training run:

```bash
GPU_ID=0 BATCH_SIZE=16 scripts/train_full.sh
```

## Evaluate

```bash
python -m brisc_mtl.evaluate --checkpoint outputs/convnext_base_mtl/best.pt
```

This evaluates on the official BRISC `test` split and writes:

```text
outputs/convnext_base_mtl/test_eval/metrics.json
```

Main metrics:

- `classification_accuracy`: 4-class tumor type classification
- `classification_macro_f1` and `classification_weighted_f1`: class-imbalance-aware classification metrics
- `classification_ece`: expected calibration error for predicted class confidence
- `binary_detection_accuracy`: tumor vs no tumor
- `binary_detection`: sensitivity, specificity, precision, recall, F1, balanced accuracy, ROC-AUC, and PR-AUC
- `dice`: segmentation mask Dice
- `segmentation`: IoU, precision, and recall for binary tumor masks
- `confusion_matrix`: 4-class confusion matrix

## Predict One Image

```bash
python -m brisc_mtl.predict \
  --checkpoint outputs/convnext_base_mtl/best.pt \
  --image data/brisc2025/segmentation_task/test/images/brisc2025_test_00001_gl_ax_t1.jpg
```

The command prints class probabilities and writes a predicted binary mask to `outputs/predictions/`.

## Interactive Demo

Install the optional demo dependency:

```bash
pip install -e ".[demo]"
```

Launch the local Gradio app:

```bash
python app/gradio_app.py --checkpoint outputs/convnext_base_mtl/best.pt --device cuda
```

The app accepts one MRI image and returns the predicted class, class probabilities, predicted mask, and mask overlay. It is a research demo only and is not for clinical diagnosis.

## Notes

This code uses the segmentation task images as the source of truth because each sample has both an image-level label and a pixel-wise mask. The class label is inferred from the BRISC filename convention, for example `_gl_`, `_me_`, `_pi_`, and `_nt_`.
