# Model Card

## Model Details

BrainTumorMRI is a PyTorch + MONAI research prototype for multitask brain MRI analysis on BRISC 2025 2D slices.
The current headline checkpoint is `outputs/convnext_tiny_mtl/best.pt`, trained with a shared ConvNeXt-Tiny encoder,
a 4-class classification head, and a U-Net style binary segmentation decoder.

The model predicts:

- 4-class label: `no_tumor`, `glioma`, `meningioma`, `pituitary`
- binary tumor detection: derived from the 4-class classification output
- binary tumor mask: predicted by the segmentation decoder with a default threshold of 0.5

## Intended Use

This model is intended for research, education, portfolio demonstration, and reproducible engineering experiments. It
can be used to demonstrate multitask learning, evaluation metrics, calibration reporting, segmentation overlays, and a
local Gradio inference workflow.

It is not intended for clinical diagnosis, clinical triage, treatment planning, or automated medical decision-making.

## Dataset

The model is trained and evaluated on the BRISC 2025 Kaggle brain MRI dataset. The repository uses the segmentation
task samples because they provide both image-level labels and pixel-wise masks.

The dataset is not included in this repository. Users must download it separately from Kaggle and follow the dataset
terms.

## Metrics

Current headline result on the official BRISC test split:

| Metric | Value |
| --- | ---: |
| Classification Accuracy | 0.9940 |
| Macro F1 | 0.9947 |
| Binary Detection Accuracy | 1.0000 |
| Sensitivity | 1.0000 |
| Specificity | 1.0000 |
| ROC-AUC | 1.0000 |
| PR-AUC | 1.0000 |
| Dice | 0.8387 |
| IoU | 0.7814 |
| ECE | 0.0055 |

Detailed results are documented in [reports/report.md](reports/report.md).

## Stability Across Seeds

ConvNeXt-Tiny was evaluated across 3 seeds:

| Metric | Mean | Std |
| --- | ---: | ---: |
| Test Classification Accuracy | 0.9920 | 0.0020 |
| Test Detection Accuracy | 1.0000 | 0.0000 |
| Test Dice | 0.8360 | 0.0024 |

## Limitations

- The model is evaluated on one official test split and has not been externally validated.
- It is a 2D slice model and does not use full volumetric MRI context.
- Binary detection is derived from the 4-class classification head, not from a clinically selected operating point.
- Segmentation quality is weaker than classification and detection quality. Boundaries and small tumor regions should
  be reviewed manually.
- Performance may shift across MRI acquisition protocols, scanners, institutions, preprocessing choices, and patient
  populations.

## Ethical Considerations

Medical AI outputs can create false confidence if presented without uncertainty, limitations, and human review. This
project should be presented as a research prototype only. Any downstream clinical use would require data governance,
external validation, clinical workflow review, privacy review, calibration review, and regulatory assessment.

## Not For Clinical Diagnosis

BrainTumorMRI is a research demo only. It is not a medical device and must not be used to diagnose patients or make
clinical decisions.
