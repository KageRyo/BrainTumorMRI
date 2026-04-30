# Dataset Card

## Dataset Source

This project uses the BRISC 2025 brain MRI dataset from Kaggle:

<https://www.kaggle.com/datasets/briscdataset/brisc2025>

The data is downloaded separately by the user. Dataset files are not included in this repository and are excluded from
Git.

## Classes

BrainTumorMRI uses four image-level classes:

- `no_tumor`
- `glioma`
- `meningioma`
- `pituitary`

For binary tumor detection, `glioma`, `meningioma`, and `pituitary` are treated as tumor-positive, while `no_tumor` is
treated as tumor-negative.

## Splits

The project follows the BRISC-provided split layout when available. The current report evaluates the headline
checkpoint on 1,000 official test samples with this class support:

| Class | Support |
| --- | ---: |
| no_tumor | 140 |
| glioma | 254 |
| meningioma | 306 |
| pituitary | 300 |

## Data Used By The Project

The repository primarily uses the segmentation task samples because they include both:

- an MRI image
- a binary tumor mask

The image label is parsed from the BRISC filename convention by the data pipeline.

## Known Limitations

- The dataset may not represent all scanner vendors, acquisition protocols, institutions, demographics, or clinical
  presentations.
- The project currently trains a 2D slice model rather than a volumetric model.
- Segmentation masks are treated as binary masks; boundary uncertainty and inter-rater variability are not modeled.
- The current evaluation is not an external validation study.

## License And Terms

The repository code is licensed under Apache License 2.0. The BRISC 2025 dataset is separate from this repository and
is governed by the dataset provider's Kaggle terms. The Apache License 2.0 for this repository does not apply to the
downloaded dataset.

## Data Not Included In Repository

Do not commit downloaded dataset files, masks, local Kaggle credentials, checkpoints, or generated `outputs/`
artifacts. Keep local data under ignored directories such as `data/` and `outputs/`.
