# Threshold Analysis

Binary tumor detection scores are computed as `1 - P(no_tumor)` from the 4-class classification head.

- ROC-AUC: 1.0000
- PR-AUC: 1.0000

| Threshold | Sensitivity | Specificity | Precision | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0.30 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.50 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.70 | 0.9988 | 1.0000 | 1.0000 | 0.9994 |

Figures:

- [ROC curve](figures/roc_curve.png)
- [PR curve](figures/pr_curve.png)

The current headline checkpoint separates tumor and no-tumor samples cleanly on this test split. External
validation is still required before choosing any clinical operating threshold.
