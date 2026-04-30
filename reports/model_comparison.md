# BRISC Model Comparison

| run | epochs | best epoch | best val score | val cls acc | val det acc | val dice | test cls acc | test det acc | test dice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke_1epoch | 1 | 1 | 0.8625 | 0.9733 | 0.9947 | 0.7517 | 0.9620 | 0.9860 | 0.7554 |
| convnext_base_mtl | 40 | 31 | 0.9223 | 0.9947 | 1.0000 | 0.8498 | 0.9920 | 0.9980 | 0.8406 |
| convnext_tiny_mtl | 40 | 35 | 0.9233 | 0.9973 | 1.0000 | 0.8493 | 0.9940 | 1.0000 | 0.8387 |
| resnet50_mtl | 40 | 24 | 0.9140 | 0.9947 | 1.0000 | 0.8333 | 0.9910 | 0.9980 | 0.8302 |
| convnext_tiny_seg15_wd1e4_mtl | 23 | 15 | 0.9193 | 0.9973 | 1.0000 | 0.8412 | 0.9900 | 0.9990 | 0.8278 |

Validation score = 0.5 * classification accuracy + 0.5 * Dice.

## Takeaways

- `convnext_tiny_mtl` still has the best validation score and best test classification/detection accuracy.
- `convnext_base_mtl` still has the best test Dice, but the margin over `convnext_tiny_mtl` is small.
- `resnet50_mtl` is a useful CNN baseline, but it underperforms both ConvNeXt variants on validation score and test Dice.
- `convnext_tiny_seg15_wd1e4_mtl` stopped early at 23 epochs and did not improve over the untuned Tiny run.
- The current best practical default is `convnext_tiny_mtl` when classification/detection is the priority, and `convnext_base_mtl` when segmentation Dice is the priority.
- Train/validation curve and overfitting-gap details are in [history_analysis.md](history_analysis.md).
