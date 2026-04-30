# ConvNeXt-Tiny Multi-Seed Results

Base config: `configs/convnext_tiny_mtl.yaml`

| seed | epochs | best epoch | best val score | val cls acc | val dice | test cls acc | test det acc | test dice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 40 | 35 | 0.9233 | 0.9973 | 0.8493 | 0.9940 | 1.0000 | 0.8387 |
| 7 | 20 | 12 | 0.9043 | 0.9907 | 0.8179 | 0.9900 | 1.0000 | 0.8349 |
| 123 | 27 | 19 | 0.9195 | 0.9960 | 0.8431 | 0.9920 | 1.0000 | 0.8344 |

| metric | mean | std |
| --- | --- | --- |
| best val score | 0.9157 | 0.0101 |
| val dice | 0.8367 | 0.0166 |
| test cls acc | 0.9920 | 0.0020 |
| test det acc | 1.0000 | 0.0000 |
| test dice | 0.8360 | 0.0024 |

## Takeaways

- Test metrics are stable across seeds, especially binary detection and segmentation Dice.
- Validation Dice is more seed-sensitive than test Dice in this split, with seed 7 underperforming seed 42.
- Early stopping reduced wasted training: seed 7 stopped at 20 epochs and seed 123 stopped at 27 epochs.
- The single-seed headline result remains representative for test Dice, but reporting mean/std is more rigorous.
