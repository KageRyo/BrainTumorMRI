# BRISC Model Comparison

| run | epochs | best epoch | best val score | val cls acc | val det acc | val dice | test cls acc | test det acc | test dice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke_1epoch | 1 | 1 | 0.8625 | 0.9733 | 0.9947 | 0.7517 | 0.9620 | 0.9860 | 0.7554 |
| convnext_base_mtl | 40 | 31 | 0.9223 | 0.9947 | 1.0000 | 0.8498 | 0.9920 | 0.9980 | 0.8406 |

Validation score = 0.5 * classification accuracy + 0.5 * Dice.

## Notes

- Full training used `configs/convnext_base_mtl.yaml` with 40 epochs, batch size 16, eval batch size 16, and GPU 0.
- Test metrics are from the official BRISC `test` split using each run's `best.pt`.
- The full run improves over the one-epoch smoke baseline by +0.0300 classification accuracy, +0.0120 binary detection accuracy, and +0.0852 Dice on the test split.
