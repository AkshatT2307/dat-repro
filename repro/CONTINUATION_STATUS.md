# Five additional epochs per model

Updated: 2026-11-18T13:32:20+05:30

Same batch, data, optimizer and teacher; original LR schedule retained. Fresh patience; previous best retained.

| Run | Status |
|---|---|
| imagenet/idr/b192_e20 | complete; epoch 25; best L1 0.034650 at epoch 25 |
| imagenet/supervised/b192_e20 | complete; epoch 25; best L1 0.032045 at epoch 25 |
| marmousi2/idr/b192_e20 | complete; epoch 25; best L1 0.078128 at epoch 25 |
| marmousi2/supervised/b192_e20 | complete; epoch 25; best L1 0.051524 at epoch 21 |
| marmousi2/idr/b128_e30 | complete; epoch 29; best L1 0.059887 at epoch 19 |
| marmousi2/supervised/b128_e30 | complete; epoch 30; best L1 0.051388 at epoch 20 |

Outputs: runs/continuation5/DATASET/METHOD/ORIGINAL_BUDGET/.
Each continuation.json records its parent, start epoch, LR and settings.
Validation logs contain only additional epochs. Best weights may remain the parent best.

[Completed continuation results](CONTINUATION_RESULTS.md): four-model loss curves, validation comparison and final evaluation.
