# Overnight experiment status

Updated: 2026-11-18T05:24:11+05:30

Monitoring finished. Review completion states and selected configurations below.

Original plan: batch 192, up to 20 epochs, patience 5. Both methods use
the same ImageNet holdout or the separate seismic validation region.
Kodak, BSDS and seismic test scores never trigger adjustments.

| Dataset | Method | Batch / limit | Last step | Validated epochs | Best validation L1 | State |
|---|---|---|---|---|---|---|
| imagenet | idr | 192 / 20 | 5120/5120 | 20 | 0.035271 | training and evaluation complete |
| imagenet | supervised | 192 / 20 | 5120/5120 | 20 | 0.032551 | training and evaluation complete |
| marmousi2 | idr | 192 / 20 | 5220/5220 | 20 | 0.089272 | training and evaluation complete |
| marmousi2 | supervised | 192 / 20 | 5220/5220 | 20 | 0.051833 | training and evaluation complete |
| imagenet | idr | 128 / 30 | 0/? | 0 | - | not started |
| imagenet | supervised | 128 / 30 | 0/? | 0 | - | not started |
| marmousi2 | idr | 128 / 30 | 9384/11730 | 24 | 0.059887 | training and evaluation complete |
| marmousi2 | supervised | 128 / 30 | 9775/11730 | 25 | 0.051388 | training and evaluation complete |

## Decisions

- 2026-11-18T01:29:38+05:30: Monitor attached to original pipeline PID 4143634; no initial configuration change
- 2026-11-18T03:44:10+05:30: Original pipeline exited; checking completeness and validation quality
- 2026-11-18T03:44:10+05:30: imagenet: validation checks passed; keeping b192_e20
- 2026-11-18T03:44:10+05:30: marmousi2: validation quality check: Fast IDR validation L1 exceeds 1.5 times supervised L1
- 2026-11-18T03:44:10+05:30: Running marmousi2, batch 128, epoch limit 30: One paired quality retry: more updates and larger epoch ceiling
- 2026-11-18T05:24:11+05:30: marmousi2 batch 128, limit 30: pipeline exit 0; log overnight-marmousi2-b128-e30-1794953650056846247.log
- 2026-11-18T05:24:11+05:30: marmousi2: paired configuration selected by mean validation L1: b128_e30
- 2026-11-18T05:24:11+05:30: Monitoring finished. Review completion states and selected configurations below.

## Review files

- Run folders: `runs/epoch128/DATASET/METHOD/bBATCH_eEPOCHS/`.
- `validation.jsonl`: validation curves; `training_summary.json`: best epoch and stopping state.
- `best.pt` / `best_weights.pth`: model selected using validation only.
- `evaluation/summary.json`: final generalization metrics.
- `runs/epoch128/DATASET/comparison_bBATCH_eEPOCHS/results.csv`: paired results.
- `logs/overnight_events.jsonl`: persistent decisions; `logs/tmux_latest.txt`: latest training pane.

Automatic recovery is bounded: at most one recovery run and one quality retry per dataset.
Quality retry triggers only for <10% L1 gain over expected noisy-input error, or Fast IDR
L1 >1.5x supervised. These are failure heuristics, not proof of paper-level performance.
A quality retry trains BOTH methods at batch 128 with up to 30 epochs in separate folders.
If original runs pass these checks, their batch and epoch limit stay unchanged.

Stop future automatic actions: `kill -INT $(cat repro/logs/overnight_monitor.pid)` (from DAT).
This does not stop an already running training pipeline.

## Selected configurations (validation only)

- imagenet: b192_e20
- marmousi2: b128_e30

[Completed continuation results](CONTINUATION_RESULTS.md): four-model loss curves, validation comparison and final evaluation.
