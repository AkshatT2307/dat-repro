# Setup verification

- Original repository remains unchanged (paper.pdf was already untracked).
- Python 3.13 / PyTorch 2.7.1+cu128 detects RTX 5070 Ti and performs real CUDA training.
- `pip check` passes; package versions frozen in requirements.lock.txt.
- Kodak: all 24 images decode. BSDS300: all 300 images decode.
- GPU batch tuner: 24 succeeds; 25 fails the >=512 MiB headroom criterion.
  Other processes consumed approximately 10 GB when tuning. This is NOT a
  maximum for an otherwise idle 16 GB GPU. Retune before full training.
- Batch 24: 20 actual IDR steps with data loader and teacher refinement succeed.
- Both IDR and supervised: 6-step smoke training succeeds.
- IDR 6 steps uninterrupted versus 3 + resume to 6: network and teacher tensors
  match bit-for-bit, and scheduler/step/epoch/offset match. Test crosses epoch 1.
- Both smoke checkpoints evaluated on Kodak and BSDS300, sigma 25/50; all
  four summary rows and 648 per-image records generated for each model.
- The supervised evaluation used the optimized NumPy-to-tensor conversion;
  original IDR smoke evaluation started before that optimization.
- These random-image smoke models and their scores are NOT reproduction results.
- ImageNet: Hugging Face metadata resolves to revision
  49e2ee26f3810fb5a7536bbf732a7b07389a47b5, 14 validation parquet shards.
  Authentication is now successful. Four validation shards were downloaded;
  subsequent full-file transfers stalled. Downloader stopped at user request.
  The user will run download_imagenet.py locally; completed shards are cached.
  Image export and full training have not started.

Update: user completed all 14 validation shard downloads. Fixed Hugging Face's
full-repository split metadata check by setting verification_mode='no_checks'
for the intentionally validation-only load; explicit 50,000-example validation
remains. Verified successful loading of all 50,000 cached examples and access
to original image bytes. Image export remains for the user to run.

## Fixed-epoch 128 pipelines

- Added run_epoch128.sh for ImageNet and Marmousi2, Fast IDR and supervised.
- 13 complete epochs, 50,000 sample presentations per epoch, drop_last=False.
  Batch 256 resolves to 2,548 updates and LR milestones 1,274 / 2,039.
- Marmousi2 original archive downloaded from the HF mirror and SHA-256 verified.
  NMOstack_time decoded from IBM floats to a 1001x2721 floating-point section.
  Original license retained; train-only RMS and separated trace splits recorded.
- Seven CPU unit tests pass (budget, resize, scheduler, resume guards, decoding,
  spatial isolation/normalization, and metrics); pip check passes.
- Four two-epoch CPU smoke trainings pass, with batch sequence [3,3,2] per epoch.
- Mid-epoch seismic IDR resume exactly matches uninterrupted network/teacher
  tensors and scheduler. Both seismic smoke models produce complete reports
  for 21 held-out tiles at three noise levels and example PNG/NPZ artifacts.
- No new full experiments started. No GPU memory tests were performed during
  the existing GPU run. Batch 256 is configurable and checked at launch.
- CPU worker IPC needed execution outside sandbox; the stalled initial sandbox
  smoke process was explicitly stopped. Existing GPU training was untouched.

## Twenty-epoch ceiling with ImageNet holdout

- Deterministic split created from all 50,000 ImageNet source images: 49,000
  training / 1,000 validation, seed 2026, disjoint filenames recorded in
  data/imagenet_split.json. No Kodak or BSDS images enter checkpoint selection.
- Epoch validation uses fixed clean-reference float L1 at sigma 25 and 50
  (ImageNet), or the separate seismic validation region at sigma .1/.2/.4.
  Patience 5, minimum improvement 1e-5, best and latest full checkpoints saved.
- Launcher defaults to 20 maximum epochs and evaluates best_weights.pth.
  ImageNet batch 256 dry-run: 192 updates/epoch, max 3,840 updates, LR milestones
  1,920 / 3,072. Seismic remains 50,000 draws/epoch, max 3,920 updates.
- All 10 CPU unit tests pass. ImageNet Fast IDR/supervised and seismic Fast IDR
  CPU smoke runs forced early stopping at epoch 2; stopped-run resume performs
  no new optimizer updates. These are only mechanical checks, not model results.
- CPU integration check_validation_resume.py passes: epoch-boundary resume and
  resume before pending final validation reproduce network, teacher, scheduler
  and early-stopping state exactly. Best checkpoint and exported weights match.
- Shell syntax and actual 49,000/1,000 ImageNet dry-run verified. No new full
  GPU training launched and the existing GPU process was not modified.

## GPU launch at batch 192

User requested stopping the previous 50,000-step run and launching all four
experiments sequentially in tmux. Old process interrupted; saved checkpoints
retained. Session idr128 runs `bash run_epoch128.sh all 192 20`.
Preflight now accounts for reusable PyTorch cache using driver-free + current
reserved - peak allocated bytes. Batch 192 trial completed three refinement
steps with 9,290,995,200 peak allocated bytes and 4,155,222,528 estimated peak
headroom. First ImageNet Fast IDR training steps succeeded and metrics are
being written under runs/epoch128/imagenet/idr/b192_e20. The full experiments
are ongoing, not yet validated scientific results.

## Overnight supervisor

- Added a persistent tmux monitor with atomic status reports, validation-only
  quality decisions, and bounded recovery/retry of paired experiments.
- All 15 CPU unit tests pass, including healthy-run preservation, one quality
  retry, OOM batch fallback, bounded recovery failure, and process identity checks.
- Confirmed the monitor is live in idr128:monitor and reporting the current
  ImageNet run. Its first two validation losses were 0.051851 and 0.046626;
  no configuration adjustment was warranted. Tests do not establish eventual
  training quality or guarantee successful overnight completion.

## GitHub publication package

- All 15 CPU unit tests passed against the packaged source using the existing
  local benchmark data (datasets are not committed).
- Original and continuation plots regenerated from archived records using the
  README restore commands; no training or weights required.
- Tracked files scanned for common GitHub/Hugging Face/AWS token formats and
  private keys; no matches. No checkpoint, virtual environment or full dataset
  files are tracked. Largest tracked file is below 6 MB.
- Code license and upstream revision preserved. Absolute workstation paths in
  text artifacts normalized for portability; historical hashes are unchanged.

## Batch-4 fixed 50,000-step protocol

- Added train_steps50k.py as a separate driver; historical trainers/configurations
  and results are unchanged. Four fresh runs, 128px, batch 4, exactly 50,000 updates,
  LR milestones 25,000/40,000. Validation every 2,500, with no early stopping.
- Short CPU integration runs passed for both datasets and both methods. Exact
  network/teacher/Adam/scheduler resume verified at epoch boundaries, mid-epoch,
  and before pending final validation. Completed resumes add no optimizer updates.
- All 15 existing CPU unit tests still pass. Shell syntax/help and both dataset
  dry-run budgets verified. Combined report and curves tested using isolated fixtures.
- GPU benchmark: 20 warmup + 200 measured batch-4 steps for each configuration,
  actual data loaders, FP32, Fast IDR teacher active. Models discarded. Timing
  estimates saved in reports/steps50k_b4_128/timing.json.
- Fresh-machine bootstrap prepared, but not executed end-to-end on a new host;
  downloads and new-host driver compatibility remain environment-dependent.
- No production 50,000-step run was launched during preparation.
