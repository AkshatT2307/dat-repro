# Four separate runs: batch 4, exactly 50,000 optimizer updates

This is a new experiment family, independent of epoch128 and continuation5.
No old model weights are used. All four runs start from scratch:

1. ImageNet Fast IDR
2. ImageNet supervised U-Net
3. Marmousi2 Fast IDR
4. Marmousi2 supervised U-Net

## Protocol

- **Batch size 4**, no gradient accumulation, exactly **50,000 optimizer updates**.
- **128×128 remains unchanged**: ImageNet clean 256×256 crop area-downsampled to
  128×128 before noise; seismic native 128×128 patches without resizing.
- Adam LR 3e-4, halved after updates 25,000 and 40,000. FP32, TF32 disabled.
- Same AWGN generation and original Fast IDR target-refinement implementation.
  One sigma per batch per corruption call; independent pixel/channel noise.
- Same fixed ImageNet split: 49,000 train / 1,000 validation, seed 2026. This uses
  only the downloaded ImageNet validation source, never the ImageNet training split.
- Same seismic spatial split and train-only RMS normalization. An epoch draws
  50,000 training-region patches. No validation/test patches enter gradient updates.
- Validation every 2,500 steps, including step 50,000. **No early stopping**: the
  optimizer-step budget takes priority. Preserve best-validation and final models.
- Final-step evaluation is the primary fixed-budget comparison. Best-validation
  evaluation is reported separately, never silently substituted for final weights.
- Fast IDR teacher refresh stays once per data-loader epoch, as in the existing
  code. It is not refreshed at every validation or optimizer update.

Each model sees 200,000 sample presentations. For ImageNet this is approximately
4.082 epochs (12,250 updates per epoch); seismic runs exactly 4 epochs (12,500
updates per epoch). ImageNet teacher refreshes occur before steps 12,251, 24,501,
36,751 and 49,001; seismic refreshes before 12,501, 25,001 and 37,501.

This follows the requested 50k-step/batch-4 convention, **not the paper's full
original setup**: our patch resolution, ImageNet holdout, and seismic extension
remain project-specific. It is not equivalent to the earlier 20-epoch budgets.

## Fresh-machine setup

Prerequisites: Linux, a compatible NVIDIA driver, and Python 3.10–3.13 with venv
support (3.13 matches the original environment). The bootstrap installs the Python
packages, not the system driver. Allow roughly 40–50 GB free disk for the environment,
dataset downloads/caches, exported images, checkpoints and logs.

Clone the private dat-repro repository using an authorized GitHub account. Then:

```bash
cd dat-repro/repro
bash setup_steps50k.sh
```

The script creates a **separate `.venv_steps50k` environment**, installs
PyTorch 2.7.1+cu128 and pinned requirements, checks CUDA, downloads only ImageNet
validation, creates its holdout split, downloads Kodak/BSDS300 and Marmousi2,
and prints all four dry-run plans. Existing downloaded files/caches are reused.

ImageNet requires accepting the dataset's access conditions on Hugging Face.
The script prompts for local HF login when needed. For unattended setup, supply
HF_TOKEN securely via the environment; do not put it in the script or repository.
Dataset access approval cannot be automated by this bootstrap.

To set up and immediately train all four runs:

```bash
bash setup_steps50k.sh --run
```

To launch after setup, preferably in tmux:

```bash
tmux new-session -s dat-steps50k
bash run_steps50k.sh
# Detach: Ctrl+B, then D.
```

If Python 3.13 is not your default: `PYTHON_BIN=python3.13 bash setup_steps50k.sh`.
For an existing compatible environment:
`TRAIN_PYTHON="$PWD/.venv/bin/python" bash run_steps50k.sh`.
That existing environment must include matplotlib for the final plotting step.
Workers default to 4; set WORKERS before the first launch if needed, and retain
that value on resume.

## Outputs and resume

- Models/logs: `runs/steps50k_b4_128/{imagenet,marmousi2}/{idr,supervised}/`.
- Per-step loss/LR: `metrics.jsonl`; per-validation loss: `validation.jsonl`.
- `latest.pt`: full resumable state (student, teacher, Adam, LR scheduler, RNG,
  data-loader epoch/position, validation state and protocol fingerprint).
- `weights.pth`: latest/final student; `best.pt` and `best_weights.pth`: validation best.
- Latest checkpoints every 1,000 updates, every validation, and at completion.
- Kodak/BSDS full-resolution PSNR/SSIM at sigma 25/50: `evaluation_final` and
  `evaluation_best` for ImageNet. Seismic test-region SNR/RMSE/correlation at
  sigma .1/.2/.4 in the analogous folders. Test scores never select checkpoints.
- Pipeline console logs: `logs/steps50k_b4_128-*.log`.
- Final combined report, CSV/JSON and loss curves: `reports/steps50k_b4_128/`.

Rerun `bash run_steps50k.sh` to resume this experiment family. It never resumes
from epoch128 or continuation5. Keep the same environment, dataset paths, worker
count and seed on resume. To run a second independent replicate, use explicit
new output directories with train_steps50k.py and record a different seed.

## Runtime measurements

Short measurements on the current RTX 5070 Ti, PyTorch 2.7.1+cu128, FP32, four
loader workers: 20 warmup + 200 measured updates per configuration, actual data
loading, and the teacher enabled for Fast IDR. Benchmark models were discarded.

| Run | Measured seconds/update | 50k training + 20 validations | Practical planning allowance including saves/evaluation |
|---|---:|---:|---:|
| ImageNet Fast IDR | 0.02716 | 25.3 minutes | 30–40 minutes |
| ImageNet supervised | 0.01763 | 17.3 minutes | 20–30 minutes |
| Marmousi2 Fast IDR | 0.01126 | 9.4 minutes | 12–18 minutes |
| Marmousi2 supervised | 0.00899 | 7.5 minutes | 10–15 minutes |

Measured extrapolation totals about 60 minutes. Plan roughly **1.2–1.75 hours**
for all four on similar hardware; these are estimates, not completed-run timings.
Environment setup and downloads are extra and depend on network/storage. A new
GPU, CPU or disk can materially change the estimates. The ImageNet benchmark
used locally available data; cold filesystem caches can make loading slower.

Measure on the new machine after setup:

```bash
.venv_steps50k/bin/python benchmark_steps50k.py --steps 200 --warmup 20
```

Raw measurements: [timing.json](reports/steps50k_b4_128/timing.json).

## Verification

All four training paths were exercised in short CPU smoke runs. Integration tests
check exact optimizer-step counts, validation at fixed steps/final completion,
and bit-for-bit checkpoint continuation from epoch boundaries, mid-epoch saves
and saves before final validation. An already completed run performs no new
updates on resume. These tiny fixtures are not scientific experiment results.
