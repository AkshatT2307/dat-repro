# dat-repro

> New experiment: [batch 4 / 50,000 optimizer steps, four runs, fresh-machine setup](repro/RUN_STEPS50K.md). Run `bash repro/setup_steps50k.sh --run` after cloning to set up and launch.

Fast IDR and supervised U-Net denoising experiments on ImageNet and Marmousi2,
trained on one NVIDIA RTX 5070 Ti (16 GB). Includes runnable code, experiment
records, validation curves and inference examples.

This is a compute-reduced adaptation of
[IDR: Self-Supervised Image Denoising via Iterative Data Refinement](https://arxiv.org/abs/2111.14358),
using the [original implementation](https://github.com/zhangyi-3/IDR) at commit
`7061caa7360008135e7f732f82b94f7b230e59b1`. It does **not** reproduce the paper's
original 50,000-update protocol exactly. Upstream MIT attribution is retained in
[LICENSE](LICENSE) and [UPSTREAM_README.md](repro/UPSTREAM_README.md).

## Experiments

- **ImageNet:** download only the 50,000-image validation source from
  `ILSVRC/imagenet-1k`; split into 49,000 training and 1,000 held-out validation
  images with seed 2026. Crop clean images to 256×256, area-downsample to 128×128,
  then add Gaussian noise. Train Fast IDR and supervised U-Net at batch 192 for
  up to 20 epochs, followed by five additional epochs from full checkpoints.
- **Marmousi2:** processed synthetic NMO-stack section, native 128×128 patches,
  single-channel U-Net, train-only RMS normalization, separated train/validation/
  test trace regions with 128-trace buffers. Each epoch draws 50,000 patches.
  Train both methods at batch 192 (20-epoch ceiling), then a separate batch-128
  pair (30-epoch ceiling). Continue all models for up to five additional epochs.
- **Model selection:** clean-reference validation L1, patience 5, minimum
  improvement 1e-5. Even Fast IDR uses clean references for checkpoint selection;
  these references are not training targets for Fast IDR. Kodak, BSDS and seismic
  test results do not guide model selection. Continuations explicitly grant fresh
  patience while preserving prior best models, Adam state and the IDR teacher.

These are controlled **added Gaussian noise** experiments. Marmousi2 is not a
true field-noise benchmark, and ImageNet-to-seismic transfer was not tested.

## Results and figures

- [Four selected models: continuation results](repro/CONTINUATION_RESULTS.md)
- [All six original validation histories](repro/reports/validation_curves/validation_all_six.png)
- [Four selected models: original + continued histories](repro/reports/continuation_curves/four_models_full_history.png)
- [Late-epoch detail](repro/reports/continuation_curves/four_models_late_epochs.png)
- [ImageNet inference examples](repro/reports/inference_examples/imagenet_sigma50.png)
- [Seismic inference examples](repro/reports/inference_examples/marmousi2_sigma0.4.png)
- [Archived per-step metrics, configurations and evaluations](repro/results/)

Final ImageNet generalization PSNR (dB), using the best checkpoints after continuation:

| Test data | Noise σ | Fast IDR | Supervised |
|---|---:|---:|---:|
| Kodak | 25 | 30.52 | 31.10 |
| Kodak | 50 | 27.89 | 27.96 |
| BSDS300 | 25 | 29.98 | 30.52 |
| BSDS300 | 50 | 27.25 | 27.27 |

Marmousi2 batch-128 pair, held-out test mean output SNR (dB):

| Noise σ (normalized) | Fast IDR | Supervised |
|---|---:|---:|
| 0.1 | 17.67 | 19.07 |
| 0.2 | 15.16 | 16.27 |
| 0.4 | 12.43 | 13.33 |

Seismic SNR averages use 7 nonzero-signal tiles; RMSE includes all 21 test tiles.
These are single-seed measurements, not estimates of run-to-run uncertainty.

## Setup

The training environment used Python 3.13 and PyTorch 2.7.1+cu128.

```bash
cd repro
python3.13 -m venv .venv
.venv/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements-plots.txt
```

The historical environment snapshot is [requirements.lock.txt](repro/requirements.lock.txt).
Plots were originally generated with a separate Matplotlib 3.10.8 installation;
`requirements-plots.txt` adds it to the reproduction environment.

## Download data

Accept access conditions for [ImageNet on Hugging Face](https://huggingface.co/datasets/ILSVRC/imagenet-1k),
then authenticate locally. Never commit access tokens.

```bash
.venv/bin/hf auth login
.venv/bin/python download_imagenet.py
.venv/bin/python imagenet_split.py
.venv/bin/python prepare_data.py --benchmarks
.venv/bin/python prepare_seismic.py
```

Dataset files and cached downloads are excluded from Git. Small inference example
figures and numerical example arrays are included as experiment artifacts; dataset
licenses/access conditions remain separate from the source-code license.
Marmousi2 source/license metadata is preserved in [results/data/marmousi2](repro/results/data/marmousi2).

## Train and evaluate

```bash
# Both datasets; Fast IDR then supervised for each, with validation and evaluation.
bash run_epoch128.sh all 192 20

# Separate seismic retry used in this study.
bash run_epoch128.sh marmousi2 128 30

# Once all six original runs exist, continue each for at most five more epochs.
.venv/bin/python -u run_continuations.py
```

The epoch launcher resumes existing compatible checkpoints automatically. Changing
batch size creates a different run; equal epochs do not imply identical optimization.
See [RUN_128.md](repro/RUN_128.md) for preprocessing, stopping, LR and checkpoint details.

Checkpoints, virtual environments, full datasets and runtime logs are excluded
from this Git repository. Training checkpoints remain on the original machine;
retrain locally to run new inference. Precomputed results and plots are included.

## Recreate plots without training

From `repro/`, restore the archived **records** into the paths expected by plotting
scripts (no weights or dataset download required):

```bash
mkdir -p runs
cp -a results/epoch128 results/continuation5 runs/
.venv/bin/python plot_validation.py
.venv/bin/python plot_continuations.py
```

After training and downloading data, generate the four-model inference examples:

```bash
.venv/bin/python inference_examples.py
```

## Verification and provenance

[VALIDATION.md](repro/VALIDATION.md) records CPU unit/integration checks and GPU
launch checks. Some tests require the downloaded Kodak images; checkpoint smoke
tests additionally require their generated smoke fixtures. Status Markdown files
are historical snapshots, not live GitHub dashboards. Notification/tmux scripts
are optional local workstation utilities.

Machine-specific absolute paths in archived text files have been normalized for
publication. Stored manifest hashes refer to the original training manifests.
