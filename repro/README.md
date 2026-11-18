> For the completed experiments and portable setup, start with the [repository README](../README.md). The fixed-step instructions below are historical.

# Latest instructions

Use [RUN_128.md](RUN_128.md) for the current fixed-epoch ImageNet and Marmousi2 pipelines. Earlier fixed-step protocols below remain for reference.

# IDR Gaussian reproduction on one RTX 5070 Ti

Scope: RGB Gaussian denoising, Fast IDR and supervised U-Net, trained from scratch
on all 50,000 ImageNet validation images; evaluate on Kodak and BSDS300 at sigma
25 and 50. No RAW experiment. No pretrained weights required.

## Environment

The local `.venv` uses Python 3.13, PyTorch 2.7.1+cu128 (Blackwell compatible),
and `requirements.txt`. `requirements.lock.txt` records all installed packages.
To rebuild:

```bash
cd repro
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
```

## Data and execution

ImageNet comes only from `ILSVRC/imagenet-1k` on Hugging Face. Accept its
access conditions in your browser and authenticate locally with `.venv/bin/hf auth login`
if needed; never put a token in scripts or chat. Download validation only:

```bash
cd repro
.venv/bin/python download_imagenet.py
.venv/bin/python prepare_data.py --benchmarks
bash run_gaussian.sh data/imagenet_val 24
```

The downloader explicitly selects validation parquet shards and passes
`split="validation"`; no training/test shards are requested. It exports original
image bytes without re-encoding, and records source revision and image checksums
in `data/imagenet_source.json`. No labels are used for denoising.

Alternatively `bash run_gaussian.sh /absolute/path/to/extracted/validation`
accepts flat or class-subfolder layouts. The training driver rejects a count
other than 50,000 unless explicitly smoke-testing with `--allow-small-data`.

The second launcher argument sets batch size (e.g. `24` or `4`); use `auto`
to retune. The launcher trains Fast IDR and supervised sequentially, each for
exactly 50,000 iterations with that same batch size, then evaluates both.
The default paper profile rejects other iteration counts; the compact profile below uses 10,000. Run from a terminal with working CUDA;
the sandbox in this session blocks GPU access unless execution is escalated.
Other GPU processes are never stopped. The tuner finds the largest successful
integer batch with at least 512 MiB remaining, using actual FP32 optimizer steps
and IDR's frozen teacher. It does not assume 16 GB is entirely free. Retune after
other GPU jobs finish. A stored batch is valid only for similar free memory.

Individual commands:

```bash
.venv/bin/python gaussian.py tune
.venv/bin/python gaussian.py train --method idr --data data/imagenet_val
.venv/bin/python gaussian.py train --method supervised --data data/imagenet_val
# Resume either interrupted run (local trusted checkpoints only):
.venv/bin/python gaussian.py train --method idr --data data/imagenet_val --resume
.venv/bin/python evaluate.py --weights runs/idr/weights.pth --output runs/idr/evaluation
.venv/bin/python evaluate.py --weights runs/supervised/weights.pth --output runs/supervised/evaluation
```

## Protocol and deviations

- Source: https://github.com/zhangyi-3/IDR commit
  `7061caa7360008135e7f732f82b94f7b230e59b1`. Original checkout `../IDR` is unchanged.
  See `UPSTREAM_README.md`, `LICENSE`, and `upstream.patch`.
- Original U-Net, Gaussian corruption, L1 loss, Adam, and Fast IDR target
  construction are reused. The original shared noise level within each batch is
  preserved. FP32 is used, with TF32 and mixed precision disabled.
- Paper recipe: 50,000 optimizer steps, initial LR 3e-4, halved at 25k and 40k,
  256x256 crops, sigma uniform in [0,50], no augmentation. This intentionally
  corrects the public YAML's 500,000-step/cosine discrepancy.
- User-authorized larger batch: **not an exact paper reproduction**. Keeping
  50k steps with batch B processes 50,000*B crops rather than 200,000. It also
  changes the number of epochs and hence IDR teacher updates. LR is not scaled.
  Use `--batch 4` on BOTH runs for the paper batch. Do not compare runs with
  different batch settings as a controlled method comparison.
- Driver adds single-GPU support (no Slurm/memcached), validated recursive image
  discovery, exact step accounting, atomic latest checkpoint, teacher/optimizer/
  scheduler/RNG checkpointing and epoch/offset resume. It avoids original
  checkpoint off-by-one behavior and incompatible bare-vs-DataParallel weights.
- Evaluation replaces slow list-to-tensor conversion with NumPy tensor conversion,
  fixes train-RGB/test-BGR mismatch and uses modern SSIM channel_axis.
  It preserves upstream uint8 rounding/clipping and SSIM defaults. Sorted images
  and fixed Gaussian RNG give both methods identical evaluation corruption.
  The authors' original random seeds are unknown, so exact decimals may differ.
- Training records absolute image paths and a manifest digest, config and runtime
  versions. Benchmark archive hashes/URLs are in `data/downloads.json`.

## Outputs and reference

`runs/{idr,supervised}/`: config.json, manifest.txt, metrics.jsonl, latest.pt,
weights.pth, evaluation/summary.json, evaluation/per_image.csv.
`batch.json`: measured memory, batch and short-probe timing (not a full runtime
benchmark). Metrics are saved EVERY step. Console progress is printed every 100 training
steps by default (`--log-every` changes console frequency only).

| Dataset | sigma | IDR PSNR/SSIM | Supervised PSNR/SSIM |
|---|---:|---|---|
| Kodak |25|32.36 / 0.884|32.39 / 0.885|
| Kodak |50|29.27 / 0.803|29.24 / 0.803|
| BSDS300 |25|31.48 / 0.890|31.40 / 0.889|
| BSDS300 |50|28.25 / 0.802|28.17 / 0.799|

Source: supplied paper, Table 2. These are reference values, not results obtained
by this setup. Smoke tests are stored separately under `runs/smoke_*`.

## Logs

- `logs/pipeline-*.log`: full launcher stdout/stderr, including tuning and evaluation.
- `runs/idr/console.log` and `runs/supervised/console.log`: training output, warnings,
  checkpoint messages, and Python tracebacks, also captured when running the Python driver directly.
- Each run's `metrics.jsonl`: every optimizer step, including UTC timestamp,
  session ID, iteration, epoch, batch size, L1 loss, learning rate used/next,
  elapsed time, average step time, throughput, ETA, GPU allocated/peak memory.
- Config, image manifest, and checkpoints remain in each run folder. Checkpoints
  are saved every 1,000 steps and at step 50,000. Resume appends logs; session IDs
  distinguish retries/replayed steps after a checkpoint.

For Fast IDR only, with more frequent console progress:

```bash
.venv/bin/python -u gaussian.py train --method idr --data data/imagenet_val --batch 24 --steps 50000 --log-every 10
```

### Readable terminal progress

Future training processes use an aligned plain-text table by default, with a
configuration banner and columns for step, completion, epoch, loss, LR, speed,
elapsed time, ETA and peak GPU memory. Epoch is displayed starting at 1; the
machine-readable JSON retains its original zero-based epoch. Checkpoints and
IDR target refreshes get short labeled messages. There are no terminal-control
escape sequences, so tee/background log files remain readable.

Use `--console-format json` for the previous JSON progress rows. Per-step
`metrics.jsonl`, optimizer settings, iteration counts and checkpoint format are
unchanged. No dependency updates or GPU tests are required for this display
change, and existing processes retain their already-loaded display code.

## Compact 10k / 128 experiment

A separate user-requested compute-reduced profile is available. This is an
adaptation, not a paper-exact reproduction. Existing/default `paper` runs remain
50k steps, 256x256, with LR drops at 25k/40k.

| Setting | Paper profile | Compact profile |
|---|---|---|
| Optimizer steps per model | 50,000 | 10,000 |
| Clean ImageNet crop | 256x256 | 256x256 |
| Input/target after area downsampling | 256x256 | 128x128 |
| Initial LR | 0.0003 | 0.0003 |
| LR halving milestones | 25,000; 40,000 | 5,000; 8,000 |
| Fast IDR refinement | Every epoch | Every epoch |
| Output folder | runs/{method} | runs/compact10k_128/{method} |

Resize the clean crop BEFORE sampling Gaussian noise; downsampling a noisy
crop would alter the noise model. Batch stays configurable; default compact
launcher batch is 64 to match the current user's run. Increasing batch to fill
all newly available memory can consume the expected time savings and changes
the number of Fast IDR refinements. The total number of refinements is NOT
preserved by compressing the learning-rate schedule.

After the existing job finishes, or in a separately managed GPU slot:

```bash
cd repro
bash run_gaussian_compact.sh data/imagenet_val 64
```

This trains Fast IDR and supervised from scratch, 10k steps EACH, then evaluates
each on the original full-resolution Kodak/BSDS300 tests at sigma 25/50. The
network is fully convolutional; training patch size does not require resizing
the benchmark images. All prior logging/checkpoint features remain enabled.

Individual run / resume:

```bash
.venv/bin/python -u gaussian.py train --profile compact --method idr --data data/imagenet_val --batch 64
.venv/bin/python -u gaussian.py train --profile compact --method idr --data data/imagenet_val --batch 64 --resume
```

Use the same batch for the supervised run. Use `auto` in the compact launcher
to explicitly retune for 128x128; it uses `batch_compact128.json`, separate from
256x256 tuning. Cross-profile resumes are rejected to prevent silently loading
the 50k optimizer schedule into the 10k experiment. CPU regression tests cover
downsampling, legacy dataset behavior, exact LR boundaries, and resume guards:
`.venv/bin/python tests/test_compact.py`.
