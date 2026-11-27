# Plan: paper-protocol reproduction of Fast IDR + seismic adaptation

Written 2026-11-19, before any code or runs for this phase. Everything below is
what will be executed; deviations discovered while running will be appended to
the "Change log" at the bottom rather than silently edited.

## 0. Goals

1. **Reproduce** the paper's Fast IDR and supervised (Noise2Clean) results under
   the paper's exact stated training protocol for every synthetic noise type that
   also has a meaningful seismic analogue: Gaussian (RGB and gray), binomial,
   impulse, spatially correlated. Other methods (BM3D, N2V, Nr2N, DBSN, N2N,
   MPRNet) are quoted from the paper tables; two of them (Nr2N, N2N) are also
   re-run once because they are free ablations of the same code and calibrate the
   pipeline.
2. **Explain the drift** between our numbers and the paper's numbers with
   controlled training and inference-only experiments.
3. **Adapt to seismic**: the same algorithm and training protocol on synthetic
   seismic with clean references (Marmousi2, SEG/EAGE Overthrust, SEG/EAGE Salt),
   with seismic analogues of each noise type, and on field data (marine F3,
   and a land field dataset if one is downloadable) where no clean data exists.
4. A results dump report with tables, figures, spatial- and frequency-domain
   diagnostics.

Raw-camera experiments (SID, SenseNoise-500, ELD noise model) are out of scope: the
noise model is sensor photon/read noise and has no seismic analogue.

## 1. Training protocol "P" (identical for every training run)

From Section 5.1 of the paper, plus the released code where the paper is silent:

| Item | Setting | Source |
|---|---|---|
| Network | Upstream `UNet_n2n_un`, unchanged; in/out channels = 3 (RGB) or 1 (gray, seismic) | paper + code |
| Optimiser | Adam, LR 3e-4, betas (0.9, 0.999), wd 0 | paper + yaml |
| Schedule | LR halves after steps 25,000 and 40,000 | paper (yaml's 500k cosine is NOT used) |
| Budget | exactly 50,000 updates, batch 4, no accumulation | paper |
| Loss | L1 (Gaussian, correlated, seismic Gaussian/band-limited/spike); masked L2 for binomial-type noise (binomial, impulse, seismic dead traces), "as recommended in N2N" | paper §A + N2N |
| Training data (images) | **All 50,000** ImageNet validation images (no holdout), random 256x256 crops, images smaller than 256 resized (upstream rule), no augmentation | paper + code |
| Fast IDR | Epoch 0: target = noisy x, input = x + n'. At the start of every later epoch the student is copied into a frozen teacher; target = teacher(x), input = target + n'. Epoch = 12,500 steps (50k samples / batch 4) -> refreshes before steps 12,501 / 25,001 / 37,501 | Algorithm 1 + code |
| Noise sampling | upstream `add_noise`: one sigma/p per batch per call, fresh noise every visit | code |
| zero_mean (-0.5 shift) | Gaussian only (train and test), as upstream `test.py` | code |
| Checkpoint reported | **final step 50,000**. No validation, no selection | paper |
| Extra saves | resumable state every 1k steps; weight snapshots every 10k steps (analysis only) | ours |
| Numerics | FP32, TF32 off, deterministic cuDNN (paper used PyTorch 1.5, pre-TF32) | ours |
| Seeds | 0 (all runs); 1 and 2 for the central Gaussian RGB / seismic Gaussian pairs | ours |

Measured cost: about 38 ms/step on the RTX 5070 Ti, so about 30 GPU-minutes per run.
The GPU is saturated by one run, so parallelism only overlaps CPU work (decoding,
correlated-noise synthesis). The scheduler runs 4 jobs at once.

## 2. Image experiments (paper tables 2, 5, 6, 8)

| ID | Noise (train -> test) | Methods | Seeds | Test sets | Paper ref |
|---|---|---|---|---|---|
| A1 | Gaussian RGB, sigma U[0,50] -> 25, 50 | Fast IDR | 0,1,2 | Kodak (24), BSDS300 (300) | Tab. 2 |
| A2 | same | Supervised | 0,1,2 | same | Tab. 2 |
| A3 | same | Nr2N (= IDR, refinement off) | 0 | same | Tab. 2 Nr2n col. + mechanism ablation |
| A4 | same | N2N (upstream `DenoiseBase_n2n`) | 0 | same | Tab. 2 N2N col. |
| A5 | same, **one fixed noisy realisation per image** | Fast IDR | 0 | same | "single noisy images" setting |
| A6 | Gaussian gray, U[0,50] -> 25, 50 | IDR, Sup | 0 | BSD68 | Tab. 2 gray rows |
| A7 | Binomial, p U[0,0.95] -> 0.5 | IDR, Sup | 0 | Kodak, BSDS300 | Tab. 5 |
| A8 | Impulse, p U[0,0.95] -> 0.5 | IDR, Sup | 0 | Kodak, BSDS300 | Tab. 5 |
| A9 | Correlated g1..g4 (upstream `scn-k`, level 5) | IDR | 0 | Kodak, BSDS300 | Tab. 6 |
| A10 | BM3D / CBM3D, no training (CPU) | BM3D | - | Kodak, BSDS300, BSD68 | Tab. 2 BM3D col. |
| A11 | Gaussian RGB, **budget probe** 200k steps, cosine | IDR, Sup | 0 | Kodak, BSDS300 | released yaml (500k cosine); last wave, only if time allows |

The paper's g5 kernel (Table 6) is not defined in the released code, so only
g1 to g4 are reproduced.

## 3. Drift analysis (why our numbers differ from the paper's)

Inference-only, applied to the A1/A2 checkpoints and the existing runs:

| ID | Factor | Test |
|---|---|---|
| D1 | Colour order | RGB vs BGR into the network (released `test.py` feeds BGR) |
| D2 | Test-noise realisation | 10 evaluation noise seeds -> spread of mean PSNR |
| D3 | Quantisation | uint8-rounded vs float output PSNR; clipping noisy input |
| D4 | SSIM definition | skimage default vs Gaussian-weighted (Wang 2004) vs Y-channel |
| D5 | Test-set definition | BSDS300 all 300 vs test 100; image decoding path |
| D9 | Evaluator calibration | our BM3D (A10) vs the paper's BM3D column: if it matches, the evaluator is right and the drift is training-side |

Training-side factors:

| ID | Factor | Comparison |
|---|---|---|
| D6 | Training seed | A1/A2 seeds 0 to 2 |
| D7 | Data protocol | 49k holdout + selection (existing `steps50k_b4_256`) vs full 50k (A1); 128 vs 256 crops (existing); large-batch epoch runs (existing) |
| D8 | Training budget | PSNR vs step from the 10k snapshots; A11 4x budget |
| D10 | Fresh vs fixed noise | A5 vs A1 (does fresh noise per visit leak clean information?) |

The output is an attribution table of the gap to the paper per factor, plus
the authors' pretrained weights if they can be downloaded (Baidu link, probably not).

## 4. Seismic adaptation

### 4.1 Data (all float amplitudes at 4 ms; each section scaled to unit RMS over live samples)

| Key | Survey | Type | Clean ref. | Use |
|---|---|---|---|---|
| M2 | AGL Elastic Marmousi2: NMO stack (+ PreSTM, NMO-SRME as test-only) | marine 2D, elastic FD synthetic, processed | yes | train/test; dead traces (0-239, 2396-2720) removed; train traces [240,1640), 256-trace guard, test [1896,2396) |
| OT | SEG/EAGE Overthrust 3D | land 3D synthetic (thrust belt) | yes | convolutional modelling: Gardner density, normal-incidence reflectivity, depth->time, 25 Hz Ricker; inlines 0-559 train, 640-800 test |
| SA | SEG/EAGE Salt 3D | marine 3D synthetic (salt body) | yes | same modelling; inlines 0-469 train, 540-675 test |
| F3 | Netherlands F3 block | marine 3D **field** | no | self-supervised IDR on the field data itself; no-reference evaluation; semi-synthetic test |
| LF | Stratton 3D, South Texas (BEG/UT Austin via SEG open data; 100 inlines x 200 xlines, 2 ms, post-stack migrated) | land 3D **field** | no | as F3 |
| VG | Mobil Viking Graben line 12 | marine 2D field prestack | no | optional, qualitative |

Training corpus: train regions of M2 + OT + SA, sampled with equal probability
per survey, random 256x256 crops. One epoch is 50,000 crops (12,500 steps), the same
refresh cadence as ImageNet.

### 4.2 Seismic analogues of the paper's noise types (sigma in units of section RMS)

| Paper noise | Seismic analogue | Train | Test |
|---|---|---|---|
| Gaussian | SG: white random noise | sigma U[0,1] | 0.5, 1.0 (input SNR about 6 and 0 dB) |
| Correlated | SB: band-limited random noise, 5-60 Hz (inside the signal band, so bandpass cannot remove it) | sigma U[0,1] | 0.5, 1.0 |
| Impulse | SS: erratic spikes, added amplitude +/-U[2,8] RMS | p U[0,0.2] | 0.05, 0.10 |
| Binomial | SD: dead traces (whole traces zeroed) | p U[0,0.95] | 0.3, 0.5 |

### 4.3 Seismic runs (protocol P, single-channel U-Net)

| ID | Noise | Methods | Seeds |
|---|---|---|---|
| C1/C2 | SG | IDR / Sup | 0,1,2 |
| C3 | SG | Nr2N | 0 |
| C4 | SG | N2N | 0 |
| C5 | SG, one fixed noisy realisation per training section (the honest self-supervised setting) | IDR | 0 |
| C6 | SB, fixed realisation | IDR | 0 |
| C7 | SB | IDR, Sup | 0 |
| C8 | SS | IDR, Sup | 0 |
| C9 | SD | IDR, Sup | 0 |
| C10 | F3 field, noise model calibrated from data (sigma and spectrum estimated from a noise-dominated band/window) | IDR (self-supervised) | 0 |
| C11 | land field (if available) | IDR | 0 |

Classical seismic baselines (CPU, no training): Butterworth bandpass (oracle band),
f-x deconvolution (Canales), BM3D (oracle sigma; PSD for SB), median filter (SS),
trace interpolation (SD).

Inference-only seismic studies:
- **Noise-model mismatch**: SG-trained models on SB noise and the reverse; sigma outside the training range.
- **Cross-dataset generalisation**: per test survey, and M2 PreSTM/SRME (unseen processing).
- **Field domain shift**: supervised model trained on synthetics vs IDR trained on the field data, applied to F3.

Metrics: SNR, PSNR (peak = max|clean|), SSIM, RMSE over live samples of whole
test sections (not tile means), mean over 5 noise seeds. Also local similarity
between removed noise and output (signal leakage, Chen & Fomel 2015). For field
data: local similarity, spectra, visual.

## 5. Visualisations

- Images: Fig. 1-style IDR target evolution per refresh; zoom panels with error maps;
  radially averaged power spectra of output and error; removed-noise vs true-noise maps;
  per-image paired PSNR scatter (IDR vs Sup); seed spread; drift waterfall; PSNR-vs-step curves.
- Seismic: variable-density sections (clean, noisy, IDR, Sup, baselines); difference
  sections; **f-k spectra** of clean/noisy/denoised/removed; average amplitude spectra;
  trace overlays (wiggles); local-similarity maps; SNR-out vs SNR-in curves; mismatch
  heat-map; IDR target evolution on seismic; field before/after/removed with f-k.

## 6. Execution

- Code: `repro/paperproto/` (new; existing scripts untouched). Driver re-implements
  the upstream Fast IDR / N2C / N2N `preprocess` logic with pluggable noise, and is
  unit-tested for bit-identity against the upstream trainer classes on Gaussian noise.
- Outputs: `repro/runs/paperproto/<id>/`; report and figures in `repro/reports/paperproto/`.
- Scheduler: `paperproto/queue.py` runs the job list 4 at a time with auto-resume,
  under tmux session `pp`. A monitoring sub-agent watches it, restarts failures and logs status.
- Order: wave 1 = A1/A2 s0, C1/C2 s0, A5, C5, A3, C3; wave 2 = seeds 1-2, A6-A8, C7-C9, C6, A4, C4;
  wave 3 = A9, C10/C11; wave 4 (optional) = A11. The CPU baselines (A10, seismic) run alongside.
- Expected: about 37 training runs at about 30 GPU-min each, so about 19 h wall time at
  saturation, plus 4 h for the optional A11.

## Change log
- (empty)
- 2026-11-19 (user: minimise total time): TF32 enabled with deterministic cuDNN (27.8 vs 38 ms/step).
  The training algorithm is unchanged. One FP32 control run (A1 seed 0, `PP_FP32=1`) quantifies the precision effect.
- 2026-11-19: correlated noise (A9) is generated on GPU: same kernels (upstream `get_experiment_kernel`),
  same padding/crop, independent RNG stream. Verified: std within 0.2%, PSD within sampling error
  (`paperproto/tests/test_correlated.py`). Upstream "level 5" means variance 5/255, i.e. std about 0.14.
- 2026-11-19: trainer verified bit-identical to upstream fastIDR/n2c/n2n (`tests/test_equivalence.py`).
- 2026-11-19: zero_mean only for Gaussian (as upstream test.py); binomial/impulse/SD use masked L2.
  Upstream binomial/impulse training draws drop prob = U(0,1)*U(0,0.95) per pixel (not a uniform p), as coded.
- 2026-11-19: field "real noise" runs (F3, Stratton): the field data is the noisy input itself.
  The IDR noise model is white Gaussian with sigma ~ U[0.5, 1.5] x MAD estimate (Donoho).
- 2026-11-19 22:30 IST (user): removed the extra training seeds (A1/A2/C1/C2 seeds 1-2) and the A11
  long-budget probe to finish sooner. All results are single-seed (seed 0); test-noise seed spread is still reported.
  The training-budget drift question relies on the 10k-50k snapshot curve only.
- 2026-11-20 02:05 IST: Nr2N evaluation fixed. A3/C3 checkpoints were scored by applying f to y directly (= IDR stage-0 net, 25.0 dB Kodak25), which is not the Noisier2Noise estimator. Added paperproto/nr2n_correct.py: z = y + n' (oracle σ, α=1), x̂ = 2f(z) − z; K=1 (paper estimator) and K=8 average. report.py now reports both plus the uncorrected numbers as an ablation. Inference-only, no retraining.
