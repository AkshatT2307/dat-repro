# Fast IDR (Zhang et al., CVPR 2022): reproduction under the paper's protocol, and adaptation to seismic data

Companion files:
- Full tables and every figure: [RESULTS.md](RESULTS.md) (generated automatically from the run outputs).
- Experiment plan and change log: [../../PAPER_PROTOCOL_PLAN.md](../../PAPER_PROTOCOL_PLAN.md).
- Code: `paperproto/`. Raw outputs: `runs/paperproto/<run>/`.

Status: all 104 jobs are complete, with no failures (wall time 15 h 38 min on one RTX 5070 Ti). A1, A2, C1 and C2 were trained with 3 seeds (§6); every other run uses one seed (seed 0). All numbers are at the final 50k-step checkpoint, and tables outside §6 show seed 0.

---

## 1. What was done

**What we trained.** Only Fast IDR and its supervised upper bound. N2V, DBSN, MPRNet and the paper's Nr2N row are copied from the paper's tables. Two extra baselines were trained, and two were run without training:

| Method | Trained? | Purpose |
|---|---|---|
| Fast IDR, supervised | Yes | The two methods being reproduced |
| N2N (A4/C4) | Yes | Check of the paper's "N2N ≈ supervised" result |
| Nr2N stand-in (A3/C3) | Yes | Ablation; see §2.4 |
| BM3D | No | Image calibration and seismic baseline |
| Classical seismic filters | No | Seismic baselines |

**Training protocol P.** Taken from the paper text; the released YAML instead uses 500k steps with a cosine schedule, which we did not use.
- Upstream `UNet_n2n_un`, batch 4, 256-pixel crops.
- 50k Adam steps at learning rate 3e-4, halved at 25k and again at 40k.
- L1 loss, except masked L2 for binomial, impulse and dead-trace noise.
- Training data: all 50,000 ImageNet validation images, no augmentation.
- Teacher refreshed every 12.5k steps (one epoch).
- Inputs shifted by −0.5 for Gaussian noise only.
- Our trainer is bit-identical to the upstream `fastIDR` code over 5 steps that include 2 teacher refreshes (`paperproto/tests/test_equivalence.py`).

**Image evaluation.**
- Noise is generated exactly as in upstream `test.py`.
- Images are reflect-padded to a multiple of 32, and outputs are rounded to uint8.
- PSNR and SSIM come from skimage.
- Test sets: Kodak (24 images), BSDS300 (all 300) and BSD68 (gray).

**Seismic adaptation.** One seismic analogue was chosen for each of the paper's noise types. σ is in units of the section's RMS amplitude.

| Paper noise | Seismic analogue | Code | Train | Test |
|---|---|---|---|---|
| Gaussian | white Gaussian | sg | σ~U[0,1] | 0.5, 1.0 |
| spatially correlated | band-limited 5–60 Hz along time (in-band, so it cannot be bandpassed away) | sb | σ~U[0,1] | 0.5, 1.0 |
| impulse | erratic spikes ±U[2,8] at rate p | ss | p~U[0,0.2] | 0.05, 0.1 |
| binomial | dead traces | sd | upstream binomial draw per trace | 0.3, 0.5 |

**Seismic data.**
- Training corpus: Marmousi2 NMO stack, SEG/EAGE Overthrust and SEG/EAGE Salt. Overthrust and Salt are velocity models turned into convolutional synthetics: Gardner density, normal-incidence reflectivity, 30 Hz Ricker wavelet, 4 ms sampling.
- Held-out test sections from:
  - the same three surveys (Marmousi2, Overthrust, Salt);
  - three other processing versions of Marmousi2 never seen in training (PreSTM, PostSTM, SRME);
  - two field surveys: F3 Netherlands (marine 3-D) and Stratton (land 3-D).
- Field sections are scored in two ways:
  - semi-synthetically: noise is added and the original section is used as the reference;
  - on their real noise: no reference exists, so we report diagnostics only.
- SNR is the primary seismic metric, because SSIM on Marmousi2 is inflated by the strong seafloor event.

## 2. Image reproduction: findings

### 2.1 Table 2: Gaussian RGB (PSNR in dB)

| | IDR paper | IDR ours | Sup paper | Sup ours | paper IDR−Sup | ours IDR−Sup |
|---|---|---|---|---|---|---|
| Kodak σ25 | 32.36 | 31.89 | 32.39 | 32.08 | −0.03 | −0.19 |
| Kodak σ50 | 29.27 | 28.73 | 29.24 | 28.89 | +0.03 | −0.16 |
| BSDS300 σ25 | 31.48 | 31.15 | 31.40 | 31.30 | +0.08 | −0.14 |
| BSDS300 σ50 | 28.25 | 27.87 | 28.17 | 28.00 | +0.08 | −0.13 |
| BSD68 gray σ25 | 29.20 | 28.84 | 29.20 | 29.04 | 0.00 | −0.20 |
| BSD68 gray σ50 | 26.25 | 25.95 | 26.24 | 26.11 | +0.01 | −0.16 |

- **We reproduce the ranking and are within 0.5 dB of the paper.**
  - IDR is 0.33–0.54 dB below the paper.
  - Supervised is 0.10–0.35 dB below.
  - N2N comes in at 32.02 on Kodak σ25, against the paper's 32.39, and lands next to supervised as in the paper.
- **We do not reproduce "IDR matches supervised".** In our runs IDR trails supervised by 0.13–0.20 dB in every row.
  - The gap is consistent across images. The per-image paired difference is −0.145 dB at σ25, with a 95% bootstrap CI of [−0.153, −0.137].
  - IDR beats supervised on 0 of 324 images at σ25 and on 1 of 324 at σ50.
  - The paper reports IDR −0.03 to +0.08 dB relative to supervised.

### 2.2 Table 5: binomial and impulse noise, p = 0.5

| | IDR paper | IDR ours | Sup paper | Sup ours |
|---|---|---|---|---|
| Kodak binomial | 31.63 | 31.78 | 31.84 | 32.41 |
| Kodak impulse | 34.92 | 35.96 | 35.83 | 34.23 |
| BSDS300 binomial | 32.18 | 31.81 | 32.32 | 31.51 |
| BSDS300 impulse | 37.40 | 35.90 | 37.48 | 33.87 |

- IDR binomial is reproduced to within 0.4 dB. IDR impulse is +1.0 dB on Kodak and −1.5 dB on BSDS300.
- **Unexplained:** our supervised impulse model is 1.6–3.6 dB below the paper. It is also below our own IDR, which should not happen for an upper bound.
- The paper's text and printed table disagree on which row is binomial and which is impulse (§RESULTS 1). Swapping the labels does not fix the supervised number.
- We flag this cell as not reproduced and did not have time to investigate it further.

### 2.3 Table 6: spatially correlated noise (upstream kernels g1–g4)

- 7 of 8 cells reproduce within −0.25 to +1.4 dB of the paper. Six of those are within 0.5 dB, and on g3 we are above the paper.
- **Kodak g1 is the exception: 30.01 against the paper's 37.46.**
  - The same model matches the paper on BSDS300 g1 (29.73 vs 29.83).
  - The paper's own BM3D row is also inconsistent for Kodak g1: 31.56 on Kodak against 26.77 on BSDS300.
  - We believe the paper's Kodak g1 entry is wrong, not our setup.
- g5 is not defined in the released code, so it was not run.
- Upstream "level 5" means a noise **variance** of 5/255, i.e. a standard deviation of about 36/255. We used the upstream definition.

### 2.4 Ablations that support the paper's mechanism

**Refinement matters (Nr2N stand-in, A3).**
- A3 is the Fast IDR network after stage 0 only (input y + n', target y), with no refinement.
- Applied directly to y, it scores 25.00 dB on Kodak σ25.
- Used as a Noisier2Noise estimator (2f(z) − z, with z = y + n' and oracle σ), it scores 29.60 for a single draw and 31.01 averaged over 8 draws.
- Full Fast IDR scores 31.89. So the iterative refinement is responsible for the gain over stage 0.
- The paper's Nr2N (31.96) was trained with its own recipe. Our stand-in is an ablation, not a reproduction of that row.

**Fresh noise vs a fixed noise realisation (A5).**
- The released code draws new noise on every visit to an image, which slightly leaks clean information compared with a fixed noisy dataset.
- With one fixed noisy copy per image, IDR changes by only −0.02 to −0.07 dB on images.
- So this leak does not inflate the image results.

## 3. Why our numbers drift from the paper

**The evaluator is correct.**
- BM3D run through our pipeline matches the paper's BM3D to within 0.08 dB on Kodak and BSD68: 31.90 vs 31.88, and 28.60 vs 28.55.
- So the Kodak and BSD68 gaps come from the trained models, not from the metric.

**Inference-only factors**, measured on the A1 and A2 checkpoints (mean change over the four RGB rows):

| Factor | ΔPSNR | Verdict |
|---|---|---|
| Test-noise seed (9 extra seeds), as std | 0.003 dB | not a cause |
| RGB vs BGR channel order | −0.02 to −0.05 | negligible |
| PSNR in float instead of uint8 | +0.007 | negligible |
| Clipping the noisy input to [0,1] | −0.83 to −0.89 | not what the paper did; would make things worse |
| Gaussian-window SSIM | SSIM −0.001, PSNR unchanged | negligible |
| BSDS300 test-100 split instead of all 300 | −0.36 | moves us further from the paper |

**Training-side factors**, using the same evaluator on earlier checkpoints (Kodak σ25, IDR / supervised):

| Factor | ΔPSNR | Notes |
|---|---|---|
| Batch 192, epoch-based schedule (the earlier repro) | **−1.37 / −0.98** | the main cause of the earlier repro's drift; protocol P fixes it |
| 128-pixel downsampled crops vs 256-pixel crops | −0.11 / −0.18 | the earlier repro used 128 |
| 1k-image holdout (49k vs 50k training images) | −0.03 / +0.01 | negligible |
| TF32 vs FP32 matmuls (A1 FP32 control) | +0.04 / — | FP32: 31.85 / 28.73 / 31.14 / 27.87; not a cause |
| Fresh vs fixed training noise (A5) | +0.02 to +0.07 | small |

**What remains.**
- After all corrections, IDR is still about 0.45 dB and supervised about 0.3 dB below the paper on Kodak.
- Nothing we could test at inference time, or in precision and data handling, closes this.
- Remaining candidates:
  1. ~~Seed-to-seed variance~~: ruled out. Three training seeds differ by at most 0.03 dB (§6).
  2. Training details the paper does not state. The released configuration trains for 500k steps with a cosine schedule rather than the 50k stated in the paper, and longer training is the most plausible source. Per the project decision, it was not tested here.
- Because supervised is also short of the paper by about 0.3 dB on Kodak, most of the gap is shared by both methods; it is not specific to IDR. The extra 0.15 dB that IDR loses relative to supervised is what we were unable to explain.

**The paper's BSDS300 numbers are not comparable to ours.**
- Our BM3D is **+0.56 dB above the paper's on BSDS300** at both σ, while it matches on Kodak and BSD68.
- So the paper evaluated BSDS300 differently from all-300 colour images.
- It was not the test-100 split, which would move us down by 0.36 dB, not up.
- Our BSDS300 gaps should therefore be read relative to BM3D, not as absolute numbers.

## 4. Seismic adaptation: findings

SNR out in dB at σ = 0.5 (or p = 0.05 / 0.3). Input SNR is about 6 dB. Full tables, including σ = 1.0, are in RESULTS §4.

| Test section | sg IDR / Sup | sb IDR / Sup | ss IDR / Sup | sd IDR / Sup | best classical method (sg) |
|---|---|---|---|---|---|
| Marmousi2 (seen) | 19.31 / 19.34 | 19.13 / 18.78 | 18.24 / 21.70 | 21.18 / 24.03 | BM3D 19.12 |
| Overthrust | 19.00 / 19.95 | 17.26 / 18.06 | 30.65 / 35.07 | 19.51 / 19.69 | BM3D 16.47 |
| Salt | 23.21 / 24.14 | 22.30 / 23.51 | 24.34 / 27.82 | 14.38 / 15.58 | BM3D 20.89 |
| Marmousi2 PreSTM (unseen) | **17.37** / 15.75 | **16.13** / 14.43 | 32.30 / 36.85 | 28.13 / 28.52 | BM3D 16.38 |
| Marmousi2 PostSTM (unseen) | **17.08** / 15.43 | **15.82** / 14.37 | 32.26 / 36.84 | 27.72 / 28.14 | BM3D 16.34 |

1. **Self-supervised training works on seismic.** With white or band-limited noise, Fast IDR comes within 0.0–1.2 dB of supervised on the training surveys. It beats every classical baseline on the synthetic surveys: bandpass, f-x deconvolution, and BM3D given the oracle σ or noise spectrum. The margin over BM3D ranges from +0.2 dB (Marmousi2, white noise, σ=0.5) to +4.7 dB (Salt, band-limited noise, σ=1.0), and over f-x deconvolution it is about +5–14 dB.
2. **IDR generalises better than supervised to data processed differently from its training data.** On the unseen Marmousi2 processing versions, IDR beats supervised by 0.75–1.7 dB for both white and band-limited noise. On the training surveys, supervised leads by up to 1.2 dB. The ranking holds on all three unseen versions at both noise levels (RESULTS §4).
3. **The gap is larger for spikes and, on two surveys, for dead traces.** Spikes: supervised leads by 3.5–5.1 dB on the synthetic surveys, while the two are equal on the field sections. Dead traces: supervised leads by 2.9 dB on Marmousi2 and 1.2 dB on Salt; elsewhere the two are within about 0.5 dB. This matches the paper's experience with impulse and binomial noise, where IDR targets are weaker. Both methods are still far above the median filter and f-x deconvolution for spikes, and above trace interpolation for dead traces on all synthetics.
4. **Fresh noise matters when data is scarce.**
   - With a fixed noisy training copy, Marmousi2, which has a single training section, loses 2.5 dB for white noise (C5) and 1.9 dB for band-limited noise (C6).
   - Overthrust and Salt, with hundreds of sections, are unchanged.
   - So the fresh noise draw of the released code helps on small surveys. On images (A5) it was negligible.
5. **The training noise model must match.**
   - A white-noise model tested on band-limited noise collapses: about 3 dB out at σ = 1, compared with 16 dB for the band-limited model.
   - A band-limited model handles white noise almost as well as a white-noise model (16.5 vs 17.3 dB).
   - For seismic work, the band-limited (correlated) noise model is the safer choice.
6. **The Nr2N and N2N ordering matches the paper.**
   - IDR beats the Nr2N estimator: 19.31 vs 17.48 for one draw and 18.65 for 8 draws on Marmousi2.
   - N2N, which uses a second independent noisy copy that real data rarely has, is close to supervised.

## 5. Field data

**Semi-synthetic test** (white noise added to field sections; the original section is the reference):

| Model | F3 σ0.5 / σ1.0 | Stratton σ0.5 / σ1.0 |
|---|---|---|
| IDR trained on that survey's own sections (noise added during training) | **14.97 / 11.41** | **13.30 / 9.62** |
| Supervised on that survey (oracle) | 14.69 / 11.04 | 13.43 / 9.66 |
| BM3D, oracle σ | 13.84 / 10.04 | 12.13 / 8.42 |
| IDR trained on the synthetic corpus (3 seeds) | 10.16±0.48 / 7.07±0.24 | 9.88±0.23 / 6.64±0.25 |
| Supervised on the synthetic corpus (3 seeds) | 10.45±0.16 / 7.29±0.07 | 10.18±0.10 / 6.89±0.02 |

- Models trained on synthetic data lose 3–4 dB on field data because of the domain gap, and fall below BM3D.
- **Fast IDR trained directly on the target survey's sections**, which needs no clean data, recovers this. It matches the oracle supervised model and beats BM3D by about 1.1–1.4 dB.
- This is the most useful practical result for seismic: train self-supervised on the survey itself.

**Real field noise** (models applied to raw sections; no reference):

| Model | F3: % energy removed | F3: leakage (local similarity) | Stratton: % removed | Stratton: leakage |
|---|---|---|---|---|
| IDR trained on raw field data | 0.02 | 0.12 | 0.04 | 0.05 |
| IDR, synthetic corpus | 1.09 | 0.31 | 0.90 | 0.36 |
| Supervised, synthetic corpus | 1.52 | 0.26 | 0.73 | 0.34 |

- **This is a negative result.**
  - Trained on the raw field data with additive noise calibrated to the robust MAD noise estimate (F2/F4), IDR leaves the sections almost unchanged.
  - These post-stack, migrated volumes contain little random noise.
  - Their remaining noise is coherent (migration artefacts, acquisition footprint), and an additive-noise self-supervised model does not regard it as noise.
- The synthetic-trained models do remove about 1% of the energy, but much of it is signal:
  - Local similarity between output and removed component is 0.26–0.36, against 0.05–0.12 for the field-trained IDR.
  - Reflectors are clearly visible in the removed-noise panels of `figs/field_f3.png`.
  - Local similarity does not depend on amplitude, so it is unreliable when almost nothing is removed. For example, the fixed-copy Stratton IDR removes 0.03% of the energy yet scores 0.51.
- So on these two field surveys we see no evidence of real-noise removal. The honest claim is limited to semi-synthetic noise.

## 6. Training-seed variability (seeds 0, 1, 2; mean ± sd)

**Images (A1 IDR / A2 supervised), PSNR in dB:**

| | IDR | Sup | IDR−Sup (per seed) | Paper IDR / Sup |
|---|---|---|---|---|
| Kodak σ25 | 31.91 ± 0.01 | 32.09 ± 0.01 | −0.19 / −0.17 / −0.17 | 32.36 / 32.39 |
| Kodak σ50 | 28.73 ± 0.01 | 28.91 ± 0.01 | −0.16 / −0.19 / −0.18 | 29.27 / 29.24 |
| BSDS300 σ25 | 31.16 ± 0.00 | 31.30 ± 0.01 | −0.14 / −0.12 / −0.15 | 31.48 / 31.40 |
| BSDS300 σ50 | 27.87 ± 0.01 | 28.01 ± 0.01 | −0.13 / −0.14 / −0.14 | 28.25 / 28.17 |

- With the paper's protocol, training is extremely repeatable: sd ≤ 0.013 dB.
- The gap to the paper (IDR about −0.45, supervised about −0.3 on Kodak) is 30–40 times the seed sd. It is a systematic difference, not luck.
- IDR is below supervised in every seed and every row, by 0.12–0.19 dB. "IDR ≈ supervised" is not reproduced.

**Seismic, white noise (C1 IDR / C2 supervised), SNR out in dB at σ = 0.5:**

| Test section | IDR | Sup | IDR−Sup, mean [min, max] over seeds |
|---|---|---|---|
| Marmousi2 | 19.47 ± 0.15 | 19.46 ± 0.09 | +0.02 [−0.05, +0.12] |
| Marmousi2 PreSTM (unseen) | 17.37 ± 0.04 | 15.67 ± 0.17 | **+1.70 [+1.50, +1.99]** |
| Marmousi2 PostSTM (unseen) | 17.08 ± 0.05 | 15.30 ± 0.23 | **+1.78 [+1.51, +2.16]** |
| Marmousi2 SRME (unseen) | 18.52 ± 0.03 | 17.62 ± 0.08 | **+0.90 [+0.84, +0.97]** |
| Overthrust | 19.07 ± 0.06 | 19.96 ± 0.01 | −0.88 [−0.95, −0.80] |
| Salt | 23.24 ± 0.02 | 24.42 ± 0.20 | −1.18 [−1.32, −0.93] |
| F3 field (noise added) | 10.16 ± 0.48 | 10.45 ± 0.16 | −0.29 [−0.70, +0.34] |
| Stratton field (noise added) | 9.88 ± 0.23 | 10.18 ± 0.10 | −0.30 [−0.59, −0.04] |

- **IDR's advantage on unseen processing holds in all three seeds:** +0.84 to +2.16 dB at σ = 0.5, and +0.72 to +1.45 dB at σ = 1.0.
- **Supervised's advantage on Overthrust and Salt is equally consistent:** 0.5–1.3 dB.
- **On the field sections**, synthetic-trained IDR is slightly below supervised on average and varies most between seeds (sd up to 0.48 dB). The single-seed claim from §4/§5 that the two are "about equal" should be read as "within seed noise, IDR slightly lower".

## 7. Claims, and how strong each is

| Claim | Support |
|---|---|
| Fast IDR's ranking against BM3D, the Nr2N stand-in and its own stage-0 network reproduces under the paper's protocol | strong (Tables 2, 5, 6; ablations) |
| Absolute IDR PSNR is reproduced to within about 0.5 dB on Kodak and BSD68 | strong; the gap is measured and the evaluator is verified |
| "IDR ≈ supervised" is **not** reproduced; IDR is 0.12–0.19 dB below in every row and seed, and on 647 of 648 seed-0 image/σ pairs | strong (3 seeds, seed sd ≤ 0.013 dB) |
| Earlier repro drift was caused mainly by the batch-192 epoch schedule and 128-pixel crops | strong (same evaluator, controlled checkpoints) |
| The remaining 0.3–0.5 dB gap is not from evaluation, precision, data holdout, the noise-sampling leak or training-seed variance | strong, by elimination (3 seeds) |
| The paper's BSDS300 evaluation differs from all-300 colour images; the paper's Kodak g1 entry looks wrong | moderate (BM3D cross-check) |
| On seismic, Fast IDR comes within about 1.2 dB of supervised for additive noise, and beats it by 0.8–1.8 dB on unseen processing | strong for white noise (3 seeds); single seed for the other noise types |
| For field seismic, self-supervised training on the survey itself beats synthetic-trained models and BM3D | moderate, semi-synthetic only |
| Real field-noise removal is not demonstrated | negative result |

## 8. Limitations

- **Single training seed** for everything except A1, A2, C1 and C2 (§6). Image seed sd is ≤ 0.013 dB; seismic seed sd is up to 0.5 dB on field sections.
- **TF32 matmuls**, checked against one FP32 control run.
- **500k-step schedule not run.** The paper's 50k protocol was used, per the project decision.
- **Table 6 coverage.** g5 is undefined upstream and not run. The correlated-noise level follows the upstream variance convention.
- **Anomalies left open:**
  - supervised impulse model (§2.2);
  - Kodak g1 (§2.3).
- **Nr2N** is an ablation built from the IDR stage-0 network, not the paper's Nr2N recipe.
- **Seismic test sets:**
  - The Marmousi2 test region is a single 500-trace section (its dead edge traces are excluded), scored over 3 noise seeds.
  - Overthrust, Salt, F3 and Stratton have 8–10 sections each.
- **The field "clean" reference** is the field section itself, so semi-synthetic SNR measures removal of the added noise only.
- **SSIM on Marmousi2** is dominated by the seafloor event and is not used for conclusions.
