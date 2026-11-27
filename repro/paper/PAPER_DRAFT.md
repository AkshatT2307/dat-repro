# Self-Supervised Seismic Random-Noise Attenuation by Iterative Data Refinement: Reproduction of Fast IDR and Adaptation to Synthetic and Field Seismic Data

_Rough draft, 27 Nov 2026. Structure follows the DAT-102 brief (IEEE TGRS style). It will be ported to the Overleaf IEEEtran template later. `[TODO-K]` marks results that come from the Kaggle runs (per-dataset synthetic models and 2D field lines). `[FIG]` marks figures, which already exist in `reports/paperproto/figs/` unless marked new._

---

## Abstract

Deep denoisers trained with supervision need clean targets, and seismic field data never has them.
- **Method.** We study Iterative Data Refinement (IDR) [1], a self-supervised scheme. A network trained on noisier-noisy pairs is repeatedly used to refine its own training targets.
- **Reproduction.** We first reproduce the Fast IDR image results under the paper's stated protocol (50k Adam steps, batch 4, 256×256 crops).
  - The method ranking is recovered: IDR above BM3D, above a Noisier2Noise-style stage-0 estimator, and close to supervised.
  - Absolute PSNR is within 0.33–0.54 dB of the paper.
  - The claim that IDR matches supervised training is **not** reproduced: the gap is 0.12–0.19 dB in every row and seed.
  - Most of the earlier drift is traced to batch size, crop size and schedule.
- **Seismic adaptation.** We define a seismic analogue for each of the paper's noise models: white, band-limited (5–60 Hz), erratic spikes and dead traces. We train on three SEG synthetic models (Marmousi2, SEG/EAGE Overthrust, SEG/EAGE Salt), both jointly and per model `[TODO-K]`.
  - IDR comes within 0–1.2 dB SNR of supervised training.
  - It beats bandpass filtering, f-x deconvolution and oracle-σ BM3D.
  - It **outperforms** supervised training by 0.8–2.2 dB on Marmousi2 processing versions not seen during training.
- **Field data.** We apply the method to four open field datasets: 2D land (USGS NPR-A Alaska), 2D marine (USGS NAMSS, Gulf of Mexico), 3D land (Stratton) and 3D marine (F3 Netherlands).
  - Training IDR on the survey itself beats BM3D by 1.1–1.4 dB and matches an oracle supervised model on semi-synthetic tests.
  - Removal of the real, mostly coherent residual noise in post-stack data is not demonstrated. We analyse why.

**Index Terms:** seismic denoising, self-supervised learning, iterative data refinement, U-Net, reproducibility.

---

## I. Introduction

- **Problem.** Random and erratic noise attenuation is a core step in seismic processing.
  - Classical tools (f-x deconvolution [7], bandpass, median filtering, BM3D [6]) need parameter tuning, and they trade signal leakage against noise left behind.
- **Learning-based denoising.** Supervised CNN denoisers need clean/noisy pairs. For field data, the clean target does not exist. Synthetic training data carries a domain gap, which we quantify in Sec. VI: a 3–4 dB loss on field data.
- **Self-supervised alternatives.**
  - Noise2Noise [2] needs two independent noisy copies.
  - Blind-spot methods (Noise2Void [3], Noise2Self [4]) assume pixel-independent noise. Seismic noise is often trace- or band-correlated, which breaks that assumption.
  - Noisier2Noise [5] adds synthetic noise to the noisy data and corrects for it at inference.
- **IDR [1]** starts from the Noisier2Noise setting. It then iteratively replaces the noisy target with the network's own denoised estimate, which shrinks the train–test gap at each round. Fast IDR folds the rounds into a single training run by refreshing a frozen teacher once per epoch.
- **Contributions.**
  1. A protocol-faithful reproduction of Fast IDR, with a quantitative attribution of the drift between our numbers and the paper's (Sec. V-A).
  2. Seismic analogues of the paper's four noise families, and training on three SEG synthetic models, both jointly and per model.
  3. A cross-processing generalisation test on Marmousi2, where self-supervised beats supervised training.
  4. Application to one 2D-land, one 2D-marine, one 3D-land and one 3D-marine field dataset, including an honest negative result on real-noise removal.
  5. Released code, data recipes with SHA-256 checksums, and all trained weights.

## II. Methods of Analysis

### A. Problem setting
- Observed data is y = x + n, where x is the clean signal and n is noise drawn from a known *family* whose parameters are unknown.
- A denoiser f_θ is trained from y alone.

### B. Noisier2Noise stage (IDR stage 0)
- Draw a fresh synthetic noise n′ from the same family and form z = y + n′.
- Train f_θ(z) → y. The network learns to remove n′. Applied to y, it therefore under-denoises.
- The Nr2N estimator 2f(z) − z corrects for this. We use it only as an ablation (Sec. V-A).

### C. Iterative data refinement
- At round k, the target is t_k = f_{θ_{k−1}}(y) and the input is t_k + n′.
- As t_k approaches x, the training pairs approach the supervised pairs (x + n′, x).
- **Fast IDR:** within one run, the student is copied into a frozen teacher at every epoch boundary. One epoch is 12,500 steps, so there are 4 refreshes in 50k steps.
- Loss: L1, or masked L2 for drop-type noise (binomial, impulse, dead traces), following [2].

### D. Training protocol P (as stated in [1])

| Setting | Value |
|---|---|
| Batch / crop | 4 / 256×256 random crops |
| Optimiser | Adam, lr 3×10⁻⁴, halved at 25k and 40k steps, 50k steps total |
| Image training data | 50k ImageNet validation images |
| Seismic training data | 50k random crops per epoch |
| Implementation check | trainer is bit-identical to the official code over 5 steps including 2 teacher refreshes |

The released config's 500k-step cosine schedule is not used (Sec. VIII).

### E. Seismic noise models

All amplitudes are in units of the section's RMS over live samples.

| Image noise in [1] | Seismic analogue | Train | Test |
|---|---|---|---|
| Gaussian | white Gaussian (sg) | σ~U[0,1] | 0.5, 1.0 |
| spatially correlated | band-limited 5–60 Hz, zero-phase 4th-order Butterworth along time (sb) | σ~U[0,1] | 0.5, 1.0 |
| impulse | erratic spikes ±U[2,8] at rate p (ss) | p~U[0,0.2] | 0.05, 0.10 |
| binomial | dead traces (sd) | upstream binomial draw per trace | 0.3, 0.5 |

### F. Metrics and baselines
- **Metrics:** SNR (primary), PSNR, SSIM, RMSE, and a signal-leakage measure: mean |local similarity| [8] between the output and the removed component, in a 21×21 window.
- **Test noise:** 3 test-noise seeds per section, deterministic.
- **Baselines:**
  - bandpass;
  - f-x deconvolution;
  - BM3D with oracle σ, or with the oracle noise PSD for band-limited noise;
  - a 5×3 median filter for spikes;
  - oracle trace interpolation for dead traces.

## III. Network Architecture

- **Network:** the upstream `UNet_n2n_un`, the Noise2Noise U-Net [2], [9].
  - **Encoder:** five stages of 3×3 conv (48 channels) with LeakyReLU(0.1) and 2× max-pooling.
  - **Decoder:** four stages of nearest-neighbour upsampling, concatenation with the skip connection, and two 3×3 convs (96 channels).
  - **Output head:** the input is concatenated back in, followed by 64→32→C_out convs, with no output activation.
- **Size:** about 0.99 M parameters for 1→1 (seismic/gray). The overall downsampling factor is 32, so inputs are reflect-padded to a multiple of 32.
- **Inference:** whole sections, with no tiling.
- **Seismic:** the network is used unchanged, with 1 input and 1 output channel. No architectural change was needed, so any gain comes from the training scheme alone.
- `[FIG]` Block diagram of the U-Net and of the Fast-IDR teacher/student loop (new figure).

## IV. Data

### A. Image benchmarks (reproduction)
- **Training:** ImageNet val [10].
- **Testing:** Kodak (24 images), BSDS300 (all 300) [11] and BSD68 (gray) [12].
- **Test noise:** as in the official `test.py`.

### B. Synthetic seismic (SEG wiki models)

| Model | Type | Preparation | Train / test |
|---|---|---|---|
| Marmousi2 [13] | 2D elastic, marine | released NMO stack, dt 4 ms, dead edge traces excluded | 1 section 1001×1400 / 1 section 1001×500 (traces 1896–2395) |
| Marmousi2 PreSTM, PostSTM, SRME | other processing versions | test region only | unseen-processing test |
| SEG/EAGE Overthrust [14] | 3D land model | convolutional modelling: Gardner density, normal-incidence reflectivity, 30 Hz Ricker, dt 4 ms | 280 / 9 inlines (498×801) |
| SEG/EAGE Salt [14] | 3D marine model | same modelling | 235 / 10 sections (788×676) |

Two configurations are used:
- **Joint:** one model trained on all three synthetics.
- **Per-dataset:** one model per synthetic `[TODO-K]`, required by the brief.

### C. Field data

| Category | Dataset | Status |
|---|---|---|
| 2D land | USGS NPR-A Alaska regional line 1 (OFR 00-286) [20]: stacked and migrated, dt 4 ms, 1501×4751. Train traces 0–3324; 256-trace guard; 2 test sections of 512 traces | done |
| 2D marine | USGS NAMSS survey W-34-82-MB, line WS82-611 (Gulf of Mexico, 1982) [21]: migrated, dt 4 ms, 1799×4882. Train traces 0–3416; 256-trace guard; 2 test sections of 512 traces | done |
| 3D land | Stratton 3D, South Texas (BEG/SEG open data) [15]: post-stack migrated, dt 2 ms, 170 train / 8 test sections (1501×309) | done |
| 3D marine | F3 Netherlands (dGB / Zenodo 3755060) [16]: post-stack, dt 4 ms, 401 train / 10 test inlines (256×701) | done |

- All field sections are scaled to unit RMS.
- Noise level is estimated by MAD: σ̂ = 0.054 for F3, 0.092 for Stratton, **0.38 for the 2D land line and 0.22 for the 2D marine line**. The 2D lines carry 4–7× more random noise than the 3D post-stack volumes.
- Each file's SHA-256 is recorded in `paperproto/SEISMIC_SHA256SUMS`. 2D lines: R_1.SGY `764902ee…`, WS82-611__5692.sgy `db0d7376…`.
- **Data reproducibility check.** The synthetic training and test arrays were rebuilt from the public mirrors on a different OS and numpy build. They match the original arrays to floating-point precision: max relative difference about 1e-7 for Marmousi2, about 1e-4 for Overthrust and Salt (FFT round-off). Shapes are identical.

## V. Application to Synthetic Data

### A. Reproduction on images (prerequisite)

**Table I.** Gaussian noise, PSNR in dB (paper → ours).

| | IDR | Supervised |
|---|---|---|
| Kodak σ25 | 32.36 → 31.89 | 32.39 → 32.08 |
| Kodak σ50 | 29.27 → 28.73 | 29.24 → 28.89 |
| BSDS300 σ25 | 31.48 → 31.15 | 31.40 → 31.30 |
| BSDS300 σ50 | 28.25 → 27.87 | 28.17 → 28.00 |
| BSD68 σ25 | 29.20 → 28.84 | 29.20 → 29.04 |
| BSD68 σ50 | 26.25 → 25.95 | 26.24 → 26.11 |

**Variance and the IDR–supervised gap.**
- Seeds 0–2 give sd ≤ 0.013 dB.
- IDR is below supervised in every row and seed, by 0.12–0.19 dB, and on 647 of 648 image/σ pairs.

**Other noise types.**
- Binomial and impulse (Table 5 of [1]): IDR is within 0.4 dB of the paper, except impulse, which is +1.0 / −1.5 dB.
- Our supervised impulse model is anomalously low: 1.6–3.6 dB below the paper (open issue).
- Correlated noise g1–g4: 7 of 8 cells are within −0.25 to +1.4 dB. The paper's Kodak g1 entry (37.46) is inconsistent with its own BSDS300 g1 and BM3D rows.

**Drift attribution.**
- Our evaluator reproduces the paper's BM3D to within 0.08 dB on Kodak and BSD68.
- The earlier reproduction's −1.4 dB came from a batch-192 epoch schedule. 128-pixel crops cost another 0.1–0.2 dB.
- Ruled out:
  - test noise;
  - channel order;
  - float vs uint8 metrics;
  - SSIM window;
  - TF32 precision;
  - the data holdout;
  - the fresh-noise leak;
  - training seed.
- The residual 0.3–0.5 dB is unexplained. The most likely cause is the unstated 500k-step schedule.

**Mechanism ablations.**
- Stage 0 alone scores 25.00 dB, or 29.60 / 31.01 with Nr2N correction (1 and 8 draws). Full IDR scores 31.89, so the refinement provides the gain.
- `[FIG]` `targets_A1_idr_g_s0.png`: the teacher targets sharpen across epochs. `zoom_kodak_50_kodim23.png`, `psd_*.png`.

### B. Seismic, joint training on three synthetics

**Table II.** SNR out in dB at σ = 0.5, p = 0.05 or fraction 0.3; input SNR is about 6 dB. Entries are IDR / supervised.

| Test | sg | sb | ss | sd | best classical (sg) |
|---|---|---|---|---|---|
| Marmousi2 | 19.31 / 19.34 | 19.13 / 18.78 | 18.24 / 21.70 | 21.18 / 24.03 | BM3D 19.12 |
| Overthrust | 19.00 / 19.95 | 17.26 / 18.06 | 30.65 / 35.07 | 19.51 / 19.69 | BM3D 16.47 |
| Salt | 23.21 / 24.14 | 22.30 / 23.51 | 24.34 / 27.82 | 14.38 / 15.58 | BM3D 20.89 |
| M2 PreSTM (unseen) | **17.37** / 15.75 | **16.13** / 14.43 | 32.30 / 36.85 | 28.13 / 28.52 | BM3D 16.38 |
| M2 PostSTM (unseen) | **17.08** / 15.43 | **15.82** / 14.37 | 32.26 / 36.84 | 27.72 / 28.14 | BM3D 16.34 |

Findings:
1. **Additive noise.** IDR is within 0–1.2 dB of supervised, and above every classical baseline: +0.2 to +4.7 dB over BM3D and +5 to +14 dB over f-x deconvolution.
2. **Unseen processing.** IDR beats supervised by 0.8–2.2 dB, consistent over 3 seeds.
3. **Spikes and dead traces.** Supervised leads by 3.5–5 dB on spikes, and by 1.2–2.9 dB on dead traces for Marmousi2 and Salt. This is consistent with the weaker IDR targets for drop-type noise.
4. **Fixed vs fresh noise.** A fixed noisy realisation costs 2–2.5 dB on single-section Marmousi2, and nothing on the 200+-section surveys.
5. **Noise-model mismatch.** A white-trained model collapses on band-limited noise (about 3 vs 16 dB). A band-limited-trained model handles both.

`[FIG]` `seis_m2__sg__1.0.png`, `seis_spec_*.png` (f-k and amplitude spectra), `seis_bars.png`, `seis_targets.png`.

### C. Seismic, one model per synthetic dataset `[TODO-K]`

- **Runs:** 3 datasets × {IDR, supervised} × {sg, sb, ss, sd}, plus seeds 1–2 for sg and a fixed-noise ablation per dataset: 39 runs.
- **Table III:** own-dataset SNR, joint vs per-dataset model.
- **Table IV:** transfer matrix (train on one, test on the others), plus the Marmousi2 processing variants and F3/Stratton.

**Questions to answer:**
- Does joint training help or hurt each survey?
- Does the per-dataset Marmousi2 model (a single section) over-fit?
- Does IDR still generalise better than supervised when the training data is a single survey?

## VI. Application to Real Field Data

### A. Semi-synthetic evaluation

**Table V.** SNR in dB. White noise is added to the field section, and the original section is the reference.

| Model | F3 σ0.5 / 1.0 | Stratton σ0.5 / 1.0 | 2D land | 2D marine |
|---|---|---|---|---|
| IDR trained on the survey | **14.97 / 11.41** | **13.30 / 9.62** | **11.51 / 8.48** | **8.53 / 5.15** |
| Supervised on the survey (oracle) | 14.69 / 11.04 | 13.43 / 9.66 | n/a (deadline cut) | n/a (deadline cut) |
| BM3D, oracle σ | 13.84 / 10.04 | 12.13 / 8.42 | n/a (deadline cut) | n/a (deadline cut) |
| f-x deconvolution | — | — | 8.53 / 5.92 | 3.95 / 2.76 |
| bandpass 5–60 Hz | — | — | 9.38 / 3.42 | 5.85 / 1.58 |
| IDR, synthetic corpus (3 seeds; 2D: seed 0) | 10.16±0.48 / 7.07±0.24 | 9.88±0.23 / 6.64±0.25 | 9.25 / 6.27 | 2.94 / 1.96 |
| Supervised, synthetic corpus | 10.45±0.16 / 7.29±0.07 | 10.18±0.10 / 6.89±0.02 | 9.63 / 6.25 | 3.31 / 2.09 |

- Synthetic-trained models lose 3–4 dB on field data because of the domain gap.
- IDR trained on the target survey, with no clean data, closes the gap and beats BM3D.

### B. Real field noise

- No reference exists here. Diagnostics are the fraction of energy removed and the leakage measure, local similarity (LS), between the output and the removed component.

| Model | F3: % energy removed / LS | Stratton: % energy removed / LS |
|---|---|---|
| IDR trained on the raw field data | 0.02% / 0.12 | 0.04% / 0.05 |
| Synthetic-trained models | about 1% / 0.26–0.36 | about 1% / 0.26–0.36 |

- The field-trained IDR barely changes the data. The synthetic-trained models remove signal: reflectors are visible in the removed-noise panels.
- **Interpretation.** These post-stack migrated volumes contain little random noise. What remains is coherent (migration smiles, acquisition footprint) and is outside the additive-noise model.
- **2D lines** (σ̂ 4–7× higher than the 3D volumes), real-noise diagnostics:

| Model | 2D land: % energy removed / % out-of-band removed / LS | 2D marine: % energy removed / % out-of-band removed / LS |
|---|---|---|
| IDR trained on the raw line | 0.15 / 6.6 / **0.12** | 1.44 / 4.7 / 0.32 |
| IDR, synthetic corpus (white) | 1.44 / 28.6 / 0.37 | 40.3 / 83.6 / 0.28 |
| Supervised, synthetic corpus | 0.97 / 18.8 / 0.37 | 34.0 / 79.5 / 0.29 |
| IDR, synthetic corpus (band-limited) | 11.3 / >100 / 0.31 | 63.6 / 87.4 / 0.31 |

  - **2D land:** IDR trained on the raw line removes a faint, incoherent noise field. Its removed panel shows almost no reflectors (LS 0.12). The synthetic-trained models remove visible reflector energy (LS 0.37). This is the first field dataset where self-supervised training on the raw data removes real noise without visible signal damage.
  - **2D marine:** the synthetic-trained models remove 34–64% of the energy, including diffractions and scattered events: a severe domain gap. On the semi-synthetic test they score below the input SNR. IDR trained on the raw line removes 1.4%, mostly high-frequency texture at depth, with only a faint seafloor imprint.
- `[FIG]` `field_f3.png`, `field_st.png`, `new_field_l2.png`, `new_field_w2.png` (removed components shown ×10), `new_fk_l2.png`, `new_fk_w2.png`.

## VII. Results Summary

| Claim | Strength |
|---|---|
| Fast IDR ranking reproduces | strong |
| Absolute PSNR within about 0.5 dB | strong |
| "IDR ≈ supervised" | not reproduced (0.12–0.19 dB gap) |
| Self-supervised IDR is competitive on seismic, within about 1.2 dB of supervised for additive noise | strong (3 seeds for sg) |
| IDR generalises better than supervised across processing versions | strong (3 seeds) |
| Training on the survey itself beats synthetic-trained models and BM3D on field data | moderate (semi-synthetic) |
| Real field-noise removal | not demonstrated on 3D post-stack data; **shown qualitatively on the 2D land line** (incoherent noise removed, low leakage); partial on the 2D marine line |

## VIII. Discussion

- **Why IDR generalises better across processing.** The supervised model learns the joint statistics of signal and noise in the training surveys. IDR's targets come from the data distribution and the noise model alone, so it relies less on signal priors and transfers better to differently processed data.
- **Where IDR is weaker.** Drop-type noise (spikes, dead traces): the refined targets keep residual artefacts that L2 masking cannot fully remove. Possible fixes are a blind-spot hybrid or a trace-interpolation prior.
- **Small-data regime.** A 2D line is a single section, like Marmousi2. Fresh noise draws act as augmentation and matter here (2–2.5 dB). Per-dataset results `[TODO-K]` will quantify over-fitting.
- **Noise model is the key assumption.** Mismatch collapses performance. For field use, the added noise n′ should be matched to the data's noise spectrum: the band-limited model, or one estimated from noise-only windows or the pre-first-break zone. Real noise in post-stack data is largely coherent, which is outside IDR's assumptions.
- **Reproducibility notes.**
  - The paper's text and its released config differ (50k vs 500k steps).
  - The paper's Table 5 labels may be swapped.
  - The Kodak g1 entry appears wrong.
  - Upstream correlated-noise "level" is a variance, not a standard deviation.
- **Limitations.**
  - Single seeds outside the sg rows.
  - Training precision: TF32 on the RTX 5070 Ti, FP32 on Kaggle T4s. The FP32 control showed no difference.
  - The semi-synthetic reference for field data is the field data itself.
  - SSIM on Marmousi2 is dominated by the seafloor event.

## IX. Conclusions

- Fast IDR is reproducible in ranking, and to within about 0.5 dB in absolute PSNR, under its stated protocol. Its parity with supervised training is not reproduced.
- Transferred to seismic data without changing the architecture, it is a strong label-free denoiser for random and band-limited noise. It is better than supervised training when the processing differs between training and test.
- When trained on the target survey itself, it is the best of the tested methods on field data with semi-synthetic noise.
- On noisier 2D lines, IDR trained on the raw line removes real random noise with little signal leakage. Synthetic-trained models damage signal badly on field data, especially the marine line. Attenuating the coherent residual noise of post-stack 3D volumes needs a noise model beyond additive i.i.d. or band-limited noise.

## References (to be converted to IEEE style)

[1] Y. Zhang, D. Li, K. L. Law, X. Wang, H. Qin, H. Li, "IDR: Self-supervised image denoising via iterative data refinement," CVPR 2022.
[2] J. Lehtinen et al., "Noise2Noise: Learning image restoration without clean data," ICML 2018.
[3] A. Krull, T.-O. Buchholz, F. Jug, "Noise2Void — Learning denoising from single noisy images," CVPR 2019.
[4] J. Batson, L. Royer, "Noise2Self: Blind denoising by self-supervision," ICML 2019.
[5] N. Moran, D. Schmidt, Y. Zhong, P. Coady, "Noisier2Noise: Learning to denoise from unpaired noisy data," CVPR 2020.
[6] K. Dabov, A. Foi, V. Katkovnik, K. Egiazarian, "Image denoising by sparse 3-D transform-domain collaborative filtering," IEEE TIP 2007.
[7] L. L. Canales, "Random noise reduction," SEG Expanded Abstracts 1984.
[8] Y. Chen, S. Fomel, "Random noise attenuation using local signal-and-noise orthogonalization," Geophysics 2015.
[9] O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," MICCAI 2015.
[10] J. Deng et al., "ImageNet: A large-scale hierarchical image database," CVPR 2009.
[11] D. Martin et al., "A database of human segmented natural images…," ICCV 2001.
[12] S. Roth, M. Black, "Fields of experts," CVPR 2005.
[13] G. S. Martin, R. Wiley, K. J. Marfurt, "Marmousi2: An elastic upgrade for Marmousi," The Leading Edge 2006.
[14] F. Aminzadeh, J. Brac, T. Kunz, "3-D Salt and Overthrust Models," SEG/EAGE 1997.
[15] B. A. Hardage et al., "3-D seismic imaging and interpretation of fluvially deposited thin-bed reservoirs" (Stratton field), BEG / AAPG 1994 [verify exact citation].
[16] Y. Alaudah et al., "A machine-learning benchmark for facies classification," Interpretation 2019 (F3 Netherlands).
[17] C. Birnie, M. Ravasi, S. Liu, T. Alkhalifah, "The potential of self-supervised networks for random noise suppression in seismic data," AI in Geosciences 2021.
[18] G. H. F. Gardner, L. W. Gardner, A. R. Gregory, "Formation velocity and density — the diagnostic basics for stratigraphic traps," Geophysics 1974.
[19] D. P. Kingma, J. Ba, "Adam: A method for stochastic optimization," ICLR 2015.
[20] J. J. Miller et al., "Regional seismic lines reprocessed using post-stack processing techniques: National Petroleum Reserve, Alaska," USGS Open-File Report 00-286, 2000.
[21] U.S. Geological Survey, National Archive of Marine Seismic Surveys (NAMSS), survey W-34-82-MB, https://walrus.wr.usgs.gov/namss/.
