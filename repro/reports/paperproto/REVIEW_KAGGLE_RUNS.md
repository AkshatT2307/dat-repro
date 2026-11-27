# Review of the completed Kaggle runs (28 Nov 2026, 00:40 IST)

Scope: every run that has finished and been pulled so far.
- **2D field lines:** 4 training runs (G_*), plus their field diagnostics, and the joint-corpus models evaluated on both lines.
- **Per-dataset synthetic models:** 15 training runs (S_*).

Detailed numbers are in `NEW_RESULTS.md`; automated checks are in `SANITY.md`.

**Status legend**
- **OK:** checked; consistent with expectations.
- **NOTE:** a real effect that needs care when writing it up.
- **ISSUE:** a limitation or a possible error that must be stated or fixed.

---

## 1. Integrity of the runs

| Check | Result |
|---|---|
| All runs reached 50,000 steps, finite loss, no retries | **OK.** 19 / 19 |
| Training speed | **NOTE.** Kaggle T4 in FP32: supervised 5.5–5.8 it/s, IDR 4.4–4.8 it/s, IDR with band-limited noise 3.2–3.4 it/s. Earlier runs on the RTX 5070 Ti (TF32) were about 7× faster. TF32 and FP32 gave equal results in the A1 control. |
| Checkpoints vs Kaggle evaluation | **OK.** Every `weights.pth` was re-run locally on CPU on the stored noisy test section. Max \|Δ\| against the Kaggle output is 4e-6 to 2e-5, and the SNR is identical to 0.01 dB. |
| Data | **OK.** Training and test arrays were rebuilt from public mirrors (archive SHA-256 matches the repo). They equal the original kratos arrays to 1e-7 relative (Marmousi2) and 1e-4 relative (Overthrust, Salt; FFT round-off). |
| Training curves | **OK.** Smooth. IDR loss drops at each teacher refresh (25k, 37.5k), which is expected: refreshed targets are cleaner. Supervised curves are flat after about 20k. |
| SNR gain on own test set | **OK.** Positive for every model: +3.6 to +36 dB. |

## 2. Per-dataset synthetic models (S_*)

SNR in dB on each model's own test set at σ = 0.5 (white sg, band-limited sb) and p = 0.05 (spikes ss).

| Dataset | sg IDR / Sup | sb IDR / Sup | ss IDR / Sup | joint model, sg IDR / Sup |
|---|---|---|---|---|
| Marmousi2 | 19.47 / 18.17 | 18.82 / 17.40 | 17.87 / 22.49 | 19.31 / 19.34 |
| Overthrust | 19.57 / 20.23 | 17.56 / 18.16 | 31.57 / 34.83 | 19.00 / 19.95 |
| Salt | 23.39 / 24.19 | 22.38 / 22.87 | 24.53 / 26.12 | 23.21 / 24.14 |

**Findings**
1. **OK, as in the joint study.** On Overthrust and Salt, supervised leads IDR by 0.7–0.8 dB for white noise and by 1.6–3.3 dB for spikes. Supervised is the upper bound; IDR comes close for additive noise and is clearly weaker for spikes.
2. **NOTE: on Marmousi2, IDR is 1.3 dB above supervised.** We traced this to over-fitting (checkpoint sweep in `SANITY.md`):
   - Supervised peaks at 10k steps (19.14 dB) and then declines on the single 1001×1400 training section.
   - IDR keeps improving.
   - **Write-up:** "with one training section, IDR ≈ supervised with early stopping". Not "IDR beats supervised". The paper's protocol has no validation-based stopping, so this is a property of the fixed 50k schedule.
3. **OK: per-dataset vs joint.**
   - Overthrust and Salt: per-dataset models equal or slightly beat the joint model on their own survey (sg: +0.2 to +0.6 dB IDR, +0.05 to +0.3 dB Sup).
   - Marmousi2: supervised loses 1.2 dB without the other surveys, while IDR does not. More data helps supervised on the small survey.
4. **NOTE: transfer collapses.**
   - A model trained on one survey and tested on another often scores **below the input SNR** (6 dB) at σ = 0.5. Examples: m2 model on Overthrust 3.6 / 3.3 dB (IDR / Sup); ot model on Marmousi2 4.5 dB (Sup); sa model on Marmousi2 6.3 dB (Sup).
   - The joint model does not have this problem.
   - The Marmousi2-only IDR still transfers well to the other Marmousi2 processing versions: 16.1–18.2 dB, against 11.9–14.9 dB for Marmousi2-only supervised. That is the "IDR generalises across processing" result again, now with a +3.3 to +4.2 dB margin.
5. **OK: spectra (band-limited noise).** IDR outputs follow the clean amplitude spectrum inside the 5–60 Hz band on all three surveys. Above about 80 Hz on Overthrust, IDR leaves a residual about 10 dB over the clean spectrum. The energy there is low, so it costs little SNR, but the high-frequency tail is not fully cleaned.
6. **ISSUE (minor): edge artefacts.**
   - Error maps show a thin band of residual noise in the first few time samples on Marmousi2 (IDR, white noise). This is the water column above the seafloor, where the signal is zero.
   - There are blobs at the bottom edge on Overthrust and Salt (both methods).
   - Probable cause: reflect padding to a multiple of 32, combined with zero-signal regions. The effect on SNR is small. Mention it in the paper or crop edges in the figures; no rerun is needed.


## 2b. Update (01:30): runs finished after the first review

| Item | Result | Status |
|---|---|---|
| Supervised, band-limited noise (m2, ot, sa) | 17.40 / 18.16 / 22.87 dB. On Marmousi2, IDR is again above supervised (+1.42 dB), the same single-section over-fitting as for white noise. On Overthrust and Salt, supervised leads by 0.5–0.6 dB. | NOTE |
| Fixed-noise ablation (σ 0.5) | Marmousi2: 16.37 fixed vs 19.47 fresh (**−3.1 dB**). Overthrust: 19.77 vs 19.57. Salt: 23.85 vs 23.39. Repeats the joint finding: fixed noise hurts only the single-section survey. | OK |
| Seed 1, Marmousi2 IDR (white) | 19.68 dB vs 19.47 (seed 0) on its own test. On M2 PreSTM: 15.93 vs 16.53. Transfer varies more across seeds than own-test SNR does. | NOTE |
| Figures | The earlier PNGs were rendered with matplotlib anti-aliasing, which averages white noise away when downsampling. That made the removed-noise panels look emptier than they are. All figures are regenerated with nearest-neighbour rendering. Conclusions are unchanged, except the 2D land real-noise wording, which becomes "almost no signal removed": there is a faint reflector trace near 2.1 s in test section 1. | ISSUE (fixed) |

## 3. 2D field lines (G_*)

**Semi-synthetic test** (white noise added; the original line is the reference). SNR in dB, σ = 0.5 / 1.0.

| Model | 2D land (NPR-A) | 2D marine (NAMSS) |
|---|---|---|
| IDR trained on the line (fixed noisy copy) | **11.51 / 8.48** | **8.53 / 5.15** |
| best classical (bandpass or f-x) | 9.38 / 5.92 | 5.85 / 2.76 |
| IDR, joint synthetic corpus | 9.25 / 6.27 | 2.94 / 1.96 |
| Supervised, joint synthetic corpus | 9.63 / 6.25 | 3.31 / 2.09 |
| input | 6.01 / −0.02 | 4.93 / −1.09 |

1. **OK.** Training on the target line wins by 2.1–2.7 dB, the same conclusion as F3 and Stratton.
2. **NOTE: the synthetic-to-marine domain gap is severe.** The synthetic-trained models score below the input at σ = 0.5. On the raw marine line they remove 34–64% of the energy, including diffractions (see `new_field_w2.png`). Synthetic-trained models should not be used on this line.
3. **OK.** The `idr_fixed` models remove about 0% when applied to the raw line. This is expected: they learned to treat the raw line as clean.

**Real-noise application** (`idr_real`, trained on the raw line plus added noise)

| | 2D land | 2D marine |
|---|---|---|
| Energy removed | 0.15% | 1.44% |
| Out-of-band energy removed | 6.6% | 4.7% |
| Leakage (\|LS\|) | 0.12 | 0.32 |

4. **OK: land.** The removed component is incoherent, with no reflectors visible at ×10. The f-k plot of the removed part is broad and not concentrated on the signal cone.
5. **NOTE: marine.** The removed component is mainly high-frequency texture at depth plus steep dips (f-k: energy above about 60 Hz and at |k| > 0.2). Leakage is 0.32, similar to the synthetic models, so this panel does not show clean separation. The f-k plot shows the low-f, low-k signal region is mostly kept. State it as "partial".
6. **ISSUE: noise level vs energy removed don't agree.**
   - The MAD σ̂ implies random noise is about 14% (land, σ̂² = 0.38²) and 5% (marine) of the energy. `idr_real` removes only 0.15% and 1.4%.
   - Either MAD σ̂ overestimates the random noise (the finest Haar diagonal detail also catches steep-dip and high-frequency signal on these lines), or IDR is conservative.
   - Consequence: added-noise levels for the `idr_real` runs were set from σ̂ and may be too high.
   - The report must state that σ̂ is an upper bound, and must not claim "most of the noise was removed".
   - A follow-up (not run) would estimate noise from a pre-first-break or water-column window instead.
7. **ISSUE (limitation).** On the semi-synthetic test the "clean" reference is the field line itself, which contains its own noise. SNR measures removal of the *added* noise only. This is already stated for F3 and Stratton and applies here too.

## 4. What is not covered yet

| Item | Status |
|---|---|
| Supervised band-limited per dataset (m2, ot, sa) | **done** |
| Fixed-noise ablation per dataset (m2, ot, sa) | **done** |
| Seed 1–2 for white noise (12 runs) | running (seed-1 m2 IDR on acc2c, about 01:00; the rest on the round-1/round-2 kernels, about 03:00 / about 06:00) |
| Dead traces per dataset (6 runs) | running (about 03:00) |
| Supervised oracle on the 2D lines, BM3D on the 2D lines | cut for time |
| Noise-mismatch evaluation per dataset | cut (the joint model covers it) |
| F3 / Stratton as cross-tests for per-dataset models | not run: those test sets were not in the rebuilt data. They are in the kratos copy and could be added as an eval-only job later. |

## 5. Verdict

- No evidence of bugs in the finished runs. The one anomaly (IDR > supervised on Marmousi2) is explained by over-fitting, and the checkpoint sweep shows it.
- The headline results are consistent with the earlier joint-corpus study:
  - IDR ≈ supervised for additive noise;
  - weaker for spikes;
  - generalises better across processing;
  - training on the target field data beats classical methods and synthetic-trained models.
- New claims the data now supports:
  - the 2D lines (land: real-noise removal with low leakage);
  - the single-survey over-fitting of supervised training;
  - the strong cross-survey transfer failure of single-dataset models.
- Caveats that must go into the paper: §2.2 (wording), §3.6 (σ̂ vs removed energy), §3.7 (reference contains noise), §2.6 (edge artefacts).
