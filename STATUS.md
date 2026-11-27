# Project status: Fast IDR reproduction and seismic adaptation (DAT-102)

_Status as of 28 Nov 2026, 01:45 IST. Written for whoever writes the report and slides; `context.md` has the older, more detailed hand-off._

---

## 1. What the professor asked for, and where each item stands

From `context/mail2.txt` and `context/mail3.txt`:

| Requirement | Status | Where |
|---|---|---|
| Reproduce the paper's results with the theory understood | **Done** | `repro/reports/paperproto/REPORT.md` §1–3, `RESULTS.md` |
| Apply the method to open-source test data | **Done** | same |
| Two synthetic datasets from the SEG wiki models | **Done:** three (Marmousi2, SEG/EAGE Overthrust, SEG/EAGE Salt), trained jointly *and* one model per dataset | `REPORT.md` §4, `NEW_RESULTS.md` §1–2 |
| One real field dataset each: 2D land, 2D marine, 3D land, 3D marine | **Done:** NPR-A Alaska line 1 (2D land), NAMSS WS82-611 Gulf of Mexico (2D marine), Stratton (3D land), F3 Netherlands (3D marine) | `REPORT.md` §5, `NEW_RESULTS.md` §3–4 |
| Paper in IEEE TGRS format (Abstract … References) | **Rough Markdown draft only.** The LaTeX (Overleaf IEEEtran) version is still to write. | `repro/paper/PAPER_DRAFT.md` |
| Code + data + paper submitted | Code and results are on GitHub. Data recipes and checksums are in the repo. Weights are on Hugging Face. | see §5 |
| Seminar presentation with the theory and code explained | **Not started** | — |
| Copy files to Ms Ekta Yadav's system and show reproducibility | **Not started** | see §6 |

**Nothing the professor asked for is waiting on GPU time.** All the remaining runs (§3) are extra.

---

## 2. What has been run

### 2a. Before the Kaggle phase (kratos server, RTX 5070 Ti, 19–20 Nov): 104 jobs

Protocol P throughout: 50k Adam steps, batch 4, 256² crops, lr 3e-4 halved at 25k and 40k, L1 loss (masked L2 for drop-type noise).

- **Images (ImageNet-trained):**
  - A1/A2: Fast IDR and supervised, Gaussian noise, 3 seeds.
  - A3: Nr2N stand-in.
  - A4: Noise2Noise.
  - A5: fixed-noise ablation.
  - A6: gray.
  - A7/A8: binomial and impulse (Table 5).
  - A9: correlated noise g1–g4 (Table 6).
  - FP32 control.
  - Drift study.
  - BM3D calibration.
- **Seismic, one joint model on Marmousi2 + Overthrust + Salt (C1–C9):**
  - white, band-limited, spikes and dead traces;
  - 3 seeds for white noise;
  - Nr2N and N2N;
  - fixed-noise ablations;
  - noise-mismatch study;
  - classical baselines (bandpass, f-x deconvolution, BM3D, median, trace interpolation).
- **3D field data (F1–F4):** for F3 and Stratton, IDR on a fixed noisy copy, a supervised oracle, and IDR on the raw field noise.

Results: `repro/reports/paperproto/REPORT.md` (narrative) and `RESULTS.md` (all tables and figures). Inference figures for these models: `INFERENCE_PRIOR.md`.

### 2b. Kaggle phase (27–28 Nov, T4 GPUs, FP32): 26 runs finished, 17 still running

The code that runs it is in `repro/kgl/`: it splits the job list across Kaggle accounts, runs 2 GPU kernels per account, and pulls results back.

| Group | Runs | Status |
|---|---|---|
| 2D field lines: IDR on a fixed noisy copy and IDR on the raw line, for land (`l2`) and marine (`w2`) | 4 | **done** |
| Old joint models (C1, C2, C7) tested on the 2D lines; real-noise diagnostics; bandpass and f-x baselines | evals | **done** |
| Per-dataset core: {m2, ot, sa} × {IDR, Sup} × {white sg, band-limited sb} | 12 | **done** |
| Per-dataset spikes (ss): {m2, ot, sa} × {IDR, Sup} | 6 | **done** |
| Per-dataset fixed-noise ablation (IDR, sg) | 3 | **done** |
| Seed 1: m2 IDR sg | 1 | **done** |
| Per-dataset dead traces (sd): {m2, ot, sa} × {IDR, Sup} | 6 | running, ETA ~03:00 |
| Seeds 1–2 for the remaining per-dataset sg models | 11 | running, ETA ~03:00 (6 runs) and ~06:00 (5 runs) |

Data for the Kaggle phase was **rebuilt from public mirrors**. The archive SHA-256 values match `repro/paperproto/SEISMIC_SHA256SUMS`. The rebuilt arrays equal the original kratos arrays to about 1e-7 relative (Marmousi2) and 1e-4 relative (Overthrust, Salt; FFT round-off). This is a useful reproducibility point for the report.

---

## 3. What is left

| Item | Type | Needed for the professor? | Effort |
|---|---|---|---|
| Dead-trace per-dataset results (6 numbers) | table rows in `NEW_RESULTS.md` §1 | no; the joint model C9 already covers dead traces | automatic once the runs finish (`python kgl/pull_one.py …` then `python kgl/new_tables.py`) |
| Seed mean ± sd for the per-dataset models | table column | no; the joint model already has a 3-seed table (`REPORT.md` §6) | same |
| **LaTeX paper** (IEEE TGRS template) | writing | **yes** | port `repro/paper/PAPER_DRAFT.md`; figures and tables are listed there |
| **Seminar slides** | writing | **yes** | use the figures listed in §4 |
| **Reproducibility package and demo for Ms Ekta Yadav** | packaging | **yes** | see §6 |
| Supervised oracle and BM3D on the 2D lines | experiment | no (cut for time) | about 3 h GPU plus CPU BM3D |
| Upload the new Kaggle weights to Hugging Face | housekeeping | no | about 15 min |

---

## 4. Results to put in the report and slides (all final)

Full numbers: `repro/reports/paperproto/NEW_RESULTS.md` (new) and `REPORT.md` (earlier). Interpretation and caveats: `REVIEW_KAGGLE_RUNS.md`.

**Image reproduction (paper Table 2):**
- Ranking reproduced. Absolute PSNR 0.33–0.54 dB below the paper for IDR and 0.10–0.35 dB for supervised.
- "IDR ≈ supervised" is *not* reproduced: IDR is 0.12–0.19 dB below in every row and seed.
- Evaluator verified with BM3D (within 0.08 dB of the paper's BM3D numbers).

**Seismic, joint model:** IDR is within 0–1.2 dB of supervised for white and band-limited noise, and beats every classical method. It beats supervised by 0.8–2.2 dB on unseen Marmousi2 processing versions (3 seeds). Supervised is clearly better for spikes and dead traces.

**Seismic, one model per dataset (new)**, SNR in dB, σ = 0.5, IDR / Sup:

| | white | band-limited | spikes (p 0.05) |
|---|---|---|---|
| Marmousi2 | 19.47 / 18.17 | 18.82 / 17.40 | 17.87 / 22.49 |
| Overthrust | 19.57 / 20.23 | 17.56 / 18.16 | 31.57 / 34.83 |
| Salt | 23.39 / 24.19 | 22.38 / 22.87 | 24.53 / 26.12 |

- IDR > supervised on Marmousi2 comes from **supervised over-fitting the single training section**. The checkpoint sweep shows supervised peaks at 10k steps (19.14) and then declines (`SANITY.md`). Write it as "IDR ≈ supervised with early stopping", **not** "IDR beats supervised".
- Single-dataset models **transfer badly**: often below the input SNR on other surveys (`NEW_RESULTS.md` §2). The joint model does not.
- Fixed-noise ablation: −3.1 dB on Marmousi2 (single section), no effect on Overthrust and Salt.

**Field data:**
- **Semi-synthetic test:** IDR trained on the survey itself is the best method on all four field datasets.

  | Dataset | IDR on the survey, SNR at σ = 0.5 | Margin over the best classical method |
  |---|---|---|
  | F3 | 14.97 | +1.1 dB (BM3D) |
  | Stratton | 13.30 | +1.2 dB (BM3D) |
  | 2D land | 11.51 | +2.1 dB (bandpass) |
  | 2D marine | 8.53 | +2.7 dB (bandpass) |

  Synthetic-trained models fail on the marine line: below the input SNR.
- **Real noise:**
  - 2D land: IDR trained on the raw line removes incoherent noise with almost no signal.
  - 2D marine: partial.
  - F3 and Stratton (post-stack 3D): almost nothing removed (negative result).
  - Synthetic-trained models remove reflector and diffraction energy.

**Key figures**, in `repro/reports/paperproto/figs/`:
- `new_field_l2.png`, `new_field_w2.png`, `new_fk_*.png`: 2D lines on real noise.
- `infer/` (Kaggle runs) and `infer_prior/` (earlier runs): per-model inference panels, indexed in `INFERENCE.md` and `INFERENCE_PRIOR.md`.
- `infer_prior/baselines_vs_nets_sg.jpg`: networks against BM3D, f-x and bandpass on the same noisy section.
- `sanity_training_curves.png`, `sanity_panels_*.png`, `sanity_spectra_*.png`.
- From `REPORT.md`: `seis_*`, `field_f3.png`, `field_st.png`, `zoom_*`, `psd_*`, `targets_*`, `seed_spread.png`.

**Caveats the report must state** (details in `REVIEW_KAGGLE_RUNS.md`):
1. On the field semi-synthetic test, the reference is the field data itself, which contains its own noise.
2. The MAD noise estimate (σ̂) on the 2D lines implies far more noise than IDR removes. Treat σ̂ as an upper bound, and do not claim "most noise removed".
3. The Marmousi2 test region is a single section.
4. Kaggle runs used FP32 on T4. Earlier runs used TF32 on an RTX 5070 Ti; the A1 control showed no difference.
5. Minor edge artefacts from reflect padding (top of Marmousi2, bottom of Overthrust and Salt).

---

## 5. Where things are

| What | Where |
|---|---|
| Code (training, evaluation, data preparation) | `repro/paperproto/` (protocol P), `repro/models/unet.py` (network) |
| Kaggle orchestration, tables, sanity checks, galleries | `repro/kgl/`: `dispatch.py`, `make_kaggle_jobs.py`, `kernel_template.py`, `pull_one.py`, `new_tables.py`, `sanity.py`, `inference_gallery*.py` |
| Reports | `repro/reports/paperproto/`: `REPORT.md`, `RESULTS.md`, `NEW_RESULTS.md`, `REVIEW_KAGGLE_RUNS.md`, `SANITY.md`, `INFERENCE.md`, `INFERENCE_PRIOR.md` |
| Paper draft | `repro/paper/PAPER_DRAFT.md` |
| Earlier hand-off and assignment mails | `context.md`, `context/mail2.txt`, `context/mail3.txt` |
| Trained weights, earlier 42 models | Hugging Face `glitch23/fast-idr-paper-protocol` (public) |
| Trained weights, Kaggle runs | local `repro/runs/paperproto/{S_*,G_*}/weights.pth` (git-ignored, about 4 MB each). **Not uploaded yet.** |
| Run outputs (metrics, evaluation CSV/JSON) | `repro/runs/paperproto/` (git-ignored; local copy and kratos) |
| Data sources | Marmousi2, Overthrust and Salt from the Hugging Face mirror `shaowinw/seismic-benchmark-models`; F3 from Zenodo 3755060; Stratton from SEG/BEG S3; 2D land from USGS OFR 00-286 `R_1.SGY` (SHA-256 `764902ee…`); 2D marine from USGS NAMSS W-34-82-MB `WS82-611__5692.sgy` (`db0d7376…`) |

Data and runs are not in git. Rebuild the seismic data with `python paperproto/prepare_seismic.py m2 ot sa l2 w2`, after downloading the raw files into `repro/data/seis_raw/` (Marmousi2 needs `python prepare_seismic.py` at the `repro/` root first).

---

## 6. Suggested reproducibility package for Ms Ekta Yadav (not done)

1. This repo plus the Hugging Face weights.
2. A short script that downloads the public data, runs `prepare_seismic.py`, and re-scores 2–3 trained models with `paperproto/eval_seis.py`. It should match the numbers in `NEW_RESULTS.md` to 0.01 dB; CPU inference is enough (see `kgl/sanity.py`).
3. Optionally, one short training run (for example 2k steps) to show the training loop works.
