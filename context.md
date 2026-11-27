# Project context: Fast IDR reproduction and seismic adaptation

_Last updated: 27 Nov 2026. This file is the hand-off. It records everything done so far, the settings in use, and what is left under the updated task._

---

## 1. The assignment

**Original task (17–19 Nov)**
- Reproduce **Fast IDR** (Zhang et al., CVPR 2022, "IDR: Self-Supervised Image Denoising via Iterative Data Refinement", github.com/zhangyi-3/IDR) and its **supervised** upper bound.
  - Train only these two. Other methods (N2V, DBSN, MPRNet, the paper's Nr2N row) are copied from the paper's tables.
  - Training must follow the paper's protocol exactly.
- Explain the drift between our numbers and the paper's, with extra inference-only experiments where needed.
- Adapt the method to seismic data, with noise models that make sense for seismic. Data may come from the SEG wiki open-data list.
- Add visualisations (FFT and f-k spectra, spatial and frequency domain) and dump all results into a report.
- Run jobs in parallel on the GPU server, with a monitor sub-agent watching the queue.

**Updated task (27 Nov), as given by the user:**
> Synthetic data: select any two synthetic datasets from the SEG wiki models (Marmousi, Overthrust, Sesame, SEG/EAGE Salt, …).
> Real field data: 2D land, 2D marine, 3D land, 3D marine. Take one example of each.

**Decisions made with the user since the update**
- The synthetic data requirement is already met: we use **three** synthetic datasets (Marmousi2, Overthrust, SEG/EAGE Salt).
- The user now wants the **three synthetic datasets trained independently**: one model per dataset, not the single joint model used so far. See §8.
- Field data:
  - **3D land** and **3D marine** are done (Stratton, F3).
  - **2D land** and **2D marine** are still missing.
  - The recommended approach is to train on the field data as well as evaluate on it (§8).

---

## 2. Where everything is

| What | Path |
|---|---|
| Working copy (not a git repo) | `/opt/watchdog/users/glitch/DAT/repro` |
| Paper-protocol code | `repro/paperproto/` |
| Plan and change log | `repro/PAPER_PROTOCOL_PLAN.md` |
| Narrative report | `repro/reports/paperproto/REPORT.md` |
| All tables and 29 figures (auto-generated) | `repro/reports/paperproto/RESULTS.md`, `repro/reports/paperproto/figs/` |
| Run outputs, one folder per run | `repro/runs/paperproto/<run>/` (`weights.pth`, `weights_step{10..50}000.pth`, `latest.pt`, `config.json`, `metrics.jsonl`, `eval/`, …), 4.1 GB |
| Queue state, logs, monitor log | `repro/runs/paperproto/{queue_state.json,logs/,MONITOR_LOG.md}` |
| Older, pre-protocol runs | `repro/runs/{epoch128,continuation5,steps50k_b4_128,steps50k_b4_256,...}` |
| Data | `repro/data/` (26 GB). ImageNet val, Kodak, BSDS300, BSD68, `seis/` (prepared), `seis_raw/` (downloads + `SHA256SUMS`) |
| Upstream IDR clone | `/opt/watchdog/users/glitch/DAT/IDR` |
| GitHub clone | `/opt/watchdog/users/glitch/DAT/dat-repro` (remote `git@github.com:AkshatT2307/dat-repro.git`) |

**GitHub.** `main` (commit `8f59021`) contains `repro/paperproto/`, `repro/PAPER_PROTOCOL_PLAN.md` and `repro/reports/paperproto/`. The `paper-protocol` branch also still exists.
- The working copy in `DAT/repro` is **not** linked to git. After any change, copy it into `dat-repro/repro/`, then commit and push.
- `.gitignore` excludes `runs/`, `data/` and `logs/`.

**Hugging Face.** Private repo `glitch23/fast-idr-paper-protocol` holds all 42 final models. Each run folder has `weights.pth`, `config.json`, `training_summary.json` and `*_summary.json`, and there is a model card.

**Environment**

| Item | Details |
|---|---|
| Python | `repro/.venv/bin/python` (torch 2.7.1+cu128) |
| GPU | 1 × RTX 5070 Ti, 16 GB. One batch-4, 256-pixel run already uses all of it, so running jobs in parallel does not add throughput; 4 slots are used so CPU work overlaps. TF32 runs at about 28 ms/step alone; 4 jobs together reach about 35–45 it/s in total. |
| `bm3d` | Needs `LD_LIBRARY_PATH=$PWD/.venv/blasshim`. The shims `np.float = float` and `scipy.signal.gaussian = scipy.signal.windows.gaussian` live in the code. |
| Shell | `tail` is uutils, so use `tail -n N`. Foreground `sleep` is blocked. |
| Hugging Face login | `glitch23` with a write token in `~/.cache/huggingface/token` |
| GitHub | SSH works as `AkshatT2307` |

**Job queue**
- `python paperproto/make_jobs.py` writes `paperproto/jobs.json`. Jobs have dependencies, priorities and done-files.
- `python paperproto/pp_queue.py`, run inside tmux session `pp`, re-reads `jobs.json` every 20 s.
- The number of slots is in `runs/paperproto/SLOTS` (4).
- Failed jobs are retried up to 3 times and resume from `latest.pt`.
- Extra jobs can go in `paperproto/jobs_extra.json`.

---

## 3. Training settings: protocol P (the paper's text, used for every run)

| Setting | Value |
|---|---|
| Network | upstream `UNet_n2n_un` (`models/unet.py`), 3→3 for RGB, 1→1 for gray and seismic |
| Batch / crop | 4 / 256×256 random crop, no augmentation |
| Steps | 50,000 Adam steps, lr 3e-4, ×0.5 at 25k and 40k (MultiStepLR) |
| Loss | L1. **Masked L2** for binomial, impulse and dead traces, "as recommended in N2N" |
| Image training data | all 50,000 ImageNet val images (BGR→RGB, gray via IMREAD_GRAYSCALE) |
| Fast IDR | epoch 0: target = noisy y, input = y + n′. Each later epoch: copy the student into a frozen teacher; target = teacher(y), input = target + n′. Teacher refreshed every 12,500 steps (1 epoch = 50k samples / batch 4) |
| Zero-mean shift | inputs −0.5 for Gaussian image noise only |
| Precision | TF32 matmuls with deterministic cuDNN. One FP32 control run showed no difference (31.85 vs 31.89 dB). Env `PP_FP32=1` forces FP32. |
| Checkpoints | `latest.pt` every 1k steps, snapshots every 10k, final `weights.pth` |
| Fidelity check | `paperproto/tests/test_equivalence.py`: bit-identical to upstream fastIDR, n2c and n2n over 5 steps that include 2 teacher refreshes |

The released YAML instead uses 500k steps with a cosine schedule. It was **not** used, because the user decided to follow the paper's stated 50k. It is the main remaining suspect for our gap to the paper.

**Image evaluation**
- Upstream `test.py` noise with seed 2026.
- Reflect-pad to a multiple of 32, round to uint8, skimage PSNR/SSIM.
- Test sets: Kodak (24), BSDS300 (all 300), BSD68 (gray).

**Seismic evaluation**
- Whole-section inference (reflect-pad to a multiple of 32).
- 3 test-noise seeds (0, 1, 2), deterministic per section.
- Metrics: SNR (primary), PSNR, SSIM, RMSE, and local similarity between output and removed noise (21×21 window) as a measure of signal leakage.
- SSIM on Marmousi2 is inflated by the seafloor event, so it isn't used for conclusions.

---

## 4. Noise models

**Images**, exactly as upstream:

| Code | Noise | Train level | Test level |
|---|---|---|---|
| `g` | Gaussian | σ ~ U[0,50]/255 | 25, 50 |
| `binomial` | pixel drop, mask shared across channels | 0.95 (upstream draw U(0,1)·U(0,0.95)) | p = 0.5 |
| `impulse` | per-channel drop | 0.95 | p = 0.5 |
| `scn-1`…`scn-4` | spatially correlated, upstream kernels g1–g4 | level 5, which upstream treats as a **variance** of 5/255 (std ≈ 36/255) | level 5 |

Correlated noise is generated on the GPU (port in `noise.py`; checked against upstream by `tests/test_correlated.py`). g5 is not defined upstream, so it was not run.

**Seismic**, one analogue per paper noise type. Amplitudes are in units of the section's RMS (data scaled to unit RMS over live samples).

| Code | Seismic noise | Paper analogue | Train | Test |
|---|---|---|---|---|
| `sg` | white Gaussian | Gaussian | σ ~ U[0,1] | 0.5, 1.0 (and 1.5 in the mismatch study) |
| `sb` | band-limited 5–60 Hz (zero-phase 4th-order Butterworth along time, unit variance) | correlated | σ ~ U[0,1] | 0.5, 1.0 |
| `ss` | erratic spikes ±U[2,8] at probability p | impulse | p ~ U[0,0.2] | 0.05, 0.1 |
| `sd` | dead traces (upstream binomial construction, per trace) | binomial | level 0.95 | fraction 0.3, 0.5 |

Masked L2 is used for `sd`. Clean-data noise sampling is fresh on every visit by default; `--fixed` uses one stored noisy realisation instead.

**Real field noise.** `--fixed raw` trains on the raw field data as the noisy input. The added noise n′ is white with σ ~ U[0.5, 1.5] × the MAD noise estimate: 0.0544 for F3 and 0.0922 for Stratton.

---

## 5. Seismic data prepared (`repro/data/seis/<key>/{train.npy,test.npy,meta.json}`)

Arrays are float32 with shape (N, time, trace), unit RMS. Built by `paperproto/prepare_seismic.py`.

| Key | Dataset | Type | Prepared as | Train / test |
|---|---|---|---|---|
| `m2` | **Marmousi2** | synthetic 2D (marine-style elastic) | released NMO stack, dt 4 ms. Dead traces 0–239 and 2396–2720 excluded | train traces 240–1639: **1 section** (1001×1400). Test traces 1896–2395: 1 section (1001×500) |
| `m2_prestm`, `m2_postm`, `m2_srme` | other Marmousi2 processing versions | test only | same test region | unseen-processing generalisation test |
| `ot` | **SEG/EAGE Overthrust** (3D land model, 187×801×801, dz 25 m) | synthetic | convolutional modelling from velocity: Gardner density, normal-incidence reflectivity at exact interface times (10× fine grid), 30 Hz Ricker, dt 4 ms | train 280 inlines (498×801). Test 9 inlines |
| `sa` | **SEG/EAGE Salt** (3D marine model, 210×676×676, dz 20 m) | synthetic | same modelling | train 235 sections (788×676). Test 10 |
| `f3` | **F3 Netherlands** (Zenodo 3755060) | **3D marine field**, post-stack | inline sections, 255 samples reflect-padded to 256, dt 4 ms | train 401 (256×701). Test 10 |
| `st` | **Stratton 3D** (BEG / SEG open data, `Stratton3D_32bit.sgy`) | **3D land field**, post-stack migrated | sections at fixed CDP, dt 2 ms | train 170 (1501×309). Test 8 |

Downloads are in `data/seis_raw/` with SHA256SUMS. Overthrust and Salt came from the Hugging Face mirror `shaowinw/seismic-benchmark-models`, because the SEG wiki and S3 bucket returned 403. Stratton came from the S3 `open.source.geoscience` objects, which worked.

---

## 6. Everything run so far (all 50k steps, protocol P)

Queue: 104 jobs. Started 19 Nov 22:12 IST, finished 20 Nov 13:50 IST (15 h 38 min). No failures.

**Images (ImageNet training)**

| Run | What |
|---|---|
| A1_idr_g_s0/s1/s2, A2_sup_g_s0/s1/s2 | Fast IDR / supervised, Gaussian RGB, 3 seeds |
| A1_idr_g_s0_fp32 | FP32 precision control |
| A3_nr2n_g_s0 | IDR stage 0 only. Scored as Nr2N with `nr2n_correct.py` (2f(z)−z) |
| A4_n2n_g_s0 | Noise2Noise |
| A5_idrfix_g_s0 | IDR with a fixed noise realisation (tests the fresh-noise leak) |
| A6_{idr,sup}_gray_s0 | gray, BSD68 |
| A7_{idr,sup}_binomial_s0, A8_{idr,sup}_impulse_s0 | Table 5 |
| A9_idr_scn{1..4}_s0 | Table 6, correlated noise |
| A10 (CPU) | BM3D / CBM3D calibration of our evaluator |

**Seismic, joint synthetic corpus**: one model trained on Marmousi2 + Overthrust + Salt together. Each crop picks a survey uniformly, then a section weighted by crop count. Every model is tested on all 8 test sets.

| Run | What |
|---|---|
| C1_idr_sg_s0/s1/s2, C2_sup_sg_s0/s1/s2 | white noise, 3 seeds |
| C3_nr2n_sg_s0, C4_n2n_sg_s0 | Nr2N stand-in, N2N |
| C5_idrfix_sg_s0, C6_idrfix_sb_s0 | fixed noise realisation |
| C7_{idr,sup}_sb_s0 | band-limited noise |
| C8_{idr,sup}_ss_s0 | spikes |
| C9_{idr,sup}_sd_s0 | dead traces |

**Field, trained per survey**

| Run | What |
|---|---|
| F1_idrfix_f3_sg, F1_sup_f3_sg_oracle | F3: IDR on a fixed noisy copy, and supervised treating the field section as clean |
| F2_idr_f3_real | F3: IDR on the raw field noise |
| F3_idrfix_st_sg, F3_sup_st_sg_oracle, F4_idr_st_real | the same three for Stratton |

**Inference-only jobs**
- Drift study D1–D5 on A1, A2 and 6 older checkpoints.
- Noise-mismatch study on C1, C2 and C7.
- Field diagnostics: `field_f3`, `field_st`.
- Classical seismic baselines on the same noisy sections: bandpass, f-x deconvolution, BM3D with oracle σ/PSD, 5×3 median, oracle trace interpolation.

**Older runs (18 Nov, before protocol P)**
- 10 Marmousi2-only runs: epoch128 b192/b128, continuation5, steps50k_b4_128 and a smoke test. About 4 h in total.
- These used the old test split, which had only 92 live traces.

---

## 7. Key results

Full numbers are in REPORT.md and RESULTS.md.

**Images**
- **Ranking reproduced.** Fast IDR sits above BM3D, the Nr2N stand-in and its own unrefined stage-0 network, as in the paper.
- **Absolute PSNR.** IDR is 0.33–0.54 dB below the paper; supervised is 0.10–0.35 dB below.
- **Evaluator verified.** Our BM3D matches the paper's on Kodak and BSD68 to within 0.08 dB.
- **"IDR ≈ supervised" NOT reproduced.** IDR is 0.12–0.19 dB below supervised in every row and seed (seed sd ≤ 0.013 dB), and on 647 of 648 image/σ pairs.
- **Drift attribution**
  - The earlier repro drift came from the batch-192 epoch schedule (−1.4 dB for IDR) and 128-pixel crops (−0.1 to −0.2).
  - Ruled out: test noise, BGR channel order, float vs uint8 metrics, SSIM window, TF32 precision, the 1k holdout, the fresh-noise leak (A5: −0.02 to −0.07) and training seeds.
  - The remaining gap is not explained. The most likely cause is the unstated 500k-step schedule.
- **The paper's BSDS300 evaluation differs from ours.** Our BM3D is +0.56 dB above the paper's BM3D there.
- **The paper's Kodak g1 entry (Table 6) looks wrong.** We get 30.01 against 37.46, while BSDS300 g1 matches.
- **Open anomaly:** our supervised impulse model is 1.6–3.6 dB below the paper and below our own IDR.

**Seismic, joint corpus**
- With white or band-limited noise, IDR is within about 0–1.2 dB of supervised on the training surveys.
- IDR beats supervised by **0.8–2.2 dB** on the unseen Marmousi2 processing versions, in all 3 seeds.
- IDR beats every classical baseline on the synthetics.
- Spikes: supervised leads by 3.5–5 dB. Dead traces: supervised leads by 2.9 dB on Marmousi2 and 1.2 dB on Salt.
- The fixed noise realisation costs **2–2.5 dB on Marmousi2**, the single-section survey, and nothing on Overthrust or Salt.
- Noise model mismatch: a white-trained model collapses on band-limited noise. A band-limited-trained model handles both.

**Field data**
- **Semi-synthetic test.** IDR trained on the survey itself reaches 14.97 dB on F3 and 13.30 on Stratton. That beats BM3D with oracle σ (13.84, 12.13), matches the oracle supervised model, and is 3–5 dB above the synthetic-trained models.
- **Real noise.** Models trained on the raw field data remove almost nothing (0.02–0.04% of the energy). The synthetic-trained models remove about 1%, much of it signal (local similarity 0.26–0.36). Real field-noise removal is **not** demonstrated.

**Nr2N fix.** Applying the stage-0 network directly gives 25.0 dB on Kodak σ25. Proper Nr2N inference (2f(z)−z) gives 29.60, or 31.01 averaged over 8 draws. Always score `nr2n` runs with `paperproto/nr2n_correct.py`.

---

## 8. Tasks left (updated task)

### 8.1 Synthetic: train each of the three datasets independently (NEW)

Until now, one model was trained on Marmousi2, Overthrust and Salt together. The new requirement is **separate models per dataset**, keeping protocol P and the same noise models.

| | |
|---|---|
| Datasets | `m2` (Marmousi2), `ot` (Overthrust), `sa` (SEG/EAGE Salt), each with its own training set (§5) |
| Runs | 3 datasets × {Fast IDR, supervised} × {sg, sb, ss, sd} = **24 runs**, using `pp_train.py --domain seis --keys <m2\|ot\|sa>` |
| Optional additions | seeds 1–2 for the `sg` runs (+12 runs); the fixed-noise ablation per dataset |
| Evaluation | each model on its own test set, plus cross-dataset tests (train on one, test on the other two) and the Marmousi2 processing versions, all with `eval_seis.py --tests ...`. Run the same baselines (already computed per test set in `runs/paperproto/baselines_seis/`) |
| Compute | 24 × 50k steps ≈ 1.2M steps at about 40 it/s, so **about 8–9 h** of queue time. Evaluations add about 15 min |
| Caveat | Marmousi2 has only **one** 1001×1400 training section. Expect over-fitting, and keep fresh noise on every visit (the fixed-noise ablation lost 2–2.5 dB there) |
| Code changes | new entries in `make_jobs.py` (or `jobs_extra.json`). In `eval_seis.py`, the default test list is `CORPUS_TESTS` only when keys == {m2, ot, sa}, so pass `--tests` for single-dataset runs. Add a per-dataset table to `report.py` |

### 8.2 Real field data: one example of each

| Type | Status | Dataset |
|---|---|---|
| 3D land | **done** | Stratton 3D (`st`): F3/F4 runs + `field_st` |
| 3D marine | **done** | F3 Netherlands (`f3`): F1/F2 runs + `field_f3` |
| 2D land | **to do** | candidate: USGS NPR-A Alaska 2D lines (stacked/migrated SEG-Y); alternatives from the SEG wiki open-data list |
| 2D marine | **to do** | candidates: a USGS NAMSS stacked 2D marine line (already stacked), or Mobil AVO Viking Graben line 12 (pre-stack shot gathers; needs NMO and stacking first, +3–5 h) |

**The 3D items count as done.** Their models were trained on each survey separately: F1 and F2 on F3 only (`--keys f3`), F3 and F4 on Stratton only (`--keys st`). Each survey has three separate models: IDR on a fixed noisy copy, supervised oracle, and IDR on the real field noise. The joint synthetic-corpus models (C*) were only *evaluated* on F3 and Stratton, as an extra transfer test.

For each new 2D line, follow the same recipe as F3 and Stratton:
1. Download and record the SHA256. Expect the SEG wiki and S3 to return 403; try mirrors.
2. Add a branch to `prepare_seismic.py`: unit RMS, MAD σ estimate, train/test split along the line with a guard gap.
3. Train 3 models: IDR on a fixed noisy copy (`--fixed sg_s0`), supervised oracle, and IDR on the real noise (`--fixed raw --level 0.5σ̂,1.5σ̂`).
4. Evaluate the existing synthetic-trained models on the line, run the classical baselines, and run `eval_field.py` for real-noise diagnostics.

A 2D line is a single section, the same small-data situation as Marmousi2.

Estimated time: **about 5–6 h** for both lines, about 2–2.5 h of it GPU time. Evaluation-only would take about 2.5–3 h. Add 3–5 h if the marine line has to be stacked from pre-stack data.

Whether field data must be *trained on* or only *evaluated* was not stated in the task. **Training is recommended**, because training on the survey itself is what beats BM3D.

### 8.3 After the runs
- Extend `report.py` with per-dataset synthetic tables and 2D field sections, then update REPORT.md.
- Copy the new code and reports to `dat-repro/repro/`, commit, and push to `main`.
- Upload the new final models to the Hugging Face repo `glitch23/fast-idr-paper-protocol` (private).

**Total estimate** if 8.1 and 8.2 run together in one queue: about **11–13 h** wall time. The GPU work overlaps with data preparation.

---

## 9. Gotchas

- Marmousi2's dead traces (0–239 and 2396–2720) left the old test split with only 92 live traces. The current split avoids them.
- Fresh noise drawn on every visit (the released code's behaviour) leaks a little clean information. It is negligible on images but helps a lot on single-section surveys.
- Don't name modules `queue.py`, `train.py` or `data.py` in `paperproto/`; they shadow the stdlib and repo modules. That is why the files are called `pp_queue.py`, `pp_train.py` and `pp_data.py`.
- In upstream correlated noise, "level 5" is a variance, not a standard deviation.
- The paper's Table 5 binomial/impulse labels may be swapped (its text contradicts the printed table).
- The Claude session and its monitor agent run under the VS Code server, so keep VS Code connected. The tmux queue survives a logout.
