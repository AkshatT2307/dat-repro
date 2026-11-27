# paperproto: Fast IDR under the paper's exact training protocol, plus seismic adaptation

All code for the paper-protocol reproduction of Fast IDR (Zhang et al., CVPR 2022) and its supervised upper bound, and the seismic adaptation.

- **Results:** [`../reports/paperproto/REPORT.md`](../reports/paperproto/REPORT.md) is the narrative report. [`RESULTS.md`](../reports/paperproto/RESULTS.md) has all tables and figures.
- **Plan and change log:** [`../PAPER_PROTOCOL_PLAN.md`](../PAPER_PROTOCOL_PLAN.md).
- **Trained models:** all 42 final checkpoints are on Hugging Face at `glitch23/fast-idr-paper-protocol`.

## Protocol P (from the paper text)

- Network: upstream `UNet_n2n_un`.
- Optimisation: batch 4, 256-pixel crops, 50k Adam steps, learning rate 3e-4, halved at 25k and 40k.
- Loss: L1, except masked L2 for binomial, impulse and dead-trace noise.
- Data: all 50,000 ImageNet val images, no augmentation.
- Fast IDR teacher refresh: every 12,500 steps (one epoch).
- Inputs are shifted by −0.5 for Gaussian noise only.
- The trainer is bit-identical to upstream `fastIDR` (`tests/test_equivalence.py`).

## Files

| File | Role |
|---|---|
| `pp_train.py` | Trainer. Methods: `idr`, `sup`, `n2n`, `nr2n`. `--domain image\|seis`, `--fixed` for a fixed noise realisation |
| `noise.py` | Image noise (upstream) and seismic noise: `sg` white, `sb` band-limited 5–60 Hz, `ss` spikes, `sd` dead traces |
| `pp_data.py` | ImageNet and seismic training datasets |
| `prepare_seismic.py` | Builds `data/seis/<key>/` from Marmousi2, Overthrust, Salt, F3 and Stratton |
| `eval_images.py`, `eval_seis.py`, `eval_field.py` | Evaluation: upstream test noise and PSNR/SSIM for images; SNR and local similarity for seismic; no-reference diagnostics for field data |
| `nr2n_correct.py` | Proper Noisier2Noise inference (2f(z) − z) for the nr2n runs |
| `drift_images.py` | Inference-only drift factors D1–D5 |
| `baselines_seis.py`, `bm3d_images.py` | Classical baselines |
| `make_jobs.py`, `pp_queue.py` | Job list and GPU job scheduler (`python paperproto/pp_queue.py`, slots in `runs/paperproto/SLOTS`) |
| `report.py` | Rebuilds `reports/paperproto/RESULTS.md` and its figures from `runs/paperproto/` |

## Running

Run from `repro/`. Examples:

```bash
python paperproto/pp_train.py --domain image --method idr --noise g --seed 0 --out runs/paperproto/A1_idr_g_s0
python paperproto/pp_train.py --domain seis --keys m2,ot,sa --method idr --noise sg --seed 0 --out runs/paperproto/C1_idr_sg_s0
python paperproto/eval_images.py --run runs/paperproto/A1_idr_g_s0
python paperproto/eval_seis.py --run runs/paperproto/C1_idr_sg_s0
```

To run the full study: `python paperproto/make_jobs.py`, then `python paperproto/pp_queue.py`. It ran in 15 h 38 min on one RTX 5070 Ti.

Data (ImageNet val, Kodak, BSDS300, BSD68 and the seismic volumes) is not in the repository. Sources and checksums are listed in `PAPER_PROTOCOL_PLAN.md` and `SEISMIC_SHA256SUMS`.
