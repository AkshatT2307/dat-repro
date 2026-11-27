"""Inference gallery for the models trained BEFORE the Kaggle phase (kratos server, protocol P, 19–20 Nov 2026).

Source: runs/paperproto/{A*,C*,F*,field_*,baselines_seis} unpacked from the kratos runs_small.tgz. Weights were not in
that archive; all panels use the evaluation outputs saved at evaluation time (eval*/examples.npz, field_*/outputs.npz).

  A*  image models  : GT | noisy | output | |output−GT| ×4 | zoom noisy | zoom output   (PSNR on uint8, as in eval)
  C*/F* seismic     : same layout as INFERENCE.md (via kgl/inference_gallery.gallery)
  baselines vs nets : per test set, same noisy section: IDR | Sup | BM3D | f-x | bandpass
  field_f3/_st      : models on the RAW field data

Writes reports/paperproto/figs/infer_prior/*.png and reports/paperproto/INFERENCE_PRIOR.md.
"""
import json, re, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inference_gallery as G

RUNS, REP = G.RUNS, G.REP
OUT = REP / 'figs' / 'infer_prior'; OUT.mkdir(parents=True, exist_ok=True)


def to8(x):
    x = np.asarray(x, np.float64)
    if x.max() <= 2.0:
        x = x * 255.0
    return np.clip(np.round(x), 0, 255)


def psnr8(gt, x):
    mse = np.mean((to8(gt) - to8(x)) ** 2)
    return 10 * np.log10(255 ** 2 / max(mse, 1e-10))


def show(ax, img, title):
    img = to8(img).astype(np.uint8)
    ax.imshow(img[..., 0] if img.shape[-1] == 1 else img, cmap='gray' if img.shape[-1] == 1 else None,
              vmin=0, vmax=255, interpolation='nearest')
    ax.set_title(title, fontsize=8); ax.axis('off')


def image_gallery(run):
    f = RUNS / run / 'eval' / 'examples.npz'
    if not f.exists():
        return None
    z = np.load(f)
    ids = sorted({k.rsplit('__', 1)[0] for k in z.files})
    fig, ax = plt.subplots(len(ids), 6, figsize=(19, 3.1 * len(ids)), squeeze=False)
    for i, k in enumerate(ids):
        gt, noisy, out = z[k + '__gt'], z[k + '__noisy'], z[k + '__out']
        h, w = gt.shape[:2]; cy, cx = h // 2, w // 2; s = 96
        crop = (slice(cy - s, cy + s), slice(cx - s, cx + s))
        show(ax[i, 0], gt, 'ground truth')
        show(ax[i, 1], noisy, f'noisy {psnr8(gt, noisy):.2f} dB')
        show(ax[i, 2], out, f'output {psnr8(gt, out):.2f} dB')
        err = np.abs(to8(out) - to8(gt)).mean(-1)
        ax[i, 3].imshow(np.clip(err * 4, 0, 255), cmap='magma', vmin=0, vmax=255, interpolation='nearest')
        ax[i, 3].set_title('|output − GT| ×4', fontsize=8); ax[i, 3].axis('off')
        show(ax[i, 4], noisy[crop], 'zoom: noisy'); show(ax[i, 5], out[crop], 'zoom: output')
        ax[i, 0].text(-0.05, 0.5, k.replace('_', ' '), transform=ax[i, 0].transAxes, rotation=90, va='center', ha='right', fontsize=8)
    note = ' (stage-0 network applied directly; Nr2N correction 2f(z)−z not applied here)' if 'nr2n' in run else ''
    fig.suptitle(f'{run}: inference examples{note}', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    fig.savefig(OUT / f'{run}.jpg', dpi=90, pil_kwargs={'quality': 85}); plt.close(fig)
    return f'{run}.jpg'


def baseline_compare(noise, idr, sup):
    fb = RUNS / 'baselines_seis' / 'examples.npz'
    fi, fs = RUNS / idr / 'eval' / 'examples.npz', RUNS / sup / 'eval' / 'examples.npz'
    if not (fb.exists() and fi.exists() and fs.exists()):
        return None
    B, I, S = np.load(fb), np.load(fi), np.load(fs)
    keys = [k for k in I.files if k.split('__')[1] == noise and k.split('__')[2] == '0.5']
    meth = [m for m in ('bm3d', 'fx', 'bandpass') if any(x.endswith('__' + m) for x in B.files)]
    fig, ax = plt.subplots(len(keys), 4 + len(meth), figsize=(3.3 * (4 + len(meth)), 2.9 * len(keys)), squeeze=False)
    for i, k in enumerate(keys):
        clean, noisy, oi = I[k].astype(np.float64); os_ = S[k][2].astype(np.float64)
        c = np.percentile(np.abs(clean), 95) * 1.5 or 1.0
        ext = [0, clean.shape[1], clean.shape[0] * G.dt_of(k.split('__')[0]), 0]
        panels = [('clean', clean), (f'noisy {G.snr(clean, noisy):.2f}', noisy), (f'IDR {G.snr(clean, oi):.2f}', oi),
                  (f'Sup {G.snr(clean, os_):.2f}', os_)]
        for m in meth:
            bk = f'{k}__{m}'
            panels.append((f'{m} {G.snr(clean, B[bk].astype(np.float64)):.2f}' if bk in B.files else f'{m} —',
                           B[bk].astype(np.float64) if bk in B.files else np.zeros_like(clean)))
        for j, (t, a) in enumerate(panels):
            ax[i, j].imshow(a, cmap='gray', vmin=-c, vmax=c, aspect='auto', interpolation='nearest', extent=ext)
            ax[i, j].set_title(t + (' dB' if j else ''), fontsize=8); ax[i, j].tick_params(labelsize=6)
        ax[i, 0].set_ylabel(f"{G.NAMES.get(k.split('__')[0], k)}\ntime (s)", fontsize=8)
    fig.suptitle(f'{noise}, σ = 0.5: joint-corpus networks ({idr}, {sup}) vs classical baselines on the same noisy section', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    name = f'baselines_vs_nets_{noise}.jpg'; fig.savefig(OUT / name, dpi=100, pil_kwargs={'quality': 85}); plt.close(fig)
    return name


DESC = {
    'A1': 'Fast IDR, Gaussian RGB', 'A2': 'supervised, Gaussian RGB', 'A3': 'Nr2N stand-in (IDR stage 0 only)', 'A4': 'Noise2Noise',
    'A5': 'Fast IDR, fixed noise realisation', 'A6': 'gray, BSD68', 'A7': 'binomial noise (Table 5)', 'A8': 'impulse noise (Table 5)',
    'A9': 'spatially correlated noise (Table 6)', 'C1': 'Fast IDR, white noise, joint synthetic corpus', 'C2': 'supervised, white noise, joint corpus',
    'C3': 'Nr2N stand-in, white noise', 'C4': 'Noise2Noise, white noise', 'C5': 'IDR, fixed noise realisation (white)',
    'C6': 'IDR, fixed noise realisation (band-limited)', 'C7': 'band-limited noise', 'C8': 'erratic spikes', 'C9': 'dead traces',
    'F1': 'F3 field: IDR on fixed noisy copy / supervised oracle', 'F2': 'F3 field: IDR on raw field noise',
    'F3': 'Stratton field: IDR on fixed noisy copy / supervised oracle', 'F4': 'Stratton field: IDR on raw field noise'}


def main():
    runs = sorted((p.name for p in RUNS.iterdir() if re.match(r'^[ACF]\d+_', p.name) and (p / 'config.json').exists()),
                  key=lambda r: (r[0], int(re.match(r'^[ACF](\d+)', r).group(1)), r))
    L = ['# Inference gallery: models trained before the Kaggle phase (kratos, protocol P)', '',
         'Generated by `kgl/inference_gallery_prior.py` from the evaluation outputs saved when these models were evaluated '
         '(19–20 Nov 2026, RTX 5070 Ti). Checkpoints were not part of the copied archive; the final weights are on '
         'Hugging Face (`glitch23/fast-idr-paper-protocol`). Image PSNR is on uint8 as in the evaluation; seismic SNR in dB. '
         'Seismic panels: middle held-out test section, test-noise seed 0; removed/error panels use a scale tied to the '
         'added noise (printed in the titles). Rendering uses nearest-neighbour (no anti-aliasing).', '',
         '## Contents', '', '1. Image models (A1–A9)', '2. Joint-corpus seismic models (C1–C9)',
         '3. Field models (F1–F4)', '4. Networks vs classical baselines', '5. F3 and Stratton on real field noise', '']
    sec = {'A': '## 1. Image models (Kodak, BSDS300, BSD68)', 'C': '## 2. Joint-corpus seismic models (all 8 test sets)',
           'F': '## 3. Field models (semi-synthetic test: white noise added to the field sections)'}
    cur = None
    for r in runs:
        if r[0] != cur:
            cur = r[0]; L += [sec[cur], '']
        tag = re.match(r'^([ACF]\d+)', r).group(1)
        n = image_gallery(r) if r[0] == 'A' else G.gallery(r, out=OUT)
        if n:
            L += [f'### {r}: {DESC.get(tag, "")}', '', f'![{r}](figs/infer_prior/{n})', '']
        if r == 'C3_nr2n_sg_s0':
            for t in ('eval_nr2ncorr_K1', 'eval_nr2ncorr_K8'):
                n = G.gallery(r, tag=t, title_extra=f' ({t})', out=OUT)
                if n:
                    L += [f'### {r}, Nr2N-corrected ({t.split("_")[-1]})', '', f'![{r} {t}](figs/infer_prior/{n})', '']
        print(r, flush=True)
    L += ['## 4. Networks vs classical baselines (same noisy section)', '']
    for noise, idr, sup in (('sg', 'C1_idr_sg_s0', 'C2_sup_sg_s0'), ('sb', 'C7_idr_sb_s0', 'C7_sup_sb_s0')):
        n = baseline_compare(noise, idr, sup)
        if n:
            L += [f'### {noise}', '', f'![{noise}](figs/infer_prior/{n})', '']
    L += ['## 5. F3 and Stratton: models applied to the raw field data (real noise)', '']
    for k in ('f3', 'st'):
        n = G.field(k, out=OUT, max_sections=3)
        if n:
            L += [f'### {G.NAMES[k]} (first 3 test sections)', '', f'![{k}](figs/infer_prior/{n})', '']
    (REP / 'INFERENCE_PRIOR.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print(len(list(OUT.glob('*.jpg'))), 'figures')


if __name__ == '__main__':
    main()
