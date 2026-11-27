"""Sanity checks for the Kaggle runs (per-dataset S_* and 2-D field G_*).

  1. training curves from metrics.jsonl (loss, lr drops at 25k/40k, teacher refresh every epoch)
  2. example panels from eval/examples.npz (clean / noisy / IDR / Sup / errors), one figure per noise type
  3. mean amplitude spectra (clean, noisy, IDR, Sup) per dataset
  4. LOCAL CPU inference: reload each weights.pth, denoise the stored noisy section, compare to the Kaggle output
     and recompute SNR (independent check that checkpoints, eval and our metric agree)
  5. sanity table + automatic red flags -> reports/paperproto/SANITY.md

Run with a Python that has torch (CPU is fine):  python kgl/sanity.py
"""
import json, sys, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'paperproto'), str(ROOT)]
RUNS = ROOT / 'runs' / 'paperproto'
REP = ROOT / 'reports' / 'paperproto'; FIGS = REP / 'figs'; FIGS.mkdir(parents=True, exist_ok=True)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

NAMES = dict(m2='Marmousi2', ot='Overthrust', sa='Salt', l2='2D land', w2='2D marine')
LV = dict(sg=0.5, sb=0.5, ss=0.05, sd=0.3)
runs = sorted(p for p in RUNS.iterdir() if p.name.startswith(('S_', 'G_')) and (p / 'config.json').exists())
cfg = {r.name: json.loads((r / 'config.json').read_text()) for r in runs}


def snr(clean, x):
    return 10 * np.log10(np.sum(clean ** 2) / max(np.sum((x - clean) ** 2), 1e-20))


# ---------------------------------------------------------------- 1. training curves
groups = {}
for r in runs:
    c = cfg[r.name]
    groups.setdefault(c['noise'] + ('_field' if r.name.startswith('G_') else ''), []).append(r)
fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 3.8), squeeze=False)
for ax, (g, rs) in zip(axes[0], sorted(groups.items())):
    for r in rs:
        m = [json.loads(l) for l in (r / 'metrics.jsonl').read_text().splitlines() if l.strip()]
        st = np.array([x['step'] for x in m]); lo = np.array([x['loss'] for x in m])
        k = max(1, len(lo) // 200)   # smooth: moving mean over ~k points
        lo_s = np.convolve(lo, np.ones(k) / k, mode='valid'); st_s = st[k - 1:]
        ax.plot(st_s, lo_s, lw=1, label=r.name.replace('S_', '').replace('_s0', ''))
    for x in (25000, 40000):
        ax.axvline(x, color='k', ls=':', lw=.8)
    for x in (12500, 37500):
        ax.axvline(x, color='grey', ls='--', lw=.5)
    ax.set_yscale('log'); ax.set_title(f'training loss: {g}'); ax.set_xlabel('step'); ax.legend(fontsize=6)
fig.suptitle('dotted: lr ×0.5 (25k, 40k); dashed: IDR teacher refresh (every 12.5k)', fontsize=8)
fig.tight_layout(); fig.savefig(FIGS / 'sanity_training_curves.png', dpi=110); plt.close(fig)

# ---------------------------------------------------------------- 2 + 3. panels and spectra per noise type
for noise in ('sg', 'sb', 'ss', 'sd'):
    rows = []
    for k in ('m2', 'ot', 'sa'):
        ri, rs = RUNS / f'S_{k}_idr_{noise}_s0', RUNS / f'S_{k}_sup_{noise}_s0'
        ex = {}
        for tag, r in (('IDR', ri), ('Sup', rs)):
            f = r / 'eval' / 'examples.npz'
            if f.exists():
                z = np.load(f); key = f'{k}__{noise}__{LV[noise]}'
                if key in z.files:
                    ex[tag] = z[key]
        if ex:
            rows.append((k, ex))
    if not rows:
        continue
    fig, ax = plt.subplots(len(rows), 6, figsize=(18, 3.2 * len(rows)), squeeze=False)
    for i, (k, ex) in enumerate(rows):
        base = next(iter(ex.values())); clean, noisy = base[0], base[1]
        c = np.percentile(np.abs(clean), 95) * 1.5   # 99th pct is dominated by the M2 seafloor event
        panels = [('clean', clean), (f'noisy {snr(clean, noisy):.1f} dB', noisy)]
        for tag in ('IDR', 'Sup'):
            o = ex[tag][2] if tag in ex else np.zeros_like(clean)
            panels.append((f'{tag} {snr(clean, o):.2f} dB' if tag in ex else f'{tag} —', o))
        for tag in ('IDR', 'Sup'):
            o = ex[tag][2] if tag in ex else clean
            panels.append((f'{tag} error (out−clean) ×5', 5 * (o - clean)))
        for j, (t, a) in enumerate(panels):
            ax[i, j].imshow(a, cmap='gray', vmin=-c, vmax=c, aspect='auto', interpolation='nearest'); ax[i, j].set_title(t, fontsize=9)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(NAMES[k])
    fig.suptitle(f'per-dataset models, noise {noise}, level {LV[noise]} (middle test section, test seed 0)')
    fig.tight_layout(); fig.savefig(FIGS / f'sanity_panels_{noise}.png', dpi=100); plt.close(fig)

    fig, ax = plt.subplots(1, len(rows), figsize=(5 * len(rows), 3.4), squeeze=False)
    for a, (k, ex) in zip(ax[0], rows):
        base = next(iter(ex.values())); dt = json.loads((ROOT / 'data/seis' / k / 'meta.json').read_text())['dt']
        f = np.fft.rfftfreq(base.shape[1], dt)
        A = lambda x: 20 * np.log10(np.abs(np.fft.rfft(x, axis=0)).mean(1) + 1e-9)
        a.plot(f, A(base[0]), 'k', lw=1.5, label='clean'); a.plot(f, A(base[1]), color='0.7', lw=1, label='noisy')
        for tag, col in (('IDR', 'C0'), ('Sup', 'C3')):
            if tag in ex:
                a.plot(f, A(ex[tag][2]), col, lw=1, label=tag)
        a.set_title(f'{NAMES[k]} ({noise})'); a.set_xlabel('Hz'); a.set_ylabel('dB'); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / f'sanity_spectra_{noise}.png', dpi=100); plt.close(fig)

# ---------------------------------------------------------------- 4. local CPU inference check
local = {}
try:
    import torch
    from models.unet import UNet_n2n_un
    import torch.nn.functional as F
    torch.set_num_threads(8)
    for r in runs:
        c = cfg[r.name]; f = r / 'eval' / 'examples.npz'
        if not (r / 'weights.pth').exists() or not f.exists():
            continue
        z = np.load(f); own = c['keys'].split(',')[0]
        key = next((k for k in z.files if k.split('__')[0] == own and k.split('__')[1] == c['noise']), None) \
            or next((k for k in z.files if k.split('__')[1] == c['noise']), None)
        if key is None:
            continue
        clean, noisy, out_k = z[key]
        net = UNet_n2n_un(1, 1).eval()
        net.load_state_dict(torch.load(r / 'weights.pth', map_location='cpu', weights_only=True))
        h, w = noisy.shape; ph, pw = (h + 31) // 32 * 32 - h, (w + 31) // 32 * 32 - w
        with torch.no_grad():
            t = F.pad(torch.from_numpy(noisy.astype(np.float32))[None, None], (0, pw, 0, ph), mode='reflect')
            out_l = net(t)[0, 0, :h, :w].numpy()
        local[r.name] = dict(key=key, maxdiff=float(np.abs(out_l - out_k).max()),
                             snr_kaggle=float(snr(clean.astype(float), out_k.astype(float))),
                             snr_local=float(snr(clean.astype(float), out_l.astype(float))))
        print('local', r.name, local[r.name], flush=True)
except ImportError as e:
    print('torch not available, skipping local inference:', e)

# ---------------------------------------------------------------- 5. table + red flags
L = ['# Sanity checks: Kaggle runs (27 Nov 2026)', '',
     'Generated by `kgl/sanity.py` from the downloaded run folders. Local inference is on CPU (PyTorch), using the '
     'downloaded `weights.pth` and the noisy section stored by the Kaggle evaluation.', '',
     'Own-test SNR is the mean over test sections × 3 noise seeds; the local check uses one stored section (middle, seed 0).', '',
     '| run | steps | it/s | final loss (mean last 1k) | own-test SNR in → out (dB) | gain | local−Kaggle max|Δ| | 1-section SNR local / Kaggle |',
     '|---|---|---|---|---|---|---|---|']
flags = []
for r in runs:
    c = cfg[r.name]
    m = [json.loads(l) for l in (r / 'metrics.jsonl').read_text().splitlines() if l.strip()]
    tail = [x['loss'] for x in m if x['step'] > m[-1]['step'] - 1000]
    fl = float(np.mean(tail))
    ev = {}
    if (r / 'eval' / 'summary.json').exists():
        for x in json.loads((r / 'eval' / 'summary.json').read_text())['results']:
            ev[(x['test'], round(x['level'], 3))] = x
    own = c['keys'].split(',')[0]
    e = ev.get((own, LV.get(c['noise'], 0.5)))
    lc = local.get(r.name)
    L.append(f"| {r.name} | {m[-1]['step']} | {m[-1]['steps_per_s']:.2f} | {fl:.4f} | "
             + (f"{e['snr_in']:.2f} → {e['snr']:.2f} | {e['snr_gain']:+.2f}" if e else '— | —') + ' | '
             + (f"{lc['maxdiff']:.1e} | {lc['snr_local']:.2f} / {lc['snr_kaggle']:.2f}" if lc else '— | —') + ' |')
    if m[-1]['step'] != 50000: flags.append(f'{r.name}: stopped at step {m[-1]["step"]}')
    if not math.isfinite(fl): flags.append(f'{r.name}: non-finite loss')
    if e and e['snr_gain'] <= 0: flags.append(f'{r.name}: no SNR gain on own test ({e["snr_gain"]:+.2f} dB)')
    if lc and lc['maxdiff'] > 1e-2: flags.append(f'{r.name}: local inference differs from Kaggle (max |Δ| {lc["maxdiff"]:.2e})')
# IDR should not beat its supervised upper bound by a lot on the training survey (would hint at a bug or leakage)
for k in ('m2', 'ot', 'sa'):
    for n in ('sg', 'sb', 'ss', 'sd'):
        fi, fs = RUNS / f'S_{k}_idr_{n}_s0/eval/summary.json', RUNS / f'S_{k}_sup_{n}_s0/eval/summary.json'
        if fi.exists() and fs.exists():
            gi = {(x['test'], round(x['level'], 3)): x['snr'] for x in json.loads(fi.read_text())['results']}
            gs = {(x['test'], round(x['level'], 3)): x['snr'] for x in json.loads(fs.read_text())['results']}
            d = gi[(k, LV[n])] - gs[(k, LV[n])]
            if d > 0.5:
                flags.append(f'S_{k}_*_{n}: IDR beats supervised by {d:+.2f} dB on its own training survey (check)')
L += ['', '## Red flags', ''] + ([f'- {x}' for x in flags] or ['- none'])

NOTES = """
## Follow-up on red flags

**S_m2_*_sg (IDR +1.30 dB above supervised on Marmousi2).** Checked by scoring the 10k–50k checkpoints on the m2 test set
(σ 0.5, 3 seeds, local CPU inference):

| step | IDR | Sup |
|---|---|---|
| 10k | 8.20 | **19.14** |
| 20k | 10.51 | 16.04 |
| 30k | 19.20 | 18.35 |
| 40k | 19.06 | 18.35 |
| 50k | **19.47** | 18.17 |

Supervised peaks at 10k and then over-fits its single 1001×1400 training section. IDR is low before the first teacher
refresh (stage-0 network under-denoises, as expected) and keeps improving. **Not a bug: a small-data over-fitting
effect.** Fair statement: IDR at 50k ≈ supervised with early stopping (19.47 vs 19.14); the joint-corpus supervised model
(more data) reaches 19.34.
"""
L += ['', NOTES, '## Figures', '', '![training curves](figs/sanity_training_curves.png)', '']
for n in ('sg', 'sb', 'ss', 'sd'):
    if (FIGS / f'sanity_panels_{n}.png').exists():
        L += [f'![panels {n}](figs/sanity_panels_{n}.png)', '', f'![spectra {n}](figs/sanity_spectra_{n}.png)', '']
(REP / 'SANITY.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
print('\n'.join(L[:len(runs) + 6])); print('\n'.join(L[L.index('## Red flags'):L.index('## Figures')]))
