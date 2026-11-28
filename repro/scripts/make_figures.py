"""Publication figures for the IDR reproduction / seismic manuscript (report/main.tex).

Reads only stored run outputs under runs/paperproto/ (summary.json, examples.npz,
vis_target_epoch*.npy, metrics.jsonl, field outputs.npz, checkpoints). Nothing is
re-trained. The only inference performed is the Marmousi2 checkpoint sweep (CPU,
10 checkpoints x 1 section x 3 test-noise seeds), cached in report/generated/.

Usage (from repro/):  python scripts/make_figures.py [--skip-sweep]
Outputs:  report/figures/*.pdf (500 dpi raster content, vector text)
          report/generated/sweep_m2.json
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

R = Path(__file__).resolve().parents[1]
RUNS = R / 'runs' / 'paperproto'
OUT = R / 'report' / 'figures'
GEN = R / 'report' / 'generated'
OUT.mkdir(parents=True, exist_ok=True); GEN.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(R / 'paperproto'), str(R)]

# ---------------------------------------------------------------- style
COL, TXT = 3.5, 7.16              # IEEE column / text width (in)
# Okabe-Ito order (validated: CVD adjacent dE 7.6 -> always paired with marker/linestyle)
C = dict(idr='#0072B2', sup='#D55E00', bm3d='#009E73', fx='#CC79A7', bp='#E69F00', n2n='#56B4E9', ink='#222222', grid='#d0d0d0')
MK = dict(idr='o', sup='s', bm3d='^', fx='D', bp='v', n2n='P')
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'legend.frameon': False, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5,
    'ytick.major.width': 0.5, 'lines.linewidth': 1.2, 'lines.markersize': 4, 'axes.spines.top': False,
    'axes.spines.right': False, 'savefig.dpi': 500, 'figure.dpi': 150, 'pdf.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'image.interpolation': 'none',
})
DT = 0.004


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf')
    plt.close(fig)
    print('wrote', OUT / f'{name}.pdf')


def summ(run, tag='eval'):
    f = RUNS / run / tag / 'summary.json'
    rows = json.loads(f.read_text())['results']
    if isinstance(rows, dict):
        return rows
    key = (lambda r: (r['dataset'], r['level'])) if 'dataset' in rows[0] else \
          (lambda r: (r.get('method'), r['test'], r['noise'], round(r['level'], 3)) if 'method' in r else (r['test'], r['noise'], round(r['level'], 3)))
    return {key(r): r for r in rows}


def snr(ref, x):
    return 10 * np.log10(np.sum(ref ** 2) / np.sum((x - ref) ** 2))


def psnr01(ref, x):
    return 10 * np.log10(1 / np.mean((np.clip(x, 0, 1) - ref) ** 2))


def seis_show(ax, a, clip, title=None, ylabel=None):
    ax.imshow(a, cmap='gray', vmin=-clip, vmax=clip, aspect='auto', interpolation='none')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.4)
    if title:
        ax.set_title(title, pad=2)
    if ylabel:
        ax.set_ylabel(ylabel)


def label(ax, txt, loc='lr'):
    x, ha = (0.97, 'right') if loc[1] == 'r' else (0.03, 'left')
    y, va = (0.03, 'bottom') if loc[0] == 'l' else (0.97, 'top')
    ax.text(x, y, txt, transform=ax.transAxes, ha=ha, va=va, fontsize=6.5, color='k',
            bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.85))


# ---------------------------------------------------------------- Fig: target refinement
def fig_refinement():
    fig, axs = plt.subplots(2, 5, figsize=(TXT, 2.95), gridspec_kw=dict(wspace=0.04, hspace=0.18))
    # images (A1, sample 0); vis targets are stored in [0,1] (shift undone)
    v = np.load(RUNS / 'A1_idr_g_s0' / 'vis_input.npz')
    clean = v['clean'][0].transpose(1, 2, 0)
    tg = [np.load(RUNS / 'A1_idr_g_s0' / f'vis_target_epoch{e}.npy')[0].transpose(1, 2, 0) for e in range(4)]
    panels = [(clean, 'Clean $y$')] + [(tg[0], 'Noisy $x$ (target, round 0)')] + \
             [(tg[e], f'Target round {e}') for e in (1, 2, 3)]
    for ax, (im, t) in zip(axs[0], panels):
        ax.imshow(np.clip(im, 0, 1)); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, pad=2)
        if im is not clean:
            label(ax, f'{psnr01(clean, im):.2f} dB')
    axs[0, 0].set_ylabel('ImageNet, Gaussian')
    # seismic (C1 joint corpus, sample 0)
    v = np.load(RUNS / 'C1_idr_sg_s0' / 'vis_input.npz')
    clean = v['clean'][0, 0]
    tg = [np.load(RUNS / 'C1_idr_sg_s0' / f'vis_target_epoch{e}.npy')[0, 0] for e in range(4)]
    clip = np.percentile(np.abs(clean), 99)
    seis_show(axs[1, 0], clean, clip, ylabel='Seismic, white')
    for e in range(4):
        seis_show(axs[1, e + 1], tg[e], clip); label(axs[1, e + 1], f'SNR {snr(clean, tg[e]):.2f} dB')
    save(fig, 'fig_refinement')


def target_quality(run, n=4):
    v = np.load(RUNS / run / 'vis_input.npz')
    out = []
    for e in range(4):
        t = np.load(RUNS / run / f'vis_target_epoch{e}.npy')
        if run.startswith('A'):
            out.append([psnr01(v['clean'][i], t[i]) for i in range(n)])
        else:
            out.append([snr(v['clean'][i, 0], t[i, 0]) for i in range(n)])
    return np.array(out)  # (rounds, samples)


def loss_curve(run, win=20):
    m = [json.loads(l) for l in (RUNS / run / 'metrics.jsonl').read_text().splitlines() if l.strip()]
    s = np.array([r['step'] for r in m]); L = np.array([r['loss'] for r in m])
    k = np.ones(win) / win
    return s[win - 1:], np.convolve(L, k, 'valid')


def load_unet():
    """Import models/unet.py without the package __init__ (which pulls in bm3d via utils)."""
    import types, importlib.util
    reg = types.ModuleType('utils.registry')
    reg.ARCH_REGISTRY = types.SimpleNamespace(register=lambda: (lambda c: c))
    sys.modules.setdefault('utils', types.ModuleType('utils')); sys.modules['utils.registry'] = reg
    spec = importlib.util.spec_from_file_location('unet_src', R / 'models' / 'unet.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.UNet_n2n_un


# ---------------------------------------------------------------- checkpoint sweep (Marmousi2, single-survey models)
def sweep_m2():
    f = GEN / 'sweep_m2.json'
    if f.exists():
        return json.loads(f.read_text())
    import torch, torch.nn.functional as F
    UNet_n2n_un = load_unet()
    import seis_common as S
    secs, meta = S.load_test('m2')
    res = {}
    for run in ('S_m2_idr_sg_s0', 'S_m2_sup_sg_s0'):
        res[run] = {}
        for step in (10000, 20000, 30000, 40000, 50000):
            net = UNet_n2n_un(1, 1).eval()
            net.load_state_dict(torch.load(RUNS / run / f'weights_step{step}.pth', map_location='cpu', weights_only=True))
            vals = []
            for seed in S.TEST_SEEDS:
                for i, cl in enumerate(secs):
                    cl = cl.astype(np.float64)
                    noisy, _ = S.test_noise(cl, 'sg', 0.5, seed, meta['dt'], 'm2', i)
                    h, w = noisy.shape; ph, pw = (h + 31) // 32 * 32 - h, (w + 31) // 32 * 32 - w
                    with torch.no_grad():
                        t = F.pad(torch.from_numpy(noisy.astype(np.float32))[None, None], (0, pw, 0, ph), mode='reflect')
                        out = net(t)[0, 0, :h, :w].numpy().astype(np.float64)
                    vals.append(S.metrics(cl, noisy, out)['snr'])
            res[run][step] = float(np.mean(vals))
            print(run, step, res[run][step], flush=True)
    f.write_text(json.dumps(res, indent=2))
    return res


def fig_convergence(do_sweep=True):
    fig, axs = plt.subplots(1, 4, figsize=(TXT, 1.75), gridspec_kw=dict(wspace=0.42))
    # (a) training loss, image
    ax = axs[0]
    for run, k, lab in (('A1_idr_g_s0', 'idr', 'Fast IDR'), ('A2_sup_g_s0', 'sup', 'Supervised')):
        s, L = loss_curve(run); ax.plot(s / 1e3, L, color=C[k], lw=0.8, label=lab)
    for x in (12.5, 25, 37.5):
        ax.axvline(x, color=C['grid'], lw=0.6, ls='--', zorder=0)
    ax.set_xlabel('Step (k)'); ax.set_ylabel('Training loss ($L_1$)'); ax.set_title('(a) Image training loss')
    ax.legend(loc='upper right', handlelength=1.2)
    # (b) target PSNR over rounds (image)
    ax = axs[1]; q = target_quality('A1_idr_g_s0')
    for i in range(q.shape[1]):
        ax.plot(range(4), q[:, i], color=C['idr'], lw=0.5, alpha=0.35)
    ax.plot(range(4), q.mean(1), color=C['idr'], marker='o', label='mean of 4 crops')
    ax.set_xticks(range(4)); ax.set_xlabel('Refinement round $m$'); ax.set_ylabel('Target PSNR (dB)')
    ax.set_title('(b) Image targets'); ax.legend(loc='lower right', handlelength=1.2)
    # (c) target SNR over rounds (seismic)
    ax = axs[2]; q = target_quality('C1_idr_sg_s0')
    for i in range(q.shape[1]):
        ax.plot(range(4), q[:, i], color=C['idr'], lw=0.5, alpha=0.35)
    ax.plot(range(4), q.mean(1), color=C['idr'], marker='o')
    ax.set_xticks(range(4)); ax.set_xlabel('Refinement round $m$'); ax.set_ylabel('Target SNR (dB)')
    ax.set_title('(c) Seismic targets')
    # (d) Marmousi2 checkpoint sweep
    ax = axs[3]
    if do_sweep:
        sw = sweep_m2()
        for run, k, lab in (('S_m2_idr_sg_s0', 'idr', 'Fast IDR'), ('S_m2_sup_sg_s0', 'sup', 'Supervised')):
            st = sorted(int(s) for s in sw[run]); ax.plot([s / 1e3 for s in st], [sw[run][str(s)] if str(s) in sw[run] else sw[run][s] for s in st],
                                                        color=C[k], marker=MK[k], label=lab)
        ax.set_xlabel('Step (k)'); ax.set_ylabel('Test SNR (dB)'); ax.set_title('(d) Marmousi2-only models')
        ax.legend(loc='lower right', handlelength=1.2)
    for a in axs:
        a.grid(True, color=C['grid'], lw=0.3); a.set_axisbelow(True)
    save(fig, 'fig_convergence')


# ---------------------------------------------------------------- image fidelity and per-image scatter
PAPER_T2 = {  # Zhang et al. 2022, Table 2 (PSNR): (IDR, Supervised)
    ('kodak', 25): (32.36, 32.39), ('kodak', 50): (29.27, 29.24), ('BSDS300', 25): (31.48, 31.40),
    ('BSDS300', 50): (28.25, 28.17), ('BSD68', 25): (29.20, 29.20), ('BSD68', 50): (26.25, 26.24)}
NAME = {'kodak': 'Kodak', 'BSDS300': 'BSDS300', 'BSD68': 'BSD68'}


def image_scores():
    """mean, min, max over seeds (RGB rows); single seed for gray BSD68 (A6)."""
    out = {}
    for (ds, lv) in PAPER_T2:
        if ds == 'BSD68':
            i = [summ('A6_idr_gray_s0')[(ds, lv)]['psnr']]; s = [summ('A6_sup_gray_s0')[(ds, lv)]['psnr']]
        else:
            i = [summ(f'A1_idr_g_s{k}')[(ds, lv)]['psnr'] for k in range(3)]
            s = [summ(f'A2_sup_g_s{k}')[(ds, lv)]['psnr'] for k in range(3)]
        out[(ds, lv)] = (np.array(i), np.array(s))
    return out


def fig_fidelity():
    sc = image_scores()
    rows = list(PAPER_T2)
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.9), sharey=True, gridspec_kw=dict(wspace=0.12))
    y = np.arange(len(rows))[::-1]
    for ax, j, k, t in ((axs[0], 0, 'idr', '(a) Fast IDR'), (axs[1], 1, 'sup', '(b) Supervised')):
        d = np.array([sc[r][j].mean() - PAPER_T2[r][j] for r in rows])
        lo = np.array([sc[r][j].min() - PAPER_T2[r][j] for r in rows]); hi = np.array([sc[r][j].max() - PAPER_T2[r][j] for r in rows])
        ax.axvline(0, color=C['ink'], lw=0.5)
        ax.hlines(y, 0, d, color=C[k], lw=1.0)
        ax.errorbar(d, y, xerr=[d - lo, hi - d], fmt=MK[k], color=C[k], ms=4, capsize=1.5, elinewidth=0.6)
        for yy, dd in zip(y, d):
            ax.text(dd - 0.05, yy, f'{dd:+.2f}', ha='right', va='center', fontsize=6)
        ax.set_xlim(-0.8, 0.08); ax.set_title(t); ax.set_xlabel('Ours $-$ paper PSNR (dB)')
        ax.grid(True, axis='x', color=C['grid'], lw=0.3); ax.set_axisbelow(True)
    axs[0].set_yticks(y); axs[0].set_yticklabels([f'{NAME[d]} $\\sigma$={l}' for d, l in rows])
    save(fig, 'fig_fidelity')


def fig_scatter():
    import csv
    def read(run):
        with open(RUNS / run / 'eval' / 'per_image.csv') as f:
            return {(r['dataset'], int(r['level']), r['image']): float(r['psnr']) for r in csv.DictReader(f)}
    a, b = read('A1_idr_g_s0'), read('A2_sup_g_s0')
    keys = [k for k in a if k in b]
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.75), gridspec_kw=dict(wspace=0.45))
    for ax, lv in zip(axs, (25, 50)):
        ks = [k for k in keys if k[1] == lv]
        s = np.array([b[k] for k in ks]); d = np.array([a[k] - b[k] for k in ks])
        isk = np.array([k[0] == 'kodak' for k in ks])
        ax.axhline(0, color=C['ink'], lw=0.5)
        ax.scatter(s[~isk], d[~isk], s=3, color=C['idr'], marker='o', lw=0, alpha=0.7, label='BSDS300')
        ax.scatter(s[isk], d[isk], s=8, color=C['sup'], marker='s', lw=0, label='Kodak')
        ax.set_xlabel('Supervised PSNR (dB)'); ax.set_title(f'$\\sigma$={lv}: {int((d > 0).sum())}/{len(d)} above 0')
        ax.grid(True, color=C['grid'], lw=0.3); ax.set_axisbelow(True)
    axs[0].set_ylabel('IDR $-$ Sup. (dB)'); axs[1].legend(loc='lower right', markerscale=1.5, handletextpad=0.2)
    save(fig, 'fig_scatter')


# ---------------------------------------------------------------- synthetic seismic panels
SURV = {'m2_prestm': 'M2 PreSTM (unseen)', 'ot': 'Overthrust', 'sa': 'Salt', 'm2': 'Marmousi2'}
ZOOM = {'m2_prestm': (420, 620, 150, 350), 'ot': (150, 330, 300, 480), 'sa': (250, 450, 200, 400)}


def synth_arrays(key, noise='sg', lv=0.5):
    i = np.load(RUNS / 'C1_idr_sg_s0' / 'eval' / 'examples.npz')[f'{key}__{noise}__{lv}']
    s = np.load(RUNS / 'C2_sup_sg_s0' / 'eval' / 'examples.npz')[f'{key}__{noise}__{lv}']
    b = np.load(RUNS / 'baselines_seis' / 'examples.npz')
    clean, noisy = i[0].astype(np.float64), i[1].astype(np.float64)
    assert np.allclose(clean, s[0]) and np.allclose(noisy, s[1]), 'IDR and Sup examples are not the same noisy section'
    return dict(clean=clean, noisy=noisy, idr=i[2].astype(np.float64), sup=s[2].astype(np.float64),
                bm3d=b[f'{key}__{noise}__{lv}__bm3d'].astype(np.float64), fx=b[f'{key}__{noise}__{lv}__fx'].astype(np.float64))


def fig_synth(keys=('m2_prestm', 'ot', 'sa'), lv=0.5):
    cols = [('clean', 'Clean'), ('noisy', 'Noisy'), ('bm3d', 'BM3D (oracle $\\sigma$)'), ('fx', '$f$-$x$ decon.'),
            ('idr', 'Fast IDR'), ('sup', 'Supervised'), ('rem', 'Removed (IDR)'), ('err', 'Error $\\times$2')]
    fig, axs = plt.subplots(2 * len(keys), len(cols), figsize=(TXT, 1.62 * 2 * len(keys) * 0.62),
                            gridspec_kw=dict(wspace=0.03, hspace=0.06, height_ratios=[1.25, 1] * len(keys)))
    for r, key in enumerate(keys):
        d = synth_arrays(key, lv=lv)
        d['rem'] = d['noisy'] - d['idr']; d['err'] = 2 * (d['idr'] - d['clean'])
        clip = np.percentile(np.abs(d['clean']), 98)
        t0, t1, x0, x1 = ZOOM[key]
        for c, (k, t) in enumerate(cols):
            a = d[k]
            seis_show(axs[2 * r, c], a, clip, title=t if r == 0 else None, ylabel=SURV[key] if c == 0 else None)
            seis_show(axs[2 * r + 1, c], a[t0:t1, x0:x1], clip, ylabel='zoom' if c == 0 else None)
            if k in ('noisy', 'bm3d', 'fx', 'idr', 'sup'):
                label(axs[2 * r, c], f'{snr(d["clean"], a):.2f}')
            if c == 0:
                axs[2 * r, c].add_patch(Rectangle((x0, t0), x1 - x0, t1 - t0, fill=False, ec=C['bp'], lw=0.8))
        axs[2 * r, 0].yaxis.label.set_size(7)
    save(fig, 'fig_synth')


def fk(a):
    s = np.abs(np.fft.fftshift(np.fft.fft2(a), axes=1))[: a.shape[0] // 2 + 1]
    return 20 * np.log10(s / s.max() + 1e-8)


def fig_fk(key='m2_prestm', lv=0.5):
    d = synth_arrays(key, lv=lv)
    nt, nx = d['clean'].shape
    f = np.fft.rfftfreq(nt, DT)
    pan = [('clean', 'Clean'), ('noisy', 'Noisy'), ('bm3d', 'BM3D'), ('idr', 'Fast IDR'), ('sup', 'Supervised')]
    fig = plt.figure(figsize=(TXT, 1.85))
    gs = fig.add_gridspec(1, 8, width_ratios=[1, 1, 1, 1, 1, 0.08, 0.42, 2.0], wspace=0.08)
    ref = np.abs(np.fft.fftshift(np.fft.fft2(d['clean']), axes=1))[: nt // 2 + 1].max()
    for i, (k, t) in enumerate(pan):
        ax = fig.add_subplot(gs[0, i])
        s = np.abs(np.fft.fftshift(np.fft.fft2(d[k]), axes=1))[: nt // 2 + 1]
        im = ax.imshow(20 * np.log10(s / ref + 1e-8), cmap='magma', vmin=-60, vmax=0, aspect='auto', origin='upper',
                       extent=[-0.5, 0.5, f[-1], 0], interpolation='none')
        ax.set_ylim(100, 0); ax.set_title(t, pad=2); ax.set_xlabel('$k$ (cyc/trace)')
        ax.set_xticks([-0.4, 0, 0.4])
        if i == 0:
            ax.set_ylabel('Frequency (Hz)')
        else:
            ax.set_yticklabels([])
    cax = fig.add_subplot(gs[0, 5]); cb = fig.colorbar(im, cax=cax); cb.set_label('dB', labelpad=1); cb.outline.set_linewidth(0.4)
    ax = fig.add_subplot(gs[0, 7])
    for k, lab, col, ls in (('clean', 'Clean', C['ink'], '-'), ('noisy', 'Noisy', '#999999', '-'), ('bm3d', 'BM3D', C['bm3d'], ':'),
                            ('idr', 'Fast IDR', C['idr'], '-'), ('sup', 'Supervised', C['sup'], '--')):
        A = np.abs(np.fft.rfft(d[k], axis=0)).mean(1)
        ax.plot(f, 20 * np.log10(A / np.abs(np.fft.rfft(d['clean'], axis=0)).mean(1).max()), color=col, ls=ls, lw=0.9, label=lab)
    ax.set_xlim(0, 125); ax.set_ylim(-70, 3); ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Mean amplitude (dB)')
    ax.yaxis.set_label_position('right'); ax.yaxis.tick_right(); ax.spines['right'].set_visible(True); ax.spines['left'].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=3, handlelength=1.6, columnspacing=0.8); ax.grid(True, color=C['grid'], lw=0.3); ax.set_title('Average spectrum', pad=2)
    save(fig, 'fig_fk')


# ---------------------------------------------------------------- field data
FIELD = [('l2', '2D land (NPR-A)', 'idr_real', 0), ('w2', '2D marine (NAMSS)', 'idr_real', 0),
         ('st', '3D land (Stratton)', 'idr_real', 4), ('f3', '3D marine (F3)', 'idr_real', 5)]


def fig_field():
    cols = ['Raw', 'Fast IDR (survey)', 'Removed $\\times$10 (survey)', 'Removed $\\times$10 (synthetic)']
    fig, axs = plt.subplots(4, 4, figsize=(TXT, 6.4), gridspec_kw=dict(wspace=0.04, hspace=0.16, height_ratios=[1.2, 1.2, 1.2, 0.75]))
    for r, (k, name, mdl, i) in enumerate(FIELD):
        z = np.load(RUNS / f'field_{k}' / 'outputs.npz'); sm = json.loads((RUNS / f'field_{k}' / 'summary.json').read_text())['results']
        raw = z['raw'][i].astype(np.float64); out = z[mdl][i]; syn = z['idr_synth'][i]
        if k in ('l2', 'w2'):        # crop first 6 s of the 512-trace test section for legibility
            raw, out, syn = raw[:1500], out[:1500], syn[:1500]
        clip = np.percentile(np.abs(raw), 99)
        seis_show(axs[r, 0], raw, clip, title=cols[0] if r == 0 else None, ylabel=name)
        seis_show(axs[r, 1], out, clip, title=cols[1] if r == 0 else None)
        seis_show(axs[r, 2], 10 * (raw - out), clip, title=cols[2] if r == 0 else None)
        seis_show(axs[r, 3], 10 * (raw - syn), clip, title=cols[3] if r == 0 else None)
        for c, m in ((2, mdl), (3, 'idr_synth')):
            label(axs[r, c], f'{100 * sm[m]["removed_energy_frac"]:.2f}% | LS {sm[m]["leak_ls"]:.2f}')
        axs[r, 0].yaxis.label.set_size(7)
    save(fig, 'fig_field')


def fig_field_spectra():
    fig, axs = plt.subplots(1, 4, figsize=(TXT, 1.6), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, (k, name, mdl, _) in zip(axs, FIELD):
        z = np.load(RUNS / f'field_{k}' / 'outputs.npz'); raw = z['raw'].astype(np.float64)
        f = np.fft.rfftfreq(raw.shape[1], DT)
        spec = lambda a: np.abs(np.fft.rfft(a, axis=1)).mean((0, 2))
        ref = spec(raw).max()
        ax.plot(f, 20 * np.log10(spec(raw) / ref), color=C['ink'], lw=0.9, label='Raw')
        ax.plot(f, 20 * np.log10(spec(raw - z[mdl]) / ref + 1e-12), color=C['idr'], lw=0.9, label='Removed, IDR (survey)')
        ax.plot(f, 20 * np.log10(spec(raw - z['idr_synth']) / ref + 1e-12), color=C['sup'], ls='--', lw=0.9, label='Removed, IDR (synthetic)')
        ax.axvspan(5, 60, color=C['grid'], alpha=0.35, lw=0, zorder=0)
        ax.set_xlim(0, 125); ax.set_ylim(-90, 3); ax.set_title(name, pad=2); ax.set_xlabel('Frequency (Hz)')
        ax.grid(True, color=C['grid'], lw=0.3); ax.set_axisbelow(True)
    axs[0].set_ylabel('Amplitude (dB)')
    axs[1].legend(loc='lower left', handlelength=1.4, fontsize=6)
    save(fig, 'fig_field_spectra')


# ---------------------------------------------------------------- cross-dataset summary
def fig_summary():
    fig, axs = plt.subplots(1, 2, figsize=(TXT, 2.0), gridspec_kw=dict(wspace=0.55, width_ratios=[1.1, 1]))
    # (a) IDR - Sup over 3 training seeds, sg sigma 0.5 (joint corpus)
    tests = ['m2', 'm2_prestm', 'm2_postm', 'm2_srme', 'ot', 'sa', 'f3', 'st']
    lab = ['Marmousi2', 'M2 PreSTM$^\\ast$', 'M2 PostSTM$^\\ast$', 'M2 SRME$^\\ast$', 'Overthrust', 'Salt', 'F3 (field)', 'Stratton (field)']
    ax = axs[0]; y = np.arange(len(tests))[::-1]
    for lv, mk, off, col in ((0.5, 'o', 0.15, C['idr']), (1.0, 's', -0.15, C['sup'])):
        d = np.array([[summ(f'C1_idr_sg_s{s}')[(t, 'sg', lv)]['snr'] - summ(f'C2_sup_sg_s{s}')[(t, 'sg', lv)]['snr'] for s in range(3)] for t in tests])
        ax.errorbar(d.mean(1), y + off, xerr=[d.mean(1) - d.min(1), d.max(1) - d.mean(1)], fmt=mk, color=col, ms=3.5,
                    capsize=1.5, elinewidth=0.6, label=f'$\\sigma$={lv}')
    ax.axvline(0, color=C['ink'], lw=0.5); ax.set_yticks(y); ax.set_yticklabels(lab)
    ax.set_xlabel('SNR, Fast IDR $-$ supervised (dB)'); ax.set_title('(a) Joint-corpus models, 3 seeds (min/mean/max)')
    ax.legend(loc='lower right', handletextpad=0.2); ax.grid(True, axis='x', color=C['grid'], lw=0.3); ax.set_axisbelow(True)
    # (b) field semi-synthetic, sigma 0.5
    ax = axs[1]
    surveys = [('l2', '2D land'), ('w2', '2D marine'), ('st', '3D land'), ('f3', '3D marine')]
    vals = field_semisynth()
    meths = [('idr_survey', 'Fast IDR (survey)', 'idr'), ('best_classical', 'Best classical', 'bm3d'),
             ('sup_synth', 'Supervised (synthetic)', 'sup'), ('idr_synth', 'Fast IDR (synthetic)', 'n2n')]
    x = np.arange(len(surveys))
    for j, (m, l, k) in enumerate(meths):
        ax.plot(x + (j - 1.5) * 0.12, [vals[s][m][0.5] for s, _ in surveys], MK[k], color=C[k], ms=4, ls='none', label=l)
    ax.plot(x, [vals[s]['input'][0.5] for s, _ in surveys], '_', color=C['ink'], ms=12, mew=1.0, label='Input')
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in surveys]); ax.set_ylabel('SNR (dB), $\\sigma$=0.5')
    ax.set_title('(b) Field data, semi-synthetic'); ax.grid(True, axis='y', color=C['grid'], lw=0.3); ax.set_axisbelow(True)
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), handletextpad=0.2)
    save(fig, 'fig_summary')


def field_semisynth():
    """SNR on field sections with added white noise. Best classical = max over bandpass/f-x/BM3D present."""
    out = {}
    for k, idr_run, sup_run in (('f3', 'F1_idrfix_f3_sg', 'F1_sup_f3_sg_oracle'), ('st', 'F3_idrfix_st_sg', 'F3_sup_st_sg_oracle'),
                                ('l2', 'G_l2_idrfix_sg', None), ('w2', 'G_w2_idrfix_sg', None)):
        tag = 'eval' if k in ('f3', 'st') else 'eval_2d'
        base = summ('baselines_seis' if k in ('f3', 'st') else 'baselines_seis_2d', '')
        o = {'idr_survey': {}, 'best_classical': {}, 'best_classical_name': {}, 'sup_synth': {}, 'idr_synth': {}, 'input': {}, 'sup_oracle': {}}
        for lv in (0.5, 1.0):
            r = summ(idr_run)[(k, 'sg', lv)]
            o['idr_survey'][lv] = r['snr']; o['input'][lv] = r['snr_in']
            o['sup_oracle'][lv] = summ(sup_run)[(k, 'sg', lv)]['snr'] if sup_run else None
            o['idr_synth'][lv] = summ('C1_idr_sg_s0', tag)[(k, 'sg', lv)]['snr']
            o['sup_synth'][lv] = summ('C2_sup_sg_s0', tag)[(k, 'sg', lv)]['snr']
            cand = {m: base[(m, k, 'sg', lv)]['snr'] for m in ('bandpass', 'fx', 'bm3d') if (m, k, 'sg', lv) in base}
            best = max(cand, key=cand.get); o['best_classical'][lv] = cand[best]; o['best_classical_name'][lv] = best
        out[k] = o
    return out


def summ_any(path):
    return json.loads(Path(path).read_text())


# patch summ for baseline folders (summary.json directly in the run folder)
_summ = summ
def summ(run, tag='eval'):  # noqa: F811
    if tag == '':
        rows = json.loads((RUNS / run / 'summary.json').read_text())['results']
        return {(r['method'], r['test'], r['noise'], round(r['level'], 3)): r for r in rows}
    return _summ(run, tag)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--skip-sweep', action='store_true'); a = p.parse_args()
    fig_refinement(); fig_convergence(not a.skip_sweep); fig_fidelity(); fig_scatter()
    fig_synth(); fig_fk(); fig_field(); fig_field_spectra(); fig_summary()
    (GEN / 'field_semisynth.json').write_text(json.dumps({k: {m: {str(l): v for l, v in d.items()} for m, d in o.items()}
                                                          for k, o in field_semisynth().items()}, indent=2))


if __name__ == '__main__':
    main()
