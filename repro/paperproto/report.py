"""Builds reports/paperproto/RESULTS.md (tables) + figs/*.png from whatever results exist.
Narrative findings are written separately (REPORT.md). Safe to re-run any time."""
import csv, json, sys, traceback
from pathlib import Path
R = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R / 'paperproto'), str(R)]
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seis_common as S

RUNS = R / 'runs/paperproto'
OUT = R / 'reports/paperproto'; FIG = OUT / 'figs'; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 130, 'font.size': 9, 'axes.titlesize': 9})
MD = []

# ------------------------------------------------------------------ paper tables (transcribed)
P2_COLS = ['BM3D', 'N2V', 'Nr2N', 'DBSN', 'N2N', 'Supervised', 'Ours (Fast IDR)', 'MPRNet']
P2 = {('kodak', 25): ['31.88/0.869', '31.63/0.869', '31.96/0.869', '32.07/0.875', '32.39/0.886', '32.39/0.885', '32.36/0.884', '32.85/0.893'],
      ('kodak', 50): ['28.64/0.772', '28.57/0.776', '28.73/0.770', '28.81/0.783', '29.23/0.803', '29.24/0.803', '29.27/0.803', '29.78/0.817'],
      ('BSDS300', 25): ['30.47/0.863', '30.72/0.874', '29.57/0.815', '31.12/0.881', '31.39/0.889', '31.40/0.889', '31.48/0.890', '31.87/0.897'],
      ('BSDS300', 50): ['27.14/0.745', '27.60/0.775', '26.18/0.684', '27.87/0.782', '28.17/0.799', '28.17/0.799', '28.25/0.802', '28.65/0.814'],
      ('BSD68', 25): ['28.55/0.782', '27.64/0.781', 'NA', '28.81/0.818', '29.15/0.831', '29.20/0.835', '29.20/0.835', '29.40/0.840'],
      ('BSD68', 50): ['25.59/0.670', '25.46/0.681', 'NA', '25.95/0.703', '26.23/0.725', '26.24/0.727', '26.25/0.726', '26.48/0.736']}
P5 = {('kodak', 'binomial'): ['31.48/0.939', '31.84/0.946', '31.63/0.944'], ('kodak', 'impulse'): ['36.04/0.975', '35.83/0.978', '34.92/0.977'],
      ('BSDS300', 'binomial'): ['31.50/0.930', '32.32/0.938', '32.18/0.936'], ('BSDS300', 'impulse'): ['36.55/0.976', '37.48/0.980', '37.40/0.979']}
P6 = {'kodak': {'BM3D': ['31.56/0.900', '29.20/0.798', '31.38/0.857', '27.23/0.742', '26.53/0.859'],
                'Ours': ['37.46/0.976', '33.10/0.929', '41.88/0.993', '28.60/0.803', '32.97/0.973']},
      'BSDS300': {'BM3D': ['26.77/0.841', '27.84/0.787', '29.59/0.838', '26.14/0.722', '26.31/0.870'],
                  'Ours': ['29.83/0.922', '31.28/0.910', '38.22/0.982', '27.39/0.793', '33.69/0.974']}}
pp = lambda s: float(s.split('/')[0]) if s != 'NA' else np.nan


def jl(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def img_res(run, tag='eval'):
    d = jl(RUNS / run / tag / 'summary.json')
    return {(r['dataset'], r['level']): r for r in d['results']} if d else None


def seis_res(run, tag='eval'):
    d = jl(RUNS / run / tag / 'summary.json')
    return {(r['test'], r['noise'], r['level']): r for r in d['results']} if d else None


def f2(x, y=None):
    return '—' if x is None or (isinstance(x, float) and np.isnan(x)) else (f'{x:.2f}' if y is None else f'{x:.2f}/{y:.3f}')


def table(head, rows):
    MD.append('| ' + ' | '.join(head) + ' |'); MD.append('|' + '---|' * len(head))
    for r in rows:
        MD.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    MD.append('')


def seed_stats(prefix, seeds=(0, 1, 2), tag='eval'):
    rs = [img_res(f'{prefix}_s{s}', tag) for s in seeds]
    rs = [r for r in rs if r]
    if not rs:
        return None, 0
    keys = rs[0].keys()
    return {k: (np.mean([r[k]['psnr'] for r in rs]), np.std([r[k]['psnr'] for r in rs]),
                np.mean([r[k]['ssim'] for r in rs])) for k in keys}, len(rs)


def safe(fn):
    def w(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:
            MD.append(f'> figure/table `{fn.__name__}` skipped: {type(e).__name__}: {e}\n')
            traceback.print_exc()
    return w


# ------------------------------------------------------------------ image tables
@safe
def image_tables():
    MD.append('## 1. Image reproduction (protocol P)\n')
    MD.append('PSNR (dB) / SSIM, final 50k-step checkpoints. Seeds are mean ± sd over training seeds 0–2. '
              'Paper columns from Table 2 are copied; "ours" columns are measured.\n')
    idr, ni = seed_stats('A1_idr_g'); sup, ns = seed_stats('A2_sup_g')
    bm = jl(RUNS / 'A10_bm3d_images/summary.json'); bm = {(r['dataset'], r['level']): r for r in bm['results']} if bm else {}
    nr = img_res('A3_nr2n_g_s0', 'eval_nr2ncorr_K1'); nr8 = img_res('A3_nr2n_g_s0', 'eval_nr2ncorr_K8'); nr0 = img_res('A3_nr2n_g_s0'); n2n = img_res('A4_n2n_g_s0'); fx = img_res('A5_idrfix_g_s0'); fp = img_res('A1_idr_g_s0_fp32')
    gi = img_res('A6_idr_gray_s0') or {}; gs = img_res('A6_sup_gray_s0') or {}
    rows = []
    for (ds, lv), pcols in P2.items():
        gray = ds == 'BSD68'
        def get(d, k=(ds, lv)):
            return f2(d[k]['psnr'], d[k]['ssim']) if d and k in d else '—'
        i_s = (f"{idr[(ds, lv)][0]:.2f}±{idr[(ds, lv)][1]:.2f}" if idr and (ds, lv) in idr else get(gi))
        s_s = (f"{sup[(ds, lv)][0]:.2f}±{sup[(ds, lv)][1]:.2f}" if sup and (ds, lv) in sup else get(gs))
        rows.append([f'{ds} σ={lv}', pcols[0], get(bm), pcols[2], '—' if gray else get(nr), '—' if gray else get(nr8), '—' if gray else get(nr0), pcols[4], '—' if gray else get(n2n),
                     pcols[5], s_s, pcols[6], i_s, '—' if gray else get(fx), '—' if gray else get(fp), pcols[7]])
    table(['set', 'BM3D paper', 'BM3D ours', 'Nr2N paper', 'Nr2N ours (2f(z)−z, K=1)', 'Nr2N ours (K=8 avg)', 'IDR stage-0 net applied to y', 'N2N paper', 'N2N ours', 'Sup paper', f'Sup ours (n={ns})',
           'IDR paper', f'IDR ours (n={ni})', 'IDR fixed-noise ours', 'IDR FP32 ours', 'MPRNet paper'], rows)
    # gap table
    if idr and sup:
        MD.append('**Gap to paper (ours − paper, PSNR dB) and IDR − Sup within each study:**\n')
        rows = []
        for (ds, lv) in [('kodak', 25), ('kodak', 50), ('BSDS300', 25), ('BSDS300', 50)]:
            pi, ps = pp(P2[(ds, lv)][6]), pp(P2[(ds, lv)][5])
            rows.append([f'{ds} σ={lv}', f'{idr[(ds, lv)][0] - pi:+.2f}', f'{sup[(ds, lv)][0] - ps:+.2f}',
                         f'{pi - ps:+.2f}', f'{idr[(ds, lv)][0] - sup[(ds, lv)][0]:+.2f}'])
        table(['set', 'IDR ours−paper', 'Sup ours−paper', 'paper IDR−Sup', 'ours IDR−Sup'], rows)
    # table 5
    rows = []
    for (ds, nz), pc in P5.items():
        i = img_res(f'A7_idr_binomial_s0' if nz == 'binomial' else 'A8_idr_impulse_s0')
        s = img_res(f'A7_sup_binomial_s0' if nz == 'binomial' else 'A8_sup_impulse_s0')
        g = lambda d: f2(d[(ds, .5)]['psnr'], d[(ds, .5)]['ssim']) if d else '—'
        rows.append([f'{ds} {nz} p=0.5', pc[0], pc[1], g(s), pc[2], g(i)])
    MD.append('**Table 5 analogue (binomial / impulse, p = 0.5).** Paper column order N2N/Sup/Ours as printed; '
              'the paper text\'s "N2N +0.2 dB on Kodak binomial" matches the printed *impulse* row, so the B/I labels '
              'in the paper may be swapped.\n')
    table(['set', 'N2N paper', 'Sup paper', 'Sup ours', 'IDR paper', 'IDR ours'], rows)
    rows = []
    for ds in ('kodak', 'BSDS300'):
        for k in range(1, 5):
            r = img_res(f'A9_idr_scn{k}_s0')
            rows.append([f'{ds} g{k}', P6[ds]['BM3D'][k - 1], P6[ds]['Ours'][k - 1], f2(r[(ds, 5)]['psnr'], r[(ds, 5)]['ssim']) if r else '—'])
    MD.append('**Table 6 analogue (spatially correlated noise, upstream kernels g1–g4, level 5).** '
              'Upstream "level 5" is a *variance* of 5/255 (std ≈ 0.14 ≈ 36/255). Paper g5 is not defined in the released code.\n')
    table(['set/kernel', 'BM3D paper', 'IDR paper', 'IDR ours'], rows)


# ------------------------------------------------------------------ drift
@safe
def drift():
    MD.append('## 2. Drift analysis\n')
    rows = []
    names = ['A1_idr_g_s0', 'A2_sup_g_s0', 'old_b4_256_idr', 'old_b4_256_sup', 'old_b4_128_idr', 'old_b4_128_sup', 'old_b192_idr', 'old_b192_sup']
    keys = ['kodak_25', 'kodak_50', 'BSDS300_25', 'BSDS300_50']
    D = {n: jl(RUNS / 'drift' / f'{n}.json') for n in names}
    for n, d in D.items():
        if not d:
            continue
        b = d['base']
        def delta(v):
            return np.mean([d[v][k]['psnr'] - b[k]['psnr'] for k in keys]) if v in d else np.nan
        seeds = d.get('D2_seeds', {})
        sd = np.mean([np.std([b[k]['psnr']] + [s[k]['psnr'] for s in seeds.values()]) for k in keys]) if seeds else np.nan
        rows.append([n, *[f"{b[k]['psnr']:.2f}" for k in keys], f'{delta("D1_bgr"):+.3f}', f'{delta("D3_float"):+.3f}',
                     f'{delta("D3_clip_input"):+.3f}', f"{np.mean([d['D4_ssim_gauss'][k]['ssim'] - b[k]['ssim'] for k in keys]):+.4f}",
                     f"{d['D5_bsds_test100']['BSDS300_25']['psnr'] - b['BSDS300_25']['psnr']:+.3f}", f'{sd:.3f}'])
    MD.append('Inference-only factors (mean Δ over the four RGB rows, PSNR dB; SSIM Δ for D4). '
              '"old_*" are the earlier runs: 49k/1k holdout (b4_256), 128-px downsampled crops (b4_128), batch 192 epochs (b192).\n')
    table(['checkpoint', 'Kodak25', 'Kodak50', 'BSDS25', 'BSDS50', 'D1 BGR in', 'D3 float PSNR', 'D3 clip noisy', 'D4 Gauss-SSIM ΔSSIM',
           'D5 BSDS test-100 (σ25)', 'D2 test-noise sd'], rows)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for j, (meth, col) in enumerate([('idr', 'C0'), ('sup', 'C1')]):
        labels, vals = [], []
        for n in ['old_b192', 'old_b4_128', 'old_b4_256', f"A{1 if meth == 'idr' else 2}"]:
            key = f'{n}_{meth}' if n.startswith('old') else (f'{n}_idr_g_s0' if meth == 'idr' else f'{n}_sup_g_s0')
            if D.get(key):
                labels.append({'old_b192': 'b192 epochs', 'old_b4_128': 'b4 50k\n128px+holdout', 'old_b4_256': 'b4 50k\n256px+holdout',
                               'A1': 'protocol P\n(full 50k)', 'A2': 'protocol P\n(full 50k)'}[n])
                vals.append(np.mean([D[key]['base'][k]['psnr'] for k in keys]))
        paper = np.mean([pp(P2[(k.split('_')[0], int(k.split('_')[1]))][6 if meth == 'idr' else 5]) for k in keys])
        ax[j].bar(labels, np.array(vals) - paper, color=col); ax[j].axhline(0, color='k', lw=.8)
        ax[j].set_title(f'{meth.upper()}: mean PSNR − paper (4 RGB rows)'); ax[j].set_ylabel('dB')
    fig.tight_layout(); fig.savefig(FIG / 'drift_protocol_steps.png'); plt.close(fig)
    MD.append('![](figs/drift_protocol_steps.png)\n')


@safe
def budget_curve():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for j, ds in enumerate([('kodak', 25), ('BSDS300', 25)]):
        for run, lab, col in [('A1_idr_g_s0', 'IDR', 'C0'), ('A2_sup_g_s0', 'Sup', 'C1')]:
            xs, ys = [], []
            for k in (10, 20, 30, 40):
                r = img_res(run, f'snap{k}k')
                if r: xs.append(k * 1000); ys.append(r[ds]['psnr'])
            r = img_res(run)
            if r: xs.append(50000); ys.append(r[ds]['psnr'])
            if xs: ax[j].plot(xs, ys, 'o-', color=col, label=f'{lab} 50k paper schedule')
            run2 = 'A11_idr_g_cos200k' if lab == 'IDR' else 'A11_sup_g_cos200k'
            xs, ys = [], []
            for k in (50, 100, 150):
                r = img_res(run2, f'snap{k}k')
                if r: xs.append(k * 1000); ys.append(r[ds]['psnr'])
            r = img_res(run2)
            if r: xs.append(200000); ys.append(r[ds]['psnr'])
            if xs: ax[j].plot(xs, ys, 's--', color=col, alpha=.6, label=f'{lab} 200k cosine')
        ax[j].axhline(pp(P2[ds][6]), color='C0', ls=':', label='paper IDR'); ax[j].axhline(pp(P2[ds][5]), color='C1', ls=':', label='paper Sup')
        ax[j].set_xscale('log'); ax[j].set_title(f'{ds[0]} σ={ds[1]}: PSNR vs training steps'); ax[j].set_xlabel('step'); ax[j].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / 'budget_curve.png'); plt.close(fig)
    MD.append('![](figs/budget_curve.png)\n')


@safe
def seed_plot():
    fig, ax = plt.subplots(figsize=(8, 3.2))
    keys = [('kodak', 25), ('kodak', 50), ('BSDS300', 25), ('BSDS300', 50)]
    for off, (pref, col, pi) in enumerate([('A1_idr_g', 'C0', 6), ('A2_sup_g', 'C1', 5)]):
        for s in (0, 1, 2):
            r = img_res(f'{pref}_s{s}')
            if r:
                ax.plot(np.arange(4) + off * .2 - .1, [r[k]['psnr'] - pp(P2[k][pi]) for k in keys], 'o', color=col, alpha=.7,
                        label=('IDR' if pi == 6 else 'Sup') + ' seeds' if s == 0 else None)
    ax.axhline(0, color='k', lw=.8); ax.set_xticks(range(4)); ax.set_xticklabels([f'{a} σ{b}' for a, b in keys])
    ax.set_ylabel('ours − paper (dB)'); ax.legend(); ax.set_title('Training-seed spread vs paper value')
    fig.tight_layout(); fig.savefig(FIG / 'seed_spread.png'); plt.close(fig)
    MD.append('![](figs/seed_spread.png)\n')


# ------------------------------------------------------------------ image visuals
def radial_psd(img):
    g = img.mean(-1) if img.ndim == 3 else img
    g = g - g.mean(); h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    P = np.abs(np.fft.fftshift(np.fft.fft2(g * win))) ** 2 / (win ** 2).sum()
    y, x = np.indices(P.shape); r = np.hypot(y - h / 2, x - w / 2) / max(h, w)
    bins = np.linspace(0, .5, 60); idx = np.digitize(r.ravel(), bins)
    return bins[1:], np.array([P.ravel()[idx == i].mean() if (idx == i).any() else np.nan for i in range(1, len(bins))])


@safe
def image_visuals():
    MD.append('## 3. Image visualisations\n')
    # target evolution
    for run, title in [('A1_idr_g_s0', 'Fast IDR (fresh noise)'), ('A5_idrfix_g_s0', 'Fast IDR (one fixed noisy realisation)')]:
        d = RUNS / run
        if not (d / 'vis_input.npz').exists():
            continue
        v = np.load(d / 'vis_input.npz'); tg = sorted(d.glob('vis_target_epoch*.npy'))
        fig, ax = plt.subplots(2, 2 + len(tg), figsize=(2.1 * (2 + len(tg)), 4.4))
        for row in range(2):
            ims = [v['clean'][row], v['noisy'][row]] + [np.load(t)[row] for t in tg]
            names = ['clean', 'noisy input x'] + [f'target, epoch {t.stem[-1]}' for t in tg]
            for a, im, nm in zip(ax[row], ims, names):
                im = np.clip(im.transpose(1, 2, 0), 0, 1)
                a.imshow(im[64:192, 64:192]); a.set_axis_off()
                cl = v['clean'][row].transpose(1, 2, 0)[64:192, 64:192]
                a.set_title(nm + ('' if nm == 'clean' else f'\nPSNR {10*np.log10(1/np.mean((im[64:192,64:192]-cl)**2)):.1f}'), fontsize=7)
        fig.suptitle(f'Training-target refinement: {title} (paper Fig. 1)'); fig.tight_layout()
        fig.savefig(FIG / f'targets_{run}.png'); plt.close(fig)
        MD.append(f'![](figs/targets_{run}.png)\n')
    # zoom + error + spectra
    runs = [('A2_sup_g_s0', 'Supervised'), ('A1_idr_g_s0', 'Fast IDR'), ('A3_nr2n_g_s0', 'IDR stage-0 net on y (no refinement)'), ('A5_idrfix_g_s0', 'IDR fixed-noise')]
    ex = {n: np.load(RUNS / r / 'eval/examples.npz') for r, n in runs if (RUNS / r / 'eval/examples.npz').exists()}
    if not ex:
        return
    for key in ['kodak_50_kodim23', 'BSDS300_25_86000']:
        first = next(iter(ex.values()))
        if f'{key}__gt' not in first:
            continue
        gt = first[f'{key}__gt'] / 255.; noisy = first[f'{key}__noisy']
        h, w = gt.shape[:2]; ys, xs = slice(h // 2 - 80, h // 2 + 80), slice(w // 2 - 80, w // 2 + 80)
        cols = [('clean', gt), ('noisy', noisy)] + [(n, e[f'{key}__out']) for n, e in ex.items()]
        fig, ax = plt.subplots(3, len(cols), figsize=(2.2 * len(cols), 6.8))
        for j, (n, im) in enumerate(cols):
            im = np.clip(im, 0, 1)
            ax[0, j].imshow(im[ys, xs]); ax[0, j].set_title(n if j < 2 else f'{n}\n{10*np.log10(1/np.mean((im-gt)**2)):.2f} dB', fontsize=7)
            if j >= 2:
                ax[1, j].imshow(np.abs(im - gt).mean(-1)[ys, xs], cmap='magma', vmin=0, vmax=.15); ax[1, j].set_title('|error|', fontsize=7)
                ax[2, j].imshow((noisy - im).mean(-1)[ys, xs], cmap='gray', vmin=-.3, vmax=.3); ax[2, j].set_title('removed = noisy − out', fontsize=7)
            elif j == 1:
                ax[2, j].imshow((noisy - gt).mean(-1)[ys, xs], cmap='gray', vmin=-.3, vmax=.3); ax[2, j].set_title('true noise', fontsize=7)
        for a in ax.ravel(): a.set_axis_off()
        fig.tight_layout(); fig.savefig(FIG / f'zoom_{key}.png'); plt.close(fig)
        MD.append(f'![](figs/zoom_{key}.png)\n')
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        f, p = radial_psd(gt); ax[0].semilogy(f, p, 'k', label='clean')
        f, p = radial_psd(noisy); ax[0].semilogy(f, p, 'k:', label='noisy')
        for n, e in ex.items():
            o = e[f'{key}__out']; f, p = radial_psd(o); ax[0].semilogy(f, p, label=n)
            f, p = radial_psd(o - gt); ax[1].semilogy(f, p, label=n)
        ax[0].set_title('Radially averaged power spectrum of output'); ax[1].set_title('Power spectrum of error (output − clean)')
        for a in ax: a.set_xlabel('cycles / pixel'); a.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIG / f'psd_{key}.png'); plt.close(fig)
        MD.append(f'![](figs/psd_{key}.png)\n')


@safe
def per_image_scatter():
    def load(run):
        p = RUNS / run / 'eval/per_image.csv'
        return {(r['dataset'], r['level'], r['image']): float(r['psnr']) for r in csv.DictReader(p.open())} if p.exists() else None
    a, b = load('A1_idr_g_s0'), load('A2_sup_g_s0')
    if not (a and b):
        return
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    MD.append('Per-image paired difference IDR − Sup (seed 0), with 95% bootstrap CI of the mean:\n')
    rows = []
    for j, lv in enumerate(('25', '50')):
        ks = [k for k in a if k[1] == lv and k in b]
        x = np.array([b[k] for k in ks]); y = np.array([a[k] for k in ks])
        ax[j].scatter(x, y - x, s=5, alpha=.5); ax[j].axhline(0, color='k', lw=.8)
        ax[j].set_xlabel('Supervised PSNR'); ax[j].set_ylabel('IDR − Sup'); ax[j].set_title(f'σ={lv}: per-image (Kodak+BSDS300)')
        d = y - x; bs = np.random.default_rng(0).choice(d, (10000, len(d))).mean(1)
        rows.append([f'σ={lv}', f'{d.mean():+.3f}', f'[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]',
                     f'{(d > 0).sum()}/{len(d)}'])
    table(['σ', 'mean IDR−Sup', '95% CI', 'IDR wins'], rows)
    fig.tight_layout(); fig.savefig(FIG / 'per_image_scatter.png'); plt.close(fig)
    MD.append('![](figs/per_image_scatter.png)\n')


# ------------------------------------------------------------------ seismic tables
SEIS_RUNS = {'sg': [('IDR', 'C1_idr_sg', True), ('Sup', 'C2_sup_sg', True), ('Nr2N (2f(z)−z)', 'C3_nr2n_sg_s0:eval_nr2ncorr_K1', False), ('Nr2N K=8', 'C3_nr2n_sg_s0:eval_nr2ncorr_K8', False),
                    ('IDR stage-0 on y', 'C3_nr2n_sg_s0', False),
                    ('N2N', 'C4_n2n_sg_s0', False), ('IDR fixed', 'C5_idrfix_sg_s0', False)],
             'sb': [('IDR', 'C7_idr_sb_s0', False), ('Sup', 'C7_sup_sb_s0', False), ('IDR fixed', 'C6_idrfix_sb_s0', False)],
             'ss': [('IDR', 'C8_idr_ss_s0', False), ('Sup', 'C8_sup_ss_s0', False)],
             'sd': [('IDR', 'C9_idr_sd_s0', False), ('Sup', 'C9_sup_sd_s0', False)]}
NAMES = {'sg': 'white Gaussian (paper: Gaussian)', 'sb': 'band-limited 5–60 Hz (paper: correlated)',
         'ss': 'erratic spikes (paper: impulse)', 'sd': 'dead traces (paper: binomial)'}


def seis_get(run, seeded):
    if not seeded:
        run, _, tag = run.partition(':'); r = seis_res(run, tag or 'eval'); return ({k: (v['snr'], np.nan, v['ssim']) for k, v in r.items()} if r else None), 1
    rs = [seis_res(f'{run}_s{s}') for s in (0, 1, 2)]; rs = [r for r in rs if r]
    if not rs:
        return None, 0
    return {k: (np.mean([r[k]['snr'] for r in rs]), np.std([r[k]['snr'] for r in rs]), np.mean([r[k]['ssim'] for r in rs])) for k in rs[0]}, len(rs)


@safe
def seismic_tables():
    MD.append('## 4. Seismic adaptation: synthetic surveys with clean reference\n')
    MD.append('Output SNR (dB) over whole held-out sections, mean over 3 test-noise seeds (and training seeds 0–2 where shown ±sd). '
              'Input SNR shown for reference. Baselines on the identical noisy sections.\n')
    bl = jl(RUNS / 'baselines_seis/summary.json')
    blr = {(r['method'], r['test'], r['noise'], r['level']): r for r in bl['results']} if bl else {}
    for nz, runs in SEIS_RUNS.items():
        MD.append(f'### {nz}: {NAMES[nz]}\n')
        got = [(n, *seis_get(r, sd)) for n, r, sd in runs]
        bm = [m for m in ['bandpass', 'fx', 'bm3d', 'median', 'interp'] if any(k[0] == m and k[2] == nz for k in blr)]
        rows = []
        for t in S.CORPUS_TESTS:
            for lv in S.LEVELS[nz]:
                row = [t, lv]
                sin = next((d[(t, nz, lv)] for n, d, c in got if d and (t, nz, lv) in d), None)
                ref = seis_res(runs[0][1] + ('_s0' if runs[0][2] else ''))
                row.append(f"{ref[(t, nz, lv)]['snr_in']:.2f}" if ref and (t, nz, lv) in ref else '—')
                for n, d, c in got:
                    v = d.get((t, nz, lv)) if d else None
                    row.append('—' if v is None else (f'{v[0]:.2f}±{v[1]:.2f}' if c > 1 else f'{v[0]:.2f}'))
                for m in bm:
                    v = blr.get((m, t, nz, lv)); row.append('—' if v is None else f"{v['snr']:.2f}")
                rows.append(row)
        table(['test', 'level', 'SNR in'] + [f'{n} (n={c})' if c > 1 else n for n, d, c in got] + bm, rows)


@safe
def mismatch():
    rows = []
    for run in ('C1_idr_sg_s0', 'C2_sup_sg_s0', 'C7_idr_sb_s0', 'C7_sup_sb_s0'):
        r = seis_res(run, 'mismatch')
        if not r:
            continue
        for nz in ('sg', 'sb'):
            for lv in (0.5, 1.0, 1.5):
                v = [r[(t, nz, lv)]['snr'] for t in ('m2', 'ot', 'sa') if (t, nz, lv) in r]
                vin = [r[(t, nz, lv)]['snr_in'] for t in ('m2', 'ot', 'sa') if (t, nz, lv) in r]
                if v: rows.append([run, nz, lv, f'{np.mean(vin):.2f}', f'{np.mean(v):.2f}'])
    if rows:
        MD.append('### Noise-model mismatch (mean over M2/OT/SA; σ=1.5 is outside the training range)\n')
        table(['model (trained on)', 'test noise', 'σ', 'SNR in', 'SNR out'], rows)


# ------------------------------------------------------------------ seismic visuals
def fk(sec, dt, dx=1.0):
    S_ = np.fft.fftshift(np.abs(np.fft.fft2(sec * np.outer(np.hanning(sec.shape[0]), np.hanning(sec.shape[1])))), axes=1)
    S_ = S_[: sec.shape[0] // 2]
    f = np.fft.fftfreq(sec.shape[0], dt)[: sec.shape[0] // 2]
    return 20 * np.log10(S_ / S_.max() + 1e-6), f


@safe
def seismic_visuals():
    MD.append('## 5. Seismic visualisations\n')
    bl = RUNS / 'baselines_seis/examples.npz'
    blx = np.load(bl) if bl.exists() else {}
    for t, nz, lv in [('ot', 'sg', 1.0), ('m2', 'sg', 1.0), ('sa', 'sg', 1.0), ('ot', 'sb', 1.0), ('m2_prestm', 'sg', 1.0),
                      ('ot', 'ss', 0.1), ('ot', 'sd', 0.5)]:
        runs = {'sg': [('IDR', 'C1_idr_sg_s0'), ('Sup', 'C2_sup_sg_s0'), ('IDR fixed', 'C5_idrfix_sg_s0')],
                'sb': [('IDR', 'C7_idr_sb_s0'), ('Sup', 'C7_sup_sb_s0'), ('IDR fixed', 'C6_idrfix_sb_s0')],
                'ss': [('IDR', 'C8_idr_ss_s0'), ('Sup', 'C8_sup_ss_s0')], 'sd': [('IDR', 'C9_idr_sd_s0'), ('Sup', 'C9_sup_sd_s0')]}[nz]
        k = f'{t}__{nz}__{lv}'
        ex = {}
        for n, r in runs:
            p = RUNS / r / 'eval/examples.npz'
            if p.exists():
                z = np.load(p)
                if k in z: ex[n] = z[k]
        if not ex:
            continue
        clean, noisy = next(iter(ex.values()))[:2]
        outs = [(n, e[2]) for n, e in ex.items()]
        for m in ('bm3d', 'fx', 'bandpass', 'median', 'interp'):
            if f'{k}__{m}' in blx: outs.append((m, blx[f'{k}__{m}']))
        dt = S.load_test(t)[1]['dt']
        cols = [('clean', clean), ('noisy', noisy)] + outs
        snr = lambda o: 10 * np.log10((clean ** 2).sum() / ((o - clean) ** 2).sum())
        fig, ax = plt.subplots(3, len(cols), figsize=(2.3 * len(cols), 8.2))
        for j, (n, o) in enumerate(cols):
            ax[0, j].imshow(o, cmap='gray_r', vmin=-2.5, vmax=2.5, aspect='auto')
            ax[0, j].set_title(n if j < 1 else f'{n}\nSNR {snr(o):.2f} dB', fontsize=7)
            if j >= 1:
                ax[1, j].imshow(noisy - o if j > 1 else noisy - clean, cmap='gray_r', vmin=-2.5, vmax=2.5, aspect='auto')
                ax[1, j].set_title('removed' if j > 1 else 'true noise', fontsize=7)
            F, f = fk(o, dt)
            ax[2, j].imshow(F, cmap='viridis', vmin=-60, vmax=0, aspect='auto', extent=[-.5, .5, f[-1], 0])
            ax[2, j].set_title('f-k (dB)', fontsize=7); ax[2, j].set_ylabel('Hz' if j == 0 else '')
        for a in ax[:2].ravel(): a.set_xticks([]); a.set_yticks([])
        ax[1, 0].set_axis_off()
        fig.suptitle(f'{t}: {NAMES[nz]}, level {lv} (section, removed noise, f-k spectrum)'); fig.tight_layout()
        fig.savefig(FIG / f'seis_{k}.png'); plt.close(fig)
        MD.append(f'![](figs/seis_{k}.png)\n')
        # spectra + traces + local similarity
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
        f = np.fft.rfftfreq(clean.shape[0], dt)
        for n, o in cols:
            ax[0].semilogy(f, np.abs(np.fft.rfft(o, axis=0)).mean(1), label=n, lw=1.4 if n in ('clean', 'IDR', 'Sup') else .8)
        ax[0].set_xlabel('Hz'); ax[0].set_title('mean amplitude spectrum'); ax[0].legend(fontsize=6)
        tr = clean.shape[1] // 2; tt = np.arange(clean.shape[0]) * dt
        for i, (n, o) in enumerate(cols[:5]):
            ax[1].plot(o[:, tr] + 6 * i, tt, lw=.7, label=n)
            if n != 'clean': ax[1].plot(clean[:, tr] + 6 * i, tt, 'k', lw=.4, alpha=.5)
        ax[1].invert_yaxis(); ax[1].set_title(f'trace {tr} (offset per method; black = clean)'); ax[1].set_ylabel('s')
        if 'IDR' in ex:
            o = ex['IDR'][2]; ls = S.local_similarity(o.astype(float), (noisy - o).astype(float))
            im = ax[2].imshow(ls, cmap='RdBu_r', vmin=-.6, vmax=.6, aspect='auto'); plt.colorbar(im, ax=ax[2])
            ax[2].set_title('IDR local similarity(output, removed)\n(≈0 = no signal leakage)')
        fig.tight_layout(); fig.savefig(FIG / f'seis_spec_{k}.png'); plt.close(fig)
        MD.append(f'![](figs/seis_spec_{k}.png)\n')
    # seismic target evolution
    d = RUNS / 'C1_idr_sg_s0'
    if (d / 'vis_input.npz').exists():
        v = np.load(d / 'vis_input.npz'); tg = sorted(d.glob('vis_target_epoch*.npy'))
        fig, ax = plt.subplots(1, 2 + len(tg), figsize=(2.3 * (2 + len(tg)), 2.8))
        ims = [v['clean'][1, 0], v['noisy'][1, 0]] + [np.load(t)[1, 0] for t in tg]
        for a, im, nm in zip(ax, ims, ['clean', 'noisy'] + [f'target ep{t.stem[-1]}' for t in tg]):
            a.imshow(im, cmap='gray_r', vmin=-2.5, vmax=2.5, aspect='auto'); a.set_axis_off()
            a.set_title(nm if nm == 'clean' else f'{nm}\nSNR {10*np.log10((ims[0]**2).sum()/((im-ims[0])**2).sum()):.1f} dB', fontsize=7)
        fig.tight_layout(); fig.savefig(FIG / 'seis_targets.png'); plt.close(fig)
        MD.append('![](figs/seis_targets.png)\n')


@safe
def seismic_bars():
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    bl = jl(RUNS / 'baselines_seis/summary.json'); blr = {(r['method'], r['test'], r['noise'], r['level']): r for r in bl['results']} if bl else {}
    for j, nz in enumerate(('sg', 'sb', 'ss', 'sd')):
        lv = S.LEVELS[nz][-1]
        series = [(n, *seis_get(r, sd)) for n, r, sd in SEIS_RUNS[nz]]
        meths = [n for n, d, c in series if d] + [m for m in ('bm3d', 'fx', 'bandpass', 'median', 'interp') if any(k[0] == m and k[2] == nz for k in blr)]
        tests = S.CORPUS_TESTS; w = .8 / max(len(meths), 1)
        for i, m in enumerate(meths):
            d = next((d for n, d, c in series if n == m), None)
            vals = [(d.get((t, nz, lv), (np.nan,))[0] if d else blr.get((m, t, nz, lv), {}).get('snr', np.nan)) for t in tests]
            ax[j].bar(np.arange(len(tests)) + i * w, vals, w, label=m)
        ref = seis_res(SEIS_RUNS[nz][0][1] + ('_s0' if SEIS_RUNS[nz][0][2] else ''))
        if ref: ax[j].plot(np.arange(len(tests)) + .4, [ref.get((t, nz, lv), {}).get('snr_in', np.nan) for t in tests], 'k_', ms=14, label='input')
        ax[j].set_xticks(np.arange(len(tests)) + .4); ax[j].set_xticklabels(tests, rotation=45, fontsize=7)
        ax[j].set_title(f'{nz} level {lv}: output SNR (dB)'); ax[j].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / 'seis_bars.png'); plt.close(fig)
    MD.append('![](figs/seis_bars.png)\n')


@safe
def field():
    MD.append('## 6. Field data (no clean reference)\n')
    for key, title in [('f3', 'F3 Netherlands (marine 3-D field)'), ('st', 'Stratton 3-D (land field)')]:
        semi = [('IDR trained on noisy field data (fixed realisation)', 'F1_idrfix_f3_sg' if key == 'f3' else 'F3_idrfix_st_sg'),
                ('Supervised on field-as-clean (oracle, domain-matched)', 'F1_sup_f3_sg_oracle' if key == 'f3' else 'F3_sup_st_sg_oracle'),
                ('IDR on synthetic corpus', 'C1_idr_sg_s0'), ('Supervised on synthetic corpus', 'C2_sup_sg_s0')]
        rows = []
        for n, r in semi:
            d = seis_res(r)
            if d:
                rows.append([n] + [f"{d[(key, 'sg', lv)]['snr']:.2f}" if (key, 'sg', lv) in d else '—' for lv in (0.5, 1.0)])
        if rows:
            MD.append(f'**{title}, semi-synthetic** (white noise added to field sections; reference = original field section):\n')
            table(['model', 'SNR out σ=0.5', 'SNR out σ=1.0'], rows)
        s = jl(RUNS / f'field_{key}/summary.json')
        if not s:
            continue
        MD.append(f'**{title}, real noise** (models applied to raw field sections):\n')
        table(['model', 'removed energy %', 'removed % in 5–60 Hz', 'removed % outside band', 'mean |local sim| (leakage)'],
              [[n, f"{100*v['removed_energy_frac']:.2f}", f"{100*v['removed_frac_in_band']:.2f}", f"{100*v['removed_frac_out_band']:.2f}",
                f"{v['leak_ls']:.3f}"] for n, v in s['results'].items()])
        z = np.load(RUNS / f'field_{key}/outputs.npz'); raw = z['raw']; i = len(raw) // 2
        names = [n for n in z.files if n != 'raw']
        fig, ax = plt.subplots(3, 1 + len(names), figsize=(2.6 * (1 + len(names)), 8.5))
        vm = np.percentile(np.abs(raw[i]), 99)
        ax[0, 0].imshow(raw[i], cmap='gray_r', vmin=-vm, vmax=vm, aspect='auto'); ax[0, 0].set_title('raw field section', fontsize=7)
        F, f = fk(raw[i], s['dt']); ax[2, 0].imshow(F, vmin=-60, vmax=0, aspect='auto', extent=[-.5, .5, f[-1], 0]); ax[2, 0].set_title('f-k raw', fontsize=7)
        ax[1, 0].set_axis_off()
        for j, n in enumerate(names, 1):
            o = z[n][i]
            ax[0, j].imshow(o, cmap='gray_r', vmin=-vm, vmax=vm, aspect='auto'); ax[0, j].set_title(n, fontsize=7)
            ax[1, j].imshow(raw[i] - o, cmap='gray_r', vmin=-vm / 3, vmax=vm / 3, aspect='auto'); ax[1, j].set_title('removed (×3 gain)', fontsize=7)
            F, f = fk(raw[i] - o, s['dt']); ax[2, j].imshow(F, vmin=-60, vmax=0, aspect='auto', extent=[-.5, .5, f[-1], 0]); ax[2, j].set_title('f-k of removed', fontsize=7)
        for a in ax[:2].ravel(): a.set_xticks([]); a.set_yticks([])
        fig.suptitle(title); fig.tight_layout(); fig.savefig(FIG / f'field_{key}.png'); plt.close(fig)
        MD.append(f'![](figs/field_{key}.png)\n')


@safe
def training_curves():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
    for j, runs in enumerate([['A1_idr_g_s0', 'A2_sup_g_s0', 'A3_nr2n_g_s0', 'A5_idrfix_g_s0'],
                              ['C1_idr_sg_s0', 'C2_sup_sg_s0', 'C3_nr2n_sg_s0', 'C5_idrfix_sg_s0']]):
        for r in runs:
            p = RUNS / r / 'metrics.jsonl'
            if not p.exists(): continue
            L = [json.loads(l) for l in p.open()]
            st = np.array([l['step'] for l in L]); lo = np.array([l['loss'] for l in L])
            k = 20; sm = np.convolve(lo, np.ones(k) / k, 'valid')
            ax[j].semilogy(st[k - 1:], sm, label=r, lw=.9)
        for e in (12500, 25000, 37500): ax[j].axvline(e, color='gray', ls=':', lw=.7)
        ax[j].set_title(['ImageNet', 'Seismic corpus'][j] + ' training loss (dotted = teacher refresh)'); ax[j].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / 'training_loss.png'); plt.close(fig)
    MD.append('![](figs/training_loss.png)\n')


def status():
    st = jl(RUNS / 'queue_state.json')
    if st:
        MD.append(f"_Queue status at build time: {len(st['done'])}/{st['total']} jobs done; running: {', '.join(st['running']) or 'none'}; "
                  f"failed: {', '.join(st['failed']) or 'none'}._\n")


if __name__ == '__main__':
    MD.append('# Results dump: paper-protocol Fast IDR reproduction + seismic adaptation\n')
    status()
    image_tables(); drift(); budget_curve(); seed_plot(); image_visuals(); per_image_scatter()
    seismic_tables(); mismatch(); seismic_bars(); seismic_visuals(); field(); training_curves()
    (OUT / 'RESULTS.md').write_text('\n'.join(MD) + '\n')
    print('wrote', OUT / 'RESULTS.md', len(list(FIG.glob('*.png'))), 'figures')
