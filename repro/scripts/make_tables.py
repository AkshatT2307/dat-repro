"""LaTeX tables for report/main.tex, generated directly from run outputs (no hand-typed results).

Usage (from repro/):  python scripts/make_tables.py
Writes report/tables/*.tex. Paper (Zhang et al. 2022) values are transcribed constants, marked PAPER_*.
Bold = best, underline = second best among methods that use neither clean data nor a second noisy
realisation (i.e. supervised and N2N are excluded from the ranking; they are reference upper bounds).
"""
import csv, json, datetime as dt
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1]
RUNS = R / 'runs' / 'paperproto'
OUT = R / 'report' / 'tables'; OUT.mkdir(parents=True, exist_ok=True)


def J(p):
    return json.loads(Path(p).read_text())


def img(run, tag='eval'):
    return {(r['dataset'], r['level']): r for r in J(RUNS / run / tag / 'summary.json')['results']}


def seis(run, tag='eval'):
    return {(r['test'], r['noise'], round(r['level'], 3)): r for r in J(RUNS / run / tag / 'summary.json')['results']}


def base(folder):
    return {(r['method'], r['test'], r['noise'], round(r['level'], 3)): r for r in J(RUNS / folder / 'summary.json')['results']}


def f2(x):
    return '--' if x is None else f'{x:.2f}'


def rank_fmt(vals, rankable):
    """vals: list of floats/None; rankable: list of bool -> list of strings with bold/underline."""
    cand = sorted({v for v, r in zip(vals, rankable) if r and v is not None}, reverse=True)
    out = []
    for v, r in zip(vals, rankable):
        s = f2(v)
        if r and v is not None and cand:
            if v == cand[0]:
                s = r'\textbf{' + s + '}'
            elif len(cand) > 1 and v == cand[1]:
                s = r'\underline{' + s + '}'
        out.append(s)
    return out


def write(name, body):
    import re  # typeset signed numbers in text mode as math minus/plus
    body = re.sub(r'(?<![-\w$])([-+])(\d)', lambda m: ('$-$' if m.group(1) == '-' else '$+$') + m.group(2), body)
    (OUT / f'{name}.tex').write_text(body)
    print('wrote', OUT / f'{name}.tex')


# ------------------------------------------------------------------ network table
def t_unet():
    import sys, types, importlib.util, torch
    reg = types.ModuleType('utils.registry'); reg.ARCH_REGISTRY = types.SimpleNamespace(register=lambda: (lambda c: c))
    sys.modules.setdefault('utils', types.ModuleType('utils')); sys.modules['utils.registry'] = reg
    spec = importlib.util.spec_from_file_location('unet_src', R / 'models' / 'unet.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    rows = []
    tot = {}
    for C in (1, 3):
        net = m.UNet_n2n_un(C, C)
        tot[C] = sum(p.numel() for p in net.parameters())
        if C == 1:
            H = 256
            spec_rows = [
                ('en\\_block1', r'2$\times$[conv3$\times$3, LReLU], maxpool', f'{C}$\\to$48', f'{H//2}', net.en_block1),
                ('en\\_block2', r'conv3$\times$3, LReLU, maxpool', '48$\\to$48', f'{H//4}', net.en_block2),
                ('en\\_block3', r'conv3$\times$3, LReLU, maxpool', '48$\\to$48', f'{H//8}', net.en_block3),
                ('en\\_block4', r'conv3$\times$3, LReLU, maxpool', '48$\\to$48', f'{H//16}', net.en_block4),
                ('en\\_block5', r'conv, LReLU, maxpool, conv, LReLU, up$\times$2', '48$\\to$48', f'{H//16}', net.en_block5),
                ('de\\_block1', r'[cat e4] 2$\times$[conv, LReLU], up$\times$2', '96$\\to$96', f'{H//8}', net.de_block1),
                ('de\\_block2', r'[cat e3] 2$\times$[conv, LReLU], up$\times$2', '144$\\to$96', f'{H//4}', net.de_block2),
                ('de\\_block3', r'[cat e2] 2$\times$[conv, LReLU], up$\times$2', '144$\\to$96', f'{H//2}', net.de_block3),
                ('de\\_block4', r'[cat e1] 2$\times$[conv, LReLU], up$\times$2', '144$\\to$96', f'{H}', net.de_block4),
                ('de\\_block5', r'[cat input] conv, LReLU, conv, LReLU, conv', f'{96+C}$\\to$64$\\to$32$\\to${C}', f'{H}', net.de_block5),
            ]
            for name, ops, ch, sz, mod in spec_rows:
                n = sum(p.numel() for p in mod.parameters())
                rows.append(f'{name} & {ops} & {ch} & ${sz}^2$ & {n:,} \\\\')
    body = r"""\begin{table}[!t]
\centering
\caption{Shallow U-Net \texttt{UNet\_n2n\_un} (no normalization layers), parsed from \texttt{models/unet.py}. Channels and parameters for one input/output channel (seismic, $C=1$); output size for a $256\times256$ crop. All convolutions are $3\times3$, stride 1, padding 1, with bias; LReLU slope 0.1; up$\times$2 is nearest-neighbour; [cat $\cdot$] is channel concatenation with the named skip tensor.}
\label{tab:unet}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llccr}
\toprule
Stage & Operations & Channels & Output & Params \\
\midrule
""" + '\n'.join(rows) + f"""
\\midrule
\\multicolumn{{4}}{{l}}{{Total, $C=1$ (seismic, gray) / $C=3$ (RGB)}} & {tot[1]:,} / {tot[3]:,} \\\\
\\bottomrule
\\end{{tabular}}}}
\\end{{table}}
"""
    write('tab_unet', body)
    return tot


# ------------------------------------------------------------------ image tables
PAPER_T2 = {  # (BM3D, N2N, Nr2N, Sup, IDR) PSNR from Zhang et al. 2022 Table 2 / Table 3 (BSD68 gray)
    ('kodak', 25): (31.88, 32.39, 31.96, 32.39, 32.36), ('kodak', 50): (28.64, 29.23, 28.73, 29.24, 29.27),
    ('BSDS300', 25): (30.47, 31.39, 29.57, 31.40, 31.48), ('BSDS300', 50): (27.14, 28.17, 26.18, 28.17, 28.25),
    ('BSD68', 25): (28.55, 29.15, None, 29.20, 29.20), ('BSD68', 50): (25.59, 26.23, None, 26.24, 26.25)}
NM = {'kodak': 'Kodak', 'BSDS300': 'BSDS300', 'BSD68': 'BSD68'}


def t_image_gauss():
    bm = img('A10_bm3d_images', '.'); n2n = img('A4_n2n_g_s0'); nr = img('A3_nr2n_g_s0', 'eval_nr2ncorr_K8')
    lines = []
    for (ds, lv), pp in PAPER_T2.items():
        if ds == 'BSD68':
            idr = [img('A6_idr_gray_s0')[(ds, lv)]['psnr']]; sup = [img('A6_sup_gray_s0')[(ds, lv)]['psnr']]
        else:
            idr = [img(f'A1_idr_g_s{s}')[(ds, lv)]['psnr'] for s in range(3)]; sup = [img(f'A2_sup_g_s{s}')[(ds, lv)]['psnr'] for s in range(3)]
        ours = [bm[(ds, lv)]['psnr'], nr.get((ds, lv), {}).get('psnr'), np.mean(idr), n2n.get((ds, lv), {}).get('psnr'), np.mean(sup)]
        fm = rank_fmt(ours, [True, True, True, False, False])
        sd = lambda v: f'$\\pm${np.std(v):.2f}' if len(v) > 1 else ''
        lines.append(f'{NM[ds]} & {lv} & {f2(pp[0])} & {fm[0]} & {f2(pp[2])} & {fm[1]} & {f2(pp[4])} & {fm[2]}{sd(idr)} & {f2(pp[1])} & {fm[3]} & {f2(pp[3])} & {fm[4]}{sd(sup)} \\\\')
    body = r"""\begin{table*}[!t]
\centering
\caption{Additive Gaussian noise, PSNR (dB). ``Paper'' columns are copied from Zhang et al.~\cite{zhang2022idr}; ``ours'' are measured with protocol P (50k steps, batch 4, $256^2$ crops). Fast IDR and supervised RGB rows: mean$\pm$sd over training seeds 0--2; BSD68 (gray) and other methods: seed 0. Our Nr2N is the stage-0 IDR network used as a Noisier2Noise estimator ($2f(z)-z$, averaged over $K=8$ draws), an ablation rather than a reproduction of the paper's Nr2N row. Bold/underline: best/second among methods needing neither clean data nor a second noisy copy (BM3D, Nr2N, Fast IDR); N2N and supervised are reference bounds.}
\label{tab:img_gauss}
\begin{tabular}{llcccccccccc}
\toprule
& & \multicolumn{2}{c}{BM3D} & \multicolumn{2}{c}{Nr2N} & \multicolumn{2}{c}{Fast IDR} & \multicolumn{2}{c}{N2N} & \multicolumn{2}{c}{Supervised} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}
Set & $\sigma$ & paper & ours & paper & ours & paper & ours & paper & ours & paper & ours \\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write('tab_image_gauss', body)


PAPER_T5 = {('kodak', 'binomial'): (31.84, 31.63), ('kodak', 'impulse'): (35.83, 34.92),
            ('BSDS300', 'binomial'): (32.32, 32.18), ('BSDS300', 'impulse'): (37.48, 37.40)}  # (Sup, IDR)
PAPER_T6 = {('kodak', 1): (31.56, 37.46), ('kodak', 2): (29.20, 33.10), ('kodak', 3): (31.38, 41.88), ('kodak', 4): (27.23, 28.60),
            ('BSDS300', 1): (26.77, 29.83), ('BSDS300', 2): (27.84, 31.28), ('BSDS300', 3): (29.59, 38.22), ('BSDS300', 4): (26.14, 27.39)}  # (BM3D, IDR)


def t_image_other():
    l1 = []
    for (ds, nz), (ps, pi) in PAPER_T5.items():
        run = 'A7' if nz == 'binomial' else 'A8'
        oi = img(f'{run}_idr_{nz}_s0')[(ds, 0.5)]['psnr']; os_ = img(f'{run}_sup_{nz}_s0')[(ds, 0.5)]['psnr']
        l1.append(f'{NM[ds]} & {nz} & {pi:.2f} & {oi:.2f} & {oi - pi:+.2f} & {ps:.2f} & {os_:.2f} & {os_ - ps:+.2f} \\\\')
    l2 = []
    for (ds, k), (pb, pi) in PAPER_T6.items():
        oi = img(f'A9_idr_scn{k}_s0')[(ds, 5)]['psnr']
        l2.append(f'{NM[ds]} & g{k} & {pb:.2f} & {pi:.2f} & {oi:.2f} & {oi - pi:+.2f} \\\\')
    body = r"""\begin{table}[!t]
\centering
\caption{Non-Gaussian image noise, PSNR (dB), seed 0. (a) Binomial and impulse noise, $p=0.5$ (paper Table~5; the paper's text and table may swap the two row labels). (b) Spatially correlated noise, upstream kernels g1--g4 at upstream ``level 5'' (a variance of $5/255$); g5 is undefined in the released code. $\Delta$ = ours $-$ paper.}
\label{tab:img_other}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcccccc}
\toprule
\multicolumn{8}{l}{(a) Binomial / impulse} \\
& & \multicolumn{3}{c}{Fast IDR} & \multicolumn{3}{c}{Supervised} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Set & Noise & paper & ours & $\Delta$ & paper & ours & $\Delta$ \\
\midrule
""" + '\n'.join(l1) + r"""
\midrule
\multicolumn{8}{l}{(b) Spatially correlated} \\
Set & Kernel & BM3D (paper) & IDR (paper) & IDR (ours) & $\Delta$ & & \\
\midrule
""" + '\n'.join(l2) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    write('tab_image_other', body)


def t_drift():
    a1 = J(RUNS / 'drift' / 'A1_idr_g_s0.json'); a2 = J(RUNS / 'drift' / 'A2_sup_g_s0.json')
    keys = ['kodak_25', 'kodak_50', 'BSDS300_25', 'BSDS300_50']
    k25 = lambda d: d['base']['kodak_25']['psnr']
    tr = []
    D = lambda n: J(RUNS / 'drift' / (n + '.json'))
    for lab, i, s, ri, rs in (('Batch 192, epoch schedule (earlier repro)', 'old_b192_idr', 'old_b192_sup', a1, a2),
                              (r'128-px crops (vs.\ 256-px, same split)', 'old_b4_128_idr', 'old_b4_128_sup', D('old_b4_256_idr'), D('old_b4_256_sup')),
                              ('49k/1k train/holdout split', 'old_b4_256_idr', 'old_b4_256_sup', a1, a2)):
        tr.append(f'{lab} & {k25(D(i)) - k25(ri):+.2f} & {k25(D(s)) - k25(rs):+.2f} \\\\')
    fp = img('A1_idr_g_s0_fp32')[('kodak', 25)]['psnr'] - img('A1_idr_g_s0')[('kodak', 25)]['psnr']
    fx = img('A5_idrfix_g_s0')[('kodak', 25)]['psnr'] - img('A1_idr_g_s0')[('kodak', 25)]['psnr']
    tr.append(f'FP32 instead of TF32 (A1 control) & {fp:+.2f} & -- \\\\')
    tr.append(f'Fixed noisy copy instead of fresh noise (A5) & {fx:+.2f} & -- \\\\')

    def mean_delta(d, k):
        return np.mean([d[k][x]['psnr'] - d['base'][x]['psnr'] for x in keys if x in d[k]])
    inf = []
    for lab, k in (('BGR instead of RGB channel order', 'D1_bgr'), ('PSNR on float output (no uint8 rounding)', 'D3_float'),
                   ('Clipping the noisy input to $[0,1]$', 'D3_clip_input')):
        inf.append(f'{lab} & {mean_delta(a1, k):+.2f} & {mean_delta(a2, k):+.2f} \\\\')
    t100 = lambda d: d['D5_bsds_test100']['BSDS300_25']['psnr'] - d['base']['BSDS300_25']['psnr']
    inf.append(f'BSDS300 test-100 split instead of all 300 ($\\sigma$=25) & {t100(a1):+.2f} & {t100(a2):+.2f} \\\\')
    body = r"""\begin{table}[!t]
\centering
\caption{Attribution of the drift between our earlier reproduction, the paper protocol, and the paper (PSNR change in dB, same evaluator). Training-side rows: Kodak $\sigma$=25, relative to the protocol-P seed-0 model. Inference-side rows: mean over Kodak and BSDS300 at $\sigma$=25/50 (test-100 row: BSDS300 $\sigma$=25 only).}
\label{tab:drift}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcc}
\toprule
Factor & IDR & Supervised \\
\midrule
\multicolumn{3}{l}{\emph{Training side}} \\
""" + '\n'.join(tr) + r"""
\midrule
\multicolumn{3}{l}{\emph{Inference side}} \\
""" + '\n'.join(inf) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    write('tab_drift', body)


# ------------------------------------------------------------------ seismic tables
TESTS = [('m2', 'Marmousi2'), ('m2_prestm', 'M2 PreSTM$^\\ast$'), ('m2_postm', 'M2 PostSTM$^\\ast$'), ('m2_srme', 'M2 SRME$^\\ast$'),
         ('ot', 'Overthrust'), ('sa', 'Salt')]


def t_seis_additive():
    b = base('baselines_seis')
    lines = []
    for nz, title in (('sg', 'White Gaussian (sg)'), ('sb', 'Band-limited 5--60 Hz (sb)')):
        lines.append(r'\midrule' + '\n' + f'\\multicolumn{{12}}{{l}}{{\\emph{{{title}}}}} \\\\')
        if nz == 'sg':
            idr = [seis(f'C1_idr_sg_s{s}') for s in range(3)]; sup = [seis(f'C2_sup_sg_s{s}') for s in range(3)]
            nr = seis('C3_nr2n_sg_s0', 'eval_nr2ncorr_K8'); n2n = seis('C4_n2n_sg_s0'); fix = seis('C5_idrfix_sg_s0')
        else:
            idr = [seis('C7_idr_sb_s0')]; sup = [seis('C7_sup_sb_s0')]; nr = n2n = None; fix = seis('C6_idrfix_sb_s0')
        for key, name in TESTS:
            for lv in (0.5, 1.0):
                k = (key, nz, lv)
                vi = np.mean([d[k]['snr'] for d in idr]); vs = np.mean([d[k]['snr'] for d in sup])
                vals = [b[('bandpass',) + k]['snr'], b[('fx',) + k]['snr'], b[('bm3d',) + k]['snr'],
                        nr[k]['snr'] if nr else None, fix[k]['snr'], vi, n2n[k]['snr'] if n2n else None, vs]
                fm = rank_fmt(vals, [True] * 6 + [False, False])
                lines.append(f'{name if lv == 0.5 else ""} & {lv} & {idr[0][k]["snr_in"]:.2f} & ' + ' & '.join(fm) + ' \\\\')
    body = r"""\begin{table*}[!t]
\centering
\caption{Synthetic seismic, additive noise: output SNR (dB) of the joint-corpus models (trained on Marmousi2, Overthrust and Salt) on held-out sections; mean over test sections and 3 test-noise seeds. $^\ast$Processing versions of Marmousi2 never seen in training. sg Fast IDR and supervised: mean over training seeds 0--2; all other entries seed 0. BM3D is given the oracle noise $\sigma$ (sg) or PSD (sb). Nr2N: stage-0 network, $2f(z)-z$, $K=8$. IDR-fix: Fast IDR trained on one fixed noisy realization. Bold/underline: best/second among methods without clean data or a second noisy copy (supervised and N2N excluded).}
\label{tab:seis_add}
\begin{tabular}{llcccccccccc}
\toprule
Test & $\sigma$ & Input & Bandpass & $f$-$x$ & BM3D & Nr2N & IDR-fix & Fast IDR & N2N & Supervised \\
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    body = body.replace(r'\multicolumn{12}', r'\multicolumn{11}').replace('{llcccccccccc}', '{llccccccccc}')
    write('tab_seis_add', body)


def t_seis_sparse():
    b = base('baselines_seis')
    lines = []
    for nz, title, lvls, cl, cln in (('ss', 'Erratic spikes (ss), rate $p$', (0.05, 0.1), ('fx', 'median'), ('$f$-$x$', 'Median')),
                                     ('sd', 'Dead traces (sd), fraction', (0.3, 0.5), ('interp',), ('Interp.',))):
        idr = seis(f'C{8 if nz == "ss" else 9}_idr_{nz}_s0'); sup = seis(f'C{8 if nz == "ss" else 9}_sup_{nz}_s0')
        lines.append(r'\midrule' + '\n' + f'\\multicolumn{{7}}{{l}}{{\\emph{{{title}}}; classical: {", ".join(cln)}}} \\\\')
        for key, name in TESTS + [('f3', 'F3$^\\dagger$'), ('st', 'Stratton$^\\dagger$')]:
            for lv in lvls:
                k = (key, nz, lv)
                cvals = [b[(c,) + k]['snr'] for c in cl]
                vals = cvals + [idr[k]['snr'], sup[k]['snr']]
                fm = rank_fmt(vals, [True] * (len(cl) + 1) + [False])
                if len(cl) == 1:
                    fm = ['--'] + fm
                lines.append(f'{name if lv == lvls[0] else ""} & {lv} & {idr[k]["snr_in"]:.2f} & ' + ' & '.join(fm) + ' \\\\')
    body = r"""\begin{table}[!t]
\centering
\caption{Synthetic seismic, sparse noise (joint-corpus models, seed 0): output SNR (dB). Masked-$L_2$ loss for dead traces. Classical: $f$-$x$ deconvolution and $5\times3$ median for spikes; trace interpolation for dead traces (second classical column). $^\ast$Unseen processing; $^\dagger$field section with noise added (reference = original section). Bold/underline among classical and Fast IDR.}
\label{tab:seis_sparse}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llccccc}
\toprule
Test & Level & Input & Classical 1 & Classical 2 & Fast IDR & Sup. \\
""" + '\n'.join(lines).replace('& -- & ', '& -- & ', 1) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    write('tab_seis_sparse', body)


def t_mismatch():
    lines = []
    for run, lab in (('C1_idr_sg_s0', 'Fast IDR, sg'), ('C2_sup_sg_s0', 'Supervised, sg'), ('C7_idr_sb_s0', 'Fast IDR, sb'), ('C7_sup_sb_s0', 'Supervised, sb')):
        rows = J(RUNS / run / 'mismatch' / 'summary.json')['results']
        cells = []
        for nz in ('sg', 'sb'):
            for lv in (0.5, 1.0, 1.5):
                v = [r['snr'] for r in rows if r['noise'] == nz and abs(r['level'] - lv) < 1e-6 and r['test'] in ('m2', 'ot', 'sa')]
                cells.append(f'{np.mean(v):.2f}')
        lines.append(f'{lab} & ' + ' & '.join(cells) + ' \\\\')
    body = r"""\begin{table}[!t]
\centering
\caption{Noise-model mismatch (joint-corpus models, seed 0): output SNR (dB), mean over Marmousi2, Overthrust and Salt test sets. $\sigma=1.5$ lies outside the training range $U[0,1]$. Input SNR: 5.99, $-0.03$, $-3.55$ dB.}
\label{tab:mismatch}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{3}{c}{Test: white (sg)} & \multicolumn{3}{c}{Test: band-limited (sb)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Trained on & 0.5 & 1.0 & 1.5 & 0.5 & 1.0 & 1.5 \\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    write('tab_mismatch', body)


JOINT = {'sg': ('C1_idr_sg_s0', 'C2_sup_sg_s0'), 'sb': ('C7_idr_sb_s0', 'C7_sup_sb_s0'), 'ss': ('C8_idr_ss_s0', 'C8_sup_ss_s0')}


def t_perdataset():
    S3 = lambda run_fmt, k: np.array([seis(run_fmt.format(s=s))[k]['snr'] for s in range(3)])
    ms = lambda v: f'{v.mean():.2f}$\\pm${v.std():.2f}'
    lines = []
    for key, name in (('m2', 'Marmousi2'), ('ot', 'Overthrust'), ('sa', 'Salt')):
        k = (key, 'sg', 0.5)
        cells = [ms(S3(f'S_{key}_idr_sg_s{{s}}', k)), ms(S3(f'S_{key}_sup_sg_s{{s}}', k)),
                 f"{S3('C1_idr_sg_s{s}', k).mean():.2f}", f"{S3('C2_sup_sg_s{s}', k).mean():.2f}"]
        for nz, lv in (('sb', 0.5), ('ss', 0.05), ('sd', 0.3)):
            kk = (key, nz, lv)
            cells += [f"{seis(f'S_{key}_idr_{nz}_s0')[kk]['snr']:.2f}", f"{seis(f'S_{key}_sup_{nz}_s0')[kk]['snr']:.2f}"]
        fx = seis(f'S_{key}_idrfix_sg_s0')[k]['snr']
        lines.append(f'{name} & ' + ' & '.join(cells) + f' & {fx:.2f} \\\\')
    # transfer, sg 0.5, mean over training seeds 0-2
    tl = []
    cols = ['m2', 'm2_prestm', 'm2_postm', 'm2_srme', 'ot', 'sa']
    sds = []
    for tr, name in (('m2', 'Marmousi2'), ('ot', 'Overthrust'), ('sa', 'Salt'), ('joint', 'Joint')):
        fi = f'S_{tr}_idr_sg_s{{s}}' if tr != 'joint' else 'C1_idr_sg_s{s}'
        fs = f'S_{tr}_sup_sg_s{{s}}' if tr != 'joint' else 'C2_sup_sg_s{s}'
        cells = []
        for c in cols:
            vi, vs = S3(fi, (c, 'sg', 0.5)), S3(fs, (c, 'sg', 0.5))
            sds += [vi.std(), vs.std()]
            cells.append(f'{vi.mean():.2f}/{vs.mean():.2f}')
        tl.append(f'{name} & ' + ' & '.join(cells) + ' \\\\')
    inp = seis('C1_idr_sg_s0')
    body = r"""\begin{table*}[!t]
\centering
\caption{One model per synthetic survey (Kaggle T4, FP32) vs.\ the joint-corpus model (subscript $J$). Top: own-survey test SNR (dB). White noise $\sigma=0.5$: mean$\pm$sd over training seeds 0--2 (joint: mean over seeds 0--2); band-limited $\sigma=0.5$, spikes $p=0.05$, dead traces 0.3 and ``fix'' (Fast IDR on one fixed noisy copy, white): seed 0. Bottom: transfer, white noise $\sigma=0.5$, Fast IDR / supervised, mean over seeds 0--2 (seed sd up to """ + f'{max(sds):.2f}' + r"""\,dB); input SNR """ + ', '.join(f'{inp[(c, "sg", 0.5)]["snr_in"]:.2f}' for c in cols) + r""" dB respectively. $^\ast$Unseen processing.}
\label{tab:perdataset}
\begin{tabular}{lccccccccccc}
\toprule
& \multicolumn{4}{c}{White (sg)} & \multicolumn{2}{c}{Band-limited (sb)} & \multicolumn{2}{c}{Spikes (ss)} & \multicolumn{2}{c}{Dead traces (sd)} & sg \\
\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-12}
Survey & IDR & Sup & IDR$_J$ & Sup$_J$ & IDR & Sup & IDR & Sup & IDR & Sup & fix \\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}

\vspace{4pt}
\begin{tabular}{lcccccc}
\toprule
Trained on $\downarrow$ / tested on $\rightarrow$ & Marmousi2 & M2 PreSTM$^\ast$ & M2 PostSTM$^\ast$ & M2 SRME$^\ast$ & Overthrust & Salt \\
\midrule
""" + '\n'.join(tl) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write('tab_perdataset', body)


# ------------------------------------------------------------------ field tables
FIELDK = [('l2', '2D land'), ('w2', '2D marine'), ('st', '3D land'), ('f3', '3D marine')]


def t_field():
    fs = J(R / 'report' / 'generated' / 'field_semisynth.json')
    rows = [('idr_survey', 'Fast IDR, trained on the survey'), ('sup_oracle', 'Supervised on survey (oracle)'),
            ('best_classical', 'Best classical'), ('idr_synth', 'Fast IDR, synthetic corpus'), ('sup_synth', 'Supervised, synthetic corpus'),
            ('input', 'Input')]
    nm = {'bm3d': 'B', 'fx': 'F', 'bandpass': 'P'}
    lines = []
    for m, lab in rows:
        cells = []
        for k, _ in FIELDK:
            for lv in ('0.5', '1.0'):
                v = fs[k][m][lv]
                s = f2(v)
                if m == 'best_classical':
                    s += f'$^{{{nm[fs[k]["best_classical_name"][lv]]}}}$'
                if m == 'idr_survey':
                    s = r'\textbf{' + s + '}'
                cells.append(s)
        if m in ('sup_oracle', 'input'):
            lab = r'\textit{' + lab + '}'
        lines.append(f'{lab} & ' + ' & '.join(cells) + ' \\\\')
    body = r"""\begin{table*}[!t]
\centering
\caption{Field data, semi-synthetic test: white noise added to field sections, original section as reference; output SNR (dB), mean over test sections and 3 noise seeds, seed-0 models. ``Trained on the survey'': Fast IDR trained on one fixed noisy copy of the survey's training sections (no clean data). Best classical: $^B$BM3D with oracle $\sigma$, $^P$bandpass 5--60 Hz, $^F$$f$-$x$ deconvolution; BM3D was not run on the 2D lines. The supervised oracle treats the field section as clean and was not run on the 2D lines. Bold: best non-oracle method.}
\label{tab:field_semi}
\begin{tabular}{lcccccccc}
\toprule
& \multicolumn{2}{c}{2D land (NPR-A)} & \multicolumn{2}{c}{2D marine (NAMSS)} & \multicolumn{2}{c}{3D land (Stratton)} & \multicolumn{2}{c}{3D marine (F3)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Method & $\sigma$=0.5 & 1.0 & 0.5 & 1.0 & 0.5 & 1.0 & 0.5 & 1.0 \\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write('tab_field_semi', body)

    # real noise
    lines = []
    for k, name in FIELDK:
        sm = J(RUNS / f'field_{k}' / 'summary.json')['results']
        fixed = 'idr_fixed' if k in ('l2', 'w2') else f'idr_fixed_{k}'
        cfg = J(RUNS / {'l2': 'G_l2_idr_real', 'w2': 'G_w2_idr_real', 'f3': 'F2_idr_f3_real', 'st': 'F4_idr_st_real'}[k] / 'config.json')
        lo, hi = cfg['level']
        first = True
        for m, lab in (('idr_real', 'Fast IDR, survey (raw + added noise)'), (fixed, 'Fast IDR, survey (fixed noisy copy)'),
                       ('idr_synth', 'Fast IDR, synthetic corpus'), ('sup_synth', 'Supervised, synthetic corpus')):
            r = sm[m]
            hdr = f'\\multirow{{4}}{{*}}{{\\shortstack[l]{{{name}\\\\$\\hat\\sigma$={(lo + hi) / 2:.3f}}}}}' if first else ''
            first = False
            lines.append(f'{hdr} & {lab} & {100 * r["removed_energy_frac"]:.2f} & {100 * r["removed_frac_in_band"]:.2f} & {100 * r["removed_frac_out_band"]:.2f} & {r["leak_ls"]:.3f} \\\\')
        lines.append(r'\midrule')
    lines = lines[:-1]
    body = r"""\begin{table}[!t]
\centering
\caption{Real field noise (no reference). Removed energy $\|x-\hat y\|^2/\|x\|^2$ in \%, overall and inside / outside 5--60 Hz (each relative to the raw energy in that band), and leakage = mean $|$local similarity$|$ between output and removed component (lower is better; unreliable when almost nothing is removed). Out-of-band fractions also count energy a model adds where the raw data have little (see Fig.~\ref{fig:field_spectra}). $\hat\sigma$: MAD noise estimate; the ``raw + added noise'' model was trained with $\sigma\sim U[0.5\hat\sigma,1.5\hat\sigma]$.}
\label{tab:field_real}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcccc}
\toprule
Data & Model & Removed & In band & Out of band & Leakage \\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    write('tab_field_real', body)


# ------------------------------------------------------------------ runtime + reproducibility
def runtime():
    def t(r):
        c = J(RUNS / r / 'config.json'); s = J(RUNS / r / 'training_summary.json')
        m = [json.loads(l) for l in (RUNS / r / 'metrics.jsonl').read_text().splitlines() if l.strip()]
        a, b = dt.datetime.fromisoformat(c['started_utc']), dt.datetime.fromisoformat(s['finished_utc'])
        return m[-1]['steps_per_s'], (b - a).total_seconds() / 60, c['gpu']
    rows = []
    for i, s, lab in (('A1_idr_g_s0', 'A2_sup_g_s0', 'Images, Gaussian'), ('C1_idr_sg_s0', 'C2_sup_sg_s0', 'Seismic joint, sg'),
                      ('C7_idr_sb_s0', 'C7_sup_sb_s0', 'Seismic joint, sb'), ('S_m2_idr_sg_s0', 'S_m2_sup_sg_s0', 'Marmousi2 only, sg'),
                      ('S_ot_idr_sg_s0', 'S_ot_sup_sg_s0', 'Overthrust only, sg')):
        ri, rs = t(i), t(s)
        gpu = 'RTX 5070 Ti' if '5070' in ri[2] else 'T4'
        rows.append(f'{lab} & {gpu} & {ri[0]:.2f} & {rs[0]:.2f} & {ri[1]:.0f} & {rs[1]:.0f} & {100 * (ri[1] / rs[1] - 1):+.0f}\\% \\\\')
    body = r"""\begin{table}[!t]
\centering
\caption{Training cost for 50k steps: average throughput (steps/s) and wall-clock time (min). Several jobs shared one GPU, so absolute times are indicative; the IDR/supervised ratio within a row is comparable.}
\label{tab:runtime}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llccccc}
\toprule
& & \multicolumn{2}{c}{steps/s} & \multicolumn{2}{c}{min} & \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Setting & GPU & IDR & Sup & IDR & Sup & IDR overhead \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    write('tab_runtime', body)


def t_repro():
    rows = [
        ('Image Gaussian (Tab.~\\ref{tab:img_gauss})', 'A1/A2 (s0--2), A3, A4, A6, A10', 'pp\\_train.py, eval\\_images.py, nr2n\\_correct.py, bm3d\\_images.py'),
        ('Image binomial/impulse/corr.', 'A7, A8, A9 (scn1--4)', 'pp\\_train.py, eval\\_images.py'),
        ('Drift attribution (Tab.~\\ref{tab:drift})', 'drift/, A1\\_fp32, A5', 'drift\\_images.py'),
        ('Seismic joint corpus', 'C1/C2 (s0--2), C3--C9', 'pp\\_train.py, eval\\_seis.py, baselines\\_seis.py'),
        ('Noise mismatch (Tab.~\\ref{tab:mismatch})', 'C1, C2, C7 \\texttt{mismatch/}', 'eval\\_seis.py --noises'),
        ('Per-survey models', 'S\\_\\{m2,ot,sa\\}\\_*', 'kgl/kernel\\_template.py, eval\\_seis.py'),
        ('3D field (F3, Stratton)', 'F1--F4, field\\_f3, field\\_st', 'pp\\_train.py, eval\\_field.py'),
        ('2D field (NPR-A, NAMSS)', 'G\\_\\{l2,w2\\}\\_*, field\\_\\{l2,w2\\}, baselines\\_seis\\_2d', 'kgl/kernel\\_template.py, eval\\_field.py'),
        ('Figures and tables', 'all of the above', 'scripts/make\\_figures.py, scripts/make\\_tables.py'),
    ]
    body = r"""\begin{table}[!t]
\centering
\caption{Reproducibility map. Run folders are under \texttt{runs/paperproto/}; each holds \texttt{config.json}, \texttt{metrics.jsonl}, \texttt{console.log} and the evaluation \texttt{summary.json}; checkpoints are \texttt{weights.pth} (final) and, for later runs, \texttt{weights\_step\{10..50\}000.pth}. Scripts are in \texttt{paperproto/} unless a path is given.}
\label{tab:repro}
\scriptsize
\begin{tabular}{@{}p{0.28\columnwidth}p{0.29\columnwidth}p{0.32\columnwidth}@{}}
\toprule
Experiment & Runs & Scripts \\
\midrule
""" + '\n'.join(f'{a} & {b} & {c} \\\\' for a, b, c in rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write('tab_repro', body)


if __name__ == '__main__':
    tot = t_unet(); print('params', tot)
    t_image_gauss(); t_image_other(); t_drift(); t_seis_additive(); t_seis_sparse(); t_mismatch(); t_perdataset()
    t_field(); runtime(); t_repro()
