"""Classical seismic baselines on the identical test noise used for the networks (CPU, parallel).

  bandpass : zero-phase Butterworth 5-60 Hz along time (order 4) - the signal band
  fx       : f-x deconvolution (Canales 1984): complex AR prediction across traces per
             frequency, forward+backward averaged, overlapping tapered windows
  bm3d     : BM3D with the true noise PSD (oracle sigma; band-limited PSD for 'sb')
  median   : 2-D 5x3 median filter (spikes)
  interp   : linear interpolation of dead traces from live neighbours (oracle mask)
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
R = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R / 'paperproto'), str(R)]
import numpy as np
from scipy.ndimage import median_filter
import seis_common as S
import scipy.signal
if not hasattr(scipy.signal, "gaussian"):
    scipy.signal.gaussian = scipy.signal.windows.gaussian   # shim for bm3d (SciPy >= 1.13)

METHODS = {'sg': ['bandpass', 'fx', 'bm3d'], 'sb': ['bandpass', 'fx', 'bm3d'], 'ss': ['median', 'fx'], 'sd': ['interp']}


def bandpass(d, dt):
    g = S.bp_gain_np(d.shape[0], dt)
    return np.fft.irfft(np.fft.rfft(d, axis=0) * g[:, None], d.shape[0], axis=0)


def _taper(n):
    return np.hanning(n + 2)[1:-1]


def fx_decon(d, dt, fmin=2., fmax=100., L=6, mu=0.01, wt=128, wx=32):
    T, X = d.shape
    out = np.zeros_like(d); wsum = np.zeros_like(d)
    ts = list(range(0, max(T - wt, 0) + 1, wt // 2)); ts += [T - wt] if ts[-1] != T - wt else []
    xs = list(range(0, max(X - wx, 0) + 1, wx // 2)); xs += [X - wx] if xs[-1] != X - wx else []
    tap = _taper(wt)[:, None] * _taper(wx)[None, :]
    freqs = np.fft.rfftfreq(wt, dt)
    band = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    for t0 in ts:
        for x0 in xs:
            w = d[t0:t0 + wt, x0:x0 + wx]
            D = np.fft.rfft(w, axis=0)
            P = D.copy()
            for k in band:
                s = D[k]
                def predict(s):
                    A = np.stack([s[L - 1 - j: len(s) - 1 - j] for j in range(L)], 1)   # rows n=L..N-1
                    b = s[L:]
                    AH = A.conj().T
                    a = np.linalg.solve(AH @ A + mu * np.trace(AH @ A).real / L * np.eye(L) + 1e-12 * np.eye(L), AH @ b)
                    p = s.copy(); p[L:] = A @ a
                    return p
                f = predict(s); bk = predict(s[::-1])[::-1]
                p = (f + bk) / 2
                p[:L] = bk[:L]; p[-L:] = f[-L:]
                P[k] = p
            P[[i for i in range(len(freqs)) if i not in set(band)]] = 0
            out[t0:t0 + wt, x0:x0 + wx] += np.fft.irfft(P, wt, axis=0) * tap
            wsum[t0:t0 + wt, x0:x0 + wx] += tap
    return out / np.maximum(wsum, 1e-8)


def bm3d_den(d, noise, level, dt):
    if not hasattr(np, 'float'):
        np.float = float   # shim: bm3d 3.x uses the removed np.float alias
    import bm3d
    if noise == 'sg':
        return bm3d.bm3d(d, sigma_psd=level)
    T, X = d.shape
    g = S.bp_gain_np(T, dt); n = T
    w = np.ones_like(g); w[1:] = 2
    if n % 2 == 0: w[-1] = 1
    std2 = (w * g ** 2).sum() / n
    gf = np.concatenate([g, g[1:(T + 1) // 2][::-1]]) if T % 2 else np.concatenate([g, g[1:T // 2][::-1]])
    psd = np.repeat((level ** 2 * gf[:T] ** 2 / std2)[:, None], X, 1) * T * X
    return bm3d.bm3d(d, sigma_psd=psd)


def interp(d, live):
    idx = np.where(live)[0]
    if len(idx) == 0:
        return d
    out = d.copy()
    for t in range(d.shape[0]):
        out[t] = np.interp(np.arange(d.shape[1]), idx, d[t, idx])
    return out


def task(args):
    key, i, noise, level, seed, method = args
    secs, meta = S.load_test(key)
    clean = secs[i].astype(np.float64); dt = meta['dt']
    noisy, live = S.test_noise(clean, noise, level, seed, dt, key, i)
    if method == 'bandpass': out = bandpass(noisy, dt)
    elif method == 'fx': out = fx_decon(noisy, dt)
    elif method == 'bm3d': out = bm3d_den(noisy, noise, level, dt)
    elif method == 'median': out = median_filter(noisy, size=(5, 3))
    elif method == 'interp': out = interp(noisy, live)
    keep = (seed == S.TEST_SEEDS[0] and i == len(secs) // 2)
    return dict(test=key, noise=noise, level=level, seed=seed, section=i, method=method, **S.metrics(clean, noisy, out)), \
        ((f'{key}__{noise}__{level}__{method}', out.astype(np.float32)) if keep else None)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--procs', type=int, default=10)
    p.add_argument('--noises', default='sg,sb,ss,sd'); p.add_argument('--out', default='runs/paperproto/baselines_seis')
    p.add_argument('--tests', default=None, help='comma list (default: corpus tests)')
    p.add_argument('--methods', default=None, help='comma list filter, e.g. bandpass,fx')
    a = p.parse_args()
    tasks = []
    for noise in a.noises.split(','):
        for key in (a.tests.split(',') if a.tests else S.CORPUS_TESTS):
            n = len(S.load_test(key)[0])
            for level in S.LEVELS[noise]:
                for seed in S.TEST_SEEDS:
                    for i in range(n):
                        for m in [m for m in METHODS[noise] if not a.methods or m in a.methods.split(',')]:
                            tasks.append((key, i, noise, level, seed, m))
    tasks.sort(key=lambda t: t[-1] != 'bm3d')   # longest first
    print(len(tasks), 'tasks', flush=True)
    out = R / a.out; out.mkdir(parents=True, exist_ok=True)
    rows, kept = [], {}
    with Pool(a.procs) as pool:
        for j, (row, k) in enumerate(pool.imap_unordered(task, tasks, chunksize=1)):
            rows.append(row)
            if k: kept[k[0]] = k[1]
            if j % 200 == 0: print(j, flush=True)
    import csv
    with (out / 'per_section.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez_compressed(out / 'examples.npz', **kept)
    keys = ['snr_in', 'snr', 'snr_gain', 'psnr', 'ssim', 'rmse', 'leak_ls']
    groups = {}
    for r in rows:
        groups.setdefault((r['method'], r['test'], r['noise'], r['level']), []).append(r)
    summ = [dict(method=m, test=t, noise=n, level=l, n=len(v), **{k: float(np.mean([x[k] for x in v])) for k in keys})
            for (m, t, n, l), v in sorted(groups.items())]
    (out / 'summary.json').write_text(json.dumps(dict(results=summ), indent=1) + '\n')
    (out / 'DONE').write_text('ok\n')


if __name__ == '__main__':
    main()
