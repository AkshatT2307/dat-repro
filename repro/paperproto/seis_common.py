"""Shared seismic test protocol: test sets, deterministic test noise, metrics."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity

R = Path(__file__).resolve().parents[1]
SEIS = R / 'data' / 'seis'
LEVELS = {'sg': [0.5, 1.0], 'sb': [0.5, 1.0], 'ss': [0.05, 0.10], 'sd': [0.3, 0.5]}
CORPUS_TESTS = ['m2', 'm2_prestm', 'm2_postm', 'm2_srme', 'ot', 'sa', 'f3', 'st']
TEST_SEEDS = [0, 1, 2]


def load_test(key):
    meta = json.loads((SEIS / key / 'meta.json').read_text())
    return np.load(SEIS / key / 'test.npy'), meta


def bp_gain_np(n, dt, band=(5., 60.), order=4):
    f = np.fft.rfftfreq(n, dt)
    g = 1 / np.sqrt(1 + (band[0] / np.maximum(f, 1e-6)) ** (2 * order)) / np.sqrt(1 + (f / band[1]) ** (2 * order))
    g[0] = 0
    return g


def test_noise(clean, noise, level, seed, dt, key='', idx=0):
    """Deterministic in (noise, level, seed, key, section index); clean: (T, X)."""
    rng = np.random.default_rng([seed, idx, sum(map(ord, key)), int(level * 1000), sum(map(ord, noise))])
    if noise == 'sg':
        return clean + level * rng.standard_normal(clean.shape), None
    if noise == 'sb':
        n = clean.shape[0]; g = bp_gain_np(n, dt)
        w = np.ones_like(g); w[1:] = 2
        if n % 2 == 0: w[-1] = 1
        std = np.sqrt((w * g ** 2).sum() / n)
        x = np.fft.irfft(np.fft.rfft(rng.standard_normal(clean.shape), axis=0) * g[:, None], n, axis=0) / std
        return clean + level * x, None
    if noise == 'ss':
        hit = rng.random(clean.shape) < level
        amp = rng.uniform(2, 8, clean.shape) * rng.choice([-1, 1], clean.shape)
        return clean + hit * amp, None
    if noise == 'sd':
        live = rng.random(clean.shape[1]) >= level
        return clean * live[None], live
    raise ValueError(noise)


def local_similarity(a, b, win=(21, 21)):
    """Windowed correlation coefficient map (simplified local similarity, cf. Chen & Fomel 2015)."""
    ma, mb = uniform_filter(a, win), uniform_filter(b, win)
    cov = uniform_filter(a * b, win) - ma * mb
    va = uniform_filter(a * a, win) - ma ** 2
    vb = uniform_filter(b * b, win) - mb ** 2
    return cov / np.sqrt(np.maximum(va * vb, 1e-12))


def metrics(clean, noisy, out):
    err = out - clean
    mse = np.mean(err ** 2)
    dr = float(clean.max() - clean.min())
    snr = 10 * np.log10(np.sum(clean ** 2) / max(np.sum(err ** 2), 1e-20))
    snr_in = 10 * np.log10(np.sum(clean ** 2) / max(np.sum((noisy - clean) ** 2), 1e-20))
    removed = noisy - out
    ls = local_similarity(out, removed)
    return dict(snr_in=float(snr_in), snr=float(snr), snr_gain=float(snr - snr_in),
                psnr=float(10 * np.log10(dr ** 2 / max(mse, 1e-20))),
                ssim=float(structural_similarity(clean, out, data_range=dr)),
                rmse=float(np.sqrt(mse)), leak_ls=float(np.mean(np.abs(ls))))
