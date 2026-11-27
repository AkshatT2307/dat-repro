"""Noise models for the paper-protocol runs.

Image noises reproduce the released IDR `utils.add_noise` exactly (Gaussian and
correlated call it directly; binomial/impulse are re-implemented with identical
torch RNG calls so that the drop mask can also be returned for the masked loss).
Seismic noises are the seismic analogues defined in PAPER_PROTOCOL_PLAN.md.
All functions take/return (B, C, H, W) tensors; for seismic H = time, W = trace.
"""
import math
import numpy as np
import scipy.signal
import torch

# Environment shim: bm3d.gaussian_kernel (used by upstream correlated noise) calls
# scipy.signal.gaussian, which SciPy >= 1.13 moved to scipy.signal.windows (same function).
if not hasattr(scipy.signal, 'gaussian'):
    scipy.signal.gaussian = scipy.signal.windows.gaussian

from utils import add_noise as upstream_add_noise

# Training ranges (paper / released configs) -------------------------------------------------
IMAGE_TRAIN = {
    'g': dict(noise_type='g', noise_level=[0, 50]),
    'binomial': dict(noise_type='binomial', noise_level=0.95),
    'impulse': dict(noise_type='impulse', noise_level=0.95),
    'scn-1': dict(noise_type='scn-1', noise_level=5),
    'scn-2': dict(noise_type='scn-2', noise_level=5),
    'scn-3': dict(noise_type='scn-3', noise_level=5),
    'scn-4': dict(noise_type='scn-4', noise_level=5),
}
SEIS_TRAIN = {
    'sg': dict(level=(0.0, 1.0)),      # sigma in RMS units
    'sb': dict(level=(0.0, 1.0), band=(5.0, 60.0)),
    'ss': dict(level=(0.0, 0.2)),      # spike probability
    'sd': dict(level=0.95),            # dead-trace level, upstream binomial construction
}
MASKED = {'binomial', 'impulse', 'sd'}   # masked-L2 loss ("as recommended in N2N")


class Cfg(dict):
    __getattr__ = dict.__getitem__

    def __setattr__(self, k, v):
        self[k] = v


def image_cfg(noise):
    c = Cfg(IMAGE_TRAIN[noise])
    c.mixed_batch_nl = False
    return c


def bandpass_gain(n, dt, band, order=4, device='cpu'):
    """Zero-phase Butterworth-shaped amplitude response on rfft frequencies."""
    f = torch.fft.rfftfreq(n, d=dt, device=device)
    lo, hi = band
    g = 1 / torch.sqrt(1 + (lo / f.clamp(min=1e-6)) ** (2 * order)) / torch.sqrt(1 + (f / hi) ** (2 * order))
    g[0] = 0
    return g


def band_limited(shape, dt, band, device, generator=None):
    """Unit-variance Gaussian noise band-limited along the time axis (dim -2)."""
    white = torch.randn(shape, device=device, generator=generator)
    n = shape[-2]
    g = bandpass_gain(n, dt, band, device=device)
    spec = torch.fft.rfft(white, dim=-2) * g[:, None]
    out = torch.fft.irfft(spec, n=n, dim=-2)
    # Theoretical std of filtered unit white noise (Parseval on the rfft grid).
    w = torch.ones_like(g); w[1:] = 2
    if n % 2 == 0:
        w[-1] = 1
    std = torch.sqrt((w * g ** 2).sum() / n)
    return out / std


_KERNELS = {}


def correlated_gpu(x, k, level=5):
    """GPU port of upstream get_experiment_noise('g<k>', level/255, ., (h,w,3)): white
    Gaussian of size sz+2*half_kernel, 'same' FFT convolution with the same kernel,
    then the same edge crop. Same distribution; RNG stream differs from scipy's."""
    from utils.util import get_experiment_kernel
    b, c, h, w = x.shape
    key = (k, h, w, x.device)
    if key not in _KERNELS:
        ker = get_experiment_kernel('g%d' % k, level / 255.0, (h, w, 3))
        _KERNELS[key] = torch.tensor(np.real(ker), dtype=torch.float32, device=x.device)
    ker = _KERNELS[key]
    kh, kw = ker.shape
    hh, hw = int(np.ceil(kh / 2)), int(np.ceil(kw / 2))
    H, W = h + 2 * hh, w + 2 * hw
    white = torch.randn(b, c, H, W, device=x.device)
    fh, fw = H + kh - 1, W + kw - 1
    full = torch.fft.irfft2(torch.fft.rfft2(white, s=(fh, fw)) * torch.fft.rfft2(ker, s=(fh, fw)), s=(fh, fw))
    sh, sw = (kh - 1) // 2, (kw - 1) // 2          # scipy 'same' = centred slice of the full output
    same = full[..., sh:sh + H, sw:sw + W]
    return x + same[..., hh:hh + h, hw:hw + w]


def add_image_noise(x, noise):
    """Returns (noisy, keep_mask or None). Exactly the upstream sampling."""
    cfg = image_cfg(noise)
    if noise.startswith('scn') and x.is_cuda:
        return correlated_gpu(x, int(noise.split('-')[1]), cfg.noise_level).float(), None
    if noise in ('binomial', 'impulse'):
        # Mirrors utils.add_noise binomial/impulse line by line, but keeps the mask.
        sigma = cfg.noise_level
        b, c, h, w = x.shape
        mask_shape = (b, 1, h, w) if noise == 'binomial' else (b, c, h, w)
        mask = torch.empty(mask_shape, device=x.device).uniform_(0, 1)
        mask = mask * torch.empty(b, 1, 1, 1, device=x.device).uniform_(0, sigma)
        mask = 1 - torch.bernoulli(mask)
        return (x * mask).float(), mask.expand_as(x)
    noisy, _, _ = upstream_add_noise(x, cfg)
    return noisy, None


def add_seis_noise(x, noise, dt=0.004, level=None):
    """Seismic analogues. level overrides the training range (fixed test value)."""
    b, c, h, w = x.shape
    spec = SEIS_TRAIN[noise]
    if noise in ('sg', 'sb'):
        if level is None:
            lo, hi = spec['level']
            level = float(np.random.uniform(lo, hi))   # one sigma per batch (upstream convention)
        n = torch.randn_like(x) if noise == 'sg' else band_limited(x.shape, dt, spec['band'], x.device)
        return x + level * n, None
    if noise == 'ss':
        if level is None:
            level = float(np.random.uniform(*spec['level']))
        hit = torch.bernoulli(torch.full_like(x, level))
        amp = torch.empty_like(x).uniform_(2, 8) * (torch.randint_like(x, 0, 2) * 2 - 1)
        return x + hit * amp, None
    if noise == 'sd':
        if level is None:   # upstream binomial construction, applied per trace
            mask = torch.empty(b, 1, 1, w, device=x.device).uniform_(0, 1)
            mask = mask * torch.empty(b, 1, 1, 1, device=x.device).uniform_(0, spec['level'])
            mask = 1 - torch.bernoulli(mask)
        else:               # test: each trace dead with probability exactly `level`
            mask = 1 - torch.bernoulli(torch.full((b, 1, 1, w), float(level), device=x.device))
        mask = mask.expand_as(x).contiguous()
        return x * mask, mask
    raise ValueError(noise)


def add_noise(x, noise, dt=0.004):
    if noise in IMAGE_TRAIN:
        return add_image_noise(x, noise)
    return add_seis_noise(x, noise, dt)


def loss_fn(out, target, noise, mask=None):
    if noise in MASKED:
        if mask is None:
            return torch.mean((out - target) ** 2)
        return torch.sum(mask * (out - target) ** 2) / mask.sum().clamp(min=1)
    return torch.mean(torch.abs(out - target))
