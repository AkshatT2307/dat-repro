"""Benchmark evaluation reproducing upstream test.py: same test noise per type, same
reflect-to-32 padding, uint8 rounding/clipping of the output, skimage PSNR/SSIM
(data_range 255, multichannel defaults), image-wise means.

Defaults: RGB input (consistent with the RGB training loader), test seed 2026.
Variant flags exist for the drift study (D1-D5).
"""
import argparse, csv, json, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R / 'paperproto'), str(R)]
import cv2, numpy as np, torch, torch.nn.functional as F
import noise as _N  # scipy shim for bm3d kernels
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from models.unet import UNet_n2n_un
from utils.util import get_experiment_noise

SETS = {'kodak': (R / 'data/kodak', 24), 'BSDS300': (R / 'data/BSDS300', 300), 'BSD68': (R / 'data/BSD68', 68)}


def files(name):
    d, n = SETS[name]
    f = sorted(p for p in d.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    assert len(f) == n, (name, len(f))
    return f


def protocol(noise, gray):
    if noise == 'g':
        return (['BSD68'] if gray else ['kodak', 'BSDS300']), [25, 50], True
    if noise in ('binomial', 'impulse'):
        return ['kodak', 'BSDS300'], [0.5], False
    return ['kodak', 'BSDS300'], [5], False          # scn-k


def test_noise(gt, noise, level, rng):
    """upstream test.py add_noise (gt float in [0,1], HWC)."""
    sigma = level / 255. if level > 1 else level
    if noise == 'g':
        return gt + rng.normal(0, sigma, gt.shape)
    h, w, c = gt.shape
    if noise == 'binomial':
        return gt * rng.binomial(n=1, p=1 - sigma, size=(h, w, 1))
    if noise == 'impulse':
        return gt * rng.binomial(n=1, p=1 - sigma, size=gt.shape)
    k = int(noise.split('-')[1])
    n, _, _ = get_experiment_noise('g%d' % k, sigma, 0, (h, w, 3))   # upstream: realisation 0
    return gt + n


def forward(net, x):
    h, w, _ = x.shape
    pw, ph = (w + 31) // 32 * 32 - w, (h + 31) // 32 * 32 - h
    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(x)).float().cuda().permute(2, 0, 1)[None]
        t = F.pad(t, (0, pw, 0, ph), mode='reflect')
        o = net(t)[..., :h, :w]
    return o[0].permute(1, 2, 0).cpu().numpy()


def load_net(weights, ch):
    net = UNet_n2n_un(ch, ch).cuda().eval()
    sd = torch.load(weights, map_location='cpu', weights_only=True)
    net.load_state_dict({k.removeprefix('module.'): v for k, v in sd.items()})
    return net


def evaluate(net, noise, gray, seed=2026, bgr=False, float_metrics=False, clip_input=False,
             ssim_gauss=False, sets=None, keep=()):
    names, levels, zero_mean = protocol(noise, gray)
    rows, kept = [], {}
    for name in (sets or names):
        fl = files(name)
        for level in levels:
            rng = np.random.RandomState(seed)
            for f in fl:
                if gray:
                    gt = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)[..., None]
                else:
                    gt = cv2.imread(str(f))
                    if not bgr:
                        gt = gt[..., ::-1]
                gt = np.ascontiguousarray(gt)
                noisy = test_noise(gt.astype(float) / 255., noise, level, rng)
                if clip_input:
                    noisy = np.clip(noisy, 0, 1)
                x = noisy - .5 if zero_mean else noisy
                den = forward(net, x) + (.5 if zero_mean else 0)
                if float_metrics:
                    pred, ref, dr = np.clip(den, 0, 1), gt / 255., 1.0
                else:
                    pred, ref, dr = np.clip(den * 255. + .5, 0, 255).astype(np.uint8), gt, 255
                kw = dict(gaussian_weights=True, sigma=1.5, use_sample_covariance=False) if ssim_gauss else {}
                rows.append(dict(dataset=name, level=level, image=f.name,
                                 psnr=float(psnr(ref, pred, data_range=dr)),
                                 ssim=float(ssim(ref, pred, data_range=dr, channel_axis=-1, **kw))))
                if f.name in keep:
                    kept[f'{name}_{level}_{f.stem}'] = dict(gt=gt, noisy=noisy.astype(np.float32), out=den.astype(np.float32))
    summary = []
    for name in (sets or names):
        for level in levels:
            sub = [r for r in rows if r['dataset'] == name and r['level'] == level]
            summary.append(dict(dataset=name, level=level, n=len(sub), psnr=float(np.mean([r['psnr'] for r in sub])),
                                ssim=float(np.mean([r['ssim'] for r in sub]))))
    return rows, summary, kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True); p.add_argument('--weights', default='weights.pth')
    p.add_argument('--tag', default='eval'); p.add_argument('--seed', type=int, default=2026)
    a = p.parse_args()
    run = Path(a.run); cfg = json.loads((run / 'config.json').read_text())
    gray = bool(cfg.get('gray')); ch = 1 if gray else 3
    torch.backends.cudnn.deterministic = True
    net = load_net(run / a.weights, ch)
    keep = ('kodim23.png', 'kodim05.png', '86000.jpg', 'test003.png')
    rows, summary, kept = evaluate(net, cfg['noise'], gray, a.seed, keep=keep)
    out = run / a.tag; out.mkdir(exist_ok=True)
    with (out / 'per_image.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if kept:
        np.savez_compressed(out / 'examples.npz', **{f'{k}__{kk}': vv for k, v in kept.items() for kk, vv in v.items()})
    (out / 'summary.json').write_text(json.dumps(dict(run=str(run), weights=a.weights, seed=a.seed, noise=cfg['noise'],
                                                      method=cfg['method'], results=summary), indent=2) + '\n')
    for s in summary:
        print(json.dumps(s))


if __name__ == '__main__':
    main()
