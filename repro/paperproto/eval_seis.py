"""Evaluate a seismic run on all held-out test surveys at the test levels of its noise type
(and optionally other noise types for the mismatch study). Whole-section inference."""
import argparse, csv, json, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R / 'paperproto'), str(R)]
import numpy as np, torch, torch.nn.functional as F
from models.unet import UNet_n2n_un
import seis_common as S


def load_net(path):
    net = UNet_n2n_un(1, 1).cuda().eval()
    net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return net


def infer(net, x):
    h, w = x.shape
    ph, pw = (h + 31) // 32 * 32 - h, (w + 31) // 32 * 32 - w
    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).cuda()[None, None]
        t = F.pad(t, (0, pw, 0, ph), mode='reflect')
        return net(t)[0, 0, :h, :w].cpu().numpy().astype(np.float64)


def run_eval(net, noise, tests, levels, seeds, keep_first=True):
    rows, kept = [], {}
    for key in tests:
        secs, meta = S.load_test(key)
        for level in levels:
            for seed in seeds:
                for i, clean in enumerate(secs):
                    clean = clean.astype(np.float64)
                    noisy, _ = S.test_noise(clean, noise, level, seed, meta['dt'], key, i)
                    out = infer(net, noisy)
                    rows.append(dict(test=key, noise=noise, level=level, seed=seed, section=i, **S.metrics(clean, noisy, out)))
                    if keep_first and seed == seeds[0] and i == len(secs) // 2:
                        kept[f'{key}__{noise}__{level}'] = np.stack([clean, noisy, out]).astype(np.float32)
    return rows, kept


def summarise(rows):
    out = {}
    for r in rows:
        out.setdefault((r['test'], r['noise'], r['level']), []).append(r)
    keys = ['snr_in', 'snr', 'snr_gain', 'psnr', 'ssim', 'rmse', 'leak_ls']
    return [dict(test=t, noise=n, level=l, n=len(v), **{k: float(np.mean([x[k] for x in v])) for k in keys},
                 snr_seed_std=float(np.std([np.mean([x['snr'] for x in v if x['seed'] == s]) for s in sorted({x['seed'] for x in v})])))
            for (t, n, l), v in out.items()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True); p.add_argument('--weights', default='weights.pth')
    p.add_argument('--tag', default='eval'); p.add_argument('--tests', default=None)
    p.add_argument('--noises', default=None, help='comma list; default = the training noise (mismatch study otherwise)')
    a = p.parse_args()
    run = Path(a.run); cfg = json.loads((run / 'config.json').read_text())
    torch.backends.cudnn.deterministic = True
    net = load_net(run / a.weights)
    keys = cfg['keys'].split(',')
    tests = a.tests.split(',') if a.tests else (S.CORPUS_TESTS if set(keys) == {'m2', 'ot', 'sa'} else keys)
    noises = a.noises.split(',') if a.noises else [cfg['noise']]
    rows, kept = [], {}
    for nz in noises:
        lv = S.LEVELS[nz] + ([1.5] if nz in ('sg', 'sb') and a.noises else [])
        r, k = run_eval(net, nz, tests, lv, S.TEST_SEEDS)
        rows += r; kept.update(k)
    out = run / a.tag; out.mkdir(exist_ok=True)
    with (out / 'per_section.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez_compressed(out / 'examples.npz', **kept)
    summ = summarise(rows)
    (out / 'summary.json').write_text(json.dumps(dict(run=str(run), method=cfg['method'], train_noise=cfg['noise'],
                                                      fixed=cfg.get('fixed'), results=summ), indent=2) + '\n')
    for s in summ:
        print(f"{s['test']:>10} {s['noise']} {s['level']:.2f}  in {s['snr_in']:6.2f}  out {s['snr']:6.2f} dB  "
              f"ssim {s['ssim']:.3f}  leak {s['leak_ls']:.3f}")


if __name__ == '__main__':
    main()
