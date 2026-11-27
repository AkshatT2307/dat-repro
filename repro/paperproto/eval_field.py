"""Apply models to RAW field test sections (real noise, no reference). Saves outputs and
no-reference diagnostics: removed-noise energy, |local similarity| between output and removed
noise (signal leakage), and the fraction of energy removed inside/outside 5-60 Hz."""
import argparse, json, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R / 'paperproto'), str(R)]
import numpy as np, torch
import seis_common as S
from eval_seis import load_net, infer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', required=True, help='name=run_dir,... (weights.pth)')
    p.add_argument('--key', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    secs, meta = S.load_test(a.key); dt = meta['dt']
    out = Path(R / a.out); out.mkdir(parents=True, exist_ok=True)
    res, arrays = {}, {'raw': secs}
    for item in a.models.split(','):
        name, run = item.split('=')
        net = load_net(R / run / 'weights.pth')
        outs = np.stack([infer(net, s.astype(np.float64)) for s in secs]).astype(np.float32)
        arrays[name] = outs
        rem = secs - outs
        f = np.fft.rfftfreq(secs.shape[1], dt); inb = (f >= 5) & (f <= 60)
        P = lambda x: (np.abs(np.fft.rfft(x, axis=1)) ** 2).sum((0, 2))
        pr, ps = P(rem), P(secs)
        res[name] = dict(removed_energy_frac=float((rem ** 2).sum() / (secs ** 2).sum()),
                         removed_frac_in_band=float(pr[inb].sum() / ps[inb].sum()),
                         removed_frac_out_band=float(pr[~inb].sum() / max(ps[~inb].sum(), 1e-12)),
                         leak_ls=float(np.mean([np.mean(np.abs(S.local_similarity(o.astype(float), r.astype(float))))
                                                for o, r in zip(outs, rem)])))
        print(name, res[name], flush=True)
    np.savez_compressed(out / 'outputs.npz', **arrays)
    (out / 'summary.json').write_text(json.dumps(dict(key=a.key, dt=dt, results=res), indent=2) + '\n')


if __name__ == '__main__':
    main()
