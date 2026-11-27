"""Proper Noisier2Noise inference for the nr2n runs (A3/C3), inference only.

The nr2n checkpoints are trained exactly like Fast IDR epoch 0 (input y+n', target y). Applying
that net directly to y (what eval_images/eval_seis do) is NOT the Nr2N estimator. Moran et al.
(CVPR 2020) estimate x = 2 f(z) - z with z = y + n', n' ~ same noise model as n (alpha = 1),
i.e. the oracle test sigma is used for n'. We add n' at test time (fixed torch seed) and apply
the correction; K>1 averages over K draws of n' (reported separately; K=1 is the paper's estimator).
Writes <run>/eval_nr2ncorr_K{K}/summary.json in the same format as the normal evals.
"""
import argparse, json, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R / 'paperproto'), str(R)]
import numpy as np, torch


class Nr2N(torch.nn.Module):
    def __init__(self, net, K):
        super().__init__(); self.net, self.K, self.sigma = net, K, None
        self.g = torch.Generator(device='cuda')

    def forward(self, y):
        self.g.manual_seed(1234)
        acc = 0
        for _ in range(self.K):
            z = y + self.sigma * torch.randn(y.shape, device=y.device, generator=self.g)
            acc = acc + 2 * self.net(z) - z
        return acc / self.K


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', required=True); p.add_argument('--K', type=int, default=1)
    a = p.parse_args()
    run = R / a.run; cfg = json.loads((run / 'config.json').read_text())
    assert cfg['method'] == 'nr2n' and cfg['noise'] in ('g', 'sg'), cfg
    torch.backends.cudnn.deterministic = True
    out = run / f'eval_nr2ncorr_K{a.K}'; out.mkdir(exist_ok=True)
    if cfg['domain'] == 'image':
        import eval_images as E
        net = Nr2N(E.load_net(run / 'weights.pth', 3), a.K)
        orig = E.test_noise

        def tn(gt, noise, level, rng):
            net.sigma = level / 255. if level > 1 else level
            return orig(gt, noise, level, rng)
        E.test_noise = tn
        rows, summary, _ = E.evaluate(net, 'g', False)
        res = dict(run=str(a.run), method='nr2n_corrected', K=a.K, noise='g', results=summary)
    else:
        import eval_seis as E, seis_common as S
        net = Nr2N(E.load_net(run / 'weights.pth'), a.K)
        orig = S.test_noise

        def tn(clean, noise, level, *args, **kw):
            net.sigma = level
            return orig(clean, noise, level, *args, **kw)
        S.test_noise = tn
        rows, _ = E.run_eval(net, 'sg', S.CORPUS_TESTS, S.LEVELS['sg'], S.TEST_SEEDS, keep_first=False)
        res = dict(run=str(a.run), method='nr2n_corrected', K=a.K, train_noise='sg', results=E.summarise(rows))
    (out / 'summary.json').write_text(json.dumps(res, indent=2) + '\n')
    for s in res['results']:
        print(json.dumps(s))


if __name__ == '__main__':
    main()
