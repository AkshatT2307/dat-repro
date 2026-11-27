"""Drift study D1-D5 (inference-only) for one Gaussian-RGB checkpoint. Writes runs/paperproto/drift/<name>.json."""
import argparse, json, sys, tarfile
from pathlib import Path
R = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R / 'paperproto'), str(R)]
import numpy as np, torch
from eval_images import evaluate, load_net


def bsds_test_ids():
    with tarfile.open(R / 'data/downloads/BSDS300-images.tgz') as t:
        return {l.strip() + '.jpg' for l in t.extractfile('BSDS300/iids_test.txt').read().decode().split() if l.strip()}


def table(rows):
    out = {}
    for r in rows:
        out.setdefault(f"{r['dataset']}_{r['level']}", []).append(r)
    return {k: dict(psnr=float(np.mean([x['psnr'] for x in v])), ssim=float(np.mean([x['ssim'] for x in v])), n=len(v))
            for k, v in out.items()}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--weights', required=True); p.add_argument('--name', required=True)
    p.add_argument('--seeds', type=int, default=10)
    a = p.parse_args()
    torch.backends.cudnn.deterministic = True
    net = load_net(R / a.weights, 3)
    res = {}
    rows, _, _ = evaluate(net, 'g', False)
    res['base'] = table(rows)
    tid = bsds_test_ids()
    res['D5_bsds_test100'] = table([r for r in rows if r['dataset'] == 'BSDS300' and r['image'] in tid])
    res['D5_bsds_train200'] = table([r for r in rows if r['dataset'] == 'BSDS300' and r['image'] not in tid])
    res['per_image_base'] = rows
    res['D1_bgr'] = table(evaluate(net, 'g', False, bgr=True)[0])
    res['D3_float'] = table(evaluate(net, 'g', False, float_metrics=True)[0])
    res['D3_clip_input'] = table(evaluate(net, 'g', False, clip_input=True)[0])
    res['D4_ssim_gauss'] = table(evaluate(net, 'g', False, ssim_gauss=True)[0])
    res['D2_seeds'] = {str(s): table(evaluate(net, 'g', False, seed=s)[0]) for s in range(2027, 2026 + a.seeds)}
    out = R / 'runs/paperproto/drift'; out.mkdir(parents=True, exist_ok=True)
    (out / f'{a.name}.json').write_text(json.dumps(dict(weights=a.weights, **res), indent=1) + '\n')
    print({k: v for k, v in res.items() if k not in ('per_image_base', 'D2_seeds')})


if __name__ == '__main__':
    main()
