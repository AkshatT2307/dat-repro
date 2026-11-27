"""A10: (C)BM3D on the identical test noise used for the networks (evaluator calibration, D9)."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
R = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R / 'paperproto'), str(R)]
import numpy as np, cv2, scipy.signal
if not hasattr(scipy.signal, 'gaussian'): scipy.signal.gaussian = scipy.signal.windows.gaussian
if not hasattr(np, 'float'): np.float = float
import bm3d
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from eval_images import files, test_noise

def task(args):
    name, sigma, idx = args
    f = files(name)[idx]; gray = name == 'BSD68'
    rng = np.random.RandomState(2026)
    for j in range(idx):            # advance the shared RandomState exactly as eval_images does
        g = cv2.imread(str(files(name)[j]), cv2.IMREAD_GRAYSCALE if gray else 1)
        rng.normal(0, 1, g.shape if not gray else g.shape + (1,))
    gt = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)[..., None] if gray else np.ascontiguousarray(cv2.imread(str(f))[..., ::-1])
    noisy = test_noise(gt.astype(float) / 255., 'g', sigma, rng)
    den = bm3d.bm3d(noisy[..., 0], sigma / 255.)[..., None] if gray else bm3d.bm3d_rgb(noisy, sigma / 255.)
    pred = np.clip(den * 255. + .5, 0, 255).astype(np.uint8)
    return dict(dataset=name, level=sigma, image=f.name, psnr=float(psnr(gt, pred, data_range=255)),
                ssim=float(ssim(gt, pred, data_range=255, channel_axis=-1)))

if __name__ == '__main__':
    tasks = [(n, s, i) for n in ('kodak', 'BSDS300', 'BSD68') for s in (25, 50) for i in range(len(files(n)))]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 10) as p:
        rows = p.map(task, tasks, chunksize=1)
    out = R / 'runs/paperproto/A10_bm3d_images'; out.mkdir(parents=True, exist_ok=True)
    import csv
    with (out / 'per_image.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = [dict(dataset=n, level=s, n=len(v), psnr=float(np.mean([r['psnr'] for r in v])), ssim=float(np.mean([r['ssim'] for r in v])))
            for n in ('kodak', 'BSDS300', 'BSD68') for s in (25, 50) for v in [[r for r in rows if r['dataset'] == n and r['level'] == s]]]
    (out / 'summary.json').write_text(json.dumps(dict(results=summ), indent=2) + '\n'); print(summ)
