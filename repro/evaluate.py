"""Evaluate both models on identical seeded RGB Gaussian corruptions."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from models.unet import UNet_n2n_un
from test import model_forward


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', required=True)
    p.add_argument('--data', default='data')
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()
    net = UNet_n2n_un(3, 3).cuda().eval()
    state = torch.load(args.weights, map_location='cpu', weights_only=True)
    net.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()})
    rows, summary = [], []
    for dataset, count in [('kodak', 24), ('BSDS300', 300)]:
        files = sorted(f for f in (Path(args.data)/dataset).rglob('*') if f.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        if len(files) != count:
            raise ValueError(f'{dataset}: expected {count} images, got {len(files)}')
        for sigma in (25, 50):
            rng = np.random.RandomState(args.seed)
            subset = []
            for file in files:
                gt = cv2.cvtColor(cv2.imread(str(file)), cv2.COLOR_BGR2RGB)
                noisy = gt.astype(float)/255 + rng.normal(0, sigma/255, gt.shape)
                denoised = model_forward(net, noisy-0.5)+0.5
                # Preserve upstream uint8 rounding/clipping and metric defaults.
                pred = np.clip(denoised*255+0.5, 0, 255).astype(np.uint8)
                row = dict(dataset=dataset, sigma=sigma, image=file.name,
                           psnr=float(peak_signal_noise_ratio(gt, pred, data_range=255)),
                           ssim=float(structural_similarity(gt, pred, data_range=255, channel_axis=-1)))
                subset.append(row); rows.append(row)
            result = dict(dataset=dataset, sigma=sigma,
                          psnr=float(np.mean([r['psnr'] for r in subset])),
                          ssim=float(np.mean([r['ssim'] for r in subset])))
            summary.append(result)
            print(json.dumps(result), flush=True)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    with (out/'per_image.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(dict(weights=args.weights, seed=args.seed, results=summary), indent=2)+'\n')

if __name__ == '__main__':
    main()
