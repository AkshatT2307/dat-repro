"""GPU correlated noise must match upstream scipy noise in variance and 2-D power spectrum."""
import sys; from pathlib import Path
R = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(R / 'paperproto'), str(R)]
import numpy as np, torch, noise as N
from utils.util import get_experiment_noise
torch.manual_seed(0)
for k in (1, 2, 3, 4):
    up = np.stack([get_experiment_noise('g%d' % k, 5/255., s, (256, 256, 3))[0] for s in range(48)]).transpose(0, 3, 1, 2)
    gp = N.correlated_gpu(torch.zeros(48, 3, 256, 256, device='cuda'), k).cpu().numpy()
    psd = lambda a: (np.abs(np.fft.fft2(a)) ** 2).mean((0, 1))
    pu, pg = psd(up), psd(gp)
    rel = np.abs(pu - pg).sum() / pu.sum()
    print(f'g{k}: std upstream {up.std():.5f} gpu {gp.std():.5f}  PSD rel. L1 diff {rel:.3f}')
    assert abs(up.std() / gp.std() - 1) < 0.05 and rel < 0.25
print('ok')
