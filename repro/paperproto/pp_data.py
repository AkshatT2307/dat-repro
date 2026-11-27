"""Training datasets for protocol P.

ImageNet: all 50,000 validation images, upstream crop rule (random 256 crop,
resize when a side is <= 256), BGR->RGB as upstream, [0,1] floats, no augmentation.
Seismic: prepared surveys under data/seis/<key>/ (see prepare_seismic.py).
"""
import json
from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
SEIS = ROOT / 'data' / 'seis'


def upstream_crop(img, size, rng=np.random):
    h, w = img.shape[:2]
    if w <= size or h <= size:
        img = cv2.resize(img, (size, size))
        if img.ndim == 2:
            img = img[..., None]
        return img, None
    hs, ws = rng.randint(h - size), rng.randint(w - size)
    return img[hs:hs + size, ws:ws + size], (hs, ws)


class ImageNetTrain(Dataset):
    def __init__(self, path, gray=False, fixed_noise=False, seed=0, size=256):
        self.files = sorted(str(p) for p in Path(path).rglob('*')
                            if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
        if len(self.files) != 50000:
            raise ValueError(f'expected 50000 ImageNet validation images, found {len(self.files)}')
        self.gray, self.fixed, self.seed, self.size = gray, fixed_noise, seed, size
        self.metas = self.files

    def __len__(self):
        return len(self.files)

    def load(self, i):
        if self.gray:
            img = cv2.imread(self.files[i], cv2.IMREAD_GRAYSCALE)[..., None]
        else:
            img = cv2.imread(self.files[i])[..., ::-1]
        return np.asarray(img, dtype=np.float64) / 255.

    def __getitem__(self, i):
        img = self.load(i)
        if not self.fixed:
            crop, _ = upstream_crop(img, self.size)
            return {'gt': crop.astype(np.float32)}
        # One fixed noisy realisation per image: sigma_i ~ U[0,50]/255 and the noise
        # field are functions of (seed, image index) only, so every visit sees the
        # same noisy pixels; only the crop position changes.
        h, w = img.shape[:2]
        if w <= self.size or h <= self.size:
            img = cv2.resize(img, (self.size, self.size))
            if img.ndim == 2:
                img = img[..., None]
        rng = np.random.default_rng([self.seed, i, 7])
        sigma = rng.uniform(0, 50) / 255.
        noisy = img + rng.normal(0, sigma, img.shape)
        crop_clean, pos = upstream_crop(img, self.size)
        if pos is None:
            crop_noisy = noisy
        else:
            hs, ws = pos
            crop_noisy = noisy[hs:hs + self.size, ws:ws + self.size]
        return {'gt': crop_clean.astype(np.float32), 'noisy': crop_noisy.astype(np.float32)}


class SeisTrain(Dataset):
    """Random 256x256 crops; each sample picks a survey uniformly, then a section
    with probability proportional to its number of valid crop positions."""

    def __init__(self, keys, samples=50000, size=256, fixed=None):
        self.keys, self.samples, self.size = list(keys), samples, size
        self.fixed = fixed  # e.g. 'sg_s0' -> data/seis/<key>/train_fixed_sg_s0.npy
        self.metas = []
        self.meta = {}
        for k in self.keys:
            m = json.loads((SEIS / k / 'meta.json').read_text())
            self.meta[k] = m
            self.metas.append(f"{k}:{m['train_shape']}:{m['train_sha256']}")
        self._arrays = None

    def arrays(self):
        if self._arrays is None:  # lazily per worker (memmap)
            self._arrays = {}
            for k in self.keys:
                a = np.load(SEIS / k / 'train.npy', mmap_mode='r')
                f = np.load(SEIS / k / f'train_fixed_{self.fixed}.npy', mmap_mode='r') if self.fixed else None
                n, t, x = a.shape
                wts = np.full(n, max(t - self.size + 1, 1) * max(x - self.size + 1, 1), float)
                self._arrays[k] = (a, f, wts / wts.sum())
        return self._arrays

    def __len__(self):
        return self.samples

    def __getitem__(self, i):
        arrs = self.arrays()
        k = self.keys[np.random.randint(len(self.keys))]
        a, f, p = arrs[k]
        j = np.random.choice(len(p), p=p)
        _, t, x = a.shape
        y0 = np.random.randint(t - self.size + 1)
        x0 = np.random.randint(x - self.size + 1)
        out = {'gt': np.array(a[j, y0:y0 + self.size, x0:x0 + self.size], np.float32)[..., None]}
        if f is not None:
            out['noisy'] = np.array(f[j, y0:y0 + self.size, x0:x0 + self.size], np.float32)[..., None]
        return out
