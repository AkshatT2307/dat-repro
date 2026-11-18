"""Float-amplitude Marmousi2 patches; no image quantization or amplitude clipping."""
import hashlib
import json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset


def read_segy(path):
    """Read fixed-length big-endian SEG-Y (IBM/IEEE float or integer samples)."""
    formats = {2: '>i4', 3: '>i2', 5: '>f4', 8: 'i1'}
    with open(path, 'rb') as f:
        header = f.read(3600)
        if len(header) != 3600:
            raise ValueError('Truncated SEG-Y header')
        ns = int.from_bytes(header[3220:3222], 'big')
        dt = int.from_bytes(header[3216:3218], 'big')
        fmt = int.from_bytes(header[3224:3226], 'big')
        extended = int.from_bytes(header[3504:3506], 'big', signed=True)
        if extended < 0:
            raise ValueError('Variable extended headers not supported')
        f.seek(3600 + extended * 3200)
        if fmt != 1 and fmt not in formats:
            raise ValueError(f'Unsupported SEG-Y sample format {fmt}')
        size = 4 if fmt == 1 else np.dtype(formats[fmt]).itemsize
        traces = []
        while th := f.read(240):
            if len(th) != 240:
                raise ValueError('Truncated trace header')
            count = int.from_bytes(th[114:116], 'big') or ns
            if ns and count != ns:
                raise ValueError('Variable-length traces not supported')
            data = f.read(count * size)
            if len(data) != count * size:
                raise ValueError('Truncated trace samples')
            if fmt == 1:
                bits = np.frombuffer(data, dtype='>u4').astype(np.uint32)
                fraction = (bits & 0x00ffffff).astype(np.float64) / 2**24
                exponent = ((bits >> 24) & 0x7f).astype(np.int32) - 64
                values = np.where(bits >> 31, -1., 1.) * fraction * np.power(16., exponent)
            else:
                values = np.frombuffer(data, dtype=formats[fmt])
            traces.append(values.astype(np.float32))
    section = np.stack(traces, axis=1)
    if not np.isfinite(section).all():
        raise ValueError('Nonfinite seismic amplitudes')
    return section, dict(sample_format=fmt, sample_interval_header=dt,
                         shape=list(section.shape), orientation='time_samples, traces')


def spatial_splits(width, patch=128):
    train_end, val_end = int(width * .6), int(width * .8)
    splits = dict(train=[0, train_end], validation=[train_end + patch, val_end],
                  test=[val_end + patch, width])
    if any(end-start < patch for start,end in splits.values()):
        raise ValueError('Section too narrow for three splits with 128-trace buffers')
    return splits


class SeismicPatches(Dataset):
    def __init__(self, root, samples=50000, patch=128):
        self.root = Path(root)
        self.metadata = json.loads((self.root/'metadata.json').read_text())
        self.section = np.load(self.root/'section.npy', mmap_mode='r')
        start, end = self.metadata['splits']['train']
        self.train = self.section[:, start:end]
        self.scale = self.metadata['train_rms']
        self.samples, self.patch = samples, patch
        self.metas = [str((self.root/'section.npy').resolve()),
                      json.dumps(self.metadata, sort_keys=True)]
    def __len__(self):
        return self.samples
    def __getitem__(self, index):
        h, w = self.train.shape
        y = np.random.randint(h-self.patch+1)
        x = np.random.randint(w-self.patch+1)
        patch = np.array(self.train[y:y+self.patch,x:x+self.patch],dtype=np.float32)/self.scale
        return {'gt': patch[...,None]}


def heldout_tiles(root, split, patch=128):
    root = Path(root)
    metadata = json.loads((root/'metadata.json').read_text())
    section = np.load(root/'section.npy', mmap_mode='r')
    start, end = metadata['splits'][split]
    # No overlap with training; fixed non-overlapping held-out evaluation tiles.
    for y in range(0,section.shape[0]-patch+1,patch):
        for x in range(start,end-patch+1,patch):
            yield y,x,np.asarray(section[y:y+patch,x:x+patch],dtype=np.float32)/metadata['train_rms']
