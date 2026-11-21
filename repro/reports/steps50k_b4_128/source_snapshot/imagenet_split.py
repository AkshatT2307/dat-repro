"""Stable file-level split; no images are moved and no crops cross image splits."""
import hashlib
import json
from pathlib import Path
import numpy as np


def split_paths(paths, count=1000, seed=2026):
    paths=sorted(str(p) for p in paths)
    if count < 1 or count >= len(paths):
        raise ValueError('Validation count must be between 1 and N-1')
    indices=set(np.random.RandomState(seed).permutation(len(paths))[:count].tolist())
    train=[p for i,p in enumerate(paths) if i not in indices]
    validation=[p for i,p in enumerate(paths) if i in indices]
    fingerprint=hashlib.sha256(json.dumps(dict(seed=seed,train=train,validation=validation),sort_keys=True).encode()).hexdigest()
    return train,validation,fingerprint


def write_split(root, output, count=1000, seed=2026):
    root=Path(root).resolve()
    paths=[p for p in root.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}]
    if len(paths)!=50000:
        raise ValueError(f'Expected 50000 source images, found {len(paths)}')
    train,validation,fingerprint=split_paths(paths,count,seed)
    Path(output).write_text(json.dumps(dict(source=str(root),seed=seed,fingerprint=fingerprint,
        train=train,validation=validation),indent=2)+'\n')
    print(f'Fixed split: {len(train)} training / {len(validation)} validation; manifest {output}')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/imagenet_val')
    p.add_argument('--output',default='data/imagenet_split.json');p.add_argument('--count',type=int,default=1000)
    p.add_argument('--seed',type=int,default=2026);a=p.parse_args()
    write_split(a.data,a.output,a.count,a.seed)
