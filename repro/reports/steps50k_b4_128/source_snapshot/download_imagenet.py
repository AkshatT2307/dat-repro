"""Download ONLY Hugging Face ImageNet validation shards; preserve image bytes."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Keep downloaded datasets inside this workspace, but retain the user's login location.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
os.environ['HF_DATASETS_CACHE'] = str(ROOT/'data/hf_cache/datasets')
os.environ['HF_HUB_CACHE'] = str(ROOT/'data/hf_cache/hub')
from datasets import load_dataset, Image
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import HfApi, get_token
from huggingface_hub.utils import GatedRepoError


def main():
    repo = 'ILSVRC/imagenet-1k'
    token = get_token()
    print(f'Hugging Face login available: {bool(token)}', flush=True)
    api = HfApi(token=token)
    info = api.dataset_info(repo)
    # Explicit file selection prevents preparation/downloading of other splits.
    files = sorted(s.rfilename for s in info.siblings
                   if Path(s.rfilename).name.startswith('validation-')
                   and s.rfilename.endswith('.parquet'))
    if not files:
        raise RuntimeError('No validation parquet shards found; inspect upstream layout before downloading.')
    print(f'Fetching validation only: {len(files)} shards, revision {info.sha}', flush=True)
    ds = load_dataset(repo, data_files={'validation': files}, split='validation',
                      revision=info.sha, token=token,
                      # The repository metadata describes train/test as well.
                      # We intentionally prepare validation only; check its size below.
                      verification_mode='no_checks',
                      cache_dir=str(ROOT/'data/hf_cache/datasets'))
    if len(ds) != 50000:
        raise ValueError(f'Expected 50,000 validation images, received {len(ds)}')
    ds = ds.cast_column('image', Image(decode=False))
    dest = ROOT/'data/imagenet_val'; dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, row in enumerate(ds):
        payload = row['image']['bytes']
        if payload is None:
            payload = Path(row['image']['path']).read_bytes()
        # Preserve originals, including JPEG compression; no image re-encoding.
        name = f'val_{index:08d}.JPEG'
        path = dest/name
        if not path.exists() or path.read_bytes() != payload:
            temporary = path.with_suffix('.part')
            temporary.write_bytes(payload); temporary.replace(path)
        manifest.append(dict(file=name, sha256=hashlib.sha256(payload).hexdigest()))
        if (index+1) % 5000 == 0:
            print(f'Exported {index+1}/50000 images', flush=True)
    count = sum(p.suffix.lower() in {'.jpg','.jpeg','.png'} for p in dest.iterdir())
    if count != 50000:
        raise RuntimeError('Destination has extra images; use a clean destination before training.')
    (ROOT/'data/imagenet_source.json').write_text(json.dumps(dict(
        repository=repo, revision=info.sha, split='validation', shards=files,
        images=manifest), indent=2)+'\n')
    print(f'Ready: {dest} (50,000 images). No training/test shards requested.', flush=True)

if __name__ == '__main__':
    try:
        main()
    except (GatedRepoError, DatasetNotFoundError) as error:
        if isinstance(error, DatasetNotFoundError) and 'gated' not in str(error).lower():
            raise
        raise SystemExit('Access denied: accept the dataset conditions on Hugging Face and log in locally using .venv/bin/hf auth login. Do not paste your token into chat.')
