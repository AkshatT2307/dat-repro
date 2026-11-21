"""Download public evaluation sets; optionally extract a user-provided ImageNet archive."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent

def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix+'.part')
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(path)
    return dict(url=url, path=str(path.relative_to(ROOT)), sha256=hashlib.file_digest(path.open('rb'), 'sha256').hexdigest())

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--imagenet-tar', type=Path)
    p.add_argument('--benchmarks', action='store_true')
    args = p.parse_args()
    if args.benchmarks:
        records = []
        for i in range(1,25):
            name = f'kodim{i:02d}.png'
            records.append(fetch('https://r0k.us/graphics/kodak/kodak/'+name, ROOT/'data/kodak'/name))
        archive = ROOT/'data/downloads/BSDS300-images.tgz'
        records.append(fetch('https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz', archive))
        dest = ROOT/'data/BSDS300'
        dest.mkdir(parents=True, exist_ok=True)
        # Only color images, both original BSDS train/test partitions (300 total).
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                if member.isfile() and '/images/' in member.name and member.name.lower().endswith('.jpg'):
                    source = tar.extractfile(member)
                    (dest/Path(member.name).name).write_bytes(source.read())
        count = len(list(dest.glob('*.jpg')))
        if count != 300:
            raise ValueError(f'Expected 300 BSDS images; found {count}')
        (ROOT/'data/downloads.json').write_text(json.dumps(records, indent=2)+'\n')
        print('Prepared Kodak (24) and BSDS300 (300).')
    if args.imagenet_tar:
        dest = ROOT/'data/imagenet_val'
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.imagenet_tar) as tar:
            members = [m for m in tar.getmembers() if m.isfile() and Path(m.name).suffix.lower() in {'.jpg','.jpeg','.png'}]
            if len(members) != 50000 or len({Path(m.name).name for m in members}) != 50000:
                raise ValueError('Archive must contain 50,000 uniquely named validation images.')
            for m in members:
                (dest/Path(m.name).name).write_bytes(tar.extractfile(m).read())
        print(f'Extracted ImageNet validation images to {dest}')

if __name__ == '__main__':
    main()
