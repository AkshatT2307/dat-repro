"""Fetch verified Marmousi2 archive and prepare its NMO time-stack amplitudes."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request
import numpy as np
from seismic_data import read_segy, spatial_splits

ROOT=Path(__file__).resolve().parent/'data/marmousi2'
SHA='b6fe8d10ce6a6dfadf203a8d0e1a040ce9f2c6cb52c1366193d35113d61797fe'
URLS=[
 'https://huggingface.co/datasets/shaowinw/seismic-benchmark-models/resolve/main/marmousi2_elastic/elastic-marmousi-model.tar.gz',
 'https://zenodo.org/records/23469300/files/elastic-marmousi-model.tar.gz',
 'https://s3.amazonaws.com/open.source.geoscience/open_data/elastic-marmousi/elastic-marmousi-model.tar.gz',
]

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    downloads=ROOT/'downloads'; downloads.mkdir(exist_ok=True)
    archive=downloads/'elastic-marmousi-model.tar.gz'
    if not archive.exists():
        for url in URLS:
            try:
                print('Downloading',url,flush=True)
                temporary=archive.with_suffix('.part')
                with urllib.request.urlopen(url,timeout=60) as r, temporary.open('wb') as f:
                    while chunk:=r.read(1024*1024): f.write(chunk)
                if hashlib.sha256(temporary.read_bytes()).hexdigest()!=SHA:
                    raise ValueError('Archive checksum mismatch')
                temporary.replace(archive)
                break
            except Exception as e: print(type(e).__name__,str(e)[:120],flush=True)
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest()!=SHA:
        raise ValueError('Verified Marmousi2 archive unavailable')
    member='elastic-marmousi-model/processed_data/SEGY-Time/NMOstack_time.segy.tar.gz'
    with tarfile.open(archive) as outer:
        for name in ('README','LICENSE'):
            (ROOT/name).write_bytes(outer.extractfile('elastic-marmousi-model/'+name).read())
        payload=outer.extractfile(member).read()
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as inner:
        files=[m for m in inner.getmembers() if m.isfile() and m.name.lower().endswith(('.segy','.sgy'))]
        if len(files)!=1: raise ValueError('Expected one NMO time-stack SEG-Y')
        raw=inner.extractfile(files[0]).read()
    segy=ROOT/'NMOstack_time.segy'; segy.write_bytes(raw)
    section,header=read_segy(segy)
    splits=spatial_splits(section.shape[1])
    start,end=splits['train']
    rms=float(np.sqrt(np.mean(section[:,start:end].astype(np.float64)**2)))
    if not np.isfinite(rms) or rms<=0: raise ValueError('Invalid training RMS')
    np.save(ROOT/'section.npy',section)
    meta=dict(dataset='AGL Elastic Marmousi2 NMOstack_time',archive_sha256=SHA,
              section_sha256=hashlib.sha256(raw).hexdigest(),archive_member=member,
              sources=URLS,license='CC BY 4.0; see LICENSE and README',
              header=header,splits=splits,buffer_traces=128,patch=128,train_rms=rms,
              normalization='divide by training-region RMS, no clipping or mean shift',
              task='remove added Gaussian noise from processed reference amplitudes')
    (ROOT/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))
    print('Prepared',ROOT,flush=True)

if __name__=='__main__': main()
