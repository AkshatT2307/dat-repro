"""Controlled added-Gaussian-noise evaluation on held-out Marmousi2 amplitudes."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from models.unet import UNet_n2n_un
from seismic_data import heldout_tiles


def measures(clean, noisy, pred, scale):
    eps=1e-30
    signal=float(np.sum(clean.astype(float)**2))
    before=float(np.sum((noisy-clean).astype(float)**2))
    after=float(np.sum((pred-clean).astype(float)**2))
    corr=lambda a,b: float(np.corrcoef(a.ravel(),b.ravel())[0,1]) if np.std(a)>0 and np.std(b)>0 else None
    sin=10*np.log10((signal+eps)/(before+eps))
    sout=10*np.log10((signal+eps)/(after+eps))
    valid_signal=signal>clean.size*1e-12
    return dict(input_snr_db=float(sin) if valid_signal else None,
                output_snr_db=float(sout) if valid_signal else None,
                snr_gain_db=float(sout-sin) if valid_signal else None,
                rmse_normalized=float(np.sqrt(after/clean.size)),
                rmse_original_amplitude=float(np.sqrt(after/clean.size)*scale),
                signal_correlation=corr(clean,pred),residual_signal_correlation=corr(clean,noisy-pred))


def quicklook(path, clean, noisy, pred):
    panels=[('Reference',clean),('Noisy',noisy),('Denoised',pred),('Removed',noisy-pred)]
    canvas=Image.new('RGB',(4*256,280),'white'); draw=ImageDraw.Draw(canvas)
    for i,(title,array) in enumerate(panels):
        gray=np.round((np.clip(array,-3,3)+3)*255/6).astype(np.uint8)
        canvas.paste(Image.fromarray(gray).resize((256,256)),(i*256,24))
        draw.text((i*256+8,5),title,fill='black')
    canvas.save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',required=True);p.add_argument('--data',default='data/marmousi2')
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--split',choices=['validation','test'],default='test')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    a=p.parse_args()
    net=UNet_n2n_un(1,1).to(a.device).eval()
    net.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True))
    meta=json.loads((Path(a.data)/'metadata.json').read_text())
    tiles=list(heldout_tiles(a.data,a.split))
    if not tiles: raise ValueError('No held-out tiles')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rows=[];summaries=[]
    for sigma in (.1,.2,.4):
        rng=np.random.RandomState(a.seed)
        subset=[]
        for index,(y,x,clean) in enumerate(tiles):
            noisy=(clean+rng.normal(0,sigma,clean.shape)).astype(np.float32)
            with torch.no_grad():
                pred=net(torch.from_numpy(noisy.copy())[None,None].to(a.device))[0,0].cpu().numpy()
            if not np.isfinite(pred).all(): raise ValueError('Nonfinite prediction')
            row=dict(split=a.split,sigma=sigma,time_index=y,trace_index=x,
                     **measures(clean,noisy,pred,meta['train_rms']))
            rows.append(row);subset.append(row)
            if index==0:
                np.savez_compressed(out/f'example_sigma{sigma}.npz',reference=clean,noisy=noisy,
                                    denoised=pred,removed=noisy-pred,amplitude_scale=meta['train_rms'])
                quicklook(out/f'example_sigma{sigma}.png',clean,noisy,pred)
        keys=list(measures(clean,noisy,pred,meta['train_rms']))
        averages={}
        for key in keys:
            values=[r[key] for r in subset if r[key] is not None]
            averages[key]=float(np.mean(values)) if values else None
        summary=dict(sigma=sigma,tiles=len(subset),
                     signal_tiles=sum(r['input_snr_db'] is not None for r in subset), **averages)
        summaries.append(summary)
        print(f"sigma={sigma:.1f} | input SNR {summary['input_snr_db']:.2f} dB | "
              f"output {summary['output_snr_db']:.2f} dB | gain {summary['snr_gain_db']:.2f} dB",flush=True)
    with (out/'per_tile.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(dict(dataset=meta['dataset'],split=a.split,
        seed=a.seed,results=summaries,metric_aggregation='mean over nonoverlapping 128x128 tiles',
        reference='processed section before synthetic noise; not a noise-free field measurement'),indent=2)+'\n')

if __name__=='__main__': main()
