"""Deterministic held-out examples using the four requested continued models."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from models.unet import UNet_n2n_un
from seismic_data import heldout_tiles

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/inference_examples'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def net_for(dataset,method,budget):
    weights=ROOT/'runs/continuation5'/dataset/method/budget/'best_weights.pth'
    channels=3 if dataset=='imagenet' else 1
    net=UNet_n2n_un(channels,channels).cuda().eval()
    net.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    return net,str(weights)


def predict(net,noisy,rgb):
    x=torch.from_numpy(noisy.copy()).permute(2,0,1)[None] if rgb else torch.from_numpy(noisy.copy())[None,None]
    with torch.inference_mode():
        result=net(x.cuda()-(.5 if rgb else 0))+(.5 if rgb else 0)
    return result[0].permute(1,2,0).cpu().numpy() if rgb else result[0,0].cpu().numpy()


def grid(rows,title,footnote,path,rgb):
    w,h=1104,120+len(rows)*318+45
    canvas=Image.new('RGB',(w,h),'#f5f6f8');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(FONT,16);large=ImageFont.truetype(FONT,21)
    draw.text((20,15),title,font=large,fill='#111827')
    for col,name in enumerate(['Clean reference','Noisy input','Fast IDR','Supervised U-Net']):
        draw.text((20+col*272,62),name,font=font,fill='#111827')
    for i,(arrays,labels,source) in enumerate(rows):
        y=100+i*318
        draw.text((20,y),source,font=font,fill='#374151')
        limit=max(float(np.percentile(np.abs(arrays[0]),99)),.1) if not rgb else None
        for col,(array,label) in enumerate(zip(arrays,labels)):
            display=np.clip(array,0,1) if rgb else (np.clip(array,-limit,limit)+limit)/(2*limit)
            im=Image.fromarray(np.uint8(np.round(display*255))).convert('RGB')
            canvas.paste(im.resize((256,256),Image.Resampling.NEAREST),(20+col*272,y+25))
            draw.text((20+col*272,y+285),label,font=font,fill='#111827')
    draw.text((20,h-33),footnote,font=ImageFont.truetype(FONT,13),fill='#374151')
    canvas.save(path)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    manifest=dict(seed=721,selection='Fixed random ImageNet holdout images; evenly spaced nonzero test seismic tiles. No selection by scores.',models={},examples=[])
    for dataset,budget,sigmas in [('imagenet','b192_e20',(25/255,50/255)),('marmousi2','b128_e30',(.2,.4))]:
        rgb=dataset=='imagenet'
        nets={}
        for method in ['idr','supervised']:
            nets[method],path=net_for(dataset,method,budget)
            manifest['models'][dataset+'/'+method]=path
        samples=[]
        if rgb:
            paths=json.loads((ROOT/'data/imagenet_split.json').read_text())['validation']
            indices=np.random.RandomState(721).choice(len(paths),3,replace=False)
            for index in indices:
                path=Path(paths[index]);im=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
                h,w=im.shape[:2]
                if h<=256 or w<=256: crop=cv2.resize(im.astype(float),(256,256))
                else:
                    y,x=(h-256)//2,(w-256)//2;crop=im[y:y+256,x:x+256].astype(float)
                clean=(cv2.resize(crop,(128,128),interpolation=cv2.INTER_AREA)/255).astype(np.float32)
                samples.append((path.name,clean))
        else:
            tiles=[tile for tile in heldout_tiles(ROOT/'data/marmousi2','test') if np.mean(tile[2]**2)>1e-6]
            for index in np.linspace(0,len(tiles)-1,3,dtype=int):
                y,x,clean=tiles[index];samples.append((f'Test tile: time {y}, trace {x}',clean.copy()))
        for sigma in sigmas:
            rng=np.random.RandomState(721);rows=[]
            tag=str(round(sigma*255)) if rgb else str(sigma)
            for i,(source,clean) in enumerate(samples):
                noisy=(clean+rng.normal(0,sigma,clean.shape)).astype(np.float32)
                idr=predict(nets['idr'],noisy,rgb);supervised=predict(nets['supervised'],noisy,rgb)
                arrays=[clean,noisy,idr,supervised]
                scores={}
                for name,array in zip(['noisy','idr','supervised'],arrays[1:]):
                    error=float(np.mean((array.astype(float)-clean)**2))
                    scores[name]=10*np.log10((1 if rgb else float(np.mean(clean.astype(float)**2)))/max(error,1e-30))
                metric='PSNR' if rgb else 'SNR'
                rows.append((arrays,['Reference']+[f'{metric}: {scores[k]:.2f} dB' for k in ['noisy','idr','supervised']],source))
                stem=f'{dataset}_sigma{tag}_example{i+1}'
                np.savez_compressed(OUT/f'{stem}.npz',reference=clean,noisy=noisy,idr=idr,supervised=supervised)
                manifest['examples'].append(dict(dataset=dataset,source=source,sigma=sigma,metric=metric,scores_db=scores,arrays=stem+'.npz'))
            title=f"{'ImageNet validation holdout' if rgb else 'Marmousi2 held-out test region'} | Gaussian sigma {tag}{'/255' if rgb else ' (normalized)'} | 128 x 128"
            footer='Identical noisy input for both models. Scores use unclipped float arrays; display only is clipped.' if rgb else 'Identical noisy input; same amplitude display scale across each row. Scores use unclipped float arrays.'
            grid(rows,title,footer,OUT/f'{dataset}_sigma{tag}.png',rgb)
        del nets
        torch.cuda.empty_cache()
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Saved four comparison grids, raw arrays and provenance to {OUT}')


if __name__=='__main__':main()
