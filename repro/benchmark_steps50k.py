"""Measure real batch-4 data loading/training; estimate fixed-update run times."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_steps50k import ROOT,create_dataset,make_model,configure_torch
from gaussian import seed_all
from validation import validate


def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=200)
    p.add_argument('--warmup',type=int,default=20);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--output',default='reports/steps50k_b4_128/timing.json');a=p.parse_args()
    configure_torch();results=[]
    for name,data in [('imagenet','data/imagenet_val'),('marmousi2','data/marmousi2')]:
        dataset,heldout,_=create_dataset(name,data)
        for method in ['idr','supervised']:
            seed_all(2026);model=make_model(name,method,'cuda')
            model.epoch=1  # Include teacher forward pass; conservative for IDR's first epoch.
            loader=iter(DataLoader(dataset,batch_size=4,shuffle=True,num_workers=a.workers,pin_memory=True))
            torch.cuda.reset_peak_memory_stats()
            for step in range(a.warmup+a.steps):
                if step==a.warmup:
                    torch.cuda.synchronize();started=time.perf_counter()
                data=next(loader)
                model.preprocess({'gt':data['gt'].cuda(non_blocking=True)})
                loss=model.optimize_parameters();model.scheduler.step()
                loss.item()
            torch.cuda.synchronize();seconds=(time.perf_counter()-started)/a.steps
            started=time.perf_counter();val=validate(model.net,name,heldout,'cuda')
            torch.cuda.synchronize();val_seconds=time.perf_counter()-started
            row=dict(dataset=name,method=method,batch=4,output_size=128,measured_steps=a.steps,
                     seconds_per_step=seconds,validation_seconds=val_seconds,
                     estimated_training_hours=seconds*50000/3600,
                     estimated_training_validation_hours=(seconds*50000+20*val_seconds)/3600,
                     peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
            results.append(row);print(json.dumps(row),flush=True)
            del model,loader,loss,data;gc.collect();torch.cuda.empty_cache()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,workers=a.workers,
        note='Real loader; FP32; teacher active for IDR; estimates exclude setup/downloads and final evaluations. Benchmark models are discarded.',results=results),indent=2)+'\n')


if __name__=='__main__':main()
