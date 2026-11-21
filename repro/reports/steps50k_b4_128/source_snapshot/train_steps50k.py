"""Separate fixed-update protocol: 128px, batch 4, 50,000 optimizer updates."""
import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback
from datetime import datetime, timezone

import gaussian  # Establishes the reproduction root before loading upstream configs.
import numpy as np
import torch
from torch.utils.data import DataLoader
from configs.synthetic_config import cfgs
from dataset.imagefolder import Imagefolder
from imagenet_split import split_paths
from seismic_data import SeismicPatches
from validation import validate

ROOT = Path(__file__).resolve().parent
PROTOCOL = 'steps50k_b4_128'


def create_dataset(name, data, validation_count=1000, seed=2026, smoke=False):
    if name == 'imagenet':
        cfg = copy.deepcopy(cfgs['idr-g'].dataset.train)
        cfg.update(path=str(Path(data).resolve()), crop_size=256, output_size=128, mc=False)
        with contextlib.redirect_stdout(io.StringIO()):
            dataset = Imagefolder(**cfg)
        if not smoke and len(dataset) != 50000:
            raise ValueError(f'Expected 50000 ImageNet source images; got {len(dataset)}')
        train, heldout, fingerprint = split_paths(dataset.metas, validation_count, seed)
        dataset.metas, dataset.num = train, len(train)
        return dataset, heldout, fingerprint
    dataset = SeismicPatches(data, samples=50000 if not smoke else 8)
    fingerprint = hashlib.sha256(json.dumps(dataset.metadata, sort_keys=True).encode()).hexdigest()
    return dataset, str(Path(data).resolve()), fingerprint


def make_model(name, method, device):
    model = gaussian.make_model(method, 'compact', channels=3 if name == 'imagenet' else 1,
                                zero_mean=name == 'imagenet', noise_max=50 if name == 'imagenet' else 127.5,
                                device=device)
    model.scheduler = torch.optim.lr_scheduler.MultiStepLR(model.optimizer, [25000, 40000], gamma=.5)
    return model


def configure_torch():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def train(a):
    if min(a.steps, a.validate_every, a.save_every, a.log_every, a.validation_count) < 1 or a.workers < 1:
        raise ValueError('Steps, intervals, validation count and workers must be positive')
    if not a.smoke_test and (a.steps != 50000 or a.validation_count != 1000):
        raise ValueError('Production protocol requires exactly 50000 steps and 1000 ImageNet holdout images')
    configure_torch()
    gaussian.seed_all(a.seed)
    dataset, heldout, fingerprint = create_dataset(a.dataset, a.data, a.validation_count, a.seed, a.smoke_test)
    if len(dataset) < 4:
        raise ValueError('Need at least four training samples')
    steps_per_epoch = math.ceil(len(dataset)/4)
    metadata = dict(vars(a), protocol=PROTOCOL, batch=4, output_size=128,
                    training_samples=len(dataset), steps_per_epoch=steps_per_epoch,
                    expected_sample_presentations=a.steps*4 if not a.smoke_test else None,
                    epochs_equivalent=a.steps/steps_per_epoch, split_fingerprint=fingerprint,
                    lr=3e-4, milestones=[25000,40000], early_stopping=False,
                    teacher_refresh='once per data-loader epoch, starting at epoch 2',
                    noise_model='AWGN; one sigma per batch and noise-generation call',
                    noise_range=[0,50/255] if a.dataset=='imagenet' else [0,.5],
                    precision='FP32; TF32 disabled')
    if a.dry_run:
        print(json.dumps(metadata, indent=2)); return
    out = Path(a.output)
    if (out/'latest.pt').exists() and not a.resume:
        raise ValueError('Output already has a checkpoint: use --resume or a new output')
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use an appropriate NVIDIA driver and CUDA-enabled PyTorch')
    model = make_model(a.dataset, a.method, a.device)
    step = epoch = offset = last_validation_step = 0
    best_loss, best_step, saved_rng = math.inf, 0, None
    if a.resume:
        state = torch.load(out/'latest.pt', map_location='cpu', weights_only=False)
        for key in ('protocol','dataset','method','steps','batch','seed','workers','output_size',
                    'split_fingerprint','validate_every','device','smoke_test'):
            if state['metadata'][key] != metadata[key]:
                raise ValueError(f'Resume mismatch: {key}')
        model.net.load_state_dict(state['net'])
        model.optimizer.load_state_dict(state['optimizer'])
        model.scheduler.load_state_dict(state['scheduler'])
        step, epoch, offset = state['step'], state['epoch'], state['offset']
        last_validation_step, best_loss, best_step = state['last_validation_step'], state['best_loss'], state['best_step']
        if a.method == 'idr':
            model.net_copy.load_state_dict(state['teacher']); model.net_copy_epoch = state['teacher_epoch']
        saved_rng = state['rng'] if offset else None
    metadata.update(torch=torch.__version__, gpu=torch.cuda.get_device_name() if a.device=='cuda' else 'CPU')
    (out/'config.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (out/'manifest.txt').write_text('\n'.join(dataset.metas)+'\n')
    if a.dataset == 'imagenet':
        (out/'validation_manifest.txt').write_text('\n'.join(heldout)+'\n')

    def save(best=False):
        state = dict(metadata=metadata, net=model.net.state_dict(), optimizer=model.optimizer.state_dict(),
                     scheduler=model.scheduler.state_dict(), step=step, epoch=epoch, offset=offset,
                     best_loss=best_loss, best_step=best_step, last_validation_step=last_validation_step,
                     rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),
                          torch.cuda.get_rng_state_all() if a.device=='cuda' else []))
        if a.method=='idr':
            state.update(teacher=model.net_copy.state_dict(), teacher_epoch=model.net_copy_epoch)
        def atomic(obj, name):
            temp=out/(name+'.tmp');torch.save(obj,temp);temp.replace(out/name)
        if best:
            atomic(state,'best.pt');atomic(model.net.state_dict(),'best_weights.pth')
        atomic(model.net.state_dict(),'weights.pth');atomic(state,'latest.pt')

    def validation_if_due():
        nonlocal best_loss, best_step, last_validation_step
        if step == 0 or last_validation_step >= step or not (step % a.validate_every == 0 or step == a.steps):
            return
        started=time.perf_counter()
        result=validate(model.net,a.dataset,heldout,a.device,a.seed)
        improved=result['loss'] < best_loss
        if improved: best_loss,best_step=result['loss'],step
        last_validation_step=step
        record=dict(step=step, epoch=epoch, **result, best_loss=best_loss, best_step=best_step,
                    seconds=time.perf_counter()-started, timestamp_utc=datetime.now(timezone.utc).isoformat())
        with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(f"  [validation] step {step:,} | L1 {result['loss']:.6f} | best {best_loss:.6f} at {best_step:,}",flush=True)
        save(best=improved)

    print(f'\n{a.dataset} | {a.method} | batch 4 | 128x128 | {a.steps:,} optimizer steps | start {step:,}',flush=True)
    print(f'{len(dataset):,} samples/epoch | {steps_per_epoch:,} updates/epoch | LR 3e-4, halves at 25k/40k | no early stopping',flush=True)
    print('       Step       Epoch        L1 loss          LR       step/s       ETA (h)',flush=True)
    validation_if_due()  # Recover a checkpoint saved just before scheduled validation.
    start_step=step;started=time.perf_counter()
    while step < a.steps:
        gaussian.seed_all(a.seed+epoch)
        generator=torch.Generator().manual_seed(a.seed+epoch)
        loader=DataLoader(dataset,batch_size=4,shuffle=True,drop_last=False,num_workers=a.workers,
                          generator=generator,pin_memory=a.device=='cuda')
        model.epoch=epoch
        for index,data in enumerate(loader):
            if index < offset:continue
            if saved_rng:
                random.setstate(saved_rng[0]);np.random.set_state(saved_rng[1]);torch.set_rng_state(saved_rng[2])
                if a.device=='cuda':torch.cuda.set_rng_state_all(saved_rng[3])
                saved_rng=None
            model.preprocess({'gt':data['gt'].to(a.device,non_blocking=True)})
            lr=model.optimizer.param_groups[0]['lr'];loss=model.optimize_parameters();model.scheduler.step()
            loss=float(loss.item())
            if not math.isfinite(loss):raise RuntimeError('Nonfinite training loss; previous checkpoint retained')
            step+=1;offset=index+1
            record_epoch=epoch
            if offset==len(loader):epoch+=1;offset=0
            elapsed=time.perf_counter()-started;rate=(step-start_step)/max(elapsed,1e-9)
            record=dict(step=step,total_steps=a.steps,epoch=record_epoch,actual_batch_size=len(data['gt']),loss=loss,
                        learning_rate=lr,elapsed_seconds=elapsed,steps_per_second=rate,
                        timestamp_utc=datetime.now(timezone.utc).isoformat())
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if step==start_step+1 or step%a.log_every==0 or step==a.steps:
                print(f'{step:>7,}/{a.steps:<7,} {record_epoch+1:>6} {loss:>14.6f} {lr:>11.2e} {rate:>12.2f} {(a.steps-step)/rate/3600:>13.2f}',flush=True)
            if step%a.save_every==0 or step==a.steps:save()
            validation_if_due()
            if step==a.steps:break
    (out/'training_summary.json').write_text(json.dumps(dict(completed_steps=step,completed_epochs=epoch,
        partial_epoch_batches=offset,best_loss=best_loss,best_step=best_step,early_stopping=False,
        final_weights=str(out/'weights.pth'),best_weights=str(out/'best_weights.pth')),indent=2)+'\n')
    print(f'Finished exactly {step:,} updates. Results: {out}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['imagenet','marmousi2'],required=True)
    p.add_argument('--method',choices=['idr','supervised'],required=True)
    p.add_argument('--data',required=True);p.add_argument('--output')
    p.add_argument('--steps',type=int,default=50000);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--validation-count',type=int,default=1000)
    p.add_argument('--validate-every',type=int,default=2500);p.add_argument('--save-every',type=int,default=1000)
    p.add_argument('--log-every',type=int,default=250);p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--smoke-test',action='store_true',help='Permit tiny local correctness tests; never scientific results')
    a=p.parse_args();a.output=a.output or str(Path('runs')/PROTOCOL/a.dataset/a.method)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with (out/'console.log').open('a',buffering=1) as f:
        with contextlib.redirect_stdout(gaussian.Tee(sys.stdout,f)),contextlib.redirect_stderr(gaussian.Tee(sys.stderr,f)):
            print('\nInvocation:',sys.argv,flush=True)
            try:
                train(a)
            except Exception:
                traceback.print_exc()
                return 1
    return 0


if __name__=='__main__':sys.exit(main())
