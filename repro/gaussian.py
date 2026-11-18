"""Single-GPU driver; uses the original IDR model, loss and target generation."""
import argparse
import contextlib
import io
from datetime import datetime, timezone
import traceback
import uuid
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
import numpy as np
import torch
from torch.utils.data import DataLoader
from configs.synthetic_config import cfgs
from dataset.imagefolder import Imagefolder
from training_display import TrainingDisplay
from epoch_protocol import plan
from validation import EarlyStopping, validate
from imagenet_split import split_paths
from seismic_data import SeismicPatches
from experiment_profiles import PROFILES, default_output, validate_resume, make_scheduler
from trainer.base_trainer import DenoiseBase_fastIDR, DenoiseBase_n2c


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DriverMixin:
    def model_to_device(self, net):
        if self.cfg.get('device', 'cuda') == 'cpu':
            return net.cpu()
        return super().model_to_device(net)

    def print_iter_info(self, *args, **kwargs):
        pass  # Driver owns logging and checkpointing (avoids upstream off-by-one saves).


class IDR(DriverMixin, DenoiseBase_fastIDR):
    def preprocess(self, data):
        if self.epoch >= 1 and self.net_copy_epoch < self.epoch:
            # Replace only the upstream status message; target computation is unchanged.
            with contextlib.redirect_stdout(io.StringIO()):
                super().preprocess(data)
            print(f"  [refine] Updated target network for epoch {self.epoch + 1}", flush=True)
        else:
            super().preprocess(data)


class Supervised(DriverMixin, DenoiseBase_n2c):
    pass


def make_model(method, profile='paper', channels=3, zero_mean=True, noise_max=50, device='cuda'):
    protocol = PROFILES[profile]
    cfg = copy.deepcopy(cfgs['idr-g' if method == 'idr' else 'n2c-g'])
    cfg.dist = False
    cfg.rank = 0
    cfg.resume = False
    cfg.total_iter = protocol['steps']
    cfg.dataset.train.mc = False
    cfg.device = device
    cfg.model_args = {'in_channels': channels, 'out_channels': channels}
    cfg.zero_mean = zero_mean
    cfg.noise_level = [0, noise_max]
    model = (IDR if method == 'idr' else Supervised)(cfg)
    # Match the paper rather than the released 500k/cosine config.
    model.scheduler = make_scheduler(model.optimizer, profile)
    return model


def probe(batch, steps, profile='paper'):
    seed_all(2026)
    model = make_model('idr', profile)  # Includes frozen target network and refinement forward pass.
    model.epoch = 1
    size = PROFILES[profile]['output_size']
    clean = torch.rand(batch, size, size, 3, device='cuda')
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(steps):
        model.preprocess({'gt': clean})
        model.optimize_parameters()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    reserved = torch.cuda.memory_reserved()
    peak_allocated = torch.cuda.max_memory_allocated()
    # Cached allocator blocks are reusable by subsequent batches. Driver-free
    # memory alone can reject a fitting batch merely because the cache grew.
    peak_headroom = free + reserved - peak_allocated
    result = dict(batch=batch, profile=profile, output_size=size, seconds_per_step=(time.perf_counter()-start)/steps,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(), free_bytes=free,
                  peak_allocated_bytes=peak_allocated, peak_headroom_bytes=peak_headroom,
                  total_bytes=total, gpu=torch.cuda.get_device_name())
    print(json.dumps(result), flush=True)
    # Leave space for transfers, data batches and transient desktop usage.
    return 0 if peak_headroom >= 512 * 1024**2 else 2


def tune(output, profile='paper'):
    def fits(batch):
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'probe',
                               '--batch', str(batch), '--steps', '3', '--profile', profile], capture_output=True, text=True)
        if proc.returncode == 0:
            result = json.loads(proc.stdout.strip().splitlines()[-1])
            print(f"batch={batch}: fits, {result['seconds_per_step']:.3f}s/step", flush=True)
            return result
        if proc.returncode == 2 or 'out of memory' in proc.stderr.lower():
            print(f'batch={batch}: insufficient memory', flush=True)
            return None
        raise RuntimeError(proc.stdout + proc.stderr)
    best = None
    lo, hi = 0, 1
    while hi <= 256:
        result = fits(hi)
        if result is None:
            break
        best, lo, hi = result, hi, hi * 2
    if best is None:
        raise RuntimeError('Even batch size 1 does not fit; free GPU memory before retrying.')
    if hi > 256:
        raise RuntimeError('Search cap reached; increase cap before claiming maximum batch size.')
    while hi - lo > 1:
        mid = (lo + hi)//2
        result = fits(mid)
        if result:
            best, lo = result, mid
        else:
            hi = mid
    best['note'] = 'Largest probed FP32 batch with >=512 MiB free; retune if GPU usage changes.'
    best['torch'] = torch.__version__
    Path(output).write_text(json.dumps(best, indent=2)+'\n')
    print(f'Saved {output}: batch={best["batch"]}', flush=True)


def train(args):
    if args.workers < 1 or args.steps < 1 or args.save_every < 1 or args.log_every < 1:
        raise ValueError('workers, steps, save-every and log-every must be positive.')
    protocol = PROFILES[args.profile]
    if args.validate and args.epochs is None:
        raise ValueError('Epoch validation requires --epochs')
    if args.patience < 1 or args.min_delta < 0:
        raise ValueError('patience must be positive and min-delta nonnegative')
    if args.validate and args.profile != 'compact':
        raise ValueError('Validation/early stopping is configured for the compact 128 profile.')
    if args.epochs is not None and args.epochs < 1:
        raise ValueError('epochs must be positive')
    if args.dataset == 'marmousi2' and (args.profile != 'compact' or args.epochs is None):
        raise ValueError('Marmousi2 requires --profile compact --epochs N')
    if args.epochs is None and not args.allow_small_data and args.steps != protocol['steps']:
        raise ValueError(f"Profile {args.profile} requires {protocol['steps']:,} iterations.")
    if args.batch == 'auto':
        if not Path(args.tuning).exists():
            tune(args.tuning, args.profile)
        tuning = json.loads(Path(args.tuning).read_text())
        if tuning.get('profile', 'paper') != args.profile:
            raise ValueError('Batch tuning belongs to a different resolution/profile; retune.')
        batch = tuning['batch']
    else:
        batch = int(args.batch)
    seed_all(args.seed)
    cfg = copy.deepcopy(cfgs['idr-g'].dataset.train)
    cfg.crop_size = protocol['crop_size']
    cfg.output_size = protocol['output_size']
    cfg.path, cfg.mc, cfg.worker_num = str(Path(args.data).resolve()), False, args.workers
    with contextlib.redirect_stdout(io.StringIO()):
        dataset = (Imagefolder(**cfg) if args.dataset == 'imagenet' else
                   SeismicPatches(args.data, samples=args.samples_per_epoch))
    if args.dataset == 'imagenet' and len(dataset) != 50000 and not args.allow_small_data:
        raise ValueError(f'Expected 50,000 ImageNet validation images, found {len(dataset)}. '
                         'Use --allow-small-data only for explicit smoke tests.')
    validation_paths = None
    split_fingerprint = None
    if args.validate and args.dataset == 'imagenet':
        train_paths, validation_paths, split_fingerprint = split_paths(dataset.metas, args.validation_count, args.split_seed)
        dataset.metas = train_paths
        dataset.num = len(train_paths)
    if batch < 1 or batch > len(dataset):
        raise ValueError('Batch must be between 1 and the dataset size.')
    budget = plan(len(dataset), batch, args.epochs) if args.epochs else None
    if budget:
        args.steps = budget['steps']
    milestones = budget['milestones'] if budget else protocol['milestones']
    if args.lr_milestones is not None:
        milestones = args.lr_milestones
        if any(value < 1 for value in milestones):
            raise ValueError('LR milestones must be positive optimizer steps')
        if budget:
            budget['milestones'] = milestones
    if args.dry_run:
        print(json.dumps(dict(dataset=args.dataset, method=args.method, output=args.output,
                             budget=budget or dict(steps=args.steps, batch=batch),
                             milestones=milestones, validation_images=len(validation_paths) if validation_paths else None, output_size=protocol['output_size']), indent=2))
        return
    out = Path(args.output or default_output(args.method, args.profile))
    out.mkdir(parents=True, exist_ok=True)
    if (out/'latest.pt').exists() and not args.resume:
        raise ValueError('Checkpoint already exists. Use --resume or another --output.')
    manifest = '\n'.join(dataset.metas)+'\n'
    digest = hashlib.sha256(manifest.encode()).hexdigest()
    metadata = dict(vars(args), batch=batch, images=len(dataset), manifest_sha256=digest,
                    torch=torch.__version__, gpu=torch.cuda.get_device_name() if args.device == 'cuda' else 'CPU',
                    precision='FP32 (TF32 disabled)', crop=128 if args.dataset == 'marmousi2' else protocol['crop_size'],
                    output_size=protocol['output_size'], learning_rate=3e-4,
                    milestones=milestones, budget=budget, split_fingerprint=split_fingerprint,
                    validation_images=len(validation_paths) if validation_paths else None, channels=1 if args.dataset == 'marmousi2' else 3,
                    noise_max=0.5 if args.dataset == 'marmousi2' else 50/255, loss='L1', upstream='7061caa7360008135e7f732f82b94f7b230e59b1')
    model = make_model(args.method, args.profile, channels=metadata['channels'],
                       zero_mean=args.dataset == 'imagenet',
                       noise_max=127.5 if args.dataset == 'marmousi2' else 50, device=args.device)
    model.scheduler = torch.optim.lr_scheduler.MultiStepLR(model.optimizer, milestones, gamma=0.5)
    step, epoch, offset = 0, 0, 0
    saved_rng = None
    stopping = EarlyStopping(args.patience, args.min_delta)
    if args.resume:
        # Only load checkpoints produced by this local script, never untrusted files.
        state = torch.load(out/'latest.pt', map_location='cpu', weights_only=False)
        validate_resume(state['metadata'], metadata)
        model.net.load_state_dict(state['net'])
        model.optimizer.load_state_dict(state['optimizer'])
        model.scheduler.load_state_dict(state['scheduler'])
        step, epoch, offset = state['step'], state['epoch'], state['offset']
        if args.method == 'idr':
            model.net_copy.load_state_dict(state['teacher'])
            model.net_copy_epoch = state['teacher_epoch']
        # A fresh epoch reseeds identically in both uninterrupted and resumed runs.
        # Restore mid-epoch RNG only after replaying the already consumed batches.
        saved_rng = state['rng'] if offset > 0 else None
        if args.validate:
            if 'early_stopping' not in state:
                raise ValueError('Checkpoint has no validation state; use a new output directory.')
            stopping = EarlyStopping(**state['early_stopping'])
    metadata['effective_start_lr'] = model.optimizer.param_groups[0]['lr']
    (out/'manifest.txt').write_text(manifest)
    if validation_paths:
        (out/'validation_manifest.txt').write_text('\n'.join(validation_paths)+'\n')
    (out/'config.json').write_text(json.dumps(metadata, indent=2)+'\n')

    def save(best=False):
        state = dict(net=model.net.state_dict(), optimizer=model.optimizer.state_dict(),
                     scheduler=model.scheduler.state_dict(), step=step, epoch=epoch, offset=offset,
                     metadata=metadata, rng=(random.getstate(), np.random.get_state(),
                     torch.get_rng_state(), torch.cuda.get_rng_state_all() if args.device == 'cuda' else []))
        if args.validate:
            state['early_stopping'] = stopping.state_dict()
        if args.method == 'idr':
            state.update(teacher=model.net_copy.state_dict(), teacher_epoch=model.net_copy_epoch)
        # Publish best artifacts before latest state can reference the new best.
        if best:
            torch.save(state, out/'best.tmp')
            (out/'best.tmp').replace(out/'best.pt')
            torch.save(model.net.state_dict(), out/'best_weights.tmp')
            (out/'best_weights.tmp').replace(out/'best_weights.pth')
        torch.save(model.net.state_dict(), out/'weights.tmp')
        (out/'weights.tmp').replace(out/'weights.pth')
        torch.save(state, out/'latest.tmp')
        (out/'latest.tmp').replace(out/'latest.pt')

    display = None
    if args.console_format == 'json':
        print(json.dumps(dict(event='training_start', config=metadata, resumed_step=step)), flush=True)
    else:
        display = TrainingDisplay(metadata, step)
    session = uuid.uuid4().hex
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    start_step = step
    processed_samples = 0
    while not stopping.stopped and (step < args.steps or (args.validate and offset > 0 and epoch < args.epochs)):
        # Epoch-specific loader seeds permit deterministic reconstruction on resume.
        generator = torch.Generator().manual_seed(args.seed + epoch)
        seed_all(args.seed + epoch)
        loader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=args.epochs is None,
                            num_workers=args.workers, pin_memory=args.device == 'cuda', generator=generator)
        model.epoch = epoch
        for index, data in enumerate(loader):
            if index < offset:
                continue
            if saved_rng:
                random.setstate(saved_rng[0]); np.random.set_state(saved_rng[1])
                torch.set_rng_state(saved_rng[2])
                if args.device == 'cuda':
                    torch.cuda.set_rng_state_all(saved_rng[3])
                saved_rng = None
            model.preprocess({'gt': data['gt'].to(args.device, non_blocking=True)})
            learning_rate = model.optimizer.param_groups[0]['lr']
            loss = model.optimize_parameters()
            model.scheduler.step()
            step += 1
            offset = index + 1
            processed_samples += len(data['gt'])
            loss_value = loss.item()  # Synchronizes the completed step for timing/logging.
            seconds = (time.perf_counter()-started)/(step-start_step)
            record = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), session=session,
                          method=args.method, step=step, total_steps=args.steps, epoch=epoch,
                          batch_size=batch, loss=loss_value, learning_rate=learning_rate,
                          next_learning_rate=model.optimizer.param_groups[0]['lr'],
                          elapsed_seconds=time.perf_counter()-started, seconds_per_step=seconds,
                          images_per_second=processed_samples/max(time.perf_counter()-started,1e-9),
                          actual_batch_size=len(data['gt']),
                          estimated_remaining_hours=seconds*(args.steps-step)/3600,
                          gpu_allocated_mib=torch.cuda.memory_allocated()/1024**2 if args.device == 'cuda' else 0,
                          gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if args.device == 'cuda' else 0)
            with (out/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            if step == start_step + 1 or step % args.log_every == 0 or step == args.steps:
                if display:
                    display.progress(record)
                else:
                    print(json.dumps(record), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                save()
                if display:
                    print(f"  [saved] Step {step:,} -> {out / 'latest.pt'}", flush=True)
                else:
                    print(json.dumps(dict(event='checkpoint_saved', step=step, path=str(out/'latest.pt'))), flush=True)
            if step == args.steps:
                break
        if offset == len(loader):
            epoch += 1
            offset = 0
            if args.validate and stopping.last_epoch < epoch:
                val_started = time.perf_counter()
                result = validate(model.net, args.dataset,
                                  validation_paths if args.dataset == 'imagenet' else args.data,
                                  args.device, args.validation_seed)
                best = stopping.update(result['loss'], epoch)
                val_record = dict(epoch=epoch, step=step, session=session,
                                  timestamp_utc=datetime.now(timezone.utc).isoformat(), **result,
                                  seconds=time.perf_counter()-val_started,
                                  **stopping.state_dict())
                with (out/'validation.jsonl').open('a') as f:
                    f.write(json.dumps(val_record)+'\n')
                print(f"  [validation] Epoch {epoch} | L1 {result['loss']:.6f} | "
                      f"best {stopping.best_loss:.6f} (epoch {stopping.best_epoch}) | "
                      f"patience {stopping.bad_epochs}/{stopping.patience}" +
                      (' | NEW BEST' if best else ''), flush=True)
                save(best=best)  # Canonical epoch boundary, including stopping/teacher/RNG state.
                if stopping.stopped:
                    print(f"  [early stop] No significant validation improvement for {stopping.patience} epochs.", flush=True)
    if args.validate:
        summary = dict(completed_steps=step, completed_epochs=epoch,
                       maximum_epochs=args.epochs, **stopping.state_dict(),
                       evaluation_weights=str(out/'best_weights.pth'))
        (out/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'Finished {step} steps. Checkpoint: {out / "latest.pt"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('probe'); p.add_argument('--batch', type=int, required=True); p.add_argument('--steps', type=int, default=3)
    p = subs.add_parser('tune'); p.add_argument('--output')
    p = subs.add_parser('train')
    p.add_argument('--method', choices=['idr', 'supervised'], required=True)
    p.add_argument('--epochs', type=int, help='Fixed full epochs; overrides profile step budget and scales LR milestones')
    p.add_argument('--lr-milestones', nargs='*', type=int, help='Explicit absolute optimizer-step LR milestones; continuation preserves the original schedule')
    p.add_argument('--dataset', choices=['imagenet', 'marmousi2'], default='imagenet')
    p.add_argument('--samples-per-epoch', type=int, default=50000, help='Random seismic patch draws per epoch')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--validate', action='store_true', help='Validate each epoch and enable early stopping')
    p.add_argument('--validation-count', type=int, default=1000, help='Held-out ImageNet images')
    p.add_argument('--split-seed', type=int, default=2026)
    p.add_argument('--validation-seed', type=int, default=2026)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--min-delta', type=float, default=1e-5)
    p.add_argument('--data', required=True); p.add_argument('--batch', default='auto')
    p.add_argument('--tuning'); p.add_argument('--steps', type=int)
    p.add_argument('--workers', type=int, default=4); p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--output'); p.add_argument('--resume', action='store_true')
    p.add_argument('--console-format', choices=['table', 'json'], default='table')
    p.add_argument('--log-every', type=int, default=100, help='Console frequency; metrics.jsonl records every step')
    p.add_argument('--save-every', type=int, default=1000); p.add_argument('--allow-small-data', action='store_true')
    for subparser in subs.choices.values():
        subparser.add_argument('--profile', choices=tuple(PROFILES), default='paper')
    args = parser.parse_args()
    batch_file = 'batch.json' if args.profile == 'paper' else 'batch_compact128.json'
    if args.command == 'tune' and args.output is None:
        args.output = batch_file
    if args.command == 'train':
        if args.epochs is not None and args.steps is not None:
            parser.error('Use --epochs OR --steps, not both.')
        args.steps = args.steps if args.steps is not None else PROFILES[args.profile]['steps']
        args.tuning = args.tuning or batch_file
        args.output = args.output or (f'runs/epoch128/{args.dataset}/{args.method}/b{args.batch}_e{args.epochs}'
                                     if args.epochs else default_output(args.method, args.profile))
    if args.command == 'train':
        out = Path(args.output or default_output(args.method, args.profile))
        out.mkdir(parents=True, exist_ok=True)
        with (out/'console.log').open('a', buffering=1) as logfile:
            with contextlib.redirect_stdout(Tee(sys.stdout, logfile)), contextlib.redirect_stderr(Tee(sys.stderr, logfile)):
                print(f"\nInvocation UTC {datetime.now(timezone.utc).isoformat()}: {sys.argv}", flush=True)
                try:
                    return execute(args)
                except BaseException:
                    traceback.print_exc()
                    return 1
    return execute(args)


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile
    def write(self, text):
        self.terminal.write(text)
        self.logfile.write(text)
        return len(text)
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def execute(args):
    if not (getattr(args, 'dry_run', False) or getattr(args, 'device', 'cuda') == 'cpu') and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Check driver/GPU access; run outside the sandbox if necessary.')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == 'probe':
        try:
            return probe(args.batch, args.steps, args.profile)
        except torch.cuda.OutOfMemoryError:
            return 2
    if args.command == 'tune':
        tune(args.output, args.profile)
    else:
        train(args)
    return 0

if __name__ == '__main__':
    sys.exit(main())
