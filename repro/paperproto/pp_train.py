"""Protocol-P trainer (paper Sec. 5.1): batch 4, 50,000 Adam updates, LR 3e-4 halved
at 25k/40k, L1 (masked L2 for binomial-type noise), 256x256 crops, final checkpoint.

Methods reproduce the released trainers' `preprocess` logic:
  idr  : DenoiseBase_fastIDR  (target = noisy in epoch 0, frozen teacher(noisy) after;
                               teacher refreshed from the student at each epoch start)
  sup  : DenoiseBase_n2c      (input = noisy, target = clean)
  n2n  : DenoiseBase_n2n      (input = second independent noisy copy, target = noisy)
  nr2n : idr without refinement (Noisier2Noise-style target = noisy throughout)
"""
import argparse, contextlib, copy, json, math, os, random, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "paperproto"), str(ROOT)]
os.chdir(ROOT)
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.unet import UNet_n2n_un
import noise as N
from pp_data import ImageNetTrain, SeisTrain

STEPS_PER_EPOCH = 12500  # 50,000 samples / batch 4 (ImageNet); seismic epochs use 50,000 crops


def seed_all(s):
    random.seed(s); np.random.seed(s % 2**32); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def configure():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tf32 = os.environ.get('PP_FP32') != '1'   # TF32 default (27% faster); PP_FP32=1 for the FP32 control
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32


class Tee:
    def __init__(self, *s): self.s = s
    def write(self, d):
        for x in self.s: x.write(d)
    def flush(self):
        for x in self.s: x.flush()


def build(a):
    if a.domain == 'image':
        ds = ImageNetTrain(a.data, gray=a.gray, fixed_noise=a.fixed is not None, seed=a.seed)
        ch = 1 if a.gray else 3
    else:
        ds = SeisTrain(a.keys.split(','), samples=50000, fixed=a.fixed)
        ch = 1
    return ds, ch


class Model:
    def __init__(self, a, ch, device):
        self.a, self.device = a, device
        self.net = UNet_n2n_un(ch, ch).to(device)
        self.teacher = UNet_n2n_un(ch, ch).to(device).eval() if a.method == 'idr' else None
        self.teacher_epoch = 0
        self.opt = torch.optim.Adam(self.net.parameters(), lr=3e-4, weight_decay=0)
        if a.schedule == 'paper':
            self.sched = torch.optim.lr_scheduler.MultiStepLR(self.opt, [25000, 40000], gamma=.5)
        else:
            self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=a.steps)
        self.zero_mean = a.domain == 'image' and a.noise == 'g'

    def noisy(self, x):
        if self.a.level:   # field runs: calibrated sigma range instead of the default training range
            return N.add_seis_noise(x, self.a.noise, self.a.dt, level=float(np.random.uniform(*self.a.level)))
        return N.add_noise(x, self.a.noise, self.a.dt)

    def targets(self, clean, given_noisy, epoch):
        """Returns input, target, target_mask for one batch (upstream order of RNG use)."""
        if given_noisy is None:
            noisy, nmask = self.noisy(clean)
        else:
            noisy, nmask = given_noisy, None
        if self.zero_mean:
            clean, noisy = clean - .5, noisy - .5
        m = self.a.method
        if m == 'sup':
            return noisy, clean, None
        if m == 'n2n':
            inp, _ = self.noisy(clean)  # upstream adds the second noise to the (shifted) clean
            return inp, noisy, nmask
        target, tmask = noisy, nmask
        if m == 'idr' and epoch >= 1:
            if self.teacher_epoch < epoch:
                self.teacher_epoch = epoch
                self.teacher.load_state_dict(self.net.state_dict())
                print(f'  [refine] teacher <- student at start of epoch {epoch + 1}', flush=True)
            with torch.no_grad():
                target = self.teacher(noisy)
            tmask = None
        inp, _ = self.noisy(target)
        return inp, target, tmask

    def step(self, clean, given_noisy, epoch):
        inp, target, tmask = self.targets(clean, given_noisy, epoch)
        self.opt.zero_grad()
        out = self.net(inp)
        loss = N.loss_fn(out, target, self.a.noise, tmask)
        loss.backward()
        self.opt.step()
        self.sched.step()
        return loss


def vis_batch(ds, a, device):
    """A fixed batch for the target-evolution figure (independent RNG)."""
    st = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    seed_all(12345)
    items = [ds[i] for i in ([0, 101, 2024, 4999] if a.domain == 'image' else range(4))]
    clean = torch.stack([torch.from_numpy(it['gt']) for it in items]).permute(0, 3, 1, 2).to(device)
    if a.fixed is not None:
        noisy = torch.stack([torch.from_numpy(it['noisy']) for it in items]).permute(0, 3, 1, 2).to(device)
    else:
        noisy, _ = N.add_noise(clean, a.noise, a.dt)
    random.setstate(st[0]); np.random.set_state(st[1]); torch.set_rng_state(st[2])
    return clean, noisy


def train(a):
    configure(); seed_all(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ds, ch = build(a)
    device = a.device
    model = Model(a, ch, device)
    meta = dict(vars(a), protocol='P', batch=4, lr=3e-4, milestones=[25000, 40000] if a.schedule == 'paper' else None,
                loss='masked-L2' if a.noise in N.MASKED else 'L1', zero_mean=model.zero_mean, channels=ch,
                samples_per_epoch=len(ds), steps_per_epoch=math.ceil(len(ds) / 4),
                torch=torch.__version__, tf32=os.environ.get('PP_FP32') != '1', gpu=torch.cuda.get_device_name() if device == 'cuda' else 'cpu',
                started_utc=datetime.now(timezone.utc).isoformat())
    step = epoch = offset = 0
    saved_rng = None
    if (out / 'latest.pt').exists():
        s = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
        for k in ('domain', 'method', 'noise', 'seed', 'steps', 'fixed', 'schedule', 'keys', 'gray', 'tf32'):
            if s['meta'].get(k) != meta.get(k):
                raise ValueError(f'resume mismatch on {k}: {s["meta"].get(k)} vs {meta.get(k)}')
        model.net.load_state_dict(s['net']); model.opt.load_state_dict(s['opt']); model.sched.load_state_dict(s['sched'])
        if model.teacher is not None:
            model.teacher.load_state_dict(s['teacher']); model.teacher_epoch = s['teacher_epoch']
        step, epoch, offset, saved_rng = s['step'], s['epoch'], s['offset'], s['rng']
        meta['started_utc'] = s['meta']['started_utc']
        print(f'** resumed at step {step} (epoch {epoch}, offset {offset})', flush=True)
    (out / 'config.json').write_text(json.dumps(meta, indent=2) + '\n')
    if step >= a.steps:
        print('already complete'); return

    def save():
        st = dict(meta=meta, net=model.net.state_dict(), opt=model.opt.state_dict(), sched=model.sched.state_dict(),
                  step=step, epoch=epoch, offset=offset,
                  rng=(random.getstate(), np.random.get_state(), torch.get_rng_state(),
                       torch.cuda.get_rng_state_all() if device == 'cuda' else []))
        if model.teacher is not None:
            st.update(teacher=model.teacher.state_dict(), teacher_epoch=model.teacher_epoch)
        tmp = out / 'latest.pt.tmp'; torch.save(st, tmp); tmp.replace(out / 'latest.pt')
        tmp = out / 'weights.pth.tmp'; torch.save(model.net.state_dict(), tmp); tmp.replace(out / 'weights.pth')

    vclean, vnoisy = vis_batch(ds, a, device)
    np.savez_compressed(out / 'vis_input.npz', clean=vclean.cpu().numpy(), noisy=vnoisy.cpu().numpy())

    def save_targets(ep):
        if a.method not in ('idr', 'nr2n'):
            return
        with torch.no_grad():
            x = vnoisy - .5 if model.zero_mean else vnoisy
            tgt = model.teacher(x) if (a.method == 'idr' and ep >= 1) else x
            tgt = tgt + .5 if model.zero_mean else tgt
        np.save(out / f'vis_target_epoch{ep}.npy', tgt.cpu().numpy())

    print(f'{a.domain}|{a.method}|{a.noise}|seed {a.seed}|fixed={a.fixed}|{len(ds)} samples/epoch|steps {a.steps}', flush=True)
    t0 = time.perf_counter(); s0 = step
    mf = (out / 'metrics.jsonl').open('a')
    while step < a.steps:
        seed_all(a.seed * 1000 + epoch)
        g = torch.Generator().manual_seed(a.seed * 1000 + epoch)
        loader = DataLoader(ds, batch_size=4, shuffle=True, drop_last=False, num_workers=a.workers,
                            generator=g, pin_memory=True, persistent_workers=False,
                            worker_init_fn=lambda w: np.random.seed(torch.initial_seed() % 2**32))
        for idx, batch in enumerate(loader):
            if idx < offset:
                continue
            if saved_rng is not None:
                random.setstate(saved_rng[0]); np.random.set_state(saved_rng[1]); torch.set_rng_state(saved_rng[2])
                if device == 'cuda' and saved_rng[3]:
                    torch.cuda.set_rng_state_all(saved_rng[3])
                saved_rng = None
            clean = batch['gt'].to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
            given = batch['noisy'].to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous() if 'noisy' in batch else None
            lr = model.opt.param_groups[0]['lr']
            loss = float(model.step(clean, given, epoch).item())
            if not math.isfinite(loss):
                raise RuntimeError('non-finite loss')
            if idx == 0:
                save_targets(epoch)
            step += 1; offset = idx + 1
            ep_rec = epoch
            if offset == len(loader):
                epoch += 1; offset = 0
            if step % 50 == 0 or step == a.steps:
                rate = (step - s0) / (time.perf_counter() - t0)
                mf.write(json.dumps(dict(step=step, epoch=ep_rec, loss=loss, lr=lr, steps_per_s=rate)) + '\n'); mf.flush()
            if step % 1000 == 0 or step == a.steps:
                rate = (step - s0) / (time.perf_counter() - t0)
                print(f'step {step:>6}/{a.steps} ep {ep_rec + 1} loss {loss:.5f} lr {lr:.2e} {rate:.1f} it/s '
                      f'ETA {(a.steps - step) / rate / 60:.0f} min', flush=True)
                save()
            if step % a.snap_every == 0:
                torch.save(model.net.state_dict(), out / f'weights_step{step}.pth')
            if step == a.steps:
                break
    save()
    (out / 'training_summary.json').write_text(json.dumps(dict(completed_steps=step, epochs=epoch, offset=offset,
        finished_utc=datetime.now(timezone.utc).isoformat()), indent=2) + '\n')
    print('DONE', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', choices=['image', 'seis'], required=True)
    p.add_argument('--method', choices=['idr', 'sup', 'n2n', 'nr2n'], required=True)
    p.add_argument('--noise', required=True)
    p.add_argument('--data', default='data/imagenet_val')
    p.add_argument('--keys', default='m2,ot,sa', help='seismic surveys for training')
    p.add_argument('--gray', action='store_true')
    p.add_argument('--fixed', default=None, help='fixed noisy realisation id (seismic file tag; images: any value)')
    p.add_argument('--dt', type=float, default=0.004)
    p.add_argument('--level', default=None, help='lo,hi noise-level range override (field runs)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--schedule', choices=['paper', 'cosine'], default='paper')
    p.add_argument('--snap-every', type=int, default=10000)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if a.noise not in N.IMAGE_TRAIN and a.noise not in N.SEIS_TRAIN:
        raise SystemExit(f'unknown noise {a.noise}')
    if a.domain == 'image':
        a.keys = None
    a.level = tuple(float(v) for v in a.level.split(',')) if a.level else None
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with (Path(a.out) / 'console.log').open('a', buffering=1) as f, \
            contextlib.redirect_stdout(Tee(sys.stdout, f)), contextlib.redirect_stderr(Tee(sys.stderr, f)):
        print('\nInvocation:', ' '.join(sys.argv), datetime.now(timezone.utc).isoformat(), flush=True)
        try:
            train(a)
        except Exception:
            traceback.print_exc(); return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
