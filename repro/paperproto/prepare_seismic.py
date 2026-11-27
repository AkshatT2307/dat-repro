"""Build data/seis/<key>/{train.npy,test.npy,meta.json} for every survey.

Arrays are float32 (N, time, trace), dt in meta. Each section is scaled to unit RMS
over its live (non-zero) samples. Splits are spatially disjoint with guard bands.

  m2        Marmousi2 NMO stack (time); dead traces 0-239, 2396-2720 removed.
            train traces [240,1640), guard 256, test [1896,2396).
  m2_prestm / m2_postm / m2_srme : same test region of other processed versions (test only).
  ot / sa   SEG/EAGE Overthrust (land) / Salt (marine) 3-D velocity -> convolutional
            synthetic (Gardner density, normal incidence, 30 Hz Ricker, dt 4 ms).
  f3        Netherlands F3 field (marine 3-D), facies-benchmark arrays; time padded 255->256.
  st        Stratton 3-D field (land), crossline-direction sections (309 traces), dt 2 ms.
"""
import argparse, hashlib, io, json, sys, tarfile, zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seismic_data import read_segy

RAW = ROOT / 'data' / 'seis_raw'
OUT = ROOT / 'data' / 'seis'


def unit_rms(sec):
    live = sec != 0
    r = np.sqrt(np.mean(sec[live] ** 2)) if live.any() else 1.0
    return (sec / r).astype(np.float32), float(r)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write(key, train, test, **meta):
    d = OUT / key; d.mkdir(parents=True, exist_ok=True)
    m = dict(key=key, **meta)
    if train is not None:
        np.save(d / 'train.npy', train); m.update(train_shape=list(train.shape), train_sha256=sha(train))
        m['train_zero_fraction'] = float((train == 0).mean())
    np.save(d / 'test.npy', test); m.update(test_shape=list(test.shape), test_sha256=sha(test))
    m['test_zero_fraction'] = float((test == 0).mean())
    (d / 'meta.json').write_text(json.dumps(m, indent=2) + '\n')
    print(key, 'train', None if train is None else train.shape, 'test', test.shape, flush=True)


# ------------------------------------------------------------------ Marmousi2
def marmousi():
    nmo = np.load(ROOT / 'data/marmousi2/section.npy')
    tr, te = (240, 1640), (1896, 2396)
    train, r = unit_rms(nmo[:, tr[0]:tr[1]])
    test = (nmo[:, te[0]:te[1]] / np.sqrt(np.mean(nmo[:, te[0]:te[1]][nmo[:, te[0]:te[1]] != 0] ** 2))).astype(np.float32)
    write('m2', train[None], test[None], dt=0.004, survey='AGL Elastic Marmousi2 NMOstack_time',
          kind='marine 2D elastic FD synthetic (processed)', train_traces=tr, test_traces=te, guard=256,
          note='dead traces 0-239 and 2396-2720 excluded')
    for key, f in [('m2_prestm', 'Kirchhoff_PreSTM_time'), ('m2_postm', 'Kirchhoff_PoSTM_time'),
                   ('m2_srme', 'NMOstack_SRME_time')]:
        a, _ = read_segy(RAW / 'marmousi2' / f'{f}.segy')
        t, _ = unit_rms(a[:, te[0]:te[1]])
        write(key, None, t[None], dt=0.004, survey=f'Marmousi2 {f}', kind='test-only processing variant', test_traces=te)


# ------------------------------------------------------------------ convolutional synthetics
def ricker(f0, dt, n=None):
    n = n or int(2 * 1.5 / f0 / dt) | 1
    t = (np.arange(n) - n // 2) * dt
    a = (np.pi * f0 * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def conv_section(v, dz, dt=0.004, f0=30.0, nt=None, up=10):
    """v: (nz, nx) velocity m/s. Exact interface times on a 10x finer grid, then decimated."""
    v = v.astype(np.float64)
    rho = 310.0 * v ** 0.25
    z = rho * v
    r = (z[1:] - z[:-1]) / (z[1:] + z[:-1])                     # (nz-1, nx) at interfaces
    tb = 2 * np.cumsum(dz / v, axis=0)                           # two-way time at cell bottoms
    ti = tb[:-1]                                                 # interface times
    fdt = dt / up
    nfine = int(np.ceil(tb[-1].max() / fdt)) + 1
    refl = np.zeros((nfine, v.shape[1]))
    idx = ti / fdt
    i0 = np.floor(idx).astype(int); w1 = idx - i0
    cols = np.broadcast_to(np.arange(v.shape[1]), r.shape)
    np.add.at(refl, (i0, cols), r * (1 - w1))
    np.add.at(refl, (np.minimum(i0 + 1, nfine - 1), cols), r * w1)
    wav = ricker(f0, fdt)
    nfft = 1 << int(np.ceil(np.log2(nfine + len(wav))))
    syn = np.fft.irfft(np.fft.rfft(refl, nfft, axis=0) * np.fft.rfft(wav, nfft)[:, None], nfft, axis=0)
    syn = syn[len(wav) // 2: len(wav) // 2 + nfine][::up]
    tmax = tb[-1].min()                                          # stop above the shallowest model bottom
    n = nt or int(tmax / dt)
    return syn[:n].astype(np.float32)


def load_ot():
    with tarfile.open(RAW / 'overthrust/Overthrust_3D_CD1.tar.gz') as t:
        m = [x for x in t.getmembers() if x.name.endswith('overthrust.vites')][0]
        raw = t.extractfile(m).read()
    v = np.frombuffer(raw, '>f4').reshape(187, 801, 801)          # z slowest, x fastest
    return v, 25.0


def load_sa():
    with tarfile.open(RAW / 'salt/Salt_Model_3D.tar.gz') as t:
        m = [x for x in t.getmembers() if x.name.upper().endswith('SALTF.ZIP')][0]
        zb = t.extractfile(m).read()
    with zipfile.ZipFile(io.BytesIO(zb)) as z:
        name = [n for n in z.namelist() if 'saltf' in n.lower()][0]
        raw = z.read(name)
    v = np.frombuffer(raw, '>f4').reshape(210, 676, 676)
    return v, 20.0


def synthetic(key, loader, train_y, test_y, desc):
    v, dz = loader()
    zmeans = v.reshape(v.shape[0], -1).mean(1)
    assert zmeans[-1] > zmeans[0], 'velocity should increase with depth; check axis order'
    print(key, 'velocity', v.shape, v.min(), v.max(), flush=True)
    nt = min(int((2 * np.cumsum(dz / v[:, y, :].astype(float), 0)[-1]).min() / 0.004) for y in range(0, v.shape[1], 25))
    secs = lambda ys: np.stack([unit_rms(conv_section(v[:, y, :], dz, nt=nt))[0] for y in ys])
    write(key, secs(train_y), secs(test_y), dt=0.004, survey=desc, kind='convolutional synthetic from 3-D velocity',
          modelling='Gardner rho=310 V^0.25; normal-incidence reflectivity at exact interface times; 30 Hz Ricker; dt 4 ms',
          train_inlines=[int(train_y[0]), int(train_y[-1]), int(train_y[1] - train_y[0])],
          test_inlines=[int(y) for y in test_y], dz=dz)


# ------------------------------------------------------------------ field data
def mad_sigma(sec):
    """Donoho-style noise std estimate from the finest diagonal Haar detail."""
    s = sec[: sec.shape[0] // 2 * 2, : sec.shape[1] // 2 * 2]
    d = (s[0::2, 0::2] - s[1::2, 0::2] - s[0::2, 1::2] + s[1::2, 1::2]) / 2
    return float(np.median(np.abs(d)) / 0.6745)


def f3():
    with zipfile.ZipFile(RAW / 'f3/data.zip') as z:
        names = z.namelist()
        def get(suffix):
            n = [x for x in names if x.endswith(suffix)][0]
            return np.load(io.BytesIO(z.read(n)))
        tr, te = get('train/train_seismic.npy'), get('test_once/test1_seismic.npy')
    print('f3 raw', tr.shape, te.shape, flush=True)
    sec = lambda a, i: np.pad(a[i].T, ((0, 1), (0, 0)), mode='reflect')   # (time 256, xline 701)
    train = np.stack([unit_rms(sec(tr, i))[0] for i in range(tr.shape[0])])
    test = np.stack([unit_rms(sec(te, i))[0] for i in range(0, te.shape[0], 20)])
    sig = [mad_sigma(s) for s in train[::20]]
    write('f3', train, test, dt=0.004, survey='Netherlands F3 (Alaudah et al. facies benchmark, Zenodo 3755060)',
          kind='marine 3-D field (post-stack), real noise; no clean reference',
          note='inline sections; 255 time samples reflect-padded to 256', mad_sigma_median=float(np.median(sig)),
          test_inlines='test1 every 20th')


def read_segy_headers(path):
    with open(path, 'rb') as f:
        h = f.read(3600)
        ns = int.from_bytes(h[3220:3222], 'big'); dt = int.from_bytes(h[3216:3218], 'big')
        fmt = int.from_bytes(h[3224:3226], 'big'); assert fmt == 1
        raw = np.fromfile(f, dtype=np.uint8)
    rec = 240 + 4 * ns
    raw = raw[: len(raw) // rec * rec].reshape(-1, rec)
    hdr = raw[:, :240]
    il = hdr[:, 8:12].copy().view('>i4').ravel(); cdp = hdr[:, 20:24].copy().view('>i4').ravel()
    bits = raw[:, 240:].copy().view('>u4').astype(np.uint32)
    frac = (bits & 0x00ffffff).astype(np.float64) / 2 ** 24
    ex = ((bits >> 24) & 0x7f).astype(np.int32) - 64
    vals = np.where(bits >> 31, -1., 1.) * frac * np.power(16., ex)
    return vals.astype(np.float32), il, cdp, dt / 1e6


def stratton():
    vals, il, cdp, dt = read_segy_headers(RAW / 'stratton/Stratton3D_32bit.sgy')
    ils, cdps = np.unique(il), np.unique(cdp)
    cube = np.zeros((len(cdps), vals.shape[1], len(ils)), np.float32)    # (xline, time, inline)
    cube[np.searchsorted(cdps, cdp), :, np.searchsorted(ils, il)] = vals
    print('stratton cube', cube.shape, 'dt', dt, 'inlines', ils.min(), ils.max(), 'cdps', cdps.min(), cdps.max(), flush=True)
    train = np.stack([unit_rms(cube[i])[0] for i in range(0, 170)])
    test = np.stack([unit_rms(cube[i])[0] for i in range(190, len(cdps), 5)])
    sig = [mad_sigma(s) for s in train[::10]]
    write('st', train, test, dt=dt, survey='Stratton 3-D, South Texas (BEG, UT Austin; SEG open data)',
          kind='land 3-D field (post-stack migrated), real noise; no clean reference',
          note='sections along the inline axis at fixed CDP; train CDP index 0-169, guard 20, test every 5th from 190',
          mad_sigma_median=float(np.median(sig)))


def line2d(key, path, desc, kind, train_frac=0.7, guard=256, test_w=512):
    """Single post-stack 2-D field line: train on the first part, guard band, test sections after it."""
    a, hdr = read_segy(RAW / path)
    a = np.asarray(a, np.float32)                       # (time, trace)
    dt = hdr['sample_interval_header'] / 1e6            # microseconds in the binary header
    live_t = np.where(np.abs(a).sum(1) > 0)[0]          # trim all-zero samples at the end
    a = a[: live_t[-1] + 1]
    n = a.shape[1]; ntr = int(n * train_frac)
    train, _ = unit_rms(a[:, :ntr])
    tests = [unit_rms(a[:, s:s + test_w])[0] for s in range(ntr + guard, n - test_w + 1, test_w)]
    sig = [mad_sigma(train[:, s:s + 512]) for s in range(0, ntr - 511, 512)]
    write(key, train[None], np.stack(tests), dt=dt, survey=desc, kind=kind,
          note=f'single line {a.shape}; train traces [0,{ntr}), guard {guard}, test sections of {test_w} traces',
          mad_sigma_median=float(np.median(sig)), raw_file=str(path))


# ------------------------------------------------------------------ fixed noisy realisations
def fixed(key, noise, seed):
    sys.path.insert(0, str(ROOT / 'paperproto'))
    import torch
    from noise import band_limited
    d = OUT / key; meta = json.loads((d / 'meta.json').read_text())
    a = np.load(d / 'train.npy', mmap_mode='r')
    out = np.lib.format.open_memmap(d / f'train_fixed_{noise}_s{seed}.npy', 'w+', np.float32, a.shape)
    rng = np.random.default_rng([seed, 99])
    sig = rng.uniform(0, 1, len(a))
    g = torch.Generator().manual_seed(seed)
    for i in range(len(a)):
        if noise == 'sg':
            n = torch.randn(a.shape[1:], generator=g).numpy()
        else:
            n = band_limited((1, 1) + a.shape[1:], meta['dt'], (5.0, 60.0), 'cpu', g)[0, 0].numpy()
        out[i] = a[i] + sig[i] * n
    out.flush()
    meta.setdefault('fixed', {})[f'{noise}_s{seed}'] = dict(sigmas=sig.tolist())
    (d / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('fixed', key, noise, seed, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('what', nargs='+'); a = p.parse_args()
    for w in a.what:
        if w == 'm2': marmousi()
        elif w == 'ot': synthetic('ot', load_ot, np.arange(0, 560, 2), np.arange(640, 801, 20), 'SEG/EAGE 3-D Overthrust (land)')
        elif w == 'sa': synthetic('sa', load_sa, np.arange(0, 470, 2), np.arange(540, 676, 15), 'SEG/EAGE 3-D Salt (marine)')
        elif w == 'f3': f3()
        elif w == 'st': stratton()
        elif w == 'l2': line2d('l2', 'npra/R_1.SGY', 'USGS NPR-A Alaska regional line 1 (OFR 00-286), stacked + migrated',
                               'land 2-D field (post-stack migrated), real noise; no clean reference')
        elif w == 'w2': line2d('w2', 'namss/WS82-611__5692.sgy', 'USGS NAMSS W-34-82-MB line WS82-611 (Gulf of Mexico), migrated',
                               'marine 2-D field (post-stack migrated), real noise; no clean reference')
        elif w.startswith('fixed:'):
            _, key, noise, seed = w.split(':'); fixed(key, noise, int(seed))
        else: raise SystemExit(w)
