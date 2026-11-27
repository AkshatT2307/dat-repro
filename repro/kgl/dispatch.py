"""Local orchestrator: fan the Kaggle job list out over several Kaggle accounts.

Credentials: one file per account in ~/.kaggle/accounts/: <name>.txt with just the API token
(username is looked up from the token), or <name>.json with the legacy
kaggle.json ({"username": ..., "key": ...}). Never commit these.

  python kgl/dispatch.py accounts                      # check every account authenticates
  python kgl/dispatch.py upload-data --src a.tar b.tar  # private <user>/dat-seis on every account
  python kgl/dispatch.py smoke --acc NAME              # 2k-step benchmark on both T4s
  python kgl/dispatch.py plan [--only REGEX]           # balance jobs over accounts -> kgl/assignments.json
  python kgl/dispatch.py push [--acc NAME]             # one kernel per account (resumes from its last version)
  python kgl/dispatch.py status
  python kgl/dispatch.py pull                          # outputs -> runs/paperproto/<run>/

Run from repro/ with the venv that has the kaggle CLI.
"""
import argparse, base64, io, json, os, re, shutil, subprocess, sys, tarfile, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ACC_DIR = Path.home() / '.kaggle' / 'accounts'
WORKDIR = HERE / '_build'                # gitignored
STATE = HERE / 'state.json'             # push history (gitignored)
ASSIGN = HERE / 'assignments.json'
CODE = ['paperproto', 'models', 'utils', 'configs', 'seismic_data.py']
DATASET = 'dat-seis'


def accounts():
    """<name>.txt holding just an API token, or <name>.json with {"token"} or legacy {"username", "key"}.
    For tokens the username is looked up once from Kaggle (token introspection) and cached."""
    accs = {}
    cache_f = WORKDIR / '_cfg' / 'usernames.json'
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    for f in sorted(ACC_DIR.glob('*.json')) + sorted(ACC_DIR.glob('*.txt')):
        acc = json.loads(f.read_text()) if f.suffix == '.json' else {'token': f.read_text().strip()}
        acc['_name'] = f.stem
        if 'username' not in acc:
            if f.stem not in cache:
                cache[f.stem] = whoami(acc)
            acc['username'] = cache[f.stem]
        accs[f.stem] = acc
    if not accs:
        sys.exit(f'no account files in {ACC_DIR}')
    cache_f.parent.mkdir(parents=True, exist_ok=True); cache_f.write_text(json.dumps(cache, indent=1))
    return accs


def whoami(acc):
    code = ('from kaggle.api.kaggle_api_extended import KaggleApi\n'
            'a = KaggleApi(); a.authenticate(); print(a.get_config_value("username"))')
    p = subprocess.run([sys.executable, '-c', code], env=kenv(acc), capture_output=True, text=True, cwd=WORKDIR)
    u = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ''
    if p.returncode or not u or u == 'None':
        sys.exit(f"could not resolve username for account '{acc['_name']}': {p.stdout}{p.stderr}"[-500:])
    return u


def kenv(acc):
    e = {k: v for k, v in os.environ.items() if not k.startswith('KAGGLE_')}
    e['KAGGLE_CONFIG_DIR'] = str(WORKDIR / '_cfg' / acc['_name'])   # isolate from any default login
    Path(e['KAGGLE_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
    if 'key' in acc:
        e['KAGGLE_USERNAME'], e['KAGGLE_KEY'] = acc['username'], acc['key']
    if 'token' in acc:
        e['KAGGLE_API_TOKEN'] = acc['token']
    e['PYTHONIOENCODING'] = 'utf-8'
    return e


def kaggle(acc, *args, check=True, cwd=None):
    p = subprocess.run([sys.executable, '-m', 'kaggle', *args], env=kenv(acc), capture_output=True, text=True, encoding='utf-8', errors='replace',
                       cwd=cwd or WORKDIR)
    if check and p.returncode:
        raise RuntimeError(f"kaggle {' '.join(args)} [{acc['username']}]: {p.stdout}{p.stderr}")
    return ((p.stdout or '') + (p.stderr or '')).strip()


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(s):
    STATE.write_text(json.dumps(s, indent=1) + '\n')


def jobs():
    subprocess.run([sys.executable, str(HERE / 'make_kaggle_jobs.py')], check=True, capture_output=True)
    return json.loads((HERE / 'jobs_kaggle.json').read_text())


def code_b64():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as t:
        for c in CODE:
            t.add(ROOT / c, arcname=c, filter=lambda ti: None if '__pycache__' in ti.name else ti)
    return base64.b64encode(buf.getvalue()).decode()


def build_kernel(acc, slug, shard_jobs, cfg, kernel_sources=()):
    d = WORKDIR / slug
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    src = (HERE / 'kernel_template.py').read_text()
    src = (src.replace('__CFG__', json.dumps(cfg)).replace('__JOBS__', json.dumps(shard_jobs))
              .replace('__CODE_B64__', code_b64()))
    (d / 'kernel.py').write_text(src, encoding='utf-8')
    u = acc['username']
    (d / 'kernel-metadata.json').write_text(json.dumps(dict(
        id=f'{u}/{slug}', title=slug, code_file='kernel.py', language='python', kernel_type='script',
        is_private=True, enable_gpu=True, enable_internet=True, dataset_sources=[f'{u}/{DATASET}'],
        kernel_sources=list(kernel_sources), competition_sources=[]), indent=1))
    return d


def has_dataset(acc):
    return kaggle(acc, 'datasets', 'status', f"{acc['username']}/{DATASET}", check=False).strip().endswith('ready')


# ------------------------------------------------------------------ commands
def cmd_accounts(a):
    for name, acc in accounts().items():
        try:
            kaggle(acc, 'quota')
            has = has_dataset(acc)
            print(f'{name:>10} {acc["username"]:<24} ok   dataset {DATASET}: {"yes" if has else "no"}')
        except RuntimeError as e:
            print(f'{name:>10} {acc["username"]:<24} FAIL {str(e)[-200:]}')


def cmd_upload_data(a):
    srcs = [Path(x).resolve() for x in a.src]
    for name, acc in accounts().items():
        if a.acc and name not in a.acc.split(','):
            continue
        d = WORKDIR / f'data_{name}'; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
        for src in srcs:
            os.link(src, d / src.name) if src.drive == d.drive else shutil.copy(src, d / src.name)
        (d / 'dataset-metadata.json').write_text(json.dumps(dict(
            title=DATASET, id=f"{acc['username']}/{DATASET}", licenses=[dict(name='other')]), indent=1))
        if has_dataset(acc):
            print(name, kaggle(acc, 'datasets', 'version', '-p', str(d), '-m', a.msg)[-300:])
        else:
            print(name, kaggle(acc, 'datasets', 'create', '-p', str(d))[-300:])   # private by default


def cmd_smoke(a):
    acc = accounts()[a.acc]
    base = next(j for j in jobs() if j['id'] == 'S_m2_idr_sg_s0')
    sj = []
    for i in range(2):     # two concurrent copies = one per T4
        rid = f'smoke{i}_m2_idr_sg'
        cmd = [x if x != 'runs/paperproto/S_m2_idr_sg_s0' else f'runs/paperproto/{rid}' for x in base['cmd']]
        sj.append(dict(id=rid, cmd=cmd + ['--steps', str(a.steps)], done_file=f'runs/paperproto/{rid}/training_summary.json',
                       deps=[], env={}))
    slug = 'dat-idr-smoke'
    d = build_kernel(acc, slug, sj, dict(shard='smoke', budget_s=3 * 3600, env={'PP_FP32': '1'}))
    print(kaggle(acc, 'kernels', 'push', '-p', str(d), '--accelerator', a.accelerator))
    s = load_state(); s.setdefault('pushes', []).append(dict(acc=a.acc, slug=slug, user=acc['username'])); save_state(s)


def cmd_plan(a):
    accs = list(accounts())
    if a.accounts:
        accs = a.accounts.split(',')
    js = jobs()
    if a.only:
        keep = {j['id'] for j in js if re.search(a.only, j['id'])}
        js = [j for j in js if j['id'] in keep or any(d in keep for d in j['deps'])]
    groups = {}
    for j in js:
        groups.setdefault(j['group'], []).append(j)
    per = {x: [] for x in accs}; load = {x: 0.0 for x in accs}
    for gname, group in sorted(groups.items(), key=lambda kv: (not kv[0].startswith('G_'), -sum(j['cost'] for j in kv[1]))):
        x = min(accs, key=lambda k: load[k])
        per[x] += [g['id'] for g in group]; load[x] += sum(g['cost'] for g in group)
    ASSIGN.write_text(json.dumps(per, indent=1) + '\n')
    for x in accs:
        n = sum(1 for i in per[x] if not i.startswith(('eval', 'mismatch_', 'prep_', 'field_')))
        print(f'{x:>10}: {n} runs, load {load[x]:.1f} run-units  (~{load[x] * a.hours_per_run / a.gpus:.1f} h on {a.gpus} GPUs)')


def cmd_push(a):
    accs = accounts(); per = json.loads(ASSIGN.read_text()); js = {j['id']: j for j in jobs()}
    s = load_state(); s.setdefault('pushes', [])
    for name, ids in per.items():
        if a.acc and name not in a.acc.split(','):
            continue
        acc = accs[name]
        prev = [p for p in s['pushes'] if p['acc'] == name and p['slug'].startswith('dat-idr-shard')]
        ver = len(prev) + 1
        slug = f'dat-idr-shard-{name}-v{ver}'
        ksrc = [f"{acc['username']}/{prev[-1]['slug']}"] if prev else []
        d = build_kernel(acc, slug, [js[i] for i in ids], dict(shard=name, budget_s=a.budget_h * 3600, env={'PP_FP32': '1'}), ksrc)
        print(name, kaggle(acc, 'kernels', 'push', '-p', str(d), '--accelerator', a.accelerator)[-300:])
        s['pushes'].append(dict(acc=name, slug=slug, user=acc['username'], jobs=ids)); save_state(s)


def latest_pushes():
    last = {}
    for p in load_state().get('pushes', []):
        last[(p['acc'], p['slug'].rsplit('-v', 1)[0])] = p
    return list(last.values())


def cmd_status(a):
    accs = accounts()
    for p in latest_pushes():
        print(f"{p['acc']:>10} {p['slug']:<32} {kaggle(accs[p['acc']], 'kernels', 'status', p['user'] + '/' + p['slug'], check=False)[-120:]}")


def cmd_pull(a):
    accs = accounts(); dst = ROOT / 'runs' / 'paperproto'; dst.mkdir(parents=True, exist_ok=True)
    for p in latest_pushes():
        o = WORKDIR / 'out' / p['slug']; o.mkdir(parents=True, exist_ok=True)
        print(p['slug'], kaggle(accs[p['acc']], 'kernels', 'output', p['user'] + '/' + p['slug'], '-p', str(o), check=False)[-200:])
        for r in (o / 'runs' / 'paperproto').glob('*'):
            shutil.copytree(r, dst / r.name, dirs_exist_ok=True)
        if (o / 'shard_state.json').exists():
            st = json.loads((o / 'shard_state.json').read_text())
            print(f"   done {len(st['done'])}  unfinished {st['unfinished']}  failed {st['failed']}  {st['elapsed_min']} min")
        if (o / 'logs').exists():
            shutil.copytree(o / 'logs', dst / 'logs_kaggle', dirs_exist_ok=True)


def main():
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest='cmd', required=True)
    sp.add_parser('accounts')
    u = sp.add_parser('upload-data'); u.add_argument('--src', required=True, nargs='+'); u.add_argument('--acc'); u.add_argument('--msg', default='update')
    s = sp.add_parser('smoke'); s.add_argument('--acc', required=True); s.add_argument('--steps', type=int, default=2000)
    s.add_argument('--accelerator', default='NvidiaTeslaT4')
    pl = sp.add_parser('plan'); pl.add_argument('--accounts'); pl.add_argument('--only')
    pl.add_argument('--gpus', type=int, default=2); pl.add_argument('--hours-per-run', type=float, default=1.5)
    pu = sp.add_parser('push'); pu.add_argument('--acc'); pu.add_argument('--budget-h', type=float, default=11.0)
    pu.add_argument('--accelerator', default='NvidiaTeslaT4')
    sp.add_parser('status'); sp.add_parser('pull')
    a = p.parse_args()
    WORKDIR.mkdir(exist_ok=True)
    globals()['cmd_' + a.cmd.replace('-', '_')](a)


if __name__ == '__main__':
    main()
