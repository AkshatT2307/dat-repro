"""Pull one finished kernel's output into runs/paperproto and print its shard state + speed.
Usage: python kgl/pull_one.py <slug> [<slug> ...]"""
import sys, json, shutil
sys.path.insert(0, 'kgl'); import dispatch as D

accs = D.accounts(); pushes = {p['slug']: p for p in D.load_state()['pushes']}
dst = D.ROOT / 'runs' / 'paperproto'
for slug in sys.argv[1:]:
    p = pushes[slug]; o = D.WORKDIR / 'out' / slug; o.mkdir(parents=True, exist_ok=True)
    D.kaggle(accs[p['acc']], 'kernels', 'output', f"{p['user']}/{slug}", '-p', str(o), check=False)
    if not (o / 'shard_state.json').exists():
        print(slug, 'no shard_state.json (kernel crashed?)'); continue
    st = json.loads((o / 'shard_state.json').read_text())
    print(slug, 'done', len(st.get('done', [])), 'failed', st.get('failed'), 'unfinished', st.get('unfinished'), st.get('elapsed_min'), 'min')
    mine = {j.split('/')[2] for j in p['jobs'] for j in [f'runs/paperproto/{j}']}
    for r in (o / 'runs' / 'paperproto').glob('*'):
        if r.name in p['jobs'] or r.name in mine or r.name.startswith(('G_', 'field_')) or (r / 'eval_2d').exists():
            shutil.copytree(r, dst / r.name, dirs_exist_ok=True)
            m = r / 'metrics.jsonl'
            if m.exists():
                last = json.loads(m.read_text().strip().splitlines()[-1])
                print(f"   {r.name}: step {last['step']}, {last['steps_per_s']:.2f} it/s")
    if (o / 'logs').exists():
        shutil.copytree(o / 'logs', dst / 'logs_kaggle', dirs_exist_ok=True)
