"""Writes paperproto/jobs.json (training + data-prep jobs; eval/analysis jobs appended by later sections)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = '.venv/bin/python'
R = 'runs/paperproto'
jobs = []


def J(id, cmd, done, prio=50, deps=(), env=None):
    jobs.append(dict(id=id, cmd=cmd, done_file=done, prio=prio, deps=list(deps), env=env or {}))


def prep(id, what, prio=5, deps=(), wait_for=None):
    w = f'until grep -q DONE data/seis_raw/download.log && [ -s {wait_for} ]; do sleep 30; done; ' if wait_for else ''
    J(id, ['bash', '-c', f'{w}{PY} paperproto/prepare_seismic.py {what} && touch data/seis/.done_{id}'],
      f'data/seis/.done_{id}', prio, deps)


def train(id, prio, args, deps=(), env=None, workers=4):
    J(id, [PY, 'paperproto/pp_train.py', '--out', f'{R}/{id}', '--workers', str(workers)] + args,
      f'{R}/{id}/training_summary.json', prio, deps, env)


# ---------------------------------------------------------------- data prep (CPU)
prep('prep_sa', 'sa', wait_for='data/seis_raw/salt/Salt_Model_3D.tar.gz')
prep('prep_fixed', 'fixed:sa:sg:0 fixed:sa:sb:0', deps=['prep_sa'])
prep('prep_f3', 'f3 fixed:f3:sg:0', wait_for='data/seis_raw/f3/data.zip')
J('prep_st', ['bash', '-c', 'until [ "$(stat -c %s data/seis_raw/stratton/Stratton3D_32bit.sgy)" = 443764680 ]; do sleep 30; done; '
              f'{PY} paperproto/prepare_seismic.py st fixed:st:sg:0 && touch data/seis/.done_prep_st'],
  'data/seis/.done_prep_st', 5)
SEIS = ['prep_sa', 'prep_fixed']

# ---------------------------------------------------------------- wave 1: central comparisons
img = ['--domain', 'image']
seis = ['--domain', 'seis', '--keys', 'm2,ot,sa']
train('A1_idr_g_s0', 10, img + ['--method', 'idr', '--noise', 'g', '--seed', '0'])
train('A2_sup_g_s0', 10, img + ['--method', 'sup', '--noise', 'g', '--seed', '0'])
train('C1_idr_sg_s0', 11, seis + ['--method', 'idr', '--noise', 'sg', '--seed', '0'], SEIS)
train('C2_sup_sg_s0', 11, seis + ['--method', 'sup', '--noise', 'sg', '--seed', '0'], SEIS)
train('A5_idrfix_g_s0', 12, img + ['--method', 'idr', '--noise', 'g', '--seed', '0', '--fixed', 'img'])
train('C5_idrfix_sg_s0', 12, seis + ['--method', 'idr', '--noise', 'sg', '--seed', '0', '--fixed', 'sg_s0'], SEIS)
train('A3_nr2n_g_s0', 13, img + ['--method', 'nr2n', '--noise', 'g', '--seed', '0'])
train('C3_nr2n_sg_s0', 13, seis + ['--method', 'nr2n', '--noise', 'sg', '--seed', '0'], SEIS)

# ---------------------------------------------------------------- wave 2: seeds, other noises
# (user 2026-11-19: extra training seeds 1-2 removed to save time)
for n, tag in [('sb', 'C7'), ('ss', 'C8'), ('sd', 'C9')]:
    for m in ('idr', 'sup'):
        train(f'{tag}_{m}_{n}_s0', 22, seis + ['--method', m, '--noise', n, '--seed', '0'], SEIS)
train('C6_idrfix_sb_s0', 22, seis + ['--method', 'idr', '--noise', 'sb', '--seed', '0', '--fixed', 'sb_s0'], SEIS)
for m in ('idr', 'sup'):
    train(f'A6_{m}_gray_s0', 23, img + ['--method', m, '--noise', 'g', '--gray', '--seed', '0'])
    train(f'A7_{m}_binomial_s0', 23, img + ['--method', m, '--noise', 'binomial', '--seed', '0'])
    train(f'A8_{m}_impulse_s0', 23, img + ['--method', m, '--noise', 'impulse', '--seed', '0'])
train('A4_n2n_g_s0', 24, img + ['--method', 'n2n', '--noise', 'g', '--seed', '0'])
train('C4_n2n_sg_s0', 24, seis + ['--method', 'n2n', '--noise', 'sg', '--seed', '0'], SEIS)

# ---------------------------------------------------------------- wave 3: correlated, field
for k in (1, 2, 3, 4):
    train(f'A9_idr_scn{k}_s0', 30, img + ['--method', 'idr', '--noise', f'scn-{k}', '--seed', '0'])
F3 = ['--domain', 'seis', '--keys', 'f3']
ST = ['--domain', 'seis', '--keys', 'st', '--dt', '0.002']
train('F1_idrfix_f3_sg', 31, F3 + ['--method', 'idr', '--noise', 'sg', '--fixed', 'sg_s0'], ['prep_f3'])
train('F1_sup_f3_sg_oracle', 31, F3 + ['--method', 'sup', '--noise', 'sg'], ['prep_f3'])
train('F3_idrfix_st_sg', 31, ST + ['--method', 'idr', '--noise', 'sg', '--fixed', 'sg_s0'], ['prep_st'])
train('F3_sup_st_sg_oracle', 31, ST + ['--method', 'sup', '--noise', 'sg'], ['prep_st'])

# ---------------------------------------------------------------- wave 4: controls / budget probe
train('A1_idr_g_s0_fp32', 40, img + ['--method', 'idr', '--noise', 'g', '--seed', '0'], env={'PP_FP32': '1'})
# (user 2026-11-19: A11 long-budget probe removed)

# ---------------------------------------------------------------- real field noise (self-supervised on raw data)
import json as _j
s3 = _j.loads((ROOT / 'data/seis/f3/meta.json').read_text())['mad_sigma_median']
sst = _j.loads((ROOT / 'data/seis/st/meta.json').read_text())['mad_sigma_median']
train('F2_idr_f3_real', 32, F3 + ['--method', 'idr', '--noise', 'sg', '--fixed', 'raw', '--level', f'{.5*s3:.5f},{1.5*s3:.5f}'], ['prep_f3'])
train('F4_idr_st_real', 32, ST + ['--method', 'idr', '--noise', 'sg', '--fixed', 'raw', '--level', f'{.5*sst:.5f},{1.5*sst:.5f}'], ['prep_st'])

# ---------------------------------------------------------------- evaluation jobs (run as soon as deps finish)
trains = [j for j in jobs if 'pp_train.py' in ' '.join(j['cmd'])]
for j in trains:
    tid = j['id']
    if '--domain' in j['cmd'] and j['cmd'][j['cmd'].index('--domain') + 1] == 'image':
        J(f'eval_{tid}', [PY, 'paperproto/eval_images.py', '--run', f'{R}/{tid}'], f'{R}/{tid}/eval/summary.json', 1, [tid])
    else:
        J(f'eval_{tid}', [PY, 'paperproto/eval_seis.py', '--run', f'{R}/{tid}'], f'{R}/{tid}/eval/summary.json', 1, [tid])
# (user: training length fixed at 50k by the paper; snapshot sweep dropped)
DR = 'runs/paperproto/drift'
for name, w, dep in [('A1_idr_g_s0', f'{R}/A1_idr_g_s0/weights.pth', ['A1_idr_g_s0']),
                     ('A2_sup_g_s0', f'{R}/A2_sup_g_s0/weights.pth', ['A2_sup_g_s0']),
                     ('old_b4_256_idr', 'runs/steps50k_b4_256/imagenet/idr/weights.pth', []),
                     ('old_b4_256_sup', 'runs/steps50k_b4_256/imagenet/supervised/weights.pth', []),
                     ('old_b4_128_idr', 'runs/steps50k_b4_128/imagenet/idr/weights.pth', []),
                     ('old_b4_128_sup', 'runs/steps50k_b4_128/imagenet/supervised/weights.pth', []),
                     ('old_b192_idr', 'runs/continuation5/imagenet/idr/b192_e20/weights.pth', []),
                     ('old_b192_sup', 'runs/continuation5/imagenet/supervised/b192_e20/weights.pth', [])]:
    J(f'drift_{name}', [PY, 'paperproto/drift_images.py', '--weights', w, '--name', name] + ([] if dep else ['--seeds', '1']),
      f'{DR}/{name}.json', 3 if dep else 35, dep)
for tid in ('C1_idr_sg_s0', 'C2_sup_sg_s0', 'C7_idr_sb_s0', 'C7_sup_sb_s0'):
    J(f'mismatch_{tid}', [PY, 'paperproto/eval_seis.py', '--run', f'{R}/{tid}', '--noises', 'sg,sb', '--tag', 'mismatch',
                          '--tests', 'm2,ot,sa'], f'{R}/{tid}/mismatch/summary.json', 2, [tid])
J('field_f3', [PY, 'paperproto/eval_field.py', '--key', 'f3', '--out', f'{R}/field_f3', '--models',
               f'idr_real={R}/F2_idr_f3_real,idr_synth={R}/C1_idr_sg_s0,sup_synth={R}/C2_sup_sg_s0,idr_fixed_f3={R}/F1_idrfix_f3_sg'],
  f'{R}/field_f3/summary.json', 2, ['F2_idr_f3_real', 'C1_idr_sg_s0', 'C2_sup_sg_s0', 'F1_idrfix_f3_sg'])
J('field_st', [PY, 'paperproto/eval_field.py', '--key', 'st', '--out', f'{R}/field_st', '--models',
               f'idr_real={R}/F4_idr_st_real,idr_synth={R}/C1_idr_sg_s0,sup_synth={R}/C2_sup_sg_s0,idr_fixed_st={R}/F3_idrfix_st_sg'],
  f'{R}/field_st/summary.json', 2, ['F4_idr_st_real', 'C1_idr_sg_s0', 'C2_sup_sg_s0', 'F3_idrfix_st_sg'])

# ---------------------------------------------------------------- auto results dump, then optional seeds
main_ids = [j['id'] for j in jobs if not j['id'].startswith('prep')]
J('report_main', ['bash', '-c', f'LD_LIBRARY_PATH=$PWD/.venv/blasshim {PY} paperproto/report.py && touch {R}/.report_main'],
  f'{R}/.report_main', 0, main_ids)
seed_ids = []
for s_ in (1, 2):
    for tid, args in [(f'A1_idr_g_s{s_}', img + ['--method', 'idr', '--noise', 'g']), (f'A2_sup_g_s{s_}', img + ['--method', 'sup', '--noise', 'g']),
                      (f'C1_idr_sg_s{s_}', seis + ['--method', 'idr', '--noise', 'sg']), (f'C2_sup_sg_s{s_}', seis + ['--method', 'sup', '--noise', 'sg'])]:
        train(tid, 90, args + ['--seed', str(s_)], [t for t in main_ids if t.startswith(('A', 'C', 'F'))])
        ev = 'eval_images.py' if tid.startswith('A') else 'eval_seis.py'
        J(f'eval_{tid}', [PY, f'paperproto/{ev}', '--run', f'{R}/{tid}'], f'{R}/{tid}/eval/summary.json', 1, [tid])
        seed_ids += [tid, f'eval_{tid}']
J('report_seeds', ['bash', '-c', f'{PY} paperproto/report.py && touch {R}/.report_seeds'], f'{R}/.report_seeds', 0, seed_ids + ['report_main'])

if __name__ == '__main__':
    extra = ROOT / 'paperproto' / 'jobs_extra.json'
    if extra.exists():
        jobs += json.loads(extra.read_text())
    (ROOT / 'paperproto' / 'jobs.json').write_text(json.dumps(jobs, indent=1) + '\n')
    print(len(jobs), 'jobs')
