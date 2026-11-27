"""Job list for the Kaggle phase (27 Nov 2026 task update).

Writes kgl/jobs_kaggle.json in the same format as paperproto/jobs.json, with
commands relative to the repro/ root and the interpreter called `python`.

S-series: one model per synthetic dataset (m2, ot, sa) instead of the joint corpus.
  S_<key>_<method>_<noise>_s<seed>   protocol P, 50k steps, fresh noise per visit
  S_<key>_idrfix_sg_s0                fixed-noise ablation per dataset
Each training job has an eval job (all 8 corpus test sets, so the cross-dataset
transfer and the Marmousi2 processing variants come for free) and the s0 sg/sb
runs also get the noise-mismatch eval.
G-series (2D field lines) is appended once those lines are prepared.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = 'runs/paperproto'
TESTS = 'm2,m2_prestm,m2_postm,m2_srme,ot,sa'   # f3/st dropped: kratos data unavailable, rebuilt from public mirrors
jobs = []


def J(id, cmd, done, cost, deps=(), env=None, group=None):
    jobs.append(dict(id=id, cmd=cmd, done_file=done, deps=list(deps), env=env or {}, cost=cost, group=group or id))


def train(id, args, cost=1.0):
    J(id, ['python', 'paperproto/pp_train.py', '--out', f'{R}/{id}', '--workers', '2'] + args,
      f'{R}/{id}/training_summary.json', cost, group=id)
    J(f'eval_{id}', ['python', 'paperproto/eval_seis.py', '--run', f'{R}/{id}', '--tests', TESTS],
      f'{R}/{id}/eval/summary.json', 0.08, [id], group=id)


import os
FULL = os.environ.get('KGL_FULL') == '1'   # 27 Nov deadline: default is the reduced set (12 runs)
for key in ('m2', 'ot', 'sa'):
    base = ['--domain', 'seis', '--keys', key]
    for noise in (('sg', 'sb', 'ss', 'sd') if FULL else ('sg', 'sb')):
        for m in ('idr', 'sup'):
            seeds = (0, 1, 2) if (noise == 'sg' and FULL) else (0,)
            for s in seeds:
                train(f'S_{key}_{m}_{noise}_s{s}', base + ['--method', m, '--noise', noise, '--seed', str(s)])
    if FULL:
        J(f'prep_{key}_fixed_sg', ['python', 'paperproto/prepare_seismic.py', f'fixed:{key}:sg:0'],
          f'data/seis/{key}/train_fixed_sg_s0.npy', 0.02, group=f'S_{key}_idrfix_sg_s0')
        train(f'S_{key}_idrfix_sg_s0', base + ['--method', 'idr', '--noise', 'sg', '--seed', '0', '--fixed', 'sg_s0'])
        jobs[-2]['deps'].append(f'prep_{key}_fixed_sg')
        for m in ('idr', 'sup'):
            for noise in ('sg', 'sb'):
                tid = f'S_{key}_{m}_{noise}_s0'
                J(f'mismatch_{tid}', ['python', 'paperproto/eval_seis.py', '--run', f'{R}/{tid}', '--noises', 'sg,sb',
                                      '--tag', 'mismatch', '--tests', 'm2,ot,sa'], f'{R}/{tid}/mismatch/summary.json', 0.05, [tid], group=tid)

# ---------------------------------------------------------------- G-series: 2-D field lines (l2 land, w2 marine)
# Same recipe as F3/Stratton minus the supervised oracle (deadline cut).
SEISDIR = HERE.parent / 'data' / 'seis'
JOINT = ['C1_idr_sg_s0', 'C2_sup_sg_s0', 'C7_idr_sb_s0', 'C7_sup_sb_s0']   # from prior_runs.tar (HF weights)
for key in ('l2', 'w2'):
    g = f'G_{key}'
    sig = json.loads((SEISDIR / key / 'meta.json').read_text())['mad_sigma_median']
    base = ['--domain', 'seis', '--keys', key, '--method', 'idr', '--noise', 'sg', '--seed', '0']
    J(f'G_{key}_idr_real', ['python', 'paperproto/pp_train.py', '--out', f'{R}/G_{key}_idr_real', '--workers', '2'] + base +
      ['--fixed', 'raw', '--level', f'{.5 * sig:.5f},{1.5 * sig:.5f}'], f'{R}/G_{key}_idr_real/training_summary.json', 1.0, group=g)
    J(f'prep_{key}_fixed', ['python', 'paperproto/prepare_seismic.py', f'fixed:{key}:sg:0'],
      f'data/seis/{key}/train_fixed_sg_s0.npy', 0.01, group=g)
    J(f'G_{key}_idrfix_sg', ['python', 'paperproto/pp_train.py', '--out', f'{R}/G_{key}_idrfix_sg', '--workers', '2'] + base +
      ['--fixed', 'sg_s0'], f'{R}/G_{key}_idrfix_sg/training_summary.json', 1.0, [f'prep_{key}_fixed'], group=g)
    J(f'eval_G_{key}_idrfix_sg', ['python', 'paperproto/eval_seis.py', '--run', f'{R}/G_{key}_idrfix_sg', '--tests', key],
      f'{R}/G_{key}_idrfix_sg/eval/summary.json', 0.02, [f'G_{key}_idrfix_sg'], group=g)
    models = ','.join([f'idr_real={R}/G_{key}_idr_real', f'idr_fixed={R}/G_{key}_idrfix_sg',
                       f'idr_synth={R}/C1_idr_sg_s0', f'sup_synth={R}/C2_sup_sg_s0', f'idr_synth_sb={R}/C7_idr_sb_s0'])
    J(f'field_{key}', ['python', 'paperproto/eval_field.py', '--key', key, '--out', f'{R}/field_{key}', '--models', models],
      f'{R}/field_{key}/summary.json', 0.02, [f'G_{key}_idr_real', f'G_{key}_idrfix_sg'], group=g)
for r in JOINT:   # joint-corpus models, semi-synthetic test on the 2-D lines
    J(f'eval2d_{r}', ['python', 'paperproto/eval_seis.py', '--run', f'{R}/{r}', '--tests', 'l2,w2', '--tag', 'eval_2d'],
      f'{R}/{r}/eval_2d/summary.json', 0.02, group='G_l2')

extra = HERE / 'jobs_kaggle_extra.json'   # anything else, same format
if extra.exists():
    jobs += json.loads(extra.read_text())

if __name__ == '__main__':
    (HERE / 'jobs_kaggle.json').write_text(json.dumps(jobs, indent=1) + '\n')
    n = sum('pp_train.py' in ' '.join(j['cmd']) for j in jobs)
    print(len(jobs), 'jobs,', n, 'training runs')
