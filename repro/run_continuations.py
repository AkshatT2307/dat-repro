"""Continue all six original runs sequentially, preserving each original configuration."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from continue_training import prepare, ROOT

RUNS = [(dataset, method, budget) for dataset, budget in
        [('imagenet', 'b192_e20'), ('marmousi2', 'b192_e20'), ('marmousi2', 'b128_e30')]
        for method in ('idr', 'supervised')]
status = []
report = ROOT/'CONTINUATION_STATUS.md'


def update():
    lines = ['# Five additional epochs per model', '',
             'Updated: '+datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(timespec='seconds'), '',
             'Same batch, data, optimizer and teacher; original LR schedule retained. Fresh patience; previous best retained.', '',
             '| Run | Status |', '|---|---|']
    lines += [f'| {name} | {value} |' for name, value in status]
    lines += ['', 'Outputs: runs/continuation5/DATASET/METHOD/ORIGINAL_BUDGET/.',
              'Each continuation.json records its parent, start epoch, LR and settings.',
              'Validation logs contain only additional epochs. Best weights may remain the parent best.']
    report.write_text('\n'.join(lines)+'\n')


for dataset, method, budget in RUNS:
    name = f'{dataset}/{method}/{budget}'
    source = ROOT/'runs/epoch128'/name
    out = ROOT/'runs/continuation5'/name
    status.append((name, 'running'))
    update()
    try:
        if (out/'resume_command.json').exists():
            provenance = json.loads((out/'continuation.json').read_text())
            if provenance['source'] != str(source.resolve()) or provenance['additional_epochs'] != 5:
                raise ValueError('Existing continuation does not match requested source/budget')
            cmd = json.loads((out/'resume_command.json').read_text())
        else:
            cmd, provenance = prepare(source, out, 5)
        print(f'\nCONTINUING {name}: {json.dumps(provenance)}', flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        if dataset == 'imagenet':
            cmd = [sys.executable, 'evaluate.py', '--weights', str(out/'best_weights.pth'), '--output', str(out/'evaluation')]
        else:
            cmd = [sys.executable, 'evaluate_seismic.py', '--weights', str(out/'best_weights.pth'),
                   '--data', 'data/marmousi2', '--split', 'test', '--output', str(out/'evaluation')]
        subprocess.run(cmd, cwd=ROOT, check=True)
        summary = json.loads((out/'training_summary.json').read_text())
        status[-1] = (name, f"complete; epoch {summary['completed_epochs']}; best L1 {summary['best_loss']:.6f} at epoch {summary['best_epoch']}")
    except Exception as error:
        status[-1] = (name, f'FAILED: {error}')
        print(f'FAILED {name}: {error}', flush=True)
    update()
print('Continuation queue finished; see CONTINUATION_STATUS.md', flush=True)
