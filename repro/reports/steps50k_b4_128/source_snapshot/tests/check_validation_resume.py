"""CPU integration: resume at epoch boundary and before final validation."""
from pathlib import Path
import shutil
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import gaussian

root = Path(tempfile.mkdtemp(prefix='validation_resume_', dir='runs'))
full = root/'full'
targets = {name: root/name for name in ('boundary', 'pending_validation')}
for target in targets.values():
    target.mkdir()
original_save = torch.save
captured = set()

def save(obj, path, *args, **kwargs):
    original_save(obj, path, *args, **kwargs)
    if Path(path) != full/'latest.tmp':
        return
    name = ('boundary' if obj['epoch'] == 1 and obj['offset'] == 0 else
            'pending_validation' if obj['step'] == 4 and obj['offset'] == 2 else None)
    if name and name not in captured:
        original_save(obj, targets[name]/'latest.pt')
        for artifact in ('best.pt', 'best_weights.pth'):
            shutil.copyfile(full/artifact, targets[name]/artifact)
        captured.add(name)

torch.save = save
base = ['gaussian.py', 'train', '--profile', 'compact', '--dataset', 'imagenet',
        '--method', 'idr', '--data', 'data/smoke', '--epochs', '2', '--batch', '3',
        '--workers', '1', '--device', 'cpu', '--allow-small-data', '--save-every', '2',
        '--validate', '--validation-count', '2']
sys.argv = base + ['--output', str(full)]
assert gaussian.main() == 0
assert captured == set(targets)
reference = torch.load(full/'latest.pt', weights_only=False, map_location='cpu')
for name, target in targets.items():
    sys.argv = base + ['--output', str(target), '--resume']
    assert gaussian.main() == 0
    state = torch.load(target/'latest.pt', weights_only=False, map_location='cpu')
    for key in ('net', 'teacher'):
        assert all(torch.equal(reference[key][k], state[key][k]) for k in reference[key]), (name, key)
    assert reference['scheduler'] == state['scheduler']
    assert reference['early_stopping'] == state['early_stopping']
    assert state['step'] == 4 and state['epoch'] == 2 and state['offset'] == 0
    best = torch.load(target/'best.pt', weights_only=False, map_location='cpu')
    weights = torch.load(target/'best_weights.pth', weights_only=True, map_location='cpu')
    assert all(torch.equal(best['net'][k], weights[k]) for k in weights)
print(f'PASS: epoch-boundary and pending-validation resumes match exactly; artifacts: {root}')
