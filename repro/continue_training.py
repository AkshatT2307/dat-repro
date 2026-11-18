"""Branch a completed epoch checkpoint without changing batch, optimizer, teacher or LR."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import torch
from epoch_protocol import plan

ROOT = Path(__file__).resolve().parent


def prepare(source, output, extra_epochs):
    if extra_epochs < 1:
        raise ValueError('Additional epochs must be positive')
    if output.exists():
        raise ValueError(f'Output already exists: {output}; resume the prepared run explicitly')
    state = torch.load(source/'latest.pt', map_location='cpu', weights_only=False)
    if state['offset'] != 0 or not state['metadata'].get('validate'):
        raise ValueError('Continuation requires a validated epoch-boundary checkpoint')
    if state['early_stopping']['last_epoch'] != state['epoch']:
        raise ValueError('Source checkpoint has pending validation')
    cfg = copy.deepcopy(state['metadata'])
    old_epoch = state['epoch']
    epochs = old_epoch + extra_epochs
    budget = plan(cfg['images'], cfg['batch'], epochs)
    # Preserve original absolute LR milestones; do not move or repeat past decays.
    budget['milestones'] = cfg['milestones']
    cfg.update(epochs=epochs, steps=budget['steps'], budget=budget, output=str(output), resume=True,
               lr_milestones=cfg['milestones'])
    state['metadata'] = cfg
    # A deliberate new continuation grants fresh patience while retaining the global best.
    state['early_stopping'].update(bad_epochs=0, stopped=False)
    output.mkdir(parents=True)
    for name in ('best.pt', 'best_weights.pth'):
        shutil.copyfile(source/name, output/name)
    torch.save(state, output/'latest.pt')
    provenance = dict(source=str(source.resolve()), start_epoch=old_epoch,
                      start_step=state['step'], additional_epochs=extra_epochs,
                      maximum_epoch=epochs, batch=cfg['batch'],
                      learning_rate=state['optimizer']['param_groups'][0]['lr'],
                      milestones=cfg['milestones'], patience_reset=True,
                      best_validation_preserved=state['early_stopping']['best_loss'])
    (output/'continuation.json').write_text(json.dumps(provenance, indent=2)+'\n')
    cmd = [sys.executable, str(ROOT/'gaussian.py'), 'train', '--resume', '--validate',
           '--output', str(output), '--epochs', str(epochs), '--lr-milestones',
           *map(str, cfg['milestones'])]
    for key in ('profile', 'dataset', 'method', 'data', 'batch', 'workers', 'seed',
                'samples_per_epoch', 'device', 'validation_count', 'split_seed',
                'validation_seed', 'patience', 'min_delta', 'save_every', 'log_every'):
        if key in cfg:
            cmd += ['--'+key.replace('_', '-'), str(cfg[key])]
    if cfg.get('allow_small_data'):
        cmd.append('--allow-small-data')
    (output/'resume_command.json').write_text(json.dumps(cmd, indent=2)+'\n')
    return cmd, provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--extra-epochs', type=int, default=5)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    cmd, provenance = prepare(a.source.resolve(), a.output.resolve(), a.extra_epochs)
    print(json.dumps(provenance, indent=2), flush=True)
    if not a.prepare_only:
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
