"""Export the six completed validation histories without smoothing."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/idr-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'reports/validation_curves'
RUNS = [(dataset, method, budget) for dataset, budget in
        [('imagenet', 'b192_e20'), ('marmousi2', 'b192_e20'), ('marmousi2', 'b128_e30')]
        for method in ('idr', 'supervised')]


def draw(ax, dataset, method, budget, zoom=False):
    folder = ROOT/'runs/epoch128'/dataset/method/budget
    records = [json.loads(line) for line in (folder/'validation.jsonl').read_text().splitlines()]
    summary = json.loads((folder/'training_summary.json').read_text())
    config = json.loads((folder/'config.json').read_text())
    epochs = [r['epoch'] for r in records]
    losses = [r['loss'] for r in records]
    color = '#2166ac' if method == 'idr' else '#17805c'
    ax.plot(epochs, losses, '-o', color=color, lw=1.8, ms=3.5, label='Validation L1')
    best_epoch, best_loss = summary['best_epoch'], summary['best_loss']
    ax.scatter([best_epoch], [best_loss], marker='*', s=160, color='#bd3737', zorder=5,
               label=f'Best: {best_loss:.5f} (epoch {best_epoch})')
    ax.axvline(best_epoch, color='#bd3737', lw=.7, alpha=.4)
    for i, step in enumerate(config['milestones']):
        epoch = step/config['budget']['steps_per_epoch']
        if epoch <= epochs[-1]:
            ax.axvline(epoch, ls='--', color='#bd831c', lw=1,
                       label='LR halved at boundary' if i == 0 else None)
    if zoom:
        start = max(1, epochs[-1] - 9)
        ax.set_xlim(start-.3, epochs[-1]+.4)
        visible = [r['loss'] for r in records if r['epoch'] >= start]
        pad = max((max(visible)-min(visible))*.25, .0002)
        ax.set_ylim(min(visible)-pad, max(visible)+pad)
    else:
        ax.set_xlim(.5, max(20, epochs[-1])+.5)
    name = 'ImageNet' if dataset == 'imagenet' else 'Marmousi2'
    model = 'Fast IDR' if method == 'idr' else 'Supervised U-Net'
    end = 'early stopped' if summary['stopped'] else 'epoch limit reached'
    ax.set_title(f'{name} · {model}\nBatch {config["batch"]}, limit {config["epochs"]}; {end} at {epochs[-1]}', fontsize=10)
    ax.set_xlabel('Completed epoch')
    ax.set_ylabel('Validation mean L1 (lower is better)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    ax.grid(alpha=.2)
    ax.legend(fontsize=7, loc='best')
    return [dict(dataset=dataset, method=method, configuration=budget,
                 epoch=r['epoch'], optimizer_step=r['step'], validation_l1=r['loss'],
                 best_so_far=r['best_loss'], selected_checkpoint=r['epoch']==best_epoch)
            for r in records]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    rows = []
    for zoom in (False, True):
        fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
        fig.suptitle('Validation histories — six trained models' + (' · final 10 epochs' if zoom else ''), fontsize=16)
        for ax, run in zip(axes.flat, RUNS):
            data = draw(ax, *run, zoom=zoom)
            if not zoom:
                rows.extend(data)
        fig.supxlabel('Fixed held-out data and noise; no smoothing. ImageNet and seismic use different amplitude scales.', fontsize=9)
        name = 'validation_last10' if zoom else 'validation_all_six'
        fig.savefig(OUT/f'{name}.png', dpi=180)
        fig.savefig(OUT/f'{name}.pdf')
        plt.close(fig)
    for run in RUNS:
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        draw(ax, *run)
        fig.savefig(OUT/('_'.join(run)+'.png'), dpi=180)
        plt.close(fig)
    with (OUT/'validation_losses.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Exported combined plots, final-10-epoch zoom, six individual plots, and CSV to {OUT}')


if __name__ == '__main__':
    main()
