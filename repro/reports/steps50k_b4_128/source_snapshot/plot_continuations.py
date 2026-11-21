"""Plot original plus continued validation losses for the four requested models."""
import csv
import json
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
os.environ.setdefault('MPLCONFIGDIR','/tmp/idr-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parent
RUNS=[(d,m,b) for d,b in [('imagenet','b192_e20'),('marmousi2','b128_e30')] for m in ['idr','supervised']]
OUT=ROOT/'reports/continuation_curves'


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    series=[]; table=[]; csvrows=[]
    for dataset,method,budget in RUNS:
        original=ROOT/'runs/epoch128'/dataset/method/budget
        continued=ROOT/'runs/continuation5'/dataset/method/budget
        old=records(original/'validation.jsonl'); new=records(continued/'validation.jsonl')
        before=json.loads((original/'training_summary.json').read_text())
        after=json.loads((continued/'training_summary.json').read_text())
        provenance=json.loads((continued/'continuation.json').read_text())
        evaluation=json.loads((continued/'evaluation/summary.json').read_text())
        assert new[0]['epoch']==old[-1]['epoch']+1
        assert new[-1]['epoch']==after['completed_epochs']
        title=f"{'ImageNet' if dataset=='imagenet' else 'Marmousi2'} — {'Fast IDR' if method=='idr' else 'Supervised'} (batch {provenance['batch']})"
        series.append((title,old,new,after,provenance))
        gain=100*(before['best_loss']-after['best_loss'])/before['best_loss']
        table.append((title,before,after,gain,evaluation,continued))
        for phase,rows in [('original',old),('continuation',new)]:
            csvrows.extend(dict(dataset=dataset,method=method,batch=provenance['batch'],phase=phase,
                                epoch=r['epoch'],optimizer_step=r['step'],validation_l1=r['loss']) for r in rows)
    for zoom in (False,True):
        fig,axes=plt.subplots(2,2,figsize=(12,8.5),constrained_layout=True)
        for ax,(title,old,new,summary,provenance) in zip(axes.flat,series):
            ax.plot([r['epoch'] for r in old],[r['loss'] for r in old],'-o',ms=3,color='#2166ac',label='Original training')
            bridge=old[-1:]+new
            ax.plot([r['epoch'] for r in bridge],[r['loss'] for r in bridge],'-o',ms=4,color='#d97706',label='Continuation')
            ax.axvline(provenance['start_epoch']+.5,color='gray',ls='--',lw=1,label='Continuation starts')
            ax.scatter([summary['best_epoch']],[summary['best_loss']],s=130,marker='*',color='#b91c1c',zorder=5,
                       label=f"Best {summary['best_loss']:.5f}, epoch {summary['best_epoch']}")
            if zoom:
                start=max(1,provenance['start_epoch']-5)
                visible=[r['loss'] for r in old+new if r['epoch']>=start]
                pad=max(.0002,(max(visible)-min(visible))*.2)
                ax.set_xlim(start-.3,new[-1]['epoch']+.5); ax.set_ylim(min(visible)-pad,max(visible)+pad)
            ax.set_title(title,fontsize=10);ax.set_xlabel('Completed epoch');ax.set_ylabel('Validation mean L1 ↓')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle('Validation loss before and after checkpoint continuation'+(' — late epochs' if zoom else ''),fontsize=14)
        fig.supxlabel('Raw epoch losses; fixed validation samples/noise. ImageNet and seismic have different amplitude scales.',fontsize=9)
        stem='four_models_late_epochs' if zoom else 'four_models_full_history'
        fig.savefig(OUT/f'{stem}.png',dpi=180);fig.savefig(OUT/f'{stem}.pdf');plt.close(fig)
    with (OUT/'validation_losses.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    lines=['# Continuation results: four requested models','',
           'Generated: '+datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(timespec='seconds'),'',
           'ImageNet batch 192 and Marmousi2 batch 128; Fast IDR and supervised for each.',
           'Continued from latest full checkpoints for up to five additional epochs, with unchanged batch/data and LR 7.5e-5.',
           'Fresh early-stopping patience; original best models retained unless validation improved.','',
           '| Model | Previous best L1 | Best after continuation | Improvement | Best epoch | Last epoch |',
           '|---|---:|---:|---:|---:|---:|']
    for title,before,after,gain,evaluation,folder in table:
        lines.append(f"| {title} | {before['best_loss']:.6f} | {after['best_loss']:.6f} | {gain:.2f}% | {after['best_epoch']} | {after['completed_epochs']} |")
    lines += ['', 'Lower validation L1 is better. Best-after values include the original best; improvements are descriptive, not statistical significance.', '',
              '![Full validation histories](reports/continuation_curves/four_models_full_history.png)','',
              '![Late epochs](reports/continuation_curves/four_models_late_epochs.png)','',
              '[Full-history PDF](reports/continuation_curves/four_models_full_history.pdf) · [Raw CSV](reports/continuation_curves/validation_losses.csv)', '',
              '## Final evaluation', '', 'These scores do not influence checkpoint selection.', '']
    for title,before,after,gain,evaluation,folder in table:
        lines += [f'### {title}', '', f"Best epoch: {after['best_epoch']}; completed epoch: {after['completed_epochs']}; early stopped: {after['stopped']}.", '',
                  '```json',json.dumps(evaluation['results'],indent=2),'```','']
    (ROOT/'CONTINUATION_RESULTS.md').write_text('\n'.join(lines)+'\n')
    for filename in ('CONTINUATION_STATUS.md','OVERNIGHT_STATUS.md'):
        path=ROOT/filename
        text=path.read_text()
        if '[Completed continuation results]' not in text:
            path.write_text(text+'\n[Completed continuation results](CONTINUATION_RESULTS.md): four-model loss curves, validation comparison and final evaluation.\n')
    print('Updated CONTINUATION_RESULTS.md, status reports, four-model plots and CSV.',flush=True)


if __name__=='__main__':
    main()
