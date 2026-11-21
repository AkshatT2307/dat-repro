"""Summarize fixed-step experiments, keeping final-step and best-validation results separate."""
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/idr-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent

def main():
    base=ROOT/'runs/steps50k_b4_128';out=ROOT/'reports/steps50k_b4_128';out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    rows=[];lines=['# Batch 4 / 50,000-step experiment results','',
        'All models trained for exactly 50,000 optimizer updates at 128x128. No early stopping.',
        'Final-step scores are the primary fixed-budget comparison; best-validation scores are reported separately.', '',
        '| Dataset | Method | Updates | Best validation L1 | Best step |','|---|---|---:|---:|---:|']
    for ax,(dataset,method) in zip(axes.flat,[(d,m) for d in ['imagenet','marmousi2'] for m in ['idr','supervised']]):
        folder=base/dataset/method;s=json.loads((folder/'training_summary.json').read_text())
        assert s['completed_steps']==50000
        # Deduplicate a possible replay of validation after a process interruption.
        records={r['step']:r for r in map(json.loads,(folder/'validation.jsonl').read_text().splitlines())}
        vals=[records[k] for k in sorted(records)]
        ax.plot([r['step'] for r in vals],[r['loss'] for r in vals],'-o',ms=3)
        ax.axvline(25000,color='gray',ls='--');ax.axvline(40000,color='gray',ls='--')
        ax.scatter([s['best_step']],[s['best_loss']],marker='*',s=120,color='red',zorder=5)
        ax.set(title=f'{dataset}: {method}',xlabel='Optimizer updates',ylabel='Validation L1');ax.grid(alpha=.2)
        lines.append(f"| {dataset} | {method} | {s['completed_steps']} | {s['best_loss']:.6f} | {s['best_step']} |")
        for selection in ['final','best']:
            data=json.loads((folder/f'evaluation_{selection}/summary.json').read_text())
            for result in data['results']:
                rows.append(dict(training_dataset=dataset,method=method,selection=selection,**result))
    fig.savefig(out/'validation_curves.png',dpi=180);fig.savefig(out/'validation_curves.pdf');plt.close(fig)
    (out/'evaluation_results.json').write_text(json.dumps(rows,indent=2)+'\n')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'evaluation_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
    lines+=['','![Validation curves](validation_curves.png)','','[All evaluation scores](evaluation_results.csv)',
            '', 'ImageNet: Kodak/BSDS300 PSNR and SSIM at sigma 25/50. Seismic: test-region SNR, RMSE and correlation at sigma .1/.2/.4.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(out/'RESULTS.md')

if __name__=='__main__':main()
