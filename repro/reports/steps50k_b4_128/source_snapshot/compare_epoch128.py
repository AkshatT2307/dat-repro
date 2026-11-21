"""Collect both method evaluations without conflating ImageNet and seismic metrics."""
import argparse
import csv
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataset',choices=['imagenet','marmousi2'],required=True)
p.add_argument('--batch',type=int,required=True);p.add_argument('--epochs',type=int,required=True)
a=p.parse_args()
base=Path('runs/epoch128')/a.dataset
rows=[]
for method in ['idr','supervised']:
    source=base/method/f'b{a.batch}_e{a.epochs}'/'evaluation/summary.json'
    data=json.loads(source.read_text())
    for result in data['results']:
        rows.append(dict(method=method,**result))
out=base/f'comparison_b{a.batch}_e{a.epochs}';out.mkdir(parents=True,exist_ok=True)
(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
with (out/'results.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(f'Combined comparison saved: {out}/results.csv')
