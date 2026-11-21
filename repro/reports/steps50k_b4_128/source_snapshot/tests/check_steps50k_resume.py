"""CPU integration checks for fixed steps, final validation and exact resume."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import train_steps50k as driver

root=Path(tempfile.mkdtemp(prefix='steps50k_check_',dir='runs'))
original_save=torch.save
captured=set()
full=root/'imagenet_idr'
targets={name:root/name for name in ['boundary','middle','pending_final']}
for p in targets.values():p.mkdir()


def capture(obj,path,*args,**kwargs):
    original_save(obj,path,*args,**kwargs)
    if Path(path)!=full/'latest.pt.tmp':return
    name=('boundary' if obj['step']==2 and obj['last_validation_step']==2 else
          'middle' if obj['step']==3 else
          'pending_final' if obj['step']==6 and obj['last_validation_step']==4 else None)
    if name and name not in captured:
        original_save(obj,targets[name]/'latest.pt')
        for artifact in ['best.pt','best_weights.pth']:
            shutil.copyfile(full/artifact,targets[name]/artifact)
        captured.add(name)


def args(dataset,method,output,resume=False):
    return SimpleNamespace(dataset=dataset,method=method,data='data/smoke' if dataset=='imagenet' else 'data/marmousi2',
        output=str(output),steps=6,workers=1,seed=2026,validation_count=2,validate_every=2,save_every=1,
        log_every=6,device='cpu',resume=resume,dry_run=False,smoke_test=True)


def load(p):return torch.load(p/'latest.pt',map_location='cpu',weights_only=False)


def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

full.mkdir();torch.save=capture
driver.train(args('imagenet','idr',full));torch.save=original_save
reference=load(full)
assert captured==set(targets)
assert reference['step']==6 and reference['epoch']==3 and reference['offset']==0
assert reference['last_validation_step']==6
for name,p in targets.items():
    driver.train(args('imagenet','idr',p,True));state=load(p)
    for key in ['net','teacher','optimizer','scheduler','step','epoch','offset','best_loss','best_step','last_validation_step']:
        assert same(reference[key],state[key]),(name,key)
    # Completed resumes must not add optimizer steps or validations.
    before=(p/'metrics.jsonl').read_text() if (p/'metrics.jsonl').exists() else ''
    driver.train(args('imagenet','idr',p,True))
    after=(p/'metrics.jsonl').read_text() if (p/'metrics.jsonl').exists() else ''
    assert before==after
for dataset,method in [('imagenet','supervised'),('marmousi2','idr'),('marmousi2','supervised')]:
    p=root/f'{dataset}_{method}';p.mkdir();driver.train(args(dataset,method,p))
    s=load(p);assert s['step']==6 and s['last_validation_step']==6
    assert len((p/'validation.jsonl').read_text().splitlines())==3
print(f'PASS: four CPU training paths; exact epoch-boundary, mid-epoch and pending-final-validation resumes. Artifacts: {root}')
