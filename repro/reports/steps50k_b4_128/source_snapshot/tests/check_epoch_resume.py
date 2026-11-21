"""CPU integration check: interrupt a logical run at a saved mid-epoch checkpoint."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import gaussian

full=Path('runs/smoke_epoch128/resume_reference')
resumed=Path('runs/smoke_epoch128/resumed')
resumed.mkdir(parents=True,exist_ok=True)
original_save=torch.save
captured=False

def save(obj,path,*args,**kwargs):
    global captured
    original_save(obj,path,*args,**kwargs)
    if isinstance(obj,dict) and obj.get('step')==4 and not captured:
        original_save(obj,resumed/'latest.pt')
        captured=True

torch.save=save
base=['gaussian.py','train','--profile','compact','--dataset','marmousi2',
      '--method','idr','--data','data/marmousi2','--samples-per-epoch','8',
      '--epochs','2','--batch','3','--workers','1','--device','cpu',
      '--allow-small-data','--save-every','2']
sys.argv=base+['--output',str(full)]
assert gaussian.main()==0
sys.argv=base+['--output',str(resumed),'--resume']
assert gaussian.main()==0
a=torch.load(full/'latest.pt',weights_only=False,map_location='cpu')
b=torch.load(resumed/'latest.pt',weights_only=False,map_location='cpu')
for name in ['net','teacher']:
    assert all(torch.equal(a[name][k],b[name][k]) for k in a[name]),name
assert a['scheduler']==b['scheduler']
assert a['step']==b['step']==6
print('PASS: mid-epoch resume reproduces full seismic IDR weights and teacher bit-for-bit.')
