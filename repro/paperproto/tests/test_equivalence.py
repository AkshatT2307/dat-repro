"""Our Model.step must reproduce upstream DenoiseBase_fastIDR / _n2c / _n2n bit-for-bit (Gaussian)."""
import sys, types
from pathlib import Path
R = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(R / "paperproto"), str(R)]
import torch, numpy as np, random
import gaussian
from trainer.base_trainer import DenoiseBase_n2n
import pp_train as T

def seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)

def run(method, up_method):
    a = types.SimpleNamespace(domain='image', method=method, noise='g', dt=.004, schedule='paper', steps=50000)
    ours = T.Model(a, 3, 'cpu')
    if up_method == 'n2n':
        up = gaussian.make_model('supervised', 'paper', 3, True, 50, 'cpu')
        up.__class__ = type('N2N', (gaussian.DriverMixin, DenoiseBase_n2n), {})
    else:
        up = gaussian.make_model(up_method, 'paper', 3, True, 50, 'cpu')
    ours.net.load_state_dict(up.net.state_dict())
    ours.opt = torch.optim.Adam(ours.net.parameters(), lr=3e-4)
    up.optimizer = torch.optim.Adam(up.net.parameters(), lr=3e-4)
    clean = torch.rand(4, 3, 64, 64)
    for step, epoch in enumerate([0, 0, 1, 1, 2]):
        seed(step); lo = ours.step(clean, None, epoch)
        seed(step); up.epoch = epoch; up.preprocess({'gt': clean.permute(0, 2, 3, 1)}); lu = up.optimize_parameters()
        assert torch.equal(lo, lu), (method, step, lo.item(), lu.item())
    for p, q in zip(ours.net.parameters(), up.net.parameters()):
        assert torch.equal(p, q)
    print(method, 'bit-identical to upstream over 5 steps incl. 2 teacher refreshes')

torch.use_deterministic_algorithms(True)
run('idr', 'idr'); run('sup', 'supervised'); run('n2n', 'n2n')
