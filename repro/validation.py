"""Deterministic clean-reference L1 validation, separate from training RNGs."""
from dataclasses import dataclass, asdict
import math
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from seismic_data import heldout_tiles


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 1e-5
    best_loss: float = math.inf
    patience_best: float = math.inf
    best_epoch: int = 0
    bad_epochs: int = 0
    last_epoch: int = 0
    stopped: bool = False

    def update(self, loss, epoch):
        if not math.isfinite(loss):
            raise ValueError('Validation loss is nonfinite')
        if epoch <= self.last_epoch:
            raise ValueError('Validation epochs must increase')
        best = loss < self.best_loss
        if best:
            self.best_loss, self.best_epoch = loss, epoch
        if loss < self.patience_best - self.min_delta:
            self.patience_best, self.bad_epochs = loss, 0
        else:
            self.bad_epochs += 1
        self.last_epoch = epoch
        self.stopped = self.bad_epochs >= self.patience
        return best

    def state_dict(self):
        return asdict(self)


def validate(net, dataset, root, device, seed=2026):
    was_training = net.training
    net.eval()
    scores = {}
    counts = {}
    try:
        with torch.inference_mode():
            if dataset == 'imagenet':
                files = root  # Explicit held-out ImageNet filenames, never benchmark directories.
                if not files:
                    raise ValueError('Empty ImageNet validation split')
                for sigma in (25, 50):
                    rng = np.random.RandomState(seed)
                    losses = []
                    inputs, targets = [], []
                    for index,file in enumerate(files):
                        image = cv2.imread(str(file))
                        if image is None:
                            raise ValueError(f'Unreadable validation image: {file}')
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        h,w = image.shape[:2]
                        if h <= 256 or w <= 256:
                            crop = cv2.resize(image.astype(float),(256,256))
                        else:
                            y,x = (h-256)//2,(w-256)//2
                            crop = image[y:y+256,x:x+256].astype(float)
                        clean = (cv2.resize(crop,(128,128),interpolation=cv2.INTER_AREA)/255).astype(np.float32)
                        noisy = (clean+rng.normal(0,sigma/255,clean.shape)).astype(np.float32)
                        inputs.append(noisy);targets.append(clean)
                        if len(inputs)==16 or index==len(files)-1:
                            x=torch.from_numpy(np.stack(inputs)).permute(0,3,1,2).to(device)-.5
                            target=torch.from_numpy(np.stack(targets)).permute(0,3,1,2).to(device)
                            pred=net(x)+.5
                            losses.extend((pred-target).abs().mean(dim=(1,2,3)).cpu().tolist())
                            inputs,targets=[],[]
                    key=f'ImageNet-heldout/sigma{sigma}'
                    scores[key],counts[key]=float(np.mean(losses)),len(losses)
            else:
                tiles = list(heldout_tiles(root,'validation'))
                if not tiles:
                    raise ValueError('Empty seismic validation split')
                for sigma in (.1,.2,.4):
                    rng = np.random.RandomState(seed)
                    losses=[]
                    for _,_,clean in tiles:
                        noisy = (clean+rng.normal(0,sigma,clean.shape)).astype(np.float32)
                        pred = net(torch.from_numpy(noisy.copy())[None,None].to(device))
                        target = torch.from_numpy(clean.copy())[None,None].to(device)
                        losses.append(float(F.l1_loss(pred,target).item()))
                    key=f'validation/sigma{sigma}'
                    scores[key],counts[key]=float(np.mean(losses)),len(losses)
    finally:
        net.train(was_training)
    # Give each fixed validation noise level equal weight.
    return dict(loss=float(np.mean(list(scores.values()))), groups=scores, counts=counts,
                metric='float L1, no clipping/quantization; equally weighted group means')
