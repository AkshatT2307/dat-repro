import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from imagenet_split import split_paths
from validation import EarlyStopping, validate
from epoch_protocol import plan

class Tests(unittest.TestCase):
    def test_split_is_stable_and_disjoint(self):
        files=[str(i) for i in range(50000)]
        train,val,digest=split_paths(files)
        self.assertEqual((len(train),len(val)),(49000,1000))
        self.assertFalse(set(train)&set(val))
        self.assertEqual(len(set(train)|set(val)),50000)
        self.assertEqual(split_paths(list(reversed(files))), (train,val,digest))
        self.assertNotEqual(split_paths(files,seed=7)[2],digest)
        self.assertEqual(plan(len(train),256,20)['steps'],3840)

    def test_stopping_best_and_resume(self):
        s=EarlyStopping(patience=2,min_delta=.01)
        self.assertTrue(s.update(.2,1))
        self.assertTrue(s.update(.195,2))  # true best, but insignificant improvement
        self.assertEqual(s.bad_epochs,1)
        s=EarlyStopping(**s.state_dict())
        self.assertFalse(s.update(.21,3))
        self.assertTrue(s.stopped)
        self.assertEqual(s.best_epoch,2)
        self.assertEqual(s.best_loss,.195)
        with self.assertRaises(ValueError): s.update(float('nan'),4)

    def test_validation_repeatability_rng_and_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            files=[]
            for i in range(3):
                p=Path(folder)/f'{i}.png'
                Image.fromarray(np.full((270,300,3),64+i,dtype=np.uint8)).save(p)
                files.append(str(p))
            model=torch.nn.Identity().train()
            np.random.seed(123)
            before=np.random.get_state()
            first=validate(model,'imagenet',files,'cpu')
            after=np.random.get_state()
            self.assertTrue(model.training)
            self.assertTrue(np.array_equal(before[1],after[1]))
            self.assertEqual(first,validate(model,'imagenet',files,'cpu'))
            self.assertEqual(first['counts'],{'ImageNet-heldout/sigma25':3,'ImageNet-heldout/sigma50':3})
            # Identity denoiser should have Gaussian absolute error sqrt(2/pi)*sigma.
            self.assertAlmostEqual(first['groups']['ImageNet-heldout/sigma25'],np.sqrt(2/np.pi)*25/255,places=3)

if __name__=='__main__': unittest.main()
