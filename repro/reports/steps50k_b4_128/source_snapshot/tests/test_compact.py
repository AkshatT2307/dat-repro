"""CPU-only regression checks: do not contend with the running GPU experiment."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np
import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_profiles import PROFILES, default_output, validate_resume, make_scheduler
from dataset.imagefolder import Imagefolder, random_crop


class CompactTests(unittest.TestCase):
    def test_real_image_downsample_and_legacy_compatibility(self):
        options = dict(path=str(ROOT/'data/kodak'), crop_size=256, repeat=1,
                       mc=False, aug=False, preload=False, img_range=1.)
        paper = Imagefolder(**options)
        compact = Imagefolder(**options, output_size=128)
        image = cv2.imread(paper.metas[0])[..., ::-1].astype(float)
        np.random.seed(13)
        crop = random_crop(image, 256)
        expected = (cv2.resize(crop, (128,128), interpolation=cv2.INTER_AREA)/255).astype(np.float32)
        np.random.seed(13)
        actual = compact[0]['gt']
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (128,128,3))
        # Existing, pickled paper-run instances may not have the new attribute.
        del paper.output_size
        np.random.seed(13)
        np.testing.assert_array_equal(paper[0]['gt'], (crop/255).astype(np.float32))

    def test_exact_schedule_boundaries(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([parameter], lr=3e-4)
        scheduler = make_scheduler(optimizer, 'compact')
        used = {}
        for step in range(1,10001):
            if step in (1,5000,5001,8000,8001,10000):
                used[step] = optimizer.param_groups[0]['lr']
            optimizer.step()
            scheduler.step()
        self.assertEqual(used, {1:3e-4,5000:3e-4,5001:1.5e-4,8000:1.5e-4,8001:7.5e-5,10000:7.5e-5})
        self.assertEqual(PROFILES['paper']['milestones'], [25000,40000])

    def test_no_cross_protocol_resume(self):
        paper = dict(batch=64,manifest_sha256='same',method='idr',seed=2026,workers=4,
                     crop=256,milestones=[25000,40000])
        current = dict(paper, profile='paper',output_size=256)
        validate_resume(paper,current)  # Old checkpoints remain compatible.
        compact = dict(current, profile='compact',output_size=128,milestones=[5000,8000])
        with self.assertRaises(ValueError):
            validate_resume(paper,compact)
        self.assertNotEqual(default_output('idr','paper'),default_output('idr','compact'))

if __name__ == '__main__':
    unittest.main()
