import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from epoch_protocol import plan
from seismic_data import read_segy, spatial_splits, SeismicPatches, heldout_tiles
from evaluate_seismic import measures

class Tests(unittest.TestCase):
    def test_budget_preserves_exposure(self):
        for batch,steps in [(64,10166),(128,5083),(256,2548)]:
            p=plan(50000,batch,13)
            self.assertEqual(p['samples_total'],650000)
            self.assertEqual(p['steps'],steps)
        self.assertEqual(plan(50000,256,13)['milestones'],[1274,2039])

    def test_ibm_trace_decoding(self):
        header=bytearray(3600)
        header[3220:3222]=struct.pack('>H',3)
        header[3224:3226]=struct.pack('>H',1)
        th=bytearray(240);th[114:116]=struct.pack('>H',3)
        with tempfile.NamedTemporaryFile() as f:
            f.write(header+th+struct.pack('>III',0x41100000,0xc1200000,0));f.flush()
            section,_=read_segy(f.name)
        np.testing.assert_array_equal(section[:,0],[1,-2,0])

    def test_seismic_splits_and_float_patches(self):
        root=ROOT/'data/marmousi2'
        d=SeismicPatches(root,samples=8)
        p=d[0]['gt']
        self.assertEqual(p.shape,(128,128,1))
        self.assertEqual(p.dtype,np.float32)
        self.assertTrue(np.isfinite(p).all())
        splits=d.metadata['splits']
        self.assertGreaterEqual(splits['validation'][0]-splits['train'][1],128)
        self.assertGreaterEqual(splits['test'][0]-splits['validation'][1],128)
        for split in ['validation','test']:
            tiles=list(heldout_tiles(root,split))
            self.assertEqual(len(tiles),21)
            for y,x,t in tiles:
                self.assertGreaterEqual(x,splits[split][0])
                self.assertLessEqual(x+128,splits[split][1])
        original=np.load(root/'section.npy')[:,:splits['train'][1]]
        normalized=original.astype(float)/d.scale
        self.assertAlmostEqual(float(np.sqrt(np.mean(normalized**2))),1,places=6)

    def test_amplitude_metrics_no_clipping(self):
        clean=np.array([[-10.,-2.],[2.,10.]],dtype=np.float32)
        noisy=clean+1
        r=measures(clean,noisy,clean,.001)
        self.assertEqual(r['rmse_original_amplitude'],0)
        self.assertGreater(r['snr_gain_db'],100)

if __name__=='__main__': unittest.main()
