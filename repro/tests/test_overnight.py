import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import monitor_overnight as watch


class OvernightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/'logs').mkdir()
        self.patch = patch.object(watch, 'ROOT', self.root)
        self.patch.start()
        self.monitor = watch.Monitor('test')

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def completed(self, batch, epochs, losses):
        for method, loss in zip(watch.METHODS, losses):
            folder = watch.run_dir('imagenet', method, batch, epochs)
            (folder/'evaluation').mkdir(parents=True, exist_ok=True)
            (folder/'training_summary.json').write_text(json.dumps({'best_loss': loss}))
            # Generalization quality is deliberately absent; only completeness is checked.
            (folder/'evaluation/summary.json').write_text('{"results": []}')

    def test_healthy_runs_keep_configuration(self):
        self.completed(192, 20, (.04, .03))
        with patch.object(self.monitor, 'launch') as launch:
            self.monitor.finish_dataset('imagenet')
        launch.assert_not_called()
        self.assertEqual(self.monitor.selected['imagenet'], 'b192_e20')

    def test_bad_validation_triggers_one_paired_retry(self):
        self.completed(192, 20, (.12, .10))
        def retry(dataset, batch, epochs, reason):
            self.completed(batch, epochs, (.04, .03))
            return True
        with patch.object(self.monitor, 'launch', side_effect=retry) as launch:
            self.monitor.finish_dataset('imagenet')
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(launch.call_args.args[:3], ('imagenet', 128, 30))
        self.assertEqual(self.monitor.selected['imagenet'], 'b128_e30')

    def test_oom_uses_fresh_smaller_batch(self):
        folder = watch.run_dir('imagenet', 'idr', 192, 20)
        folder.mkdir(parents=True)
        (folder/'console.log').write_text('torch.cuda.OutOfMemoryError: CUDA out of memory')
        def recover(dataset, batch, epochs, reason):
            self.completed(batch, epochs, (.04, .03))
            return True
        with patch.object(self.monitor, 'launch', side_effect=recover) as launch:
            self.monitor.finish_dataset('imagenet')
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(launch.call_args.args[:3], ('imagenet', 128, 20))

    def test_failed_recovery_is_bounded(self):
        with patch.object(self.monitor, 'launch', return_value=False) as launch:
            self.monitor.finish_dataset('imagenet')
        self.assertEqual(launch.call_count, 1)
        self.assertIn('INCOMPLETE', self.monitor.selected['imagenet'])

    def test_quality_and_process_checks(self):
        bad = {'idr': {'best_loss': float('nan')}, 'supervised': {'best_loss': .03}}
        self.assertTrue(watch.quality_reasons('imagenet', bad))
        self.assertIsNotNone(watch.process_identity(os.getpid()))
        self.assertIsNone(watch.process_identity(999999999))


if __name__ == '__main__':
    unittest.main()
