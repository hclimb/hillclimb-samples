import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from utils import public_checkpoint
from verifiers import benchmark


class PublicCheckpointTests(unittest.TestCase):
    def test_public_workers_use_retained_package_when_live_checkout_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            live = root / 'live'
            site = live / '.flashmla-build/site'
            site.mkdir(parents=True)
            (site / 'kernel.so').write_bytes(b'tested kernel')
            (live / 'source.cu').write_text('tested source')
            output = root / 'reports'
            events = []

            def snapshot(saved, source):
                events.append('checkpoint')
                self.assertEqual((source / 'source.cu').read_text(), 'tested source')

            def evaluate(checkout, seeds, results, case, implementation):
                self.assertEqual(events, ['checkpoint'])
                (site / 'kernel.so').write_bytes(b'next build')
                (live / 'source.cu').write_text('next source')
                self.assertEqual((checkout / 'source.cu').read_text(), 'tested source')
                self.assertEqual((implementation / 'kernel.so').read_bytes(), b'tested kernel')
                self.assertIn(root / 'archive', implementation.parents)
                (results / 'reward.json').write_text('{"valid": 1, "reward": 0}')

            with patch.object(public_checkpoint, 'ARCHIVE_ROOT', root / 'archive'), \
                    patch.object(public_checkpoint.PublicRun, 'checkpoint', snapshot), \
                    patch.object(benchmark, 'evaluate', side_effect=evaluate), \
                    patch.object(sys, 'argv', ['benchmark', '--checkout', str(live), '--output', str(output)]):
                benchmark.main()
            metadata = json.loads((output / 'checkpoint.json').read_text())
            retained = Path(metadata['artifacts']['kernel']['path'])
            self.assertEqual((retained / 'kernel.so').read_bytes(), b'tested kernel')
            self.assertEqual(metadata['status'], 'finished')
            self.assertEqual(json.loads((output / 'reward.json').read_text())['valid'], 1)
