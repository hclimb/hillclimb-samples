from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from verifiers.runner import run, run_one


class PublicCheckpointTests(unittest.TestCase):
    def test_source_is_saved_before_training_and_weights_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            live = root / 'live'
            live.mkdir()
            (live / 'train.sh').write_text('original source')
            saved = Mock()
            events = []

            def snapshot(source):
                self.assertEqual((source / 'train.sh').read_text(), 'original source')
                events.append('source')
                (live / 'train.sh').write_text('next source')

            def keep(weights, name):
                self.assertEqual(name, 'model.safetensors')
                self.assertEqual(weights.read_bytes(), b'trained weights')
                (root / 'retained.safetensors').write_bytes(weights.read_bytes())
                events.append('weights')

            def phase(command, cwd, output, name, *args):
                if name == 'training':
                    self.assertEqual(events, ['source'])
                    self.assertEqual((cwd / 'train.sh').read_text(), 'original source')
                    destination = Path(command[command.index('--output_dir') + 1])
                    (destination / 'model.safetensors').write_bytes(b'trained weights')
                    (output / 'training.log').write_text('ok')
                    events.append('training')
                else:
                    self.assertEqual(events, ['source', 'training', 'weights'])
                    weights = Path(command[command.index('--weights') + 1])
                    self.assertEqual(weights.read_bytes(), (root / 'retained.safetensors').read_bytes())
                    (output / 'retrieval.log').write_text('valid')
                    events.append('inference')
                return dict(seconds=0.1)

            saved.checkpoint.side_effect = snapshot
            saved.keep.side_effect = keep
            with patch('verifiers.runner.check_environment'), patch('verifiers.runner.phase', side_effect=phase):
                result = run(live, live / 'train.sh', root, root, root / 'output', root,
                             smoke=True, public_checkpoint=saved)
            self.assertEqual(result['valid'], 1)
            self.assertEqual(events, ['source', 'training', 'weights', 'inference'])
            self.assertEqual((root / 'retained.safetensors').read_bytes(), b'trained weights')

    def test_private_run_has_no_public_archive_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch('verifiers.runner.snapshot'), patch('verifiers.runner.check_environment'), \
                    patch('verifiers.runner.run_one', return_value=dict(valid=1, ndcg_at10=0.446)) as candidate:
                result = run(root, root / 'train.sh', root, root, root / 'output', root)
            self.assertIsNone(candidate.call_args.args[-1])
            self.assertEqual(result['R'], 0)
