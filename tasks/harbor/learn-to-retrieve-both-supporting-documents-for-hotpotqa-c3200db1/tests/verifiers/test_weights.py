import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Tensor tests run in the compiled PyTorch image')
class WeightTests(unittest.TestCase):
    def test_invalid_tensor_child_status(self):
        from transformers import BertConfig
        from verifiers.runner import isolated_command
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / 'config.json'
            BertConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=2,
                       intermediate_size=512).to_json_file(config)
            (root / 'model.safetensors').write_bytes(b'invalid tensor archive')
            command = isolated_command(Path(__file__).resolve().parents[1], 'verifiers.inference',
                                       ['--models', str(root), '--weights', str(root / 'model.safetensors'),
                                        '--corpus', '/unused', '--queries', '/unused', '--validity',
                                        '--output', str(root / 'result.json')])
            invalid = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(invalid.returncode, 2, invalid.stderr)
            self.assertIn('InvalidWeights', invalid.stderr)
            config.unlink()
            infrastructure = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(infrastructure.returncode, 1, infrastructure.stderr)
            self.assertIn('FileNotFoundError', infrastructure.stderr)
            self.assertFalse((root / 'result.json').exists())

    def test_exact_encoder_state(self):
        import torch
        from safetensors.torch import save_file
        from transformers import BertConfig
        from utils.model import architecture, load_weights
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            BertConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=2,
                       intermediate_size=512).to_json_file(root / 'config.json')
            state = architecture(root).state_dict()
            path = root / 'model.safetensors'
            save_file(state, str(path))
            load_weights(root, path)
            alias = root / 'alias.safetensors'
            alias.symlink_to(path)
            with self.assertRaises(ValueError):
                load_weights(root, alias)
            key = next(iter(state))
            original = state[key]
            for replacement in (original[:1], original.to(torch.int32), original * float('nan')):
                state[key] = replacement.contiguous()
                save_file(state, str(path))
                with self.assertRaises(ValueError):
                    load_weights(root, path)
            state[key] = original
            state['unexpected'] = torch.zeros(1)
            save_file(state, str(path))
            with self.assertRaises(ValueError):
                load_weights(root, path)
