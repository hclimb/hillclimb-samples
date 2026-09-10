import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.profiles import make_manifest
from utils.protocol import PUBLIC_SEEDS, REPETITIONS, TIMED_CALLS, WARMUP_CALLS
from verifiers.test_benchmark import BenchmarkTests
from verifiers.test_metrics import MetricTests
from verifiers.test_worker import WorkerTests
from verifiers.test_public_checkpoint import PublicCheckpointTests


ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def test_build_retains_early_error(self):
        import os
        from verifiers.build import run
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / 'build.log'
            command = [sys.executable, '-c',
                       "print('error: original compiler diagnostic'); print('x' * 20000); exit(1)"]
            with self.assertRaisesRegex(RuntimeError, 'original compiler diagnostic'):
                run(command, ROOT, os.environ, log)
            self.assertGreater(log.stat().st_size, 20000)
            self.assertIn('original compiler diagnostic', log.read_text())

    def test_manifest(self):
        manifest = json.loads((ROOT / 'utils/manifest.json').read_text())
        self.assertEqual(manifest, make_manifest())
        self.assertEqual(manifest['repetitions'], REPETITIONS)
        self.assertEqual(len(set(PUBLIC_SEEDS)), 9)
        self.assertEqual(len(manifest['cases']), 23)
        self.assertEqual(sum(case['scored'] for case in manifest['cases']), 15)
        self.assertEqual(sum(case['throughput_weight'] for case in manifest['cases']), 1)
        self.assertEqual(manifest['warmup_calls'], WARMUP_CALLS)
        self.assertEqual(manifest['timed_calls'], TIMED_CALLS)
        for case in manifest['cases']:
            for scope in case['scopes']:
                self.assertEqual(scope['page_stride'] % 576, 0)
                self.assertEqual(scope['capacity'] % (4 * scope['page']), 0)
                self.assertEqual(len(scope['lengths']), case['batch'])
        for case in manifest['cases'][12:14]:
            self.assertEqual((case['batch'], case['queries']), (148, 2))
            self.assertEqual(case['scopes'][0]['topk'], 16384)

    def test_fixed_scored_setting(self):
        cases = make_manifest()['cases']
        self.assertEqual([case['name'] for case in cases if case['throughput_weight'] > 0],
                         ['roofline-b128-q3-h128-k1152-c32768'])
        fixed = cases[-1]
        self.assertEqual((fixed['batch'], fixed['queries'], fixed['heads']), (128, 3, 128))
        self.assertEqual(fixed['sink_categories'], [0] * 128)
        self.assertEqual([(scope['topk'], scope['page']) for scope in fixed['scopes']],
                         [(128, 256), (1024, 64)])
        for scope in fixed['scopes']:
            self.assertEqual(scope['capacity'], 32768)
            self.assertEqual(scope['lengths'], [32768] * 128)
            self.assertEqual(scope['valid_counts'], [[scope['topk']] * 3 for _ in range(128)])

    def test_child_failure_diagnostic(self):
        from utils.process import execute
        with self.assertRaisesRegex(RuntimeError, 'underlying child failure'):
            execute([sys.executable, '-c', "raise RuntimeError('underlying child failure')"], ROOT, 5)

    def test_repetition_processes(self):
        from utils.process import repetitions
        observed = []

        def execute(command, directory, timeout):
            seed = int(command[command.index('--seed') + 1])
            observed.append((seed, directory, timeout))
            Path(command[command.index('--output') + 1]).write_text(json.dumps({'seed': seed}))

        with patch('utils.process.execute', side_effect=execute):
            reports = repetitions(ROOT, PUBLIC_SEEDS)
        self.assertEqual([report['seed'] for report in reports], list(PUBLIC_SEEDS))
        self.assertEqual(len({directory for seed, directory, timeout in observed}), REPETITIONS)
        self.assertTrue(all(timeout == 180 for seed, directory, timeout in observed))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch unavailable; numerical tests not executed')
class NumericalTests(unittest.TestCase):
    def test_result_arity(self):
        import torch
        from utils.numerics import compare
        manifest = make_manifest()
        tolerances = [manifest['output_tolerance'], manifest['lse_tolerance']]
        expected = (torch.ones(2, dtype=torch.bfloat16), torch.ones(2))
        self.assertEqual(set(compare(expected, expected, tolerances)), {'output', 'lse'})
        for answer in [(), expected[:1], (*expected, expected[0])]:
            with self.subTest(length=len(answer)), self.assertRaises(ValueError):
                compare(answer, expected, tolerances)

    def test_malformed_candidate_gets_zero_reward(self):
        from utils.process import execute
        from verifiers.benchmark import evaluate
        manifest = make_manifest()
        for answer in ['()', '(output,)', '(output, lse, output)']:
            with self.subTest(answer=answer), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                command = [sys.executable, '-c',
                           'import torch; from utils.numerics import compare; '
                           'output = torch.ones(2, dtype=torch.bfloat16); lse = torch.ones(2); '
                           f'compare({answer}, (output, lse), '
                           f'{[manifest["output_tolerance"], manifest["lse_tolerance"]]!r})']

                def repetitions(implementation, seeds, case, collected):
                    execute(command, ROOT, 30)

                with patch('verifiers.benchmark.repetitions', side_effect=repetitions), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    reward = evaluate(directory, PUBLIC_SEEDS, directory / 'result')
                self.assertEqual(reward, {'valid': 0, 'reward': 0})
                self.assertEqual(json.loads((directory / 'result/reward.json').read_text()), reward)
                diagnostics = json.loads((directory / 'result/diagnostics.json').read_text())
                self.assertFalse(diagnostics['evaluator_failure'])
                self.assertIn('ValueError', diagnostics['error'])
                self.assertIn('unpack', diagnostics['error'])

    def test_oracle_and_mutants(self):
        import torch
        from utils.native_quant import dequantize_k_cache, FP8KVCacheLayout
        from utils.numerics import compare
        from utils.workload import build_inputs, replace_values_in_place
        from verifiers.reference import decode_selected, reference
        torch.set_num_threads(2)
        manifest = make_manifest()
        tolerances = [manifest['output_tolerance'], manifest['lse_tolerance']]
        for case in manifest['cases'][14:22]:
            inputs = build_inputs(case, 101, 103, device='cpu')
            answer = reference(inputs)
            compare(answer, answer, tolerances)
            self.assertTrue(torch.isfinite(answer[0]).all())
            self.assertTrue(torch.isposinf(answer[1][0]).all())
            self.assertEqual(answer[0][0].count_nonzero(), 0)
            scope = inputs['scopes'][0]
            valid = scope['indices'][2, 0, :283].long()
            decoded = decode_selected(scope['cache'], valid)
            native = dequantize_k_cache(scope['cache'], FP8KVCacheLayout.MODEL1_FP8Sparse)
            self.assertTrue(torch.equal(decoded, native.view(-1, 512)[valid].float()))
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(AssertionError):
                    compare((torch.zeros_like(answer[0]), answer[1]), answer, tolerances)
                saved = scope['indices'].clone()
                scope['indices'][scope['indices'] >= 0] = valid[0].to(torch.int32)
                wrong = reference(inputs)
                with self.assertRaises(AssertionError):
                    compare(wrong, answer, tolerances)
                scope['indices'].copy_(saved)
                replacement = build_inputs(case, 101, 107, device='cpu')
                pointers = [inputs['q'].data_ptr(), scope['cache'].data_ptr()]
                replace_values_in_place(inputs, replacement)
                self.assertEqual(pointers, [inputs['q'].data_ptr(), scope['cache'].data_ptr()])
                changed = reference(inputs)
                with self.assertRaises(AssertionError):
                    compare(answer, changed, tolerances)

    def test_nonfinite_masks(self):
        import torch
        from utils.numerics import compare
        tolerances = [[1e-3, 2.01 / 128, 5e-6], [1e-6, 8.01 / 65536, 1e-7]]
        reference = (torch.zeros(2, dtype=torch.bfloat16), torch.tensor([float('inf'), 1.0]))
        compare(reference, reference, tolerances)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(AssertionError):
            compare((reference[0], torch.tensor([-float('inf'), 1.0])), reference, tolerances)

    def test_public_private_index_profiles(self):
        import torch
        from utils.workload import logical_indices
        torch.set_num_threads(2)
        for case in make_manifest()['cases']:
            for scope_id, scope in enumerate(case['scopes']):
                for seed in [771731, 181081]:
                    torch.manual_seed(seed)
                    indices = logical_indices(case, scope, scope_id)
                    valid = indices >= 0
                    if scope['topk_lengths'] is not None:
                        lengths = torch.tensor(scope['topk_lengths']).view(-1, 1, 1)
                        valid &= torch.arange(scope['topk']).view(1, 1, -1) < lengths
                    self.assertEqual(valid.sum(-1).tolist(), scope['valid_counts'])

    def test_uniform_attention_reference(self):
        import torch
        from utils.workload import build_inputs
        from verifiers.reference import decode_selected, reference
        case = make_manifest()['cases'][14]
        inputs = build_inputs(case, 191, 193, device='cpu')
        inputs['q'].zero_()
        output, lse = reference(inputs)
        scope = inputs['scopes'][0]
        selected = scope['indices'][1, 0]
        values = decode_selected(scope['cache'], selected[selected >= 0])
        mean = values.mean(0).to(torch.bfloat16)
        torch.testing.assert_close(output[1, 0, 0], mean, rtol=2.01 / 128, atol=1e-3)
        self.assertAlmostEqual(float(lse[1, 0, 0]), float(torch.tensor(7.).log()), places=6)


if __name__ == '__main__':
    unittest.main()
