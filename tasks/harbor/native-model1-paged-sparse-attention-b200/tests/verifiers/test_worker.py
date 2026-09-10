import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, call, patch


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch unavailable; worker tests not executed')
class WorkerTests(unittest.TestCase):
    def test_sampling_initialization_before_warmup(self):
        import torch
        original_mode = torch.get_deterministic_debug_mode()
        try:
            for debug_mode in [0, 1, 2]:
                with self.subTest(debug_mode=debug_mode):
                    torch.set_deterministic_debug_mode(debug_mode)
                    self.check_sampling_initialization()
        finally:
            torch.set_deterministic_debug_mode(original_mode)

    def check_sampling_initialization(self):
        import torch
        from verifiers import worker
        timeline = Mock()
        rng_state = torch.get_rng_state()
        cuda_initialized = torch.cuda.is_initialized()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        implementation = Mock()
        if 'flash_mla' in sys.modules:
            self.addCleanup(sys.modules.__setitem__, 'flash_mla', sys.modules['flash_mla'])
        else:
            self.addCleanup(sys.modules.pop, 'flash_mla', None)
        sys.modules['flash_mla'] = implementation
        self.addCleanup(setattr, torch.backends.cuda.matmul, 'allow_tf32',
                        torch.backends.cuda.matmul.allow_tf32)
        arguments = ['worker', '--implementation', '.', '--seed', '123', '--output', 'unused.json']
        with patch.object(sys, 'argv', arguments), patch.object(sys, 'path', sys.path.copy()), \
                patch.object(worker, 'import_module', wraps=worker.import_module) as initialize, \
                patch.object(worker.signal, 'signal'), patch.object(worker.signal, 'alarm') as alarm, \
                patch.object(torch.cuda, 'get_device_name', return_value='NVIDIA B200'), \
                patch.object(torch, 'set_num_threads'), patch.object(torch, 'set_float32_matmul_precision'), \
                patch.object(worker, 'build_inputs', side_effect=RuntimeError('warmup reached')) as build:
            timeline.attach_mock(initialize, 'initialize')
            timeline.attach_mock(alarm, 'alarm')
            timeline.attach_mock(build, 'build')
            with self.assertRaisesRegex(RuntimeError, 'warmup reached'):
                worker.main()
        self.assertEqual(timeline.mock_calls, [
            call.initialize('torch._inductor.config'), call.alarm(30),
            call.build(build.call_args.args[0], 124, 125),
        ])
        for dependency in ['torch._inductor.config', 'torch._dynamo', 'sympy']:
            self.assertIn(dependency, sys.modules)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_state))
        self.assertEqual(torch.cuda.is_initialized(), cuda_initialized)
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), deterministic)
        self.assertEqual(torch.is_deterministic_algorithms_warn_only_enabled(), warn_only)
        self.assertEqual(implementation.mock_calls, [])

    def test_block_timing_waits_for_all_work_without_per_call_synchronization(self):
        from verifiers.worker import time_call
        timeline = Mock()
        operation = Mock(side_effect=['first', 'second', 'last'])
        timeline.attach_mock(operation, 'operation')
        with patch('verifiers.worker.time.perf_counter', side_effect=[10, 10.3]) as clock, \
                patch('verifiers.worker.torch.cuda.synchronize') as synchronize:
            timeline.attach_mock(synchronize, 'synchronize')
            timeline.attach_mock(clock, 'clock')
            answer, seconds = time_call(operation, 'input', 'scheduler', calls=3)
        self.assertEqual(timeline.mock_calls, [
            call.synchronize(), call.clock(), *[call.operation('input', 'scheduler')] * 3,
            call.synchronize(), call.clock(),
        ])
        self.assertEqual(answer, 'last')
        self.assertAlmostEqual(seconds, 0.1)
        for calls in [0, -1, 1.5]:
            with self.assertRaises(ValueError):
                time_call(operation, calls=calls)

    def test_warmed_path_reuses_metadata_checks_changed_values_and_timed_output(self):
        import contextlib
        import io
        import torch
        from utils.profiles import make_manifest
        from utils.protocol import TIMED_CALLS, WARMUP_CALLS
        from verifiers import sequence, worker

        def inputs(case, seed, values):
            return dict(q=torch.tensor([[[[float(values)]]]]), sink=None,
                        scopes=[dict(cache=torch.tensor([float(values * 2)]),
                                     indices=torch.tensor([0]), length=None)])

        def reference(data):
            value = data['q'] + data['scopes'][0]['cache']
            return value.to(torch.bfloat16), value

        manifest = make_manifest()
        tolerances = [manifest['output_tolerance'], manifest['lse_tolerance']]
        for timed in [False, True]:
            for mutant in [None, 'stale', 'cached_timed', 'wrong_timed', 'shared_output']:
                if not timed and mutant not in [None, 'stale']:
                    continue
                with self.subTest(timed=timed, mutant=mutant):
                    implementation, scheduler = Mock(), object()
                    implementation.get_mla_metadata.return_value = scheduler, None
                    calls = []
                    cached = None
                    shared = None

                    def native(module, data, metadata):
                        nonlocal cached, shared
                        self.assertIs(module, implementation)
                        self.assertIs(metadata, scheduler)
                        calls.append((data['q'].data_ptr(), data['q'].item(),
                                      data['scopes'][0]['cache'].data_ptr()))
                        answer = reference(data)
                        if cached is None or len(calls) == 3:
                            cached = answer
                        if mutant == 'stale' or mutant == 'cached_timed' and len(calls) > 2:
                            return cached
                        if mutant == 'wrong_timed' and len(calls) > 2 + WARMUP_CALLS:
                            return torch.zeros_like(answer[0]), answer[1]
                        if mutant == 'shared_output':
                            if shared is None:
                                shared = tuple(torch.empty_like(value) for value in answer)
                            for buffer, value in zip(shared, answer):
                                buffer.copy_(value)
                            return shared
                        return answer

                    with patch.object(worker, 'build_inputs', side_effect=inputs), \
                            patch.object(worker, 'reference', side_effect=reference), \
                            patch.object(sequence, 'reference', side_effect=reference), \
                            patch.object(worker, 'call_native', side_effect=native), \
                            patch.object(sequence, 'call_native', side_effect=native), \
                            patch.object(torch.cuda, 'synchronize'), \
                            patch.object(torch.cuda, 'empty_cache') as empty_cache, \
                            patch.object(worker.time, 'perf_counter', side_effect=[0, 0.01, 1, 1.128]), \
                            contextlib.redirect_stdout(io.StringIO()):
                        case = dict(throughput_weight=int(timed))
                        if mutant in ['stale', 'cached_timed', 'wrong_timed']:
                            with self.assertRaises(AssertionError):
                                worker.benchmark_case(implementation, case, 123, tolerances)
                        else:
                            result = worker.benchmark_case(implementation, case, 123, tolerances)
                            self.assertAlmostEqual(result['cold_seconds'], 0.01)
                            self.assertIn('same_storage', result)
                            if timed:
                                self.assertAlmostEqual(result['seconds'], 0.128 / TIMED_CALLS)
                                self.assertEqual(result['timed_calls'], TIMED_CALLS)
                                self.assertEqual(len(result['timed_output_checks']), 3)
                                self.assertEqual(len(calls), 2 + WARMUP_CALLS + TIMED_CALLS)
                                self.assertEqual(len({pointer for pointer, _, _ in calls[2:]}),
                                                 WARMUP_CALLS + TIMED_CALLS)
                            else:
                                self.assertNotIn('seconds', result)
                                self.assertEqual(len(calls), 2)
                        empty_cache.assert_not_called()
                    self.assertEqual(calls[0][0], calls[1][0])
                    self.assertNotEqual(calls[0][1], calls[1][1])
                    self.assertEqual(len({cache for _, _, cache in calls}), 1)

    def test_query_sequence_has_local_rng_and_disjoint_values(self):
        import torch
        from verifiers.sequence import query_sequence
        query = torch.zeros((1, 1, 2, 8), dtype=torch.bfloat16)
        state = torch.get_rng_state()
        first = query_sequence(query, 731, 10)
        torch.manual_seed(99)
        second = query_sequence(query, 731, 10)
        torch.set_rng_state(state)
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(first, second)))
        self.assertTrue(all(not torch.equal(first[i], first[j]) for i in range(10) for j in range(i)))
        self.assertEqual(len({item.data_ptr() for item in first}), 10)

    def test_generation_and_reference_precede_warmup_and_timing(self):
        import torch
        from utils.profiles import make_manifest
        from utils.protocol import TIMED_CALLS, WARMUP_CALLS
        from verifiers import sequence
        events = []
        data = dict(q=torch.ones((1, 1, 1, 2), dtype=torch.bfloat16))
        query_sequence = sequence.query_sequence

        def generate(*arguments):
            events.append('generate')
            return query_sequence(*arguments)

        def oracle(inputs):
            events.append('reference')
            return inputs['q'].clone(), torch.zeros(1)

        def native(implementation, inputs, scheduler):
            events.append('call')
            return inputs['q'].clone(), torch.zeros(1)

        def clock():
            events.append('clock')
            return 1 if events.count('clock') == 1 else 1.128

        manifest = make_manifest()
        with patch.object(sequence, 'query_sequence', side_effect=generate), \
                patch.object(sequence, 'reference', side_effect=oracle), \
                patch.object(sequence, 'call_native', side_effect=native), \
                patch.object(torch.cuda, 'synchronize'), \
                patch.object(sequence.time, 'perf_counter', side_effect=clock):
            sequence.benchmark_sequence(None, data, data, None, 731,
                                        [manifest['output_tolerance'], manifest['lse_tolerance']])
        self.assertEqual(events[:4], ['generate', 'reference', 'reference', 'reference'])
        clocks = [index for index, event in enumerate(events) if event == 'clock']
        self.assertEqual(clocks[0], 4 + WARMUP_CALLS)
        self.assertEqual(events[clocks[0] + 1:clocks[1]], ['call'] * TIMED_CALLS)


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    unittest.main()
