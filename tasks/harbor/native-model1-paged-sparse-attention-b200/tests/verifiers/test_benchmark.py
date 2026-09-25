import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.profiles import make_manifest
from utils.protocol import PUBLIC_SEEDS, REPETITIONS
from utils.process import ROOT
from verifiers.benchmark import TARGET_RATIO, evaluate
from verifiers.test_metrics import passing_reports


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.output = self.directory / 'result'
        self.output.mkdir()
        (self.output / 'reward.json').write_text('{"valid": 1, "reward": 2}')
        self.events = []
        self.failure = None
        self.bad_timing = None
        self.baseline_scale = 2
        self.candidate_scale = 1
        self.manifest = make_manifest()
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        self.enterContext(patch('verifiers.benchmark.repetitions', side_effect=self.run_worker))
        self.build = self.enterContext(patch('verifiers.benchmark.execute', side_effect=self.run_build))

    def run_worker(self, implementation, seeds, case, reports):
        role = 'baseline' if implementation == ROOT / 'utils/incumbent/site' else 'candidate'
        if role == 'candidate':
            self.assertEqual(implementation, self.directory / '.flashmla-build/site')
        self.assertEqual(len(seeds), 1)
        occurrence = len(reports)
        seed = seeds[0]
        self.events.append((role, seed, case))
        if self.failure and self.failure[:2] == (role, occurrence):
            raise self.failure[2]
        seconds = (PUBLIC_SEEDS.index(seed) + 1) * 0.001
        seconds *= self.baseline_scale if role == 'baseline' else self.candidate_scale
        cases = [item for item in self.manifest['cases'] if case is None or item['name'] == case]
        report = passing_reports(cases, 0 if self.bad_timing == role else seconds)[0]
        reports.append(dict(report, seed=seed))

    def run_build(self, command, checkout, timeout):
        self.assertEqual((command, checkout, timeout), (['bash', 'solve.sh'], self.directory, 1800))
        self.events.append(('build', None, None))
        if self.failure and self.failure[:2] == ('build', 0):
            raise self.failure[2]
        return 'candidate build completed'

    def evaluate(self, build=True, case=None):
        return evaluate(self.directory, PUBLIC_SEEDS, self.output, build_candidate=build, case=case)

    def diagnostics(self):
        return json.loads((self.output / 'diagnostics.json').read_text())

    def test_paired_build_and_complete_accounting(self):
        for build in [False, True]:
            with self.subTest(build=build):
                self.events.clear()
                reward = self.evaluate(build=build)
                expected = [('build', None, None)] if build else []
                for index, seed in enumerate(PUBLIC_SEEDS):
                    order = ('baseline', 'candidate') if index % 2 == 0 else ('candidate', 'baseline')
                    expected.extend((role, seed, None) for role in order)
                self.assertEqual(self.events, expected)
                diagnostics = self.diagnostics()
                for role in ('baseline', 'candidate'):
                    self.assertEqual(len(diagnostics[role]), REPETITIONS)
                    self.assertTrue(all(len(report['cases']) == 23 for report in diagnostics[role]))
                self.assertEqual(diagnostics['metric'], 'paired_throughput_ratio')
                self.assertEqual(reward['reward'], 1)
                self.assertAlmostEqual(reward['throughput_ratio'], 2)
                self.assertAlmostEqual(reward['candidate_rate'], 384 / 0.005)
                self.assertAlmostEqual(reward['baseline_rate'], 384 / 0.010)
                self.assertEqual(diagnostics['phase'], 'complete')
                self.assertEqual(diagnostics['run_index'], REPETITIONS - 1)

    def test_speedup_tracks_measured_baseline_and_common_gpu_slowdown(self):
        for baseline, candidate, expected in [(1, 1, 1), (2, 1, 2), (1, 2, 0.5), (20, 10, 2)]:
            with self.subTest(baseline=baseline, candidate=candidate):
                self.baseline_scale, self.candidate_scale = baseline, candidate
                result = self.evaluate()
                self.assertAlmostEqual(result['throughput_ratio'], expected)
                self.assertAlmostEqual(result['reward'], min(1, max(0, ((expected - 1) / (TARGET_RATIO - 1) - .01) / .98)))

    def test_one_percent_endpoint_margins(self):
        for progress, expected in ((.005, 0), (.01, 0), (.255, .25), (.5, .5), (.99, 1), (.995, 1)):
            with self.subTest(progress=progress):
                self.baseline_scale = 1 + progress * (TARGET_RATIO - 1)
                self.candidate_scale = 1
                self.assertAlmostEqual(self.evaluate()["reward"], expected)

    def test_baseline_failures_are_infrastructure_errors_even_after_candidate_runs(self):
        for occurrence in [0, 1, REPETITIONS - 1]:
            with self.subTest(occurrence=occurrence):
                self.failure = ('baseline', occurrence, RuntimeError('baseline failure'))
                with self.assertRaisesRegex(RuntimeError, 'infrastructure'):
                    self.evaluate()
                self.assertFalse((self.output / 'reward.json').exists())
                self.assertTrue(self.diagnostics()['evaluator_failure'])
                self.assertEqual(self.diagnostics()['phase'], 'baseline')

    def test_invalid_baseline_statistic_is_an_evaluator_error(self):
        self.bad_timing = 'baseline'
        with self.assertRaisesRegex(RuntimeError, 'infrastructure'):
            self.evaluate()
        self.assertFalse((self.output / 'reward.json').exists())
        self.assertEqual(self.diagnostics()['phase'], 'baseline')

    def test_candidate_build_and_numerical_failures_are_zero(self):
        for phase, occurrence in [('build', 0), ('candidate', 0), ('candidate', REPETITIONS - 1)]:
            with self.subTest(phase=phase, occurrence=occurrence):
                self.events.clear()
                self.failure = (phase, occurrence, RuntimeError('candidate failure'))
                self.assertEqual(self.evaluate(), dict(valid=0, reward=0))
                self.assertEqual(json.loads((self.output / 'reward.json').read_text()), dict(valid=0, reward=0))
                diagnostics = self.diagnostics()
                self.assertFalse(diagnostics['evaluator_failure'])
                self.assertEqual(diagnostics['phase'], phase)
                self.assertIn('candidate failure', diagnostics['error'])
                self.assertEqual(len(diagnostics['candidate']), occurrence)

    def test_candidate_oserror_is_infrastructure_failure(self):
        self.failure = ('candidate', 1, OSError('No space left on device'))
        with self.assertRaisesRegex(RuntimeError, 'infrastructure'):
            self.evaluate()
        self.assertFalse((self.output / 'reward.json').exists())
        self.assertTrue(self.diagnostics()['evaluator_failure'])
        self.assertEqual(len(self.diagnostics()['candidate']), 1)

    def test_invalid_candidate_statistic_is_zero(self):
        self.bad_timing = 'candidate'
        self.assertEqual(self.evaluate(), dict(valid=0, reward=0))
        self.assertFalse(self.diagnostics()['evaluator_failure'])

    def test_case_subsets_cannot_skip_correctness_for_official_reward(self):
        for name in ['production-1-b2-q1', self.manifest['cases'][14]['name'],
                     self.manifest['cases'][-1]['name']]:
            with self.subTest(name=name):
                self.events.clear()
                reward = self.evaluate(case=name)
                self.assertEqual((reward['valid'], reward['reward']), (0, 0))
                diagnostics = self.diagnostics()
                self.assertTrue(diagnostics['subset_diagnostic_only'])
                self.assertTrue(diagnostics['correctness_passed'])
                self.assertTrue(all(list(report['cases']) == [name] for report in diagnostics['candidate']))

    def test_unknown_case_and_duplicate_seeds_are_setup_errors(self):
        for seeds, case in [(PUBLIC_SEEDS, 'missing'), ([PUBLIC_SEEDS[0]] * REPETITIONS, None)]:
            with self.subTest(case=case), self.assertRaisesRegex(RuntimeError, 'infrastructure'):
                evaluate(self.directory, seeds, self.output, case=case)
            self.assertFalse((self.output / 'reward.json').exists())


if __name__ == '__main__':
    unittest.main()
