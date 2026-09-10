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
from utils.protocol import PUBLIC_SEEDS, REPETITIONS, STARTER_EFFICIENCY
from utils.roofline import estimate
from verifiers.benchmark import evaluate
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
        self.bad_timing = False
        self.manifest = make_manifest()
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        self.enterContext(patch('verifiers.benchmark.repetitions', side_effect=self.run_worker))
        self.build = self.enterContext(patch('verifiers.benchmark.execute', side_effect=self.run_build))

    def run_worker(self, implementation, seeds, case, reports):
        self.assertEqual(implementation, self.directory / '.flashmla-build/site')
        self.assertEqual(len(seeds), 1)
        occurrence = len(reports)
        seed = seeds[0]
        self.events.append(('candidate', seed, case))
        if self.failure and self.failure[:2] == ('candidate', occurrence):
            raise self.failure[2]
        seconds = (PUBLIC_SEEDS.index(seed) + 1) * 0.001
        cases = [item for item in self.manifest['cases'] if case is None or item['name'] == case]
        report = passing_reports(cases, 0 if self.bad_timing else seconds)[0]
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

    def test_candidate_only_build_and_complete_accounting_without_incumbent(self):
        for build in [False, True]:
            with self.subTest(build=build):
                self.events.clear()
                reward = self.evaluate(build=build)
                expected = [('build', None, None)] if build else []
                expected += [('candidate', seed, None) for seed in PUBLIC_SEEDS]
                self.assertEqual(self.events, expected)
                diagnostics = self.diagnostics()
                self.assertNotIn('baseline', diagnostics)
                self.assertEqual(len(diagnostics['candidate']), REPETITIONS)
                self.assertTrue(all(len(report['cases']) == 23 for report in diagnostics['candidate']))
                self.assertEqual(diagnostics['metric'], 'estimated_dense_bf16_roofline')
                self.assertAlmostEqual(reward['reward'],
                                       (estimate(self.manifest['cases'][-1])['ideal_seconds'] / 0.005 -
                                        STARTER_EFFICIENCY) / (1 - STARTER_EFFICIENCY))
                self.assertEqual(diagnostics['phase'], 'complete')
                self.assertEqual(diagnostics['run_index'], REPETITIONS - 1)

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
        self.bad_timing = True
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
