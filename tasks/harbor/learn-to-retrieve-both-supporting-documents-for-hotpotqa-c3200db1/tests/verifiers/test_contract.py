import copy
import json
import math
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from utils.oracle import reward, score
from utils.process import PhaseFailure, phase
from verifiers.runner import isolated_command, run, snapshot


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.queries = [dict(id='one', question='Question?')]
        self.gold = [dict(id='one', pos_doc_ids=[0, 1])]
        self.slices = [dict(id='one', type='bridge')]
        self.results = [dict(query_id='one', query='Question?',
                             retrieved=[dict(doc_index=index) for index in range(10)])]

    def calculate(self, results=None):
        return score(self.results if results is None else results, self.queries, self.gold, self.slices, 20)

    def test_normalization_and_rank(self):
        self.assertEqual(self.calculate()['ndcg_at10'], 1)
        self.assertEqual(self.calculate()['both_supports_at2'], 1)
        self.results[0]['retrieved'].reverse()
        expected = (1 / math.log2(10) + 1 / math.log2(11)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(self.calculate()['ndcg_at10'], expected)
        self.assertEqual(self.calculate()['both_supports_at2'], 0)

    def test_partial_and_absent(self):
        self.results[0]['retrieved'][1]['doc_index'] = 15
        self.assertAlmostEqual(self.calculate()['ndcg_at10'], 1 / (1 + 1 / math.log2(3)))
        self.results[0]['retrieved'][0]['doc_index'] = 16
        self.assertEqual(self.calculate()['ndcg_at10'], 0)

    def test_invalid_results(self):
        for invalid in (-1, 20, True, 1.0, '1', 2):
            with self.subTest(invalid=invalid):
                changed = copy.deepcopy(self.results)
                changed[0]['retrieved'][0]['doc_index'] = invalid
                with self.assertRaises(ValueError):
                    self.calculate(changed)
        with self.assertRaises(ValueError):
            self.calculate([])
        with self.assertRaises(ValueError):
            self.calculate(self.results * 2)

    def test_reward(self):
        self.assertEqual(reward(0.446), 0)
        self.assertEqual(reward(1), 1)
        self.assertAlmostEqual(reward(0), -0.446 / 0.554)
        self.assertLess(reward(0.445), 0)
        self.assertGreater(reward(0.447), 0)
        self.assertGreater(reward(0.7), reward(0.6))
        for invalid in (float('nan'), float('inf'), -0.1, 1.1):
            with self.assertRaises(ValueError):
                reward(invalid)

    def test_query_association_and_slices(self):
        self.queries.append(dict(id='two', question='Another?'))
        self.gold.append(dict(id='two', pos_doc_ids=[0, 1]))
        self.slices.append(dict(id='two', type='comparison'))
        self.results.append(dict(self.results[0], query_id='two', query='Another?'))
        self.assertEqual(self.calculate(list(reversed(self.results)))['ndcg_at10'], 1)
        self.assertEqual(self.calculate()['slices']['comparison']['both_supports_at2'], 1)
        for key, value in [('query_id', 'one'), ('query_id', 'unknown'), ('query', 'Changed')]:
            changed = copy.deepcopy(self.results)
            changed[1][key] = value
            with self.assertRaises(ValueError):
                self.calculate(changed)


class ProcessTests(unittest.TestCase):
    def test_fixed_anchor_in_candidate_only_and_paired_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('candidate', 'trusted/incumbent'):
                (root / name).mkdir(parents=True)
                (root / name / 'train_retriever.sh').write_text('exit 0')
            for baseline in (None, 0, 0.446, 0.8, 1):
                for candidate in (0, 0.446, 0.7, 1):
                    records = [dict(valid=1, ndcg_at10=candidate)]
                    if baseline is not None:
                        records.insert(0, dict(valid=1, ndcg_at10=baseline))
                    with self.subTest(baseline=baseline, candidate=candidate), \
                            patch('verifiers.runner.check_environment'), \
                            patch('verifiers.runner.run_one', side_effect=records):
                        result = run(root / 'candidate', root / 'candidate/train_retriever.sh', root, root,
                                     root / 'output', root / 'trusted', paired=baseline is not None)
                        self.assertEqual(result['R'], (candidate - 0.446) / (1 - 0.446))
                        self.assertEqual(result['valid'], 1)

    def test_paired_validity_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('candidate', 'trusted/incumbent'):
                (root / name).mkdir(parents=True)
                (root / name / 'train_retriever.sh').write_text('exit 0')
            with patch('verifiers.runner.check_environment'), \
                    patch('verifiers.runner.run_one', side_effect=[dict(valid=1, ndcg_at10=0.5), dict(valid=0)]):
                result = run(root / 'candidate', root / 'candidate/train_retriever.sh', root, root,
                             root / 'output', root / 'trusted', paired=True)
                self.assertEqual((result['valid'], result['R']), (0, 0))
            with patch('verifiers.runner.check_environment'), patch('verifiers.runner.run_one', return_value=dict(valid=0)):
                with self.assertRaises(RuntimeError):
                    run(root / 'candidate', root / 'candidate/train_retriever.sh', root, root,
                        root / 'output', root / 'trusted', paired=True)

    def test_snapshot_retains_sources_not_generated_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            (source / 'training/nested').mkdir(parents=True)
            (source / 'training/nested/helper.py').write_text('value = 4')
            for name in ('assets', 'runs', '.venv', '.cache'):
                (source / name).mkdir()
                (source / name / 'excluded').write_text('not source')
            snapshot(source, root / 'snapshot')
            (source / 'training/nested/helper.py').write_text('value = 5')
            self.assertEqual((root / 'snapshot/training/nested/helper.py').read_text(), 'value = 4')
            self.assertFalse((root / 'snapshot/assets').exists())
            self.assertFalse((root / 'snapshot/.venv').exists())

    def test_failure_excerpt_and_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(PhaseFailure) as caught:
                phase([sys.executable, '-c', 'raise RuntimeError("underlying failure")'], root, root, 'failure', 10)
            self.assertIn('underlying failure', caught.exception.detail['output_excerpt'])
            with self.assertRaises(PhaseFailure) as caught:
                phase([sys.executable, '-c', 'import time;time.sleep(30)'], root, root, 'timeout', 0.1)
            self.assertTrue(caught.exception.detail['timeout'])

    def test_isolated_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'probe.py').write_text('import sys; assert sys.flags.isolated; print("isolated")')
            result = phase(isolated_command(root, 'probe', []), root, root, 'isolation', 10)
            self.assertIn('isolated', result['output_excerpt'])
