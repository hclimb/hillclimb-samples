import copy
import unittest

from utils.metrics import aggregate
from utils.profiles import make_manifest
from utils.protocol import REPETITIONS, STARTER_EFFICIENCY, TIMED_CALLS, WARMUP_CALLS
from utils.roofline import estimate


def numerical_report():
    return {name: dict(passed=True, max_absolute=0, distance=0) for name in ['output', 'lse']}


def passing_reports(cases, seconds=0.001):
    reports = []
    for _ in range(REPETITIONS):
        entries = {}
        for case in cases:
            item = dict(numerical=numerical_report(), same_storage=numerical_report())
            if case['throughput_weight'] > 0:
                item.update(seconds=seconds, block_seconds=seconds * TIMED_CALLS,
                            timed_calls=TIMED_CALLS, warmup_calls=WARMUP_CALLS,
                            distinct_queries=TIMED_CALLS, snapshot_copies=3,
                            timed_output_checks={str(index): numerical_report() for index in [0, 64, 127]})
            entries[case['name']] = item
        reports.append(dict(cases=entries))
    return reports


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.cases = make_manifest()['cases']
        self.name = self.cases[-1]['name']
        self.runs = passing_reports(self.cases)

    def test_roofline_formula_and_units(self):
        model = estimate(self.cases[-1])
        flops = 4 * 128 * 3 * 128 * 1152 * 512
        self.assertEqual(model['attention_flops'], flops)
        self.assertAlmostEqual(model['compute_seconds'], flops / 2.25e15)
        self.assertEqual(model['minimum_kv_bytes_per_block'], 128 * 1154 * 584)
        modeled_bytes = 128 * 3 * 128 * 512 * 4 + 128 * 3 * 128 * 4
        modeled_bytes += 128 * 3 * 1152 * 4 + 128 * 4 + 128 * 1154 * 584 / 128
        self.assertEqual(model['modeled_bytes_per_call'], modeled_bytes)
        self.assertAlmostEqual(model['memory_seconds'], modeled_bytes / 8e12)
        self.assertEqual(model['limiting_resource'], 'compute')
        self.assertEqual(model['ideal_seconds'], model['compute_seconds'])
        self.assertAlmostEqual(model['query_tokens_per_second'], 384 / model['ideal_seconds'])

    def test_absolute_reward_needs_no_baseline_and_uses_median(self):
        reward, metrics = aggregate(self.runs, self.cases)
        ideal = estimate(self.cases[-1])['ideal_seconds']
        self.assertAlmostEqual(reward['reward'], (ideal / 0.001 - STARTER_EFFICIENCY) / (1 - STARTER_EFFICIENCY))
        self.assertAlmostEqual(reward['candidate_rate'], 384000)
        self.assertNotIn('baseline_rate', reward)
        self.runs[0]['cases'][self.name].update(seconds=999, block_seconds=999 * TIMED_CALLS)
        self.assertEqual(aggregate(self.runs, self.cases)[1]['median_seconds'], 0.001)
        self.assertTrue(metrics['correctness_passed'])

    def test_estimate_is_not_clipped(self):
        ideal = estimate(self.cases[-1])['ideal_seconds']
        reward, metrics = aggregate(passing_reports(self.cases, ideal / 2), self.cases)
        self.assertAlmostEqual(reward['reward'], (2 - STARTER_EFFICIENCY) / (1 - STARTER_EFFICIENCY))
        self.assertTrue(metrics['exceeds_estimated_roofline'])

    def test_fixed_anchor_endpoints_and_negative_rewards(self):
        ideal = estimate(self.cases[-1])['ideal_seconds']
        rewards = []
        for efficiency in [STARTER_EFFICIENCY / 2, STARTER_EFFICIENCY, 1]:
            value, metrics = aggregate(passing_reports(self.cases, ideal / efficiency), self.cases)
            self.assertEqual(value['valid'], 1)
            self.assertEqual(metrics['starter_efficiency_anchor'], STARTER_EFFICIENCY)
            rewards.append(value['reward'])
        self.assertLess(rewards[0], 0)
        self.assertAlmostEqual(rewards[1], 0)
        self.assertAlmostEqual(rewards[2], 1)

    def test_missing_runs_or_cases_fail(self):
        for runs in [[], self.runs[:3], self.runs + self.runs[:1]]:
            with self.assertRaises(ValueError):
                aggregate(runs, self.cases)
        for index in [0, 14, 22]:
            broken = copy.deepcopy(self.runs)
            del broken[0]['cases'][self.cases[index]['name']]
            with self.assertRaisesRegex(ValueError, 'every requested correctness case'):
                aggregate(broken, self.cases)
        for cases in [[], self.cases * 2, self.cases[:-1]]:
            with self.assertRaises(ValueError):
                aggregate(self.runs, cases)

    def test_invalid_timing_or_protocol_fails(self):
        changes = [dict(seconds=value) for value in [0, -1, float('nan'), float('inf')]]
        changes += [dict(timed_calls=1), dict(warmup_calls=0), dict(distinct_queries=1),
                    dict(snapshot_copies=0), dict(timed_output_checks={}), dict(block_seconds=1)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                runs = copy.deepcopy(self.runs)
                runs[0]['cases'][self.name].update(change)
                aggregate(runs, self.cases)

    def test_all_correctness_gates_are_required(self):
        for case_index in [0, 14, 22]:
            for gate in ['numerical', 'same_storage']:
                with self.subTest(case=case_index, gate=gate), self.assertRaises(ValueError):
                    runs = copy.deepcopy(self.runs)
                    runs[0]['cases'][self.cases[case_index]['name']][gate]['output']['passed'] = False
                    aggregate(runs, self.cases)
        runs = copy.deepcopy(self.runs)
        runs[0]['cases'][self.name]['timed_output_checks']['127']['lse']['passed'] = False
        with self.assertRaises(ValueError):
            aggregate(runs, self.cases)

    def test_subset_is_never_official_reward(self):
        for case in [self.cases[0], self.cases[-1]]:
            reward, metrics = aggregate(passing_reports([case]), [case], diagnostic_only=True)
            self.assertEqual((reward['valid'], reward['reward']), (0, 0))
            self.assertTrue(metrics['correctness_passed'])
            self.assertTrue(metrics['diagnostic_only'])


if __name__ == '__main__':
    unittest.main()
