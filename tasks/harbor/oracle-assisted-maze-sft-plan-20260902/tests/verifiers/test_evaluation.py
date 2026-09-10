import copy
import unittest
from unittest import mock

import numpy as np
import torch

from utils import maze, model, runner


class PreferencePolicy(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))
        self.batches = []

    def forward(self, inputs):
        self.batches.append(len(inputs))
        return self.values.expand(len(inputs), -1).clone()


class BatchSensitivePolicy(PreferencePolicy):
    def forward(self, inputs):
        logits = super().forward(inputs)
        if len(inputs) == 1:
            logits[:, 1] += 0.0032
        return logits


def scalar_metrics(policy, groups):
    scores, lengths, losses = [], [], []
    outcomes = {"success": 0, "collision": 0, "timeout": 0, "malformed": 0}
    with torch.inference_mode():
        for group in groups:
            path = model.rollout(policy, group, torch.device("cpu"))
            quality, outcome = maze.score_path(group, path)
            scores.append(quality)
            lengths.append(len(path))
            outcomes[outcome] += 1
            for state, target in group["canonical_examples"]:
                logits = policy(state.unsqueeze(0))
                losses.append(torch.nn.functional.cross_entropy(
                    logits, torch.tensor([target])).item())
    return {"quality": sum(scores) / len(groups),
            "success_rate": outcomes["success"] / len(groups),
            "collision_rate": outcomes["collision"] / len(groups),
            "mean_path_length": sum(lengths) / len(groups),
            "validation_nll": sum(losses) / len(losses), "outcomes": outcomes}


class EvaluationTests(unittest.TestCase):
    def test_light_groups_keep_canonical_arrays_and_fixture_metadata(self):
        for partition in ("public_train", "public_eval", "private_train", "private_eval"):
            with self.subTest(partition=partition):
                ids = maze.partition_ids(partition)[:2]
                full = maze.build_groups(ids)
                with mock.patch.object(maze, "states_for", wraps=maze.states_for) as states, \
                        mock.patch.object(maze, "failed_examples") as failed, \
                        mock.patch.object(maze, "recovery_examples") as recovery:
                    light = maze.build_groups(ids, include_catalog=False)
                self.assertEqual(states.call_count, len(ids))
                failed.assert_not_called()
                recovery.assert_not_called()
                for left, right in zip(runner.canonical_arrays(full), runner.canonical_arrays(light)):
                    np.testing.assert_array_equal(left, right)
                for left, right in zip(full, light):
                    for key in left.keys() - {"canonical_examples", "alternative_examples",
                                             "failure_examples", "recovery_examples"}:
                        self.assertEqual(left[key], right[key], key)
                    self.assertNotIn("alternative_examples", right)
                    self.assertNotIn("failure_examples", right)
                    self.assertNotIn("recovery_examples", right)

    def test_light_groups_still_validate_unused_alternative_paths(self):
        prompt = maze.partition_ids("public_eval")[0]
        row = copy.deepcopy(maze._rows("public_eval")[prompt])
        row["samples"][0]["L"] += 1
        for include_catalog in (True, False):
            with self.subTest(include_catalog=include_catalog), \
                    self.assertRaisesRegex(RuntimeError, "sample length"):
                maze._group(row, include_catalog=include_catalog)

    def test_light_groups_skip_catalog_tensors_with_a_custom_failure_policy(self):
        with mock.patch.object(maze, "frozen_failure", return_value=[]) as failure, \
                mock.patch.object(maze, "failed_examples") as failed, \
                mock.patch.object(maze, "recovery_examples") as recovery:
            groups = maze.build_groups(maze.partition_ids("public_train")[:2],
                                       object(), include_catalog=False)
        self.assertEqual(failure.call_count, 2)
        failed.assert_not_called()
        recovery.assert_not_called()
        self.assertEqual([group["failure"] for group in groups], [[], []])

    def test_batched_rollout_matches_scalar_paths_and_does_not_mutate_groups(self):
        groups = maze.build_groups(maze.partition_ids("public_eval")[:6], include_catalog=False)
        policies = [PreferencePolicy([4, 3, 2, 1]), PreferencePolicy([1, 1, 1, 1]),
                    PreferencePolicy([float("-inf")] * 4),
                    PreferencePolicy([float("nan"), 3, 2, 1]),
                    BatchSensitivePolicy([0, 2, 2.0001, -2]), model.initialized_model()]
        for policy in policies:
            for limit in (0, 1, 60):
                with self.subTest(policy=type(policy).__name__, limit=limit):
                    expected = [model.rollout(policy, group, torch.device("cpu"), limit)
                                for group in groups]
                    actual = model.rollout_batched(policy, groups, torch.device("cpu"), limit)
                    self.assertEqual(actual, expected)
                    self.assertEqual(model.rollout_batched(policy, groups, torch.device("cpu"), limit),
                                     expected)
        self.assertEqual(model.rollout_batched(policies[0], [], torch.device("cpu")), [])

    def test_well_separated_choices_need_no_single_maze_rechecks(self):
        groups = maze.build_groups(maze.partition_ids("public_eval")[:6], include_catalog=False)
        policy = PreferencePolicy([4, 3, 2, 1])
        model.rollout_batched(policy, groups, torch.device("cpu"), upper_bound=1)
        self.assertEqual(policy.batches, [len(groups)])

    def test_batched_metrics_match_scalar_reference(self):
        groups = maze.build_groups(maze.partition_ids("public_eval")[:4], include_catalog=False)
        for policy in (PreferencePolicy([4, 3, 2, 1]), model.initialized_model()):
            with self.subTest(policy=type(policy).__name__):
                expected = scalar_metrics(policy, groups)
                actual = runner.evaluate(policy, groups, torch.device("cpu"))
                for key in expected.keys() - {"validation_nll"}:
                    self.assertEqual(actual[key], expected[key], key)
                self.assertAlmostEqual(actual["validation_nll"], expected["validation_nll"], places=5)

    def test_validation_nll_uses_bounded_batches(self):
        groups = maze.build_groups(maze.partition_ids("public_eval")[:16], include_catalog=False)
        policy = PreferencePolicy([4, 3, 2, 1])
        with mock.patch.object(runner, "rollout_batched", return_value=[g["canonical"] for g in groups]):
            metrics = runner.evaluate(policy, groups, torch.device("cpu"))
        count = sum(len(group["canonical_examples"]) for group in groups)
        self.assertEqual(policy.batches, [min(256, count - start) for start in range(0, count, 256)])
        self.assertEqual(metrics["quality"], 1)


if __name__ == "__main__":
    unittest.main()
