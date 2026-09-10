import statistics
import unittest
from unittest import mock

import torch

import utils.maze as maze_data
from utils.maze import ACTIONS, build_groups, fixture_manifest, partition_ids, score_path, shortest_actions
from utils.model import initialized_model, rollout
from utils.runner import CONFIGS, SEEDS, canonical_arrays, evaluate, seed_sets, train_model


class FixedPreferencePolicy(torch.nn.Module):
    def forward(self, inputs):
        return torch.tensor([[4.0, 3.0, 2.0, 1.0]]).repeat(len(inputs), 1)


class MazeTests(unittest.TestCase):
    def test_provenance_digests_schema_and_all_supplied_paths(self):
        manifest = fixture_manifest()
        self.assertEqual(manifest["upstream"]["revision"], "9b9ed56991cb045ba4227d9120dad337085db439")
        self.assertEqual(manifest["upstream"]["source_file"], "main_1.3M.jsonl")
        for partition in manifest["partitions"]:
            prompt_ids = partition_ids(partition)
            self.assertEqual(len(prompt_ids), manifest["partitions"][partition]["rows"])
            # Do not materialize all alternative-path tensors for the enlarged pool at once.
            for group in (build_groups([prompt_id])[0] for prompt_id in prompt_ids):
                self.assertEqual(len(group["grid"]), 17)
                self.assertEqual(len(group["canonical"]), group["L_star"])
                self.assertEqual(group["ub"], 60)
                self.assertEqual(len(group["alternatives"]), group["n_samples"])
                self.assertTrue(all(score_path(group, path)[1] == "success"
                                    for path in group["alternatives"]))

    def test_prompt_partitions_are_disjoint(self):
        partitions = [set(partition_ids(name)) for name in fixture_manifest()["partitions"]]
        self.assertEqual(sum(map(len, partitions)), len(set().union(*partitions)))

    def test_runtime_fixture_loading_needs_no_network(self):
        maze_data.fixture_manifest.cache_clear()
        maze_data._rows.cache_clear()
        with mock.patch("socket.create_connection", side_effect=AssertionError("network used")):
            self.assertEqual(len(build_groups(partition_ids("public_eval")[:1])), 1)

    def test_canonical_bfs_and_scoring_use_dataset_bound(self):
        maze = build_groups(partition_ids("public_eval")[:1])[0]
        shortest = shortest_actions(maze)
        self.assertEqual(len(shortest), maze["L_star"])
        self.assertEqual(score_path(maze, shortest), (1.0, "success"))
        self.assertEqual(score_path(maze, [99])[1], "malformed")
        self.assertEqual(score_path(maze, [])[1], "timeout")
        inefficient = max(maze["alternatives"], key=len)
        self.assertAlmostEqual(score_path(maze, inefficient)[0],
                               max(0, (60 - len(inefficient)) / (60 - maze["L_star"])))

    def test_catalog_sources_derive_from_real_group(self):
        group = build_groups(partition_ids("public_train")[:1])[0]
        self.assertTrue(group["canonical_examples"])
        self.assertTrue(group["alternative_examples"])
        self.assertTrue(group["failure_examples"])
        self.assertTrue(group["recovery_examples"])

    def test_packaged_failures_match_fixed_baseline_inference(self):
        group = build_groups(partition_ids("private_train")[:1])[0]
        expected = rollout(initialized_model(), group, torch.device("cpu"))
        self.assertEqual(group["failure"], expected)

    def test_generation_masks_walls_and_revisited_cells(self):
        maze = build_groups(partition_ids("public_eval")[:1])[0]
        actions = rollout(FixedPreferencePolicy(), maze, torch.device("cpu"))
        point, visited = maze["start"], {maze["start"]}
        for action in actions:
            dr, dc = ACTIONS[action]
            point = point[0] + dr, point[1] + dc
            self.assertFalse(maze["grid"][point[0]][point[1]])
            self.assertNotIn(point, visited)
            visited.add(point)

    def test_three_run_gold_baseline_has_positive_mean_quality(self):
        training_runs, evaluation_runs = seed_sets("full", 1_000_000)
        qualities = []
        for run_index, run_seed in enumerate(SEEDS):
            training = build_groups(training_runs[run_index])
            evaluation = build_groups(evaluation_runs[run_index])
            model = train_model(initialized_model(), canonical_arrays(training),
                                {**CONFIGS["full"], "steps": 40, "seed": run_seed},
                                torch.device("cpu"))
            qualities.append(evaluate(model, evaluation, torch.device("cpu"))["quality"])
        self.assertGreater(statistics.mean(qualities), 0)


if __name__ == "__main__":
    unittest.main()
