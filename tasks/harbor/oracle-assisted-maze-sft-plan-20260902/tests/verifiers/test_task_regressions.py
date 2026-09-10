import contextlib
import io
import json
import sys
import unittest
from unittest import mock

import torch

from utils.contract import build_catalog
from utils.maze import (ACTIONS, _rows, distance_map, failed_examples,
                        frozen_failures, partition_ids, recovery_examples)
from utils.runner import seed_sets, training_groups
from verifiers import run_public


class RecoveryTests(unittest.TestCase):
    def maze(self, cells, goal):
        grid = [[1] * 17 for _ in range(17)]
        for row, col in cells:
            grid[row][col] = 0
        return {"grid": grid, "start": (1, 1), "goal": goal}

    def test_irrecoverable_suffix_has_no_positive_or_negative_labels(self):
        maze = self.maze({(1, 1), (1, 2), (1, 3), (2, 1), (3, 1)}, (1, 3))
        failure = [2, 2]
        recovery = recovery_examples(maze, failure)
        self.assertEqual(len(recovery), 1)
        self.assertEqual(recovery[0][1], 1)
        group = {**maze, "canonical_examples": [], "alternative_examples": [],
                 "failure_examples": failed_examples(maze, failure),
                 "recovery_examples": recovery}
        _, groups = build_catalog([group])
        records = groups[0]["catalog"]
        self.assertEqual([r["kind"] for r in records], ["recovery", "failure_error"])
        self.assertEqual(records[1]["action"], 2)
        self.assertEqual(float(records[1]["state"][3].sum()), 1)

    def test_recovery_can_require_a_detour_away_from_original_shortest_path(self):
        cells = {(1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
                 (2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4)}
        maze = self.maze(cells, (1, 5))
        examples = recovery_examples(maze, [1, 1, 2, 3, 2])
        at_detour = [action for state, action in examples if state[2, 2, 2]]
        self.assertEqual(at_detour, [2])
        original = distance_map(maze["grid"], maze["goal"])
        self.assertGreater(original[(3, 2)], original[(2, 2)])

    def test_all_packaged_recovery_targets_are_legal_and_goal_reachable(self):
        checked = 0
        failures = frozen_failures()
        for partition in ("public_train", "private_train"):
            for row in _rows(partition).values():
                maze = {**row, "start": (1, 1), "goal": (15, 15)}
                examples = recovery_examples(maze, failures[row["prompt_id"]])
                for state, action in examples:
                    point = tuple(state[2].nonzero()[0].tolist())
                    visited = {tuple(x) for x in state[3].nonzero().tolist()}
                    dr, dc = ACTIONS[action]
                    target = point[0] + dr, point[1] + dc
                    self.assertNotIn(target, visited)
                    self.assertFalse(maze["grid"][target[0]][target[1]])
                    # After leaving the current cell, it too must remain blocked.
                    self.assertIn(target, distance_map(maze["grid"], maze["goal"], visited))
                    checked += 1
        self.assertGreater(checked, 0)


class PublicValidationTests(unittest.TestCase):
    def test_contract_cli_does_not_require_cuda_or_train(self):
        stdout = io.StringIO()
        with mock.patch.object(sys, "argv", ["public_test", "--mode", "contract"]), \
                mock.patch("torch.cuda.is_available", return_value=False), \
                mock.patch.object(run_public, "construct_examples") as construct, \
                mock.patch.object(run_public, "load_examples", return_value=(None, [1], None, None)), \
                mock.patch.object(run_public, "compare") as compare, \
                contextlib.redirect_stdout(stdout):
            run_public.main()
        self.assertEqual(len(construct.call_args.args[1]), 4)
        compare.assert_not_called()
        self.assertEqual(json.loads(stdout.getvalue())["valid"], 1)

    def test_public_panels_keep_full_budget_and_disjoint_holdouts(self):
        self.assertEqual(seed_sets("full", 1_000_000),
                         seed_sets("full", 1_000_000, public_fold=0))
        public_train = set(partition_ids("public_train"))
        public_eval = set(partition_ids("public_eval"))
        private = set(partition_ids("private_train")) | set(partition_ids("private_eval"))
        train_panels, eval_panels = [], []
        for panel in range(2):
            training, evaluation = seed_sets("full", 1_000_000, public_fold=panel)
            self.assertEqual((training, evaluation),
                             seed_sets("full", 1_000_000, public_fold=panel))
            self.assertEqual(len(training), 3)
            self.assertEqual(len(evaluation), 3)
            for train, evaluate in zip(training, evaluation):
                self.assertEqual(len(set(train)), 192)
                self.assertEqual(len(set(evaluate)), 96)
                self.assertTrue(set(train) <= public_train)
                self.assertTrue(set(evaluate) <= public_eval)
                self.assertTrue(set(train).isdisjoint(evaluate))
                self.assertTrue((set(train) | set(evaluate)).isdisjoint(private))
            self.assertEqual(len(set().union(*map(set, training))), 576)
            self.assertEqual(len(set().union(*map(set, evaluation))), 288)
            train_panels.extend(map(set, training))
            eval_panels.extend(map(set, evaluation))
        self.assertEqual(set.union(*train_panels), public_train)
        self.assertEqual(set.union(*eval_panels), public_eval)
        self.assertEqual(sum(map(len, train_panels)), len(public_train))
        self.assertEqual(sum(map(len, eval_panels)), len(public_eval))

    def test_private_runs_use_six_disjoint_paired_panels(self):
        training, evaluation = seed_sets("full", 2_000_000, private=True)
        self.assertEqual(len(training), 6)
        self.assertEqual(len(evaluation), 6)
        self.assertEqual(len(set().union(*map(set, training))), 1152)
        self.assertEqual(len(set().union(*map(set, evaluation))), 576)
        self.assertEqual(set().union(*map(set, training)), set(partition_ids("private_train")))
        self.assertEqual(set().union(*map(set, evaluation)), set(partition_ids("private_eval")))
        for train, evaluate in zip(training, evaluation):
            self.assertEqual((len(train), len(evaluate)), (192, 96))
            self.assertTrue(set(train).isdisjoint(evaluate))

    def test_panels_reject_quick_private_and_invalid_inputs(self):
        for mode, private, panel in (("quick", False, 0), ("full", True, 1),
                                     ("full", False, -1), ("full", False, 2)):
            with self.subTest(mode=mode, private=private, panel=panel):
                with self.assertRaises(ValueError):
                    seed_sets(mode, 1_000_000, private=private, public_fold=panel)
        with self.assertRaises(ValueError):
            seed_sets("full", 2_000_000, public_fold=0)

    def test_training_groups_reject_all_evaluation_partitions(self):
        for name in ("public_eval", "private_eval"):
            with self.assertRaisesRegex(ValueError, "never evaluation"):
                training_groups(partition_ids(name)[:1])
        prompt_ids = partition_ids("public_train")[:1] + partition_ids("private_train")[:1]
        groups = training_groups(prompt_ids)
        self.assertEqual(tuple(group["prompt_id"] for group in groups), prompt_ids)

    def test_public_cli_labels_metric_and_passes_fold(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", ["public_test", "--mode", "full",
                                             "--validation-fold", "1"]), \
                mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch.dict("os.environ", {"STARTER_ROOT": "/test/starter"}), \
                mock.patch.object(run_public, "compare", return_value={"reward": 0.55,
                                  "candidate": 0.55, "baseline": 0.5, "ratio": 1.1}) as compare, \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            run_public.main()
        compare.assert_called_once_with(
            "/test/starter", "full", 1_000_000, torch.device("cuda"), public_fold=1)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["metric"], "mean_generated_path_quality")
        self.assertEqual(result["reward"], 0.55)
        self.assertEqual(result["ratio"], 1.1)
        self.assertEqual(result["settings"]["steps"], 600)
        self.assertEqual(result["validation_fold"], 1)
        self.assertIn("[maze]", stderr.getvalue())

    def test_quick_cli_is_labeled_as_smoke_only(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", ["public_test"]), \
                mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch.object(run_public, "compare", return_value={"ratio": 1.0}), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            run_public.main()
        result = json.loads(stdout.getvalue())
        self.assertIn("Smoke test only", result["note"])
        self.assertIn("Smoke test only", stderr.getvalue())

    def test_quick_fold_cli_fails_before_starting_any_work(self):
        stderr = io.StringIO()
        with mock.patch.object(sys, "argv", ["public_test", "--validation-fold", "1"]), \
                mock.patch.object(run_public, "compare") as compare, \
                contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            run_public.main()
        self.assertEqual(error.exception.code, 2)
        compare.assert_not_called()
        self.assertIn("requires --mode full", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
