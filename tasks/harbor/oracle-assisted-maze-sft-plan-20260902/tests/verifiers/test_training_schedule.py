import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from utils.runner import CONFIGS, run_schedule
from verifiers import run_training


class TrainingScheduleTests(unittest.TestCase):
    def test_worker_uses_same_panel_and_initial_checkpoint_for_both_seeds(self):
        for mode, panel_count in (("quick", 3), ("full", 3), ("full", 6)):
            for method in ("baseline", "candidate"):
                with self.subTest(mode=mode, method=method, panels=panel_count), tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    training, evaluation, output = (directory / name for name in
                                                    ("training.json", "evaluation.json", "result.json"))
                    training.write_text(json.dumps([[10 * (i + 1)] for i in range(panel_count)]))
                    evaluation.write_text(json.dumps([[100 * (i + 1)] for i in range(panel_count)]))
                    model = SimpleNamespace(state_dict=lambda: {"fixed": None})
                    artifacts = [f"examples-{i}.npz" for i in range(panel_count)]
                    argv = ["worker", method, mode, str(training), str(output), str(evaluation), "cpu"]
                    if method == "candidate":
                        argv.extend(artifacts)
                    metrics = {"quality": 0.5, "outcomes": {"success": CONFIGS[mode]["eval_count"]}}
                    with mock.patch.object(sys, "argv", argv), \
                            mock.patch.object(run_training, "build_groups", side_effect=lambda ids, **kw: ids) as build, \
                            mock.patch.object(run_training, "load_examples", side_effect=lambda path: path) as load, \
                            mock.patch.object(run_training, "canonical_arrays", side_effect=lambda groups: groups) as canonical, \
                            mock.patch.object(run_training, "initialized_model", return_value=model) as initialize, \
                            mock.patch.object(run_training, "train_model", return_value=model) as train, \
                            mock.patch.object(run_training, "validate_model") as validate, \
                            mock.patch.object(run_training, "evaluate", return_value=metrics) as evaluate:
                        run_training.main()
                    schedule = run_schedule(mode, panel_count)
                    self.assertEqual(initialize.call_count, len(schedule))
                    self.assertEqual(validate.call_count, len(schedule))
                    self.assertEqual(load.call_count, panel_count if method == "candidate" else 0)
                    self.assertEqual(canonical.call_count, panel_count if method == "baseline" else 0)
                    expected_builds = []
                    for panel in range(panel_count):
                        if method == "baseline":
                            expected_builds.append(mock.call([10 * (panel + 1)], include_catalog=False))
                        expected_builds.append(mock.call([100 * (panel + 1)], include_catalog=False))
                    self.assertEqual(build.call_args_list, expected_builds)
                    rows = json.loads(output.read_text())
                    self.assertEqual([(r["panel"], r["seed"]) for r in rows], list(schedule))
                    for call, evaluation_call, (panel, seed) in zip(
                            train.call_args_list, evaluate.call_args_list, schedule):
                        self.assertEqual(call.args[1], artifacts[panel] if method == "candidate"
                                         else [10 * (panel + 1)])
                        self.assertEqual(call.args[2], {**CONFIGS[mode], "seed": seed})
                        self.assertEqual(evaluation_call.args[1], [100 * (panel + 1)])
