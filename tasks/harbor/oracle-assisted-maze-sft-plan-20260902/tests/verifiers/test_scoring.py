import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from utils.contract import CandidateError
from utils.runner import (CONFIGS, SEEDS, _run, aggregate, compare, construct_examples,
                          run_schedule, seed_sets)


class ScoringTests(unittest.TestCase):
    def results(self, qualities, mode="full"):
        def runs(values):
            return [{"panel": panel, "seed": seed, "quality": values[panel],
                     "outcomes": {"success": CONFIGS[mode]["eval_count"]}}
                    for panel, seed in run_schedule(mode)]
        return {"baseline": runs([0.5] * 3), "candidate": runs(qualities)}

    def test_every_panel_counts_and_bad_panel_cannot_hide_behind_median(self):
        fragile = aggregate(self.results([0.0, 0.6, 0.6]), "full")
        robust = aggregate(self.results([0.55] * 3), "full")
        self.assertAlmostEqual(fragile["ratio"], 0.8)
        self.assertAlmostEqual(robust["ratio"], 1.1)
        self.assertLess(fragile["ratio"], robust["ratio"])

    def test_missing_panels_cases_and_nonfinite_metrics_are_evaluation_errors(self):
        original = self.results([0.5] * 3)
        for method in ("baseline", "candidate"):
            for mutation in (lambda runs: runs.pop(),
                             lambda runs: runs[0].update(outcomes={"success": 95}),
                             lambda runs: runs[1].update(seed=17),
                             lambda runs: runs[1].update(panel=1),
                             lambda runs: runs[1].pop("panel"),
                             lambda runs: runs[0].update(quality=float("nan")),
                             lambda runs: runs[0].update(quality=float("inf")),
                             lambda runs: runs[0].update(quality=-0.1),
                             lambda runs: runs[0].update(quality=1.1)):
                result = copy.deepcopy(original)
                mutation(result[method])
                with self.subTest(method=method), self.assertRaises(RuntimeError):
                    aggregate(result, "full")

    def test_baseline_control_reward_is_quality_and_diagnostic_ratio_is_one(self):
        result = aggregate(self.results([0.5] * 3), "full")
        self.assertEqual(result["reward"], 0.5)
        self.assertEqual(result["ratio"], 1.0)

    def test_reward_is_candidate_mean_quality_for_every_schedule(self):
        for mode, private, panels in (("quick", False, 3), ("full", False, 3),
                                      ("full", True, 6)):
            for quality in (0.0, 0.4, 0.5, 0.75, 1.0):
                with self.subTest(mode=mode, private=private, quality=quality):
                    def runs(value):
                        return [{"panel": p, "seed": s, "quality": value,
                                 "outcomes": {"success": CONFIGS[mode]["eval_count"]}}
                                for p, s in run_schedule(mode, panels)]
                    result = aggregate({"baseline": runs(0.5), "candidate": runs(quality)},
                                       mode, private=private)
                    self.assertEqual(result["reward"], quality)
                    self.assertEqual(result["candidate"], quality)
                    self.assertEqual(result["ratio"], quality / 0.5)
                    self.assertEqual(result["evaluation_version"], "quality-v1")

    def test_official_verifier_writes_quality_not_ratio(self):
        import verify

        for quality in (0.0, 0.4, 0.5, 0.75, 1.0):
            result = aggregate(self.results([quality] * 3), "full")
            with self.subTest(quality=quality), tempfile.TemporaryDirectory() as temporary, \
                    mock.patch.object(verify, "LOG_DIR", Path(temporary)), \
                    mock.patch("torch.cuda.is_available", return_value=True), \
                    mock.patch.object(verify, "compare", return_value=result):
                verify.main()
                self.assertEqual(json.loads((Path(temporary) / "reward.json").read_text()),
                                 {"valid": 1, "reward": quality})
                diagnostics = json.loads((Path(temporary) / "diagnostics.json").read_text())
                self.assertEqual(diagnostics["baseline"], 0.5)
                self.assertEqual(diagnostics["ratio"], quality / 0.5)

    def test_full_has_two_training_order_seeds_per_panel_and_quick_is_unchanged(self):
        self.assertEqual(run_schedule("full"),
                         ((0, 17), (0, 101), (1, 29), (1, 101), (2, 43), (2, 101)))
        self.assertEqual(run_schedule("quick"), tuple(enumerate(SEEDS)))
        self.assertEqual(CONFIGS["full"],
                         {"train_count": 192, "eval_count": 96, "steps": 600, "batch_size": 64})
        self.assertEqual(CONFIGS["quick"],
                         {"train_count": 48, "eval_count": 24, "steps": 120, "batch_size": 32})
        self.assertEqual(aggregate(self.results([0.6] * 3, "quick"), "quick")["ratio"], 1.2)

    def test_second_seed_counts_and_aggregation_is_ratio_of_means(self):
        result = self.results([0.6] * 3)
        for run in result["candidate"]:
            if run["seed"] == 101:
                run["quality"] = 0.4
        summary = aggregate(result, "full")
        self.assertEqual(summary["ratio"], 1.0)
        self.assertEqual(summary["evaluation_version"], "quality-v1")
        self.assertEqual(summary["aggregation"], "mean_over_panel_seed_pairs")
        result = self.results([0.5] * 3)
        for run in result["baseline"]:
            run["quality"] = (0.2, 0.5, 0.8)[run["panel"]]
        self.assertEqual(aggregate(result, "full")["ratio"], 1.0)

    def test_trusted_timeout_is_not_candidate_failure(self):
        with mock.patch("utils.runner.CHILD_TIMEOUT", 0.05):
            for untrusted, error in ((False, RuntimeError), (True, CandidateError)):
                with self.subTest(untrusted=untrusted), self.assertRaises(error):
                    _run([sys.executable, "-I", "-c", "import time; time.sleep(1)"],
                         untrusted=untrusted, cwd="/tmp")

    def test_trusted_fixture_failure_stays_an_evaluation_error(self):
        with mock.patch("utils.runner.training_groups", side_effect=RuntimeError("bad fixture")):
            with self.assertRaisesRegex(RuntimeError, "bad fixture"):
                construct_examples(Path("/unused"), [], Path("/unused"))

    def test_builder_setup_failure_stays_an_evaluation_error(self):
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch("utils.runner._run", return_value=(1, "worker dependency unavailable")):
            with self.assertRaisesRegex(RuntimeError, "preflight"):
                construct_examples(Path(temporary), seed_sets("quick", 1_000_000)[0][0][:1],
                                   Path(temporary) / "examples.npz")

    def test_official_verifier_preserves_error_distinction(self):
        import verify

        for failure in (CandidateError("bad submission"), RuntimeError("bad evaluator")):
            with tempfile.TemporaryDirectory() as temporary, \
                    mock.patch.object(verify, "LOG_DIR", Path(temporary)), \
                    mock.patch("torch.cuda.is_available", return_value=True), \
                    mock.patch.object(verify, "compare", side_effect=failure):
                if isinstance(failure, CandidateError):
                    verify.main()
                    self.assertEqual(json.loads((Path(temporary) / "reward.json").read_text()),
                                     {"valid": 0, "reward": 0.0})
                else:
                    with self.assertRaises(RuntimeError):
                        verify.main()
                    self.assertFalse((Path(temporary) / "reward.json").exists())

    def test_six_private_panels_all_count_and_missing_new_panels_fail(self):
        schedule = run_schedule("full", 6)
        self.assertEqual(schedule, tuple((p, s) for p in range(6)
                                         for s in (SEEDS[p % 3], 101)))
        def runs(qualities):
            return [{"panel": p, "seed": s, "quality": qualities[p], "outcomes": {"success": 96}}
                    for p, s in schedule]
        result = {"baseline": runs([0.5] * 6), "candidate": runs([0.6] * 3 + [0.0] * 3)}
        summary = aggregate(result, "full", private=True)
        self.assertAlmostEqual(summary["ratio"], 0.6)
        self.assertEqual(summary["evaluation_version"], "quality-v1")
        for method in ("baseline", "candidate"):
            changed = copy.deepcopy(result)
            changed[method] = changed[method][:6]
            with self.assertRaisesRegex(RuntimeError, "omitted"):
                aggregate(changed, "full", private=True)

    def test_compare_sets_private_worker_timeout_without_changing_candidate_limit(self):
        import utils.runner as runner
        for private, offset, panels, timeout in ((False, 1_000_000, 3, 900),
                                                (True, 2_000_000, 6, 1800),
                                                (False, 2_000_000, 6, 1800)):
            def worker(command, **kwargs):
                self.assertEqual(kwargs["timeout"], timeout)
                self.assertEqual(len(json.loads(Path(command[6]).read_text())), panels)
                if command[4] == "candidate":
                    self.assertEqual(len(command[10:]), panels)
                Path(command[7]).write_text(json.dumps([
                    {"panel": p, "seed": s, "quality": 0.5, "outcomes": {"success": 96}}
                    for p, s in run_schedule("full", panels)]))
                return 0, ""
            with self.subTest(private=private, offset=offset), \
                    mock.patch.object(runner, "construct_examples") as construct, \
                    mock.patch.object(runner, "_run", side_effect=worker):
                result = compare(Path("/unused"), "full", offset, "cpu", private=private)
            self.assertEqual(construct.call_count, panels)
            self.assertEqual(result["ratio"], 1.0)
            self.assertEqual(runner.CHILD_TIMEOUT, 900)
