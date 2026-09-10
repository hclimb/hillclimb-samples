from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from SFT.eval.aggregate_new_solvers import (
    AggregationValidationError,
    collect_runs,
    load_manifest,
)


PRIMARY_METRIC = {
    "samsum": "rougeL",
    "tydiqa": "f1_score",
    "nq_open": "f1_score",
    "squad": "f1_score",
}


class AggregateNewSolverReliabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.runs_dir = Path(self._temporary_directory.name) / "runs"
        self.runs_dir.mkdir()

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def _run_dir(
        self,
        setting: str,
        method: str,
        optimizer: str,
        *,
        lr: str = "1.00e-05",
    ) -> Path:
        path = self.runs_dir / (
            f"{setting}-{method}-{optimizer}-p0.5-lr{lr}-b8-v16-s42-"
            "Llama-3.2-1B"
        )
        path.mkdir()
        return path

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _complete_run(
        self,
        setting: str,
        task: str,
        method: str,
        optimizer: str,
        *,
        lr: str = "1.00e-05",
        soft_diagnostics: bool = False,
    ) -> Path:
        run_dir = self._run_dir(setting, method, optimizer, lr=lr)
        self._write_json(
            run_dir / "evaluation_results.json",
            [
                {
                    "eval_perplexity": 2.0,
                    "val_perplexity": 2.5,
                    "wall_time": 12.0,
                    "train_wall_time": 10.0,
                }
            ],
        )
        self._write_json(
            run_dir / f"{task}_results.json",
            {"task": task, PRIMARY_METRIC[task]: 0.75},
        )
        if soft_diagnostics:
            self._write_json(
                run_dir / "selection_diagnostics.json",
                [
                    {
                        "soft/entropy": 0.5,
                        "soft/ess": 2.0,
                        "soft/converged": 1.0,
                        "soft/iterations": 7.0,
                        "soft/objective_improvement": 3.0,
                    }
                ],
            )
        return run_dir

    @staticmethod
    def _entry(
        setting: str,
        train: str,
        task: str,
        method: str,
        optimizer: str,
    ) -> dict[str, str]:
        return {
            "setting": setting,
            "train": train,
            "task": task,
            "method": method,
            "optimizer": optimizer,
        }

    def test_manifest_gates_new_runs_and_normalizes_trivia_setting(self) -> None:
        self._complete_run(
            "triviaqa_nq_open", "nq_open", "GlobalOptA", "hybrid"
        )
        self._complete_run(
            "triviaqa_nq_open", "nq_open", "GlobalSoft", "hybrid",
            soft_diagnostics=True,
        )
        self._complete_run(
            "triviaqa_nq_open", "nq_open", "GlobalMuonSur", "hybrid"
        )
        # This valid but unrelated new-solver run must not alter a
        # manifest-gated report.
        self._complete_run(
            "triviaqa_nq_open", "nq_open", "GlobalRandom", "adamw"
        )
        manifest = [
            self._entry(
                "triviaqa_nq", "triviaqa", "nq_open", "GlobalSoft", "hybrid"
            ),
            self._entry(
                "triviaqa_nq", "triviaqa", "nq_open", "GlobalMuonSur", "hybrid"
            ),
        ]

        rows = collect_runs(self.runs_dir, 42, manifest)

        self.assertEqual(
            {row["method"] for row in rows},
            {"GlobalOptA", "GlobalSoft", "GlobalHybridMuonSur"},
        )
        self.assertEqual({row["setting"] for row in rows}, {"triviaqa_nq"})
        self.assertTrue(all(row["task"] == "nq_open" for row in rows))
        soft_row = next(row for row in rows if row["method"] == "GlobalSoft")
        self.assertEqual(soft_row["soft_entropy"], 0.5)

    def test_random_and_muon_do_not_require_solver_diagnostics(self) -> None:
        self._complete_run("alpaca_samsum", "samsum", "GlobalRandom", "adamw")
        self._complete_run(
            "alpaca_samsum", "samsum", "LayerwiseMuonSur", "hybrid"
        )
        manifest = [
            self._entry(
                "alpaca_samsum", "alpaca", "samsum", "GlobalRandom", "adamw"
            ),
            self._entry(
                "alpaca_samsum", "alpaca", "samsum", "LayerwiseMuonSur", "hybrid"
            ),
        ]

        rows = collect_runs(self.runs_dir, 42, manifest)

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {row["method"] for row in rows},
            {"GlobalRandom", "LayerwiseHybridMuonSur"},
        )
        self.assertTrue(all(row.get("soft_entropy") is None for row in rows))

    def test_canonical_muon_and_hybrid_surrogate_labels_are_supported(self) -> None:
        methods = (
            ("GlobalMuonSur", "muon"),
            ("LayerwiseMuonSur", "muon"),
            ("LayerwiseMuonPSur", "muon"),
            ("LayerwiseMuonSatSur", "muon"),
            ("LayerwiseMuonSatPSur", "muon"),
            ("GlobalHybridMuonSur", "hybrid"),
            ("LayerwiseHybridMuonSur", "hybrid"),
            ("GlobalHybridMuonMatrixSur", "hybrid"),
            ("LayerwiseHybridMuonMatrixSur", "hybrid"),
        )
        manifest = []
        for method, optimizer in methods:
            self._complete_run("less_squad", "squad", method, optimizer)
            manifest.append(
                self._entry(
                    "less_squad", "less", "squad", method, optimizer
                )
            )

        rows = collect_runs(self.runs_dir, 42, manifest)

        self.assertEqual(
            {(row["method"], row["optimizer"]) for row in rows}, set(methods)
        )
        self.assertTrue(all(row.get("soft_entropy") is None for row in rows))

    def test_historical_muon_matrix_labels_are_normalized(self) -> None:
        self._complete_run(
            "less_squad", "squad", "GlobalMuonMatrixSur", "muon"
        )
        self._complete_run(
            "less_squad", "squad", "LayerwiseMuonMatrixSur", "hybrid"
        )
        self._complete_run(
            "less_squad", "squad", "GlobalMuonMatrixSur", "hybrid"
        )
        self._complete_run(
            "less_squad", "squad", "LayerwiseMuonMatrixSur", "muon"
        )
        manifest = [
            self._entry(
                "less_squad", "less", "squad", "GlobalMuonMatrixSur", "muon"
            ),
            self._entry(
                "less_squad", "less", "squad", "LayerwiseMuonMatrixSur", "hybrid"
            ),
            self._entry(
                "less_squad", "less", "squad", "GlobalMuonMatrixSur", "hybrid"
            ),
            self._entry(
                "less_squad", "less", "squad", "LayerwiseMuonMatrixSur", "muon"
            ),
        ]

        rows = collect_runs(self.runs_dir, 42, manifest)

        self.assertEqual(
            {row["method"] for row in rows},
            {
                "GlobalMuonSur",
                "LayerwiseMuonSur",
                "GlobalHybridMuonMatrixSur",
                "LayerwiseHybridMuonMatrixSur",
            },
        )
        self.assertTrue(all(row.get("soft_entropy") is None for row in rows))

    def test_historical_muon_variant_labels_are_normalized(self) -> None:
        aliases = (
            ("LayerwiseMuonOnlyPSur", "LayerwiseMuonPSur"),
            ("LayerwiseMuonOnlySatSur", "LayerwiseMuonSatSur"),
            ("LayerwiseMuonOnlySatPSur", "LayerwiseMuonSatPSur"),
        )
        manifest = []
        for historical, _ in aliases:
            self._complete_run("less_squad", "squad", historical, "muon")
            manifest.append(
                self._entry(
                    "less_squad", "less", "squad", historical, "muon"
                )
            )

        rows = collect_runs(self.runs_dir, 42, manifest)

        self.assertEqual(
            {row["method"] for row in rows},
            {canonical for _, canonical in aliases},
        )
        self.assertTrue(all(row.get("soft_entropy") is None for row in rows))

    def test_probability_simplex_soft_label_requires_soft_diagnostics(self) -> None:
        manifest = []
        for optimizer in ("adamw", "muon"):
            self._complete_run(
                "less_squad",
                "squad",
                "LayerwiseSoftP",
                optimizer,
                soft_diagnostics=True,
            )
            manifest.append(
                self._entry(
                    "less_squad", "less", "squad", "LayerwiseSoftP", optimizer
                )
            )

        rows = collect_runs(self.runs_dir, 42, manifest)

        self.assertEqual({row["optimizer"] for row in rows}, {"adamw", "muon"})
        self.assertTrue(all(row["soft_entropy"] == 0.5 for row in rows))

    def test_incomplete_soft_run_reports_every_required_artifact(self) -> None:
        run_dir = self._run_dir(
            "alpaca_samsum", "LayerwiseSoft", "adamw"
        )
        self._write_json(run_dir / "evaluation_results.json", [])
        self._write_json(
            run_dir / "selection_diagnostics.json", [{"soft/entropy": 0.5}]
        )
        manifest = [
            self._entry(
                "alpaca_samsum", "alpaca", "samsum", "LayerwiseSoft", "adamw"
            )
        ]

        with self.assertRaises(AggregationValidationError) as caught:
            collect_runs(self.runs_dir, 42, manifest)

        message = str(caught.exception)
        self.assertIn("Incomplete runs", message)
        self.assertIn("nonempty evaluation_results.json", message)
        self.assertIn("finite downstream primary metric", message)
        self.assertIn("soft_ess", message)
        self.assertIn(str(run_dir), message)

    def test_missing_and_duplicate_expected_runs_are_actionable(self) -> None:
        first = self._complete_run(
            "less_squad", "squad", "GlobalRandom", "adamw"
        )
        second = self._complete_run(
            "less_squad", "squad", "GlobalRandom", "adamw", lr="2.00e-05"
        )
        manifest = [
            self._entry(
                "less_squad", "less", "squad", "GlobalRandom", "adamw"
            ),
            self._entry(
                "less_squad", "less", "squad", "LayerwiseRandom", "adamw"
            ),
        ]

        with self.assertRaises(AggregationValidationError) as caught:
            collect_runs(self.runs_dir, 42, manifest)

        message = str(caught.exception)
        self.assertIn("Missing expected runs", message)
        self.assertIn("method=LayerwiseRandom", message)
        self.assertIn("Duplicate run candidates", message)
        self.assertIn(str(first), message)
        self.assertIn(str(second), message)

    def test_load_manifest_rejects_duplicate_exact_rows(self) -> None:
        manifest_path = Path(self._temporary_directory.name) / "manifest.tsv"
        manifest_path.write_text(
            "setting\ttrain\ttask\tmethod\toptimizer\ttrain_job_id\teval_job_id\n"
            "alpaca_samsum\talpaca\tsamsum\tGlobalRandom\tadamw\t1\t2\n"
            "alpaca_samsum\talpaca\tsamsum\tGlobalRandom\tadamw\t3\t4\n",
            encoding="utf-8",
        )

        with self.assertRaises(AggregationValidationError) as caught:
            load_manifest(manifest_path)

        self.assertIn("Duplicate manifest rows", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
