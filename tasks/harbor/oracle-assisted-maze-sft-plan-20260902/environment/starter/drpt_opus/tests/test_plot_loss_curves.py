from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from SFT.eval import plot_loss_curves as plot_module
from SFT.eval.plot_loss_curves import (
    BASELINE9_METHODS,
    GENERAL_REDUCTION_METRIC,
    _expected_keys,
    add_general_reduction_points,
    build_parser,
    collect_completed_general_runs,
    curve_summary,
    deduplicate_eval_records,
    discover_runs,
    optimizer_methods_for_profile,
    parse_train_loss,
    parse_wandb_metadata,
    settings_for_profile,
    write_completed_general_outputs,
    write_markdown,
)


class LossCurveParsingTests(unittest.TestCase):
    def test_train_parser_ignores_eval_and_summary_losses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            log.write_text(
                "{'loss': 3.0, 'grad_norm': 1.0, 'learning_rate': 0.0, 'epoch': 0.1}\n"
                "{'eval_loss': 2.0, 'val_loss': 1.5, 'epoch': 0.1}\n"
                "{'train_runtime': 2.0, 'train_loss': 2.5, 'epoch': 1.0}\n"
                "{'loss': 1.0, 'grad_norm': 1.0, 'learning_rate': 1e-5, 'epoch': 1.0}\n",
                encoding="utf-8",
            )

            self.assertEqual(
                parse_train_loss(log),
                [
                    {"step": 1.0, "epoch": 0.1, "value": 3.0},
                    {"step": 2.0, "epoch": 1.0, "value": 1.0},
                ],
            )

    def test_eval_records_are_validated_sorted_and_deduplicated(self) -> None:
        source = Path("evaluation_results.json")
        records = deduplicate_eval_records(
            [
                {"step": 10, "val_loss": 2.0, "eval_loss": 3.0},
                {"step": 0, "val_loss": 4.0, "eval_loss": 5.0},
                {"step": 10, "val_loss": 1.5, "eval_loss": 2.5},
            ],
            source,
        )

        self.assertEqual([row["step"] for row in records], [0.0, 10.0])
        self.assertEqual(records[-1]["target_val_loss"], 1.5)
        self.assertEqual(records[-1]["general_eval_loss"], 2.5)

    def test_curve_reduction_and_normalized_auc(self) -> None:
        summary = curve_summary(
            [
                {"step": 0.0, "value": 3.0},
                {"step": 5.0, "value": 2.0},
                {"step": 10.0, "value": 1.0},
            ]
        )

        self.assertEqual(summary["reduction"], 2.0)
        self.assertEqual(summary["relative_reduction_pct"], 200.0 / 3.0)
        self.assertTrue(math.isclose(summary["normalized_auc"], 2.0))

    def test_wandb_parser_uses_final_run_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            log.write_text(
                "wandb_group=less_squad-adamw-s42,\n"
                "wandb: View run at https://wandb.ai/leena12/drpt_opus/runs/first\n"
                "wandb: View run at https://wandb.ai/leena12/drpt_opus/runs/final42\n",
                encoding="utf-8",
            )

            metadata = parse_wandb_metadata(log)
            self.assertEqual(metadata["wandb_group"], "less_squad-adamw-s42")
            self.assertEqual(metadata["wandb_run_id"], "final42")
            self.assertTrue(metadata["wandb_url"].endswith("/final42"))


class LossCurveProfileTests(unittest.TestCase):
    @staticmethod
    def _write_completed_run(
        root: Path,
        method: str,
        optimizer: str = "hybrid",
        completed: bool = True,
        metadata: dict | None = None,
    ) -> Path:
        run_dir = root / (
            f"alpaca_samsum-{method}-{optimizer}-p0.4-lr1.00e-05-"
            "b8-v16-s42-Llama-3.2-1B"
        )
        run_dir.mkdir()
        (run_dir / "evaluation_results.json").write_text(
            json.dumps(
                [
                    {"step": 0, "val_loss": 3.0, "eval_loss": 4.0},
                    {"step": 1, "val_loss": 2.0, "eval_loss": 3.0},
                ]
            ),
            encoding="utf-8",
        )
        (run_dir / "train.log").write_text(
            "{'train_runtime': 1.0}\n", encoding="utf-8"
        )
        if completed:
            (run_dir / "model.safetensors").write_bytes(b"")
        if metadata is not None:
            (run_dir / "run_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
        return run_dir

    def test_loss52_profile_has_exact_requested_52_runs(self) -> None:
        methods = optimizer_methods_for_profile("loss52")

        self.assertEqual(
            methods["adamw"],
            (
                "FullTraining",
                "GlobalRaw",
                "LayerwiseRaw",
                "GlobalOptA",
                "LayerwiseOptA",
                "GlobalSoft",
                "LayerwiseSoft",
            ),
        )
        self.assertEqual(
            methods["muon"],
            (
                "FullTraining",
                "GlobalRaw",
                "LayerwiseRaw",
                "LayerwiseMuonSur",
                "GlobalSoft",
                "LayerwiseSoft",
            ),
        )
        self.assertEqual(len(list(_expected_keys(methods))), 52)

    def test_baseline9_profile_has_exact_layerwise_5_plus_8_bundle(self) -> None:
        methods = optimizer_methods_for_profile("baseline9")

        self.assertEqual(
            methods["adamw"],
            (
                "FullTraining",
                "LayerwiseRaw",
                "LayerwiseSoft",
                "LayerwiseSoftP",
                "LayerwiseOptA",
            ),
        )
        self.assertEqual(
            methods["muon"],
            (
                "FullTraining",
                "LayerwiseRaw",
                "LayerwiseSoft",
                "LayerwiseSoftP",
                "LayerwiseMuonSur",
                "LayerwiseMuonPSur",
                "LayerwiseMuonSatSur",
                "LayerwiseMuonSatPSur",
            ),
        )
        self.assertEqual(len(list(_expected_keys(methods))), 52)

    def test_baseline9_optimizer_reports_are_independent(self) -> None:
        adamw = optimizer_methods_for_profile("baseline9-adamw")
        muon = optimizer_methods_for_profile("baseline9-muon")

        self.assertEqual(len(list(_expected_keys(adamw))), 20)
        self.assertEqual(len(list(_expected_keys(muon))), 32)
        baseline9 = optimizer_methods_for_profile("baseline9")
        self.assertEqual(adamw, {"adamw": baseline9["adamw"]})
        self.assertEqual(muon, {"muon": baseline9["muon"]})

    def test_dolci32k_profiles_use_canonical_reason_code_and_25_40_runs(self) -> None:
        settings = settings_for_profile("dolci32k")
        combined = optimizer_methods_for_profile("dolci32k")
        self.assertEqual(
            settings,
            ("inst_if", "reason_math", "reason_code", "mixed_if", "mixed_math"),
        )
        self.assertEqual(len(settings) * len(combined["adamw"]), 25)
        self.assertEqual(len(settings) * len(combined["muon"]), 40)
        self.assertEqual(
            optimizer_methods_for_profile("dolci32k-adamw"),
            {"adamw": combined["adamw"]},
        )
        self.assertEqual(
            optimizer_methods_for_profile("dolci32k-muon"),
            {"muon": combined["muon"]},
        )

    def test_default_profile_preserves_legacy_bundle(self) -> None:
        args = build_parser().parse_args([])
        legacy = optimizer_methods_for_profile(args.profile)

        self.assertEqual(args.profile, "legacy")
        self.assertFalse(args.completed_general_only)
        self.assertEqual(args.additional_runs_dir, [])
        self.assertEqual(len(list(_expected_keys(legacy))), 64)
        self.assertIn("GlobalOptA", legacy["hybrid"])
        self.assertIn("GlobalHybridMuonSur", legacy["hybrid"])

    def test_baseline9_profile_has_exact_optimizer_scopes_and_52_runs(self) -> None:
        methods = optimizer_methods_for_profile("baseline9")

        self.assertEqual(methods, BASELINE9_METHODS)
        self.assertEqual(
            methods["adamw"],
            (
                "FullTraining",
                "LayerwiseRaw",
                "LayerwiseSoft",
                "LayerwiseSoftP",
                "LayerwiseOptA",
            ),
        )
        self.assertEqual(
            methods["muon"],
            (
                "FullTraining",
                "LayerwiseRaw",
                "LayerwiseSoft",
                "LayerwiseSoftP",
                "LayerwiseMuonSur",
                "LayerwiseMuonPSur",
                "LayerwiseMuonSatSur",
                "LayerwiseMuonSatPSur",
            ),
        )
        conceptual_methods = {
            method for optimizer in methods.values() for method in optimizer
        }
        self.assertEqual(len(conceptual_methods), 9)
        self.assertEqual(len(list(_expected_keys(methods))), 52)

    def test_completed_general_cli_is_explicit_and_repeatable(self) -> None:
        args = build_parser().parse_args(
            [
                "--profile",
                "baseline9",
                "--completed-general-only",
                "--additional-runs-dir",
                "/tmp/campaign-a",
                "--additional-runs-dir",
                "/tmp/campaign-b",
            ]
        )

        self.assertTrue(args.completed_general_only)
        self.assertEqual(
            args.additional_runs_dir,
            [Path("/tmp/campaign-a"), Path("/tmp/campaign-b")],
        )

    def test_muon_scorer_source_profile_is_paired_and_isolated(self) -> None:
        methods = optimizer_methods_for_profile("muon-source")

        self.assertEqual(methods["adamw"], ())
        self.assertEqual(
            methods["hybrid"],
            (
                "GlobalHybridMuonSur",
                "LayerwiseHybridMuonSur",
                "GlobalHybridMuonMatrixSur",
                "LayerwiseHybridMuonMatrixSur",
            ),
        )
        self.assertEqual(len(list(_expected_keys(methods))), 16)
        self.assertNotIn(
            "GlobalHybridMuonSur",
            optimizer_methods_for_profile("loss52")["muon"],
        )

    def test_loss52_optimizer_reports_are_independent(self) -> None:
        adamw = optimizer_methods_for_profile("loss52-adamw")
        muon = optimizer_methods_for_profile("loss52-muon")
        hybrid = optimizer_methods_for_profile("loss52-hybrid")

        self.assertEqual(len(list(_expected_keys(adamw))), 28)
        self.assertEqual(len(list(_expected_keys(muon))), 24)
        self.assertEqual(len(list(_expected_keys(hybrid))), 24)
        self.assertNotIn("muon", adamw)
        self.assertNotIn("adamw", muon)
        self.assertIn("LayerwiseMuonSur", muon["muon"])
        self.assertNotIn("LayerwiseHybridMuonSur", muon["muon"])
        self.assertIn("LayerwiseHybridMuonSur", hybrid["hybrid"])

    def test_loss52_legacy_profile_preserves_old_combined_array(self) -> None:
        legacy = optimizer_methods_for_profile("loss52-legacy")

        self.assertEqual(len(list(_expected_keys(legacy))), 52)
        self.assertIn("LayerwiseHybridMuonSur", legacy["hybrid"])
        self.assertNotIn("LayerwiseMuonSur", legacy["hybrid"])

    def test_muon_surrogate_variant_profile_is_optimizer_scoped(self) -> None:
        methods = optimizer_methods_for_profile("muon-surrogate-variants")

        self.assertEqual(
            methods,
            {
                "muon": (
                    "LayerwiseMuonSur",
                    "LayerwiseMuonPSur",
                    "LayerwiseMuonSatSur",
                    "LayerwiseMuonSatPSur",
                )
            },
        )
        self.assertEqual(len(list(_expected_keys(methods))), 16)

    def test_soft_variant_profile_pairs_constraints_and_optimizers(self) -> None:
        methods = optimizer_methods_for_profile("soft-variants")

        self.assertEqual(
            methods,
            {
                "adamw": ("LayerwiseSoft", "LayerwiseSoftP"),
                "muon": ("LayerwiseSoft", "LayerwiseSoftP"),
            },
        )
        self.assertEqual(len(list(_expected_keys(methods))), 16)

    def test_soft_probability_runs_are_discovered_for_both_optimizers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adamw = self._write_completed_run(
                root, "LayerwiseSoftP", optimizer="adamw"
            )
            muon = self._write_completed_run(
                root, "LayerwiseSoftP", optimizer="muon"
            )

            found, incomplete = discover_runs(
                root, 42, optimizer_methods_for_profile("soft-variants")
            )

            self.assertEqual(incomplete, [])
            self.assertEqual(
                found,
                {
                    ("alpaca_samsum", "adamw", "LayerwiseSoftP"): adamw,
                    ("alpaca_samsum", "muon", "LayerwiseSoftP"): muon,
                },
            )

    def test_discovery_filters_runs_using_selected_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            included = self._write_completed_run(
                root, "LayerwiseMuonMatrixSur", optimizer="muon"
            )
            excluded = self._write_completed_run(root, "GlobalOptA")

            loss52_found, loss52_incomplete = discover_runs(
                root, 42, optimizer_methods_for_profile("loss52")
            )
            legacy_found, legacy_incomplete = discover_runs(
                root, 42, optimizer_methods_for_profile("legacy")
            )

            self.assertEqual(loss52_incomplete, [])
            self.assertEqual(legacy_incomplete, [])
            self.assertEqual(
                loss52_found,
                {("alpaca_samsum", "muon", "LayerwiseMuonSur"): included},
            )
            self.assertEqual(
                legacy_found[("alpaca_samsum", "hybrid", "GlobalOptA")],
                excluded,
            )

    def test_discovery_recognizes_paired_muon_scorer_source_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            global_mixed = self._write_completed_run(root, "GlobalMuonSur")
            layer_mixed = self._write_completed_run(root, "LayerwiseMuonSur")
            global_run = self._write_completed_run(root, "GlobalMuonMatrixSur")
            layer_run = self._write_completed_run(root, "LayerwiseMuonMatrixSur")

            found, incomplete = discover_runs(
                root, 42, optimizer_methods_for_profile("muon-source")
            )

            self.assertEqual(incomplete, [])
            self.assertEqual(
                found,
                {
                    ("alpaca_samsum", "hybrid", "GlobalHybridMuonSur"): global_mixed,
                    ("alpaca_samsum", "hybrid", "LayerwiseHybridMuonSur"): layer_mixed,
                    ("alpaca_samsum", "hybrid", "GlobalHybridMuonMatrixSur"): global_run,
                    ("alpaca_samsum", "hybrid", "LayerwiseHybridMuonMatrixSur"): layer_run,
                },
            )

    def test_markdown_uses_profile_specific_expected_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.md"
            write_markdown(
                report,
                runs=[],
                missing=[],
                optimizer_methods=optimizer_methods_for_profile("loss52"),
            )

            self.assertIn(
                "Successful requested runs: **0 / 52**", report.read_text(encoding="utf-8")
            )

    def test_dolci_markdown_describes_disjoint_monitoring_and_five_settings(self) -> None:
        previous_settings = plot_module.ACTIVE_SETTINGS
        try:
            plot_module.ACTIVE_SETTINGS = settings_for_profile("dolci32k")
            with tempfile.TemporaryDirectory() as directory:
                report = Path(directory) / "report.md"
                write_markdown(
                    report,
                    runs=[],
                    missing=[],
                    optimizer_methods=optimizer_methods_for_profile("dolci32k"),
                )
                text = report.read_text(encoding="utf-8")
        finally:
            plot_module.ACTIVE_SETTINGS = previous_settings

        self.assertIn("disjoint 128-example target monitoring split", text)
        self.assertIn("It is never used for selection", text)
        self.assertIn("Counts are across the 5 settings", text)
        self.assertNotIn("not independent of selection", text)

    def test_completed_general_collector_combines_roots_and_skips_incomplete(self) -> None:
        methods = {
            "adamw": ("FullTraining", "LayerwiseRaw"),
            "muon": ("LayerwiseMuonSur",),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            full = self._write_completed_run(
                first, "FullTraining", optimizer="adamw"
            )
            incomplete = self._write_completed_run(
                first,
                "LayerwiseRaw",
                optimizer="adamw",
                completed=False,
            )
            matrix = self._write_completed_run(
                second, "LayerwiseMuonMatrixSur", optimizer="muon"
            )

            runs, points, missing = collect_completed_general_runs(
                (first, second), 42, methods
            )

            self.assertEqual(
                {(row["method"], Path(row["run_dir"])) for row in runs},
                {("FullTraining", full), ("LayerwiseMuonSur", matrix)},
            )
            self.assertEqual(len(points), 4)
            raw_entry = next(
                row
                for row in missing
                if row["setting"] == "alpaca_samsum"
                and row["optimizer"] == "adamw"
                and row["method"] == "LayerwiseRaw"
            )
            self.assertEqual(raw_entry["status"], "incomplete")
            self.assertEqual(raw_entry["run_dir"], str(incomplete))
            self.assertEqual(len(runs) + len(missing), 12)

    def test_completed_general_collector_excludes_ambiguous_matches(self) -> None:
        methods = {"adamw": ("FullTraining",)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            self._write_completed_run(first, "FullTraining", optimizer="adamw")
            self._write_completed_run(second, "FullTraining", optimizer="adamw")

            runs, points, missing = collect_completed_general_runs(
                (first, second), 42, methods
            )

            self.assertEqual(runs, [])
            self.assertEqual(points, [])
            ambiguous = next(
                row for row in missing if row["setting"] == "alpaca_samsum"
            )
            self.assertEqual(ambiguous["status"], "ambiguous")
            self.assertIn("multiple completed runs", ambiguous["reason"])
            self.assertEqual(len(missing), 4)

    def test_general_reduction_curve_is_initial_minus_current(self) -> None:
        points = [
            {
                "setting": "alpaca_samsum",
                "optimizer": "adamw",
                "method": "FullTraining",
                "metric": "general_eval_loss",
                "step": step,
                "progress": progress,
                "value": value,
                "smoothed_value": value,
                "run_dir": "/tmp/run",
            }
            for step, progress, value in ((0, 0.0, 4.0), (1, 0.5, 3.5), (2, 1.0, 2.0))
        ]

        combined = add_general_reduction_points(points)
        reduction = [
            row for row in combined if row["metric"] == GENERAL_REDUCTION_METRIC
        ]

        self.assertEqual(len(combined), 6)
        self.assertEqual([row["value"] for row in reduction], [0.0, 0.5, 2.0])

    def test_completed_general_writer_emits_manifest_report_and_plots(self) -> None:
        methods = {"adamw": ("FullTraining",)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs_root = root / "runs"
            output = root / "report"
            runs_root.mkdir()
            self._write_completed_run(
                runs_root,
                "FullTraining",
                optimizer="adamw",
                metadata={
                    "soft_weighting_constraint": "capped_simplex",
                    "soft_replay_precision": "bf16_fp32",
                },
            )
            runs, points, missing = collect_completed_general_runs(
                (runs_root,), 42, methods
            )

            write_completed_general_outputs(
                output, runs, points, missing, methods, (runs_root,)
            )

            expected_files = (
                "included_runs.csv",
                "missing_runs.csv",
                "requested_runs_manifest.csv",
                "general_eval_loss_summary.csv",
                "general_eval_loss_points.csv.gz",
                "plot_styles.csv",
                "general_eval_loss_grid.png",
                "general_eval_loss_reduction_grid.png",
                "adamw_general_eval_loss_grid.png",
                "adamw_general_eval_loss_reduction_grid.png",
                "general_eval_loss_report.md",
                "alpaca_samsum_general_eval_loss.png",
                "alpaca_samsum_general_eval_loss_reduction.png",
            )
            for name in expected_files:
                self.assertTrue((output / name).is_file(), name)
            with (output / "requested_runs_manifest.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                manifest = list(csv.DictReader(handle))
            self.assertEqual(len(manifest), 4)
            self.assertEqual(
                [row["status"] for row in manifest].count("completed"), 1
            )
            completed = next(
                row for row in manifest if row["status"] == "completed"
            )
            self.assertEqual(
                completed["soft_weighting_constraint"], "capped_simplex"
            )
            self.assertEqual(
                completed["soft_replay_precision"], "bf16_fp32"
            )
            with (output / "included_runs.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                included = list(csv.DictReader(handle))
            self.assertEqual(included[0]["soft_replay_precision"], "bf16_fp32")
            report = (output / "general_eval_loss_report.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Completed requested runs included: **1 / 4**", report)
            self.assertIn("Rankings are only among completed runs", report)


if __name__ == "__main__":
    unittest.main()
