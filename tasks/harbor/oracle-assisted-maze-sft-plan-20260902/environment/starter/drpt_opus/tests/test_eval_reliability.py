from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from SFT.eval import eval as eval_module


def _make_model_dir(root: str, name: str) -> str:
    path = os.path.join(root, name)
    os.makedirs(path)
    with open(os.path.join(path, "config.json"), "w") as handle:
        json.dump({}, handle)
    return path


class EvalDiscoveryReliabilityTests(unittest.TestCase):
    def test_token_normalized_opta_run_name_is_parsed(self) -> None:
        name = (
            "alpaca_samsum-GlobalOptANorm-adamw-p0.4-"
            "lr1.00e-05-b8-v16-s42-Llama-3.2-1B"
        )

        parsed = eval_module.parse_model_name(name)

        self.assertEqual(parsed["selection"], "OptimizerAwareGlobalSubset")
        self.assertEqual(parsed["optimizer_aware_target_mode"], "opta")
        self.assertEqual(parsed["optimizer_type"], "adamw")
        self.assertEqual(parsed["seed"], "42")

    def test_canonical_muon_matrix_surrogate_names_are_parsed(self) -> None:
        expected = {
            "GlobalMuonSur": "GlobalMuonMatrixSpectral",
            "LayerwiseMuonSur": "LayerWiseMuonMatrixSpectral",
            "LayerwiseMuonPSur": "LayerWiseMuonMatrixSpectralP",
            "LayerwiseMuonSatSur": "LayerWiseMuonMatrixSpectralSat",
            "LayerwiseMuonSatPSur": "LayerWiseMuonMatrixSpectralSatP",
        }
        for label, selection in expected.items():
            with self.subTest(label=label):
                name = (
                    f"alpaca_samsum-{label}-muon-p0.4-"
                    "lr1.00e-05-b8-v16-s42-Llama-3.2-1B"
                )
                parsed = eval_module.parse_model_name(name)
                self.assertEqual(parsed["selection"], selection)
                self.assertEqual(parsed["optimizer_type"], "muon")
                self.assertEqual(parsed["seed"], "42")

    def test_explicit_hybrid_muon_scorer_names_are_parsed(self) -> None:
        expected = {
            "GlobalHybridMuonSur": "GlobalMuonSpectral",
            "LayerwiseHybridMuonSur": "LayerWiseMuonSpectral",
            "GlobalHybridMuonMatrixSur": "GlobalMuonMatrixSpectral",
            "LayerwiseHybridMuonMatrixSur": "LayerWiseMuonMatrixSpectral",
        }
        for label, selection in expected.items():
            with self.subTest(label=label):
                name = (
                    f"alpaca_samsum-{label}-hybrid-p0.4-"
                    "lr1.00e-05-b8-v16-s42-Llama-3.2-1B"
                )
                parsed = eval_module.parse_model_name(name)
                self.assertEqual(parsed["selection"], selection)
                self.assertEqual(parsed["optimizer_type"], "hybrid")

    def test_historical_muon_labels_preserve_scorer_semantics(self) -> None:
        expected = {
            ("GlobalMuonSur", "hybrid"): "GlobalMuonSpectral",
            ("LayerwiseMuonSur", "hybrid"): "LayerWiseMuonSpectral",
            ("GlobalMuonMatrixSur", "muon"): "GlobalMuonMatrixSpectral",
            ("LayerwiseMuonMatrixSur", "muon"): "LayerWiseMuonMatrixSpectral",
            ("GlobalMuonMatrixSur", "hybrid"): "GlobalMuonMatrixSpectral",
            ("LayerwiseMuonMatrixSur", "hybrid"): "LayerWiseMuonMatrixSpectral",
            ("LayerwiseMuonOnlyPSur", "muon"): "LayerWiseMuonMatrixSpectralP",
            ("LayerwiseMuonOnlySatSur", "muon"): "LayerWiseMuonMatrixSpectralSat",
            ("LayerwiseMuonOnlySatPSur", "muon"): "LayerWiseMuonMatrixSpectralSatP",
        }
        for (label, optimizer), selection in expected.items():
            with self.subTest(label=label, optimizer=optimizer):
                name = (
                    f"alpaca_samsum-{label}-{optimizer}-p0.4-"
                    "lr1.00e-05-b8-v16-s42-Llama-3.2-1B"
                )
                parsed = eval_module.parse_model_name(name)
                self.assertEqual(parsed["selection"], selection)
                self.assertEqual(parsed["optimizer_type"], optimizer)

    def test_first_class_muon_run_name_and_historical_filter_are_supported(self) -> None:
        name = (
            "alpaca_samsum-LayerwiseMuonSur-muon-p0.4-"
            "lr1.00e-05-b8-v16-s42-Llama-3.2-1B"
        )

        parsed = eval_module.parse_model_name(name)

        self.assertEqual(parsed["selection"], "LayerWiseMuonMatrixSpectral")
        self.assertEqual(parsed["optimizer_type"], "muon")
        with tempfile.TemporaryDirectory() as root:
            wanted = _make_model_dir(root, name)
            matches = eval_module.find_models(
                root, "alpaca_samsum", "LayerwiseMuonMatrixSur", "muon", seed=42
            )
            self.assertEqual(matches, [wanted])

    def test_canonical_filters_discover_historical_muon_run_labels(self) -> None:
        aliases = (
            ("GlobalMuonSur", "GlobalMuonMatrixSur", "muon"),
            ("LayerwiseMuonSur", "LayerwiseMuonMatrixSur", "muon"),
            ("LayerwiseMuonPSur", "LayerwiseMuonOnlyPSur", "muon"),
            ("LayerwiseMuonSatSur", "LayerwiseMuonOnlySatSur", "muon"),
            ("LayerwiseMuonSatPSur", "LayerwiseMuonOnlySatPSur", "muon"),
            ("GlobalHybridMuonSur", "GlobalMuonSur", "hybrid"),
            ("LayerwiseHybridMuonSur", "LayerwiseMuonSur", "hybrid"),
            ("GlobalHybridMuonMatrixSur", "GlobalMuonMatrixSur", "hybrid"),
            (
                "LayerwiseHybridMuonMatrixSur",
                "LayerwiseMuonMatrixSur",
                "hybrid",
            ),
        )
        for canonical, historical, optimizer in aliases:
            with self.subTest(canonical=canonical, historical=historical):
                with tempfile.TemporaryDirectory() as root:
                    name = (
                        f"alpaca_samsum-{historical}-{optimizer}-p0.4-"
                        "lr1.00e-05-b8-v16-s42-Llama-3.2-1B"
                    )
                    wanted = _make_model_dir(root, name)
                    matches = eval_module.find_models(
                        root,
                        "alpaca_samsum",
                        canonical,
                        optimizer,
                        seed=42,
                    )
                    self.assertEqual(matches, [wanted])

    def test_probability_simplex_soft_name_supports_adamw_and_muon(self) -> None:
        for optimizer in ("adamw", "muon"):
            with self.subTest(optimizer=optimizer):
                name = (
                    f"less_squad-LayerwiseSoftP-{optimizer}-p0.005-"
                    "lr1.00e-05-b8-v16-s42-Llama-3.2-1B"
                )
                parsed = eval_module.parse_model_name(name)
                self.assertEqual(parsed["selection"], "LayerWiseSoftProbability")
                self.assertEqual(parsed["optimizer_type"], optimizer)
                self.assertEqual(parsed["seed"], "42")

    def test_find_models_matches_exact_parsed_seed(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            wanted = _make_model_dir(
                root,
                "alpaca_samsum-GlobalSoft-adamw-p0.5-lr2e-5-b8-v16-s42-Llama-3.2-1B",
            )
            _make_model_dir(
                root,
                "alpaca_samsum-GlobalSoft-adamw-p0.5-lr2e-5-b8-v16-s420-Llama-3.2-1B",
            )
            _make_model_dir(
                root,
                "alpaca_samsum-GlobalSoft-hybrid-p0.5-lr2e-5-b8-v16-s42-Llama-3.2-1B",
            )

            matches = eval_module.find_models(
                root, "alpaca_samsum", "GlobalSoft", "adamw", seed=42
            )

            self.assertEqual(matches, [wanted])

    def test_require_single_match_rejects_ambiguous_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            for model in ("Llama-3.2-1B", "Llama-3.2-3B"):
                _make_model_dir(
                    root,
                    f"alpaca_samsum-GlobalSoft-adamw-p0.5-lr2e-5-b8-v16-s42-{model}",
                )

            argv = [
                "eval.py",
                "--models_dir",
                root,
                "--train",
                "alpaca",
                "--task",
                "samsum",
                "--method",
                "GlobalSoft",
                "--optimizer_type",
                "adamw",
                "--seed",
                "42",
                "--require_single_match",
            ]
            with patch.object(sys, "argv", argv), patch.object(
                eval_module, "evaluate_model"
            ) as evaluate:
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        eval_module.main()

            self.assertEqual(raised.exception.code, 1)
            evaluate.assert_not_called()

    def test_require_single_match_rejects_zero_matches(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            argv = [
                "eval.py",
                "--models_dir",
                root,
                "--train",
                "alpaca",
                "--task",
                "samsum",
                "--method",
                "GlobalSoft",
                "--optimizer_type",
                "adamw",
                "--seed",
                "42",
                "--require_single_match",
            ]
            with patch.object(sys, "argv", argv), patch.object(
                eval_module, "evaluate_model"
            ) as evaluate:
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        eval_module.main()

            self.assertEqual(raised.exception.code, 1)
            evaluate.assert_not_called()


class EvalResultReliabilityTests(unittest.TestCase):
    def test_required_result_must_exist_and_match_task(self) -> None:
        with tempfile.TemporaryDirectory() as model_path:
            missing = eval_module._validate_required_result(model_path, "nq_open")
            self.assertIn("was not written", missing)

            result_path = os.path.join(model_path, "nq_open_results.json")
            with open(result_path, "w") as handle:
                json.dump({"task": "squad", "em": 0.5}, handle)
            wrong_task = eval_module._validate_required_result(model_path, "nq_open")
            self.assertIn("expected 'nq_open'", wrong_task)

            with open(result_path, "w") as handle:
                json.dump({"task": "nq_open", "em": 0.5}, handle)
            invalid_metric = eval_module._validate_required_result(
                model_path, "nq_open"
            )
            self.assertIn("primary metric 'f1'", invalid_metric)

            with open(result_path, "w") as handle:
                json.dump({"task": "nq_open", "em": 0.5, "f1": float("nan")}, handle)
            nonfinite_metric = eval_module._validate_required_result(
                model_path, "nq_open"
            )
            self.assertIn("primary metric 'f1'", nonfinite_metric)

            with open(result_path, "w") as handle:
                json.dump({"task": "nq_open", "em": 0.5, "f1": 0.6}, handle)
            self.assertIsNone(
                eval_module._validate_required_result(model_path, "nq_open")
            )

    def test_single_model_failure_exits_nonzero(self) -> None:
        argv = ["eval.py", "--model_path", "/tmp/model", "--task", "nq_open"]
        failed = {"model_name": "model", "error": "generation failed"}
        with patch.object(sys, "argv", argv), patch.object(
            eval_module, "evaluate_model", return_value=failed
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    eval_module.main()

        self.assertEqual(raised.exception.code, 1)

    def test_discovered_model_failure_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            model_path = _make_model_dir(
                root,
                "alpaca_samsum-GlobalSoft-adamw-p0.5-lr2e-5-b8-v16-s42-Llama-3.2-1B",
            )
            argv = [
                "eval.py",
                "--models_dir",
                root,
                "--train",
                "alpaca",
                "--task",
                "samsum",
                "--method",
                "GlobalSoft",
                "--optimizer_type",
                "adamw",
                "--seed",
                "42",
                "--require_single_match",
            ]
            failed = {"model_name": os.path.basename(model_path), "error": "failed"}
            with patch.object(sys, "argv", argv), patch.object(
                eval_module, "evaluate_model", return_value=failed
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        eval_module.main()

        self.assertEqual(raised.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
