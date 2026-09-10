from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from SFT.eval import campaign_downstream as campaign
from SFT.eval.task_registry import (
    TASK_SPECS,
    profile32k_downstream_cells,
    profile32k_evaluator_pins,
    profile32k_model_metadata,
)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_completed_run(root: Path, optimizer: str = "muon") -> Path:
    run_dir = root / (
        "alpaca_samsum-LayerwiseMuonSur-"
        f"{optimizer}-p0.4-lr1.00e-05-b8-v16-s42-Llama-3.2-1B"
    )
    run_dir.mkdir()
    _write_json(run_dir / "run_metadata.json", {"optimizer_type": optimizer})
    _write_json(run_dir / "config.json", {})
    (run_dir / "model.safetensors").write_bytes(b"weights")
    (run_dir / "train.log").write_text(
        "{'train_runtime': 1.25, 'train_loss': 1.0}\n", encoding="utf-8"
    )
    _write_json(
        run_dir / "evaluation_results.json",
        [
            {"step": 0, "eval_loss": 2.0},
            {"step": 1, "eval_loss": 1.5},
        ],
    )
    return run_dir



def _dolci_provenance(task: str) -> dict:
    spec = TASK_SPECS[task]
    return {
        "schema_version": "drpt.eval.result.v1",
        "evaluator": dict(profile32k_evaluator_pins("dolci32k")[task]),
        "generation": {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": spec.default_max_new_tokens,
            "thinking": spec.thinking,
            "thinking_output_stripped": True,
        },
    }


class CampaignRunValidationTests(unittest.TestCase):
    def test_dolci32k_downstream_mapping_has_exact_family_cell_counts(self) -> None:
        adamw = profile32k_downstream_cells("dolci32k", "adamw")
        muon = profile32k_downstream_cells("dolci32k", "muon")
        self.assertEqual(len(adamw), 35)
        self.assertEqual(len(muon), 56)
        self.assertEqual(
            (adamw[0].setting, adamw[0].method, adamw[0].task),
            ("inst_if", "FullTraining", "ifeval"),
        )
        self.assertEqual(
            (adamw[1].setting, adamw[1].method, adamw[1].task),
            ("inst_if", "FullTraining", "ifbench"),
        )
        self.assertEqual(
            (muon[-1].setting, muon[-1].method, muon[-1].task),
            ("mixed_math", "LayerwiseMuonSatPSur", "math500"),
        )

    def test_dolci32k_registry_uses_canonical_reason_code(self) -> None:
        adamw = profile32k_downstream_cells("dolci32k", "adamw")
        muon = profile32k_downstream_cells("dolci32k", "muon")
        self.assertEqual(len(adamw), 35)
        self.assertEqual(len(muon), 56)
        self.assertIn(
            ("reason_code", "FullTraining", "mbpp_plus"),
            [(cell.setting, cell.method, cell.task) for cell in adamw],
        )
        self.assertNotIn("reason_mbpp", {cell.setting for cell in adamw})

    def test_dolci_completion_requires_exact_model_profile_tuple(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _make_completed_run(Path(temporary), "adamw")
            model = profile32k_model_metadata("dolci32k", "qwen3_4b")
            _write_json(
                run_dir / "run_metadata.json",
                {
                    "optimizer_type": "adamw",
                    "experiment_profile": "dolci32k",
                    "model_profile": "qwen3_4b",
                    **model,
                },
            )
            _write_json(
                run_dir / "run_status.json",
                {
                    "status": "complete",
                    "profile": "dolci32k",
                    "artifact_build_id": "abc123",
                },
            )
            (run_dir / "_SUCCESS").write_text("", encoding="utf-8")
            with mock.patch.dict(
                os.environ,
                {
                    "DRPT_ARTIFACT_BUILD_ID": "abc123",
                    "DRPT_MODEL_PROFILE": "qwen3_4b",
                },
                clear=False,
            ):
                campaign.validate_completed_run(run_dir, "adamw", "dolci32k")
                metadata = json.loads(
                    (run_dir / "run_metadata.json").read_text(encoding="utf-8")
                )
                metadata["model_revision"] = "drifted"
                _write_json(run_dir / "run_metadata.json", metadata)
                with self.assertRaisesRegex(
                    campaign.CampaignEvaluationError, "model_revision"
                ):
                    campaign.validate_completed_run(run_dir, "adamw", "dolci32k")

    def test_completed_run_requires_matching_public_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _make_completed_run(Path(temporary), "muon")
            campaign.validate_completed_run(run_dir, "muon")
            with self.assertRaisesRegex(
                campaign.CampaignEvaluationError, "expected 'hybrid'"
            ):
                campaign.validate_completed_run(run_dir, "hybrid")

    def test_completed_run_requires_post_training_general_eval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _make_completed_run(Path(temporary), "adamw")
            _write_json(run_dir / "evaluation_results.json", [{"step": 0, "eval_loss": 2.0}])
            with self.assertRaisesRegex(
                campaign.CampaignEvaluationError, "post-training eval_loss"
            ):
                campaign.validate_completed_run(run_dir, "adamw")

    def test_dolci_completion_accepts_sharded_model_and_checks_all_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _make_completed_run(Path(temporary), "adamw")
            model = profile32k_model_metadata("dolci32k", "qwen3_4b")
            _write_json(
                run_dir / "run_metadata.json",
                {
                    "optimizer_type": "adamw",
                    "experiment_profile": "dolci32k",
                    "model_profile": "qwen3_4b",
                    **model,
                },
            )
            (run_dir / "model.safetensors").unlink()
            shard_name = "model-00001-of-00001.safetensors"
            (run_dir / shard_name).write_bytes(b"weights")
            _write_json(
                run_dir / "model.safetensors.index.json",
                {"weight_map": {"model.embed_tokens.weight": shard_name}},
            )
            _write_json(
                run_dir / "run_status.json",
                {
                    "status": "complete",
                    "profile": "dolci32k",
                    "artifact_build_id": "abc123",
                },
            )
            (run_dir / "_SUCCESS").write_text("", encoding="utf-8")
            with mock.patch.dict(
                os.environ,
                {
                    "DRPT_ARTIFACT_BUILD_ID": "abc123",
                    "DRPT_MODEL_PROFILE": "qwen3_4b",
                },
                clear=False,
            ):
                campaign.validate_completed_run(run_dir, "adamw", "dolci32k")
                (run_dir / shard_name).unlink()
                with self.assertRaisesRegex(
                    campaign.CampaignEvaluationError, "config/weights"
                ):
                    campaign.validate_completed_run(run_dir, "adamw", "dolci32k")

    def test_downstream_result_uses_task_native_metric(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _make_completed_run(Path(temporary))
            _write_json(
                run_dir / "samsum_results.json",
                {"task": "samsum", "rouge1": 0.5, "rougeL": 0.4},
            )
            result = campaign.read_downstream_result(run_dir, "samsum")
            self.assertEqual(result["primary_metric_name"], "rougeL")
            self.assertEqual(result["primary_metric"], 0.4)

    def test_legacy_qa_f1_is_normalized_to_current_percent_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _make_completed_run(Path(temporary))
            _write_json(
                run_dir / "nq_open_results.json",
                {"task": "nq_open", "f1": 0.375, "em": 0.25},
            )
            result = campaign.read_downstream_result(run_dir, "nq_open")
            self.assertEqual(result["primary_metric_name"], "f1_score")
            self.assertEqual(result["primary_metric"], 37.5)

    def test_dolci_task_native_metrics_remain_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _make_completed_run(Path(temporary))
            fixtures = {
                "ifeval": ("ifeval_results.json", "prompt_level_strict_acc", 61.0),
                "ifbench": ("ifbench_results.json", "prompt_level_loose_acc", 52.0),
                "math500": ("math500_results.json", "accuracy", 43.0),
                "mbpp_plus": (
                    "mbpp_plus_results.json",
                    "base_plus_extra_pass_at_1",
                    34.0,
                ),
            }
            for task, (filename, metric, expected) in fixtures.items():
                _write_json(run_dir / filename, {"task": task, metric: expected})
                result = campaign.read_downstream_result(run_dir, task)
                self.assertEqual(result["primary_metric_name"], metric)
                self.assertEqual(result["primary_metric"], expected)

    def test_legacy_mbpp_plus_primary_remains_on_percent_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _make_completed_run(Path(temporary))
            _write_json(
                run_dir / "mbpp_plus_results.json",
                {"task": "mbpp_plus", "plus_pass_at_1": 34.0},
            )
            result = campaign.read_downstream_result(run_dir, "mbpp_plus")
            self.assertEqual(
                result["primary_metric_name"], "base_plus_extra_pass_at_1"
            )
            self.assertEqual(result["primary_metric"], 34.0)

    def test_dolci_official_report_rejects_limited_smoke_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _make_completed_run(Path(temporary))
            _write_json(
                run_dir / "math500_results.json",
                {
                    "task": "math500",
                    "evaluation_scope": "limited",
                    "accuracy": 50.0,
                },
            )
            with self.assertRaisesRegex(
                campaign.CampaignEvaluationError, "rejects non-full result"
            ):
                campaign.read_downstream_result(run_dir, "math500", "dolci32k")


    def test_dolci_full_result_requires_exact_generation_and_evaluator_pins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _make_completed_run(Path(temporary))
            payload = {
                "task": "math500",
                "evaluation_scope": "full",
                "accuracy": 50.0,
                "provenance": _dolci_provenance("math500"),
            }
            _write_json(run_dir / "math500_results.json", payload)
            result = campaign.read_downstream_result(run_dir, "math500", "dolci32k")
            self.assertEqual(result["primary_metric"], 50.0)

            payload["provenance"]["generation"]["max_new_tokens"] = 2048
            _write_json(run_dir / "math500_results.json", payload)
            with self.assertRaisesRegex(
                campaign.CampaignEvaluationError, r"generation\.max_new_tokens"
            ):
                campaign.read_downstream_result(run_dir, "math500", "dolci32k")

            payload["provenance"] = _dolci_provenance("math500")
            payload["provenance"]["evaluator"]["version"] = "0.8.0"
            _write_json(run_dir / "math500_results.json", payload)
            with self.assertRaisesRegex(
                campaign.CampaignEvaluationError, r"evaluator\.version"
            ):
                campaign.read_downstream_result(run_dir, "math500", "dolci32k")

    def test_dolci_mbpp_requires_dynamic_container_task_count_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _make_completed_run(Path(temporary))
            provenance = _dolci_provenance("mbpp_plus")
            provenance["dataset"] = {"n_tasks": 378}
            provenance["evaluator"]["dataset_task_count"] = 378
            payload = {
                "task": "mbpp_plus",
                "evaluation_scope": "full",
                "base_plus_extra_pass_at_1": 50.0,
                "provenance": provenance,
            }
            result_path = run_dir / "mbpp_plus_results.json"
            _write_json(result_path, payload)
            result = campaign.read_downstream_result(
                run_dir, "mbpp_plus", "dolci32k"
            )
            self.assertEqual(result["primary_metric"], 50.0)

            payload["provenance"]["evaluator"]["dataset_task_count"] = 377
            _write_json(result_path, payload)
            with self.assertRaisesRegex(
                campaign.CampaignEvaluationError, "dynamic task-count"
            ):
                campaign.read_downstream_result(run_dir, "mbpp_plus", "dolci32k")

class CampaignCollectorTests(unittest.TestCase):
    def test_dolci_collector_reports_every_missing_array_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status_dir = root / "status"
            status_dir.mkdir()
            statuses, results = campaign.collect_results(
                status_dir, root / "report", "campaign", "adamw", "dolci32k"
            )
            self.assertEqual(len(statuses), 35)
            self.assertFalse(results)
            self.assertEqual({row["status"] for row in statuses}, {"missing_status"})

    def test_collector_keeps_skips_and_collects_only_valid_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = _make_completed_run(root)
            _write_json(
                run_dir / "samsum_results.json",
                {"task": "samsum", "rougeL": 0.4},
            )
            status_dir = root / "status"
            status_dir.mkdir()
            rows = [
                {
                    "array_task": "0",
                    "campaign_id": "campaign",
                    "optimizer": "muon",
                    "setting": "alpaca_samsum",
                    "train": "alpaca",
                    "task": "samsum",
                    "method": "LayerwiseMuonSur",
                    "seed": "42",
                    "status": "evaluated",
                    "run_dir": str(run_dir),
                    "result_file": str(run_dir / "samsum_results.json"),
                    "detail": "ok",
                },
                {
                    "array_task": "1",
                    "campaign_id": "campaign",
                    "optimizer": "muon",
                    "setting": "alpaca_samsum",
                    "train": "alpaca",
                    "task": "samsum",
                    "method": "GlobalSoft",
                    "seed": "42",
                    "status": "skipped_missing",
                    "run_dir": "",
                    "result_file": "",
                    "detail": "missing",
                },
            ]
            for index, row in enumerate(rows):
                with (status_dir / f"{index:03d}.tsv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=campaign.STATUS_FIELDS, delimiter="\t"
                    )
                    writer.writeheader()
                    writer.writerow(row)

            output_dir = root / "report"
            statuses, results = campaign.collect_results(
                status_dir, output_dir, "campaign", "muon"
            )

            self.assertEqual(len(statuses), 2)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["primary_metric"], 0.4)
            self.assertTrue((output_dir / "evaluation_status.tsv").is_file())
            self.assertTrue((output_dir / "downstream_results.csv").is_file())
            self.assertTrue((output_dir / "downstream_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
