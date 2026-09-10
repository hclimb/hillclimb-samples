from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from SFT.eval.task_registry import default_max_new_tokens
from SFT.eval.tasks.common import clean_model_response, render_generation_chat
from SFT.eval.tasks.math500 import score_math_answer
from SFT.eval.tasks.bench_data import load_bench_records
from SFT.eval.tasks.mbpp_plus import (
    DEFAULT_EVALPLUS_IMAGE,
    add_official_smoke_fillers,
    build_evalplus_command,
    build_evalplus_probe_command,
    evalplus_result_filename,
    inspect_evalplus_runtime,
    parse_evalplus_results,
    resolve_evalplus_dataset_path,
    resolve_evalplus_image,
    select_task_rows,
    validate_official_registry_coverage,
)


class _TemplateTokenizer:
    chat_template = "stub"

    def __init__(self) -> None:
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return f"PROMPT:{messages[0]['content']}"


class DolciGenerationPolicyTests(unittest.TestCase):
    def test_fixed_final_benchmark_generation_lengths(self) -> None:
        self.assertEqual(default_max_new_tokens("ifeval"), 2048)
        self.assertEqual(default_max_new_tokens("ifbench"), 2048)
        self.assertEqual(default_max_new_tokens("math500"), 4096)
        self.assertEqual(default_max_new_tokens("mbpp_plus"), 2048)

    def test_render_chat_passes_explicit_thinking_policy(self) -> None:
        tokenizer = _TemplateTokenizer()
        render_generation_chat(tokenizer, "question", enable_thinking=True)
        self.assertIs(tokenizer.kwargs["enable_thinking"], True)
        self.assertIs(tokenizer.kwargs["add_generation_prompt"], True)

    def test_scored_response_strips_thinking_but_raw_is_retained(self) -> None:
        raw = "<think>private trace</think>\n\\boxed{2}<|im_end|>ignored"
        scored = score_math_answer(
            "2",
            raw,
            parse_fn=lambda value: [value],
            verify_fn=lambda gold, prediction: "2" in prediction[0],
        )
        self.assertTrue(scored["correct"])
        self.assertEqual(scored["raw_generation"], raw)
        self.assertEqual(scored["scored_response"], "\\boxed{2}")
        self.assertEqual(clean_model_response(raw), "\\boxed{2}")


class DolciArtifactPinTests(unittest.TestCase):
    def test_dolci_profile_never_falls_back_without_campaign_build_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fallback = Path(temporary) / "eval" / "ifeval"
            fallback.mkdir(parents=True)
            (fallback / "ifeval_bench_data.jsonl").write_text(
                json.dumps({"id": "legacy"}) + "\n", encoding="utf-8"
            )
            with mock.patch.dict(
                os.environ, {"DRPT_DOWNSTREAM_PROFILE": "dolci32k"}, clear=True
            ):
                with self.assertRaisesRegex(RuntimeError, "DRPT_ARTIFACT_BUILD_ID"):
                    load_bench_records(temporary, "ifeval")

    def test_dolci_profile_forwards_pinned_build_id_to_artifact_api(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "ifeval.jsonl"
            artifact.write_text(json.dumps({"id": "official"}) + "\n", encoding="utf-8")
            environment = {
                "DRPT_DOWNSTREAM_PROFILE": "dolci32k",
                "DRPT_ARTIFACT_BUILD_ID": "abc123",
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                with mock.patch(
                    "SFT.data.dolci32k.artifact_path", return_value=artifact
                ) as resolver:
                    self.assertEqual(load_bench_records(temporary, "ifeval")[0]["id"], "official")
            resolver.assert_called_once_with(
                temporary, "benchmarks", "ifeval", build_id="abc123", validate=True
            )


class EvalPlusPolicyTests(unittest.TestCase):
    def test_apptainer_command_is_contained_and_network_off(self) -> None:
        command = build_evalplus_command(
            runner="apptainer",
            image="image.sif",
            output_dir="/tmp/eval-output",
            dataset_path="/tmp/MbppPlus-v0.2.0.jsonl",
            runtime_cache_path="/tmp/eval-output/runtime_cache",
        )
        self.assertIn("--containall", command)
        self.assertIn("--cleanenv", command)
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertNotIn("--output-file", command)
        self.assertIn(
            "/tmp/MbppPlus-v0.2.0.jsonl:/opt/drpt/evalplus/MbppPlus-v0.2.0.jsonl:ro",
            command,
        )
        self.assertIn(
            "MBPP_OVERRIDE_PATH=/opt/drpt/evalplus/MbppPlus-v0.2.0.jsonl",
            command,
        )
        self.assertIn(
            "/tmp/eval-output/mbpp_plus_samples.jsonl:/workspace/mbpp_plus_samples.jsonl:ro",
            command,
        )
        self.assertIn("/tmp/eval-output:/workspace", command)
        self.assertIn("XDG_CACHE_HOME=/workspace/runtime_cache", command)
        self.assertNotIn("/opt/drpt/evalplus-runtime-cache", command)
        self.assertEqual(
            evalplus_result_filename("mbpp_plus_samples.jsonl"),
            "mbpp_plus_samples_eval_results.json",
        )

    def test_plus_metric_is_explicitly_base_plus_extra(self) -> None:
        payload = {
            "eval": {
                "Mbpp/1": [{"base_status": "pass", "plus_status": "pass"}],
                "Mbpp/2": [{"base_status": "pass", "plus_status": "fail"}],
            },
        }
        result = parse_evalplus_results(payload, ["Mbpp/1", "Mbpp/2"])
        self.assertEqual(result["base_pass_at_1"], 100.0)
        self.assertEqual(result["plus_pass_at_1"], 50.0)
        self.assertEqual(result["base_plus_extra_pass_at_1"], 50.0)

    def test_limited_metrics_allow_official_canonical_fillers(self) -> None:
        payload = {
            "eval": {
                "Mbpp/1": [{"base_status": "pass", "plus_status": "fail"}],
                "Mbpp/2": [{"base_status": "pass", "plus_status": "pass"}],
            }
        }
        with self.assertRaisesRegex(RuntimeError, "coverage mismatch"):
            parse_evalplus_results(payload, ["Mbpp/1"])
        result = parse_evalplus_results(
            payload, ["Mbpp/1"], allow_extra_tasks=True
        )
        self.assertEqual(result["n_test"], 1)
        self.assertEqual(result["base_plus_extra_pass_at_1"], 0.0)

    def test_limited_smoke_fills_unselected_registry_tasks(self) -> None:
        problems = {
            "Mbpp/1": {"prompt": "def one():", "canonical_solution": "\n return 1"},
            "Mbpp/2": {"prompt": "def two():", "canonical_solution": "\n return 2"},
        }
        samples = [{"task_id": "Mbpp/1", "solution": "def one():\n return 0"}]
        filled = add_official_smoke_fillers(samples, problems, ["Mbpp/1"])
        self.assertEqual([row["task_id"] for row in filled], ["Mbpp/1", "Mbpp/2"])
        self.assertEqual(filled[1]["solution"], "def two():\n return 2")

    def test_limited_raw_result_requires_full_dynamic_registry(self) -> None:
        payload = {"eval": {"Mbpp/1": [], "Mbpp/2": []}}
        validate_official_registry_coverage(payload, ["Mbpp/1", "Mbpp/2"])
        with self.assertRaisesRegex(RuntimeError, "canonical-filler coverage"):
            validate_official_registry_coverage(payload, ["Mbpp/1", "Mbpp/2", "Mbpp/3"])

    def test_limited_smoke_uses_prefix_after_full_registry_validation(self) -> None:
        records = [
            {
                "id": task_id,
                "metadata": {"task_id": task_id, "prompt": f"prompt {task_id}"},
            }
            for task_id in ("Mbpp/1", "Mbpp/2", "Mbpp/3")
        ]
        selected, scope = select_task_rows(
            records, ["Mbpp/1", "Mbpp/2", "Mbpp/3"], 2
        )
        self.assertEqual([task_id for task_id, _ in selected], ["Mbpp/1", "Mbpp/2"])
        self.assertEqual(scope, "limited")

        selected, scope = select_task_rows(
            records, ["Mbpp/1", "Mbpp/2", "Mbpp/3"], -1
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(scope, "full")

    def test_limited_smoke_still_rejects_incomplete_full_artifact(self) -> None:
        records = [
            {"id": "Mbpp/1", "metadata": {"task_id": "Mbpp/1", "prompt": "p1"}},
            {"id": "Mbpp/2", "metadata": {"task_id": "Mbpp/2", "prompt": "p2"}},
        ]
        with self.assertRaisesRegex(RuntimeError, "does not match get_mbpp_plus"):
            select_task_rows(records, ["Mbpp/1", "Mbpp/2", "Mbpp/3"], 1)

    def test_evalplus_image_override_must_equal_profile_pin(self) -> None:
        self.assertEqual(resolve_evalplus_image(), DEFAULT_EVALPLUS_IMAGE)
        with mock.patch.dict(
            os.environ, {"DRPT_EVALPLUS_IMAGE": "docker://ganler/evalplus:v0.3.1"}
        ):
            with self.assertRaisesRegex(RuntimeError, "exact EvalPlus OCI image pin"):
                resolve_evalplus_image()

    def test_dataset_cache_is_hash_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "MbppPlus-v0.2.0.jsonl"
            path.write_text("fixture\n", encoding="utf-8")
            with mock.patch(
                "SFT.eval.tasks.mbpp_plus._sha256_file",
                return_value=(
                    "b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad"
                ),
            ):
                resolved, digest = resolve_evalplus_dataset_path(str(path))
        self.assertEqual(resolved, str(path))
        self.assertEqual(
            digest,
            "b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad",
        )

    def test_container_preflight_checks_real_image_distribution_version(self) -> None:
        command = build_evalplus_probe_command(
            runner="apptainer",
            image="image.sif",
            dataset_path="/tmp/MbppPlus-v0.2.0.jsonl",
            runtime_cache_path="/tmp/evalplus-probe/runtime_cache",
        )
        self.assertIn("python", command)
        completed = mock.Mock(
            stdout=(
                '{"distribution_version":"0.4.0.dev2",'
                '"dataset_sha256":"b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad",'
                '"task_count":378,"runtime_cache_writable":true}\n'
            )
        )
        with mock.patch("SFT.eval.tasks.mbpp_plus.subprocess.run", return_value=completed):
            payload = inspect_evalplus_runtime(
                runner="apptainer",
                image="image.sif",
                dataset_path="/tmp/MbppPlus-v0.2.0.jsonl",
                runtime_cache_path="/tmp/evalplus-probe/runtime_cache",
                expected_task_count=378,
            )
        self.assertEqual(payload["distribution_version"], "0.4.0.dev2")
        self.assertEqual(payload["task_count"], 378)
        self.assertIs(payload["runtime_cache_writable"], True)



if __name__ == "__main__":
    unittest.main()
