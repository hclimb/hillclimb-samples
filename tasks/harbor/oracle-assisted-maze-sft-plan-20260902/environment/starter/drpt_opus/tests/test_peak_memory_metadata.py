from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from SFT.train.train import (
    _process_peak_rss_bytes,
    _reset_cuda_peak_memory_stats,
    _runtime_environment,
    _write_run_metadata,
)


def _training_args(output_dir: str) -> SimpleNamespace:
    return SimpleNamespace(
        local_rank=-1,
        output_dir=output_dir,
        method="FullTraining",
        optimizer_type="adamw",
        learning_rate=1e-5,
        muon_learning_rate=None,
        aux_adamw_learning_rate=None,
        optimizer_aware_muon_backend="auto",
        selection_frac=1.0,
        selection_mode="topk",
        scoring_method="reduced_ghost",
        subset_mode="one_pass",
        val_strategy="separate_batch",
        seed=42,
        data_seed=42,
        train_dataset_names=["toy"],
        analysis_dataset="toy",
        device="cpu",
        soft_weighting_steps=20,
        soft_weighting_lr=0.1,
        soft_weighting_tol=1e-5,
        soft_weighting_patience=3,
        soft_weighting_gamma=0.0,
        soft_weighting_use_optimizer_state=True,
        soft_weighting_constraint="capped_simplex",
        soft_replay_precision="bf16_fp32",
        muon_surrogate_alpha=1.0,
        muon_surrogate_rank=8,
        muon_surrogate_full_svd_max_dim=64,
        muon_surrogate_rtol=1e-6,
        muon_surrogate_oversample=4,
        muon_surrogate_power_iters=1,
        muon_surrogate_include_adamw_scores=False,
        muon_surrogate_mode_weighting="uniform",
        muon_surrogate_saturation=False,
        optimizer_aware_token_normalized_selection=False,
    )


class _RuntimeOptimizer:
    def get_runtime_metadata(self):
        return {
            "optimizer_runtime_class": "tests.ResolvedOptimizer",
            "muon_backend_resolved": "torch",
        }


class PeakMemoryMetadataTests(unittest.TestCase):
    def test_linux_process_peak_rss_is_converted_from_kib_to_bytes(self):
        usage = SimpleNamespace(ru_maxrss=123)
        with (
            patch("SFT.train.train.sys.platform", "linux"),
            patch("SFT.train.train.resource.getrusage", return_value=usage),
        ):
            self.assertEqual(_process_peak_rss_bytes(), 123 * 1024)

    def test_cpu_runtime_environment_keeps_explicit_cuda_fields(self):
        with (
            patch("SFT.train.train.torch.cuda.is_available", return_value=False),
            patch("SFT.train.train._process_peak_rss_bytes", return_value=987654),
        ):
            payload = _runtime_environment(SimpleNamespace(device="cpu"))

        self.assertEqual(payload["process_peak_rss_bytes"], 987654)
        self.assertIsNone(payload["cuda_max_memory_allocated_bytes"])
        self.assertIsNone(payload["cuda_max_memory_reserved_bytes"])

    def test_cuda_runtime_environment_reads_current_device_peak_counters(self):
        with (
            patch("SFT.train.train.torch.cuda.is_available", return_value=True),
            patch("SFT.train.train.torch.cuda.current_device", return_value=2),
            patch(
                "SFT.train.train.torch.cuda.max_memory_allocated",
                return_value=111,
            ) as allocated,
            patch(
                "SFT.train.train.torch.cuda.max_memory_reserved",
                return_value=222,
            ) as reserved,
            patch("SFT.train.train.torch.cuda.get_device_name", return_value="A40"),
            patch(
                "SFT.train.train.torch.cuda.get_device_capability",
                return_value=(8, 6),
            ),
            patch("SFT.train.train._process_peak_rss_bytes", return_value=333),
        ):
            payload = _runtime_environment(SimpleNamespace(device="cuda:2"))

        allocated.assert_called_once_with(2)
        reserved.assert_called_once_with(2)
        self.assertEqual(payload["cuda_max_memory_allocated_bytes"], 111)
        self.assertEqual(payload["cuda_max_memory_reserved_bytes"], 222)
        self.assertEqual(payload["process_peak_rss_bytes"], 333)
        self.assertEqual(payload["gpu_name"], "A40")

    def test_cuda_peak_reset_targets_current_device(self):
        with (
            patch("SFT.train.train.torch.cuda.is_available", return_value=True),
            patch("SFT.train.train.torch.cuda.current_device", return_value=3),
            patch("SFT.train.train.torch.cuda.reset_peak_memory_stats") as reset,
        ):
            _reset_cuda_peak_memory_stats()
        reset.assert_called_once_with(3)

    def test_training_complete_metadata_atomically_keeps_peaks_and_optimizer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = _training_args(temp_dir)
            model_args = SimpleNamespace(
                model_name_or_path="toy/model", model_revision="revision"
            )
            data_args = SimpleNamespace(
                max_seq_length=128,
                experiment_profile="dolci32k",
                setting_id="inst_if",
                artifact_build_id="build-id",
            )
            runtime = {
                "device": "cpu",
                "process_peak_rss_bytes": 444,
                "cuda_max_memory_allocated_bytes": 555,
                "cuda_max_memory_reserved_bytes": 666,
            }
            with patch("SFT.train.train._runtime_environment", return_value=runtime):
                _write_run_metadata(
                    args,
                    model_args,
                    data_args,
                    optimizer=_RuntimeOptimizer(),
                    profile_metadata={"artifact": "sha256"},
                    lifecycle_phase="training_complete",
                )

            output_dir = Path(temp_dir)
            payload = json.loads((output_dir / "run_metadata.json").read_text())
            self.assertEqual(payload["metadata_lifecycle_phase"], "training_complete")
            self.assertEqual(payload["runtime_environment"], runtime)
            self.assertEqual(payload["muon_backend_resolved"], "torch")
            self.assertEqual(payload["soft_replay_precision"], "bf16_fp32")
            self.assertEqual(payload["profile"], {"artifact": "sha256"})
            self.assertEqual(list(output_dir.glob("run_metadata.json.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
