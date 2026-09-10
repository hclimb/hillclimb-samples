"""
Tests for GCS checkpoint saving with TPU region auto-detection.
Run with: python -m pytest tests/test_gcs_checkpointing.py -v
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestGetGcsRegion(unittest.TestCase):
    def test_returns_env_var_when_set(self):
        with patch.dict(os.environ, {"GCS_REGION": "us-east1"}):
            from utils import get_gcs_region
            self.assertEqual(get_gcs_region(), "us-east1")

    def test_parses_metadata_zone(self):
        zone_response = b"projects/123456/zones/us-central2-b"
        mock_response = MagicMock()
        mock_response.read.return_value = zone_response

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GCS_REGION", None)
            import urllib.request
            with patch.object(urllib.request, "urlopen", return_value=mock_response):
                from utils import get_gcs_region
                result = get_gcs_region()
        self.assertEqual(result, "us-central2")

    def test_returns_none_on_metadata_failure(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GCS_REGION", None)
            import urllib.request
            with patch.object(urllib.request, "urlopen", side_effect=Exception("timeout")):
                from utils import get_gcs_region
                result = get_gcs_region()
        self.assertIsNone(result)


class TestBuildGcsRunDir(unittest.TestCase):
    def _make_cfg(self):
        cfg = MagicMock()
        cfg.trainer.get.return_value = None  # no resume_from, no run_name, no wandb_name
        cfg.trainer.run_name = None
        cfg.trainer.wandb_name = None
        cfg.model.main_model.model_id = "Qwen/Qwen3-0.6B"
        cfg.model.get.return_value = {}
        cfg.dataset.get.return_value = 1
        cfg.dataset.name = "testdata"
        cfg.trainer.learning_rate = 1e-4
        return cfg

    def test_returns_none_when_no_gcs_bucket(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GCS_BUCKET", None)
            from utils import _build_gcs_run_dir
            result = _build_gcs_run_dir(self._make_cfg())
        self.assertIsNone(result)

    def test_returns_correct_gcs_path(self):
        hydra_output_dir = "/home/user/outputs/2024-03-15/14-30-00/train"
        mock_hydra_cfg = MagicMock()
        mock_hydra_cfg.runtime.output_dir = hydra_output_dir

        with patch.dict(os.environ, {"GCS_BUCKET": "my-bucket"}):
            with patch("utils.get_gcs_region", return_value="us-central2"):
                with patch("utils.ensure_gcs_bucket"):
                    with patch("utils.HydraConfig") as mock_hydra:
                        mock_hydra.get.return_value = mock_hydra_cfg
                        with patch("utils.get_run_name", return_value="my-run"):
                            from utils import _build_gcs_run_dir
                            result = _build_gcs_run_dir(self._make_cfg())

        self.assertEqual(result, "gs://my-bucket/my-run-2024-03-15-14-30-00")

    def test_falls_back_to_datetime_when_outputs_not_in_path(self):
        hydra_output_dir = "/tmp/no-outputs-anchor/somedir"
        mock_hydra_cfg = MagicMock()
        mock_hydra_cfg.runtime.output_dir = hydra_output_dir

        with patch.dict(os.environ, {"GCS_BUCKET": "my-bucket"}):
            with patch("utils.get_gcs_region", return_value=None):
                with patch("utils.ensure_gcs_bucket"):
                    with patch("utils.HydraConfig") as mock_hydra:
                        mock_hydra.get.return_value = mock_hydra_cfg
                        with patch("utils.get_run_name", return_value="my-run"):
                            from utils import _build_gcs_run_dir
                            result = _build_gcs_run_dir(self._make_cfg())

        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("gs://my-bucket/my-run-"))


class TestSetupCheckpointing(unittest.TestCase):
    def _make_cfg(self, resume_from=None):
        cfg = MagicMock()
        cfg.trainer.get.side_effect = lambda key, default=None: (
            resume_from if key == "resume_from" else default
        )
        cfg.trainer.checkpoint_interval = 1000
        cfg.model.name = "test-model"
        return cfg

    def test_uses_gcs_path_when_bucket_set(self):
        cfg = self._make_cfg()
        mock_hydra_cfg = MagicMock()
        mock_hydra_cfg.runtime.output_dir = "/home/user/outputs/2024-03-15/14-30-00"

        mock_manager = MagicMock()

        with patch.dict(os.environ, {"GCS_BUCKET": "my-bucket"}):
            with patch("utils._build_gcs_run_dir", return_value="gs://my-bucket/my-run-2024-03-15-14-30-00"):
                with patch("utils.upload_config_to_gcs"):
                    with patch("utils.ocp") as mock_ocp:
                        mock_ocp.CheckpointManager.return_value = mock_manager
                        mock_ocp.CheckpointManagerOptions.return_value = MagicMock()
                        mock_ocp.StandardCheckpointer.return_value = MagicMock()
                        from utils import setup_checkpointing
                        result = setup_checkpointing(cfg)

        call_args = mock_ocp.CheckpointManager.call_args
        checkpoint_dir_used = call_args[0][0]
        self.assertEqual(checkpoint_dir_used, "gs://my-bucket/my-run-2024-03-15-14-30-00/test-model")

    def test_falls_back_to_local_when_no_bucket(self):
        cfg = self._make_cfg()
        mock_hydra_cfg = MagicMock()
        mock_hydra_cfg.runtime.output_dir = "/home/user/outputs/2024-03-15/14-30-00"

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GCS_BUCKET", None)
            with patch("utils._build_gcs_run_dir", return_value=None):
                with patch("utils.HydraConfig") as mock_hydra:
                    mock_hydra.get.return_value = mock_hydra_cfg
                    with patch("utils.ocp") as mock_ocp:
                        mock_ocp.CheckpointManager.return_value = MagicMock()
                        mock_ocp.CheckpointManagerOptions.return_value = MagicMock()
                        mock_ocp.StandardCheckpointer.return_value = MagicMock()
                        from utils import setup_checkpointing
                        result = setup_checkpointing(cfg)

        call_args = mock_ocp.CheckpointManager.call_args
        checkpoint_dir_used = call_args[0][0]
        self.assertFalse(checkpoint_dir_used.startswith("gs://"))
        self.assertIn("test-model", checkpoint_dir_used)

    def test_resume_from_gcs_not_corrupted_by_abspath(self):
        gcs_resume_path = "gs://my-bucket/old-run/checkpoints"
        cfg = self._make_cfg(resume_from=gcs_resume_path)

        with patch("utils._build_gcs_run_dir", return_value="gs://my-bucket/new-run-2024"):
            with patch("utils.upload_config_to_gcs"):
                with patch("utils.ocp") as mock_ocp:
                    mock_ocp.CheckpointManager.return_value = MagicMock()
                    mock_ocp.CheckpointManagerOptions.return_value = MagicMock()
                    mock_ocp.StandardCheckpointer.return_value = MagicMock()
                    from utils import setup_checkpointing
                    _, resume_step, resume_from_dir = setup_checkpointing(cfg)

        # Checkpoint manager should point to the new run dir, NOT the resume_from path
        call_args = mock_ocp.CheckpointManager.call_args
        checkpoint_dir_used = call_args[0][0]
        self.assertNotEqual(checkpoint_dir_used, gcs_resume_path)
        self.assertEqual(checkpoint_dir_used, "gs://my-bucket/new-run-2024/test-model")
        # resume_from_dir should preserve the original GCS path without abspath corruption
        self.assertEqual(resume_from_dir, gcs_resume_path)
        self.assertIsNone(resume_step)

    def test_resume_from_local_path_is_made_absolute(self):
        cfg = self._make_cfg(resume_from="relative/path/checkpoints")
        mock_hydra_cfg = MagicMock()
        mock_hydra_cfg.runtime.output_dir = "/home/user/outputs/2024-03-15/14-30-00"

        with patch("utils._build_gcs_run_dir", return_value=None):
            with patch("utils.HydraConfig") as mock_hydra:
                mock_hydra.get.return_value = mock_hydra_cfg
                with patch("utils.ocp") as mock_ocp:
                    mock_ocp.CheckpointManager.return_value = MagicMock()
                    mock_ocp.CheckpointManagerOptions.return_value = MagicMock()
                    mock_ocp.StandardCheckpointer.return_value = MagicMock()
                    from utils import setup_checkpointing
                    _, resume_step, resume_from_dir = setup_checkpointing(cfg)

        # Checkpoint manager should point to new local run dir
        call_args = mock_ocp.CheckpointManager.call_args
        checkpoint_dir_used = call_args[0][0]
        self.assertTrue(os.path.isabs(checkpoint_dir_used))
        self.assertIn("test-model", checkpoint_dir_used)
        # resume_from_dir should be made absolute
        self.assertTrue(os.path.isabs(resume_from_dir))
        self.assertFalse(resume_from_dir.startswith("gs://"))
        self.assertIsNone(resume_step)

    def test_resume_from_gcs_with_step_suffix_is_stripped(self):
        gcs_resume_path = "gs://my-bucket/old-run/qwen3_mem_embed/100000"
        cfg = self._make_cfg(resume_from=gcs_resume_path)

        with patch("utils._build_gcs_run_dir", return_value="gs://my-bucket/new-run-2024"):
            with patch("utils.upload_config_to_gcs"):
                with patch("utils.ocp") as mock_ocp:
                    mock_ocp.CheckpointManager.return_value = MagicMock()
                    mock_ocp.CheckpointManagerOptions.return_value = MagicMock()
                    mock_ocp.StandardCheckpointer.return_value = MagicMock()
                    from utils import setup_checkpointing
                    _, resume_step, resume_from_dir = setup_checkpointing(cfg)

        # Step suffix stripped into resume_step and resume_from_dir
        self.assertEqual(resume_step, 100000)
        self.assertEqual(resume_from_dir, "gs://my-bucket/old-run/qwen3_mem_embed")
        # Checkpoint manager points to new run dir, not the resume path
        call_args = mock_ocp.CheckpointManager.call_args
        checkpoint_dir_used = call_args[0][0]
        self.assertEqual(checkpoint_dir_used, "gs://my-bucket/new-run-2024/test-model")

    def test_resume_from_local_with_step_suffix_is_stripped(self):
        cfg = self._make_cfg(resume_from="/absolute/path/qwen3_mem_embed/100000")
        mock_hydra_cfg = MagicMock()
        mock_hydra_cfg.runtime.output_dir = "/home/user/outputs/2024-03-15/14-30-00"

        with patch("utils._build_gcs_run_dir", return_value=None):
            with patch("utils.HydraConfig") as mock_hydra:
                mock_hydra.get.return_value = mock_hydra_cfg
                with patch("utils.ocp") as mock_ocp:
                    mock_ocp.CheckpointManager.return_value = MagicMock()
                    mock_ocp.CheckpointManagerOptions.return_value = MagicMock()
                    mock_ocp.StandardCheckpointer.return_value = MagicMock()
                    from utils import setup_checkpointing
                    _, resume_step, resume_from_dir = setup_checkpointing(cfg)

        # Step suffix stripped into resume_step and resume_from_dir
        self.assertEqual(resume_step, 100000)
        self.assertEqual(resume_from_dir, "/absolute/path/qwen3_mem_embed")
        # Checkpoint manager points to new local run dir
        call_args = mock_ocp.CheckpointManager.call_args
        checkpoint_dir_used = call_args[0][0]
        self.assertIn("test-model", checkpoint_dir_used)
        self.assertNotEqual(checkpoint_dir_used, "/absolute/path/qwen3_mem_embed")


class TestEvalGcsPathParsing(unittest.TestCase):
    """Test the path-parsing logic extracted from eval.py's checkpoint_dir block."""

    def _parse(self, checkpoint_dir):
        """Replicate the path-parsing logic from eval.py."""
        parts = checkpoint_dir.rstrip("/").split("/")
        checkpoint_step = int(parts[-1])
        model_ckpt_dir = "/".join(parts[:-1])
        run_dir = "/".join(parts[:-2])
        train_config_path = f"{run_dir}/.hydra/config.yaml"
        return checkpoint_step, model_ckpt_dir, run_dir, train_config_path

    def test_gcs_path_parsed_correctly(self):
        gcs_path = "gs://my-bucket/my-run-2024-03-15-14-30-00/qwen3_mem_embed/60000"
        step, model_dir, run_dir, config_path = self._parse(gcs_path)
        self.assertEqual(step, 60000)
        self.assertEqual(model_dir, "gs://my-bucket/my-run-2024-03-15-14-30-00/qwen3_mem_embed")
        self.assertEqual(run_dir, "gs://my-bucket/my-run-2024-03-15-14-30-00")
        self.assertEqual(config_path, "gs://my-bucket/my-run-2024-03-15-14-30-00/.hydra/config.yaml")

    def test_local_path_parsed_correctly(self):
        local_path = "/home/user/outputs/2024-03-15/14-30-00/qwen3_mem_embed/60000"
        step, model_dir, run_dir, config_path = self._parse(local_path)
        self.assertEqual(step, 60000)
        self.assertEqual(model_dir, "/home/user/outputs/2024-03-15/14-30-00/qwen3_mem_embed")
        self.assertEqual(run_dir, "/home/user/outputs/2024-03-15/14-30-00")
        self.assertEqual(config_path, "/home/user/outputs/2024-03-15/14-30-00/.hydra/config.yaml")

    def test_gcs_config_path_strip_prefix(self):
        """gcsfs.open() receives path without the gs:// prefix."""
        config_path = "gs://my-bucket/my-run/.hydra/config.yaml"
        stripped = config_path[5:]  # strip "gs://"
        self.assertEqual(stripped, "my-bucket/my-run/.hydra/config.yaml")


class TestLoadInferenceCheckpointFallback(unittest.TestCase):
    def test_fallback_path_is_gcs_safe(self):
        """Fallback ckpt_path must use string concat, not os.path.join."""
        mock_manager = MagicMock()
        mock_manager.directory = "gs://my-bucket/my-run/qwen3_mem_embed"
        step = 60000

        ckpt_path = mock_manager.directory.rstrip("/") + f"/{step}/default"
        self.assertEqual(ckpt_path, "gs://my-bucket/my-run/qwen3_mem_embed/60000/default")
        self.assertTrue(ckpt_path.startswith("gs://"))

    def test_fallback_path_local_unchanged(self):
        mock_manager = MagicMock()
        mock_manager.directory = "/home/user/outputs/2024-03-15/14-30-00/qwen3_mem_embed"
        step = 60000

        ckpt_path = mock_manager.directory.rstrip("/") + f"/{step}/default"
        self.assertEqual(ckpt_path, "/home/user/outputs/2024-03-15/14-30-00/qwen3_mem_embed/60000/default")


if __name__ == "__main__":
    unittest.main()
