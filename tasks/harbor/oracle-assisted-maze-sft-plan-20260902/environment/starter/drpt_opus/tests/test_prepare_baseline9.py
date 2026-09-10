from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from SFT.data import prepare_baseline9 as baseline9


def _write_record(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset": "fixture",
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


class Baseline9ManifestTests(unittest.TestCase):
    def test_manifest_has_the_exact_14_unique_campaign_files(self) -> None:
        self.assertEqual(len(baseline9.BASELINE9_REQUIRED_FILES), 14)
        self.assertEqual(len(set(baseline9.BASELINE9_REQUIRED_FILES)), 14)
        self.assertIn("train/alpaca/alpaca_data.jsonl", baseline9.BASELINE9_REQUIRED_FILES)
        self.assertIn(
            "eval/nq_open/nq_open_test_data.jsonl",
            baseline9.BASELINE9_REQUIRED_FILES,
        )

    def test_data_dir_prefers_cli_then_environment_then_repo_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cli_path = Path(temporary) / "cli"
            env_path = Path(temporary) / "env"
            with mock.patch.dict(os.environ, {"DRPT_DATA_DIR": str(env_path)}):
                self.assertEqual(baseline9.resolve_data_dir(str(cli_path)), cli_path.resolve())
                self.assertEqual(baseline9.resolve_data_dir(), env_path.resolve())

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(baseline9.resolve_data_dir(), baseline9.SCRIPT_DIR)

    def test_complete_manifest_is_idempotent_and_runs_no_downloader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            for relative_path in baseline9.BASELINE9_REQUIRED_FILES:
                _write_record(data_dir / relative_path)

            with mock.patch.object(baseline9.subprocess, "run") as run:
                exit_code = baseline9.main(["--data-dir", str(data_dir)])

            self.assertEqual(exit_code, 0)
            run.assert_not_called()

    def test_corrupt_or_missing_file_marks_only_its_source_group_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            for relative_path in baseline9.BASELINE9_REQUIRED_FILES:
                _write_record(data_dir / relative_path)

            corrupt = data_dir / "train/alpaca/alpaca_data.jsonl"
            corrupt.write_text("not-json\n", encoding="utf-8")
            statuses = baseline9.inspect_manifest(data_dir)
            missing = baseline9.missing_requirements(statuses)

            self.assertEqual([dataset.key for dataset in missing], ["alpaca"])
            self.assertIn("invalid JSONL", statuses["train/alpaca/alpaca_data.jsonl"].detail)

    def test_check_only_returns_nonzero_without_calling_downloader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(baseline9.subprocess, "run") as run:
                exit_code = baseline9.main(
                    ["--data-dir", temporary, "--check-only"]
                )

            self.assertEqual(exit_code, 1)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
