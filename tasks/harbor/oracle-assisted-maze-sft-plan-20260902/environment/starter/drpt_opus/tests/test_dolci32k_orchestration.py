from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Iterable, Sequence

from SFT.data.dolci32k.profile import (
    ADAMW_METHODS,
    MUON_METHODS,
    SETTINGS,
    SETTING_ORDER,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_LAUNCHER = REPO_ROOT / "SFT/train/submit_general_loss_comparison.sh"
TRAIN_WORKER = REPO_ROOT / "SFT/train/general_loss_comparison_job.sh"

def _isolated_environment(root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "DRPT_REPO_ROOT": str(REPO_ROOT),
            "DRPT_DATA_DIR": str(root / "data"),
            "DRPT_RUNS_DIR": str(root / "runs"),
            "DRPT_REPORTS_DIR": str(root / "reports"),
            "DRPT_LOGS_DIR": str(root / "logs"),
            "DRPT_PYTHON": sys.executable,
            "DRPT_SLURM_PARTITION": "test-partition",
            "DRPT_SLURM_QOS": "test-qos",
            "DRPT_ARTIFACT_BUILD_ID": "deadbeef",
            "DRPT_MODEL_PROFILE": "qwen3_1_7b",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _run_launcher(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["bash", str(TRAIN_LAUNCHER), *arguments, "--dry-run"],
        cwd=REPO_ROOT,
        env=_isolated_environment(root),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        raise AssertionError(
            f"launcher exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def _tab_rows(output: str) -> list[tuple[str, ...]]:
    return [
        tuple(line.split("\t"))
        for line in output.splitlines()
        if line.count("\t") == 3
    ]


def _dry_run_commands(output: str) -> list[list[str]]:
    commands: list[list[str]] = []
    for line in output.splitlines():
        if not line.startswith("[DRY-RUN] sbatch "):
            continue
        commands.append(shlex.split(line.removeprefix("[DRY-RUN]")))
    return commands


def _option(command: Sequence[str], name: str) -> str | None:
    prefix = f"{name}="
    values = [argument[len(prefix) :] for argument in command if argument.startswith(prefix)]
    if len(values) > 1:
        raise AssertionError(f"duplicate {name} options in {command!r}")
    return values[0] if values else None


def _training_matrix(
    families: Iterable[tuple[str, Sequence[str]]],
    settings: Sequence[str] = SETTING_ORDER,
) -> list[tuple[str, str, str, str]]:
    return [
        (setting, family, family, method)
        for family, methods in families
        for setting in settings
        for method in methods
    ]


def _downstream_matrix(
    families: Iterable[tuple[str, Sequence[str]]],
) -> list[tuple[str, str, str, str]]:
    return [
        (setting, family, method, str(task))
        for family, methods in families
        for setting in SETTING_ORDER
        for method in methods
        for task in SETTINGS[setting]["benchmarks"]
    ]


class Dolci32KLauncherDryRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.output = _run_launcher(
            cls.root,
            "--campaign-id",
            "dolci32k-orchestration-test",
            "--profile",
            "dolci32k",
            "--family",
            "all",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_training_matrix_is_canonical_25_plus_40_equals_65(self) -> None:
        rows = _tab_rows(self.output)
        training_rows = [row for row in rows if row[2] in ("adamw", "muon")]
        expected_adamw = _training_matrix((("adamw", ADAMW_METHODS),))
        expected_muon = _training_matrix((("muon", MUON_METHODS),))

        self.assertEqual(len(expected_adamw), 25)
        self.assertEqual(len(expected_muon), 40)
        self.assertEqual(len(training_rows), 65)
        self.assertEqual(training_rows, expected_adamw + expected_muon)
        self.assertIn("Default matrix: AdamW 25 + Muon 40 = 65 runs", self.output)

    def test_adamw_and_muon_smoke_dependency_graph_and_resources(self) -> None:
        commands = _dry_run_commands(self.output)
        training = [
            command
            for command in commands
            if command[-1] == "SFT/train/general_loss_comparison_job.sh"
        ]
        self.assertEqual(len(training), 5)
        by_array = {_option(command, "--array"): command for command in training}

        adamw_smoke = by_array["0-4%1"]
        self.assertEqual(_option(adamw_smoke, "--mem"), "48G")
        self.assertIsNone(_option(adamw_smoke, "--dependency"))
        self.assertIn("DRPT_MAX_STEPS=3", _option(adamw_smoke, "--export") or "")

        adamw_main = by_array["0-24%4"]
        self.assertEqual(_option(adamw_main, "--mem"), "48G")
        self.assertEqual(
            _option(adamw_main, "--dependency"), "afterok:<adamw-smoke-job>"
        )

        muon_smoke = by_array["0-7%1"]
        self.assertEqual(_option(muon_smoke, "--mem"), "192G")
        self.assertEqual(
            _option(muon_smoke, "--dependency"), "afterany:<adamw-main-job>"
        )
        self.assertIn("DRPT_MAX_STEPS=3", _option(muon_smoke, "--export") or "")

        light_array = "0-1,4-9,12-17,20-25,28-33,36-39%4"
        heavy_array = "2-3,10-11,18-19,26-27,34-35%2"
        muon_light = by_array[light_array]
        muon_heavy = by_array[heavy_array]
        self.assertEqual(_option(muon_light, "--mem"), "48G")
        self.assertEqual(_option(muon_heavy, "--mem"), "192G")
        self.assertEqual(
            _option(muon_light, "--dependency"), "afterok:<muon-smoke-job>"
        )
        self.assertEqual(
            _option(muon_heavy, "--dependency"), "afterok:<muon-smoke-job>"
        )

    def test_downstream_arrays_are_35_and_56_cells_afterany(self) -> None:
        rows = _tab_rows(self.output)
        downstream_rows = [row for row in rows if row[2] not in ("adamw", "muon")]
        expected_adamw = _downstream_matrix((("adamw", ADAMW_METHODS),))
        expected_muon = _downstream_matrix((("muon", MUON_METHODS),))
        self.assertEqual(len(expected_adamw), 35)
        self.assertEqual(len(expected_muon), 56)
        self.assertEqual(downstream_rows, expected_adamw + expected_muon)

        commands = _dry_run_commands(self.output)
        evaluation = [
            command
            for command in commands
            if command[-1] == "SFT/eval/campaign_downstream_eval_job.sh"
        ]
        self.assertEqual(len(evaluation), 2)
        by_array = {_option(command, "--array"): command for command in evaluation}
        self.assertEqual(
            _option(by_array["0-34%4"], "--dependency"), "afterany:900001"
        )
        self.assertEqual(
            _option(by_array["0-55%4"], "--dependency"),
            "afterany:900002:900003",
        )

    def test_family_and_combined_reports_wait_for_terminal_training(self) -> None:
        commands = _dry_run_commands(self.output)
        reports = [
            command
            for command in commands
            if command[-1] == "SFT/eval/plot_general_loss_campaign.sh"
        ]
        self.assertEqual(len(reports), 3)
        by_family: dict[str, list[str]] = {}
        for command in reports:
            export = _option(command, "--export") or ""
            report_family = next(
                field.split("=", 1)[1]
                for field in export.split(",")
                if field.startswith("DRPT_REPORT_FAMILY=")
            )
            by_family[report_family] = command

        self.assertEqual(
            _option(by_family["dolci32k-adamw"], "--dependency"),
            "afterany:<adamw-main-job>",
        )
        muon_terminal = (
            "afterany:<muon-main-light-job>:<muon-main-heavy-job>"
        )
        self.assertEqual(
            _option(by_family["dolci32k-muon"], "--dependency"), muon_terminal
        )
        self.assertEqual(_option(by_family["dolci32k"], "--dependency"), muon_terminal)

    def test_formal_dry_run_does_not_create_campaign_files(self) -> None:
        self.assertEqual(list(self.root.iterdir()), [])


class Dolci32KWorkerDryRunTests(unittest.TestCase):
    def _make_stub_repository(self, root: Path) -> Path:
        sft = root / "SFT"
        train = sft / "train"
        train.mkdir(parents=True)
        (sft / "__init__.py").write_text("", encoding="utf-8")
        (sft / "data").symlink_to(REPO_ROOT / "SFT/data", target_is_directory=True)
        stub = train / "train.sh"
        stub.write_text(
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            "printf '[STUB-TRAIN]'\n"
            "printf '\\t%s' \"$@\"\n"
            "printf '\\n'\n",
            encoding="utf-8",
        )
        stub.chmod(0o755)
        return stub

    def _run_worker(
        self, root: Path, family: str, array_index: int
    ) -> subprocess.CompletedProcess[str]:
        environment = _isolated_environment(root)
        environment.update(
            {
                "DRPT_REPO_ROOT": str(root),
                "DRPT_CAMPAIGN_ID": "dolci32k-worker-test",
                "DRPT_COMPARISON_PROFILE": "dolci32k",
                "DRPT_OPTIMIZER_FAMILY": family,
                "DRPT_DRY_RUN": "true",
                "SLURM_ARRAY_TASK_ID": str(array_index),
            }
        )
        return subprocess.run(
            ["bash", str(TRAIN_WORKER)],
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def test_every_training_worker_index_preserves_setting_major_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._make_stub_repository(root)
            family_methods = (("adamw", ADAMW_METHODS), ("muon", MUON_METHODS))
            observed: list[tuple[str, str, str]] = []

            for family, methods in family_methods:
                for array_index in range(len(SETTING_ORDER) * len(methods)):
                    expected_setting = SETTING_ORDER[array_index // len(methods)]
                    expected_method = methods[array_index % len(methods)]
                    result = self._run_worker(root, family, array_index)
                    self.assertEqual(
                        result.returncode,
                        0,
                        msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
                    )
                    comparison = next(
                        line
                        for line in result.stdout.splitlines()
                        if line.startswith("[comparison] profile=dolci32k ")
                    )
                    self.assertIn(f"family={family}", comparison)
                    self.assertIn(f"array_task={array_index}", comparison)
                    self.assertIn(f"setting={expected_setting}", comparison)
                    self.assertIn(f"optimizer={family}", comparison)
                    self.assertIn(f"method={expected_method}", comparison)

                    stub_line = next(
                        line
                        for line in result.stdout.splitlines()
                        if line.startswith("[STUB-TRAIN]\t")
                    )
                    arguments = stub_line.split("\t")[1:]
                    self.assertEqual(
                        arguments[:4],
                        [
                            "-c",
                            f"configs/dolci32k/{expected_setting}",
                            "-m",
                            expected_method,
                        ],
                    )
                    self.assertIn("--dry-run", arguments)
                    self.assertEqual(
                        arguments[arguments.index("--optimizer_type") + 1], family
                    )
                    self.assertEqual(
                        arguments[arguments.index("--artifact-build-id") + 1],
                        "deadbeef",
                    )
                    self.assertEqual(arguments[arguments.index("--lr") + 1], "1e-5")
                    if family == "muon":
                        self.assertEqual(
                            arguments[arguments.index("--muon-lr") + 1], "3e-4"
                        )
                        self.assertEqual(
                            arguments[arguments.index("--aux-adamw-lr") + 1],
                            "1e-5",
                        )
                    else:
                        self.assertNotIn("--muon-lr", arguments)
                        self.assertNotIn("--aux-adamw-lr", arguments)
                    observed.append((expected_setting, family, expected_method))

            self.assertEqual(
                len([row for row in observed if row[1] == "adamw"]), 25
            )
            self.assertEqual(len([row for row in observed if row[1] == "muon"]), 40)
            self.assertEqual(len(observed), 65)

            for family, invalid_index in (("adamw", 25), ("muon", 40)):
                result = self._run_worker(root, family, invalid_index)
                self.assertEqual(result.returncode, 2)
                self.assertIn("array task must be in", result.stderr)
                self.assertNotIn("[STUB-TRAIN]", result.stdout)


if __name__ == "__main__":
    unittest.main()
