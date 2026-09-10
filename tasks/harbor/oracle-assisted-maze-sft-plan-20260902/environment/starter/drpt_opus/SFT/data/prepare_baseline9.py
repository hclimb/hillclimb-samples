#!/usr/bin/env python3
"""Prepare and preflight the 14 files required by the baseline9 campaign.

The command is safe to rerun: a source dataset is prepared only when one of
its required baseline9 outputs is missing, empty, or not valid JSONL. The
underlying conversion work remains in ``prepare_datasets.py``; this file is
only the baseline-specific manifest and orchestration layer.

The data root is resolved in this order: ``--data-dir``, ``DRPT_DATA_DIR``,
then the directory containing this script (``SFT/data``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PREPARE_SCRIPT = SCRIPT_DIR / "prepare_datasets.py"


@dataclass(frozen=True)
class DatasetRequirement:
    """One Hugging Face source/conversion and its baseline9 outputs."""

    key: str
    description: str
    required_files: tuple[str, ...]


# Keep this manifest in sync with get_train_dataset.py, get_val_dataset.py,
# and the four baseline9 defaults.yaml files. LR/dev files are useful extra
# outputs produced by the converters, but the canonical campaign uses the
# validation split for selection and the test split for evaluation.
BASELINE9_DATASETS: tuple[DatasetRequirement, ...] = (
    DatasetRequirement(
        "alpaca",
        "Alpaca training pool",
        ("train/alpaca/alpaca_data.jsonl",),
    ),
    DatasetRequirement(
        "dolly",
        "LESS Dolly component",
        ("train/dolly/dolly_data.jsonl",),
    ),
    DatasetRequirement(
        "oasst1",
        "LESS OASST1 component",
        ("train/oasst1/oasst1_data.jsonl",),
    ),
    DatasetRequirement(
        "triviaqa_train",
        "TriviaQA training pool",
        ("train/triviaqa/triviaqa_data.jsonl",),
    ),
    DatasetRequirement(
        "samsum",
        "SamSUM selection/evaluation splits",
        (
            "eval/samsum/samsum_validation_data.jsonl",
            "eval/samsum/samsum_test_data.jsonl",
        ),
    ),
    DatasetRequirement(
        "tydiqa",
        "TyDiQA selection/evaluation splits",
        (
            "eval/tydiqa/tydiqa_validation_data.jsonl",
            "eval/tydiqa/tydiqa_test_data.jsonl",
        ),
    ),
    DatasetRequirement(
        "nq_open_eval",
        "NQ-open selection/evaluation splits",
        (
            "eval/nq_open/nq_open_validation_data.jsonl",
            "eval/nq_open/nq_open_test_data.jsonl",
        ),
    ),
    DatasetRequirement(
        "squad_eval",
        "SQuAD selection/evaluation splits",
        (
            "eval/squad/squad_validation_data.jsonl",
            "eval/squad/squad_test_data.jsonl",
        ),
    ),
    DatasetRequirement(
        "flan_v2",
        "LESS FLAN-v2 component (100k subset)",
        ("train/flan_v2/flan_v2_data.jsonl",),
    ),
    DatasetRequirement(
        "cot",
        "LESS CoT-Collection component",
        ("train/cot/cot_data.jsonl",),
    ),
)

BASELINE9_REQUIRED_FILES: tuple[str, ...] = tuple(
    relative_path
    for dataset in BASELINE9_DATASETS
    for relative_path in dataset.required_files
)

if len(BASELINE9_REQUIRED_FILES) != 14 or len(set(BASELINE9_REQUIRED_FILES)) != 14:
    raise RuntimeError("The baseline9 data manifest must contain 14 unique files.")


@dataclass(frozen=True)
class FileStatus:
    relative_path: str
    path: Path
    ready: bool
    detail: str


def resolve_data_dir(cli_value: str | None = None) -> Path:
    """Resolve the data root without depending on the current directory."""

    raw_path = cli_value or os.environ.get("DRPT_DATA_DIR") or str(SCRIPT_DIR)
    return Path(raw_path).expanduser().resolve()


def inspect_jsonl(data_dir: Path, relative_path: str) -> FileStatus:
    """Perform a cheap preflight check without scanning a potentially huge file."""

    path = data_dir / relative_path
    if not path.exists():
        return FileStatus(relative_path, path, False, "missing")
    if not path.is_file():
        return FileStatus(relative_path, path, False, "not a regular file")

    try:
        size = path.stat().st_size
    except OSError as error:
        return FileStatus(relative_path, path, False, f"cannot stat: {error}")
    if size == 0:
        return FileStatus(relative_path, path, False, "empty")

    try:
        with path.open("r", encoding="utf-8") as handle:
            first_record = None
            for line in handle:
                if line.strip():
                    first_record = json.loads(line)
                    break
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return FileStatus(relative_path, path, False, f"invalid JSONL: {error}")

    if not isinstance(first_record, dict):
        return FileStatus(relative_path, path, False, "first record is not an object")
    messages = first_record.get("messages")
    if not isinstance(messages, list) or not messages:
        return FileStatus(relative_path, path, False, "first record has no messages list")

    return FileStatus(relative_path, path, True, f"ready ({size:,} bytes)")


def inspect_manifest(data_dir: Path) -> dict[str, FileStatus]:
    return {
        relative_path: inspect_jsonl(data_dir, relative_path)
        for relative_path in BASELINE9_REQUIRED_FILES
    }


def missing_requirements(
    statuses: dict[str, FileStatus],
) -> list[DatasetRequirement]:
    return [
        dataset
        for dataset in BASELINE9_DATASETS
        if any(not statuses[path].ready for path in dataset.required_files)
    ]


def print_manifest_report(data_dir: Path, statuses: dict[str, FileStatus]) -> None:
    ready_count = sum(status.ready for status in statuses.values())
    print(f"Data root: {data_dir}")
    print(f"Baseline9 files: {ready_count}/{len(BASELINE9_REQUIRED_FILES)} ready")
    for dataset in BASELINE9_DATASETS:
        print(f"\n{dataset.description} [{dataset.key}]")
        for relative_path in dataset.required_files:
            status = statuses[relative_path]
            marker = "OK" if status.ready else "MISSING"
            print(f"  [{marker:7}] {relative_path} -- {status.detail}")


def prepare_command(dataset: DatasetRequirement, data_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(PREPARE_SCRIPT),
        "--datasets",
        dataset.key,
        "--output_dir",
        str(data_dir),
    ]


def prepare_missing_datasets(
    data_dir: Path,
    requirements: Sequence[DatasetRequirement],
    *,
    dry_run: bool = False,
) -> list[str]:
    """Prepare only incomplete source groups and return failed group keys."""

    failures: list[str] = []
    environment = os.environ.copy()
    environment["DRPT_DATA_DIR"] = str(data_dir)

    for index, dataset in enumerate(requirements, start=1):
        command = prepare_command(dataset, data_dir)
        print(
            f"\n[{index}/{len(requirements)}] Preparing {dataset.description} "
            f"({dataset.key})"
        )
        print("Command:", " ".join(command))
        if dry_run:
            continue

        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            check=False,
        )
        statuses = [inspect_jsonl(data_dir, path) for path in dataset.required_files]
        if completed.returncode != 0 or not all(status.ready for status in statuses):
            failures.append(dataset.key)
            print(f"ERROR: preparation did not complete for {dataset.key}.", file=sys.stderr)
            for status in statuses:
                if not status.ready:
                    print(f"  {status.path}: {status.detail}", file=sys.stderr)
        else:
            print(f"Prepared {dataset.key}; all required outputs are ready.")

    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Idempotently prepare or preflight the 14 JSONL files required by "
            "the four-setting baseline9 campaign."
        )
    )
    parser.add_argument(
        "--data-dir",
        "--output-dir",
        dest="data_dir",
        help=(
            "Prepared-data root. Defaults to DRPT_DATA_DIR, then the repository's "
            "SFT/data directory."
        ),
    )
    parser.add_argument(
        "--check-only",
        "--preflight",
        action="store_true",
        help="Report readiness and exit without downloading or writing data.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the preparation commands for missing groups without running them.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)
    statuses = inspect_manifest(data_dir)
    print_manifest_report(data_dir, statuses)
    missing = missing_requirements(statuses)

    if not missing:
        print("\nBaseline9 data preflight passed: all 14 required files are ready.")
        return 0

    print(
        f"\n{len(missing)} source group(s) need preparation: "
        + ", ".join(dataset.key for dataset in missing)
    )
    if args.check_only:
        print("Run this command without --check-only to prepare only those groups.")
        return 1

    data_dir.mkdir(parents=True, exist_ok=True)
    failures = prepare_missing_datasets(data_dir, missing, dry_run=args.dry_run)
    if args.dry_run:
        print("\nDry run complete; no data was written.")
        return 0

    final_statuses = inspect_manifest(data_dir)
    print("\nFinal preflight")
    print_manifest_report(data_dir, final_statuses)
    remaining = missing_requirements(final_statuses)
    if failures or remaining:
        failed_keys = sorted(set(failures) | {dataset.key for dataset in remaining})
        print(
            "\nBaseline9 data preparation is incomplete. Failed/missing groups: "
            + ", ".join(failed_keys),
            file=sys.stderr,
        )
        print("Rerun the same command after resolving the reported download error.")
        return 1

    print("\nBaseline9 data preparation complete: all 14 required files are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
