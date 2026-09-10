#!/usr/bin/env python3
"""Validate and collect optimizer-scoped downstream campaign evaluations.

This helper deliberately uses only the Python standard library.  Array jobs use
``check-run`` before loading a model, so an ``afterany`` dependency can safely
skip failed or incomplete training tasks.  A final CPU job uses ``collect`` to
merge the per-array status files without averaging metrics across tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Array jobs invoke this file by path. Ensure the repository package root is
# importable even when PYTHONPATH is not preconfigured by the login shell.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from SFT.eval.task_registry import (
    IMMUTABLE_32K_PROFILES,
    TASK_SPECS,
    profile32k_downstream_cells,
    profile32k_evaluator_pins,
    profile32k_model_metadata,
)


PUBLIC_OPTIMIZERS = ("adamw", "muon", "hybrid")

RESULT_SPECS: Mapping[str, Tuple[str, str, Optional[str]]] = {
    "samsum": ("samsum_results.json", "rougeL", None),
    "tydiqa": ("tydiqa_results.json", "f1_score", None),
    # The current QA evaluators expose f1_score on a 0--100 scale and retain
    # legacy f1 on a 0--1 scale.  Reports use f1_score consistently; old result
    # files that only contain f1 are converted to the same 0--100 scale.
    "nq_open": ("nq_open_results.json", "f1_score", "f1"),
    "squad": ("squad_results.json", "f1_score", "f1"),
    # Dolci32k final benchmarks. Primary metrics are already on a 0--100 scale.
    "ifeval": ("ifeval_results.json", "prompt_level_strict_acc", None),
    "ifbench": ("ifbench_results.json", "prompt_level_loose_acc", None),
    "math500": ("math500_results.json", "accuracy", None),
    "mbpp_plus": (
        "mbpp_plus_results.json",
        "base_plus_extra_pass_at_1",
        "plus_pass_at_1",
    ),
}

LEGACY_PRIMARY_SCALES: Mapping[str, float] = {
    "nq_open": 100.0,
    "squad": 100.0,
    "mbpp_plus": 1.0,
}

# Tasks whose primary metric may be legitimately absent. Every Dolci32k
# benchmark is scored offline, so this is currently empty.
OPTIONAL_METRIC_TASKS: frozenset[str] = frozenset()
UNSCORED_STATUS = "generated_unscored"

DIRECT_MODEL_ARTIFACTS = (
    "model.safetensors",
    "pytorch_model.bin",
    "adapter_model.safetensors",
    "adapter_model.bin",
)
MODEL_INDEX_ARTIFACTS = ("model.safetensors.index.json", "pytorch_model.bin.index.json")

STATUS_FIELDS = (
    "array_task",
    "campaign_id",
    "optimizer",
    "setting",
    "train",
    "task",
    "method",
    "seed",
    "status",
    "run_dir",
    "result_file",
    "detail",
)

RESULT_FIELDS = (
    "campaign_id",
    "optimizer",
    "setting",
    "train",
    "task",
    "method",
    "seed",
    "primary_metric_name",
    "primary_metric",
    "metrics",
    "status",
    "run_dir",
    "result_file",
    "result_status",
)


class CampaignEvaluationError(RuntimeError):
    """Raised when a campaign artifact is malformed or internally inconsistent."""


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        raise CampaignEvaluationError(f"could not read valid JSON from {path}: {exc}") from exc


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _validate_immutable_result_provenance(
    result: Mapping[str, Any], task: str, result_path: Path, profile: str
) -> None:
    """Reject official immutable-profile results with drifted policy."""
    provenance = result.get("provenance")
    if not isinstance(provenance, Mapping):
        raise CampaignEvaluationError(
            f"{profile} result has no provenance object: {result_path}"
        )
    if provenance.get("schema_version") != "drpt.eval.result.v1":
        raise CampaignEvaluationError(
            f"{profile} result has unsupported provenance schema: {result_path}"
        )
    generation = provenance.get("generation")
    if not isinstance(generation, Mapping):
        raise CampaignEvaluationError(
            f"{profile} result has no generation provenance: {result_path}"
        )
    spec = TASK_SPECS[task]
    generation_checks = {
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": spec.default_max_new_tokens,
        "thinking": spec.thinking,
        "thinking_output_stripped": True,
    }
    for key, expected in generation_checks.items():
        actual = generation.get(key)
        if isinstance(expected, float):
            actual_number = _finite_number(actual)
            matches = actual_number is not None and actual_number == expected
        elif isinstance(expected, bool):
            matches = actual is expected
        else:
            matches = (
                not isinstance(actual, bool)
                and _finite_number(actual) == float(expected)
            )
        if not matches:
            raise CampaignEvaluationError(
                f"{profile} {task} provenance generation.{key}={actual!r}, "
                f"expected {expected!r}: {result_path}"
            )

    evaluator = provenance.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise CampaignEvaluationError(
            f"{profile} result has no evaluator provenance: {result_path}"
        )
    for key, expected in profile32k_evaluator_pins(profile)[task].items():
        if expected is None or evaluator.get(key) != expected:
            raise CampaignEvaluationError(
                f"{profile} {task} provenance evaluator.{key}="
                f"{evaluator.get(key)!r}, expected {expected!r}: {result_path}"
            )
    if task == "mbpp_plus":
        dataset = provenance.get("dataset")
        expected_count = dataset.get("n_tasks") if isinstance(dataset, Mapping) else None
        actual_count = evaluator.get("dataset_task_count")
        valid_count = (
            isinstance(expected_count, int)
            and not isinstance(expected_count, bool)
            and expected_count > 0
            and isinstance(actual_count, int)
            and not isinstance(actual_count, bool)
            and actual_count == expected_count
        )
        if not valid_count:
            raise CampaignEvaluationError(
                f"{profile} mbpp_plus dynamic task-count provenance mismatch: "
                f"dataset.n_tasks={expected_count!r}, "
                f"evaluator.dataset_task_count={actual_count!r}: {result_path}"
            )



def has_complete_model_weights(run_dir: Path) -> bool:
    has_weights = any((run_dir / name).is_file() for name in DIRECT_MODEL_ARTIFACTS)
    for index_name in MODEL_INDEX_ARTIFACTS:
        index_path = run_dir / index_name
        if not index_path.is_file():
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map", {})
        except (OSError, ValueError, AttributeError):
            continue
        shards = {str(name) for name in weight_map.values()} if weight_map else set()
        if shards and all(
            Path(name).name == name and (run_dir / name).is_file() for name in shards
        ):
            has_weights = True
    return has_weights


def has_completed_model(run_dir: Path) -> bool:
    has_config = (run_dir / "config.json").is_file() or (
        run_dir / "adapter_config.json"
    ).is_file()
    return has_config and has_complete_model_weights(run_dir)


def _validate_immutable_completion_marker(
    run_dir: Path, metadata: Mapping[str, Any], profile: str
) -> None:
    status_path = run_dir / "run_status.json"
    status = _read_json(status_path)
    if not isinstance(status, dict) or status.get("status") != "complete":
        raise CampaignEvaluationError(f"{profile} run_status.json is not complete")
    if not (run_dir / "_SUCCESS").is_file():
        raise CampaignEvaluationError(f"{profile} _SUCCESS marker is missing")
    expected_build_id = os.environ.get("DRPT_ARTIFACT_BUILD_ID")
    if expected_build_id and status.get("artifact_build_id") != expected_build_id:
        raise CampaignEvaluationError(
            f"{profile} run artifact_build_id does not match the downstream campaign pin"
        )
    if status.get("profile") != profile:
        raise CampaignEvaluationError(
            f"dolci32k run_status profile is {status.get('profile')!r}"
        )
    if metadata.get("experiment_profile") != profile:
        raise CampaignEvaluationError(
            "dolci32k run metadata has a mismatched experiment_profile"
        )
    expected_alias = os.environ.get("DRPT_MODEL_PROFILE")
    if not expected_alias:
        raise CampaignEvaluationError(
            "dolci32k downstream validation requires DRPT_MODEL_PROFILE"
        )
    expected = profile32k_model_metadata(profile, expected_alias)
    checks = {"model_profile": expected_alias, **expected}
    for key, value in checks.items():
        if metadata.get(key) != value:
            raise CampaignEvaluationError(
                f"dolci32k run metadata {key}={metadata.get(key)!r}, "
                f"expected {value!r}"
            )


def _has_train_runtime(log_path: Path) -> bool:
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            return any("'train_runtime':" in line for line in handle)
    except OSError:
        return False


def validate_completed_run(
    run_dir: Path, optimizer: str, profile: Optional[str] = None
) -> None:
    """Require artifacts that are written only after a successful training pass."""
    if optimizer not in PUBLIC_OPTIMIZERS:
        raise CampaignEvaluationError(f"unsupported optimizer label: {optimizer}")
    if not run_dir.is_dir():
        raise CampaignEvaluationError(f"run directory does not exist: {run_dir}")

    metadata_path = run_dir / "run_metadata.json"
    metadata = _read_json(metadata_path)
    if not isinstance(metadata, dict):
        raise CampaignEvaluationError(f"run metadata is not an object: {metadata_path}")
    recorded_optimizer = str(metadata.get("optimizer_type", ""))
    if recorded_optimizer != optimizer:
        raise CampaignEvaluationError(
            f"metadata optimizer_type={recorded_optimizer!r}, expected {optimizer!r}"
        )

    if not has_completed_model(run_dir):
        raise CampaignEvaluationError("final model config/weights are missing")
    if profile in IMMUTABLE_32K_PROFILES:
        _validate_immutable_completion_marker(run_dir, metadata, profile)
    if not _has_train_runtime(run_dir / "train.log"):
        raise CampaignEvaluationError("train.log has no completed train_runtime record")

    history_path = run_dir / "evaluation_results.json"
    history = _read_json(history_path)
    if not isinstance(history, list) or not history:
        raise CampaignEvaluationError("general evaluation history is empty")
    usable = []
    for record in history:
        if not isinstance(record, dict):
            continue
        step = _finite_number(record.get("step"))
        eval_loss = _finite_number(
            record.get("general_val_loss", record.get("eval_loss"))
        )
        if step is not None and eval_loss is not None:
            usable.append((step, eval_loss))
    if not usable or max(step for step, _ in usable) <= 0:
        raise CampaignEvaluationError(
            "general evaluation history has no finite post-training eval_loss"
        )


def read_downstream_result(
    run_dir: Path, task: str, profile: Optional[str] = None
) -> Dict[str, Any]:
    """Read one task result and require a finite task-native primary metric."""
    try:
        filename, primary_name, legacy_primary_name = RESULT_SPECS[task]
    except KeyError as exc:
        raise CampaignEvaluationError(f"unsupported downstream task: {task}") from exc
    result_path = run_dir / filename
    result = _read_json(result_path)
    if not isinstance(result, dict):
        raise CampaignEvaluationError(f"result is not an object: {result_path}")
    if result.get("task") != task:
        raise CampaignEvaluationError(
            f"result task={result.get('task')!r}, expected {task!r}: {result_path}"
        )
    if profile in IMMUTABLE_32K_PROFILES:
        if result.get("evaluation_scope") != "full":
            raise CampaignEvaluationError(
                f"{profile} official report rejects non-full result "
                f"(evaluation_scope={result.get('evaluation_scope')!r}): {result_path}"
            )
        _validate_immutable_result_provenance(result, task, result_path, profile)
    primary = _finite_number(result.get(primary_name))
    if primary is None and legacy_primary_name is not None:
        legacy_primary = _finite_number(result.get(legacy_primary_name))
        primary = (
            None
            if legacy_primary is None
            else LEGACY_PRIMARY_SCALES.get(task, 1.0) * legacy_primary
        )
    unscored = False
    if primary is None:
        if task in OPTIONAL_METRIC_TASKS and result.get("status") == UNSCORED_STATUS:
            unscored = True
        else:
            raise CampaignEvaluationError(
                f"result has no finite primary metric {primary_name!r}: {result_path}"
            )
    metrics = {
        str(key): number
        for key, value in result.items()
        if key != "task" and (number := _finite_number(value)) is not None
    }
    return {
        "result_file": str(result_path),
        "primary_metric_name": primary_name,
        "primary_metric": primary,
        "metrics": metrics,
        "result_status": UNSCORED_STATUS if unscored else "scored",
    }


def _read_status_rows(status_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(status_dir.glob("*.tsv")) if status_dir.is_dir() else []:
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                if reader.fieldnames is None or set(STATUS_FIELDS).difference(reader.fieldnames):
                    raise CampaignEvaluationError(f"status file has an invalid header: {path}")
                file_rows = list(reader)
        except OSError as exc:
            raise CampaignEvaluationError(f"could not read status file {path}: {exc}") from exc
        if len(file_rows) != 1:
            raise CampaignEvaluationError(
                f"status file must contain exactly one data row: {path}"
            )
        rows.append({field: str(file_rows[0].get(field, "")) for field in STATUS_FIELDS})
    return rows


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_results(
    status_dir: Path,
    output_dir: Path,
    campaign_id: str,
    optimizer: str,
    profile: Optional[str] = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Collect valid results while retaining skipped/failed task status rows."""
    status_rows = _read_status_rows(status_dir)
    if profile in IMMUTABLE_32K_PROFILES:
        expected_cells = profile32k_downstream_cells(profile, optimizer)
        rows_by_index = {row["array_task"]: row for row in status_rows}
        for index, cell in enumerate(expected_cells):
            key = str(index)
            if key not in rows_by_index:
                status_rows.append(
                    {
                        "array_task": key,
                        "campaign_id": campaign_id,
                        "optimizer": optimizer,
                        "setting": cell.setting,
                        "train": "",
                        "task": cell.task,
                        "method": cell.method,
                        "seed": "",
                        "status": "missing_status",
                        "run_dir": "",
                        "result_file": "",
                        "detail": "array task wrote no status artifact",
                    }
                )
        status_rows.sort(key=lambda row: int(row["array_task"]))
    collected: List[Dict[str, Any]] = []
    successful_statuses = {"evaluated", "already_evaluated"}

    for row in status_rows:
        if row["campaign_id"] != campaign_id or row["optimizer"] != optimizer:
            raise CampaignEvaluationError(
                "status row belongs to another campaign/family: "
                f"{row['campaign_id']}/{row['optimizer']}"
            )
        if row["status"] not in successful_statuses:
            continue
        run_dir = Path(row["run_dir"])
        try:
            result = read_downstream_result(run_dir, row["task"], profile)
        except CampaignEvaluationError as exc:
            row["status"] = "invalid_result"
            row["detail"] = str(exc)
            continue
        collected.append(
            {
                "campaign_id": campaign_id,
                "optimizer": optimizer,
                "setting": row["setting"],
                "train": row["train"],
                "task": row["task"],
                "method": row["method"],
                "seed": row["seed"],
                "primary_metric_name": result["primary_metric_name"],
                "primary_metric": result["primary_metric"],
                "metrics": json.dumps(result["metrics"], sort_keys=True, separators=(",", ":")),
                "status": row["status"],
                "run_dir": str(run_dir),
                "result_file": result["result_file"],
                "result_status": result["result_status"],
            }
        )
        if result["result_status"] == UNSCORED_STATUS:
            row["detail"] = (
                f"{row['detail']}; " if row["detail"] else ""
            ) + "answers generated but not judge-scored"


    output_dir.mkdir(parents=True, exist_ok=True)
    _write_tsv(output_dir / "evaluation_status.tsv", status_rows, STATUS_FIELDS)
    _write_csv(output_dir / "downstream_results.csv", collected)

    counts = Counter(row["status"] for row in status_rows)
    lines = [
        f"# Downstream evaluation · {campaign_id} · {optimizer}",
        "",
        "Task-native metrics are reported separately; no cross-task mean is computed.",
        "",
        "## Status",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    if counts:
        lines.extend(f"| {status} | {count} |" for status, count in sorted(counts.items()))
    else:
        lines.append("| no status files | 0 |")

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Setting | Task | Method | Primary metric | Value |",
            "|---|---|---|---|---:|",
        ]
    )
    for row in sorted(
        collected, key=lambda item: (item["setting"], item["method"], item["task"])
    ):
        # A generated-but-unscored MT-Bench run has no number to print; showing
        # its status keeps it visible instead of silently dropping the row.
        value = (
            f"{float(row['primary_metric']):.6f}"
            if row["primary_metric"] is not None
            else UNSCORED_STATUS
        )
        lines.append(
            f"| {row['setting']} | {row['task']} | {row['method']} | "
            f"{row['primary_metric_name']} | {value} |"
        )
    if not collected:
        lines.append("| — | — | no valid completed evaluations | — | — |")
    lines.append("")
    (output_dir / "downstream_report.md").write_text("\n".join(lines), encoding="utf-8")
    return status_rows, collected


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_run = subparsers.add_parser("check-run", help="validate training completion")
    check_run.add_argument("--run-dir", type=Path, required=True)
    check_run.add_argument("--optimizer", choices=PUBLIC_OPTIMIZERS, required=True)
    check_run.add_argument(
        "--profile", choices=("loss52", "baseline9", *IMMUTABLE_32K_PROFILES)
    )

    check_result = subparsers.add_parser("check-result", help="validate downstream JSON")
    check_result.add_argument("--run-dir", type=Path, required=True)
    check_result.add_argument("--task", choices=tuple(RESULT_SPECS), required=True)
    check_result.add_argument(
        "--profile", choices=("loss52", "baseline9", *IMMUTABLE_32K_PROFILES)
    )

    collect = subparsers.add_parser("collect", help="merge array-task status/results")
    collect.add_argument("--status-dir", type=Path, required=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--campaign-id", required=True)
    collect.add_argument("--optimizer", choices=PUBLIC_OPTIMIZERS, required=True)
    collect.add_argument(
        "--profile", choices=("loss52", "baseline9", *IMMUTABLE_32K_PROFILES)
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "check-run":
            validate_completed_run(args.run_dir, args.optimizer, args.profile)
            print(f"READY\t{args.run_dir}")
        elif args.command == "check-result":
            result = read_downstream_result(args.run_dir, args.task, args.profile)
            print(json.dumps(result, sort_keys=True))
        else:
            statuses, results = collect_results(
                args.status_dir,
                args.output_dir,
                args.campaign_id,
                args.optimizer,
                args.profile,
            )
            print(
                f"Collected {len(results)} valid downstream results from "
                f"{len(statuses)} status rows into {args.output_dir}"
            )
    except CampaignEvaluationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
