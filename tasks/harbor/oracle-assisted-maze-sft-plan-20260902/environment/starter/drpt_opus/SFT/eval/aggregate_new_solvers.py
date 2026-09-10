#!/usr/bin/env python3
"""Aggregate baseline and new-solver SFT runs into CSV and Markdown.

The report keeps each task's native downstream metric separate. It never takes
an average across SamSUM, TyDiQA, NQ-open, and SQuAD; the only aggregate is a
count of setting-level wins against the scope-matched OptA baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


METHOD_LABELS = (
    "FullTraining",
    "GlobalRaw",
    "LayerwiseRaw",
    "OptGroupRaw",
    "GlobalOptA",
    "LayerwiseOptA",
    "OptGroupOptA",
    "GlobalOptB",
    "LayerwiseOptB",
    "OptGroupOptB",
    "GlobalRandom",
    "LayerwiseRandom",
    "GlobalSoft",
    "LayerwiseSoft",
    "LayerwiseSoftP",
    "GlobalHybridMuonSur",
    "LayerwiseHybridMuonSur",
    "GlobalHybridMuonMatrixSur",
    "LayerwiseHybridMuonMatrixSur",
    "GlobalMuonSur",
    "LayerwiseMuonSur",
    "LayerwiseMuonPSur",
    "LayerwiseMuonSatSur",
    "LayerwiseMuonSatPSur",
)

LEGACY_METHOD_LABELS = (
    "GlobalMuonMatrixSur",
    "LayerwiseMuonMatrixSur",
    "LayerwiseMuonOnlyPSur",
    "LayerwiseMuonOnlySatSur",
    "LayerwiseMuonOnlySatPSur",
)

RUN_RE = re.compile(
    r"^(?P<setting>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)"
    rf"-(?P<method>{'|'.join(METHOD_LABELS + LEGACY_METHOD_LABELS)})"
    r"-(?P<optimizer>adamw|muon|hybrid)"
    r"-(?:p(?P<percentage>[\d.]+)|ms(?P<max_steps>\d+))"
    r"-lr(?P<learning_rate>[\d.]+e-?\d+)"
    r"-b(?P<batch_size>\d+)-v(?P<n_val>\d+)-s(?P<seed>\d+)-(?P<model>.+)$"
)

PRIMARY_METRIC = {
    "samsum": "rougeL",
    "tydiqa": "f1_score",
    "nq_open": "f1_score",
    "squad": "f1_score",
}

NEW_METHODS = {
    "GlobalRandom",
    "LayerwiseRandom",
    "GlobalSoft",
    "LayerwiseSoft",
    "LayerwiseSoftP",
    "GlobalHybridMuonSur",
    "LayerwiseHybridMuonSur",
    "GlobalHybridMuonMatrixSur",
    "LayerwiseHybridMuonMatrixSur",
    "GlobalMuonSur",
    "LayerwiseMuonSur",
    "LayerwiseMuonPSur",
    "LayerwiseMuonSatSur",
    "LayerwiseMuonSatPSur",
}

SOFT_METHODS = {"GlobalSoft", "LayerwiseSoft", "LayerwiseSoftP"}

SOFT_DIAGNOSTIC_FIELDS = (
    "soft_entropy",
    "soft_ess",
    "soft_converged",
    "soft_iterations",
    "soft_objective_improvement",
)

# The run directory records the target task's public name, while the config
# directory keeps the historical shorthand.  A supplied manifest takes
# precedence over this fallback mapping.
SETTING_ALIASES = {
    "triviaqa_nq_open": "triviaqa_nq",
}

REPORT_FIELDS = (
    "setting",
    "task",
    "method",
    "optimizer",
    "seed",
    "primary_metric_name",
    "primary_metric",
    "downstream_metrics",
    "final_eval_perplexity",
    "min_eval_perplexity",
    "final_val_perplexity",
    "min_val_perplexity",
    "wall_time",
    "train_wall_time",
    "soft_entropy",
    "soft_ess",
    "soft_converged",
    "soft_iterations",
    "soft_objective_improvement",
    "run_dir",
)


class AggregationValidationError(RuntimeError):
    """Raised when a report would contain missing, ambiguous, or partial runs."""


def canonical_method_label(method: str, optimizer: str) -> str:
    """Map historical labels to the baseline names used by current reports."""
    legacy_aliases = {
        "GlobalMuonMatrixSur": "GlobalMuonSur",
        "LayerwiseMuonMatrixSur": "LayerwiseMuonSur",
        "LayerwiseMuonOnlyPSur": "LayerwiseMuonPSur",
        "LayerwiseMuonOnlySatSur": "LayerwiseMuonSatSur",
        "LayerwiseMuonOnlySatPSur": "LayerwiseMuonSatPSur",
    }
    if optimizer == "hybrid":
        if method == "GlobalMuonSur":
            return "GlobalHybridMuonSur"
        if method == "LayerwiseMuonSur":
            return "LayerwiseHybridMuonSur"
        if method == "GlobalMuonMatrixSur":
            return "GlobalHybridMuonMatrixSur"
        if method == "LayerwiseMuonMatrixSur":
            return "LayerwiseHybridMuonMatrixSur"
    return legacy_aliases.get(method, method)


def _manifest_description(entry: Dict[str, str]) -> str:
    return ", ".join(
        f"{name}={entry[name]}"
        for name in ("setting", "train", "task", "method", "optimizer")
    )


def _manifest_run_key(entry: Dict[str, str]) -> tuple[str, str, str]:
    return (
        f"{entry['train']}_{entry['task']}",
        entry["method"],
        entry["optimizer"],
    )


def _validate_manifest_entries(
    entries: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    if not entries:
        raise AggregationValidationError("Manifest contains no experiment rows.")

    exact_seen: Dict[tuple[str, ...], int] = {}
    run_key_seen: Dict[tuple[str, str, str], Dict[str, str]] = {}
    duplicate_rows = []
    ambiguous_keys = []
    normalized = []
    required = ("setting", "train", "task", "method", "optimizer")

    for index, raw_entry in enumerate(entries, start=2):
        entry = {name: str(raw_entry.get(name, "")).strip() for name in required}
        missing = [name for name in required if not entry[name]]
        if missing:
            raise AggregationValidationError(
                f"Manifest row {index} is missing required fields: {', '.join(missing)}"
            )
        entry["method"] = canonical_method_label(
            entry["method"], entry["optimizer"]
        )
        if entry["method"] not in METHOD_LABELS:
            raise AggregationValidationError(
                f"Manifest row {index} has unsupported method={entry['method']!r}."
            )
        if entry["optimizer"] not in {"adamw", "muon", "hybrid"}:
            raise AggregationValidationError(
                f"Manifest row {index} has unsupported optimizer={entry['optimizer']!r}."
            )

        exact_key = tuple(entry[name] for name in required)
        if exact_key in exact_seen:
            duplicate_rows.append(
                f"rows {exact_seen[exact_key]} and {index}: {_manifest_description(entry)}"
            )
        else:
            exact_seen[exact_key] = index

        run_key = _manifest_run_key(entry)
        previous = run_key_seen.get(run_key)
        if previous is not None and previous != entry:
            ambiguous_keys.append(
                f"{run_key}: {_manifest_description(previous)}; {_manifest_description(entry)}"
            )
        else:
            run_key_seen[run_key] = entry
        normalized.append(entry)

    problems = []
    if duplicate_rows:
        problems.append("Duplicate manifest rows:\n  - " + "\n  - ".join(duplicate_rows))
    if ambiguous_keys:
        problems.append(
            "Manifest rows map to the same run-name key:\n  - "
            + "\n  - ".join(ambiguous_keys)
        )
    if problems:
        raise AggregationValidationError(
            "Manifest validation failed:\n" + "\n".join(problems)
        )
    return normalized


def load_manifest(path: Path) -> List[Dict[str, str]]:
    """Load and validate the experiment TSV used to gate new solver runs."""
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                raise AggregationValidationError(f"Manifest has no header: {path}")
            required = {"setting", "train", "task", "method", "optimizer"}
            missing = sorted(required.difference(reader.fieldnames))
            if missing:
                raise AggregationValidationError(
                    f"Manifest {path} is missing columns: {', '.join(missing)}"
                )
            entries = list(reader)
    except OSError as exc:
        raise AggregationValidationError(f"Could not read manifest {path}: {exc}") from exc
    return _validate_manifest_entries(entries)


def _read_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _finite(values: Iterable[Any]) -> List[float]:
    result = []
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            result.append(float(value))
    return result


def _last_finite(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = _finite(record.get(key) for record in records)
    return values[-1] if values else None


def _min_finite(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = _finite(record.get(key) for record in records)
    return min(values) if values else None


def _diagnostic_summary(run_dir: Path) -> Dict[str, Optional[float]]:
    """Read optional local diagnostics emitted by the solver implementation."""
    payload = None
    for name in ("solver_diagnostics.json", "selection_diagnostics.json"):
        payload = _read_json(run_dir / name)
        if payload is not None:
            break
    if payload is None:
        return {}
    records = payload if isinstance(payload, list) else payload.get("steps", [payload])
    if not isinstance(records, list):
        return {}

    aliases = {
        "soft_entropy": ("soft_entropy", "soft/entropy", "selection/soft_entropy"),
        "soft_ess": ("soft_ess", "soft/ess", "selection/soft_ess"),
        "soft_converged": ("soft_converged", "soft/converged", "selection/soft_converged"),
        "soft_iterations": ("soft_iterations", "soft/iterations", "selection/soft_iterations"),
        "soft_objective_improvement": (
            "soft_objective_improvement",
            "soft/objective_improvement",
            "selection/soft_objective_improvement",
        ),
    }
    summary: Dict[str, Optional[float]] = {}
    for output_key, keys in aliases.items():
        values = []
        for record in records:
            if not isinstance(record, dict):
                continue
            for record_key, value in record.items():
                if record_key in keys or any(record_key.endswith(f"/{key.rsplit('/', 1)[-1]}") for key in keys):
                    values.extend(_finite([value]))
        summary[output_key] = sum(values) / len(values) if values else None
    return summary


def _downstream_result(
    run_dir: Path,
) -> tuple[str, Optional[str], Optional[float], Dict[str, float]]:
    candidates = sorted(run_dir.glob("*_results.json"))
    for path in candidates:
        if path.name == "evaluation_results.json":
            continue
        payload = _read_json(path)
        if not isinstance(payload, dict) or "task" not in payload:
            continue
        task = str(payload["task"])
        metric_name = PRIMARY_METRIC.get(task)
        metric = payload.get(metric_name) if metric_name else None
        metric_value = _finite([metric])
        scalar_metrics = {
            key: float(value)
            for key, value in payload.items()
            if key != "task"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        }
        return task, metric_name, metric_value[0] if metric_value else None, scalar_metrics
    return "", None, None, {}


def _report_setting(
    run_setting: str,
    manifest_setting_by_run_prefix: Dict[str, str],
) -> str:
    return manifest_setting_by_run_prefix.get(
        run_setting, SETTING_ALIASES.get(run_setting, run_setting)
    )


def _format_validation_sections(sections: Dict[str, List[str]]) -> str:
    lines = ["Aggregation input validation failed:"]
    for heading, items in sections.items():
        if not items:
            continue
        lines.append(f"{heading}:")
        lines.extend(f"  - {item}" for item in items)
    return "\n".join(lines)


def collect_runs(
    runs_dir: Path,
    seed: int,
    manifest_entries: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """Collect only complete, unambiguous report rows.

    When ``manifest_entries`` is supplied, baseline methods remain included,
    while new-solver methods are gated to the exact manifest experiment keys.
    The manifest's config setting name is used in the report, so
    ``triviaqa_nq_open`` run prefixes are reported as ``triviaqa_nq``.
    """
    manifest_provided = manifest_entries is not None
    if manifest_provided:
        manifest_entries = _validate_manifest_entries(manifest_entries or [])
    else:
        manifest_entries = []

    expected_by_run_key = {
        _manifest_run_key(entry): entry for entry in manifest_entries
    }
    manifest_setting_by_run_prefix: Dict[str, str] = {}
    for entry in manifest_entries:
        run_prefix, _, _ = _manifest_run_key(entry)
        previous = manifest_setting_by_run_prefix.get(run_prefix)
        if previous is not None and previous != entry["setting"]:
            raise AggregationValidationError(
                "Manifest run prefix maps to multiple settings: "
                f"{run_prefix} -> {previous}, {entry['setting']}"
            )
        manifest_setting_by_run_prefix[run_prefix] = entry["setting"]

    candidates_by_run_key: Dict[
        tuple[str, str, str], List[tuple[Path, Dict[str, str]]]
    ] = defaultdict(list)
    for run_dir in sorted(runs_dir.iterdir() if runs_dir.is_dir() else []):
        if not run_dir.is_dir():
            continue
        match = RUN_RE.match(run_dir.name)
        if match is None or int(match.group("seed")) != seed:
            continue
        groups = match.groupdict()
        groups["method"] = canonical_method_label(
            groups["method"], groups["optimizer"]
        )
        run_key = (groups["setting"], groups["method"], groups["optimizer"])
        candidates_by_run_key[run_key].append((run_dir, groups))

    sections: Dict[str, List[str]] = {
        "Missing expected runs": [],
        "Duplicate run candidates": [],
        "Incomplete runs": [],
    }
    if manifest_provided:
        for run_key, entry in expected_by_run_key.items():
            candidates = candidates_by_run_key.get(run_key, [])
            if not candidates:
                sections["Missing expected runs"].append(
                    _manifest_description(entry)
                )
            elif len(candidates) > 1:
                sections["Duplicate run candidates"].append(
                    f"{_manifest_description(entry)} -> "
                    + "; ".join(str(path) for path, _ in candidates)
                )

    included_candidates: List[
        tuple[Path, Dict[str, str], Optional[Dict[str, str]]]
    ] = []
    for run_key, candidates in sorted(candidates_by_run_key.items()):
        method = run_key[1]
        expected_entry = expected_by_run_key.get(run_key)
        if manifest_provided and method in NEW_METHODS and expected_entry is None:
            # The manifest is the authority for this sweep.  Unrelated
            # new-solver runs must not change the report row count.
            continue
        if len(candidates) > 1 and expected_entry is None:
            sections["Duplicate run candidates"].append(
                f"setting={run_key[0]}, method={run_key[1]}, "
                f"optimizer={run_key[2]} -> "
                + "; ".join(str(path) for path, _ in candidates)
            )
        included_candidates.extend(
            (run_dir, groups, expected_entry) for run_dir, groups in candidates
        )

    rows: List[Dict[str, Any]] = []
    for run_dir, groups, expected_entry in included_candidates:
        evaluation = _read_json(run_dir / "evaluation_results.json")
        evaluation_history_present = isinstance(evaluation, list) and bool(evaluation)
        evaluation = evaluation if isinstance(evaluation, list) else []
        task, metric_name, metric_value, downstream_metrics = _downstream_result(run_dir)
        row: Dict[str, Any] = {
            "setting": _report_setting(
                groups["setting"], manifest_setting_by_run_prefix
            ),
            "task": task,
            "method": groups["method"],
            "optimizer": groups["optimizer"],
            "seed": int(groups["seed"]),
            "primary_metric_name": metric_name,
            "primary_metric": metric_value,
            "downstream_metrics": json.dumps(
                downstream_metrics, sort_keys=True, separators=(",", ":")
            ),
            "final_eval_perplexity": _last_finite(evaluation, "eval_perplexity"),
            "min_eval_perplexity": _min_finite(evaluation, "eval_perplexity"),
            "final_val_perplexity": _last_finite(evaluation, "val_perplexity"),
            "min_val_perplexity": _min_finite(evaluation, "val_perplexity"),
            "wall_time": _last_finite(evaluation, "wall_time"),
            "train_wall_time": _last_finite(evaluation, "train_wall_time"),
            "run_dir": str(run_dir),
            "_run_setting": groups["setting"],
            "_expected_task": expected_entry["task"] if expected_entry else None,
            "_evaluation_history_present": evaluation_history_present,
        }
        row.update(_diagnostic_summary(run_dir))
        rows.append(row)

    report_groups: Dict[tuple[str, str, str], List[str]] = defaultdict(list)
    for row in rows:
        report_groups[(row["setting"], row["method"], row["optimizer"])].append(
            row["run_dir"]
        )
    for key, paths in sorted(report_groups.items()):
        if len(paths) > 1:
            detail = f"setting={key[0]}, method={key[1]}, optimizer={key[2]}"
            candidate_message = f"{detail} -> " + "; ".join(paths)
            if candidate_message not in sections["Duplicate run candidates"]:
                sections["Duplicate run candidates"].append(candidate_message)

    for row in rows:
        missing = []
        if not row["_evaluation_history_present"]:
            missing.append("nonempty evaluation_results.json")
        if row["primary_metric_name"] is None or row["primary_metric"] is None:
            missing.append("finite downstream primary metric")
        expected_task = row["_expected_task"]
        if expected_task is not None and row["task"] != expected_task:
            missing.append(
                f"downstream task {expected_task!r} (found {row['task']!r})"
            )
        if row["method"] in SOFT_METHODS:
            missing_soft = [
                field for field in SOFT_DIAGNOSTIC_FIELDS
                if row.get(field) is None
            ]
            if missing_soft:
                missing.append("soft diagnostics " + ", ".join(missing_soft))
        if missing:
            sections["Incomplete runs"].append(
                f"{row['run_dir']}: missing " + "; ".join(missing)
            )

    if any(sections.values()):
        raise AggregationValidationError(_format_validation_sections(sections))
    return rows


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _scope_reference(method: str) -> Optional[str]:
    if method.startswith("Global"):
        return "GlobalOptA"
    if method.startswith("Layerwise"):
        return "LayerwiseOptA"
    return None


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, Any]], output_path: Path) -> None:
    by_setting: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_setting[(row["setting"], row["optimizer"])].append(row)

    win_counts: Counter[str] = Counter()
    comparison_counts: Counter[str] = Counter()
    lines = [
        "# Soft Weighting and Muon Surrogate Results",
        "",
        "Task metrics remain in their native units; no cross-task mean is reported.",
        "",
    ]
    for (setting, optimizer), group in sorted(by_setting.items()):
        metric_name = next((row["primary_metric_name"] for row in group if row["primary_metric_name"]), "metric")
        lookup = {row["method"]: row for row in group}
        lines.extend([
            f"## {setting} · {optimizer}",
            "",
            f"Downstream metric: `{metric_name}` (higher is better).",
            "",
            "| Method | Downstream | Δ vs scope OptA | Final eval PPL | Min eval PPL | Wall time | Soft entropy | Soft ESS | Iter. | Conv. | Obj. Δ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in sorted(group, key=lambda item: item["method"]):
            reference_name = _scope_reference(row["method"])
            reference = lookup.get(reference_name) if reference_name else None
            delta = None
            if (
                reference is not None
                and row["primary_metric"] is not None
                and reference["primary_metric"] is not None
                and row["method"] != reference_name
            ):
                delta = row["primary_metric"] - reference["primary_metric"]
                if row["method"] in NEW_METHODS:
                    comparison_counts[row["method"]] += 1
                    if delta > 0:
                        win_counts[row["method"]] += 1
            lines.append(
                "| {method} | {metric} | {delta} | {final_ppl} | {min_ppl} | {wall} | {entropy} | {ess} | {iterations} | {converged} | {objective_improvement} |".format(
                    method=row["method"],
                    metric=_fmt(row["primary_metric"]),
                    delta=_fmt(delta),
                    final_ppl=_fmt(row["final_eval_perplexity"]),
                    min_ppl=_fmt(row["min_eval_perplexity"]),
                    wall=_fmt(row["wall_time"], 1),
                    entropy=_fmt(row.get("soft_entropy")),
                    ess=_fmt(row.get("soft_ess")),
                    iterations=_fmt(row.get("soft_iterations"), 2),
                    converged=_fmt(row.get("soft_converged"), 3),
                    objective_improvement=_fmt(row.get("soft_objective_improvement")),
                )
            )
        lines.append("")

    lines.extend(["## Setting-level win counts", ""])
    if comparison_counts:
        lines.extend(["| Method | Wins vs scope OptA | Compared settings |", "|---|---:|---:|"])
        for method in sorted(comparison_counts):
            lines.append(f"| {method} | {win_counts[method]} | {comparison_counts[method]} |")
    else:
        lines.append("No completed scope-matched comparisons were found.")
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    repo_root = Path(
        os.environ.get("DRPT_REPO_ROOT", Path(__file__).resolve().parents[2])
    )
    default_runs = Path(
        os.environ.get("DRPT_RUNS_DIR", repo_root / "SFT" / "runs")
    )
    default_report_dir = Path(
        os.environ.get(
            "DRPT_REPORTS_DIR", repo_root / "SFT" / "eval" / "reports"
        )
    ) / "new_solvers"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=default_runs)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional experiment TSV. Baselines are retained, while new solver "
            "rows are gated to its exact setting/train/task/method/optimizer keys."
        ),
    )
    parser.add_argument(
        "--csv", type=Path, default=default_report_dir / "new_solver_results.csv"
    )
    parser.add_argument(
        "--markdown", type=Path, default=default_report_dir / "new_solver_report.md"
    )
    args = parser.parse_args()

    try:
        manifest_entries = load_manifest(args.manifest) if args.manifest else None
        rows = collect_runs(args.runs_dir, args.seed, manifest_entries)
    except AggregationValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    write_csv(rows, args.csv)
    write_markdown(rows, args.markdown)
    print(f"Collected {len(rows)} runs into {args.csv} and {args.markdown}")


if __name__ == "__main__":
    main()
