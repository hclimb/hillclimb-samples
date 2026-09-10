#!/usr/bin/env python3
"""Collect and plot loss curves for the focused AdamW/Muon comparison.

The script is deliberately local-first: it reconstructs the curves that were
logged to W&B from ``train.log`` and ``evaluation_results.json``.  It therefore
does not require a W&B login or make any remote changes.  W&B run URLs are
included in the generated tables when they can be recovered from ``train.log``.

Both ``muon`` and legacy ``hybrid`` labels use official
``torch.optim.Muon`` for eligible matrices (with the local Muon only as a
compatibility fallback) and auxiliary AdamW for ineligible parameters. The
``muon`` experiment label uses only Muon-managed matrices for its spectral
selection score; ``hybrid`` preserves the earlier mixed Muon+AdamW score.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from SFT.eval.campaign_downstream import (
    has_complete_model_weights,
    has_completed_model,
)


IMMUTABLE_32K_PROFILES = ("dolci32k",)


def _immutable_profile_registry(profile: str):
    if profile not in IMMUTABLE_32K_PROFILES:
        raise ValueError(f"unsupported immutable profile: {profile}")
    module = importlib.import_module(f"SFT.data.{profile}.profile")
    return (
        tuple(module.SETTING_ORDER),
        {
            "adamw": tuple(module.ADAMW_METHODS),
            "muon": tuple(module.MUON_METHODS),
        },
    )


IMMUTABLE_32K_REGISTRIES = {
    profile: _immutable_profile_registry(profile)
    for profile in IMMUTABLE_32K_PROFILES
}
DOLCI32K_SETTING_ORDER, DOLCI32K_METHODS = IMMUTABLE_32K_REGISTRIES["dolci32k"]


SETTINGS = (
    "alpaca_samsum",
    "less_squad",
    "less_tydiqa",
    "triviaqa_nq",
)
ACTIVE_SETTINGS = SETTINGS

SETTING_ALIASES = {"triviaqa_nq_open": "triviaqa_nq"}

ADAMW_METHODS = (
    "FullTraining",
    "GlobalRaw",
    "LayerwiseRaw",
    "GlobalOptA",
    "LayerwiseOptA",
    "GlobalSoft",
    "LayerwiseSoft",
)

# Keep the original report bundle as the default so previously generated
# optim_0718 reports remain reproducible.  The focused ``loss52`` profile is the
# narrower comparison requested for the fresh four-task rerun.
MUON_METHODS = (
    "FullTraining",
    "GlobalRaw",
    "LayerwiseRaw",
    "GlobalOptA",
    "LayerwiseOptA",
    "GlobalHybridMuonSur",
    "LayerwiseHybridMuonSur",
    "GlobalSoft",
    "LayerwiseSoft",
)

LOSS52_HYBRID_METHODS = (
    "FullTraining",
    "GlobalRaw",
    "LayerwiseRaw",
    "LayerwiseHybridMuonSur",
    "GlobalSoft",
    "LayerwiseSoft",
)

LOSS52_MUON_METHODS = (
    "FullTraining",
    "GlobalRaw",
    "LayerwiseRaw",
    "LayerwiseMuonSur",
    "GlobalSoft",
    "LayerwiseSoft",
)

MUON_SCORER_SOURCE_METHODS = (
    "GlobalHybridMuonSur",
    "LayerwiseHybridMuonSur",
    "GlobalHybridMuonMatrixSur",
    "LayerwiseHybridMuonMatrixSur",
)

MUON_SURROGATE_VARIANT_METHODS = (
    "LayerwiseMuonSur",
    "LayerwiseMuonPSur",
    "LayerwiseMuonSatSur",
    "LayerwiseMuonSatPSur",
)

SOFT_VARIANT_METHODS = (
    "LayerwiseSoft",
    "LayerwiseSoftP",
)

# Nine conceptual baselines requested for the focused comparison.  Methods are
# optimizer-scoped: AdamW receives OPUS/OptA, while Muon receives the four
# matrix-only spectral surrogate variants.  FullTraining, LayerwiseRaw, and the
# two soft constraints are shared between optimizers.
BASELINE9_METHODS = {
    "adamw": (
        "FullTraining",
        "LayerwiseRaw",
        "LayerwiseSoft",
        "LayerwiseSoftP",
        "LayerwiseOptA",
    ),
    "muon": (
        "FullTraining",
        "LayerwiseRaw",
        "LayerwiseSoft",
        "LayerwiseSoftP",
        "LayerwiseMuonSur",
        "LayerwiseMuonPSur",
        "LayerwiseMuonSatSur",
        "LayerwiseMuonSatPSur",
    ),
}

ALL_METHODS = tuple(
    dict.fromkeys(
        ADAMW_METHODS
        + MUON_METHODS
        + MUON_SCORER_SOURCE_METHODS
        + MUON_SURROGATE_VARIANT_METHODS
        + SOFT_VARIANT_METHODS
        + (
            "GlobalMuonSur",
            "GlobalMuonMatrixSur",
            "LayerwiseMuonMatrixSur",
            "LayerwiseMuonOnlyPSur",
            "LayerwiseMuonOnlySatSur",
            "LayerwiseMuonOnlySatPSur",
        )
    )
)
OPTIMIZER_METHODS = {"adamw": ADAMW_METHODS, "hybrid": MUON_METHODS}
PROFILE_METHODS = {
    "legacy": OPTIMIZER_METHODS,
    "loss52": {"adamw": ADAMW_METHODS, "muon": LOSS52_MUON_METHODS},
    "loss52-legacy": {"adamw": ADAMW_METHODS, "hybrid": LOSS52_HYBRID_METHODS},
    "loss52-adamw": {"adamw": ADAMW_METHODS},
    "loss52-muon": {"muon": LOSS52_MUON_METHODS},
    "loss52-hybrid": {"hybrid": LOSS52_HYBRID_METHODS},
    "muon-source": {"adamw": (), "hybrid": MUON_SCORER_SOURCE_METHODS},
    "muon-surrogate-variants": {"muon": MUON_SURROGATE_VARIANT_METHODS},
    "soft-variants": {
        "adamw": SOFT_VARIANT_METHODS,
        "muon": SOFT_VARIANT_METHODS,
    },
    "baseline9": BASELINE9_METHODS,
    "baseline9-adamw": {"adamw": BASELINE9_METHODS["adamw"]},
    "baseline9-muon": {"muon": BASELINE9_METHODS["muon"]},
    "dolci32k": DOLCI32K_METHODS,
    "dolci32k-adamw": {"adamw": DOLCI32K_METHODS["adamw"]},
    "dolci32k-muon": {"muon": DOLCI32K_METHODS["muon"]},
}
OPTIMIZER_DISPLAY = {
    "adamw": "AdamW",
    "muon": "Muon",
    "hybrid": "Hybrid scoring (legacy)",
}

RUN_RE = re.compile(
    r"^(?P<setting>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)"
    rf"-(?P<method>{'|'.join(ALL_METHODS)})"
    r"-(?P<optimizer>adamw|muon|hybrid)"
    r"-(?:p(?P<percentage>[\d.]+)|ms(?P<max_steps>\d+))"
    r"-lr(?P<learning_rate>[\d.]+e-?\d+)"
    r"-b(?P<batch_size>\d+)-v(?P<n_val>\d+)-s(?P<seed>\d+)-(?P<model>.+)$"
)

FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TRAIN_LOSS_RE = re.compile(
    rf"\{{'loss':\s*(?P<loss>{FLOAT_PATTERN}).*?"
    rf"'epoch':\s*(?P<epoch>{FLOAT_PATTERN})\}}"
)
WANDB_URL_RE = re.compile(
    r"https://wandb\.ai/[^/\s]+/[^/\s]+/runs/(?P<run_id>[A-Za-z0-9_-]+)"
)
WANDB_GROUP_RE = re.compile(r"wandb_group=(?P<group>[^,\n]+)")

METRIC_LABELS = {
    "target_val_loss": "Target validation loss",
    "general_eval_loss": "General held-out evaluation loss",
    "train_loss": "Logged training loss",
}

COLORS = {
    "FullTraining": "#111111",
    "GlobalRaw": "#4C78A8",
    "LayerwiseRaw": "#4C78A8",
    "GlobalOptA": "#F58518",
    "LayerwiseOptA": "#F58518",
    "GlobalHybridMuonSur": "#54A24B",
    "LayerwiseHybridMuonSur": "#54A24B",
    "GlobalHybridMuonMatrixSur": "#B279A2",
    "LayerwiseHybridMuonMatrixSur": "#B279A2",
    "GlobalMuonSur": "#B279A2",
    "LayerwiseMuonSur": "#B279A2",
    "LayerwiseMuonPSur": "#72B7B2",
    "LayerwiseMuonSatSur": "#FF9DA6",
    "LayerwiseMuonSatPSur": "#9D755D",
    "GlobalSoft": "#E45756",
    "LayerwiseSoft": "#E45756",
    "LayerwiseSoftP": "#59A14F",
}

MARKERS = {
    "FullTraining": "o",
    "GlobalRaw": "s",
    "LayerwiseRaw": "^",
    "GlobalOptA": "s",
    "LayerwiseOptA": "s",
    "GlobalHybridMuonSur": "P",
    "LayerwiseHybridMuonSur": "P",
    "GlobalHybridMuonMatrixSur": "X",
    "LayerwiseHybridMuonMatrixSur": "X",
    "GlobalMuonSur": "X",
    "LayerwiseMuonSur": "X",
    "LayerwiseMuonPSur": "v",
    "LayerwiseMuonSatSur": "D",
    "LayerwiseMuonSatPSur": "*",
    "GlobalSoft": "h",
    "LayerwiseSoft": "d",
    "LayerwiseSoftP": ">",
}

LINESTYLES = {
    "FullTraining": "-",
    "GlobalRaw": "-",
    "LayerwiseRaw": (0, (5, 1)),
    "GlobalOptA": "-",
    "LayerwiseOptA": (0, (5, 2, 1, 2)),
    "GlobalHybridMuonSur": "-.",
    "LayerwiseHybridMuonSur": "--",
    "GlobalHybridMuonMatrixSur": "-.",
    "LayerwiseHybridMuonMatrixSur": (0, (7, 1, 1, 1)),
    "GlobalMuonSur": "-.",
    "LayerwiseMuonSur": (0, (7, 1, 1, 1)),
    "LayerwiseMuonPSur": (0, (3, 1, 1, 1)),
    "LayerwiseMuonSatSur": ":",
    "LayerwiseMuonSatPSur": (0, (6, 1, 1, 1, 1, 1)),
    "GlobalSoft": "-",
    "LayerwiseSoft": (0, (4, 1, 1, 1)),
    "LayerwiseSoftP": (0, (1, 1)),
}

GENERAL_REDUCTION_METRIC = "general_eval_loss_reduction"
GENERAL_REDUCTION_LABEL = "General held-out loss reduction (initial − current)"


class LossReportError(RuntimeError):
    """Raised when requested run records are ambiguous or malformed."""


def optimizer_methods_for_profile(profile: str) -> Mapping[str, Tuple[str, ...]]:
    """Return the immutable method sequences associated with a report profile."""
    try:
        methods = PROFILE_METHODS[profile]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILE_METHODS))
        raise LossReportError(
            f"Unknown loss-report profile {profile!r}; expected one of: {choices}"
        ) from exc
    return methods



def settings_for_profile(profile: str) -> Tuple[str, ...]:
    """Return canonical settings for a report profile."""
    base_profile = profile.partition("-")[0]
    if base_profile in IMMUTABLE_32K_REGISTRIES:
        return tuple(IMMUTABLE_32K_REGISTRIES[base_profile][0])
    return SETTINGS


def _active_settings_are_immutable_32k() -> bool:
    active = tuple(ACTIVE_SETTINGS)
    return any(
        active == tuple(settings)
        for settings, _ in IMMUTABLE_32K_REGISTRIES.values()
    )
def canonical_method_label(method: str, optimizer: str) -> str:
    """Map historical directory labels to the current baseline vocabulary."""
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


def _optimizer_methods_or_default(
    optimizer_methods: Optional[Mapping[str, Sequence[str]]],
) -> Mapping[str, Sequence[str]]:
    """Resolve an optional method map while preserving the legacy API default."""
    return OPTIMIZER_METHODS if optimizer_methods is None else optimizer_methods


def _active_optimizers(
    optimizer_methods: Mapping[str, Sequence[str]],
) -> Tuple[str, ...]:
    """Return optimizer columns that contain at least one requested method."""
    return tuple(
        optimizer
        for optimizer in ("adamw", "muon", "hybrid")
        if optimizer_methods.get(optimizer)
    )


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        raise LossReportError(f"Could not read valid JSON from {path}: {exc}") from exc


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_train_loss(log_path: Path) -> List[Dict[str, float]]:
    """Parse Trainer's step-level loss records without depending on W&B."""
    records: List[Dict[str, float]] = []
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                for match in TRAIN_LOSS_RE.finditer(line):
                    loss = float(match.group("loss"))
                    epoch = float(match.group("epoch"))
                    if math.isfinite(loss) and math.isfinite(epoch):
                        records.append(
                            {"step": float(len(records) + 1), "epoch": epoch, "value": loss}
                        )
    except OSError as exc:
        raise LossReportError(f"Could not read {log_path}: {exc}") from exc
    return records


def parse_wandb_metadata(log_path: Path) -> Dict[str, str]:
    """Recover the online W&B identity printed by the training process."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"wandb_group": "", "wandb_run_id": "", "wandb_url": ""}
    urls = list(WANDB_URL_RE.finditer(text))
    groups = list(WANDB_GROUP_RE.finditer(text))
    if not urls:
        return {
            "wandb_group": groups[-1].group("group").strip() if groups else "",
            "wandb_run_id": "",
            "wandb_url": "",
        }
    last = urls[-1]
    return {
        "wandb_group": groups[-1].group("group").strip() if groups else "",
        "wandb_run_id": last.group("run_id"),
        "wandb_url": last.group(0),
    }


def deduplicate_eval_records(records: Any, source: Path) -> List[Dict[str, float]]:
    """Validate evaluation records and keep the last observation at each step."""
    if not isinstance(records, list):
        raise LossReportError(f"Expected a list in {source}")
    by_step: Dict[int, Dict[str, float]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        step_value = _finite_float(record.get("step"))
        target = _finite_float(
            record.get("target_val_loss", record.get("val_loss"))
        )
        general = _finite_float(
            record.get("general_val_loss", record.get("eval_loss"))
        )
        if step_value is None or target is None or general is None:
            continue
        step = int(step_value)
        by_step[step] = {
            "step": float(step),
            "target_val_loss": target,
            "general_eval_loss": general,
        }
    result = [by_step[step] for step in sorted(by_step)]
    if len(result) < 2 or result[0]["step"] != 0:
        raise LossReportError(
            f"Incomplete evaluation history in {source}: expected step 0 and a later step"
        )
    return result


def normalize_curve(
    points: Sequence[Mapping[str, float]], value_key: str = "value"
) -> Tuple[np.ndarray, np.ndarray]:
    steps = np.asarray([float(point["step"]) for point in points], dtype=float)
    values = np.asarray([float(point[value_key]) for point in points], dtype=float)
    if len(steps) == 0 or not np.isfinite(steps).all() or not np.isfinite(values).all():
        raise LossReportError("A loss curve is empty or contains non-finite values")
    maximum = float(steps.max())
    progress = steps / maximum if maximum > 0 else np.zeros_like(steps)
    return progress, values


def curve_summary(
    points: Sequence[Mapping[str, float]],
    value_key: str = "value",
    use_window: bool = False,
) -> Dict[str, float]:
    """Summarize a curve; reductions are positive when loss improves."""
    progress, values = normalize_curve(points, value_key=value_key)
    if use_window:
        width = max(1, int(math.ceil(len(values) * 0.1)))
        initial = float(values[:width].mean())
        final = float(values[-width:].mean())
    else:
        initial = float(values[0])
        final = float(values[-1])
    minimum = float(values.min())
    if len(values) > 1:
        # NumPy <2 exposes trapz; NumPy >=2 exposes trapezoid and may remove
        # trapz. Resolve lazily so either API works.
        integrate = getattr(np, "trapezoid", None)
        if integrate is None:
            integrate = getattr(np, "trapz")
        auc = float(integrate(values, progress))
    else:
        auc = final
    reduction = initial - final
    relative = 100.0 * reduction / abs(initial) if initial != 0 else float("nan")
    return {
        "initial": initial,
        "final": final,
        "minimum": minimum,
        "reduction": reduction,
        "relative_reduction_pct": relative,
        "normalized_auc": auc,
        "points": float(len(values)),
        "last_step": float(points[-1]["step"]),
    }


def _is_completed_run(
    run_dir: Path,
    eval_records: Sequence[Mapping[str, float]],
    *,
    require_immutable_marker: bool = False,
) -> bool:
    log_path = run_dir / "train.log"
    if not log_path.is_file():
        return False
    if require_immutable_marker and not has_completed_model(run_dir):
        return False
    if not require_immutable_marker and not has_complete_model_weights(run_dir):
        return False
    if require_immutable_marker:
        try:
            status = _read_json(run_dir / "run_status.json")
        except LossReportError:
            return False
        if (
            not isinstance(status, Mapping)
            or status.get("status") != "complete"
            or not (run_dir / "_SUCCESS").is_file()
        ):
            return False
    try:
        tail_marker = "'train_runtime':" in log_path.read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return False
    return bool(tail_marker and eval_records and eval_records[-1]["step"] > 0)


def _discover_run_candidates(
    runs_dir: Path,
    seed: int,
    optimizer_methods: Mapping[str, Sequence[str]],
) -> Tuple[
    Dict[Tuple[str, str, str], List[Path]],
    List[Dict[str, str]],
]:
    """Discover every completed candidate without resolving duplicate keys."""

    candidates: Dict[Tuple[str, str, str], List[Path]] = defaultdict(list)
    incomplete: List[Dict[str, str]] = []
    if not runs_dir.is_dir():
        raise LossReportError(f"Runs directory does not exist: {runs_dir}")
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        match = RUN_RE.match(run_dir.name)
        if not match or int(match.group("seed")) != seed:
            continue
        setting = SETTING_ALIASES.get(match.group("setting"), match.group("setting"))
        optimizer = match.group("optimizer")
        method = canonical_method_label(match.group("method"), optimizer)
        if (
            setting not in ACTIVE_SETTINGS
            or optimizer not in optimizer_methods
            or method not in optimizer_methods[optimizer]
        ):
            continue
        key = (setting, optimizer, method)
        eval_path = run_dir / "evaluation_results.json"
        try:
            eval_records = deduplicate_eval_records(_read_json(eval_path), eval_path)
            complete = _is_completed_run(
                run_dir, eval_records,
                require_immutable_marker=_active_settings_are_immutable_32k(),
            )
        except LossReportError as exc:
            incomplete.append(
                {
                    "setting": setting,
                    "optimizer": optimizer,
                    "method": method,
                    "reason": str(exc),
                    "run_dir": str(run_dir),
                }
            )
            continue
        if not complete:
            incomplete.append(
                {
                    "setting": setting,
                    "optimizer": optimizer,
                    "method": method,
                    "reason": "missing training completion marker or complete model weights",
                    "run_dir": str(run_dir),
                }
            )
            continue
        candidates[key].append(run_dir)
    return candidates, incomplete


def discover_runs(
    runs_dir: Path,
    seed: int,
    optimizer_methods: Optional[Mapping[str, Sequence[str]]] = None,
) -> Tuple[Dict[Tuple[str, str, str], Path], List[Dict[str, str]]]:
    optimizer_methods = _optimizer_methods_or_default(optimizer_methods)
    candidates, incomplete = _discover_run_candidates(
        runs_dir, seed, optimizer_methods
    )
    found: Dict[Tuple[str, str, str], Path] = {}
    for key, matches in candidates.items():
        if len(matches) > 1:
            raise LossReportError(
                f"Multiple completed runs match {key}: "
                + " and ".join(str(path) for path in matches)
            )
        found[key] = matches[0]
    return found, incomplete


def _expected_keys(
    optimizer_methods: Optional[Mapping[str, Sequence[str]]] = None,
) -> Iterable[Tuple[str, str, str]]:
    optimizer_methods = _optimizer_methods_or_default(optimizer_methods)
    for setting in ACTIVE_SETTINGS:
        for optimizer, methods in optimizer_methods.items():
            for method in methods:
                yield setting, optimizer, method


def collect_runs(
    runs_dir: Path,
    seed: int,
    optimizer_methods: Optional[Mapping[str, Sequence[str]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, str]]]:
    optimizer_methods = _optimizer_methods_or_default(optimizer_methods)
    discovered, incomplete = discover_runs(runs_dir, seed, optimizer_methods)
    runs: List[Dict[str, Any]] = []
    points: List[Dict[str, Any]] = []
    missing: List[Dict[str, str]] = []
    incomplete_by_key = {
        (entry["setting"], entry["optimizer"], entry["method"]): entry
        for entry in incomplete
    }

    for setting, optimizer, method in _expected_keys(optimizer_methods):
        key = (setting, optimizer, method)
        run_dir = discovered.get(key)
        if run_dir is None:
            prior = incomplete_by_key.get(key)
            missing.append(
                {
                    "setting": setting,
                    "optimizer": optimizer,
                    "optimizer_display": OPTIMIZER_DISPLAY[optimizer],
                    "method": method,
                    "status": "incomplete" if prior else "missing",
                    "reason": prior["reason"] if prior else "no matching run directory",
                    "run_dir": prior["run_dir"] if prior else "",
                }
            )
            continue

        eval_path = run_dir / "evaluation_results.json"
        eval_records = deduplicate_eval_records(_read_json(eval_path), eval_path)
        train_records = parse_train_loss(run_dir / "train.log")
        if not train_records:
            raise LossReportError(f"No step-level train loss found in {run_dir / 'train.log'}")
        eval_last_step = int(eval_records[-1]["step"])
        if len(train_records) != eval_last_step:
            raise LossReportError(
                f"Train loss count/final step mismatch for {run_dir.name}: "
                f"{len(train_records)} records versus step {eval_last_step}"
            )

        wandb = parse_wandb_metadata(run_dir / "train.log")
        metadata_path = run_dir / "run_metadata.json"
        metadata = _read_json(metadata_path) if metadata_path.is_file() else {}
        row: Dict[str, Any] = {
            "setting": setting,
            "optimizer": optimizer,
            "optimizer_display": OPTIMIZER_DISPLAY[optimizer],
            "method": method,
            "seed": seed,
            "val_strategy": metadata.get("val_strategy", ""),
            "soft_weighting_constraint": metadata.get(
                "soft_weighting_constraint", ""
            ),
            "soft_replay_precision": metadata.get(
                "soft_replay_precision", ""
            ),
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            **wandb,
        }

        target_points = [
            {"step": record["step"], "value": record["target_val_loss"]}
            for record in eval_records
        ]
        general_points = [
            {"step": record["step"], "value": record["general_eval_loss"]}
            for record in eval_records
        ]
        summaries = {
            "target": curve_summary(target_points),
            "general": curve_summary(general_points),
            "train": curve_summary(train_records, use_window=True),
        }
        for prefix, summary in summaries.items():
            for name, value in summary.items():
                row[f"{prefix}_{name}"] = value
        runs.append(row)

        for metric, metric_points in (
            ("target_val_loss", target_points),
            ("general_eval_loss", general_points),
            ("train_loss", train_records),
        ):
            progress, values = normalize_curve(metric_points)
            smooth_values = rolling_mean(values) if metric == "train_loss" else values
            for point, normalized_step, value, smooth in zip(
                metric_points, progress, values, smooth_values
            ):
                points.append(
                    {
                        "setting": setting,
                        "optimizer": optimizer,
                        "method": method,
                        "metric": metric,
                        "step": int(point["step"]),
                        "progress": float(normalized_step),
                        "value": float(value),
                        "smoothed_value": float(smooth),
                        "run_dir": str(run_dir),
                    }
                )
    return runs, points, missing


def collect_completed_general_runs(
    run_roots: Sequence[Path],
    seed: int,
    optimizer_methods: Optional[Mapping[str, Sequence[str]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, str]]]:
    """Collect only completed runs for a general-eval-only partial report.

    This opt-in path deliberately does not require step-level train-loss records:
    completion is still established by the model artifact, Trainer completion
    marker, and a valid general-evaluation history.  Multiple explicit campaign
    roots can be combined.  If more than one completed run matches the same
    setting/optimizer/method key, the key is reported as ambiguous and excluded
    instead of choosing a run silently.
    """

    optimizer_methods = _optimizer_methods_or_default(optimizer_methods)
    if not run_roots:
        raise LossReportError("At least one completed-run root is required")

    candidates: Dict[Tuple[str, str, str], List[Path]] = defaultdict(list)
    incomplete_by_key: Dict[
        Tuple[str, str, str], List[Dict[str, str]]
    ] = defaultdict(list)
    for root in run_roots:
        discovered, incomplete = _discover_run_candidates(
            root, seed, optimizer_methods
        )
        for key, run_dirs in discovered.items():
            for run_dir in run_dirs:
                if run_dir not in candidates[key]:
                    candidates[key].append(run_dir)
        for entry in incomplete:
            key = (entry["setting"], entry["optimizer"], entry["method"])
            incomplete_by_key[key].append(entry)

    runs: List[Dict[str, Any]] = []
    points: List[Dict[str, Any]] = []
    missing: List[Dict[str, str]] = []
    for setting, optimizer, method in _expected_keys(optimizer_methods):
        key = (setting, optimizer, method)
        matches = candidates.get(key, [])
        if len(matches) != 1:
            prior = incomplete_by_key.get(key, [])
            if len(matches) > 1:
                status = "ambiguous"
                reason = "multiple completed runs: " + "; ".join(
                    str(path) for path in matches
                )
                run_dir_text = ";".join(str(path) for path in matches)
            elif prior:
                status = "incomplete"
                reason = "; ".join(
                    sorted({str(entry["reason"]) for entry in prior})
                )
                run_dir_text = ";".join(
                    sorted({str(entry["run_dir"]) for entry in prior})
                )
            else:
                status = "missing"
                reason = "no matching run directory in requested roots"
                run_dir_text = ""
            missing.append(
                {
                    "setting": setting,
                    "optimizer": optimizer,
                    "optimizer_display": OPTIMIZER_DISPLAY[optimizer],
                    "method": method,
                    "status": status,
                    "reason": reason,
                    "run_dir": run_dir_text,
                }
            )
            continue

        run_dir = matches[0]
        try:
            eval_path = run_dir / "evaluation_results.json"
            eval_records = deduplicate_eval_records(_read_json(eval_path), eval_path)
            metadata_path = run_dir / "run_metadata.json"
            metadata = _read_json(metadata_path) if metadata_path.is_file() else {}
        except LossReportError as exc:
            missing.append(
                {
                    "setting": setting,
                    "optimizer": optimizer,
                    "optimizer_display": OPTIMIZER_DISPLAY[optimizer],
                    "method": method,
                    "status": "malformed",
                    "reason": str(exc),
                    "run_dir": str(run_dir),
                }
            )
            continue

        general_points = [
            {"step": record["step"], "value": record["general_eval_loss"]}
            for record in eval_records
        ]
        summary = curve_summary(general_points)
        row: Dict[str, Any] = {
            "setting": setting,
            "optimizer": optimizer,
            "optimizer_display": OPTIMIZER_DISPLAY[optimizer],
            "method": method,
            "seed": seed,
            "status": "completed",
            "val_strategy": metadata.get("val_strategy", ""),
            "soft_weighting_constraint": metadata.get(
                "soft_weighting_constraint", ""
            ),
            "soft_replay_precision": metadata.get(
                "soft_replay_precision", ""
            ),
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            **parse_wandb_metadata(run_dir / "train.log"),
        }
        for name, value in summary.items():
            row[f"general_{name}"] = value
        runs.append(row)

        progress, values = normalize_curve(general_points)
        for point, normalized_step, value in zip(
            general_points, progress, values
        ):
            points.append(
                {
                    "setting": setting,
                    "optimizer": optimizer,
                    "method": method,
                    "metric": "general_eval_loss",
                    "step": int(point["step"]),
                    "progress": float(normalized_step),
                    "value": float(value),
                    "smoothed_value": float(value),
                    "run_dir": str(run_dir),
                }
            )
    return runs, points, missing


def add_general_reduction_points(
    points: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Return raw points plus ``initial_loss - current_loss`` curves.

    The reduction is computed independently for each setting/optimizer/method
    curve.  It starts at zero and is larger when held-out loss has fallen more.
    """

    result = [dict(point) for point in points]
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for point in points:
        if point.get("metric") != "general_eval_loss":
            continue
        key = (
            str(point["setting"]),
            str(point["optimizer"]),
            str(point["method"]),
        )
        grouped[key].append(point)
    for records in grouped.values():
        ordered = sorted(
            records,
            key=lambda item: (float(item["progress"]), int(item["step"])),
        )
        initial = float(ordered[0]["value"])
        for point in ordered:
            reduction = initial - float(point["value"])
            derived = dict(point)
            derived["metric"] = GENERAL_REDUCTION_METRIC
            derived["value"] = reduction
            derived["smoothed_value"] = reduction
            result.append(derived)
    return result


def rolling_mean(values: np.ndarray) -> np.ndarray:
    """Centered rolling mean with a window near 2% of training steps."""
    if len(values) <= 2:
        return values.copy()
    width = max(5, len(values) // 50)
    width = min(width, len(values))
    kernel = np.ones(width, dtype=float) / width
    left = width // 2
    right = width - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    runs: Sequence[Mapping[str, Any]],
    points: Sequence[Mapping[str, Any]],
    missing: Sequence[Mapping[str, str]],
    optimizer_methods: Optional[Mapping[str, Sequence[str]]] = None,
) -> None:
    optimizer_methods = _optimizer_methods_or_default(optimizer_methods)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_points = add_general_reduction_points(points)
    summary_fields = (
        "setting", "optimizer", "optimizer_display", "method", "seed", "val_strategy",
        "soft_weighting_constraint", "soft_replay_precision",
        "target_initial", "target_final", "target_minimum", "target_reduction",
        "target_relative_reduction_pct", "target_normalized_auc", "target_points",
        "general_initial", "general_final", "general_minimum", "general_reduction",
        "general_relative_reduction_pct", "general_normalized_auc", "general_points",
        "train_initial", "train_final", "train_minimum", "train_reduction",
        "train_relative_reduction_pct", "train_normalized_auc", "train_points",
        "wandb_group", "wandb_run_id", "wandb_url", "run_name", "run_dir",
    )
    _write_csv(output_dir / "loss_summary.csv", runs, summary_fields)
    wandb_fields = (
        "setting", "optimizer", "optimizer_display", "method", "wandb_group",
        "wandb_run_id", "wandb_url", "run_name", "run_dir",
    )
    _write_csv(output_dir / "wandb_runs.csv", runs, wandb_fields)
    missing_fields = (
        "setting", "optimizer", "optimizer_display", "method", "status", "reason", "run_dir"
    )
    _write_csv(output_dir / "missing_runs.csv", missing, missing_fields)

    point_fields = (
        "setting", "optimizer", "method", "metric", "step", "progress", "value",
        "smoothed_value", "run_dir",
    )
    with gzip.open(output_dir / "loss_curve_points.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=point_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(plot_points)

    for metric in (*METRIC_LABELS, GENERAL_REDUCTION_METRIC):
        plot_grid(
            output_dir / f"{metric}_grid.png",
            plot_points,
            metric,
            optimizer_methods,
        )
    write_markdown(
        output_dir / "loss_report.md", runs, missing, optimizer_methods
    )


def _line_style(method: str) -> Any:
    return LINESTYLES.get(
        method, "--" if method.startswith("Layerwise") else "-"
    )


def _metric_label(metric: str) -> str:
    if metric == GENERAL_REDUCTION_METRIC:
        return GENERAL_REDUCTION_LABEL
    return METRIC_LABELS[metric]


def _plot_method_curve(
    ax: Any,
    records: Sequence[Mapping[str, Any]],
    method: str,
    metric: str,
) -> None:
    records = sorted(records, key=lambda item: float(item["progress"]))
    y_key = "smoothed_value" if metric == "train_loss" else "value"
    markevery = max(1, len(records) // 8)
    ax.plot(
        [float(item["progress"]) for item in records],
        [float(item[y_key]) for item in records],
        label=method,
        color=COLORS.get(method, "#777777"),
        marker=MARKERS.get(method, "o"),
        markevery=markevery,
        markersize=4.2,
        linestyle=_line_style(method),
        linewidth=1.7 if method != "FullTraining" else 2.2,
        alpha=0.92,
    )


def _finish_axis_legend(ax: Any) -> None:
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7, ncol=2, frameon=False)
    else:
        ax.text(
            0.5,
            0.5,
            "No completed run",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#777777",
        )


def plot_grid(
    path: Path,
    all_points: Sequence[Mapping[str, Any]],
    metric: str,
    optimizer_methods: Optional[Mapping[str, Sequence[str]]] = None,
) -> None:
    optimizer_methods = _optimizer_methods_or_default(optimizer_methods)
    by_key: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for point in all_points:
        if point["metric"] == metric:
            by_key[(str(point["setting"]), str(point["optimizer"]), str(point["method"]))].append(point)

    active_optimizers = _active_optimizers(optimizer_methods)
    if not active_optimizers:
        raise LossReportError("The selected profile does not request any methods")
    fig, axes = plt.subplots(
        len(ACTIVE_SETTINGS),
        len(active_optimizers),
        figsize=(7.5 * len(active_optimizers), 18),
        squeeze=False,
    )
    for row_index, setting in enumerate(ACTIVE_SETTINGS):
        for column_index, optimizer in enumerate(active_optimizers):
            ax = axes[row_index][column_index]
            for method in optimizer_methods[optimizer]:
                records = by_key.get((setting, optimizer, method), [])
                if not records:
                    continue
                _plot_method_curve(ax, records, method, metric)
            ax.set_title(f"{setting} · {OPTIMIZER_DISPLAY[optimizer]}")
            if row_index == len(ACTIVE_SETTINGS) - 1:
                ax.set_xlabel("Normalized training progress")
            ax.set_ylabel(_metric_label(metric))
            ax.grid(alpha=0.22)
            _finish_axis_legend(ax)
    note = "rolling mean (~2% window)" if metric == "train_loss" else "raw evaluation checkpoints"
    fig.suptitle(f"{_metric_label(metric)} ({note})", fontsize=16, y=0.985)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.055, top=0.95, hspace=0.30, wspace=0.18)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_task_figures(
    output_dir: Path,
    all_points: Sequence[Mapping[str, Any]],
    metric: str,
    optimizer_methods: Mapping[str, Sequence[str]],
) -> None:
    """Write one optimizer-separated figure per task for meeting use."""

    by_key: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for point in all_points:
        if point["metric"] == metric:
            by_key[(
                str(point["setting"]),
                str(point["optimizer"]),
                str(point["method"]),
            )].append(point)
    active_optimizers = _active_optimizers(optimizer_methods)
    for setting in ACTIVE_SETTINGS:
        fig, axes = plt.subplots(
            1,
            len(active_optimizers),
            figsize=(7.2 * len(active_optimizers), 5.2),
            squeeze=False,
        )
        for column_index, optimizer in enumerate(active_optimizers):
            ax = axes[0][column_index]
            for method in optimizer_methods[optimizer]:
                records = by_key.get((setting, optimizer, method), [])
                if records:
                    _plot_method_curve(ax, records, method, metric)
            ax.set_title(f"{setting} · {OPTIMIZER_DISPLAY[optimizer]}")
            ax.set_xlabel("Normalized training progress")
            ax.set_ylabel(_metric_label(metric))
            ax.grid(alpha=0.22)
            _finish_axis_legend(ax)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{setting}_{metric}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)


def _fmt(value: Any, digits: int = 4) -> str:
    number = _finite_float(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _winner(rows: Sequence[Mapping[str, Any]], field: str, lower: bool = True) -> str:
    finite = [row for row in rows if _finite_float(row.get(field)) is not None]
    if not finite:
        return "—"
    chosen_value = (min if lower else max)(float(row[field]) for row in finite)
    methods = sorted(
        str(row["method"])
        for row in finite
        if math.isclose(float(row[field]), chosen_value, rel_tol=1e-10, abs_tol=1e-12)
    )
    return ", ".join(methods)


def write_markdown(
    path: Path,
    runs: Sequence[Mapping[str, Any]],
    missing: Sequence[Mapping[str, str]],
    optimizer_methods: Optional[Mapping[str, Sequence[str]]] = None,
) -> None:
    optimizer_methods = _optimizer_methods_or_default(optimizer_methods)
    active_optimizers = _active_optimizers(optimizer_methods)
    is_immutable_32k = _active_settings_are_immutable_32k()
    if is_immutable_32k:
        target_validation_description = (
            "- **Target validation** is `target_val_loss` on the disjoint "
            "128-example target monitoring split. It is never used for selection; "
            "lower final loss and lower normalized AUC are better."
        )
        general_validation_description = (
            "- **General held-out** is `general_val_loss` on the disjoint "
            "512-example general validation split. It is the main loss-based "
            "generalization cross-check."
        )
    else:
        target_validation_description = (
            "- **Target validation** is `val_loss` on the small target/proxy set "
            "used by the selection method. Lower final loss and lower normalized "
            "AUC are better, but this set is not independent of selection."
        )
        general_validation_description = (
            "- **General held-out** is `eval_loss` on the larger evaluation set. "
            "It is the main loss-based generalization cross-check."
        )
    by_block: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in runs:
        by_block[(str(row["setting"]), str(row["optimizer"]))].append(row)

    lines = [
        "# AdamW/Muon loss-curve comparison",
        "",
        f"Successful requested runs: **{len(runs)} / "
        f"{len(list(_expected_keys(optimizer_methods)))}**. "
        f"Missing or incomplete: **{len(missing)}**.",
        "",
        "## Reading the metrics",
        "",
        target_validation_description,
        general_validation_description,
        "- **Train** compares the mean of the first and last 10% of logged steps; plots use a centered rolling mean. For merged-batch selection runs this is the returned train+target merged loss, while FullTraining returns train-only loss, so cross-method absolute train-loss ranks are not apples-to-apples.",
        "- AUC integrates loss over normalized progress from 0 to 1, making it comparable across run lengths. Lower is better. Reduction is initial minus final, so positive is improvement.",
        "- Both `muon` and legacy `hybrid` runs prioritize official `torch.optim.Muon` for eligible matrices and use auxiliary AdamW for embeddings, norms, biases, heads, and other ineligible parameters. The local Muon is used only if the official backend is unavailable or incompatible. The `muon` label uses only Muon-managed matrices for the spectral surrogate score; `hybrid` preserves mixed Muon+AdamW scoring.",
        "",
        "## Winners by setting",
        "",
        "| Setting | Optimizer | Lowest target final | Lowest target AUC | Lowest general final | Lowest general AUC | Largest within-run train reduction |",
        "|---|---|---|---|---|---|---|",
    ]
    for setting in ACTIVE_SETTINGS:
        for optimizer in active_optimizers:
            rows = by_block[(setting, optimizer)]
            lines.append(
                f"| {setting} | {OPTIMIZER_DISPLAY[optimizer]} | "
                f"{_winner(rows, 'target_final')} | {_winner(rows, 'target_normalized_auc')} | "
                f"{_winner(rows, 'general_final')} | {_winner(rows, 'general_normalized_auc')} | "
                f"{_winner(rows, 'train_reduction', lower=False)} |"
            )

    lines.extend(
        [
            "",
            "## Win counts",
            "",
            f"Counts are across the {len(ACTIVE_SETTINGS)} settings; ties count "
            "for every tied method. Raw losses are never averaged across "
            "different tasks.",
            "",
            "| Optimizer | Method | Target final | Target AUC | General final | General AUC | Available settings |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for optimizer in active_optimizers:
        counts: Dict[str, Counter[str]] = defaultdict(Counter)
        availability: Counter[str] = Counter()
        for setting in ACTIVE_SETTINGS:
            rows = by_block[(setting, optimizer)]
            for row in rows:
                availability[str(row["method"])] += 1
            for label, field in (
                ("target_final", "target_final"),
                ("target_auc", "target_normalized_auc"),
                ("general_final", "general_final"),
                ("general_auc", "general_normalized_auc"),
            ):
                for method in _winner(rows, field).split(", "):
                    if method != "—":
                        counts[method][label] += 1
        for method in optimizer_methods[optimizer]:
            lines.append(
                f"| {OPTIMIZER_DISPLAY[optimizer]} | {method} | "
                f"{counts[method]['target_final']} | {counts[method]['target_auc']} | "
                f"{counts[method]['general_final']} | {counts[method]['general_auc']} | "
                f"{availability[method]} |"
            )

    lines.extend(["", "## Detailed loss reductions", ""])
    ordered_methods = tuple(
        dict.fromkeys(
            method
            for optimizer in active_optimizers
            for method in optimizer_methods[optimizer]
        )
    )
    method_order = {method: index for index, method in enumerate(ordered_methods)}
    for setting in ACTIVE_SETTINGS:
        for optimizer in active_optimizers:
            rows = sorted(
                by_block[(setting, optimizer)], key=lambda row: method_order[str(row["method"])]
            )
            lines.extend(
                [
                    f"### {setting} · {OPTIMIZER_DISPLAY[optimizer]}",
                    "",
                    "| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |",
                    "|---|---:|---:|---:|---|",
                ]
            )
            for row in rows:
                url = str(row.get("wandb_url", ""))
                link = f"[run]({url})" if url else "—"
                lines.append(
                    f"| {row['method']} | {_fmt(row['target_initial'])} → {_fmt(row['target_final'])} "
                    f"({_fmt(row['target_reduction'])}; {_fmt(row['target_normalized_auc'])}) | "
                    f"{_fmt(row['general_initial'])} → {_fmt(row['general_final'])} "
                    f"({_fmt(row['general_reduction'])}; {_fmt(row['general_normalized_auc'])}) | "
                    f"{_fmt(row['train_initial'])} → {_fmt(row['train_final'])} "
                    f"({_fmt(row['train_reduction'])}; {_fmt(row['train_normalized_auc'])}) | {link} |"
                )
            lines.append("")

    lines.extend(["## Missing or incomplete requested runs", ""])
    if not missing:
        lines.append("None.")
    else:
        lines.extend(
            [
                "| Setting | Optimizer | Method | Status | Reason |",
                "|---|---|---|---|---|",
            ]
        )
        for row in missing:
            reason = str(row["reason"]).replace("|", "\\|")
            lines.append(
                f"| {row['setting']} | {row['optimizer_display']} | {row['method']} | "
                f"{row['status']} | {reason} |"
            )
    lines.extend(
        [
            "",
            "## W&B panel recipe",
            "",
            "In project `leena12/drpt_opus`, filter by group `<campaign>-<setting>-<optimizer>-s42`, where optimizer is `adamw`, `muon`, or legacy `hybrid`, then filter run names to the methods in this report. Use `train/global_step` as the x-axis and add panels for `train/val_loss` (target), `eval/loss` (general held-out), and `train/loss`. Keep evaluation curves unsmoothed; smoothing is useful only for `train/loss`. Exact URLs and groups are in `wandb_runs.csv`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _style_description(method: str) -> str:
    style = _line_style(method)
    return style if isinstance(style, str) else repr(style)


def write_completed_general_markdown(
    path: Path,
    runs: Sequence[Mapping[str, Any]],
    missing: Sequence[Mapping[str, str]],
    optimizer_methods: Mapping[str, Sequence[str]],
    run_roots: Sequence[Path] = (),
) -> None:
    """Write a completed-only general-evaluation snapshot report."""

    active_optimizers = _active_optimizers(optimizer_methods)
    expected = len(list(_expected_keys(optimizer_methods)))
    by_block: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in runs:
        by_block[(str(row["setting"]), str(row["optimizer"]))].append(row)

    lines = [
        "# Completed-only general evaluation snapshot",
        "",
        f"Completed requested runs included: **{len(runs)} / {expected}**. "
        f"Missing, incomplete, malformed, or ambiguous: **{len(missing)}**.",
        "",
        "> This is a partial snapshot. Every curve comes from a run with a valid "
        "step-0-plus-later evaluation history, a Trainer completion marker, and "
        "`model.safetensors`. Rankings are only among completed runs currently "
        "available in the same task/optimizer block; they are not final campaign "
        "rankings while entries remain unavailable.",
        "",
        "## Search roots",
        "",
    ]
    if run_roots:
        lines.extend(f"- `{root}`" for root in run_roots)
    else:
        lines.append("- Not recorded")

    lines.extend(
        [
            "",
            "## Reading the curves",
            "",
            "- **General eval loss** is the raw loss on the larger held-out set; lower is better.",
            "- **General eval loss reduction** is `loss at step 0 − loss at the current checkpoint`; it starts at zero and higher is better.",
            "- Normalized AUC integrates raw loss over training progress from 0 to 1; lower is better.",
            "- Task losses are not averaged across datasets.",
            "",
            "## Completed-run winners by task and optimizer",
            "",
            "| Task | Optimizer | Available | Lowest final loss | Lowest loss AUC | Largest loss reduction |",
            "|---|---|---:|---|---|---|",
        ]
    )
    for setting in ACTIVE_SETTINGS:
        for optimizer in active_optimizers:
            rows = by_block[(setting, optimizer)]
            lines.append(
                f"| {setting} | {OPTIMIZER_DISPLAY[optimizer]} | {len(rows)} / "
                f"{len(optimizer_methods[optimizer])} | "
                f"{_winner(rows, 'general_final')} | "
                f"{_winner(rows, 'general_normalized_auc')} | "
                f"{_winner(rows, 'general_reduction', lower=False)} |"
            )

    method_order = {
        (optimizer, method): index
        for optimizer in active_optimizers
        for index, method in enumerate(optimizer_methods[optimizer])
    }
    lines.extend(
        [
            "",
            "## Included completed runs",
            "",
            "| Task | Optimizer | Method | Initial → final | Reduction | Minimum | AUC | Last step |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    ordered_runs = sorted(
        runs,
        key=lambda row: (
            ACTIVE_SETTINGS.index(str(row["setting"])),
            active_optimizers.index(str(row["optimizer"])),
            method_order[(str(row["optimizer"]), str(row["method"]))],
        ),
    )
    for row in ordered_runs:
        lines.append(
            f"| {row['setting']} | {row['optimizer_display']} | {row['method']} | "
            f"{_fmt(row['general_initial'])} → {_fmt(row['general_final'])} | "
            f"{_fmt(row['general_reduction'])} | {_fmt(row['general_minimum'])} | "
            f"{_fmt(row['general_normalized_auc'])} | {_fmt(row['general_last_step'], 0)} |"
        )
    if not ordered_runs:
        lines.append("| — | — | — | — | — | — | — | — |")

    lines.extend(["", "## Missing or excluded requested runs", ""])
    if not missing:
        lines.append("None.")
    else:
        lines.extend(
            [
                "| Task | Optimizer | Method | Status | Reason |",
                "|---|---|---|---|---|",
            ]
        )
        for row in missing:
            reason = str(row["reason"]).replace("|", "\\|")
            lines.append(
                f"| {row['setting']} | {row['optimizer_display']} | "
                f"{row['method']} | {row['status']} | {reason} |"
            )

    lines.extend(
        [
            "",
            "## Plot style legend",
            "",
            "| Method | Color | Marker | Line style |",
            "|---|---|---|---|",
        ]
    )
    requested_methods = tuple(
        dict.fromkeys(
            method
            for optimizer in active_optimizers
            for method in optimizer_methods[optimizer]
        )
    )
    for method in requested_methods:
        lines.append(
            f"| {method} | `{COLORS.get(method, '#777777')}` | "
            f"`{MARKERS.get(method, 'o')}` | `{_style_description(method)}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_completed_general_outputs(
    output_dir: Path,
    runs: Sequence[Mapping[str, Any]],
    points: Sequence[Mapping[str, Any]],
    missing: Sequence[Mapping[str, str]],
    optimizer_methods: Mapping[str, Sequence[str]],
    run_roots: Sequence[Path] = (),
) -> None:
    """Write an isolated, completed-only held-out-loss report bundle."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_fields = (
        "setting", "optimizer", "optimizer_display", "method", "seed", "status",
        "val_strategy", "soft_weighting_constraint", "soft_replay_precision",
        "general_initial",
        "general_final", "general_minimum", "general_reduction",
        "general_relative_reduction_pct", "general_normalized_auc",
        "general_points", "general_last_step", "wandb_group", "wandb_run_id",
        "wandb_url", "run_name", "run_dir",
    )
    _write_csv(output_dir / "included_runs.csv", runs, summary_fields)
    _write_csv(output_dir / "general_eval_loss_summary.csv", runs, summary_fields)

    missing_fields = (
        "setting", "optimizer", "optimizer_display", "method", "status", "reason",
        "run_dir",
    )
    _write_csv(output_dir / "missing_runs.csv", missing, missing_fields)

    completed_by_key = {
        (str(row["setting"]), str(row["optimizer"]), str(row["method"])): row
        for row in runs
    }
    missing_by_key = {
        (str(row["setting"]), str(row["optimizer"]), str(row["method"])): row
        for row in missing
    }
    manifest: List[Dict[str, Any]] = []
    for key in _expected_keys(optimizer_methods):
        if key in completed_by_key:
            row = dict(completed_by_key[key])
            row["reason"] = ""
        else:
            row = dict(
                missing_by_key.get(
                    key,
                    {
                        "setting": key[0],
                        "optimizer": key[1],
                        "optimizer_display": OPTIMIZER_DISPLAY[key[1]],
                        "method": key[2],
                        "status": "missing",
                        "reason": "not collected",
                        "run_dir": "",
                    },
                )
            )
        manifest.append(row)
    manifest_fields = (
        "setting", "optimizer", "optimizer_display", "method", "status", "reason",
        "soft_weighting_constraint", "soft_replay_precision",
        "general_initial", "general_final", "general_reduction",
        "general_normalized_auc", "general_last_step", "run_name", "run_dir",
    )
    _write_csv(output_dir / "requested_runs_manifest.csv", manifest, manifest_fields)

    all_points = add_general_reduction_points(points)
    point_fields = (
        "setting", "optimizer", "method", "metric", "step", "progress", "value",
        "smoothed_value", "run_dir",
    )
    with gzip.open(
        output_dir / "general_eval_loss_points.csv.gz",
        "wt",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=point_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_points)

    requested_methods = tuple(
        dict.fromkeys(
            method
            for optimizer in _active_optimizers(optimizer_methods)
            for method in optimizer_methods[optimizer]
        )
    )
    style_rows = [
        {
            "method": method,
            "color": COLORS.get(method, "#777777"),
            "marker": MARKERS.get(method, "o"),
            "linestyle": _style_description(method),
        }
        for method in requested_methods
    ]
    _write_csv(
        output_dir / "plot_styles.csv",
        style_rows,
        ("method", "color", "marker", "linestyle"),
    )

    for metric in ("general_eval_loss", GENERAL_REDUCTION_METRIC):
        plot_grid(
            output_dir / f"{metric}_grid.png",
            all_points,
            metric,
            optimizer_methods,
        )
        for optimizer in _active_optimizers(optimizer_methods):
            plot_grid(
                output_dir / f"{optimizer}_{metric}_grid.png",
                all_points,
                metric,
                {optimizer: optimizer_methods[optimizer]},
            )
        plot_task_figures(output_dir, all_points, metric, optimizer_methods)
    write_completed_general_markdown(
        output_dir / "general_eval_loss_report.md",
        runs,
        missing,
        optimizer_methods,
        run_roots,
    )


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(
        os.environ.get("DRPT_REPO_ROOT", Path(__file__).resolve().parents[2])
    )
    default_runs_dir = Path(
        os.environ.get("DRPT_RUNS_DIR", repo_root / "SFT" / "runs")
    )
    default_reports_dir = Path(
        os.environ.get(
            "DRPT_REPORTS_DIR", repo_root / "SFT" / "eval" / "reports"
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=default_runs_dir,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--completed-general-only",
        action="store_true",
        help=(
            "Generate a completed-only general-eval snapshot. This opt-in mode "
            "records missing/incomplete/ambiguous runs instead of requiring a "
            "fully finished campaign, and leaves the strict default report unchanged."
        ),
    )
    parser.add_argument(
        "--additional-runs-dir",
        action="append",
        type=Path,
        default=[],
        help=(
            "Additional campaign root searched only with --completed-general-only; "
            "repeat the option to combine multiple roots."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_METHODS),
        default="legacy",
        help=(
            "Method bundle to compare. 'legacy' preserves the original optim_0718 "
            "report; 'loss52' expects AdamW 7 plus Muon-matrix-score 6 methods per "
            "setting; 'loss52-adamw' and 'loss52-muon' generate independent family "
            "reports; 'loss52-legacy' preserves the previous combined AdamW+Hybrid "
            "campaign and 'loss52-hybrid' isolates its mixed-score bundle; "
            "'baseline9' expects AdamW 5 plus Muon 8 methods per setting, and "
            "its '-adamw'/'-muon' profiles generate independent family reports; "
            "'muon-source' compares the two spectral scorer sources directly."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_reports_dir / "optim_0718_loss_curves",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    global ACTIVE_SETTINGS
    ACTIVE_SETTINGS = settings_for_profile(args.profile)
    optimizer_methods = optimizer_methods_for_profile(args.profile)
    try:
        if args.completed_general_only:
            run_roots = tuple(
                dict.fromkeys([args.runs_dir, *args.additional_runs_dir])
            )
            runs, points, missing = collect_completed_general_runs(
                run_roots, args.seed, optimizer_methods
            )
            write_completed_general_outputs(
                args.output_dir,
                runs,
                points,
                missing,
                optimizer_methods,
                run_roots,
            )
        else:
            runs, points, missing = collect_runs(
                args.runs_dir, args.seed, optimizer_methods
            )
            write_outputs(
                args.output_dir, runs, points, missing, optimizer_methods
            )
    except LossReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Wrote {len(runs)} successful runs and {len(missing)} missing/incomplete "
        f"entries for profile {args.profile!r} to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
