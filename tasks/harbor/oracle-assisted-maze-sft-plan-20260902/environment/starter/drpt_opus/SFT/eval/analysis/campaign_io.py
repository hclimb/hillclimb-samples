"""Shared reader for a completed Dolci32k campaign directory.

Every analysis in this package works from the artifacts a finished run already
writes — no re-training and no checkpoint loading:

  run_status.json             setting / method label / status / artifact pin
  run_metadata.json           resolved hyperparameters and pool composition
  evaluation_results.json     target and general val-loss series
  selection_diagnostics.json  per-step selection diagnostics (selection runs)
  selection_domain_summary.json  per-domain and per-source selection mass
  selection_records.json      sampled candidate identity and per-layer decisions

The method axis table below is the one thing here that is a modelling choice
rather than a file read, so it is stated once and imported everywhere.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAMPAIGN = "dolci32k-qwen3_1_7b-s42"


def campaign_root(campaign: str, runs_dir: Optional[Path] = None) -> Path:
    root = Path(runs_dir) if runs_dir else REPO_ROOT / "SFT" / "runs"
    return root / "campaigns" / campaign


def results_root(campaign: str, out_dir: Optional[Path] = None) -> Path:
    base = Path(out_dir) if out_dir else Path(__file__).resolve().parent / "results"
    return base / campaign


# --------------------------------------------------------------------------
# The three-axis reading of each method.
#
#   data  — does the update use a target-signal-selected subset, or all data?
#   arch  — is the subset chosen per layer, or once globally?
#   optim — is the score computed in the optimizer's geometry (P g), or raw?
#
# `relaxation` separates the two continuous methods from the hard top-k ones:
# they share all three axis values with LayerwiseOptA and differ only in the
# feasible set the weights are optimized over.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MethodAxes:
    label: str
    data: bool
    architecture: bool
    optim: bool
    relaxation: str  # "none", "capped_simplex", "probability_simplex"
    order: int


METHOD_AXES: Mapping[str, MethodAxes] = {
    spec.label: spec
    for spec in (
        MethodAxes("FullTraining", False, False, False, "none", 0),
        MethodAxes("GlobalRaw", True, False, False, "none", 1),
        MethodAxes("GlobalOptA", True, False, True, "none", 2),
        MethodAxes("LayerwiseRaw", True, True, False, "none", 3),
        MethodAxes("LayerwiseOptA", True, True, True, "none", 4),
        MethodAxes("LayerwiseSoft", True, True, True, "capped_simplex", 5),
        MethodAxes("LayerwiseSoftP", True, True, True, "probability_simplex", 6),
    )
}

METHOD_ORDER: Tuple[str, ...] = tuple(
    spec.label for spec in sorted(METHOD_AXES.values(), key=lambda s: s.order)
)

# Selection methods only; FullTraining writes no selection artifacts.
SELECTION_METHODS: Tuple[str, ...] = tuple(
    label for label in METHOD_ORDER if METHOD_AXES[label].data
)


def setting_order(profile: str = "dolci32k") -> Tuple[str, ...]:
    """Canonical setting order, read from the immutable profile registry."""
    import importlib

    module = importlib.import_module(f"SFT.data.{profile}.profile")
    return tuple(module.SETTING_ORDER)


def setting_targets(profile: str = "dolci32k") -> Dict[str, str]:
    import importlib

    module = importlib.import_module(f"SFT.data.{profile}.profile")
    return {
        name: str(values["target"]) for name, values in module.SETTINGS.items()
    }


@dataclass(frozen=True)
class Run:
    setting: str
    method: str
    optimizer: str
    path: Path
    complete: bool

    def artifact(self, name: str) -> Optional[dict | list]:
        target = self.path / name
        if not target.is_file():
            return None
        with target.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @property
    def axes(self) -> Optional[MethodAxes]:
        return METHOD_AXES.get(self.method)


def discover_runs(
    campaign: str = DEFAULT_CAMPAIGN,
    *,
    runs_dir: Optional[Path] = None,
    optimizer: str = "adamw",
    complete_only: bool = True,
) -> List[Run]:
    """Enumerate runs from run_status.json rather than by parsing directory names."""
    root = campaign_root(campaign, runs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"campaign directory not found: {root}")

    runs: List[Run] = []
    for entry in sorted(root.iterdir()):
        status_path = entry / "run_status.json"
        if not entry.is_dir() or not status_path.is_file():
            continue
        with status_path.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
        complete = (
            str(status.get("status")) == "complete" and (entry / "_SUCCESS").is_file()
        )
        if complete_only and not complete:
            continue
        family = str(status.get("optimizer_family", "adamw"))
        if optimizer and family != optimizer:
            continue
        runs.append(
            Run(
                setting=str(status["setting"]),
                method=str(status["method"]),
                optimizer=family,
                path=entry,
                complete=complete,
            )
        )
    return runs


def index_runs(runs: Iterable[Run]) -> Dict[Tuple[str, str], Run]:
    return {(run.setting, run.method): run for run in runs}


def final_losses(run: Run) -> Optional[Dict[str, float]]:
    """Final-step and best-step target/general val loss for one run."""
    series = run.artifact("evaluation_results.json")
    if not series:
        return None
    rows = [row for row in series if row.get("target_val_loss") is not None]
    if not rows:
        return None
    trained = [row for row in rows if int(row.get("step", 0)) > 0] or rows
    last = trained[-1]
    target_values = [float(row["target_val_loss"]) for row in trained]
    general_values = [
        float(row["general_val_loss"])
        for row in trained
        if row.get("general_val_loss") is not None
    ]
    # Late-window spread is the cheapest honest noise proxy available from a
    # single seed: how much the curve still moves once it has flattened.
    tail = target_values[-3:] if len(target_values) >= 3 else target_values
    return {
        "final_step": float(last.get("step", 0)),
        "target_final": float(last["target_val_loss"]),
        "target_best": min(target_values),
        "general_final": float(last["general_val_loss"])
        if last.get("general_val_loss") is not None
        else float("nan"),
        "general_best": min(general_values) if general_values else float("nan"),
        "target_tail_spread": max(tail) - min(tail),
        "wall_hours": float(last.get("wall_time", float("nan"))) / 3600.0,
    }


def diagnostics_frame(run: Run) -> List[dict]:
    payload = run.artifact("selection_diagnostics.json")
    return list(payload) if payload else []


def domain_summary(run: Run) -> Optional[dict]:
    payload = run.artifact("selection_domain_summary.json")
    return payload if isinstance(payload, dict) else None


def selection_records(run: Run) -> Optional[dict]:
    """Sampled candidate identities and exact layer/group selection decisions."""
    payload = run.artifact("selection_records.json")
    return payload if isinstance(payload, dict) else None


def summarize_series(
    rows: Sequence[dict], key: str, *, head: int = 200, tail: int = 200
) -> Optional[Dict[str, float]]:
    """Mean / final / early / late for one diagnostic key across a run."""
    values = [float(row[key]) for row in rows if key in row]
    if not values:
        return None
    return {
        "mean": sum(values) / len(values),
        "final": values[-1],
        "early": sum(values[:head]) / min(len(values), head),
        "late": sum(values[-tail:]) / min(len(values), tail),
        "steps": float(len(values)),
    }


def write_table(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    """Write one table as both CSV and a GitHub-flavoured Markdown table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(str(name) for name in header) + "\n")
        for row in rows:
            handle.write(
                ",".join("" if value is None else str(value) for value in row) + "\n"
            )
    md_path = path.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(str(name) for name in header) + " |\n")
        handle.write("|" + "|".join("---" for _ in header) + "|\n")
        for row in rows:
            handle.write(
                "| "
                + " | ".join("" if value is None else str(value) for value in row)
                + " |\n"
            )


def fmt(value: Optional[float], places: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and value != value:  # NaN
        return "—"
    return f"{value:.{places}f}"
