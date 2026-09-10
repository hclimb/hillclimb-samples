#!/usr/bin/env python
"""Q3: target val loss across the data / architecture / optimizer-geometry axes.

Each method is a corner of a 2x2 (global vs layer-wise selection) x (raw vs
optimizer-aware score), anchored by FullTraining which curates nothing:

    method          data  arch  optim
    FullTraining     no    no    no
    GlobalRaw       yes    no    no
    GlobalOptA      yes    no   yes
    LayerwiseRaw    yes   yes    no
    LayerwiseOptA   yes   yes   yes
    LayerwiseSoft   yes   yes   yes   + capped-simplex relaxation
    LayerwiseSoftP  yes   yes   yes   + probability-simplex relaxation

Cells that have not been trained yet render as "pending", so the table is
usable before the Global runs land and refreshes once they do.

Usage:
  python -m SFT.eval.analysis.axis_ablation [--campaign ID] [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from SFT.eval.analysis.campaign_io import (
    DEFAULT_CAMPAIGN,
    METHOD_AXES,
    METHOD_ORDER,
    discover_runs,
    final_losses,
    fmt,
    index_runs,
    results_root,
    setting_order,
    write_table,
)

BASELINE = "FullTraining"
PENDING = "pending"

# Contrasts that isolate one axis by holding the others fixed.
MAIN_EFFECTS: Sequence[Tuple[str, str, str, str]] = (
    ("data", "GlobalRaw", "FullTraining", "curate on target signal, global + raw"),
    ("architecture", "LayerwiseRaw", "GlobalRaw", "per-layer subsets, raw score"),
    ("architecture", "LayerwiseOptA", "GlobalOptA", "per-layer subsets, OptA score"),
    ("optim", "GlobalOptA", "GlobalRaw", "optimizer geometry, global subset"),
    ("optim", "LayerwiseOptA", "LayerwiseRaw", "optimizer geometry, per-layer subsets"),
    ("relaxation", "LayerwiseSoft", "LayerwiseOptA", "capped simplex vs hard top-k"),
    ("relaxation", "LayerwiseSoftP", "LayerwiseOptA", "probability simplex vs hard top-k"),
)


def _mark(flag: bool) -> str:
    return "O" if flag else "X"


def collect(runs_by_key, settings) -> Dict[Tuple[str, str], dict]:
    table: Dict[Tuple[str, str], dict] = {}
    for setting in settings:
        for method in METHOD_ORDER:
            run = runs_by_key.get((setting, method))
            if run is None:
                continue
            losses = final_losses(run)
            if losses is None:
                continue
            table[(setting, method)] = losses
    return table


def noise_floor(table, settings) -> float:
    """Largest late-window wobble of any curve — the floor a delta must clear.

    These are single-seed runs, so there is no across-seed variance to quote.
    How much a curve still moves over its final evaluations is the cheapest
    honest lower bound on what counts as a real difference.
    """
    spreads = [
        values["target_tail_spread"]
        for values in table.values()
        if values["target_tail_spread"] == values["target_tail_spread"]
    ]
    return max(spreads) if spreads else 0.0


def build_axis_table(table, settings, out_dir: Path, metric: str) -> None:
    header = ["method", "data", "arch", "optim", "relaxation"] + list(settings)
    rows: List[List[object]] = []
    for method in METHOD_ORDER:
        axes = METHOD_AXES[method]
        row: List[object] = [
            method,
            _mark(axes.data),
            _mark(axes.architecture),
            _mark(axes.optim),
            "—" if axes.relaxation == "none" else axes.relaxation,
        ]
        for setting in settings:
            values = table.get((setting, method))
            row.append(fmt(values[metric]) if values else PENDING)
        rows.append(row)
    write_table(out_dir / f"axis_table_{metric}", header, rows)


def build_delta_table(table, settings, out_dir: Path) -> None:
    header = ["method", "data", "arch", "optim"] + [f"{s} Δ" for s in settings]
    rows: List[List[object]] = []
    for method in METHOD_ORDER:
        if method == BASELINE:
            continue
        axes = METHOD_AXES[method]
        row: List[object] = [method, _mark(axes.data), _mark(axes.architecture), _mark(axes.optim)]
        for setting in settings:
            values = table.get((setting, method))
            base = table.get((setting, BASELINE))
            if not values or not base:
                row.append(PENDING)
            else:
                row.append(f"{values['target_final'] - base['target_final']:+.4f}")
        rows.append(row)
    write_table(out_dir / "axis_delta_vs_fulltraining", header, rows)


def build_main_effects(table, settings, out_dir: Path, floor: float) -> List[List[object]]:
    header = (
        ["axis", "contrast", "holds fixed"]
        + list(settings)
        + ["mean Δ", "settings clearing floor", "verdict"]
    )
    rows: List[List[object]] = []
    for axis, treatment, control, description in MAIN_EFFECTS:
        row: List[object] = [axis, f"{treatment} − {control}", description]
        deltas: List[float] = []
        for setting in settings:
            a = table.get((setting, treatment))
            b = table.get((setting, control))
            if not a or not b:
                row.append(PENDING)
                continue
            delta = a["target_final"] - b["target_final"]
            deltas.append(delta)
            row.append(f"{delta:+.4f}")
        if len(deltas) == len(settings) and deltas:
            mean = sum(deltas) / len(deltas)
            # A mean can clear the floor on the strength of one setting, so
            # report how many settings move on their own before calling it.
            clearing = [d for d in deltas if abs(d) >= floor]
            consistent = clearing and all((d < 0) == (mean < 0) for d in clearing)
            row.append(f"{mean:+.4f}")
            row.append(f"{len(clearing)}/{len(deltas)}")
            if abs(mean) < floor:
                row.append(f"within noise (|Δ| < {floor:.4f})")
            elif len(clearing) == 1:
                row.append(("better" if mean < 0 else "worse") + " — one setting only")
            elif consistent:
                row.append("better" if mean < 0 else "worse")
            else:
                row.append("mixed sign across settings")
        else:
            row += [PENDING, PENDING, PENDING]
        rows.append(row)
    write_table(out_dir / "axis_main_effects", header, rows)
    return rows


def plot_axis_effects(table, settings, out_dir: Path, floor: float) -> None:
    from SFT.eval.analysis.viz import (
        METHOD_COLORS,
        TEXT_MUTED,
        TEXT_SECONDARY,
        apply_style,
        strip_spines,
    )
    import matplotlib.pyplot as plt

    apply_style()
    methods = [m for m in METHOD_ORDER if m != BASELINE]
    # reason_code's effect is ~100x the others', so a shared y-axis would render
    # four of five settings as flat lines. Each setting gets its own scale, and
    # every bar carries its value so nothing has to be compared by height across
    # panels.
    fig, axes = plt.subplots(
        1, len(settings), figsize=(2.55 * len(settings) + 1.0, 4.2)
    )
    axes = list(axes) if len(settings) > 1 else [axes]

    drew = False
    for ax, setting in zip(axes, settings):
        strip_spines(ax)
        ax.grid(axis="x", visible=False)
        positions, values, colors, labels = [], [], [], []
        for index, method in enumerate(methods):
            entry = table.get((setting, method))
            base = table.get((setting, BASELINE))
            if not entry or not base:
                continue
            positions.append(index)
            values.append(entry["target_final"] - base["target_final"])
            colors.append(METHOD_COLORS[method])
            labels.append(method)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([""] * len(methods))
        ax.set_title(setting, fontsize=9)
        if not values:
            ax.annotate(
                "pending", xy=(0.5, 0.5), xycoords="axes fraction",
                ha="center", color=TEXT_MUTED, fontsize=9,
            )
            continue
        drew = True
        ax.bar(positions, values, width=0.72, color=colors, linewidth=0)
        span = max(abs(min(values)), abs(max(values)), floor) * 1.45
        ax.set_ylim(-span, span)
        ax.axhspan(-floor, floor, color=TEXT_MUTED, alpha=0.16, linewidth=0, zorder=0)
        ax.axhline(0, color=TEXT_SECONDARY, linewidth=1.0)
        for position, value in zip(positions, values):
            # Outside the bar end, never on top of the fill.
            ax.annotate(
                f"{value:+.4f}",
                xy=(position, value),
                xytext=(0, 4 if value >= 0 else -10),
                textcoords="offset points",
                ha="center", fontsize=6.8, color=TEXT_SECONDARY,
            )

    if not drew:
        plt.close(fig)
        return

    axes[0].set_ylabel("target val loss − FullTraining  (negative = better)")
    # Methods keep their reserved hue even before their runs land; say so
    # rather than showing a legend entry with nothing drawn against it.
    trained = {m for m in methods if any((s, m) in table for s in settings)}
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[m], linewidth=0)
        for m in methods
    ]
    names = [m if m in trained else f"{m} (pending)" for m in methods]
    fig.legend(handles, names, loc="lower center", ncol=len(methods), fontsize=8)
    fig.suptitle(
        "Every curated method against the no-curation anchor", fontsize=11, y=1.0
    )
    fig.text(
        0.5, 0.075,
        f"shaded band = single-curve noise floor ±{floor:.4f} nats · "
        "each panel has its own y-scale",
        ha="center", fontsize=7.5, color=TEXT_MUTED,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.96))
    fig.savefig(out_dir / "axis_effects.png", bbox_inches="tight")
    plt.close(fig)


def build_report(table, settings, out_dir: Path, floor: float, effects) -> None:
    missing = [
        f"{setting}/{method}"
        for setting in settings
        for method in METHOD_ORDER
        if (setting, method) not in table
    ]
    lines = [
        "# Three-axis comparison of target validation loss",
        "",
        "Axes: **data** (curate on target signal vs use everything), "
        "**architecture** (per-layer subsets vs one global subset), "
        "**optim** (score in the optimizer's geometry vs raw gradients).",
        "",
        f"Noise floor: **±{floor:.4f} nats** — the largest late-window wobble of "
        "any single curve in this campaign. All runs are seed 42 only, so there "
        "is no across-seed variance to quote; a mean effect smaller than this "
        "floor is not a result.",
        "",
    ]
    if missing:
        lines += [
            f"**{len(missing)} cells pending**: "
            + ", ".join(sorted(missing))
            + ". Re-run this script once those runs finish.",
            "",
        ]
    lines += [
        "## Main effects",
        "",
        "Each row changes exactly one axis and holds the others fixed.",
        "",
    ]
    header = (
        ["axis", "contrast", "holds fixed"]
        + list(settings)
        + ["mean Δ", "settings clearing floor", "verdict"]
    )
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in effects:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    lines += [
        "",
        "## Files",
        "",
        "| file | contents |",
        "|---|---|",
        "| `axis_table_target_final.*` | final target val loss per cell |",
        "| `axis_table_target_best.*` | best-over-steps target val loss |",
        "| `axis_table_general_final.*` | general val loss (the retention side) |",
        "| `axis_delta_vs_fulltraining.*` | Δ against the no-curation anchor |",
        "| `axis_main_effects.*` | the one-axis contrasts above |",
        "| `axis_effects.png` | the same deltas as a figure |",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    runs = discover_runs(args.campaign, optimizer=args.optimizer)
    runs_by_key = index_runs(runs)
    settings = [s for s in setting_order() if any(k[0] == s for k in runs_by_key)]
    out_dir = results_root(args.campaign, Path(args.out) if args.out else None) / "axis_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = collect(runs_by_key, settings)
    floor = noise_floor(table, settings)
    for metric in ("target_final", "target_best", "general_final"):
        build_axis_table(table, settings, out_dir, metric)
    build_delta_table(table, settings, out_dir)
    effects = build_main_effects(table, settings, out_dir, floor)
    plot_axis_effects(table, settings, out_dir, floor)
    build_report(table, settings, out_dir, floor, effects)

    present = sorted({key[1] for key in table})
    pending = [m for m in METHOD_ORDER if m not in present]
    print(f"campaign     : {args.campaign}")
    print(f"cells        : {len(table)} of {len(settings) * len(METHOD_ORDER)}")
    print(f"noise floor  : ±{floor:.4f} nats")
    if pending:
        print(f"pending      : {', '.join(pending)}")
    print(f"wrote        : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
