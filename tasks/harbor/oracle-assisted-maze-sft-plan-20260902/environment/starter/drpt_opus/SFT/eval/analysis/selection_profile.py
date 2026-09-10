#!/usr/bin/env python
"""Q1: what weights each selection method assigned, and what data it consumed.

Reads only artifacts that finished runs already wrote. Produces, per campaign:

  weighting_contract.md    the weighting RULE per method, resolved from the
                           YAML chain plus the solver code
  weight_realization.*     what the rule actually converged to (ESS, entropy,
                           boundary saturation, agreement with hard OptA)
  domain_lift.*            which domains/sources each method over- or
                           under-sampled relative to a uniform draw
  data_budget.md           how much data each method actually integrated
  weight_concentration.png ESS over training, per setting
  domain_lift.png          lift heatmaps, per setting

Usage:
  python -m SFT.eval.analysis.selection_profile [--campaign ID] [--out DIR]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from SFT.eval.analysis.campaign_io import (
    DEFAULT_CAMPAIGN,
    diagnostics_frame,
    discover_runs,
    domain_summary,
    fmt,
    index_runs,
    results_root,
    setting_order,
    setting_targets,
    summarize_series,
    write_table,
)

# The weighting rule per method. Sources, in resolution order:
#   configs/dolci32k/<setting>/defaults.yaml  (batch 16, selection_frac 0.5)
#   configs/dolci32k_methods/<method>.yaml   (feasible set, solver settings)
#   drpt/selection/state.py, advanced_solvers.py, optimizer_aware.py
WEIGHTING_CONTRACT: Sequence[Dict[str, str]] = (
    {
        "method": "FullTraining",
        "feasible_set": "w = 1 for all 16",
        "mass": "16",
        "per_layer": "n/a",
        "score": "none (no target signal)",
        "solver": "none",
    },
    {
        "method": "GlobalRaw",
        "feasible_set": "w in {0,1}^16",
        "mass": "8",
        "per_layer": "no — one subset shared by every layer",
        "score": "sum over layers of <g_i, g_target>",
        "solver": "exact top-k",
    },
    {
        "method": "GlobalOptA",
        "feasible_set": "w in {0,1}^16",
        "mass": "8",
        "per_layer": "no — one subset shared by every layer",
        "score": "sum over layers of <P g_i, g_target>, P = AdamW inv-RMS",
        "solver": "exact top-k",
    },
    {
        "method": "LayerwiseRaw",
        "feasible_set": "w in {0,1}^16",
        "mass": "8 per layer",
        "per_layer": "yes — independent subset per layer",
        "score": "<g_i, g_target> at that layer",
        "solver": "exact top-k",
    },
    {
        "method": "LayerwiseOptA",
        "feasible_set": "w in {0,1}^16",
        "mass": "8 per layer",
        "per_layer": "yes — independent subset per layer",
        "score": "<P g_i, g_target>, P = AdamW inv-RMS",
        "solver": "exact top-k",
    },
    {
        "method": "LayerwiseSoft",
        "feasible_set": "0 <= w <= 1 (capped simplex)",
        "mass": "8 per layer",
        "per_layer": "yes — independent weights per layer",
        "score": "<P g_i, g_target>, token-weighted",
        "solver": "Projected Adam, 20 steps, lr 0.1, gamma 0",
    },
    {
        "method": "LayerwiseSoftP",
        "feasible_set": "w >= 0 (probability simplex)",
        "mass": "1 per layer",
        "per_layer": "yes — independent weights per layer",
        "score": "<P g_i, g_target>, token-weighted",
        "solver": "Projected Adam, 20 steps, lr 0.1, gamma 0",
    },
    {
        "method": "LayerwiseMuonSur",
        "feasible_set": "w in {0,1}^16",
        "mass": "8 per layer",
        "per_layer": "yes — independent subset per layer",
        "score": "uniform target singular-mode support",
        "solver": "exact top-k",
    },
    {
        "method": "LayerwiseMuonPSur",
        "feasible_set": "w in {0,1}^16",
        "mass": "8 per layer",
        "per_layer": "yes — independent subset per layer",
        "score": "singular-value-weighted target-mode support",
        "solver": "exact top-k",
    },
    {
        "method": "LayerwiseMuonSatSur",
        "feasible_set": "w in {0,1}^16",
        "mass": "8 per layer",
        "per_layer": "yes — independent subset per layer",
        "score": "uniform target-mode support with saturation",
        "solver": "greedy saturated support",
    },
    {
        "method": "LayerwiseMuonSatPSur",
        "feasible_set": "w in {0,1}^16",
        "mass": "8 per layer",
        "per_layer": "yes — independent subset per layer",
        "score": "singular-value target-mode support with saturation",
        "solver": "greedy saturated support",
    },
)

# Diagnostic keys, by what the reader should take from them.
REALIZATION_KEYS = (
    ("soft/ess", "ESS", "effective number of candidates carrying weight (of 16)"),
    ("soft/normalized_entropy", "norm. entropy", "1.0 = uniform, 0 = one candidate"),
    ("soft/boundary_fraction", "boundary frac", "share of weights pinned at 0 or 1"),
    ("soft/opta_topk_overlap", "OptA overlap", "agreement with the hard top-k it relaxes"),
)

HARD_KEYS = (
    ("selection/n_selected", "n selected", "candidates kept per layer (of 16)"),
    (
        "update/adamw_lm_head/selected_score_margin",
        "lm_head margin",
        "selected minus non-selected mean alignment at the LM head",
    ),
    (
        # Global methods only: the margin of the one pooled decision, which has
        # no layer-wise counterpart.
        "update/global/selected_score_margin",
        "pooled margin",
        "selected minus non-selected mean alignment of the shared subset",
    ),
)


def _hard_ess(rows: Sequence[dict]) -> Optional[float]:
    """A hard top-k subset has ESS exactly k; report it so panels share a scale."""
    values = [float(row["selection/n_selected"]) for row in rows if "selection/n_selected" in row]
    if not values:
        return None
    return sum(values) / len(values)


def build_weighting_contract(path: Path, optimizer: str) -> None:
    header = ["method", "feasible set", "mass", "per-layer?", "score geometry", "solver"]
    rows = [
        [
            spec["method"],
            spec["feasible_set"],
            spec["mass"],
            spec["per_layer"],
            spec["score"],
            spec["solver"],
        ]
        for spec in _weighting_contract_for_optimizer(optimizer)
    ]
    write_table(path, header, rows)

    lines = [
        "# Weighting contract",
        "",
        "The rule each method applies to one logical window of **16 candidates**,",
        "resolved from `configs/dolci32k/<setting>/defaults.yaml`",
        "(`batch_size: 16`, `selection_frac: 0.5`) and",
        "`configs/dolci32k_methods/<method>.yaml`.",
        f"Optimizer family: **{optimizer}**.",
        "",
        "**The hard methods keep 8 of 16, not 4.** Top-4 was the earlier",
        "`baseline9`/`loss52` setting, which ran at batch 8.",
        "",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    lines += ["", "Notes", ""]
    if optimizer == "adamw":
        lines += [
            "- `P` is AdamW's diagonal inverse-RMS preconditioner",
            "  (`drpt/selection/optimizer_aware.py::_get_adamw_inv_rms`). Before the",
            "  optimizer has state it falls back to identity, so step 0 is raw geometry.",
        ]
    else:
        lines += [
            "- Muon Soft forms the weighted candidate gradient, applies Muon's",
            "  matrix update map, and aligns that update with the raw target gradient.",
            "  AdamW-managed biases retain their AdamW transform.",
            "- The four Muon surrogate methods replace that nonlinear objective",
            "  with target singular-mode support; `P` variants weight modes by",
            "  target singular values and `Sat` variants use greedy saturation.",
        ]
    lines += [
        "- `LayerwiseSoft` and `LayerwiseSoftP` differ **only** in the feasible set:",
        "  mass 8 spread over [0,1] versus mass 1 on the probability simplex. Both",
        "  optimize the same family-specific token-weighted alignment objective.",
        "- Selection changes only *which* candidates contribute and with what weight;",
        "  the update itself still uses raw gradients.",
    ]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selection_methods_for_optimizer(optimizer: str) -> tuple[str, ...]:
    from SFT.data.dolci32k.profile import ADAMW_METHODS, MUON_METHODS

    family = str(optimizer).lower()
    if family == "adamw":
        # The two Global axis methods run in their own array but belong in the
        # same AdamW post-hoc selection report whenever those runs are present.
        methods = ("GlobalRaw", "GlobalOptA", *ADAMW_METHODS)
    elif family == "muon":
        methods = MUON_METHODS
    else:
        raise ValueError("optimizer must be 'adamw' or 'muon'")
    return tuple(method for method in methods if method != "FullTraining")


def _weighting_contract_for_optimizer(optimizer: str) -> tuple[Dict[str, str], ...]:
    """Return only this optimizer family's methods, with truthful Soft geometry."""

    family = str(optimizer).lower()
    ordered_methods = ("FullTraining", *_selection_methods_for_optimizer(family))
    by_method = {spec["method"]: spec for spec in WEIGHTING_CONTRACT}
    rows = [dict(by_method[method]) for method in ordered_methods]
    if family == "muon":
        muon_soft_score = (
            "<MuonMap(sum_i w_i g_i), g_target>; AdamW transform for biases"
        )
        for row in rows:
            if row["method"] in ("LayerwiseSoft", "LayerwiseSoftP"):
                row["score"] = muon_soft_score
    return tuple(rows)


def build_campaign_completeness(
    runs_by_key, settings, methods: Sequence[str], out_dir: Path
) -> List[tuple[str, str]]:
    """Write every expected selection cell so partial campaigns are explicit."""

    missing: List[tuple[str, str]] = []
    rows: List[List[str]] = []
    for setting in settings:
        for method in methods:
            present = (setting, method) in runs_by_key
            rows.append([setting, method, "complete" if present else "missing"])
            if not present:
                missing.append((setting, method))
    write_table(
        out_dir / "campaign_completeness",
        ["setting", "method", "status"],
        rows,
    )
    return missing


def build_weight_realization(
    runs_by_key, settings, out_dir: Path, methods: Sequence[str]
) -> List[dict]:
    header = [
        "setting",
        "method",
        "metric",
        "mean",
        "first 200 steps",
        "last 200 steps",
        "final",
    ]
    rows: List[List[object]] = []
    records: List[dict] = []
    for setting in settings:
        for method in methods:
            run = runs_by_key.get((setting, method))
            if run is None:
                continue
            frame = diagnostics_frame(run)
            if not frame:
                continue
            for key, label, _ in REALIZATION_KEYS + HARD_KEYS:
                stats = summarize_series(frame, key)
                if stats is None:
                    continue
                rows.append([
                    setting,
                    method,
                    label,
                    fmt(stats["mean"]),
                    fmt(stats["early"]),
                    fmt(stats["late"]),
                    fmt(stats["final"]),
                ])
                records.append({
                    "setting": setting,
                    "method": method,
                    "key": key,
                    **stats,
                })
    write_table(out_dir / "weight_realization", header, rows)
    return records


def plot_weight_concentration(runs_by_key, settings, out_dir: Path) -> None:
    from SFT.eval.analysis.viz import (
        METHOD_COLORS,
        SURFACE,
        TEXT_MUTED,
        TEXT_SECONDARY,
        apply_style,
        strip_spines,
    )
    import matplotlib.pyplot as plt

    apply_style()
    soft_methods = ("LayerwiseSoft", "LayerwiseSoftP")
    fig, axes = plt.subplots(
        1, len(settings), figsize=(3.1 * len(settings), 3.2), sharey=True
    )
    axes = list(axes) if len(settings) > 1 else [axes]

    drew_any = False
    for ax, setting in zip(axes, settings):
        strip_spines(ax)
        # Reference levels: uniform weighting over the window, and the mass the
        # hard methods spend. Together they bracket "is this actually selecting?".
        ax.axhline(16, color=TEXT_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.axhline(8, color=TEXT_MUTED, linewidth=1.0, linestyle=(0, (1, 3)))
        for method in soft_methods:
            run = runs_by_key.get((setting, method))
            if run is None:
                continue
            frame = diagnostics_frame(run)
            steps = [row["step"] for row in frame if "soft/ess" in row]
            values = [row["soft/ess"] for row in frame if "soft/ess" in row]
            if not values:
                continue
            drew_any = True
            ax.plot(steps, values, color=METHOD_COLORS[method], label=method)
            ax.annotate(
                f"{values[-1]:.1f}",
                xy=(steps[-1], values[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                color=TEXT_SECONDARY,
                fontsize=8,
                va="center",
            )
        ax.set_title(setting, fontsize=9)
        ax.set_xlabel("training step")
        ax.set_ylim(0, 17.5)

    if not drew_any:
        plt.close(fig)
        return

    axes[0].set_ylabel("effective sample size (of 16)")
    # Label the reference levels on the last panel, where the curves have
    # climbed away from y=8, and back the text with the surface colour so it
    # stays legible if a curve does cross it.
    for level, text in ((16, "uniform (16)"), (8, "hard top-8")):
        axes[-1].annotate(
            text,
            xy=(0.98, level),
            xycoords=("axes fraction", "data"),
            color=TEXT_MUTED,
            fontsize=7.5,
            ha="right",
            va="bottom",
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0},
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8.5)
    fig.suptitle(
        "How concentrated the continuous weights actually became",
        fontsize=11,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out_dir / "weight_concentration.png", bbox_inches="tight")
    plt.close(fig)


def build_domain_lift(
    runs_by_key, settings, out_dir: Path, methods: Sequence[str]
) -> Dict[str, dict]:
    header = ["setting", "method", "grouping", "name", "candidate share", "selection rate", "lift"]
    rows: List[List[object]] = []
    payloads: Dict[str, dict] = {}
    for setting in settings:
        for method in methods:
            run = runs_by_key.get((setting, method))
            if run is None:
                continue
            summary = domain_summary(run)
            if not summary:
                continue
            payloads[f"{setting}/{method}"] = summary
            for grouping, key in (("domain", "by_domain"), ("source", "by_source_dataset")):
                for name, values in sorted(summary.get(key, {}).items()):
                    rows.append([
                        setting,
                        method,
                        grouping,
                        name,
                        fmt(values["candidate_share"]),
                        fmt(values["selection_rate"]),
                        fmt(values["lift"], 3),
                    ])
    write_table(out_dir / "domain_lift", header, rows)
    return payloads


def plot_domain_lift(
    payloads: Dict[str, dict],
    settings,
    out_dir: Path,
    expected_methods: Sequence[str],
) -> None:
    from SFT.eval.analysis.viz import (
        DIVERGING,
        TEXT_MUTED,
        TEXT_PRIMARY,
        apply_style,
        diverging_norm,
    )
    import matplotlib.pyplot as plt

    apply_style()
    methods = [
        method
        for method in expected_methods
        if any(f"{setting}/{method}" in payloads for setting in settings)
    ]
    if not methods:
        return
    domains = sorted({
        name
        for key, summary in payloads.items()
        for name in summary.get("by_domain", {})
    })
    if not domains:
        return

    all_values = [
        math.log2(summary["by_domain"][domain]["lift"])
        for summary in payloads.values()
        for domain in domains
        if domain in summary.get("by_domain", {})
        and summary["by_domain"][domain]["lift"] > 0
    ]
    # Lift is a polarity around 1.0 (prefers vs avoids a uniform draw), so the
    # scale is diverging with a neutral midpoint, not a one-hue ramp. It is also
    # a ratio: colour it on log2 so "twice as often" and "half as often" sit
    # equally far from neutral, and so the scale never implies a negative lift.
    norm = diverging_norm(all_values, center=0.0)
    absent = DIVERGING.copy()
    absent.set_bad("#dcdad2")

    fig, axes = plt.subplots(
        1, len(settings), figsize=(2.55 * len(settings) + 1.4, 0.42 * len(domains) + 2.4)
    )
    axes = list(axes) if len(settings) > 1 else [axes]
    mesh = None
    for ax, setting in zip(axes, settings):
        grid, shaded = [], []
        for domain in domains:
            row, row_log = [], []
            for method in methods:
                summary = payloads.get(f"{setting}/{method}")
                entry = (summary or {}).get("by_domain", {}).get(domain)
                lift = entry["lift"] if entry else float("nan")
                row.append(lift)
                row_log.append(math.log2(lift) if lift and lift > 0 else float("nan"))
            grid.append(row)
            shaded.append(row_log)
        mesh = ax.imshow(shaded, cmap=absent, norm=norm, aspect="auto")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(domains)))
        ax.set_yticklabels(domains if ax is axes[0] else [], fontsize=7.5)
        ax.set_title(setting, fontsize=9)
        ax.grid(False)
        # The value inside every cell: this is the relief for the sub-3:1
        # contrast warning, and it lets the figure be read exactly.
        for r, domain in enumerate(domains):
            for c, _ in enumerate(methods):
                value = grid[r][c]
                shown = f"{value:.2f}" if value == value else "n/a"
                ax.text(
                    c, r, shown,
                    ha="center", va="center", fontsize=6.6,
                    color=TEXT_PRIMARY if value == value else TEXT_MUTED,
                )

    if mesh is not None:
        bar = fig.colorbar(mesh, ax=axes, fraction=0.02, pad=0.015)
        bar.set_label("selection lift (1.0 = uniform draw)", fontsize=8)
        ticks = [t for t in (0.25, 0.5, 1.0, 2.0, 4.0) if norm.vmin <= math.log2(t) <= norm.vmax]
        bar.set_ticks([math.log2(t) for t in ticks])
        bar.set_ticklabels([f"{t:g}x" for t in ticks])
        bar.ax.tick_params(labelsize=7.5)
    fig.suptitle(
        "Which domains each method over- and under-sampled", fontsize=11, y=1.0
    )
    fig.savefig(out_dir / "domain_lift.png", bbox_inches="tight")
    plt.close(fig)


def build_data_budget(
    runs_by_key, settings, out_dir: Path, records, methods: Sequence[str]
) -> None:
    targets = setting_targets()
    lines = [
        "# Data budget — what each method actually looked at",
        "",
        "Every method traverses the **same** pinned candidate order",
        "(`candidate_orders/<pool>.jsonl`): 2000 steps x 16 candidates = 32000",
        "examples, exactly one pass over the pool without replacement. The methods",
        "differ only in the weight each candidate's gradient carries.",
        "",
        "| quantity | value |",
        "|---|---|",
        "| candidates traversed | 32000 (2000 windows of 16) |",
        "| target-gradient split | 64 examples (`target_grad`) |",
        "| target val split | 128 examples (`target_val`) |",
        "| general val split | 512 examples |",
        "| domain tracking coverage | every 10th step -> 200 of 2000 windows |",
        "",
        "## Effective examples per window",
        "",
        "For the hard methods this is exactly 8 of 16. For the continuous methods",
        "it is the effective sample size of the weight vector, which is what the",
        "gradient variance actually sees:",
        "",
        "| setting | method | mean ESS (of 16) | reading |",
        "|---|---|---|---|",
    ]

    by_key = {(r["setting"], r["method"], r["key"]): r for r in records}
    for setting in settings:
        for method in methods:
            run = runs_by_key.get((setting, method))
            if run is None:
                continue
            soft = by_key.get((setting, method, "soft/ess"))
            hard = by_key.get((setting, method, "selection/n_selected"))
            if soft is not None:
                value = soft["mean"]
                if value >= 14.0:
                    reading = "near-uniform — barely selecting"
                elif value <= 6.0:
                    reading = "strongly concentrated"
                else:
                    reading = "intermediate"
            elif hard is not None:
                value = hard["mean"]
                reading = "hard top-k (ESS = k by construction)"
            else:
                continue
            lines.append(
                f"| {setting} | {method} | {value:.2f} | {reading} |"
            )

    lines += [
        "",
        "## Two caveats that change how these numbers read",
        "",
        "1. **Layer-wise methods do not discard the other 8.** Each layer picks its",
        "   own subset, so across the network almost every candidate contributes to",
        "   *some* layer's update. `selection_rate` in `domain_lift` is therefore a",
        "   mean keep-fraction over layers, not a global keep/drop decision. Only",
        "   the Global methods drop a candidate everywhere at once.",
        "2. **`lift` is already normalised** by each run's own overall selection",
        "   rate, so `LayerwiseSoftP` (rate 0.0625, mass 1) is directly comparable",
        "   to the hard methods (rate 0.5). Do not renormalise it.",
        "",
        "## Per-setting targets",
        "",
        "| setting | target |",
        "|---|---|",
    ]
    for setting in settings:
        lines.append(f"| {setting} | {targets.get(setting, '?')} |")

    lines += [
        "",
        "## What these runs cannot tell you",
        "",
        "`selection_diagnostics.json` stores layer-**averaged** values",
        "(`SelectionState.get_diagnostic_metrics` means over layers). Current",
        "Dolci defaults additionally write `selection_records.json` every 100",
        "steps, including candidate identity plus each layer's indices or weights.",
        "Older runs without that artifact still require a re-run or the separate",
        "`layer_alignment_probe` for exact layer/sample disagreement.",
    ]
    (out_dir / "data_budget.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN)
    parser.add_argument(
        "--optimizer", default="adamw", choices=("adamw", "muon")
    )
    parser.add_argument("--out", default=None, help="output root (default: analysis/results)")
    args = parser.parse_args(argv)

    runs = discover_runs(args.campaign, optimizer=args.optimizer)
    runs_by_key = index_runs(runs)
    settings = list(setting_order())
    methods = _selection_methods_for_optimizer(args.optimizer)
    out_dir = (
        results_root(args.campaign, Path(args.out) if args.out else None)
        / "selection_profile"
        / args.optimizer
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    build_weighting_contract(out_dir / "weighting_contract", args.optimizer)
    missing_cells = build_campaign_completeness(
        runs_by_key, settings, methods, out_dir
    )
    records = build_weight_realization(runs_by_key, settings, out_dir, methods)
    payloads = build_domain_lift(runs_by_key, settings, out_dir, methods)
    build_data_budget(runs_by_key, settings, out_dir, records, methods)
    plot_weight_concentration(runs_by_key, settings, out_dir)
    plot_domain_lift(payloads, settings, out_dir, methods)

    present = sorted({key[1] for key in runs_by_key})
    print(f"campaign        : {args.campaign}")
    print(f"optimizer       : {args.optimizer}")
    print(f"settings        : {', '.join(settings)}")
    print(f"methods present : {', '.join(present)}")
    print(
        f"selection cells : {len(settings) * len(methods) - len(missing_cells)} "
        f"of {len(settings) * len(methods)} complete"
    )
    if missing_cells:
        print("missing cells   : " + ", ".join(f"{s}/{m}" for s, m in missing_cells))
    print(f"wrote           : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
