#!/usr/bin/env python3
"""Zoomed AdamW comparison across all 5 dolci32k settings.

``plot_loss_curves.py`` already writes the campaign grids, but it puts every
setting on one shared y-scale per figure. On dolci32k the non-SoftP methods
finish within ~0.001 of each other, so at that scale four of the five curves
overlap into a single line and the ordering is unreadable.

This re-plots the same points (from the ``loss_curve_points.csv.gz`` that
plot_loss_curves.py wrote -- nothing is recomputed) with each panel autoscaled
to its own converged band, plus an explicit final-value ranking per panel.

Usage:  python plot_adamw_grid.py
"""

from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
POINTS = HERE / "loss_curve_points.csv.gz"

# Registry order from SFT/data/dolci32k/profile.py.
SETTINGS = ("inst_if", "reason_math", "reason_code", "mixed_if", "mixed_math")
METHODS = ("FullTraining", "LayerwiseRaw", "LayerwiseSoft", "LayerwiseSoftP", "LayerwiseOptA")

# Validated categorical slots 1-5 (light surface). Assignment is by the fixed
# registry order above so a method keeps its colour in every panel.
COLORS = {
    "FullTraining": "#2a78d6",
    "LayerwiseRaw": "#eb6834",
    "LayerwiseSoft": "#1baf7a",
    "LayerwiseSoftP": "#eda100",
    "LayerwiseOptA": "#e87ba4",
}
# Secondary encoding: three of these hues WARN on contrast against a light
# surface, and the yellow/aqua adjacent pair sits in the CVD relief band.
MARKERS = {
    "FullTraining": "o",
    "LayerwiseRaw": "s",
    "LayerwiseSoft": "^",
    "LayerwiseSoftP": "D",
    "LayerwiseOptA": "v",
}

INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

# (metric, column title, whether the curve is dense enough to need marker thinning)
COLUMNS = (
    ("target_val_loss", "Target val loss", False),
    ("general_eval_loss", "General val loss", False),
    ("train_loss", "Train loss (smoothed)", True),
)


def load():
    series = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with gzip.open(POINTS, "rt") as handle:
        for row in csv.DictReader(handle):
            if row["optimizer"] != "adamw":
                continue
            # train_loss is logged every step and is noisy; the writer already
            # stored a centred rolling mean next to the raw value.
            value = row["smoothed_value"] if row["metric"] == "train_loss" else row["value"]
            if value in ("", None):
                continue
            series[row["setting"]][row["metric"]][row["method"]].append(
                (int(row["step"]), float(value))
            )
    for setting in series:
        for metric in series[setting]:
            for method in series[setting][metric]:
                series[setting][metric][method].sort()
    return series


def main() -> int:
    series = load()
    nrow, ncol = len(SETTINGS), len(COLUMNS)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(5.6 * ncol, 3.5 * nrow), facecolor=SURFACE
    )

    for r, setting in enumerate(SETTINGS):
        for c, (metric, coltitle, dense) in enumerate(COLUMNS):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            data = series.get(setting, {}).get(metric, {})
            present = [m for m in METHODS if m in data]
            if not present:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color=MUTED)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                continue

            # Autoscale to the converged tail: step 0 sits ~0.2-0.3 above it and
            # would flatten the whole panel back into one line.
            tail = [v for m in present for s, v in data[m] if s >= 400]
            if tail:
                lo, hi = min(tail), max(tail)
                pad = max((hi - lo) * 0.18, 0.002)
                ax.set_ylim(lo - pad, hi + pad)

            for method in present:
                steps = [s for s, _ in data[method]]
                vals = [v for _, v in data[method]]
                ax.plot(
                    steps, vals,
                    color=COLORS[method],
                    marker=MARKERS[method],
                    markersize=4.5,
                    markevery=max(1, len(steps) // 18) if dense else 1,
                    linewidth=1.8,
                    markeredgecolor=SURFACE,
                    markeredgewidth=0.8,
                    label=method,
                    zorder=3,
                )

            # Ranked final values. The four non-SoftP methods land within ~0.001
            # of each other, so per-point labels would collide; this also serves
            # as the relief the palette's contrast WARN requires.
            finals = sorted(((data[m][-1][1], m) for m in present), key=lambda p: p[0])
            ax.text(0.55, 0.965, "final (best first)", transform=ax.transAxes,
                    fontsize=7.5, color=MUTED, ha="left", va="top")
            for i, (value, method) in enumerate(finals):
                y = 0.90 - i * 0.072
                ax.plot([0.575], [y + 0.014], transform=ax.transAxes,
                        marker=MARKERS[method], markersize=4.5, color=COLORS[method],
                        markeredgecolor=SURFACE, markeredgewidth=0.8, clip_on=False)
                ax.text(0.605, y, method, transform=ax.transAxes,
                        fontsize=7.5, color=INK, ha="left", va="center")
                ax.text(0.995, y, f"{value:.4f}", transform=ax.transAxes,
                        fontsize=7.5, color=INK, ha="right", va="center")

            if r == 0:
                ax.set_title(coltitle, fontsize=11.5, color=INK, pad=10, loc="left")
            if c == 0:
                ax.set_ylabel(setting, fontsize=11, color=INK, labelpad=8)
            if r == nrow - 1:
                ax.set_xlabel("Training step", fontsize=9, color=MUTED)
            ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(AXIS)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.set_xlim(120, 2120)

    handles, labels = axes[0][0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                        frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.004))
    for t in legend.get_texts():
        t.set_color(INK)

    fig.suptitle(
        "dolci32k · AdamW · Qwen3-1.7B-Base · seed 42 — all 25 runs complete",
        fontsize=15, color=INK, x=0.008, ha="left", y=0.995,
    )
    fig.text(
        0.008, 0.977,
        "Each panel autoscaled to its own post-step-400 band; step 0 is off-scale. "
        "Train loss is a centred rolling mean and is NOT comparable across methods "
        "(selection runs return a per-example-weighted loss, FullTraining a plain mean).",
        fontsize=8.5, color=MUTED, ha="left",
    )
    fig.tight_layout(rect=(0, 0.028, 1, 0.968))
    out = HERE / "adamw_all_settings_zoom.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
