#!/usr/bin/env python3
"""Zoomed inst_if/AdamW comparison.

The campaign-wide grid from ``plot_loss_curves.py`` puts every setting on one
shared y-scale, where the four non-SoftP methods sit within ~0.001 of each other
and read as a single line. This script re-plots only the completed inst_if AdamW
runs, autoscaled to the converged band, so the method ordering is actually
visible. Input is the ``loss_curve_points.csv.gz`` that plot_loss_curves.py
already wrote -- nothing is recomputed here.
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

# Registry order from SFT/data/dolci32k/profile.py::ADAMW_METHODS. Hues are the
# validated categorical slots 1-5; assignment is by this fixed order so a method
# keeps its colour no matter which runs happen to be finished.
METHODS = ("FullTraining", "LayerwiseRaw", "LayerwiseSoft", "LayerwiseSoftP", "LayerwiseOptA")
COLORS = {
    "FullTraining": "#2a78d6",
    "LayerwiseRaw": "#eb6834",
    "LayerwiseSoft": "#1baf7a",
    "LayerwiseSoftP": "#eda100",
    "LayerwiseOptA": "#e87ba4",
}
# Secondary encoding: the contrast check WARNs on three of these hues against a
# light surface, and CVD separation for yellow/aqua sits in the 6-8 relief band.
MARKERS = {
    "FullTraining": "o",
    "LayerwiseRaw": "s",
    "LayerwiseSoft": "^",
    "LayerwiseSoftP": "D",
    "LayerwiseOptA": "v",
}

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

PANELS = (
    ("target_val_loss", "Target validation loss", "precise_if, 128 held-out examples (never used for selection)"),
    ("general_eval_loss", "General validation loss", "instruction_32k, 512 held-out examples"),
)


def load(setting: str = "inst_if", optimizer: str = "adamw"):
    series = defaultdict(lambda: defaultdict(list))
    with gzip.open(POINTS, "rt") as handle:
        for row in csv.DictReader(handle):
            if row["setting"] != setting or row["optimizer"] != optimizer:
                continue
            series[row["metric"]][row["method"]].append(
                (int(row["step"]), float(row["value"]))
            )
    for metric in series:
        for method in series[metric]:
            series[metric][method].sort()
    return series


def main() -> int:
    series = load()
    fig, axes = plt.subplots(
        1, len(PANELS), figsize=(13.5, 5.4), facecolor=SURFACE
    )

    for ax, (metric, title, subtitle) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        present = [m for m in METHODS if m in series.get(metric, {})]

        # Autoscale to the converged tail so the method spread is legible; the
        # step-0 point is ~0.2 above it and would flatten everything again.
        tail = [
            value
            for method in present
            for step, value in series[metric][method]
            if step >= 400
        ]
        if tail:
            lo, hi = min(tail), max(tail)
            pad = max((hi - lo) * 0.18, 0.002)
            ax.set_ylim(lo - pad, hi + pad)

        for method in present:
            points = series[metric][method]
            steps = [s for s, _ in points]
            values = [v for _, v in points]
            ax.plot(
                steps,
                values,
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=5.5,
                linewidth=2.0,
                markeredgecolor=SURFACE,
                markeredgewidth=0.9,
                label=method,
                zorder=3,
            )

        # The four non-SoftP methods finish within ~0.001 of each other, so a
        # per-point direct label collides into an unreadable blob. A ranked
        # block gives the same relief the contrast WARN requires, and makes the
        # ordering explicit rather than something to squint at.
        finals = sorted(
            ((series[metric][m][-1][1], m) for m in present), key=lambda p: p[0]
        )
        ax.text(
            0.62, 0.955, "final (best first)", transform=ax.transAxes,
            fontsize=8.5, color=MUTED, ha="left", va="top",
        )
        for row, (value, method) in enumerate(finals):
            y = 0.90 - row * 0.062
            ax.plot(
                [0.635], [y + 0.012], transform=ax.transAxes,
                marker=MARKERS[method], markersize=5.5, color=COLORS[method],
                markeredgecolor=SURFACE, markeredgewidth=0.9, clip_on=False,
            )
            ax.text(
                0.665, y, method, transform=ax.transAxes,
                fontsize=8.5, color=INK, ha="left", va="center",
            )
            ax.text(
                0.995, y, f"{value:.4f}", transform=ax.transAxes,
                fontsize=8.5, color=INK, ha="right", va="center",
            )

        ax.set_title(title, fontsize=12.5, color=INK, pad=14, loc="left")
        ax.text(
            0.0, 1.015, subtitle, transform=ax.transAxes,
            fontsize=8.8, color=MUTED, ha="left", va="bottom",
        )
        ax.set_xlabel("Training step", fontsize=9.5, color=MUTED)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xlim(120, 2120)

    axes[0].set_ylabel("Loss", fontsize=9.5, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels,
        loc="lower center", ncol=len(labels), frameon=False,
        fontsize=9.5, bbox_to_anchor=(0.5, -0.005),
    )
    for text in legend.get_texts():
        text.set_color(INK)

    fig.suptitle(
        "inst_if · AdamW · Qwen3-1.7B-Base · seed 42 — converged band",
        fontsize=14, color=INK, x=0.012, ha="left", y=0.985,
    )
    fig.text(
        0.012, 0.925,
        "All 5 methods complete (2000 steps). y-axis is zoomed to the post-step-400 range; "
        "step 0 (~1.50 / ~1.65) is off-scale.",
        fontsize=9, color=MUTED, ha="left",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    out = HERE / "inst_if_adamw_zoom.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
