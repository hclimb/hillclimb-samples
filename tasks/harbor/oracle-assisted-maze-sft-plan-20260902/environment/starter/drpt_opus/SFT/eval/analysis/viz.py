"""Shared figure parameters for the campaign analyses.

One fixed hue per method on the shared AdamW axis, assigned in registry order
and never cycled. Muon-only surrogate comparisons use the labelled diverging
heatmap rather than categorical hues. Palette values and the diverging pair are
the validated defaults; the categorical subset used here was checked with the
six-check validator (light surface):

    Lightness band  PASS | Chroma floor PASS
    CVD separation  PASS (worst adjacent dE 9.1)
    Normal-vision   PASS (worst adjacent dE 19.6)
    Contrast        WARN -> relief is satisfied because every figure ships the
                    same numbers as a CSV/Markdown table beside it, and the
                    heatmaps print the value inside each cell.
"""

from __future__ import annotations

from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .campaign_io import METHOD_ORDER

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8880"
GRID = "#e3e2dd"

# Categorical slots, fixed order.
CATEGORICAL: Sequence[str] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

METHOD_COLORS: Dict[str, str] = {
    label: CATEGORICAL[index] for index, label in enumerate(METHOD_ORDER)
}

# Diverging pair for signed quantities (lift around 1, delta around 0):
# blue <-> red with a neutral gray midpoint. Never a hue at the midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "drpt_diverging",
    ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#8f2222"],
)


def diverging_norm(values, center: float):
    """Symmetric diverging norm so equal deviations get equal colour weight."""
    finite = [v for v in values if v == v]
    if not finite:
        return TwoSlopeNorm(vcenter=center, vmin=center - 1, vmax=center + 1)
    span = max(abs(max(finite) - center), abs(center - min(finite)), 1e-9)
    return TwoSlopeNorm(vcenter=center, vmin=center - span, vmax=center + span)


def apply_style() -> None:
    """Recessive grid and axes; text in ink tokens, never in a series colour."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.size": 9,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "figure.dpi": 130,
    })


def strip_spines(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
