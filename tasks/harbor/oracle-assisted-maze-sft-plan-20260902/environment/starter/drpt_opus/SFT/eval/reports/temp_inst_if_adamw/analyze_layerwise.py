#!/usr/bin/env python3
"""Is layer-wise selection actually using its per-layer freedom?

Everything here is recovered from what the completed dolci32k AdamW runs already
wrote to disk. Nothing is re-trained.

THE TRICK THAT MAKES THIS POSSIBLE
----------------------------------
``selection_domain_summary.json`` stores, for each tracked step, the candidate
count and the "selected mass" per domain. That mass is each example's selection
weight *averaged over layers*, summed over the examples of that domain
(Trainer._per_example_selection_weights). So whenever a domain contributes
exactly ONE candidate in a step, its selected mass is precisely that single
example's mean-over-layers weight, which we call w.

  w = 0.0  -> no layer kept this example
  w = 1.0  -> every layer kept it
  w = 0.5  -> half the layers kept it

READING w DEPENDS ON THE METHOD
-------------------------------
* Hard top-k (LayerwiseRaw, LayerwiseOptA): each layer votes 0/1, so if all
  layers agreed every w would be exactly 0 or 1. Any w in between is proof of
  layer disagreement.
* Soft weighting (LayerwiseSoft, LayerwiseSoftP): a single layer already emits
  continuous weights, so a fractional w proves nothing on its own. Instead we
  use the run's own logged ``soft/boundary_fraction`` -- the share of weights
  that sit at 0 or 1 *within one layer*. If every layer produced the same
  weights, that same share of w values would land on the boundary. Observing
  far fewer means the layers disagreed.

Both cases reduce to one comparison: observed boundary share of w
vs. the boundary share expected if all layers behaved identically.

Writes layerwise_report.md and layerwise_analysis.png next to this file.
"""

from __future__ import annotations

import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CAMPAIGN = Path(
    "/home/nakyungl/drpt-next/Dr.Post-Training-Next/drpt_opus/SFT/runs/campaigns/"
    "dolci32k-qwen3_1_7b-s42"
)

SETTINGS = ("inst_if", "reason_math", "reason_code", "mixed_if", "mixed_math")
METHODS = ("LayerwiseRaw", "LayerwiseOptA", "LayerwiseSoft", "LayerwiseSoftP")
HARD = {"LayerwiseRaw", "LayerwiseOptA"}
GROUPS = ("embedding", "attention", "mlp", "lm_head")
BOUNDARY_EPS = 0.02
N_UNITS = 198  # trainable layer-units the hook wraps; used only for the random null

TARGET_DOMAIN = {
    "inst_if": "precise_if", "mixed_if": "precise_if",
    "reason_math": "math", "mixed_math": "math", "reason_code": "mbpp",
}

COLORS = {
    "LayerwiseRaw": "#eb6834",
    "LayerwiseOptA": "#e87ba4",
    "LayerwiseSoft": "#1baf7a",
    "LayerwiseSoftP": "#eda100",
}
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def run_dir(setting, method):
    hits = glob.glob(str(CAMPAIGN / f"{setting}-{method}-adamw-*"))
    return Path(hits[0]) if hits else None


def load(setting, method, name):
    d = run_dir(setting, method)
    if d is None or not (d / name).exists():
        return None
    return json.loads((d / name).read_text())


def singleton_weights(setting, method):
    payload = load(setting, method, "selection_domain_summary.json")
    if not payload:
        return []
    out = []
    for step in payload.get("timeline", []):
        for domain, count in step.get("candidates", {}).items():
            if count == 1.0:
                out.append(float(step.get("selected", {}).get(domain, 0.0)))
    return out


def within_layer_boundary_share(setting, method):
    """Share of weights at 0/1 inside a single layer, from the run's own log."""
    if method in HARD:
        return 1.0  # a 0/1 vote is boundary by construction
    recs = load(setting, method, "selection_diagnostics.json")
    if not recs:
        return None
    vals = [r["soft/boundary_fraction"] for r in recs if "soft/boundary_fraction" in r]
    return statistics.median(vals) if vals else None


def group_margins(setting, method):
    recs = load(setting, method, "selection_diagnostics.json")
    if not recs:
        return {}
    acc = defaultdict(list)
    for rec in recs:
        for g in GROUPS:
            key = f"update/adamw_{g}/selected_score_margin"
            if isinstance(rec.get(key), (int, float)):
                acc[g].append(float(rec[key]))
    return {g: statistics.median(v) for g, v in acc.items() if v}


def domain_lift(setting, method):
    payload = load(setting, method, "selection_domain_summary.json")
    if not payload:
        return {}
    return {
        k: v.get("lift")
        for k, v in payload.get("by_domain", {}).items()
        if isinstance(v, dict) and v.get("lift") is not None
    }


def main() -> int:
    L: list[str] = []
    W = L.append

    W("# Is layer-wise selection actually layer-wise?\n")
    W("Recovered from the 25 completed dolci32k AdamW runs. Nothing re-trained.\n")

    # ---------- Q1 ----------
    W("## Q1. Did different layers choose different samples?\n")
    W("**w** = the fraction of layers that kept a given candidate "
      "(1.0 = every layer kept it, 0.0 = none did).\n")
    W("**How to read the last three columns.** `boundary(obs)` is the share of w "
      "values sitting at 0 or 1. `boundary(if identical)` is what that share "
      "*would* be if every layer behaved the same way -- 100% for the hard "
      "top-k methods (a 0/1 vote is always at the boundary), and the run's own "
      "logged within-layer `soft/boundary_fraction` for the soft methods. "
      "**A large gap between the two means the layers disagreed.**\n")
    W("| setting | method | n | median w | SD of w | boundary(obs) | boundary(if identical) | gap |")
    W("|---|---|---:|---:|---:|---:|---:|---:|")

    q1 = {}
    for setting in SETTINGS:
        for method in METHODS:
            ws = singleton_weights(setting, method)
            if len(ws) < 20:
                continue
            exp = within_layer_boundary_share(setting, method)
            obs = sum(1 for v in ws if v < BOUNDARY_EPS or v > 1 - BOUNDARY_EPS) / len(ws)
            q1[(setting, method)] = ws
            gap = (exp - obs) if exp is not None else None
            W(f"| {setting} | {method} | {len(ws)} | {statistics.median(ws):.3f} "
              f"| {statistics.pstdev(ws):.3f} | {obs*100:.1f}% "
              f"| {exp*100:.1f}% | {gap*100:+.1f}pp |")
    W("")
    W(f"For reference, if layers picked **independently at random** the SD of w "
      f"would be about {0.5/ N_UNITS**0.5:.3f} (Binomial(U≈{N_UNITS}, 0.5)/U) -- "
      f"i.e. essentially every w crowded onto 0.5.\n")

    # ---------- Q2a ----------
    W("## Q2a. Do the architecture groups differ?\n")
    W("Median selected-vs-unselected **score margin** over all 2000 steps. "
      "Bigger = that part of the network separates good from bad candidates "
      "more sharply. Near zero = it barely distinguishes them.\n")
    W("Only the hard top-k methods log this; the soft methods log solver "
      "diagnostics instead.\n")
    W("| setting | method | " + " | ".join(GROUPS) + " | lm_head ÷ attention |")
    W("|---|---|" + "---:|" * (len(GROUPS) + 1))
    for setting in SETTINGS:
        for method in ("LayerwiseRaw", "LayerwiseOptA"):
            gm = group_margins(setting, method)
            if not gm:
                continue
            cells = [f"{gm[g]:.2e}" if g in gm else "-" for g in GROUPS]
            ratio = (gm["lm_head"] / gm["attention"]
                     if gm.get("attention") else None)
            cells.append(f"{ratio:.0f}x" if ratio else "-")
            W(f"| {setting} | {method} | " + " | ".join(cells) + " |")
    W("")

    # ---------- Q2b ----------
    W("## Q2b. Does each task pull a different data mix?\n")
    W("`lift` of the **target domain** = how much more often that domain is "
      "kept than its share of the candidate pool. 1.00 = no preference at all; "
      ">1 = the method steers toward the target.\n")
    W("| setting | target domain | " + " | ".join(METHODS) + " |")
    W("|---|---|" + "---:|" * len(METHODS))
    for setting in SETTINGS:
        tgt = TARGET_DOMAIN[setting]
        cells = []
        for method in METHODS:
            lifts = domain_lift(setting, method)
            v = lifts.get(tgt)
            cells.append(f"{v:.2f}" if v is not None else "not in pool")
        W(f"| {setting} | `{tgt}` | " + " | ".join(cells) + " |")
    W("")
    W("Where a cell says *not in pool*, that domain label does not exist in that "
      "setting's candidate pool at all, so no steering toward it is even possible.\n")

    # ---------- figure ----------
    fig, axes = plt.subplots(len(SETTINGS), len(METHODS),
                             figsize=(4.0 * len(METHODS), 2.3 * len(SETTINGS)),
                             facecolor=SURFACE, squeeze=False)
    for r, setting in enumerate(SETTINGS):
        for c, method in enumerate(METHODS):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            ws = q1.get((setting, method))
            if not ws:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                        va="center", color=MUTED, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                continue
            ax.hist(ws, bins=24, range=(0, 1), color=COLORS[method],
                    edgecolor=SURFACE, linewidth=0.7)
            ax.axvline(0.5, color=MUTED, linestyle=":", linewidth=1.1)
            ax.set_xlim(0, 1)
            if r == 0:
                ax.set_title(method, fontsize=10.5, color=INK, loc="left", pad=8)
            if c == 0:
                ax.set_ylabel(setting, fontsize=10, color=INK)
            if r == len(SETTINGS) - 1:
                ax.set_xlabel("w  (0 = no layer kept it, 1 = all layers did)",
                              fontsize=8, color=MUTED)
            ax.grid(True, color=GRID, linewidth=0.6, axis="y")
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(AXIS)
            ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle("How much do layers agree on which candidates to keep?",
                 fontsize=13.5, color=INK, x=0.006, ha="left", y=0.998)
    fig.text(0.006, 0.978,
             "Mass piled at 0 and 1 would mean the layers agree. Mass piled tightly "
             "on 0.5 (dotted) would mean they choose independently at random. "
             "A broad hump in between is structured disagreement.",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.968))
    png = HERE / "layerwise_analysis.png"
    fig.savefig(png, dpi=150, facecolor=SURFACE)
    W("![layer agreement](layerwise_analysis.png)\n")

    W("## What is NOT recoverable from these runs\n")
    W("Per-*individual*-layer sample identity. The campaign ran without "
      "`--record_selections`, so `selection_records.json` -- which would store "
      "each layer's own selected indices -- exists in none of the 25 runs. "
      "Statements like \"layer 3 preferred domain X, layer 27 preferred Y\" "
      "cannot be made; only the 4 optimizer-group buckets in Q2a. Re-running one "
      "setting with `--record_selections` would close that gap.\n")

    (HERE / "layerwise_report.md").write_text("\n".join(L) + "\n")
    print(f"wrote {HERE/'layerwise_report.md'}")
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
