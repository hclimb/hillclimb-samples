#!/usr/bin/env python3
"""Which *source datasets* does curation actually prefer, and does length explain it?

The per-domain lift table is coarse: dolci32k's `math` label covers both MATH-style
symbolic problems and Tulu persona word problems, so a domain lift below 1 does not
by itself mean the method ignored target-relevant data.

This drops to source-dataset granularity for every setting, and pairs each source's
lift with the mean assistant-response length of that source and of the setting's own
target split. If lift tracked topic we would expect the target's own domain on top;
if it tracked surface form we would expect lift to fall off with distance from the
target's response length. Printing both lets the reader decide instead of guessing
from a couple of hand-picked rows.

Writes source_lift_report.md and source_lift.png next to this file.
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
ROOT = Path("/home/nakyungl/drpt-next/Dr.Post-Training-Next/drpt_opus")
CAMPAIGN = ROOT / "SFT/runs/campaigns/dolci32k-qwen3_1_7b-s42"
BUILD = (CAMPAIGN / "dolci32k_artifact_build_id.txt").read_text().strip()
ARTIFACT = ROOT / "SFT/data/dolci32k_artifacts/builds" / BUILD

SETTINGS = {
    "inst_if": ("instruction_32k", "precise_if"),
    "reason_math": ("reasoning_32k", "math"),
    "reason_code": ("reasoning_32k", "mbpp"),
    "mixed_if": ("mixed_32k", "precise_if"),
    "mixed_math": ("mixed_32k", "math"),
}
METHODS = ("LayerwiseRaw", "LayerwiseOptA", "LayerwiseSoft", "LayerwiseSoftP")

INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLORS = {
    "LayerwiseRaw": "#eb6834",
    "LayerwiseOptA": "#e87ba4",
    "LayerwiseSoft": "#1baf7a",
    "LayerwiseSoftP": "#eda100",
}


def assistant_len(record) -> int:
    return sum(len(m["content"]) for m in record["messages"] if m["role"] == "assistant")


def pool_stats(pool: str):
    """Mean assistant length and domain label for every source in a pool."""
    lens = defaultdict(list)
    domain = {}
    with (ARTIFACT / "general" / pool / "train.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            src = r.get("source_dataset")
            lens[src].append(assistant_len(r))
            domain.setdefault(src, r.get("domain"))
    return ({s: statistics.mean(v) for s, v in lens.items()}, domain)


def target_len(target: str) -> float:
    path = ARTIFACT / "targets" / target / "grad.jsonl"
    return statistics.mean(assistant_len(json.loads(l)) for l in path.open())


def source_lift(setting: str, method: str):
    hits = glob.glob(str(CAMPAIGN / f"{setting}-{method}-adamw-*" / "selection_domain_summary.json"))
    if not hits:
        return {}
    payload = json.loads(Path(hits[0]).read_text())
    return {
        k: v["lift"]
        for k, v in payload.get("by_source_dataset", {}).items()
        if isinstance(v, dict) and v.get("lift") is not None
    }


def main() -> int:
    L: list[str] = []
    W = L.append
    W("# Source-level selection lift, all five settings\n")
    W("`lift` = P(source | selected) / P(source | candidate). 1.00 means no preference.\n")
    W("`len` is the mean assistant-response length in characters. `D* len` on each")
    W("section header is the same statistic for that setting's target split, so a")
    W("source's distance from it is readable at a glance.\n")

    pools = {}
    scatter = defaultdict(list)
    for setting, (pool, target) in SETTINGS.items():
        if pool not in pools:
            pools[pool] = pool_stats(pool)
        lens, domains = pools[pool]
        tlen = target_len(target)

        lifts = {m: source_lift(setting, m) for m in METHODS}
        soft = lifts.get("LayerwiseSoft", {})
        if not soft:
            continue
        order = sorted(soft, key=lambda s: -soft[s])

        W(f"\n## {setting}  (target `{target}`, D* len {tlen:.0f} chars)\n")
        W("| source | domain | len | " + " | ".join(m.replace("Layerwise", "") for m in METHODS) + " |")
        W("|---|---|---:|" + "---:|" * len(METHODS))
        for src in order:
            cells = []
            for m in METHODS:
                v = lifts.get(m, {}).get(src)
                cells.append(f"{v:.3f}" if v is not None else "-")
            mark = " **←target domain**" if domains.get(src) == target else ""
            W(f"| {src}{mark} | `{domains.get(src)}` | {lens.get(src, 0):.0f} | "
              + " | ".join(cells) + " |")
            for m in METHODS:
                v = lifts.get(m, {}).get(src)
                if v is not None and src in lens:
                    scatter[m].append((abs(lens[src] - tlen), v, setting))

    # Does surface length explain lift?
    W("\n## Does response length explain the ordering?\n")
    W("Pearson r between a source's |len - D* len| and its lift, pooled over settings.")
    W("A strong negative r would mean curation is largely tracking response length")
    W("rather than content.\n")
    W("| method | n | r |")
    W("|---|---:|---:|")
    for m in METHODS:
        pts = scatter.get(m, [])
        if len(pts) < 4:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
        r = num / den if den else float("nan")
        W(f"| {m} | {len(pts)} | {r:+.3f} |")

    fig, axes = plt.subplots(1, len(METHODS), figsize=(4.2 * len(METHODS), 3.6),
                             facecolor=SURFACE, squeeze=False)
    for ax, m in zip(axes[0], METHODS):
        ax.set_facecolor(SURFACE)
        pts = scatter.get(m, [])
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=26,
                       color=COLORS[m], edgecolor=SURFACE, linewidth=0.7, zorder=3)
        ax.axhline(1.0, color=MUTED, linestyle=":", linewidth=1.1)
        ax.set_title(m.replace("Layerwise", ""), fontsize=11, color=INK, loc="left", pad=8)
        ax.set_xlabel("|source len − D* len|  (chars)", fontsize=8.5, color=MUTED)
        ax.set_ylabel("lift", fontsize=9, color=MUTED)
        ax.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle("Source lift vs distance from the target's response length",
                 fontsize=13, color=INK, x=0.006, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    png = HERE / "source_lift.png"
    fig.savefig(png, dpi=150, facecolor=SURFACE)
    W(f"\n![source lift](source_lift.png)\n")

    (HERE / "source_lift_report.md").write_text("\n".join(L) + "\n")
    print(f"wrote {HERE/'source_lift_report.md'}")
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
