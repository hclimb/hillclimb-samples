#!/usr/bin/env python
"""Per-domain target alignment as an absolute quantity, not a selection ratio.

`domain_lift` answers "how many times its proportional share did this domain
get selected". That is a ratio, it is attenuated toward 1 by layer averaging,
and it says nothing about how aligned the data actually is.

This report instead reads the probe's per-candidate records and reports, per
domain, the quantity the selector ranks on and its scale-free form:

  cos_raw   cos(grad_i, grad_target)          in [-1, 1], absolute
  cos_opt   the same in AdamW geometry        the scale-free OptA score
  dot_raw   <grad_i, grad_target>             the raw score, unnormalised
  loss      the example's own loss            tests "already fit -> small grad"
  P(cos>0)  share of the domain that helps at all

A domain can be semantically close to the target and still score low if the
model already fits it: low loss gives a small gradient and hence a small
alignment score regardless of direction. Reporting loss beside cosine is what
separates those two explanations.

Usage:
  python -m SFT.eval.analysis.domain_alignment_report [--split val]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from SFT.eval.analysis.campaign_io import (
    DEFAULT_CAMPAIGN,
    discover_runs,
    domain_summary,
    fmt,
    index_runs,
    results_root,
    setting_order,
    setting_targets,
    write_table,
)


def mean(values: Sequence[float]) -> float:
    finite = [v for v in values if v == v]
    return sum(finite) / len(finite) if finite else float("nan")


def stdev(values: Sequence[float]) -> float:
    finite = [v for v in values if v == v]
    if len(finite) < 2:
        return float("nan")
    m = sum(finite) / len(finite)
    return math.sqrt(sum((v - m) ** 2 for v in finite) / (len(finite) - 1))


def load_probes(out_dir: Path, split: str) -> Dict[str, dict]:
    payloads: Dict[str, dict] = {}
    for path in sorted(out_dir.glob(f"*_{split}_probe.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("candidates"):
            payloads[payload["setting"]] = payload
    return payloads


def by_domain(candidates: Sequence[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for row in candidates:
        grouped.setdefault(row["domain"], []).append(row)
    return grouped


def build_tables(payloads, settings, campaign, out_dir: Path, split: str):
    runs = index_runs(discover_runs(campaign, optimizer="adamw"))
    targets = setting_targets()

    header = [
        "setting", "target", "domain", "n",
        "cos_raw", "cos_opt", "P(cos>0)", "mean loss", "grad norm",
        "dot_raw", "lift (LayerwiseRaw)",
    ]
    rows: List[List[object]] = []
    summaries: Dict[str, Dict[str, dict]] = {}

    for setting in settings:
        payload = payloads[setting]
        grouped = by_domain(payload["candidates"])
        run = runs.get((setting, "LayerwiseRaw"))
        lift_table = (domain_summary(run) or {}).get("by_domain", {}) if run else {}
        summaries[setting] = {}
        for domain, entries in sorted(
            grouped.items(), key=lambda kv: -mean([e["cos_raw"] for e in kv[1]])
        ):
            cos_raw = [e["cos_raw"] for e in entries]
            record = {
                "n": len(entries),
                "cos_raw": mean(cos_raw),
                "cos_raw_sd": stdev(cos_raw),
                "cos_opt": mean([e["cos_opt"] for e in entries]),
                "positive": mean([1.0 if e["cos_raw"] > 0 else 0.0 for e in entries]),
                "loss": mean([e["loss"] for e in entries]),
                "grad_norm": mean([e["grad_norm"] for e in entries]),
                "dot_raw": mean([e["dot_raw"] for e in entries]),
                "lift": lift_table.get(domain, {}).get("lift"),
            }
            summaries[setting][domain] = record
            rows.append([
                setting, targets.get(setting, "?"), domain, record["n"],
                fmt(record["cos_raw"], 5), fmt(record["cos_opt"], 5),
                fmt(record["positive"], 3), fmt(record["loss"], 4),
                fmt(record["grad_norm"], 3), fmt(record["dot_raw"], 5),
                fmt(record["lift"], 3),
            ])
    write_table(out_dir / f"domain_alignment_{split}", header, rows)
    return summaries


def correlations(summaries) -> List[List[object]]:
    """Does alignment track the lift, and does loss explain alignment?"""
    def pearson(a, b):
        pairs = [(x, y) for x, y in zip(a, b) if x == x and y == y]
        if len(pairs) < 3:
            return float("nan")
        xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        dy = math.sqrt(sum((y - my) ** 2 for y in ys))
        return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")

    rows: List[List[object]] = []
    for setting, table in summaries.items():
        names = sorted(table)
        cos = [table[d]["cos_raw"] for d in names]
        rows.append([
            setting, len(names),
            fmt(pearson(cos, [table[d]["lift"] for d in names]), 3),
            fmt(pearson(cos, [table[d]["loss"] for d in names]), 3),
            fmt(pearson([table[d]["dot_raw"] for d in names],
                        [table[d]["lift"] for d in names]), 3),
        ])
    return rows


def build_report(summaries, settings, out_dir: Path, split: str, corr) -> None:
    targets = setting_targets()
    lines = [
        f"# Per-domain target alignment ({split} split)",
        "",
        "Absolute alignment per domain, measured at the base checkpoint from the",
        "probe's per-candidate gradients. Unlike `domain_lift`, nothing here is a",
        "ratio against a proportional share, and nothing is averaged over layers.",
        "",
        "- `cos_raw` = cos(grad_i, grad_target) over the whole model, in [-1, 1]",
        "- `cos_opt` = the same in AdamW geometry — the scale-free OptA score",
        "- `dot_raw` = <grad_i, grad_target>, the unnormalised score the hard rule ranks on",
        "- `mean loss` = the example's own loss; a low-loss domain gives small",
        "  gradients and therefore a small `dot_raw` even when `cos_raw` is high",
        "",
        "## Target-domain alignment vs the rest",
        "",
    ]
    for setting in settings:
        table = summaries[setting]
        target = targets.get(setting, "?")
        ordered = sorted(table, key=lambda d: -table[d]["cos_raw"])
        lines += [
            f"### {setting} (target: {target})",
            "",
            "| domain | n | cos_raw | cos_opt | P(cos>0) | mean loss | dot_raw | lift |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for domain in ordered:
            r = table[domain]
            lines.append(
                f"| {domain} | {r['n']} | {fmt(r['cos_raw'], 5)} | {fmt(r['cos_opt'], 5)} | "
                f"{fmt(r['positive'], 3)} | {fmt(r['loss'], 4)} | {fmt(r['dot_raw'], 5)} | "
                f"{fmt(r['lift'], 3)} |"
            )
        lines.append("")

    lines += [
        "## Does alignment explain the lift, or does loss explain the alignment?",
        "",
        "Per-setting correlations across domains:",
        "",
        "| setting | domains | corr(cos_raw, lift) | corr(cos_raw, loss) | corr(dot_raw, lift) |",
        "|---|---|---|---|---|",
    ]
    for row in corr:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    lines += [
        "",
        "Reading it:",
        "",
        "- **corr(dot_raw, lift) high** — the selector's domain tilt is explained by",
        "  the raw score, as it should be. If it is low, the layer-averaging",
        "  attenuation in `lift` is dominating and the lift number is not",
        "  measuring domain preference.",
        "- **corr(cos_raw, loss) strongly positive** — high-loss domains look more",
        "  aligned, i.e. the score is largely tracking how much room is left rather",
        "  than directional similarity. That is the 'already fit -> small gradient'",
        "  mechanism, and it would explain a target domain scoring below 1 in lift",
        "  while still being the semantically right data.",
        "",
        "## Caveat",
        "",
        "All of this is measured at the **base** checkpoint. During training the",
        "model fits parts of the pool at different rates, so a domain's loss (and",
        "so its gradient magnitude) moves. These numbers describe step 0, which is",
        "where selection starts, not the whole trajectory. Probing the saved final",
        "checkpoints would give the other end of the run.",
    ]
    (out_dir / f"README_domain_alignment_{split}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def plot_alignment(summaries, settings, out_dir: Path, split: str) -> None:
    from SFT.eval.analysis.viz import (
        CATEGORICAL,
        TEXT_MUTED,
        TEXT_SECONDARY,
        apply_style,
        strip_spines,
    )
    import matplotlib.pyplot as plt

    apply_style()
    targets = setting_targets()
    fig, axes = plt.subplots(
        1, len(settings), figsize=(2.9 * len(settings) + 1.0, 4.0)
    )
    axes = list(axes) if len(settings) > 1 else [axes]
    target_domain = {"precise_if": "precise_if", "math": "math", "mbpp": "coding"}

    for ax, setting in zip(axes, settings):
        strip_spines(ax)
        ax.grid(axis="y", visible=False)
        table = summaries[setting]
        ordered = sorted(table, key=lambda d: table[d]["cos_raw"])
        marked = target_domain.get(targets.get(setting, ""), None)
        values = [table[d]["cos_raw"] for d in ordered]
        # One hue for the setting's own target-adjacent domain, one for the rest,
        # so identity is never carried by position alone.
        colors = [
            CATEGORICAL[1] if d == marked else CATEGORICAL[0] for d in ordered
        ]
        ax.barh(range(len(ordered)), values, color=colors, linewidth=0, height=0.72)
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=1.0)
        ax.set_yticks(range(len(ordered)))
        ax.set_yticklabels(ordered, fontsize=7.5)
        ax.set_title(f"{setting}\n(target: {targets.get(setting,'?')})", fontsize=8.5)
        ax.set_xlabel("mean cos(grad, target grad)", fontsize=8)
        for index, value in enumerate(values):
            ax.annotate(
                f"{value:.4f}",
                xy=(value, index),
                xytext=(3 if value >= 0 else -3, 0),
                textcoords="offset points",
                ha="left" if value >= 0 else "right",
                va="center", fontsize=6.5, color=TEXT_SECONDARY,
            )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CATEGORICAL[1], linewidth=0),
        plt.Rectangle((0, 0), 1, 1, color=CATEGORICAL[0], linewidth=0),
    ]
    fig.legend(
        handles, ["setting's own target domain", "other domains"],
        loc="lower center", ncol=2, fontsize=8,
    )
    fig.suptitle(
        "Absolute gradient alignment per domain (base checkpoint)", fontsize=11, y=1.0
    )
    fig.text(
        0.5, 0.045,
        "positive = pulls toward the target direction; this is a cosine, not a selection ratio",
        ha="center", fontsize=7.5, color=TEXT_MUTED,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(out_dir / f"domain_alignment_{split}.png", bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN)
    parser.add_argument("--split", default="val", choices=("val", "grad"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    out_dir = (
        results_root(args.campaign, Path(args.out) if args.out else None)
        / "layer_alignment"
    )
    payloads = load_probes(out_dir, args.split)
    if not payloads:
        print(f"no probe output with per-candidate records in {out_dir}")
        print("run: sbatch ... SFT/eval/analysis/probe_job.sh")
        return 1

    settings = [s for s in setting_order() if s in payloads]
    summaries = build_tables(payloads, settings, args.campaign, out_dir, args.split)
    corr = correlations(summaries)
    plot_alignment(summaries, settings, out_dir, args.split)
    build_report(summaries, settings, out_dir, args.split, corr)

    print(f"settings : {', '.join(settings)}")
    print(f"wrote    : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
