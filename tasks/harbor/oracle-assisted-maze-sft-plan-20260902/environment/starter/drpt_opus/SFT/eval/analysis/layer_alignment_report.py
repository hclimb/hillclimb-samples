#!/usr/bin/env python
"""Render the layer-alignment probe into the w_l table and its verdict.

Consumes `<setting>_<split>_probe.json` written by `layer_alignment_probe` and
answers the question the probe was built for: does a target-derived per-group
weight actually separate by target, or is it the same profile every time?

The null result is a result. If the profiles agree across targets, then w_l is
target-insensitive at initialisation and deriving it automatically buys nothing
over a fixed heuristic — that is a finding, not a failure, and this script
states it either way rather than only reporting a difference.

Usage:
  python -m SFT.eval.analysis.layer_alignment_report [--campaign ID] [--split val]
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
    diagnostics_frame,
    discover_runs,
    fmt,
    index_runs,
    results_root,
    setting_order,
    summarize_series,
    write_table,
)

PROFILES = (
    ("energy_raw", "target energy share", "||grad_g L_target||^2, normalised"),
    ("energy_opt", "target energy share (AdamW geom.)", "preconditioned energy, normalised"),
    ("align_raw", "alignment share", "mean_i |<grad_g L_i, grad_g L_target>|, normalised"),
    ("align_opt", "alignment share (AdamW geom.)", "preconditioned alignment, normalised"),
)

# The five-row view requested for the writeup, as a subset of the full grid.
HEADLINE_ROWS = (
    ("Embedding", "embedding"),
    ("Early attn", "attn_early"),
    ("Mid MLP", "mlp_mid"),
    ("Late MLP", "mlp_late"),
    ("LM head", "lm_head"),
)

# Coarse groups the trained runs already log, for the free consistency check.
COARSE = {
    "embedding": ("embedding",),
    "attention": ("attn_early", "attn_mid", "attn_late"),
    "mlp": ("mlp_early", "mlp_mid", "mlp_late"),
    "lm_head": ("lm_head",),
}


def normalise(values: Dict[str, float], groups: Sequence[str]) -> Dict[str, float]:
    total = sum(max(values.get(g, 0.0), 0.0) for g in groups)
    if total <= 0:
        return {g: float("nan") for g in groups}
    return {g: max(values.get(g, 0.0), 0.0) / total for g in groups}


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation without scipy; average ranks for ties."""
    pairs = [(x, y) for x, y in zip(a, b) if x == x and y == y]
    if len(pairs) < 3:
        return float("nan")

    def ranks(values: Sequence[float]) -> List[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position
            while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
                end += 1
            average = (position + end) / 2.0 + 1.0
            for index in range(position, end + 1):
                out[order[index]] = average
            position = end + 1
        return out

    xs, ys = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def load_probes(out_dir: Path, split: str) -> Dict[str, dict]:
    payloads: Dict[str, dict] = {}
    for path in sorted(out_dir.glob(f"*_{split}_probe.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads[payload["setting"]] = payload
    return payloads


def build_tables(payloads, settings, groups, out_dir: Path, split: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    shares: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key, _, _ in PROFILES:
        shares[key] = {
            setting: normalise(payloads[setting][key], groups) for setting in settings
        }

    for key, label, _ in PROFILES:
        header = ["layer group"] + list(settings)
        rows = [
            [group] + [fmt(shares[key][s].get(group), 4) for s in settings]
            for group in groups
        ]
        write_table(out_dir / f"w_l_{key}_{split}", header, rows)

    # Scale-free cosine and a size control, so a big group is not mistaken for
    # an informative one.
    header = ["layer group", "params"] + [f"{s} cos" for s in settings]
    rows = []
    for group in groups:
        counts = payloads[settings[0]]["param_counts"].get(group, 0)
        rows.append(
            [group, f"{counts:,}"]
            + [fmt(payloads[s]["cosine"].get(group), 4) for s in settings]
        )
    write_table(out_dir / f"w_l_cosine_{split}", header, rows)
    return shares


def build_headline(shares, settings, out_dir: Path, split: str) -> List[List[object]]:
    key = "align_raw"
    header = ["Layer group"] + list(settings)
    rows = [
        [label] + [fmt(shares[key][s].get(group), 4) for s in settings]
        for label, group in HEADLINE_ROWS
    ]
    write_table(out_dir / f"w_l_headline_{split}", header, rows)
    return rows


def cross_setting_agreement(shares, settings, groups) -> Dict[str, List[List[object]]]:
    matrices: Dict[str, List[List[object]]] = {}
    for key, _, _ in PROFILES:
        matrix: List[List[object]] = []
        for a in settings:
            row: List[object] = [a]
            for b in settings:
                rho = spearman(
                    [shares[key][a][g] for g in groups],
                    [shares[key][b][g] for g in groups],
                )
                row.append(fmt(rho, 3))
            matrix.append(row)
        matrices[key] = matrix
    return matrices


def consistency_check(campaign: str, settings, shares) -> List[List[object]]:
    """Compare the probe's coarse ordering against the trained runs' own logs.

    The runs already record `update/adamw_<group>/selected_alignment` for four
    coarse groups. Agreement there is free evidence that the probe measures the
    same thing the selector reacts to.
    """
    runs = index_runs(discover_runs(campaign, optimizer="adamw"))
    rows: List[List[object]] = []
    for setting in settings:
        run = runs.get((setting, "LayerwiseRaw"))
        if run is None:
            continue
        frame = diagnostics_frame(run)
        if not frame:
            continue
        logged = {}
        for coarse in COARSE:
            stats = summarize_series(frame, f"update/adamw_{coarse}/selected_alignment")
            if stats is not None:
                logged[coarse] = stats["mean"]
        if len(logged) < 3:
            continue
        probe_coarse = {
            coarse: sum(shares["align_raw"][setting].get(g, 0.0) for g in members)
            for coarse, members in COARSE.items()
        }
        names = [c for c in COARSE if c in logged]
        rho = spearman([probe_coarse[c] for c in names], [logged[c] for c in names])
        rows.append([
            setting,
            " > ".join(sorted(names, key=lambda c: -probe_coarse[c])),
            " > ".join(sorted(names, key=lambda c: -logged[c])),
            fmt(rho, 3),
        ])
    return rows


def plot_profiles(shares, settings, groups, out_dir: Path, split: str) -> None:
    from SFT.eval.analysis.viz import (
        CATEGORICAL,
        TEXT_MUTED,
        apply_style,
        strip_spines,
    )
    import matplotlib.pyplot as plt

    apply_style()
    keys = [key for key, _, _ in PROFILES]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 4.2), sharey=True)
    axes = list(axes) if len(keys) > 1 else [axes]
    positions = range(len(groups))
    width = 0.8 / max(len(settings), 1)

    for ax, key in zip(axes, keys):
        strip_spines(ax)
        ax.grid(axis="x", visible=False)
        for index, setting in enumerate(settings):
            offsets = [p + (index - (len(settings) - 1) / 2) * width for p in positions]
            values = [shares[key][setting].get(g, float("nan")) for g in groups]
            ax.bar(
                offsets, values, width=width * 0.88,
                color=CATEGORICAL[index % len(CATEGORICAL)],
                label=setting, linewidth=0,
            )
        label = next(l for k, l, _ in PROFILES if k == key)
        ax.set_title(label, fontsize=9)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=7.5)

    axes[0].set_ylabel("share of total (sums to 1 per setting)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8)
    fig.suptitle(
        "Where the target signal lives, by layer group and target", fontsize=11, y=1.0
    )
    fig.text(
        0.5, 0.055,
        "base checkpoint, no training — differences between bars come only from the target set",
        ha="center", fontsize=7.5, color=TEXT_MUTED,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(out_dir / f"w_l_profile_{split}.png", bbox_inches="tight")
    plt.close(fig)


def build_report(payloads, shares, settings, groups, out_dir, split, headline, matrices, checks) -> None:
    key = "align_raw"
    # A profile is "target-specific" only if the group ordering actually moves
    # between targets; near-1 rank correlation everywhere means it does not.
    off_diagonal = [
        float(matrices[key][i][j + 1])
        for i in range(len(settings))
        for j in range(len(settings))
        if i != j and matrices[key][i][j + 1] not in ("—", "nan")
    ]
    mean_rho = sum(off_diagonal) / len(off_diagonal) if off_diagonal else float("nan")

    lines = [
        f"# Is w_l learnable from target gradients? ({split} split)",
        "",
        "One forward+backward per setting at the **base** checkpoint",
        f"(`{payloads[settings[0]]['model_profile']}`), no training. All settings share",
        "that checkpoint, so every difference below comes from the target set alone.",
        "",
        f"- target split: `target_{split}` "
        f"({payloads[settings[0]]['target_examples']} examples)",
        f"- candidates: {payloads[settings[0]]['candidate_examples']} examples "
        f"({payloads[settings[0]]['windows']} pinned windows of 16)",
        "- `P` = 1/(sqrt(vhat) + 1e-8), vhat = mean squared batch gradient over"
        " those windows (the stationary value AdamW's exp_avg_sq converges to)",
        "",
        "**Qwen3 ties the embedding and the LM head to one tensor.** The probe",
        "unties them with an exact-value clone so the two call sites accumulate",
        "separate gradients; no forward value changes and nothing is stepped. In",
        "the *trained* runs the optimizer updates the tied tensor, so the",
        "Embedding and LM head rows below are attributions of one shared weight,",
        "not two independently trainable blocks.",
        "",
        "## Headline table — alignment share (raw geometry)",
        "",
    ]
    header = ["Layer group"] + list(settings)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in headline:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")

    lines += [
        "",
        "## Verdict: does the profile separate by target?",
        "",
        f"Mean off-diagonal Spearman rank correlation between settings: "
        f"**{mean_rho:.3f}** (alignment share, {len(groups)} groups).",
        "",
    ]
    if mean_rho != mean_rho:
        lines.append("Not enough data to judge.")
    elif mean_rho > 0.9:
        lines += [
            "The group ordering is **essentially the same for every target**. On this",
            "evidence a target-derived `w_l` is target-insensitive at initialisation:",
            "automating it would reproduce one fixed profile, so it is not a",
            "methodological contribution on its own. The honest framings left are",
            "(a) derive `w_l` once as a principled default rather than a hand-picked",
            "heuristic, or (b) re-probe *during* training, where the profile may",
            "diverge as each run adapts to its target.",
        ]
    elif mean_rho > 0.6:
        lines += [
            "The ordering is broadly shared but not identical. There is some",
            "target-specific structure; whether it is large enough to matter needs",
            "the per-group magnitudes below, not the rank correlation alone.",
        ]
    else:
        lines += [
            "The ordering **differs materially between targets**, which is the",
            "precondition for a learned `w_l` to be worth more than a fixed",
            "heuristic. The next step is to feed these weights back into selection",
            "and measure target val loss against the fixed-`w_l` runs.",
        ]

    lines += [
        "",
        "## Cross-setting rank correlation",
        "",
    ]
    for profile_key, label, _ in PROFILES:
        lines += [f"### {label}", ""]
        matrix_header = ["setting"] + list(settings)
        lines.append("| " + " | ".join(matrix_header) + " |")
        lines.append("|" + "|".join("---" for _ in matrix_header) + "|")
        for row in matrices[profile_key]:
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
        lines.append("")

    if checks:
        lines += [
            "## Consistency check against the trained runs",
            "",
            "The completed `LayerwiseRaw` runs log",
            "`update/adamw_<group>/selected_alignment` for four coarse groups.",
            "If the probe measures the same thing the selector reacts to, the two",
            "orderings should agree.",
            "",
            "| setting | probe order | trained-run order | Spearman |",
            "|---|---|---|---|",
        ]
        for row in checks:
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
        lines.append("")

    lines += [
        "## Files",
        "",
        "| file | contents |",
        "|---|---|",
        f"| `w_l_align_raw_{split}.*` | alignment share per group (the headline profile) |",
        f"| `w_l_align_opt_{split}.*` | same under AdamW geometry |",
        f"| `w_l_energy_raw_{split}.*` | target gradient energy share |",
        f"| `w_l_energy_opt_{split}.*` | same under AdamW geometry |",
        f"| `w_l_cosine_{split}.*` | scale-free cosine + parameter counts |",
        f"| `w_l_headline_{split}.*` | the five-row view |",
        f"| `w_l_profile_{split}.png` | all four profiles as a figure |",
        "",
        "A leak note: `target_val` is the split the reported target val loss is",
        "measured on. Any `w_l` fitted on it and then used to pick training data",
        "would be tuned on the evaluation set. Re-run the probe with",
        "`--split grad` for the leak-free variant before making a method claim.",
    ]
    (out_dir / f"README_{split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return mean_rho


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
        print(f"no probe output found in {out_dir} for split={args.split}")
        print("run: python -m SFT.eval.analysis.layer_alignment_probe --all-settings")
        return 1

    settings = [s for s in setting_order() if s in payloads]
    groups = payloads[settings[0]]["groups"]

    shares = build_tables(payloads, settings, groups, out_dir, args.split)
    headline = build_headline(shares, settings, out_dir, args.split)
    matrices = cross_setting_agreement(shares, settings, groups)
    checks = consistency_check(args.campaign, settings, shares)
    plot_profiles(shares, settings, groups, out_dir, args.split)
    mean_rho = build_report(
        payloads, shares, settings, groups, out_dir, args.split,
        headline, matrices, checks,
    )

    print(f"settings : {', '.join(settings)}")
    print(f"groups   : {len(groups)}")
    print(f"mean rho : {mean_rho:.3f} (cross-setting, alignment share)")
    print(f"wrote    : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
