#!/usr/bin/env python3
"""QASPER serving latency vs MuSiQue accuracy, v2 — adds the 2026-07-21 baselines.

Modified copy of plot_longdoc_pareto.py (the original and its PNG stay as the 07-19 page's
artifact). Same mixed-axes design, both benchmarks named on the labels: x = QASPER
seconds/query (log; 8.24M-token all-splits corpus for SnapKV, 6.32M-slot bank for RLM,
27,084-tok prompt for RAG@5; each query = prefill + generation, B=1). y = MuSiQue LLM-judge
accuracy at the chosen corpus (n=128, Qwen3-4B judge).

    uv run python scripts/analysis/plot_longdoc_pareto_v2.py --corpus 2048   # default
    uv run python scripts/analysis/plot_longdoc_pareto_v2.py --corpus 512

v2 additions per corpus: SnapKV context stuffing (x = 27.33 s/query measured after a one-time
59-min index; y = that corpus's MuSiQue accuracy — c2048 run stopped at user request after 71/128 queries -> vertical marker; partial generations (unjudged) at gs://memory-layers-training/pareto/snapkv/),
plus MuSiQue-only horizontal references for the hybrid RAG@50->memory and the oracle-50
retrieval ceiling (no QASPER latency measured for those, so lines, not points).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator

FIGDIR = Path(__file__).resolve().parents[2] / "results" / "figures"

C_RAG = "#eb4034"
C_RLM = "#287dfc"
C_SNAP = "#c98500"
C_REF = "#777777"

SNAPKV_QASPER_S = 27.329        # measured, median over 64 queries, after one-time 59-min index

# Per-corpus MuSiQue judge accuracies (n=128). SnapKV c2048 pending -> None.
ACC = {
    "512":  {"rag": 0.3906, "rlm": 0.2810, "snapkv": 0.5312,
             "hybrid": 0.3984, "oracle": 0.4844,
             "ylim": (0.25, 0.56), "yticks": [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]},
    "2048": {"rag": 0.3281, "rlm": 0.3125, "snapkv": None,
             "hybrid": 0.2734, "oracle": 0.4531,
             "ylim": (0.25, 0.50), "yticks": [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["512", "2048"], default="2048")
    args = ap.parse_args()
    a = ACC[args.corpus]
    out = FIGDIR / (f"longdoc_pareto_v2_c{args.corpus}.png" if args.corpus == "512"
                    else "longdoc_pareto_v2.png")

    points = [
        ("RAG@5",                        4.430, a["rag"],    C_RAG,  "o"),
        ("Retrieval Layer Model (RLM)",  7.259, a["rlm"],    C_RLM,  "s"),
        ("RLM + Approximate Top-K",      2.198, a["rlm"],    C_RLM,  "o"),
        ("SnapKV context stuffing",      SNAPKV_QASPER_S, a["snapkv"], C_SNAP, "D"),
    ]
    ref_lines = [
        ("Hybrid RAG@50 → memory (MuSiQue only)", a["hybrid"]),
        ("Oracle-50 retrieval ceiling (MuSiQue only)", a["oracle"]),
    ]

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for label, x, y, colour, marker in points:
        if y is None:
            ax.axvline(x, color=colour, lw=1.6, ls=":", alpha=0.9)
            ax.annotate(f"{label}\n(acc not measured — run stopped)", (x, a["ylim"][0] + 0.012),
                        fontsize=8.5, color=colour, ha="right", rotation=90, va="bottom",
                        textcoords="offset points", xytext=(-4, 0))
            continue
        ax.plot(x, y, marker=marker, color=colour, markersize=11, linestyle="none", label=label)

    ref_label_x = 8.5 if args.corpus == "512" else 1.62   # keep clear of that variant's legend
    for label, y in ref_lines:
        ax.axhline(y, color=C_REF, lw=1.1, ls="--", alpha=0.8)
        ax.annotate(label, (ref_label_x, y), fontsize=8, color=C_REF,
                    textcoords="offset points", xytext=(0, 3))

    # Speedup arrow only where the endpoints sit at comparable accuracy (c2048); at c512 the
    # 0.39 -> 0.28 drop would make a "faster" arrow read as an accuracy trade.
    if args.corpus == "2048":
        _a = next(p for p in points if p[0] == "RAG@5")
        _b = next(p for p in points if p[0] == "RLM + Approximate Top-K")
        ax.annotate("", xy=(_b[1], _b[2]), xytext=(_a[1], _a[2]),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, shrinkA=9, shrinkB=9))
        ax.annotate(f"{_a[1] / _b[1]:.1f}x faster",
                    ((_a[1] * _b[1]) ** 0.5, (_a[2] + _b[2]) / 2),
                    textcoords="offset points", xytext=(-14, 7), ha="center", fontsize=10)

    ax.set_xscale("log")
    ax.set_xlim(1.5, 70 if args.corpus == "512" else 40)
    ax.set_ylim(*a["ylim"])
    ax.set_yticks(a["yticks"])
    xt = [2, 3, 4, 6, 8, 12, 20, 30] + ([50] if args.corpus == "512" else [])
    ax.set_xticks(xt)
    ax.set_xticklabels(xt)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Long Context End-to-End Latency (s/query)", fontsize=11)
    ax.set_ylabel(f"QA Accuracy — MuSiQue c{args.corpus}", fontsize=11)
    ax.grid(alpha=0.25)
    # Opaque frame so the reference lines don't strike through the legend text; anchor per
    # corpus so it sits in that variant's empty region.
    if args.corpus == "512":
        leg = ax.legend(fontsize=8, loc="lower right", bbox_to_anchor=(0.995, 0.02),
                        handletextpad=0.4, borderpad=0.5,
                        frameon=True, framealpha=0.95, edgecolor="none")
    else:
        leg = ax.legend(fontsize=9.5, loc="center right", bbox_to_anchor=(0.98, 0.62),
                        frameon=True, framealpha=0.95, edgecolor="none")
    for handle in leg.legend_handles:
        handle.set_markersize(7)

    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
