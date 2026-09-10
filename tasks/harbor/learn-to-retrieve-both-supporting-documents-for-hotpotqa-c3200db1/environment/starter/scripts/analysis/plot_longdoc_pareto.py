#!/usr/bin/env python3
"""QASPER serving latency vs MuSiQue accuracy, RLM vs RAG@5.

AXES ARE FROM DIFFERENT BENCHMARKS (both named in the axis labels). x = QASPER seconds/query:
1,169 papers, 5,404 tok/paper, 6.32M-slot bank; RAG@5 prefills a 27,084-tok prompt, RLM prefills
the 64-tok question only; each query is prefill + 100 generated tokens, B=1, single v5p chip.
y = MuSiQue LLM-judge accuracy, 2048-doc corpus, n=128, Qwen3-4B judge. Accuracy cannot be
measured on QASPER -- no checkpoint was trained on long documents, so it would be out-of-
distribution. Noise: accuracy +/-0.008, latency ~9%. All values measured; RLM is n_mem_layers=1.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Resolved from the script's location, not the working directory, so it runs from anywhere.
OUT = Path(__file__).resolve().parents[2] / "results" / "figures" / "longdoc_pareto.png"

C_RAG = "#eb4034"   # RAG
C_RLM = "#287dfc"   # ours

# label, QASPER s/query, MuSiQue accuracy, colour, marker
POINTS = [
    ("RAG@5",                                       4.430, 0.3281, C_RAG, "o"),
    ("Retrieval Layer Model (RLM)",                 7.259, 0.3125, C_RLM, "s"),
    ("RLM + Approximate Top-K",                     2.198, 0.3125, C_RLM, "o")
]

fig, ax = plt.subplots(figsize=(4.5, 3.5))
for label, x, y, colour, marker in POINTS:
    ax.plot(x, y, marker=marker, color=colour, markersize=12, linestyle="none", label=label)

# Speedup connector: RAG@5 -> RLM + Approximate Top-K. Ratio is computed from POINTS rather than
# hardcoded, so editing a latency above keeps the label correct.
_a = next(p for p in POINTS if p[0] == "RAG@5")
_b = next(p for p in POINTS if p[0] == "RLM + Approximate Top-K")
ax.annotate("", xy=(_b[1], _b[2]), xytext=(_a[1], _a[2]),   # arrow points RAG -> RLM
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2,
                            shrinkA=9, shrinkB=9))          # clear the markers at both ends
ax.annotate(f"{_a[1] / _b[1]:.1f}x faster",
            ((_a[1] * _b[1]) ** 0.5, (_a[2] + _b[2]) / 2),   # geometric mean: x axis is log
            textcoords="offset points", xytext=(-14, 7), ha="center",
            fontsize=11, color="black")

ax.set_xscale("log")
ax.set_xlim(1.5, 9.5)
ax.set_ylim(0.25, 0.35)
ax.set_yticks([ 0.25, 0.3, 0.35])
ax.set_xticks([2, 3, 4, 6, 8])
ax.set_xticklabels([2, 3, 4,  6,  8])
ax.set_xlabel("Long Context End-to-End Latency (s/query)", fontsize=12)
ax.set_ylabel("QA Accuracy - MuSiQue", fontsize=12)
ax.grid(alpha=0.25)
leg = ax.legend(fontsize=12, frameon=False, loc="lower left")
# Legend markers inherit the plot's markersize (12), which is oversized in a legend. Set them
# directly on the legend handles so the plot markers keep their size.
for handle in leg.legend_handles:
    handle.set_markersize(7)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
