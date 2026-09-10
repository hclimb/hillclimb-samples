#!/usr/bin/env python3
"""
Pareto: binary LLM-judge accuracy (x) vs corrected same-engine decode throughput (y).

Accuracy = mean binary LLM-judge (Qwen3-8B) over 3 MSA evals (popqa, nq, hotpotqa):
  RAG 0.771 | memory-layer 0.323 | MSA 0.136   (per-eval in results/PARETO_FINDINGS.md)

Throughput = repo-JAX decode-only tok/s, B=16, 512 tok, single v6e-8 (corrected;
prior RAG number was a weight-sharding artifact — see plot_decode_throughput_corrected.py):
  RAG 2102 | MSA 1711 | memory-layer 781 (exact) / 981 (approx_max_k)

Result: with throughput fixed, RAG Pareto-DOMINATES (highest acc AND highest tput).
The earlier "memory-layer 2x faster than RAG" frontier was the sharding bug.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "figures")

# (label, acc, tput, color, marker, dx, dy, ha)
PTS = [
    ("RAG (base Qwen3-4B)",          0.771, 2102, "#2a9d8f", "o", -0.02,  60, "right"),
    ("MSA-4B",                       0.136, 1711, "#e76f51", "o",  0.015,  60, "left"),
    ("Memory-layer + approx_max_k",  0.323,  981, "#4361ee", "^",  0.018,  55, "left"),
    ("Memory-layer (exact)",         0.323,  781, "#4361ee", "o",  0.018, -95, "left"),
]

fig, ax = plt.subplots(figsize=(8.6, 6))

# arrow exact -> approx (throughput gain, ~no acc change)
ax.annotate("", xy=(0.323, 970), xytext=(0.323, 792),
            arrowprops=dict(arrowstyle="->", color="#4361ee", lw=1.4, alpha=0.7))
ax.text(0.345, 880, "approx_max_k\n+26% tput\n(recall 0.95)", fontsize=7.5,
        color="#4361ee", va="center")

for label, acc, tput, color, marker, dx, dy, ha in PTS:
    ax.scatter(acc, tput, s=230, color=color, marker=marker, zorder=5,
               edgecolor="white", linewidth=1.5)
    ax.annotate(f"{label}\n(acc {acc:.2f}, {tput:,} tok/s)",
                (acc, tput), xytext=(acc+dx, tput+dy), ha=ha, fontsize=9,
                fontweight="bold" if label.startswith("RAG") else "normal")

# shade RAG's dominated region (everything below-left of RAG)
ax.axvspan(-1, 0.771, ymin=0, ymax=2102/2450, alpha=0.0)  # placeholder
ax.fill_between([0, 0.771], 0, 2102, color="#2a9d8f", alpha=0.06, zorder=0)
ax.text(0.40, 250, "RAG Pareto-dominates this region\n(higher acc AND higher throughput)",
        fontsize=8.5, color="#2a9d8f", style="italic", ha="center")

ax.set_xlabel("Binary LLM-judge accuracy  (mean of popqa / nq / hotpotqa)", fontsize=11)
ax.set_ylabel("Decode throughput (tok/s, B=16, 512 tok, repo JAX)", fontsize=11)
ax.set_title("Accuracy vs decode throughput — corrected same-engine\n"
             "RAG dominates once the sharding bug is fixed (was: false 'memory-layer 2× faster')",
             fontsize=11)
ax.set_xlim(0.0, 0.92)
ax.set_ylim(0, 2450)
ax.grid(alpha=0.3, zorder=0)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
p = os.path.join(OUT, "pareto_acc_vs_throughput_corrected.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
print("wrote", p)
