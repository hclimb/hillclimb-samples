#!/usr/bin/env python3
"""
Corrected same-engine decode throughput (repo JAX, B=16, 512 tok, decode-only).

Supersedes the earlier pareto_decode_throughput plots, whose RAG/MSA numbers were
crippled by a weight-sharding artifact (fresh `qwen3.load` shards the matmul
contraction dim on the 8-chip 'data' axis -> all-reduce every decode step).
Replicated weights remove the collective. All numbers below are repo-JAX
decode-only tok/s on a single v6e-8, greedy.

  RAG (base Qwen3-4B)  : replicated base, corpus-independent (reads top-k in ctx)
  MSA-4B               : 18 compressed-routing layers
  Memory-layer (exact) : 1 mem layer, full O(M) bank scoring (top-128 over ~1.5M vec)
  Memory-layer (approx): same, jax.lax.approx_max_k (recall 0.95)
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "figures")

# --- corrected numbers (repo JAX, B16, 512 tok, decode-only) ---
DATA = [
    # label,                value, old_note,                  corpus_note,                 attended_note
    ("RAG\n(base Qwen3-4B)",     2102, "was 376:\nsharding bug",    "corpus-indep.\n(reads top-5)",  "~2-3k ctx tok"),
    ("MSA-4B",                   1711, "was 351:\nnative-e2e metric","8.7k docs / 1.19M tok",        "12,288 tok/layer ×18"),
    ("Memory-layer\n(approx_max_k)", None, None,                    "2k docs / 0.74M tok",           "128 tok/pos"),
    ("Memory-layer\n(exact)",     781, "was 846:\ntiny corpus",     "2k docs / 0.74M tok",           "128 tok/pos"),
]

def make(approx_val):
    rows = []
    for label, val, old, corpus, attended in DATA:
        v = approx_val if val is None else val
        rows.append((label, v, old, corpus, attended))
    # sort desc by value
    rows.sort(key=lambda r: -r[1])

    labels  = [r[0] for r in rows]
    vals    = [r[1] for r in rows]
    olds    = [r[2] for r in rows]
    corpora = [r[3] for r in rows]
    att     = [r[4] for r in rows]

    colors = []
    for l in labels:
        if l.startswith("RAG"): colors.append("#2a9d8f")
        elif l.startswith("MSA"): colors.append("#e76f51")
        else: colors.append("#4361ee")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = range(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.62, zorder=3)

    for i, (b, v, old, corpus, attended) in enumerate(zip(bars, vals, olds, corpora, att)):
        ax.text(b.get_x()+b.get_width()/2, v+30, f"{v:,}", ha="center", va="bottom",
                fontweight="bold", fontsize=11)
        ax.text(b.get_x()+b.get_width()/2, v/2, f"{corpus}\n\nattends:\n{attended}",
                ha="center", va="center", fontsize=7.5, color="white", linespacing=1.2)
        if old:
            ax.text(b.get_x()+b.get_width()/2, -120, f"({old})",
                    ha="center", va="top", fontsize=7, color="#999", style="italic")

    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Decode throughput (tok/s, B=16, 512 tok)", fontsize=11)
    ax.set_title("Same-engine decode throughput (repo JAX, single v6e-8, greedy)\n"
                 "Corrected — RAG was crippled by a weight-sharding bug; all re-measured decode-only",
                 fontsize=11)
    ax.set_ylim(-260, 2450)
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "pareto_decode_throughput_corrected.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print("wrote", p)

if __name__ == "__main__":
    import sys
    approx = int(sys.argv[1]) if len(sys.argv) > 1 else 900
    make(approx)
