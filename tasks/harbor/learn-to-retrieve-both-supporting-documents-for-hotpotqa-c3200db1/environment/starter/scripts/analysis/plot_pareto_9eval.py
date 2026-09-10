#!/usr/bin/env python3
"""
9 MSA evals x 3 methods Pareto: decode throughput (y) vs binary LLM-judge accuracy (x).

One point per (method, eval). Throughput is per-method (aggregate decode tok/s, B=16, best
config each) — roughly constant across small-corpus evals, so each method forms a horizontal
band; x spreads by per-eval accuracy. Reads results/pareto_9eval_data.json.
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "..", "results", "figures")
D = json.load(open(os.path.join(HERE, "..", "..", "results", "pareto_9eval_data.json")))

ACC = D["accuracy_binary_judge"]
TPUT = D["throughput_by_method_agg_tok_s"]
EVALS = D["evals"]
STYLE = {
    "rag":          ("#2a9d8f", "o", "RAG (Qwen3-4B)"),
    "memory_layer": ("#4361ee", "s", "Memory-layer (best)"),
    "msa":          ("#e76f51", "^", "MSA-4B"),
}

SHORT = {"natural_questions": "nq", "2wikimultihopqa": "2wiki", "hotpotqa": "hotpot",
         "msmarco_v1": "msmarco", "triviaqa_10m": "triviaqa", "narrativeqa": "narrqa",
         "dureader": "dureader", "musique": "musique", "popqa": "popqa"}
OVR = D.get("throughput_per_eval_override", {})
def tput(method, e):
    return OVR.get(method, {}).get(e, TPUT.get(method))
fig, ax = plt.subplots(figsize=(11, 6.4))
OFF = {"rag": 10, "memory_layer": 10, "msa": -16}
for method, (color, marker, label) in STYLE.items():
    if TPUT.get(method) is None:
        continue
    xs, ys, n = [], [], 0
    for i, e in enumerate(EVALS):
        a = ACC[method].get(e)
        if not isinstance(a, (int, float)):
            continue  # null / OOM
        y = tput(method, e)
        xs.append(a); ys.append(y); n += 1
        dy = OFF[method] + (14 if (i % 2) else 0) * (1 if OFF[method] > 0 else -1)
        ax.annotate(SHORT[e], (a, y), fontsize=6.5, alpha=0.8,
                    xytext=(0, dy), textcoords="offset points", ha="center", color=color)
    ax.scatter(xs, ys, s=120, color=color, marker=marker, zorder=5,
               edgecolor="white", linewidth=1.2,
               label=f"{label} — {TPUT[method]:.0f} tok/s  (n={n}/9)")

# document the non-numeric cells (OOM / retrieval-N/A / measuring) so all 27 are accounted for
STATUS_LABEL = {None: "measuring", "OOM": "OOM", "N/A": "retrieval-N/A"}
notes = []
for method, (_c, _m, label) in STYLE.items():
    for e in EVALS:
        a = ACC[method].get(e)
        if not isinstance(a, (int, float)):
            notes.append(f"{label.split(' ')[0]}/{SHORT.get(e, e)}={STATUS_LABEL.get(a, a)}")
if notes:
    ax.text(0.01, 0.985, "not plotted (documented): " + ", ".join(notes),
            transform=ax.transAxes, fontsize=6.8, va="top", color="#555", style="italic")

ax.set_xlabel("Binary LLM-judge accuracy (per eval)", fontsize=11)
ax.set_ylabel("Decode throughput (agg tok/s, B=16, 512 tok, one v6e-8, repo JAX)", fontsize=11)
ax.set_title("9 MSA evals × 3 methods — decode throughput vs binary LLM-judge accuracy\n"
             "Memory-layer 439→1774 tok/s (small corpus) but DROPS on big corpora (msmarco/triviaqa); "
             "RAG is corpus-independent & leads accuracy everywhere", fontsize=10)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0, max(v for v in TPUT.values() if isinstance(v, (int, float))) * 1.15)
ax.grid(alpha=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="center", bbox_to_anchor=(0.62, 0.62), fontsize=9.5, framealpha=0.95)
fig.text(0.5, -0.01, "Memory-layer throughput is corpus-dependent (scales with bank size): 1774 (small, B16) → "
         "488 (msmarco 19.3M-vec) / 575 (triviaqa 12.1M-vec) at B8 (large corpora OOM at B16). It handles 10M "
         "tokens (triviaqa acc 0.567, its best). RAG is corpus-independent (top-5). MSA OOMs on triviaqa.",
         ha="center", fontsize=7.3, style="italic")
fig.tight_layout()
p = os.path.join(OUT, "pareto_9eval_throughput_acc.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
print("wrote", p)
# coverage report
for m in STYLE:
    filled = [e for e in EVALS if isinstance(ACC[m].get(e), (int, float))]
    print(f"{m}: {len(filled)}/9 acc filled; missing {[e for e in EVALS if e not in filled]}")
