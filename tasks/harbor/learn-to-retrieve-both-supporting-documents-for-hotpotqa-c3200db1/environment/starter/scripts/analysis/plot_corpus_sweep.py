"""Plot the qhn64 corpus-size sweep: llm_judge accuracy, lexical grounding, and doc_access
vs corpus size (# docs in the memory bank), log-x. Reads a sweep_results.json of the form
{ "<size>": {"llm_judge_accuracy":..., "lexical_grounding":..., "doc_access_acc":..., "mechanism":"batch|corpus"} }.

  python3 scripts/analysis/plot_corpus_sweep.py [results.json] [out.png]
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "sweep_results.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "corpus_sweep.png"

r = json.load(open(RESULTS))
sizes = sorted((int(k) for k in r), key=int)

def series(key):
    xs, ys = [], []
    for s in sizes:
        v = r[str(s)].get(key)
        if v is not None:
            xs.append(s); ys.append(v)
    return xs, ys

fig, ax = plt.subplots(figsize=(8, 5))
for key, style, label in [
    ("llm_judge_accuracy", "o-", "LLM-judge accuracy"),
    ("lexical_grounding",  "s--", "Lexical grounding"),
    ("doc_access_acc",     "^:", "doc_access (retrieval hit)"),
]:
    xs, ys = series(key)
    ax.plot(xs, ys, style, label=label, linewidth=2, markersize=6)

# annotate accuracy points
xa, ya = series("llm_judge_accuracy")
for x, y in zip(xa, ya):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8)

ax.set_xscale("log")
ax.set_xlabel("corpus size  (# docs in memory bank)")
ax.set_ylabel("score")
ax.set_ylim(0, 1)
ax.set_title("qhn64 (topk64) MS MARCO oracle: grounding vs corpus size (n=128)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("wrote", OUT, "sizes:", sizes)
