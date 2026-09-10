"""Plot the qhn64 mem-softmax-temperature sweep at fixed 10k corpus. Reads a
temp_sweep.json of {"<temp>": {llm_judge_accuracy, lexical_grounding, doc_access_acc,
mem_topk_entropy, ...}}. Left y-axis: score metrics [0,1]; right y-axis: mem_topk_entropy.

  python3 scripts/analysis/plot_temp_sweep.py [temp_sweep.json] [out.png]
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "temp_sweep.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "temp_sweep.png"

r = json.load(open(RESULTS))
temps = sorted((float(k) for k in r))

def series(key):
    return [r[_fmt(t)].get(key) for t in temps]

def _fmt(t):
    # match json keys ("1.0","0.5","0.25","0.1")
    for k in r:
        if float(k) == t:
            return k
    return str(t)

fig, ax = plt.subplots(figsize=(8.5, 5))
for key, style, label, dy in [
    ("llm_judge_accuracy", "o-", "LLM-judge accuracy", 8),
    ("lexical_grounding",  "s--", "Lexical grounding", -13),
    ("doc_access_acc",     "^:", "doc_access (retrieval hit)", -13),
]:
    ys = series(key)
    ax.plot(temps, ys, style, label=label, linewidth=2, markersize=6)
    for x, y in zip(temps, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=7)

ax.set_xlabel("mem softmax temperature  (lower = sharper mixture)")
ax.set_ylabel("score")
ax.set_ylim(0, 1)
ax.set_xticks(temps)
ax.invert_xaxis()  # sharper (small temp) on the right
ax.grid(True, alpha=0.3)

ax2 = ax.twinx()
ent = series("mem_topk_entropy")
ax2.plot(temps, ent, "d-", color="gray", alpha=0.7, label="mem_topk_entropy (right)")
ax2.set_ylabel("mem_topk_entropy", color="gray")
ax2.tick_params(axis="y", labelcolor="gray")

l1, la1 = ax.get_legend_handles_labels()
l2, la2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, la1 + la2, loc="center right", fontsize=8)
ax.set_title("qhn64 (topk) 10k-corpus: eval-time softmax sharpening (baseline temp=1.0)")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("wrote", OUT, "temps:", temps)
