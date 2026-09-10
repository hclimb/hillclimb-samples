"""Pareto plot: end-to-end throughput (QPS, log-x) vs binary LLM-judge accuracy.

Reads results/pareto_summary.json (accuracy) + results/throughput_data.json (QPS).
Honest first-order Pareto picture; see throughput_data.json caveats (not fully matched).
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
acc = json.load(open(os.path.join(ROOT, "results", "pareto_summary.json")))
tput = json.load(open(os.path.join(ROOT, "results", "throughput_data.json")))["methods"]

fig, ax = plt.subplots(figsize=(8, 5.5))
for m, t in tput.items():
    if m not in acc or not t.get("qps"):
        continue
    a = acc[m]["avg_binary_acc"]
    q = t["qps"]
    ax.scatter(q, a, s=160, zorder=3)
    ax.annotate(f"{t['label']}\n({q:g} QPS, {a:.2f})", (q, a),
                textcoords="offset points", xytext=(10, -4), fontsize=9)

ax.set_xscale("log")
ax.set_xlabel("End-to-end throughput (queries/sec, log scale) — native config")
ax.set_ylabel("Binary LLM-judge accuracy (avg over 3 MSA evals)")
ax.set_title("Pareto: throughput vs accuracy (3 MSA document-QA evals)")
ax.grid(True, which="both", alpha=0.3)
ax.set_ylim(0, 1)
fig.text(0.5, 0.01,
         "Caveat: output length & batching NOT matched (RAG=short answers/concurrency64; "
         "ours/MSA=512-tok CoT/B=16). See throughput_data.json.",
         ha="center", fontsize=7, color="gray")
fig.tight_layout(rect=[0, 0.03, 1, 1])
out = os.path.join(ROOT, "results", "figures", "pareto_qps.png")
fig.savefig(out, dpi=140)
print(f"wrote {out}")
for m, t in tput.items():
    if m in acc and t.get("qps"):
        print(f"  {t['label']:22s} qps={t['qps']:>6} acc={acc[m]['avg_binary_acc']:.3f}")
