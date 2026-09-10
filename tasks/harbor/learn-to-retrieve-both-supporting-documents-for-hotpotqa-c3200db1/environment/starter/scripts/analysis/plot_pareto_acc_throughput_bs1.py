#!/usr/bin/env python3
"""
Pareto (acc vs single-stream decode throughput) + a bar chart isolating two
memory-layer throughput levers: (1) sharding the bank across chips, (2) keeping
mem_v on-device vs the CPU pure_callback offload.

Single-stream = 1 sequence per chip on one v6e-8 (global B=8; per-stream = agg/8).
Literal global B=1 is not expressible for the data-parallel transformer; MSA's
separate decode path runs literal single-chip B=1 = 124.6, matching its
per-stream@B=8 = 109, validating the proxy. 512 tok, repo JAX.

Inset adds the int8 mem_k lever: quantizing the score-scan (quant_bench_bs1.py, recall@128
0.97 -> same accuracy) gives a SMALL full-decode gain at BS=1 (transformer-bound) that grows
with bank size: 512k +1%, 4M +4% (4M projected via a 1.4 TB/s bandwidth model anchored to the
measured 512k point). int4 omitted (recall 0.64 on synthetic; LLM-judge accuracy unmeasured).

Numbers: results/throughput_data.json -> bs1_single_stream_2026_06, int8_scan_projection_2026_07.
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "..", "results", "figures")
DATA = json.load(open(os.path.join(HERE, "..", "..", "results", "throughput_data.json")))
D = DATA["bs1_single_stream_2026_06"]
ACC = {"rag": 0.771, "msa": 0.136, "membed": 0.323}

rag   = D["rag_base_qwen3_4b_perstream"]
msa   = D["msa_4b_perstream"]
apx   = D["memory_layer_bank_sharded_deviceV_approx_perstream"]  # + approx_max_k (best)
best  = D["memory_layer_bank_sharded_deviceV_perstream"]   # sharded bank + on-device v (exact)
cpuv  = D["memory_layer_bank_sharded_cpuV_perstream"]       # sharded bank + CPU offload

# int8 mem_k scan (quant_bench_bs1.py): quantize the score-scan; recall@128 0.97 -> same accuracy.
Q = DATA["int8_scan_projection_2026_07"]
i8_512 = Q["membed_512k_int8_perstream"]      # 512k bank, int8   (measured-anchored)
bf_4m  = Q["membed_4M_bf16_approx_perstream"] # 4M bank,  bf16    (projected)
i8_4m  = Q["membed_4M_int8_perstream"]        # 4M bank,  int8    (projected)

# ---------------- Pareto: acc vs per-stream throughput ----------------
fig, ax = plt.subplots(figsize=(8.6, 6))
PTS = [
    ("RAG (base Qwen3-4B)",                ACC["rag"], rag,  "#2a9d8f", "o", -0.02,  6, "right"),
    ("Memory-layer + approx_max_k",        ACC["membed"], apx,  "#3a0ca3", "^", 0.018, 8, "left"),
    ("MSA-4B",                             ACC["msa"], msa,  "#e76f51", "o",  0.015, -14, "left"),
    ("Memory-layer\n(sharded bank, on-device v)", ACC["membed"], best, "#4361ee", "o", 0.018, -22, "left"),
    ("Memory-layer (CPU-offload v)",       ACC["membed"], cpuv, "#9bb0ff", "v", 0.018, -13, "left"),
]
# arrows: cpu-v -> device-v -> +approx (the three throughput levers)
ax.annotate("", xy=(ACC["membed"], best-1.5), xytext=(ACC["membed"], cpuv+1.5),
            arrowprops=dict(arrowstyle="->", color="#4361ee", lw=1.4, alpha=0.75))
ax.annotate("", xy=(ACC["membed"], apx-1.5), xytext=(ACC["membed"], best+1.5),
            arrowprops=dict(arrowstyle="->", color="#3a0ca3", lw=1.4, alpha=0.75))
ax.text(ACC["membed"]-0.13, (best+cpuv)/2, f"on-device mem_v\n(+{round((best/cpuv-1)*100)}%)",
        fontsize=7, color="#4361ee", va="center", ha="left")
ax.text(ACC["membed"]-0.13, (apx+best)/2+3, f"approx_max_k\n(+{round((apx/best-1)*100)}%)",
        fontsize=7, color="#3a0ca3", va="center", ha="left")
for label, acc, tput, color, marker, dx, dy, ha in PTS:
    ax.scatter(acc, tput, s=230, color=color, marker=marker, zorder=5,
               edgecolor="white", linewidth=1.5)
    ax.annotate(f"{label}\n(acc {acc:.2f}, {tput:.0f} tok/s)",
                (acc, tput), xytext=(acc+dx, tput+dy), ha=ha, fontsize=8.5,
                fontweight="bold" if label.startswith("RAG") else "normal")
GOLD = "#e9a000"
ax.fill_between([0, ACC["rag"]], 0, rag, color="#2a9d8f", alpha=0.06, zorder=0)
ax.text(0.40, rag*0.10, "RAG Pareto-dominates this region", fontsize=8.5,
        color="#2a9d8f", style="italic", ha="center")
# --- inset: int8 mem_k vs bf16 full-decode throughput by bank size (BS=1) ---
axi = ax.inset_axes([0.60, 0.07, 0.37, 0.30])
xs = [0, 1]
axi.plot(xs, [apx, bf_4m], "-o", color="#4361ee", ms=7, lw=1.6, label="bf16")
axi.plot(xs, [i8_512, i8_4m], "--*", color=GOLD, ms=12, lw=1.6, label="int8 (recall 0.97)")
for x, yb, yi in [(0, apx, i8_512), (1, bf_4m, i8_4m)]:
    axi.annotate("+%.0f%%" % ((yi/yb-1)*100), (x, yi), xytext=(x, yi+1.6),
                 ha="center", fontsize=7.5, color="#9a6a00", fontweight="bold")
axi.set_xticks(xs); axi.set_xticklabels(["512k", "4M"], fontsize=8)
axi.set_xlim(-0.3, 1.3); axi.set_ylim(103, 120)
axi.set_xlabel("bank size (vectors)", fontsize=7.5, labelpad=1)
axi.set_ylabel("tok/s", fontsize=7.5, labelpad=1)
axi.set_title("int8 score-scan (BS=1): gain grows with bank", fontsize=7.6)
axi.tick_params(labelsize=7); axi.legend(fontsize=7, loc="lower left", frameon=False)
axi.grid(alpha=0.3)

ax.set_xlabel("Binary LLM-judge accuracy  (mean of popqa / nq / hotpotqa)", fontsize=11)
ax.set_ylabel("Per-stream decode throughput (tok/s, 1 seq/chip, 512 tok, repo JAX)", fontsize=11)
ax.set_title("Accuracy vs single-stream decode throughput (1 seq/chip, v6e-8)\n"
             "int8 mem_k adds throughput at fixed accuracy; gain grows with bank size (transformer-bound at BS=1)", fontsize=10.0)
ax.set_xlim(0.0, 0.92)
ax.set_ylim(0, max(rag, msa) * 1.28)
ax.grid(alpha=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
p1 = os.path.join(OUT, "pareto_acc_vs_throughput_bs1.png")
fig.savefig(p1, dpi=150, bbox_inches="tight")
print("wrote", p1)

# ---------------- Bar: two memory-layer levers (aggregate B=8) ----------------
rep_a    = D["memory_layer_bank_replicated_agg_b8"]
sh_cpu_a = D["memory_layer_bank_sharded_cpuV_agg_b8"]
sh_dev_a = D["memory_layer_bank_sharded_deviceV_agg_b8"]
sh_apx_a = D["memory_layer_bank_sharded_deviceV_approx_agg_b8"]
labels = ["REPLICATED\n(on-device v)", "SHARDED\n(CPU-offload v)", "SHARDED\n+ on-device v", "SHARDED + on-device v\n+ approx_max_k"]
vals   = [rep_a, sh_cpu_a, sh_dev_a, sh_apx_a]
cols   = ["#9bb0ff", "#6f86e8", "#4361ee", "#3a0ca3"]
fig2, ax2 = plt.subplots(figsize=(7.6, 5))
bars = ax2.bar(labels, vals, color=cols, width=0.66, edgecolor="white")
for b, v in zip(bars, vals):
    ax2.text(b.get_x()+b.get_width()/2, v+10, f"{v:.0f}", ha="center", fontsize=11, fontweight="bold")
ax2.set_ylabel("Decode throughput (tok/s, B=8, 512 tok, repo JAX)", fontsize=10)
ax2.set_title(f"Memory-layer decode levers (B=8, one v6e-8) — best = {sh_apx_a:.0f}\n"
              f"shard bank {rep_a:.0f}→{sh_dev_a:.0f} ({sh_dev_a/rep_a:.2f}×) · drop CPU mem_v "
              f"{sh_cpu_a:.0f}→{sh_dev_a:.0f} ({sh_dev_a/sh_cpu_a:.2f}×) · approx_max_k "
              f"{sh_dev_a:.0f}→{sh_apx_a:.0f} ({sh_apx_a/sh_dev_a:.2f}×) → {sh_apx_a/rep_a:.2f}× total",
              fontsize=8)
ax2.tick_params(axis="x", labelsize=8)
ax2.set_ylim(0, max(vals) * 1.18)
ax2.grid(alpha=0.3, axis="y", zorder=0)
ax2.spines[["top", "right"]].set_visible(False)
fig2.tight_layout()
p2 = os.path.join(OUT, "memory_bank_sharding_effect_bs1.png")
fig2.savefig(p2, dpi=150, bbox_inches="tight")
print("wrote", p2)
