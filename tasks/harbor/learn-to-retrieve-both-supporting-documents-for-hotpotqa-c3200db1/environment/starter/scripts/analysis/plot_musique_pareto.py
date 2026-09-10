#!/usr/bin/env python3
"""Pareto: MuSiQue LLM-judge accuracy vs end-to-end throughput, memory layer vs RAG.

EVERY POINT CARRIES ITS CONDITIONS. Accuracy points are labelled with the corpus size they were
measured at, throughput points with bank size / doc length / optimization config. A plot that
puts a 512-doc accuracy next to a full-corpus one without saying so is making a false claim, and
it is the first thing a reader will ask about.

Three panels:
  LEFT    accuracy vs corpus size — both systems, identical protocol per point (same 128 queries,
          same matched corpus, same Qwen3-4B judge).
  MIDDLE  the actual Pareto: accuracy vs end-to-end throughput at MATCHED conditions. This is
          possible because the eval chunks documents at max_chunks_per_doc=1 / chunk_size=256, so
          bank slots = docs x 256 and the swept banks ARE the accuracy corpora
          (512->131072, 2048->524288, 8192->2097152). RAG's prompt is k*256+64 = 1344 tokens
          regardless of corpus size, so its throughput is a single flat value.
          RESULT: RAG wins BOTH axes at every corpus size, with one accuracy tie at c8192
          (+0.0078, at the ~0.008 noise floor) where RAG still wins throughput. No Pareto win.
  RIGHT   end-to-end queries/sec vs DOCUMENT LENGTH, with the optimization ladder. The memory
          layer overtakes RAG past ~1.7k tokens/doc — but MuSiQue's documents are 256 tokens,
          ~7x below that, so this panel is a PROJECTION to long-document QA and is labelled as
          one. It is NOT a MuSiQue result and must never be read against the accuracy panel.

WHY THE THIRD PANEL IS SEPARATED. Plotting long-document throughput next to MuSiQue accuracy as a
single head-to-head would be true numbers arranged into a false claim. The memory layer's
throughput advantage is real and the crossover is measured, but it belongs to a workload this task
does not contain; turning it into a Pareto claim requires an accuracy measurement on an actual
long-document benchmark (NovelHopQA, narrativeqa), which has not been run.

Inputs (produced by the overnight sweeps):
  gs://…/pareto/e2e.jsonl              scripts/embed/sweep_pareto_e2e.sh
  gs://…/pareto/throughput.jsonl       scripts/embed/sweep_pareto_throughput.sh
  gs://…/<run-dir>/eval/step_N/*.json  scripts/embed/sweep_mem_corpus.sh
  gs://…/pareto/rag_corpus_sweep/      scripts/embed/sweep_rag_corpus.sh

Run: python scripts/analysis/plot_musique_pareto.py   (reads ACC_JSON/E2E_JSONL from env or
     the defaults below; writes results/figures/musique_pareto.png)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "..", "results", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

ACC_JSON = os.environ.get("ACC_JSON", os.path.join(HERE, "..", "..", "results", "musique_pareto_acc.json"))
E2E_JSONL = os.environ.get("E2E_JSONL", os.path.join(HERE, "..", "..", "results", "pareto_e2e.jsonl"))

acc = json.load(open(ACC_JSON))
e2e = [json.loads(l) for l in open(E2E_JSONL)] if os.path.exists(E2E_JSONL) else []

# The eval chunks each document into ONE 256-token chunk, so a corpus of N documents occupies
# N*256 memory-bank slots. This is what lets the accuracy and throughput axes be compared at
# matched conditions rather than merely placed side by side.
SLOTS_PER_DOC = 256
MUSIQUE_DOC_LEN = 256          # tokens per document in this task
RAG_K = 5

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 5.8))

# ---------------------------------------------------------------- LEFT: accuracy vs corpus
for name, series in acc.items():
    xs = sorted(int(k) for k in series)
    ys = [series[str(x)] for x in xs]
    style = dict(marker="o", lw=2) if "RAG" in name else dict(marker="s", lw=2, ls="--")
    ax1.plot(xs, ys, label=name, **style)
    for x, y in zip(xs, ys):
        ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=7)

ax1.set_xscale("log")
ax1.set_xlabel("corpus size (documents, log scale)")
ax1.set_ylabel("MuSiQue LLM-judge accuracy  (n=128)")
ax1.set_title("Accuracy vs corpus size\nidentical protocol per point: same queries, matched corpus, Qwen3-4B judge",
              fontsize=10)
ax1.grid(alpha=0.3)
ax1.legend(fontsize=8)

rag = sorted([r for r in e2e if r["mode"] == "rag"], key=lambda r: r["config"]["kvlen"])
mem = [r for r in e2e if r["mode"] == "mem"]
doclen = lambda r: (r["config"]["kvlen"] - 64) // RAG_K
slots = lambda n: f"{n//1048576}M" if n >= 1048576 else f"{n//1024}k"

# ------------------------------------------- MIDDLE: the actual Pareto, at matched conditions
if e2e and mem:
    # RAG's prompt is k*doc_len+64 tokens no matter how big the haystack is, so on MuSiQue its
    # throughput is ONE number for every corpus size. Take it from the measured doc_len=256 point.
    rag_q = next((r["end_to_end"]["queries_per_s"] for r in rag
                  if doclen(r) == MUSIQUE_DOC_LEN), None)
    # Best (fastest) memory config per bank, and the corpus size that bank corresponds to.
    best_by_bank = {}
    for r in mem:
        b = r["config"]["bank_slots"]
        if b not in best_by_bank or r["end_to_end"]["queries_per_s"] > best_by_bank[b]["end_to_end"]["queries_per_s"]:
            best_by_bank[b] = r

    rag_acc = next((s for n, s in acc.items() if "RAG" in n), {})
    mem_series = {n: s for n, s in acc.items() if "RAG" not in n}

    if rag_q:
        xs = [rag_q] * len(rag_acc)
        ys = [rag_acc[k] for k in sorted(rag_acc, key=int)]
        ax2.plot(xs, ys, marker="o", ms=9, lw=1.4, ls="-", color="tab:blue",
                 label=f"RAG (prompt = {RAG_K}x{MUSIQUE_DOC_LEN}+64 tok, flat in corpus size)")
        for k, y in zip(sorted(rag_acc, key=int), ys):
            ax2.annotate(f"c{k}", (rag_q, y), textcoords="offset points", xytext=(8, -3),
                         fontsize=7, color="tab:blue")

    for mi, (name, series) in enumerate(mem_series.items()):
        px, py, lbl = [], [], []
        for k in sorted(series, key=int):
            bank = int(k) * SLOTS_PER_DOC
            if bank in best_by_bank:
                px.append(best_by_bank[bank]["end_to_end"]["queries_per_s"])
                py.append(series[k]); lbl.append(k)
        if px:
            c = ["tab:orange", "tab:green"][mi % 2]
            ax2.plot(px, py, marker="s", ms=8, lw=1.4, ls="--", color=c, label=name)
            for x, y, k in zip(px, py, lbl):
                ax2.annotate(f"c{k}", (x, y), textcoords="offset points", xytext=(8, -3),
                             fontsize=7, color=c)

    ax2.set_xlabel("end-to-end queries/sec  (prefill + 100 generated tokens)  →  better")
    ax2.set_ylabel("MuSiQue LLM-judge accuracy  (n=128)  →  better")
    # The dominance statement goes in the TITLE, not in an in-axes box: at c8192 the memory point
    # sits at (0.381, 0.211) in the lower-left, exactly where such a box lands, and a label that
    # hides the very data it describes is worse than no label.
    ax2.set_title("THE PARETO, matched conditions — RAG wins BOTH axes; no Pareto win\n"
                  "bank slots = docs x 256, so each memory point sits at ITS corpus's bank",
                  fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.margins(0.14)
    ax2.legend(fontsize=7, loc="upper left")

    # A corpus whose bank was not swept has no measured memory throughput, so it cannot be placed
    # on this panel. Say so on the figure rather than dropping the point silently -- an absent
    # marker reads as "not measured" only if the plot admits it.
    dropped = sorted({int(k) for s in mem_series.values() for k in s
                      if int(k) * SLOTS_PER_DOC not in best_by_bank})
    if dropped:
        ax2.annotate("not shown: corpus " + ", ".join(f"c{k}" for k in dropped) +
                     f" (bank {dropped[0]*SLOTS_PER_DOC/1e6:.2f}M slots not swept)",
                     xy=(0.5, -0.19), xycoords="axes fraction", ha="center", va="top",
                     fontsize=7, color="0.35")
else:
    ax2.text(0.5, 0.5, "no end-to-end data yet", ha="center", va="center", transform=ax2.transAxes)

# ------------------------------- RIGHT: throughput vs document length (PROJECTION, not MuSiQue)
if e2e:

    if rag:
        xs = [doclen(r) for r in rag]
        ys = [r["end_to_end"]["queries_per_s"] for r in rag]
        ax3.plot(xs, ys, marker="o", lw=2.2, color="tab:red", zorder=5,
                 label="RAG (k=5 docs in prompt)")
        for x, y in zip(xs, ys):
            ax3.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -14),
                         ha="center", fontsize=7, color="tab:red")

    # Memory-layer prompt length does not depend on doc_len, so each config is a HORIZONTAL line
    # across the same x-range — that flatness IS the claim. Shade the ladder by bank size so the
    # optimization steps (exact -> approx -> approx+int8) read as a group per bank.
    if mem and rag:
        xr = [min(doclen(r) for r in rag), max(doclen(r) for r in rag)]
        banks = sorted({r["config"]["bank_slots"] for r in mem})
        cmap = {b: c for b, c in zip(banks, ["tab:blue", "tab:green", "tab:purple", "tab:orange"])}
        for r in sorted(mem, key=lambda r: (r["config"]["bank_slots"],
                                            -r["end_to_end"]["queries_per_s"])):
            c = r["config"]
            q = r["end_to_end"]["queries_per_s"]
            ls = {"exact": ":", "approx": "--"}[c["topk_mode"]]
            lw = 2.0 if c["key_dtype"] == "int8" else 1.3
            ax3.plot(xr, [q, q], ls=ls, lw=lw, alpha=0.9, color=cmap[c["bank_slots"]],
                     label=f"mem {slots(c['bank_slots'])} slots, {c['topk_mode']}/{c['key_dtype']} "
                           f"x{c['n_mem_layers']}L")

        # Crossover: where the RAG curve drops below the BEST memory config. Interpolated in
        # log(doc_len), and only drawn when the measured RAG curve actually brackets it — an
        # extrapolated crossover would be a claim the data does not support.
        best = max(mem, key=lambda r: r["end_to_end"]["queries_per_s"])
        qb = best["end_to_end"]["queries_per_s"]
        xs = [doclen(r) for r in rag]; ys = [r["end_to_end"]["queries_per_s"] for r in rag]
        xc = None
        for i in range(len(xs) - 1):
            if (ys[i] - qb) * (ys[i + 1] - qb) < 0:
                import math
                f = (ys[i] - qb) / (ys[i] - ys[i + 1])
                xc = math.exp(math.log(xs[i]) + f * (math.log(xs[i + 1]) - math.log(xs[i])))
                break
        ax3.axhline(qb, color="0.4", lw=0.8, ls="-", alpha=0.5)
        if xc:
            ax3.axvline(xc, color="0.3", lw=1.0, ls="-.", alpha=0.8)
            ax3.annotate(f"crossover\n~{xc:,.0f} tok/doc", (xc, qb), textcoords="offset points",
                         xytext=(8, 18), fontsize=8, color="0.2",
                         bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="0.6", lw=0.6))
        else:
            ax3.annotate("no crossover within measured range", (xr[0], qb),
                         textcoords="offset points", xytext=(6, 6), fontsize=8, color="0.3")

    # Mark where MuSiQue actually lives. Without this the panel reads as a MuSiQue result, which
    # it is not: the task's documents are 256 tokens, well left of any crossover.
    ax3.axvspan(xr[0], MUSIQUE_DOC_LEN * 1.02, color="tab:blue", alpha=0.07, zorder=0)
    ax3.axvline(MUSIQUE_DOC_LEN, color="tab:blue", lw=1.2, ls="-", alpha=0.7)
    ax3.annotate(f"MuSiQue lives HERE\n({MUSIQUE_DOC_LEN} tok/doc)\nRAG wins this regime",
                 (MUSIQUE_DOC_LEN, 0.02), xycoords=("data", "axes fraction"),
                 textcoords="offset points", xytext=(6, 4), fontsize=8, color="tab:blue",
                 bbox=dict(boxstyle="round,pad=0.3", fc="aliceblue", ec="tab:blue", lw=0.8))

    ax3.set_xscale("log")
    ax3.set_xlabel("document length (tokens, log scale)")
    ax3.set_ylabel("end-to-end queries/sec  (prefill + 100 generated tokens)")
    ax3.set_title("PROJECTION to long-document QA — NOT a MuSiQue result\n"
                  "RAG prefills k*doc_len tokens per query; the memory layer prefills ~64",
                  fontsize=10, color="0.25")
    ax3.grid(alpha=0.3)
    # 9 ladder lines + RAG crowds the axes; park the legend outside rather than over the curves.
    ax3.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1,
               frameon=True, borderaxespad=0)
else:
    ax3.text(0.5, 0.5, "no end-to-end data yet", ha="center", va="center", transform=ax3.transAxes)

fig.suptitle("MuSiQue: memory layer vs classic RAG — RAG wins both axes on this task "
             "(accuracy ties at c8192); the memory layer's throughput win needs documents "
             "MuSiQue does not have", fontsize=11)
fig.tight_layout()
path = os.path.join(OUT_DIR, "musique_pareto.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
print(f"wrote {path}")
