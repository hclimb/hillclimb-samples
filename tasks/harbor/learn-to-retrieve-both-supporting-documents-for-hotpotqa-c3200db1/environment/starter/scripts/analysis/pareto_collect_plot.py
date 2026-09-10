"""Collect Pareto-study results across methods and plot throughput vs accuracy.

Accuracy = binary LLM-judge accuracy, averaged over the 9 MSA evals.
Throughput = steady-state decode tokens/s (batch) from the speed benchmark.

Sources (all optional — missing methods are skipped):
  membed : gs://memory-layers-training/pareto_eval/membed/<dataset>.json  (metrics.llm_judge_accuracy)
  rag    : gs://memory-layers-training/pareto_eval/rag/<dataset>.json
  msa    : results/msa_eval_summary.json (llm_judge_accuracy_binary)
  throughput: results/membench/bench_{membed,msa}.json (+ optional rag)

Usage:
  python scripts/analysis/pareto_collect_plot.py
"""
import json, os, statistics, sys

DATASETS = ["msmarco_v1", "natural_questions", "narrativeqa", "2wikimultihopqa",
            "hotpotqa", "musique", "dureader", "popqa", "triviaqa_10m"]
GCS_PREFIX = "memory-layers-training/pareto_eval"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Force the gcloud account for gsutil calls to the GCS identity from .env, rather than whatever
# happens to be locally active. (An older note here claimed the bucket was listable ONLY by
# ra3440@columbia.edu — no longer true: rohunagrawal@gmail.com can list it too, and .env now
# points at gmail. Inherit rather than pin, so this can't drift from what the boxes use.)
GSENV = {**os.environ,
         "CLOUDSDK_CORE_ACCOUNT": os.environ.get("GCS_USER_EMAIL") or "rohunagrawal@gmail.com"}


def collect_gcs_method(method):
    """Return {dataset: binary_acc, ...} pulled from GCS per-dataset result JSONs via gsutil."""
    import subprocess
    out = {}
    base = f"gs://{GCS_PREFIX}/{method}"
    listing = subprocess.run(["gsutil", "ls", base + "/"], capture_output=True, text=True, env=GSENV)
    if listing.returncode != 0:
        return out
    present = {os.path.basename(l.strip()) for l in listing.stdout.splitlines() if l.strip().endswith(".json")}
    for ds in DATASETS:
        if f"{ds}.json" not in present:
            continue
        cat = subprocess.run(["gsutil", "cat", f"{base}/{ds}.json"], capture_output=True, text=True, env=GSENV)
        if cat.returncode != 0:
            continue
        d = json.loads(cat.stdout)
        acc = (d.get("metrics") or {}).get("llm_judge_accuracy")
        if acc is not None:
            out[ds] = acc
    return out


def collect_msa():
    p = os.path.join(ROOT, "results", "msa_eval_summary.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    return dict(d.get("llm_judge_accuracy_binary", {}))


def decode_tps(bench):
    """Steady-state decode tokens/s (batch), warmup batch dropped."""
    b = bench.get("batches", [])[1:] or bench.get("batches", [])
    tps = []
    for x in b:
        B = x.get("B", 1)
        if "decode_s" in x and x["decode_s"] > 0:                      # MSA-style
            tps.append(x["decode_steps"] * B / x["decode_s"])
        elif x.get("gen_e2e_s", 0) > x.get("prefill_s", 0):            # membed-style (lower bound)
            tps.append(x["max_new_tokens"] * B / (x["gen_e2e_s"] - x["prefill_s"]))
    return statistics.mean(tps) if tps else None


def collect_throughput():
    out = {}
    bdir = os.path.join(ROOT, "results", "membench")
    for method, fname in [("membed", "bench_membed.json"), ("msa", "bench_msa.json"),
                          ("rag", "bench_rag.json")]:
        p = os.path.join(bdir, fname)
        if os.path.exists(p):
            t = decode_tps(json.load(open(p)))
            if t:
                out[method] = round(t, 1)
    return out


def main():
    accs = {
        "membed": collect_gcs_method("membed"),
        "rag": collect_gcs_method("rag"),
        "ttt": collect_gcs_method("ttt"),  # oracle-doc reader (1.3B), NOT corpus-level
        "msa": collect_msa(),
    }
    tput = collect_throughput()

    summary = {}
    print(f"{'method':10s} {'n_evals':>7s} {'avg_binary_acc':>15s} {'decode_tok/s':>13s}")
    for m, per in accs.items():
        if not per:
            continue
        avg = statistics.mean(per.values())
        summary[m] = {"avg_binary_acc": round(avg, 4), "n_evals": len(per),
                      "per_dataset": {k: round(v, 4) for k, v in per.items()},
                      "decode_tokens_per_s": tput.get(m)}
        print(f"{m:10s} {len(per):>7d} {avg:>15.4f} {str(tput.get(m,'-')):>13s}")

    outp = os.path.join(ROOT, "results", "pareto_summary.json")
    json.dump(summary, open(outp, "w"), indent=2)
    print(f"\nwrote {outp}")

    # plot if we have any method with both axes
    plottable = {m: s for m, s in summary.items() if s.get("decode_tokens_per_s")}
    if plottable:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 5))
            for m, s in plottable.items():
                ax.scatter(s["decode_tokens_per_s"], s["avg_binary_acc"], s=120)
                ax.annotate(m, (s["decode_tokens_per_s"], s["avg_binary_acc"]),
                            textcoords="offset points", xytext=(8, 4))
            ax.set_xlabel("Decode throughput (tokens/s, batch)")
            ax.set_ylabel("Binary LLM-judge accuracy (avg over MSA evals)")
            ax.set_title("Pareto: throughput vs accuracy on MSA document-QA evals")
            ax.grid(True, alpha=0.3)
            pp = os.path.join(ROOT, "results", "figures", "pareto_plot.png")
            fig.tight_layout(); fig.savefig(pp, dpi=140)
            print(f"wrote {pp}")
        except Exception as e:
            print(f"plot skipped: {e}")
    else:
        print("no method has both throughput + acc yet; plot skipped")


if __name__ == "__main__":
    main()
