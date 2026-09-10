"""Scale plot: memory-layer per-query cost vs corpus size (architectural efficiency).

Reads gs://memory-layers-training/pareto_eval/scale/membed_<N>.json (speed-bench outputs
at increasing max_docs) and plots steady-state per-query prefill latency + one-time corpus
encode time vs corpus size. Annotates the full-context feasibility wall (a vanilla
Qwen3-4B context tops out ~32k tokens — far below these corpora — so it cannot hold the
corpus at all; the memory-layer answers at ~constant per-query prefill).
"""
import json, os, subprocess, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# gcloud account for gsutil calls: inherit the GCS identity from .env (GCS_USER_EMAIL).
GSENV = {**os.environ,
         "CLOUDSDK_CORE_ACCOUNT": os.environ.get("GCS_USER_EMAIL") or "rohunagrawal@gmail.com"}
BASE = "gs://memory-layers-training/pareto_eval/scale"
FULL_CTX_TOKENS = 32768  # Qwen3-4B practical context cap

ls = subprocess.run(["gsutil", "ls", BASE + "/"], capture_output=True, text=True, env=GSENV)
files = [l.strip() for l in ls.stdout.splitlines() if l.strip().endswith(".json")]

pts = []
for f in files:
    cat = subprocess.run(["gsutil", "cat", f], capture_output=True, text=True, env=GSENV)
    d = json.loads(cat.stdout)
    b = d.get("batches", [])[1:] or d.get("batches", [])
    pf = statistics.mean([x["prefill_s"] / x["B"] for x in b]) if b else float("nan")
    pts.append({
        "docs": d.get("num_corpus_chunks"),                  # chunks ~ docs*chunks_per_doc
        "doc_tokens": d.get("doc_tokens_in_memory"),
        "prefill_s_per_query": pf,
        "encode_s": d.get("encode_corpus_s"),
    })
pts.sort(key=lambda p: p["doc_tokens"] or 0)
print(json.dumps(pts, indent=2))

if pts:
    xs = [p["doc_tokens"] for p in pts]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, [p["prefill_s_per_query"] for p in pts], "o-", label="Memory-layer: prefill s/query (steady)")
    ax.plot(xs, [p["encode_s"] for p in pts], "s--", color="gray", label="Memory-layer: one-time corpus encode (s)")
    ax.axvline(FULL_CTX_TOKENS, color="red", ls=":", lw=2)
    ax.text(FULL_CTX_TOKENS, ax.get_ylim()[1]*0.5,
            " full-context Qwen3-4B\n cap (~32k tok)\n → cannot hold corpus",
            color="red", fontsize=8, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("Corpus size (doc tokens in memory, log scale)")
    ax.set_ylabel("Seconds")
    ax.set_title("Memory-layer: sub-linear per-query cost over corpora far exceeding any context window")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    out = os.path.join(ROOT, "results", "figures", "scale_cost.png")
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print(f"wrote {out}")
