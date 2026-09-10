#!/usr/bin/env python3
"""Convert a rag_only OUT_DIR into a hybrid-eval-style {metrics, samples} JSON.

The RAG pipeline saves per-stage files (retrieval_results.json, generated.json) but no
per-sample judge verdicts and not in the schema scripts/analysis/qa_compare_viewer.html
reads. This joins the stages, attaches the retrieved docs (rag_docs / rag_doc_ids with
GLOBAL corpus ids — the pipeline's doc_index is a corpus ROW index), re-derives gold
ids/texts, runs the real evals.metrics.llm_judge llm_judge_accuracy for per-sample
verdicts (same protocol as every memory arm), and writes a viewer-ready JSON.

Box-side (needs the venv + a TPU for the vLLM judge):
    RAG_OUT_DIR=~/rag_msmarco_c10000_top10 QUERIES_REPO=... CORPUS_REPO=... \
      DEST=~/msmarco_c10000_rag_top10_samples.json uv run python scripts/misc/rag_to_viewer_json.py
"""
import glob
import json
import os
import sys

import dotenv

dotenv.load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datasets import load_dataset

USER = os.environ.get("HF_USERNAME", "")
OUT_DIR = os.path.expanduser(os.environ.get("RAG_OUT_DIR", "~/rag_msmarco_c10000_top10"))
QUERIES_REPO = os.environ.get("QUERIES_REPO", f"{USER}/msmarco-c10000-eval-queries")
CORPUS_REPO = os.environ.get("CORPUS_REPO", f"{USER}/msmarco-c10000-rag-corpus-evalmatched")
DEST = os.path.expanduser(os.environ.get("DEST", "~/rag_viewer.json"))
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-4B")
TP = int(os.environ.get("TP", 4))


def norm(s):
    return " ".join(str(s).split())


def load_stage(name):
    hits = glob.glob(os.path.join(OUT_DIR, "rag", "*", name))
    if len(hits) != 1:
        sys.exit(f"ERROR: expected exactly one {name} under {OUT_DIR}/rag/*/, got {hits}")
    return json.load(open(hits[0]))


def main():
    retr = load_stage("retrieval_results.json")
    gen = load_stage("generated.json")
    summary = json.load(open(os.path.join(OUT_DIR, "rag_only_summary.json")))
    assert len(retr) == len(gen), (len(retr), len(gen))

    corpus = load_dataset(CORPUS_REPO, split="train")
    row_to_id = [int(r["id"]) for r in corpus]
    id_to_text = {int(r["id"]): r["text"] for r in corpus}
    queries = load_dataset(QUERIES_REPO, split="train")
    golds_by_q = {norm(r["question"]): [int(v) for v in (r["pos_doc_ids"] or [])] for r in queries}

    samples = []
    for i, (r, g) in enumerate(zip(retr, gen)):
        q = norm(r["query"])
        assert q == norm(g["query"]), f"row {i}: stage order mismatch"
        golds = golds_by_q.get(q)
        if golds is None:
            sys.exit(f"ERROR: query not in {QUERIES_REPO}: {q!r}")
        ret_ids = [row_to_id[int(d["doc_index"])] for d in r["retrieved"]]
        samples.append({
            "prompt": f"<|im_start|>user\n{r['query']}<|im_end|>",
            "generated": g["answer"],
            "thinking": "",
            "generated_answer": g["answer"],
            "ground_truth": g["ground_truth"],
            "doc": " || ".join(id_to_text[d] for d in golds if d in id_to_text),
            "rag_doc_ids": ret_ids,
            "rag_docs": [d["document"] for d in r["retrieved"]],
            "rag_all_golds_covered": set(golds) <= set(ret_ids),
            "n_gold_docs": len(golds),
        })

    from evals.metrics.llm_judge import llm_judge_accuracy, llm_judge_score
    scores, outputs = llm_judge_accuracy(samples, model_id=JUDGE_MODEL, tensor_parallel_size=TP)
    for s, sc, out in zip(samples, scores, outputs):
        s["llm_judge_accuracy"] = sc
        s["llm_judge_accuracy_output"] = out
    s5, o5 = llm_judge_score(samples, model_id=JUDGE_MODEL, tensor_parallel_size=TP)
    for s, sc, out in zip(samples, s5, o5):
        s["llm_judge_score"] = sc
        s["llm_judge_score_output"] = out

    metrics = dict(summary.get("metrics", {}))
    metrics["generated_count"] = len(samples)
    metrics["llm_judge_accuracy"] = sum(scores) / len(scores)
    valid5 = [x for x in s5 if x is not None]
    if valid5:
        metrics["llm_judge_score"] = sum(valid5) / len(valid5)
    json.dump({"metrics": metrics, "samples": samples}, open(DEST, "w"), indent=2)
    print(f"[rag_to_viewer] {len(samples)} samples, judge={metrics['llm_judge_accuracy']:.4f} "
          f"(pipeline rag_accuracy={metrics.get('rag_accuracy')}) -> {DEST}")


if __name__ == "__main__":
    main()
