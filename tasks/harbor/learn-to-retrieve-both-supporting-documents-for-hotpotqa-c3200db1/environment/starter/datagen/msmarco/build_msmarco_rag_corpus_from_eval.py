#!/usr/bin/env python3
"""Build the EXACT corpus + query set a hybrid eval run used, for the RAG baseline.

Why not datagen/musique/build_musique_c512_rag_corpus.py's approach (HF rows[:N])? The eval's
queries are the data pipeline's first N SURVIVING rows, not the HF prefix — on msmarco HF row 0
is dropped, so rows[:128] is a shifted query set with a different gold union, and the RAG
baseline would search a subtly different haystack (measured 2026-07-22: eval != rows[:128]).

This script instead reads an eval result JSON (any hybrid arm; samples carry the prompts),
extracts the queries in eval order, re-derives their gold ids from the QA repo (same content
join as the evaluator), rebuilds the evaluator's inject_query_gold selection (gold chunks in
corpus order, then corpus-order distractors to --target-docs), and pushes:
  --out-corpus-repo   {text, id}                       — the matched haystack
  --out-queries-repo  {question, answer, pos_doc_ids}  — the matched 128 queries, eval order

    uv run python datagen/msmarco/build_msmarco_rag_corpus_from_eval.py \
        --eval-json <local path> --target-docs 10000 [--dry-run]
"""
import argparse
import json
import logging
import os
import re
import sys

import dotenv

dotenv.load_dotenv()

from datasets import Dataset, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_USER = os.environ.get("HF_USERNAME", "")


def extract_question(prompt):
    m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, re.S)
    if not m:
        sys.exit(f"ERROR: no user turn found in prompt: {prompt[:200]!r}")
    return " ".join(m.group(1).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", required=True,
                    help="local path to a hybrid eval result JSON (samples carry prompts)")
    ap.add_argument("--target-docs", type=int, required=True)
    ap.add_argument("--qa-repo", default=f"{_USER}/msa-msmarco-v1-qa-with-ids")
    ap.add_argument("--doc-repo", default=f"{_USER}/msa-msmarco-v1-docs-with-ids")
    ap.add_argument("--out-corpus-repo", default=None)
    ap.add_argument("--out-queries-repo", default=None)
    # Parity with the memory bank, and a hard requirement for long-doc corpora: the hybrid
    # embeds only the first N tokens of each doc (doc_chunk_seq_len), and the RAG generator's
    # vLLM has max_model_len 4096 — k=10 whole triviaqa_10m docs (~800 tok each) overflow it.
    ap.add_argument("--max-doc-tokens", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not _USER:
        sys.exit("set HF_USERNAME in .env")
    out_corpus = args.out_corpus_repo or f"{_USER}/msmarco-c{args.target_docs}-rag-corpus-evalmatched"
    out_queries = args.out_queries_repo or f"{_USER}/msmarco-c{args.target_docs}-eval-queries"

    # 1. Queries, in eval order, from the eval run's own record.
    ev = json.load(open(args.eval_json))
    questions = [extract_question(s["prompt"]) for s in ev["samples"]]
    logging.info(f"{len(questions)} queries from {args.eval_json}")

    # 2. Re-derive gold ids by content join against the QA repo (first exact normalized match —
    #    the evaluator's longest-substring join reduces to this when the full question is known).
    qa = load_dataset(args.qa_repo, split="train")
    by_q = {}
    for row in qa:
        by_q.setdefault(" ".join(str(row["question"]).split()), row)
    rows, gold_ids = [], set()
    for q in questions:
        row = by_q.get(q)
        if row is None:
            sys.exit(f"ERROR: eval question not found in {args.qa_repo}: {q!r}")
        golds = [int(v) for v in (row.get("pos_doc_ids") or []) if int(v) >= 0]
        gold_ids.update(golds)
        rows.append({"question": row["question"], "answer": row.get("answer", ""),
                     "pos_doc_ids": golds})
    logging.info(f"matched all {len(rows)} queries; gold union = {len(gold_ids)} doc ids")

    # 3. The evaluator's inject_query_gold selection: gold chunks in corpus order, then
    #    corpus-order distractors to target_docs.
    docs = load_dataset(args.doc_repo, split="train")
    trunc = None
    if args.max_doc_tokens:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

        def trunc(t):
            ids = tok(t, add_special_tokens=False)["input_ids"]
            return t if len(ids) <= args.max_doc_tokens else tok.decode(ids[:args.max_doc_tokens])
    gold, other = [], []
    for row in docs:
        text = trunc(row["text"]) if trunc else row["text"]
        (gold if int(row["id"]) in gold_ids else other).append(
            {"text": text, "id": int(row["id"])})
    n_other = max(0, args.target_docs - len(gold))
    corpus = gold + other[:n_other]
    missing = gold_ids - {d["id"] for d in gold}
    if missing:
        sys.exit(f"ERROR: {len(missing)} gold ids absent from {args.doc_repo}: {sorted(missing)[:10]}")
    logging.info(f"corpus (scanned {len(docs)}): {len(gold)} gold + {min(n_other, len(other))} "
                 f"distractor = {len(corpus)} docs (target={args.target_docs})")

    if args.dry_run:
        logging.info("dry run — not uploading")
        return
    Dataset.from_list(corpus).push_to_hub(out_corpus, token=token, private=False)
    Dataset.from_list(rows).push_to_hub(out_queries, token=token, private=False)
    logging.info(f"Done -> {out_corpus} ({len(corpus)} docs), {out_queries} ({len(rows)} queries)")


if __name__ == "__main__":
    main()
