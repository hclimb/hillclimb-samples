#!/usr/bin/env python3
"""Build the EXACT 512-doc corpus the memory model was evaluated on, so the RAG baseline
retrieves from the same haystack.

Why this is necessary. `gen_large_mem` with `inject_query_gold: true` does NOT take the first N
documents: it scans the full corpus, keeps every gold document for the eval queries, then fills
to `target_docs` with distractors in corpus order (evals/gen_large_mem.py:377-416). The RAG
retriever has no such notion — `single_embedding_retrieval.py --max_docs N` simply stops reading
after N documents. So pointing RAG at `--max_docs 512` gives it 512 essentially arbitrary
documents that mostly do NOT contain the answers, and it scores near zero. That is not a
baseline, it is a broken comparison that would make the memory model look good for the wrong
reason.

This reproduces the selection deterministically — same query order, same gold set, same
distractor fill — and publishes it as a plain `{text, id}` dataset both systems can read.

    uv run python datagen/musique/build_musique_c512_rag_corpus.py
    uv run python datagen/musique/build_musique_c512_rag_corpus.py --dry-run
"""
import argparse
import logging
import os
import sys

import dotenv

dotenv.load_dotenv()

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_USER = os.environ.get("HF_USERNAME", "")
QA_REPO = f"{_USER}/msa-musique-qa-with-ids"
DOC_REPO = f"{_USER}/msa-musique-docs-with-ids"
OUT_REPO = f"{_USER}/musique-c512-rag-corpus"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-queries", type=int, default=128,
                    help="Must match the memory eval's eval.num_samples")
    ap.add_argument("--target-docs", type=int, default=512,
                    help="Must match the memory eval's doc_dataset.target_docs")
    ap.add_argument("--out-repo", default=OUT_REPO)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not _USER:
        sys.exit("set HF_USERNAME in .env")

    # 1. Gold ids from the FIRST num_queries rows, in dataset order — the eval's dataset config
    #    sets shuffle: false, so its 128 samples are this same prefix.
    logging.info(f"Loading {QA_REPO} (first {args.num_queries} queries, in order)...")
    qa = load_dataset(QA_REPO, split=f"train[:{args.num_queries}]")
    gold_ids = set()
    for row in qa:
        for v in (row.get("pos_doc_ids") or []):
            if int(v) >= 0:
                gold_ids.add(int(v))
    logging.info(f"  {len(gold_ids)} gold doc ids from {len(qa)} queries")

    # 2. Partition the full corpus in order: gold first, then distractors to fill.
    logging.info(f"Scanning {DOC_REPO}...")
    docs = load_dataset(DOC_REPO, split="train")
    gold, other = [], []
    for row in docs:
        (gold if int(row["id"]) in gold_ids else other).append(
            {"text": row["text"], "id": int(row["id"])}
        )
    n_other = max(0, args.target_docs - len(gold))
    corpus = gold + other[:n_other]
    logging.info(
        f"  corpus (scanned {len(docs)}): {len(gold)} gold + {min(n_other, len(other))} "
        f"distractor = {len(corpus)} docs (target={args.target_docs})"
    )
    missing = gold_ids - {d["id"] for d in gold}
    if missing:
        # Every gold must be present or the baseline is unfairly handicapped — the exact failure
        # this script exists to prevent. Fail loudly rather than publish a corpus with holes.
        sys.exit(f"ERROR: {len(missing)} gold ids not found in {DOC_REPO}: {sorted(missing)[:10]}")
    if len(corpus) < args.target_docs:
        logging.warning(
            f"only {len(corpus)} docs available (< target {args.target_docs}); "
            "the corpus is smaller than the memory eval's — comparison is still fair but note it"
        )

    ds = Dataset.from_list(corpus)
    logging.info(f"columns: {ds.column_names}  rows: {len(ds)}")
    logging.info(f"sample: id={corpus[0]['id']} text={corpus[0]['text'][:100]!r}")

    if args.dry_run:
        logging.info("dry run — not uploading")
        return
    ds.push_to_hub(args.out_repo, token=token, private=False)
    logging.info(f"Done -> https://huggingface.co/datasets/{args.out_repo}")


if __name__ == "__main__":
    main()
