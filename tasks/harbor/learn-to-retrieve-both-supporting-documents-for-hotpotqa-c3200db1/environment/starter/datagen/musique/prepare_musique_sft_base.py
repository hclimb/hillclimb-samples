#!/usr/bin/env python3
"""
MuSiQue → SFT base dataset (stage 1 of 2).

Flattens MuSiQue's 20-paragraph rows into the framework's {pos_doc, neg_doc}
shape and writes sharded parquets to {HF_USERNAME}/musique-sft-base. Stage 2
(generate_musique_sft.py) reads those shards and adds the CoT.

Gold-labelled negatives: MuSiQue marks each paragraph is_supporting, so the
~17.7 non-supporting paragraphs per question are hard negatives by human
annotation rather than by embedding-mined similarity. Unlike the vm2825/*
sources there is no neg_scores column and no threshold to apply.

TRAIN SPLIT ONLY, deliberately. MuSiQue validation is a live eval benchmark
here (configs/eval_set/{corpus_evals,msa_evals,hard_neg_think_c512}.yaml and
7 configs/eval/tasks/gen_*_musique_*.yaml). Emitting only train makes it
impossible for the finetuning artifact to carry eval data. Pass
--split validation only if you know why you want it.

Usage:
    uv run python datagen/musique/prepare_musique_sft_base.py
    uv run python datagen/musique/prepare_musique_sft_base.py --dry-run
"""

import argparse
import json
import logging
import os
import random
import sys

import dotenv

dotenv.load_dotenv()

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ─── constants ────────────────────────────────────────────────────────────────
SOURCE_REPO = "dgslibisey/MuSiQue"
OUTPUT_REPO = f"{os.environ.get('HF_USERNAME', '')}/musique-sft-base"

SHARD_ROWS = 2000  # 19,938 train rows → 10 shards; stage-2 resumes at shard granularity
LOCAL_DIR = "./temp_musique_base"

SCHEMA = pa.schema([
    ("id",             pa.string()),
    ("question",       pa.string()),
    ("answer",         pa.string()),
    ("answer_aliases", pa.list_(pa.string())),
    ("pos_doc",        pa.list_(pa.string())),
    ("neg_doc",        pa.list_(pa.string())),
    ("hop_type",       pa.string()),
    ("decomposition",  pa.string()),  # JSON: [{question, answer, paragraph_support_idx}]
])


# ─── row processing ───────────────────────────────────────────────────────────
def format_paragraph(p: dict) -> str:
    """Match the title formatting already used by prepare_musique_supporting_list.py."""
    return f"**{p['title']}**\n{p['paragraph_text']}"


def process_row(row: dict) -> dict | None:
    if not row.get("answerable", False):
        return None

    pos_doc, neg_doc = [], []
    for p in row["paragraphs"]:
        (pos_doc if p.get("is_supporting") else neg_doc).append(format_paragraph(p))

    # A row with no gold paragraph can't supervise retrieval; drop rather than
    # fall back to "all paragraphs are positive" (which would poison pos_doc_mask).
    if not pos_doc:
        return None

    decomposition = [
        {
            "question": s["question"],
            "answer": s["answer"],
            "paragraph_support_idx": s.get("paragraph_support_idx"),
        }
        for s in row.get("question_decomposition", [])
    ]

    return {
        "id":             row["id"],
        "question":       row["question"],
        "answer":         row["answer"],
        "answer_aliases": list(row.get("answer_aliases") or []),
        "pos_doc":        pos_doc,
        "neg_doc":        neg_doc,
        # ids look like "2hop__12345_678" / "3hop1__..." / "4hop2__..."
        "hop_type":       row["id"].split("__")[0],
        "decomposition":  json.dumps(decomposition),
    }


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--output-repo", default=OUTPUT_REPO)
    parser.add_argument("--shard-rows", type=int, default=SHARD_ROWS)
    parser.add_argument("--shuffle-seed", type=int, default=42,
                        help="Seed for the pre-shard shuffle; change only to reshuffle deliberately")
    parser.add_argument("--dry-run", action="store_true", help="Skip HF uploads")
    args = parser.parse_args()

    if args.split == "validation":
        logging.warning(
            "Building from the VALIDATION split. This is the held-out MuSiQue eval set; "
            "do not train on the result."
        )

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token and not args.dry_run:
        raise ValueError("Set HF_TOKEN or HUGGINGFACE_TOKEN in your environment.")
    if not args.output_repo.strip("/") or args.output_repo.startswith("/"):
        raise ValueError("Set HF_USERNAME in .env, or pass --output-repo explicitly.")

    logging.info(f"Loading {SOURCE_REPO} split={args.split}...")
    ds = load_dataset(SOURCE_REPO, split=args.split)
    logging.info(f"  {len(ds)} raw rows.")

    records, skipped = [], 0
    for row in ds:
        rec = process_row(row)
        if rec is None:
            skipped += 1
        else:
            records.append(rec)

    logging.info(f"  {len(records)} kept | {skipped} skipped (unanswerable / no gold paragraph)")

    if not records:
        # Bail here rather than let the stats lines below die on min() of an empty
        # sequence, which buries the real cause.
        raise SystemExit(
            f"ERROR: every row was filtered out of {SOURCE_REPO}:{args.split}. "
            "Check that the source still carries `answerable` and `is_supporting`."
        )

    from collections import Counter
    hops = Counter(r["hop_type"] for r in records)
    n_pos = [len(r["pos_doc"]) for r in records]
    n_neg = [len(r["neg_doc"]) for r in records]
    logging.info(f"  hop types: {dict(sorted(hops.items()))}")
    logging.info(f"  pos_doc/row: min={min(n_pos)} mean={sum(n_pos)/len(n_pos):.1f} max={max(n_pos)}")
    logging.info(f"  neg_doc/row: min={min(n_neg)} mean={sum(n_neg)/len(n_neg):.1f} max={max(n_neg)}")

    sample = records[0]
    logging.info("Sample row:")
    logging.info(f"  id:       {sample['id']}  ({sample['hop_type']})")
    logging.info(f"  question: {sample['question']}")
    logging.info(f"  answer:   {sample['answer']}")
    logging.info(f"  pos_doc:  {len(sample['pos_doc'])}, neg_doc: {len(sample['neg_doc'])}")
    logging.info(f"  decomp:   {sample['decomposition'][:200]}...")

    # Shuffle before sharding. MuSiQue's train split is ORDERED BY HOP TYPE — shards 0-6 come
    # out pure 2hop, 7-8 pure 3hop, 9 the 4hop tail. Left unshuffled that means: a partial or
    # interrupted stage-2 run silently produces a hop-skewed dataset, per-shard yields are not
    # comparable to each other, and the harder 3/4-hop questions (the ones a tight token budget
    # is most likely to drop) are quarantined in the last shards instead of spread across them.
    random.Random(args.shuffle_seed).shuffle(records)
    logging.info(f"  shuffled with seed {args.shuffle_seed} so every shard is representative")

    os.makedirs(LOCAL_DIR, exist_ok=True)
    api = HfApi(token=hf_token) if not args.dry_run else None
    if api:
        api.create_repo(args.output_repo, repo_type="dataset", exist_ok=True)

    n_shards = (len(records) + args.shard_rows - 1) // args.shard_rows
    for i in range(n_shards):
        chunk = records[i * args.shard_rows : (i + 1) * args.shard_rows]
        name = f"data/{args.split}-{i:05d}.parquet"
        local = os.path.join(LOCAL_DIR, os.path.basename(name))
        pq.write_table(pa.Table.from_pylist(chunk, schema=SCHEMA), local, compression="snappy")
        mix = Counter(r["hop_type"] for r in chunk)
        logging.info(f"[{i+1}/{n_shards}] wrote {len(chunk)} rows → {name}  hops={dict(sorted(mix.items()))}")

        if api:
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=name,
                repo_id=args.output_repo,
                repo_type="dataset",
                commit_message=f"Add MuSiQue SFT base: {name}",
                token=hf_token,
            )
            os.remove(local)

    if args.dry_run:
        logging.info(f"Dry run — {n_shards} shards left in {LOCAL_DIR}/")
    else:
        logging.info(f"Done → https://huggingface.co/datasets/{args.output_repo}")


if __name__ == "__main__":
    sys.exit(main())
