#!/usr/bin/env python3
"""
Join `think` (CoT) text from ragrawal36/multihop_qa_sft into the mined
mihir-1999/multihop_qa_sft-hard-neg-train rows (question | answer | pos_doc_ids |
neg_doc_ids), matched by exact question-text, and publish the joined table as a new
HF dataset: ragrawal36/multihop_qa_sft-hard-neg-cot.

Why a text-match join, not a shared id: mihir-1999's mined rows/negatives repos have no
column in common with ragrawal36/multihop_qa_sft other than the question string itself --
the two were built by different pipelines from a common origin. Row counts are close
(1,341,045 mined rows vs ~176*7681=~1,352,000 in ragrawal36's train split), consistent
with "same underlying question set," but that's a hint, not a guarantee -- this script
computes and reports the ACTUAL match rate rather than assuming it.

Reads the mihir-1999 side from the box's already-staged local parquet
($GROUND_HF_PARQUET/mihir-1999__multihop_qa_sft-hard-neg-train/train.parquet) --
run datagen/download_multihop_hardneg.py first if that's not present. Downloads
ragrawal36/multihop_qa_sft's train shards fresh (question+think columns only, after
download -- the parquet files themselves must be fetched whole, HF has no column-projected
partial download).

Usage:
    HF_HUB_OFFLINE=0 uv run python datagen/join_multihop_cot.py
    HF_HUB_OFFLINE=0 uv run python datagen/join_multihop_cot.py --dry-run   # join + report, no upload
"""
import argparse
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SOURCE_REPO = "ragrawal36/multihop_qa_sft"
MIHIR_TRAIN_DIRNAME = "mihir-1999__multihop_qa_sft-hard-neg-train"
TARGET_REPO = "ragrawal36/multihop_qa_sft-hard-neg-cot"


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=os.environ.get("GROUND_HF_PARQUET",
                                                       os.path.expanduser("~/hf_parquet")))
    p.add_argument("--target-repo", default=TARGET_REPO)
    p.add_argument("--dry-run", action="store_true", help="Join + report match stats, skip upload.")
    p.add_argument("--force-download", action="store_true",
                   help="Re-download ragrawal36 shards even if already cached locally.")
    return p


def main():
    args = build_parser().parse_args()
    from huggingface_hub import HfApi, hf_hub_download, create_repo

    hf_token = os.environ.get("HF_TOKEN")
    out_dir = Path(os.path.expanduser(args.out_dir))
    mihir_train_parquet = out_dir / MIHIR_TRAIN_DIRNAME / "train.parquet"
    if not mihir_train_parquet.exists():
        print(f"ERROR: {mihir_train_parquet} not found -- run "
              f"datagen/download_multihop_hardneg.py first.", file=sys.stderr)
        return 1

    print(f"[1/5] loading mihir-1999 rows from {mihir_train_parquet} ...", flush=True)
    mihir_tbl = pq.read_table(mihir_train_parquet)
    print(f"  {mihir_tbl.num_rows:,} rows, columns={mihir_tbl.column_names}")

    print(f"[2/5] listing {SOURCE_REPO} train shards ...", flush=True)
    api = HfApi(token=hf_token)
    info = api.dataset_info(SOURCE_REPO)
    train_files = sorted(s.rfilename for s in info.siblings if s.rfilename.startswith("train/"))
    print(f"  {len(train_files)} shards")

    print(f"[3/5] downloading + building question->think map ...", flush=True)
    think_by_question = {}
    n_dup_questions = 0
    n_dup_conflicting_think = 0
    for i, rel in enumerate(train_files):
        local = hf_hub_download(SOURCE_REPO, rel, repo_type="dataset", token=hf_token,
                                 force_download=args.force_download)
        shard = pq.read_table(local, columns=["question", "think"])
        qs = shard.column("question").to_pylist()
        ts = shard.column("think").to_pylist()
        for q, t in zip(qs, ts):
            if q in think_by_question:
                n_dup_questions += 1
                if think_by_question[q] != t:
                    n_dup_conflicting_think += 1
                continue  # keep first occurrence
            think_by_question[q] = t
        if (i + 1) % 20 == 0 or i + 1 == len(train_files):
            print(f"  {i+1}/{len(train_files)} shards, {len(think_by_question):,} unique questions so far",
                  flush=True)

    print(f"[4/5] joining onto mihir-1999 rows ...", flush=True)
    mihir_questions = mihir_tbl.column("question").to_pylist()
    joined_think = [think_by_question.get(q) for q in mihir_questions]
    n_matched = sum(t is not None for t in joined_think)
    n_total = len(mihir_questions)
    print(f"  matched {n_matched:,}/{n_total:,} ({100*n_matched/n_total:.2f}%) rows to a `think` value")
    print(f"  {n_dup_questions:,} duplicate questions seen in {SOURCE_REPO} "
          f"({n_dup_conflicting_think:,} with conflicting `think` text -- kept first occurrence)")
    if n_matched / n_total < 0.95:
        print(f"  WARNING: match rate below 95% -- inspect before trusting this join for training.",
              file=sys.stderr)

    out_tbl = mihir_tbl.append_column("think", pa.array(joined_think, type=pa.string()))

    out_path = out_dir / "multihop_qa_sft_hard_neg_cot_train.parquet"
    tmp = out_path.with_suffix(".parquet.tmp")
    pq.write_table(out_tbl, tmp)
    tmp.rename(out_path)
    print(f"  wrote {out_path} ({out_path.stat().st_size/1e9:.2f} GB)")

    if args.dry_run:
        print(f"[5/5] --dry-run: skipping upload to {args.target_repo}")
        return 0

    print(f"[5/5] creating/uploading {args.target_repo} ...", flush=True)
    create_repo(args.target_repo, repo_type="dataset", token=hf_token, exist_ok=True,
                private=True)
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo="train.parquet",
        repo_id=args.target_repo,
        repo_type="dataset",
        token=hf_token,
    )
    print(f"  uploaded -> https://huggingface.co/datasets/{args.target_repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
