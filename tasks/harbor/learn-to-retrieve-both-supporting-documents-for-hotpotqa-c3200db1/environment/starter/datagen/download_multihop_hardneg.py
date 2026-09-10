#!/usr/bin/env python3
"""
Stage the multihop hard-negative data on a TPU box for training.

Produces two artifacts:

  1. $GROUND_HF_PARQUET/mihir-1999__multihop_qa_sft-hard-neg-train/train.parquet
       question | answer | pos_doc_ids | neg_doc_ids
     Joined on row_id from the two published datasets (questions live in the corpus
     repo's `rows` config, negatives in the hard-negatives repo). data/qa.py's offline
     branch picks this path up automatically when HF_HUB_OFFLINE=1.

  2. $GROUND_HF_PARQUET/multihop_doc_corpus.arrow
     The document corpus as an **Arrow IPC file**, row i == doc_id i (asserted).
     Arrow IPC rather than parquet because it is uncompressed and memory-mappable, so
     every grain worker shares one physical ~0.9 GB copy through the OS page cache
     instead of decompressing its own.

Run ONCE on the box before training. Training then reads local disk only — never the
Hub. That is not an optimization: wiki/data/hf-rate-limits.md documents 16 workers x 4
sources blowing HF's 1000-req/5-min quota during pipeline build, which qa.py retries
into a livelock (12 rebuild cycles, 47 min, still step 0).

Usage:
    python datagen/download_multihop_hardneg.py
    python datagen/download_multihop_hardneg.py --max-rows 5000    # smoke test
"""

import argparse
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

CORPUS_REPO = "mihir-1999/multihop_qa_sft-doc-corpus"
NEG_REPO = "mihir-1999/multihop_qa_sft-hard-negatives"
TRAIN_DIRNAME = "mihir-1999__multihop_qa_sft-hard-neg-train"
CORPUS_ARROW = "multihop_doc_corpus.arrow"


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-repo", default=CORPUS_REPO)
    p.add_argument("--neg-repo", default=NEG_REPO)
    p.add_argument("--out-dir", default=os.environ.get("GROUND_HF_PARQUET",
                                                       os.path.expanduser("~/hf_parquet")))
    p.add_argument("--max-rows", type=int, default=None, help="Cap rows (smoke test).")
    p.add_argument("--force", action="store_true", help="Rebuild even if outputs exist.")
    return p


def main():
    args = build_parser().parse_args()
    from datasets import load_dataset

    out_dir = Path(os.path.expanduser(args.out_dir))
    train_dir = out_dir / TRAIN_DIRNAME
    train_dir.mkdir(parents=True, exist_ok=True)
    corpus_arrow = out_dir / CORPUS_ARROW
    train_parquet = train_dir / "train.parquet"

    # ---- corpus -> Arrow IPC (row i == doc_id i) ----------------------------------
    if corpus_arrow.exists() and not args.force:
        print(f"corpus present, skipping: {corpus_arrow}")
    else:
        print(f"downloading corpus {args.corpus_repo} [corpus] ...", flush=True)
        corpus = load_dataset(args.corpus_repo, "corpus", split="train")
        tbl = corpus.data.table if hasattr(corpus.data, "table") else corpus.data
        tbl = pa.table({"doc_id": tbl.column("doc_id"), "text": tbl.column("text")})

        doc_ids = tbl.column("doc_id").to_numpy()
        n = len(doc_ids)
        # Row order IS the lookup: the normalizer indexes this table by doc_id directly.
        # If the order ever drifts, every resolved document would be the wrong one, so
        # fail loudly here rather than train on silently mismatched text.
        import numpy as np
        if not np.array_equal(doc_ids, np.arange(n)):
            print(f"ERROR: corpus rows are not doc_id 0..N-1 in order "
                  f"(min={doc_ids.min()}, max={doc_ids.max()}, n={n})", file=sys.stderr)
            return 1

        tmp = corpus_arrow.with_suffix(".arrow.tmp")
        with pa.OSFile(str(tmp), "wb") as sink:
            # No compression: the point is a mappable file the OS can share across workers.
            with pa.ipc.new_file(sink, tbl.schema) as writer:
                writer.write_table(tbl)
        tmp.rename(corpus_arrow)
        print(f"corpus: {n:,} docs -> {corpus_arrow} "
              f"({corpus_arrow.stat().st_size/1e9:.2f} GB, memory-mappable)")

    # ---- rows x negatives -> one training parquet ---------------------------------
    if train_parquet.exists() and not args.force:
        print(f"train parquet present, skipping: {train_parquet}")
    else:
        print(f"downloading rows {args.corpus_repo} [rows] ...", flush=True)
        rows = load_dataset(args.corpus_repo, "rows", split="train")
        print(f"downloading negatives {args.neg_repo} ...", flush=True)
        negs = load_dataset(args.neg_repo, split="train")

        rt = rows.data.table if hasattr(rows.data, "table") else rows.data
        nt = negs.data.table if hasattr(negs.data, "table") else negs.data

        import numpy as np
        r_ids = rt.column("row_id").to_numpy()
        n_ids = nt.column("row_id").to_numpy()
        # Both were written in row_id order by the mining pipeline; a positional join is
        # only valid if that still holds, so check instead of assuming.
        if len(r_ids) != len(n_ids) or not np.array_equal(r_ids, n_ids):
            print("ERROR: row_id ordering differs between the rows and negatives tables; "
                  "a positional join would mismatch questions to negatives.", file=sys.stderr)
            return 1

        table = pa.table({
            "question":    rt.column("question"),
            "answer":      rt.column("answer"),
            "pos_doc_ids": rt.column("pos_doc_ids"),
            "neg_doc_ids": nt.column("neg_doc_ids"),
        })
        if args.max_rows is not None:
            table = table.slice(0, args.max_rows)

        tmp = train_parquet.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp)
        tmp.rename(train_parquet)
        print(f"train rows: {table.num_rows:,} -> {train_parquet} "
              f"({train_parquet.stat().st_size/1e9:.2f} GB)")

    print("\nStaged. Train with:")
    print(f"  HF_HUB_OFFLINE=1 GROUND_HF_PARQUET={out_dir} \\")
    print(f"  MULTIHOP_CORPUS={corpus_arrow} uv run train.py ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
