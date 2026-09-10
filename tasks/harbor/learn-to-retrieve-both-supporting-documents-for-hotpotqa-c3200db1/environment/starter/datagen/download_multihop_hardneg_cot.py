#!/usr/bin/env python3
"""
Stage ragrawal36/multihop_qa_sft-hard-neg-cot (question | answer | pos_doc_ids |
neg_doc_ids | think -- see datagen/join_multihop_cot.py for how it was built) into the
local parquet layout data/qa.py's HF_HUB_OFFLINE=1 branch expects:
$GROUND_HF_PARQUET/<hf_name with "/" -> "__">/train.parquet.

Uses the same doc corpus as the plain multihop_hard_neg_full recipe (multihop_doc_corpus.arrow,
staged by datagen/download_multihop_hardneg.py) -- pos_doc_ids/neg_doc_ids index into it
unchanged; only the rows-level table gained a `think` column.

Usage:
    HF_HUB_OFFLINE=0 uv run python datagen/download_multihop_hardneg_cot.py
"""
import argparse
import os
import sys
from pathlib import Path

REPO = "ragrawal36/multihop_qa_sft-hard-neg-cot"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=REPO)
    p.add_argument("--out-dir", default=os.environ.get("GROUND_HF_PARQUET",
                                                       os.path.expanduser("~/hf_parquet")))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    from huggingface_hub import hf_hub_download

    out_dir = Path(os.path.expanduser(args.out_dir)) / args.repo.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "train.parquet"
    if dest.exists() and not args.force:
        print(f"present, skipping: {dest}")
        return 0

    print(f"downloading {args.repo} train.parquet ...", flush=True)
    local = hf_hub_download(args.repo, "train.parquet", repo_type="dataset",
                             token=os.environ.get("HF_TOKEN"))
    import shutil
    tmp = dest.with_suffix(".parquet.tmp")
    shutil.copyfile(local, tmp)
    tmp.rename(dest)
    print(f"staged: {dest} ({dest.stat().st_size/1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
