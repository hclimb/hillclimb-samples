#!/usr/bin/env python3
"""
Load sriragt/fanoutqa, split pos_doc by '<|doc_seperator|>', and upload
individual documents as rows with a single 'pos_doc' column to HuggingFace.

The output dataset is directly compatible with data/documents.py DocumentsDataset
(column='pos_doc').

Usage:
    HF_TOKEN=hf_... python datagen/create_fanoutqa_docs_dataset.py \
        --output-repo ragrawal36/fanoutqa-docs
"""

import argparse
import logging
import os

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SOURCE_REPO = "ragrawal36/fanoutqa"
DOC_SEPARATOR = "<|doc_seperator|>"
MIN_CHARS = 10
PARQUET_ROWS = 50_000
UPLOAD_THRESHOLD = 512 * 1024 * 1024  # 512 MB

SCHEMA = pa.schema([
    ("pos_doc", pa.string()),
])


def get_dir_size(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    return sum(
        os.path.getsize(os.path.join(path, f))
        for f in os.listdir(path)
        if f.endswith(".parquet")
    )


def upload_and_clear(api: HfApi, local_dir: str, repo: str, upload_count: int):
    logging.info(f"Uploading batch {upload_count} to {repo}...")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo,
        repo_type="dataset",
        allow_patterns="*.parquet",
        commit_message=f"Upload batch {upload_count}",
    )
    logging.info("Upload successful. Clearing local files...")
    for f in os.listdir(local_dir):
        if f.endswith(".parquet"):
            os.remove(os.path.join(local_dir, f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-repo", default="ragrawal36/fanoutqa-docs")
    parser.add_argument("--split", default="train")
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if args.local_dir is None:
        args.local_dir = f"./temp_{args.output_repo.split('/')[-1]}"

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    os.makedirs(args.local_dir, exist_ok=True)

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=args.output_repo, repo_type="dataset", private=args.private, exist_ok=True)
    logging.info(f"Repository {args.output_repo} ready.")

    ds = load_dataset(SOURCE_REPO, split=args.split, streaming=True, token=hf_token)

    records = []
    chunk_index = 0
    upload_count = 0
    total_written = 0
    skipped = 0

    with tqdm(desc="Processing documents") as pbar:
        for item in ds:
            raw = item.get("pos_doc", "") or ""
            if not raw:
                skipped += 1
                continue

            docs = [d.strip() for d in raw.split(DOC_SEPARATOR)]

            for doc in docs:
                if len(doc) < MIN_CHARS:
                    skipped += 1
                    continue

                records.append({"pos_doc": doc})
                pbar.update(1)

                if len(records) >= PARQUET_ROWS:
                    table = pa.Table.from_pylist(records, schema=SCHEMA)
                    path = os.path.join(args.local_dir, f"data_{chunk_index:06d}.parquet")
                    pq.write_table(table, path)
                    total_written += len(records)
                    logging.info(f"Wrote {len(records)} rows → {path}  (total: {total_written})")
                    records = []
                    chunk_index += 1

                    if get_dir_size(args.local_dir) >= UPLOAD_THRESHOLD:
                        upload_count += 1
                        upload_and_clear(api, args.local_dir, args.output_repo, upload_count)

    if records:
        table = pa.Table.from_pylist(records, schema=SCHEMA)
        path = os.path.join(args.local_dir, f"data_{chunk_index:06d}.parquet")
        pq.write_table(table, path)
        total_written += len(records)
        logging.info(f"Wrote {len(records)} rows → {path}  (total: {total_written})")

    if get_dir_size(args.local_dir) > 0:
        upload_count += 1
        upload_and_clear(api, args.local_dir, args.output_repo, upload_count)

    logging.info(f"Done. Total written: {total_written} | Skipped: {skipped}")


if __name__ == "__main__":
    main()
