#!/usr/bin/env python3
"""
Create a Q&A dataset from any HuggingFace dataset by splitting a document column
at the character midpoint: first half → "question", second half → "answer".

Usage:
    python datagen/create_doc_split_dataset.py \
        --source-repo roneneldan/TinyStories \
        --doc-column text \
        --output-repo mihir-1999/tinystories-qa-3m

    python datagen/create_doc_split_dataset.py \
        --source-repo vm2825/nemotron-cc-v21-Parsed-QA4 \
        --doc-column pos_doc \
        --output-repo mihir-1999/nemotron-qa
"""

import argparse
import logging
import os

import dotenv
dotenv.load_dotenv()

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ─── defaults ─────────────────────────────────────────────────────────────────
N_SAMPLES        = 3_000_000
MIN_CHARS        = 100
PARQUET_ROWS     = 50_000
UPLOAD_THRESHOLD = 512 * 1024 * 1024   # 512 MB

SCHEMA = pa.schema([
    ("question", pa.string()),
    ("answer",   pa.string()),
    ("pos_doc",  pa.string()),
])


# ─── helpers ──────────────────────────────────────────────────────────────────
def split_at_midpoint(text: str):
    mid = len(text) // 2
    space = text.rfind(' ', 0, mid)
    if space == -1:
        space = mid
    return text[:space].strip(), text[space:].strip()


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


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo",  required=True,  help="HF dataset repo to read from (e.g. roneneldan/TinyStories)")
    parser.add_argument("--doc-column",   required=True,  help="Column name containing the document text (e.g. text, pos_doc)")
    parser.add_argument("--output-repo",  required=True,  help="HF dataset repo to upload to (e.g. mihir-1999/my-qa-dataset)")
    parser.add_argument("--split",        default="train")
    parser.add_argument("--n-samples",    type=int, default=N_SAMPLES)
    parser.add_argument("--min-chars",    type=int, default=MIN_CHARS)
    parser.add_argument("--local-dir",    default=None,   help="Local temp dir (default: ./temp_<output-repo-name>)")
    parser.add_argument("--private",      action="store_true")
    args = parser.parse_args()

    if args.local_dir is None:
        args.local_dir = f"./temp_{args.output_repo.split('/')[-1]}"

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    os.makedirs(args.local_dir, exist_ok=True)

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=args.output_repo, repo_type="dataset", private=args.private, exist_ok=True)
    logging.info(f"Repository {args.output_repo} ready.")

    ds = load_dataset(args.source_repo, split=args.split, streaming=True, token=hf_token)

    records = []
    chunk_index = 0
    upload_count = 0
    total_written = 0
    skipped = 0

    with tqdm(total=args.n_samples, desc="Processing documents") as pbar:
        for item in ds:
            if total_written + len(records) >= args.n_samples:
                break

            text = item.get(args.doc_column, "") or ""
            if len(text) < args.min_chars:
                skipped += 1
                continue

            question, answer = split_at_midpoint(text)
            if not question or not answer:
                skipped += 1
                continue

            records.append({"question": question, "answer": answer, "pos_doc": text})

            if len(records) >= PARQUET_ROWS:
                table = pa.Table.from_pylist(records, schema=SCHEMA)
                path = os.path.join(args.local_dir, f"data_{chunk_index:06d}.parquet")
                pq.write_table(table, path)
                total_written += len(records)
                pbar.update(len(records))
                logging.info(f"Wrote {len(records)} rows → {path}  (total: {total_written})")
                records = []
                chunk_index += 1

                if get_dir_size(args.local_dir) >= UPLOAD_THRESHOLD:
                    upload_count += 1
                    upload_and_clear(api, args.local_dir, args.output_repo, upload_count)

    # Flush remaining records
    if records:
        table = pa.Table.from_pylist(records, schema=SCHEMA)
        path = os.path.join(args.local_dir, f"data_{chunk_index:06d}.parquet")
        pq.write_table(table, path)
        total_written += len(records)
        logging.info(f"Wrote {len(records)} rows → {path}  (total: {total_written})")

    # Final upload
    if get_dir_size(args.local_dir) > 0:
        upload_count += 1
        upload_and_clear(api, args.local_dir, args.output_repo, upload_count)

    logging.info(f"Done. Total written: {total_written} | Skipped: {skipped}")


if __name__ == "__main__":
    main()
