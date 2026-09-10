#!/usr/bin/env python3
"""
Rearrange mihir-1999/nemotron-qa into a completion format.

Original schema:
  question  — first half of document
  answer    — second half of document
  pos_doc   — full document (question + answer)

New schema:
  question  — first 32 whitespace-tokens of original question
  answer    — rest of full_doc after those 32 tokens (original question[32:] + original answer)
  pos_doc   — original question (first half of document)
  full_doc  — original pos_doc (full document)

Usage:
    python datagen/create_nemotron_completion_dataset.py \
        --output-repo ragrawal36/nemotron-science-completion \
        [--n-samples 3000000] \
        [--private]
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
SOURCE_REPO      = "mihir-1999/tinystories-qa-3m"
N_SAMPLES        = 3_000_000
MIN_CHARS        = 100
QUESTION_TOKENS  = 32
PARQUET_ROWS     = 50_000
UPLOAD_THRESHOLD = 512 * 1024 * 1024  # 512 MB

SCHEMA = pa.schema([
    ("question", pa.string()),
    ("answer",   pa.string()),
    ("pos_doc",  pa.string()),
    ("full_doc", pa.string()),
])


# ─── helpers ──────────────────────────────────────────────────────────────────
def split_first_n_tokens(text: str, n: int):
    """Split text at the nth whitespace token boundary."""
    tokens = text.split()
    if len(tokens) <= n:
        return text, ""
    question = " ".join(tokens[:n])
    answer = " ".join(tokens[n:])
    return question, answer


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
    parser.add_argument("--source-repo",  default=SOURCE_REPO)
    parser.add_argument("--output-repo",  required=True, help="HF dataset repo to upload to (e.g. ragrawal36/nemotron-science-completion)")
    parser.add_argument("--split",        default="train")
    parser.add_argument("--n-samples",    type=int, default=N_SAMPLES)
    parser.add_argument("--min-chars",    type=int, default=MIN_CHARS)
    parser.add_argument("--question-tokens", type=int, default=QUESTION_TOKENS)
    parser.add_argument("--local-dir",    default=None)
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

    with tqdm(total=args.n_samples, desc="Processing rows") as pbar:
        for item in ds:
            if total_written + len(records) >= args.n_samples:
                break

            orig_question = item.get("question", "") or ""
            orig_answer   = item.get("answer", "") or ""
            orig_pos_doc  = item.get("pos_doc", "") or ""

            if len(orig_pos_doc) < args.min_chars:
                skipped += 1
                continue

            # Full document is what was previously pos_doc
            full_doc = orig_pos_doc

            # pos_doc is the first half (was question)
            pos_doc = orig_question

            # Split the full_doc at 32 tokens to get new question/answer
            question, answer_suffix = split_first_n_tokens(orig_question, args.question_tokens)
            if not question or not answer_suffix:
                skipped += 1
                continue

            # Answer is everything after the first 32 tokens of the full doc
            # = remainder of original question + space + original answer
            answer = answer_suffix + " " + orig_answer if orig_answer else answer_suffix

            records.append({
                "question": question,
                "answer":   answer,
                "pos_doc":  pos_doc,
                "full_doc": full_doc,
            })

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
        pbar.update(len(records))
        logging.info(f"Wrote {len(records)} rows → {path}  (total: {total_written})")

    # Final upload
    if get_dir_size(args.local_dir) > 0:
        upload_count += 1
        upload_and_clear(api, args.local_dir, args.output_repo, upload_count)

    logging.info(f"Done. Total written: {total_written} | Skipped: {skipped}")


if __name__ == "__main__":
    main()
