#!/usr/bin/env python3
"""
Load zhuexe/fanoutqa validation split, fetch Wikipedia articles for each question's
decomposition evidence using fetch_revision_text (raw wikitext including tables),
and upload as ragrawal36/fanoutqa with columns: question, answer, pos_doc.

Uses batched revision fetching (50 revisions per API call) to avoid rate limits.

Documents are joined with '<|doc_seperator|>' in the pos_doc column, matching the
schema of sriragt/fanoutqa.

Usage:
    HF_TOKEN=hf_... python datagen/create_fanoutqa_dataset.py \
        --output-repo ragrawal36/fanoutqa
"""

import argparse
import json
import logging
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SOURCE_REPO = "zhuexe/fanoutqa"
DOC_SEPARATOR = "<|doc_seperator|>"
API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "memory-layers-fanoutqa/1.0 (research)"}

SCHEMA = pa.schema([
    ("question", pa.string()),
    ("answer", pa.string()),
    ("pos_doc", pa.string()),
])


def fetch_revision_text(rev_id: int, max_retries: int = 8) -> str:
    """Fetch raw wikitext for a specific revision ID, respecting Retry-After headers."""
    params = {
        "action": "query",
        "revids": rev_id,
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
    }
    for attempt in range(max_retries):
        resp = requests.get(API_URL, headers=HEADERS, params=params)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 2 ** attempt + 2))
            wait = max(retry_after, 2 ** attempt + 2)
            logging.warning(f"Rate limited on revid={rev_id}, waiting {wait}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        pages = resp.json()["query"]["pages"]
        page = next(iter(pages.values()))
        return page["revisions"][0]["slots"]["main"]["*"]
    raise RuntimeError(f"Failed to fetch revid={rev_id} after {max_retries} retries")


def collect_evidence(decomposition) -> list[dict]:
    """Recursively collect all unique evidence items from decomposition list."""
    seen_pageids = set()
    evidence_list = []

    def _recurse(items):
        if not items:
            return
        for sub in items:
            if not isinstance(sub, dict):
                continue
            ev = sub.get("evidence")
            if ev and isinstance(ev, dict):
                pid = ev.get("pageid")
                if pid is not None and pid not in seen_pageids:
                    seen_pageids.add(pid)
                    evidence_list.append(ev)
            _recurse(sub.get("decomposition") or [])

    _recurse(decomposition)
    return evidence_list


def answer_to_str(answer) -> str:
    """Convert answer (may be JSON string, dict, list, or primitive) to a readable string."""
    if isinstance(answer, str):
        try:
            decoded = json.loads(answer)
            if isinstance(decoded, str):
                return decoded
            return json.dumps(decoded, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return answer
    if isinstance(answer, (int, float, bool)):
        return str(answer)
    return json.dumps(answer, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-repo", default="ragrawal36/fanoutqa")
    parser.add_argument("--source-split", default="validation")
    parser.add_argument("--output-split", default="train")
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--batch-delay", type=float, default=1.5,
                        help="Delay between individual API calls (seconds)")
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

    logging.info(f"Loading {SOURCE_REPO} split={args.source_split} ...")
    ds = list(load_dataset(SOURCE_REPO, split=args.source_split, streaming=True))
    logging.info(f"Loaded {len(ds)} questions.")

    # Pass 1: collect all unique rev_ids needed
    all_rev_ids: set[int] = set()
    for item in ds:
        for ev in collect_evidence(item.get("decomposition") or []):
            rid = ev.get("revid")
            if rid:
                all_rev_ids.add(rid)

    logging.info(f"Need to fetch {len(all_rev_ids)} unique revisions (~{len(all_rev_ids)*args.batch_delay/60:.1f} min at {args.batch_delay}s/rev).")

    # Pass 2: fetch all revisions individually
    rev_cache: dict[int, str] = {}
    for rev_id in tqdm(list(all_rev_ids), desc="Fetching revisions"):
        try:
            rev_cache[rev_id] = fetch_revision_text(rev_id)
        except Exception as e:
            logging.warning(f"Failed revid={rev_id}: {e}")
            rev_cache[rev_id] = ""
        time.sleep(args.batch_delay)

    logging.info(f"Fetched {len(rev_cache)} / {len(all_rev_ids)} revisions successfully.")

    # Pass 3: build records
    records = []
    skipped = 0

    for item in tqdm(ds, desc="Building records"):
        question = item.get("question", "")
        answer_raw = item.get("answer", "")
        decomposition = item.get("decomposition") or []

        if not question:
            skipped += 1
            continue

        answer = answer_to_str(answer_raw)
        evidence_list = collect_evidence(decomposition)

        if not evidence_list:
            skipped += 1
            continue

        docs = []
        for ev in evidence_list:
            rid = ev.get("revid")
            text = rev_cache.get(rid, "").strip()
            if text:
                docs.append(text)

        if not docs:
            logging.warning(f"No docs for: {question[:80]}")
            skipped += 1
            continue

        records.append({
            "question": question,
            "answer": answer,
            "pos_doc": DOC_SEPARATOR.join(docs),
        })

    logging.info(f"Processed {len(records)} questions, skipped {skipped}.")

    if not records:
        logging.error("No records to upload!")
        return

    table = pa.Table.from_pylist(records, schema=SCHEMA)
    split_dir = os.path.join(args.local_dir, args.output_split)
    os.makedirs(split_dir, exist_ok=True)
    path = os.path.join(split_dir, "data_000000.parquet")
    pq.write_table(table, path)
    logging.info(f"Wrote {len(records)} rows to {path}")

    logging.info(f"Uploading to {args.output_repo} ...")
    api.upload_folder(
        folder_path=args.local_dir,
        repo_id=args.output_repo,
        repo_type="dataset",
        allow_patterns="*.parquet",
        commit_message=f"Upload {args.output_split} split ({len(records)} rows) with revision wikitext (includes tables)",
    )
    logging.info(f"Done. Uploaded {len(records)} rows to {args.output_repo}.")


if __name__ == "__main__":
    main()
