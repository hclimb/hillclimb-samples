#!/usr/bin/env python3
"""
Convert sentence-transformers/embedding-training-data (*.jsonl.gz) into the
hard-negative SFT parquet schema used by
vm2825/triviaqa-hotpotqa-nq-squad-msmarco-hard-neg-sft4b.

One output repo PER source file:  {HF_USERNAME}/etd-<name>-hard-neg-sft

Design decisions (see claude/msa-data-processing-plan.md):
  1. Hard negs: cheap. Carry through existing negs only; pairs/sets get none.
     neg_scores = "0.0" placeholder per neg.
  2. think / generated_answer: empty strings (no generation).
  3. Scope: all 37 source files. Streamed — 119 GB S2ORC never materialized.
  4. Per-file output repos.

Streams each .jsonl.gz straight off the Hub via HfFileSystem (no full download),
maps each line to the target schema, shards to parquet, deterministic 95/5
train/val split (hash of query), uploads periodically and clears local files.

Cross-machine safe: skips a source file if its output repo already has parquet
shards (unless --overwrite).

Usage:
    python datagen/create_hard_neg_sft_dataset.py                 # all 37 files
    python datagen/create_hard_neg_sft_dataset.py --files msmarco-triplets.jsonl.gz
    python datagen/create_hard_neg_sft_dataset.py --dry-run       # no uploads
    python datagen/create_hard_neg_sft_dataset.py --list          # list source files & exit
"""

import argparse
import gzip
import hashlib
import json
import logging
import os

import dotenv
dotenv.load_dotenv()

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ─── constants ────────────────────────────────────────────────────────────────
SOURCE_REPO      = "sentence-transformers/embedding-training-data"
DOC_SEP          = "<doc_separator>"
PARQUET_ROWS     = 100_000               # rows per shard (per split)
UPLOAD_THRESHOLD = 512 * 1024 * 1024     # flush+upload once local parquet ≥ 512 MB
VAL_FRACTION     = 0.05                  # 5% → validation, deterministic by query hash

SCHEMA = pa.schema([
    ("query",            pa.string()),
    ("pos_doc",          pa.string()),
    ("neg_docs",         pa.string()),
    ("neg_scores",       pa.string()),
    ("think",            pa.string()),
    ("generated_answer", pa.string()),
])


# ─── mapping ──────────────────────────────────────────────────────────────────
def _row(query, pos_doc, negs):
    """Build a target-schema row. negs: list[str] (may be empty)."""
    negs = [n for n in negs if n]
    return {
        "query":            query,
        "pos_doc":          pos_doc,
        "neg_docs":         DOC_SEP.join(negs),
        "neg_scores":       DOC_SEP.join(["0.0"] * len(negs)),
        "think":            "",
        "generated_answer": "",
    }


def map_line(obj):
    """Map one source JSON object to zero or more target rows.

    Handles the 5 source schemas:
      ["a","b"]                              → pair
      ["anchor","pos","neg",...]             → triplet (negs = items[2:])
      {"set": [...]}                         → set (item0 = query, rest = pos)
      {"query","pos":[...]}                  → query-pairs
      {"query","pos":[...],"neg":[...]}      → query-triplets
    """
    rows = []
    if isinstance(obj, list):
        if len(obj) == 2:
            q, p = obj
            if q and p:
                rows.append(_row(q, p, []))
        elif len(obj) >= 3:
            q, p, negs = obj[0], obj[1], obj[2:]
            if q and p:
                rows.append(_row(q, p, negs))
    elif isinstance(obj, dict):
        if "set" in obj:
            s = [t for t in obj["set"] if t]
            if len(s) >= 2:
                q = s[0]
                for p in s[1:]:
                    rows.append(_row(q, p, []))
        elif "query" in obj:
            q = obj["query"]
            pos = obj.get("pos") or []
            neg = obj.get("neg") or []
            if isinstance(pos, str):
                pos = [pos]
            if isinstance(neg, str):
                neg = [neg]
            if q:
                for p in pos:
                    if p:
                        rows.append(_row(q, p, neg))
    return rows


def is_val(query: str) -> bool:
    h = int(hashlib.md5(query.encode("utf-8")).hexdigest(), 16)
    return (h % 1000) < int(VAL_FRACTION * 1000)


# ─── io helpers ───────────────────────────────────────────────────────────────
def list_source_files(fs: HfFileSystem):
    base = f"datasets/{SOURCE_REPO}"
    return sorted(
        os.path.basename(p)
        for p in fs.ls(base, detail=False)
        if p.endswith(".jsonl.gz")
    )


def repo_for(filename: str, username: str) -> str:
    name = filename[: -len(".jsonl.gz")] if filename.endswith(".jsonl.gz") else filename
    name = name.replace("_", "-").lower()
    return f"{username}/etd-{name}-hard-neg-sft"


COMPLETE_MARKER = "complete.json"
PROGRESS_MARKER = "progress.json"


def repo_is_complete(api: HfApi, repo: str) -> bool:
    """A repo is done only if it carries a complete.json marker."""
    try:
        files = api.list_repo_files(repo_id=repo, repo_type="dataset")
    except Exception:
        return False
    return COMPLETE_MARKER in files


def load_progress(api: HfApi, repo: str):
    """Return resume state {lines, train_chunk, val_chunk, train_rows, val_rows}
    from a prior run, or None. progress.json is written after every uploaded
    batch, so it always points at a fully-uploaded boundary."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo, repo_type="dataset",
                               filename=PROGRESS_MARKER)
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def _upload_json(api: HfApi, repo: str, name: str, payload: dict):
    import io
    api.upload_file(
        path_or_fileobj=io.BytesIO(json.dumps(payload, indent=2).encode()),
        path_in_repo=name,
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"Update {name}",
    )


def dir_parquet_size(data_dir: str) -> int:
    if not os.path.isdir(data_dir):
        return 0
    return sum(
        os.path.getsize(os.path.join(data_dir, f))
        for f in os.listdir(data_dir)
        if f.endswith(".parquet")
    )


def upload_and_clear(api: HfApi, local_dir: str, data_dir: str, repo: str, batch: int, dry_run: bool):
    if dry_run:
        logging.info(f"[dry-run] would upload batch {batch} to {repo}")
        for f in os.listdir(data_dir):
            if f.endswith(".parquet"):
                os.remove(os.path.join(data_dir, f))
        return
    logging.info(f"Uploading batch {batch} → {repo} ...")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo,
        repo_type="dataset",
        allow_patterns="data/*.parquet",
        commit_message=f"Upload batch {batch}",
    )
    for f in os.listdir(data_dir):
        if f.endswith(".parquet"):
            os.remove(os.path.join(data_dir, f))


# ─── per-file processing ──────────────────────────────────────────────────────
def process_file(fs: HfFileSystem, api: HfApi, filename: str, username: str,
                 local_root: str, dry_run: bool, private: bool, overwrite: bool):
    repo = repo_for(filename, username)
    logging.info(f"=== {filename}  →  {repo} ===")

    if not dry_run and not overwrite and repo_is_complete(api, repo):
        logging.info(f"  {repo} has complete.json — skipping (use --overwrite to force).")
        return

    if not dry_run:
        api.create_repo(repo_id=repo, repo_type="dataset", private=private, exist_ok=True)

    # ── resume: pick up from last uploaded boundary, unless --overwrite ──
    skip_lines = 0
    prog = None if (overwrite or dry_run) else load_progress(api, repo)
    if prog:
        skip_lines = prog.get("lines", 0)
        logging.info(f"  resuming {filename}: skipping {skip_lines:,} already-processed lines")

    local_dir = os.path.join(local_root, repo.split("/")[-1])
    data_dir = os.path.join(local_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    buf = {"train": [], "validation": []}
    chunk = {"train": (prog or {}).get("train_chunk", 0),
             "validation": (prog or {}).get("val_chunk", 0)}
    written = {"train": (prog or {}).get("train_rows", 0),
               "validation": (prog or {}).get("val_rows", 0)}
    batch = 0
    src_path = f"datasets/{SOURCE_REPO}/{filename}"

    def flush(split):
        if not buf[split]:
            return
        table = pa.Table.from_pylist(buf[split], schema=SCHEMA)
        path = os.path.join(data_dir, f"{split}-{chunk[split]:06d}.parquet")
        pq.write_table(table, path)
        written[split] += len(buf[split])
        chunk[split] += 1
        buf[split] = []

    line_no = 0
    with fs.open(src_path, "rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc=filename, unit=" lines"):
            line_no += 1
            if line_no <= skip_lines:        # fast resume skip: no parse/write
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for row in map_line(obj):
                split = "validation" if is_val(row["query"]) else "train"
                buf[split].append(row)
                if len(buf[split]) >= PARQUET_ROWS:
                    flush(split)
            if dir_parquet_size(data_dir) >= UPLOAD_THRESHOLD:
                batch += 1
                upload_and_clear(api, local_dir, data_dir, repo, batch, dry_run)
                if not dry_run:
                    # progress.json points at a fully-uploaded boundary
                    _upload_json(api, repo, PROGRESS_MARKER, {
                        "lines": line_no, "train_chunk": chunk["train"],
                        "val_chunk": chunk["validation"],
                        "train_rows": written["train"], "val_rows": written["validation"],
                    })

    flush("train")
    flush("validation")
    if dir_parquet_size(data_dir) > 0:
        batch += 1
        upload_and_clear(api, local_dir, data_dir, repo, batch, dry_run)

    if not dry_run:
        _upload_json(api, repo, COMPLETE_MARKER, {
            "source_file": filename, "lines": line_no,
            "train_rows": written["train"], "val_rows": written["validation"],
        })
    logging.info(f"  done {filename}: train={written['train']} val={written['validation']}")


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", default=None,
                        help="Comma-separated source filenames to process (default: all 37).")
    parser.add_argument("--local-dir", default="./temp_hard_neg_sft",
                        help="Local temp root for parquet staging.")
    parser.add_argument("--list", action="store_true", help="List source files and exit.")
    parser.add_argument("--dry-run", action="store_true", help="No HF uploads / repo creation.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reprocess from scratch, ignoring complete.json / progress.json.")
    parser.add_argument("--backfill-complete", action="store_true",
                        help="Just write a complete.json marker to each --files repo (no processing). "
                             "Use to bless repos known-done by a prior run.")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")
    username = os.getenv("HF_USERNAME")
    if not username and not (args.list or args.dry_run):
        raise ValueError("HF_USERNAME environment variable not set")

    fs = HfFileSystem(token=hf_token)
    api = HfApi(token=hf_token)

    source_files = list_source_files(fs)

    if args.list:
        for fn in source_files:
            print(f"{fn:60s} → {repo_for(fn, username or '<HF_USERNAME>')}")
        print(f"\n{len(source_files)} source files.")
        return

    targets = source_files
    if args.files:
        wanted = {f.strip() for f in args.files.split(",")}
        targets = [f for f in source_files if f in wanted]
        missing = wanted - set(targets)
        if missing:
            logging.warning(f"Not found in source repo: {sorted(missing)}")

    if args.backfill_complete:
        for fn in targets:
            repo = repo_for(fn, username)
            _upload_json(api, repo, COMPLETE_MARKER,
                         {"source_file": fn, "backfilled": True})
            logging.info(f"  marked complete: {repo}")
        logging.info("Backfill done.")
        return

    os.makedirs(args.local_dir, exist_ok=True)
    logging.info(f"Processing {len(targets)} file(s).")
    for fn in targets:
        try:
            process_file(fs, api, fn, username, args.local_dir,
                         args.dry_run, args.private, args.overwrite)
        except Exception as e:
            logging.error(f"FAILED {fn}: {e}", exc_info=True)

    logging.info("All done.")


if __name__ == "__main__":
    main()
