"""Streaming concat of the 5 SFT-4B hard-neg datasets with a per-source 5% val carve-out.

Sources (single `train` split each):
  vm2825/msmarco-triplets-hard-neg-reasoning-embedding-modified-SFT-4B-parts-0-9
  vm2825/triviaqa-pairs-hard-neg-reasoning-embedding-modified-SFT-4B-parts-0-1
  vm2825/nq-train-pairs-hard-neg-reasoning-embedding-modified-SFT-4B-parts-0-1
  vm2825/squad-pairs-hard-neg-reasoning-embedding-modified-SFT-4B-parts-0-1
  vm2825/hotpotqa-hard-neg-reasoning-embedding-modified-SFT-4B-parts-0-1

For every source we stream rows, send ~VAL_FRAC to validation (deterministic RNG, seeded
per-source), and the rest to train. Rows are buffered in memory, flushed as parquet chunks
of CHUNK_ROWS, uploaded to the combined target, and the tmp files are deleted.

Target: vm2825/triviaqa-hotpotqa-nq-squad-msmarco-hard-neg-sft4b
Schema kept: query, pos_doc, neg_docs, neg_scores, think, generated_answer.
"""
import argparse
import os
import random
import sys
import tempfile
import time

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if not HF_TOKEN:
    sys.exit("HF_TOKEN missing")
login(token=HF_TOKEN)

SOURCES = [
    "vm2825/msmarco-triplets-hard-neg-reasoning-embedding-modified-SFT-4B-parts-0-9",
    "vm2825/triviaqa-pairs-hard-neg-reasoning-embedding-modified-SFT-4B-parts-0-1",
    "vm2825/nq-train-pairs-hard-neg-reasoning-embedding-modified-SFT-4B-parts-0-1",
    "vm2825/squad-pairs-hard-neg-reasoning-embedding-modified-SFT-4B-parts-0-1",
    "vm2825/hotpotqa-hard-neg-reasoning-embedding-modified-SFT-4B-parts-0-1",
]

DEFAULT_COMBINED_TARGET = "vm2825/triviaqa-hotpotqa-nq-squad-msmarco-hard-neg-sft4b"
VAL_FRAC = 0.05
SEED = 42
CHUNK_ROWS = 50_000

FIELDS = ["query", "pos_doc", "neg_docs", "neg_scores", "think", "generated_answer"]
SCHEMA = pa.schema([(f, pa.string()) for f in FIELDS])


class SplitWriter:
    """Buffers rows for a single (repo, split) and flushes chunk parquets in order."""

    def __init__(self, api: HfApi, repo_id: str, split: str):
        self.api = api
        self.repo_id = repo_id
        self.split = split
        self.buffer: list[dict] = []
        self.file_idx = 0
        self.rows_written = 0

    def add(self, row: dict) -> None:
        self.buffer.append({f: (row.get(f) or "") for f in FIELDS})
        if len(self.buffer) >= CHUNK_ROWS:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        table = pa.Table.from_pylist(self.buffer, schema=SCHEMA)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = tmp.name
        pq.write_table(table, tmp_path, compression="snappy")
        out_name = f"data/{self.split}-{self.file_idx:05d}.parquet"
        self.api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=out_name,
            repo_id=self.repo_id,
            repo_type="dataset",
        )
        os.remove(tmp_path)
        self.rows_written += len(self.buffer)
        self.file_idx += 1
        self.buffer = []


def stream_source(source: str, train_w: SplitWriter, val_w: SplitWriter, seed: int) -> tuple[int, int]:
    rng = random.Random(seed)
    ds = load_dataset(source, split="train", token=HF_TOKEN, streaming=True)
    n_train = n_val = 0
    start = time.time()
    for row in ds:
        if rng.random() < VAL_FRAC:
            val_w.add(row)
            n_val += 1
        else:
            train_w.add(row)
            n_train += 1
        if (n_train + n_val) % 20_000 == 0:
            elapsed = time.time() - start
            sys.stdout.write(
                f"\r    [{source.split('/')[-1][:40]}] "
                f"seen {n_train + n_val:,} | train={n_train:,} val={n_val:,} | "
                f"{(n_train + n_val) / max(elapsed, 1e-6):,.0f} rows/s"
            )
            sys.stdout.flush()
    print()
    return n_train, n_val


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--combined-target", default=DEFAULT_COMBINED_TARGET)
    p.add_argument("--sources", nargs="+", default=SOURCES)
    args = p.parse_args()

    api = HfApi()
    api.create_repo(repo_id=args.combined_target, repo_type="dataset", exist_ok=True, private=False)
    train_w = SplitWriter(api, args.combined_target, "train")
    val_w = SplitWriter(api, args.combined_target, "validation")

    totals = []
    for idx, src in enumerate(args.sources):
        print(f"\n{'=' * 80}\nSOURCE ({idx + 1}/{len(args.sources)}): {src}\n{'=' * 80}")
        per_src_seed = SEED * 1_000_003 + idx
        nt, nv = stream_source(src, train_w, val_w, per_src_seed)
        print(f"  {src}: train+={nt:,}  val+={nv:,}")
        totals.append((src, nt, nv))

    train_w.flush()
    val_w.flush()
    print(f"\n{'=' * 80}\nDONE -> {args.combined_target}\n{'=' * 80}")
    for src, nt, nv in totals:
        print(f"  {src}: train={nt:,}  val={nv:,}")
    print(f"  TOTAL: train={train_w.rows_written:,}  val={val_w.rows_written:,}")
    print(f"  uploaded {train_w.file_idx} train chunks and {val_w.file_idx} val chunks")


if __name__ == "__main__":
    main()
