#!/usr/bin/env python3
"""Push Stage-2 hard-negative ID shards to HuggingFace.

IDs only (row_id, pos_doc_ids, neg_doc_ids); they index the Stage-0 corpus at
mihir-1999/multihop_qa_sft-doc-corpus. Kept as IDs deliberately: materializing 200
negatives x ~5.4M exploded rows as text would be ~1.5 TB against a 5.4 GB source.

Usage:
    python datagen/multihop_push_hard_neg_ids.py --ids-dir data/cache/multihop_hard_neg_ids
"""

import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "mihir-1999/multihop_qa_sft-hard-negatives"
CORPUS_REPO = "mihir-1999/multihop_qa_sft-doc-corpus"

README = f"""---
license: cc-by-sa-4.0
task_categories:
- question-answering
configs:
- config_name: default
  data_files: data/*.parquet
---

# multihop_qa_sft — hard negative IDs

Hard negatives mined for [`ragrawal36/multihop_qa_sft`](https://huggingface.co/datasets/ragrawal36/multihop_qa_sft)
(train split). **IDs only** — they index the document corpus at
[`{CORPUS_REPO}`](https://huggingface.co/datasets/{CORPUS_REPO}).

## Schema

| column | type | meaning |
|---|---|---|
| `row_id` | int32 | source row index in `ragrawal36/multihop_qa_sft` train |
| `pos_doc_ids` | list[int32] | the row's own supporting docs (positives) |
| `neg_doc_ids` | list[int32] | 200 mined hard negatives |

`pos_doc_ids` and `neg_doc_ids` are disjoint by construction and verified.

## How they were mined

1. Every doc embedded with `Qwen/Qwen3-Embedding-0.6B` (last-token pool, L2-normalised).
2. Each question embedded with the same model using
   `Instruct: Given a web search query, retrieve relevant passages that answer the query`.
3. Top-220 retrieved by brute-force dense matmul + `jax.lax.approx_max_k` on TPU
   (doc matrix replicated, queries sharded).
4. **All** of the row's own docs removed — every paragraph of a source row supports that
   row's question, so any of them would be a false negative — then truncated to 200.

## Joining back to text

```python
from datasets import load_dataset
corpus = load_dataset("{CORPUS_REPO}", "corpus", split="train")
ids    = load_dataset("{REPO}", split="train")
text   = corpus["text"]           # index == doc_id
row    = ids[0]
positives = [text[i] for i in row["pos_doc_ids"]]
negatives = [text[i] for i in row["neg_doc_ids"]]
```
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids-dir", default="data/cache/multihop_hard_neg_ids")
    p.add_argument("--repo", default=REPO)
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        for cand in (Path(".env"), Path.home() / ".env"):
            if cand.exists():
                for line in cand.read_text().splitlines():
                    if line.strip().startswith("HF_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
            if token:
                break
    if not token:
        print("HF_TOKEN not found (env or .env)", file=sys.stderr)
        return 1

    ids_dir = Path(args.ids_dir)
    shards = sorted(ids_dir.glob("hard_neg_ids_*.parquet"))
    if not shards:
        print(f"no shards in {ids_dir}", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    print(f"authenticated as: {api.whoami().get('name')}")
    api.create_repo(repo_id=args.repo, repo_type="dataset", exist_ok=True,
                    private=args.private)

    for s in shards:
        print(f"uploading {s.name} ({s.stat().st_size/1e6:,.0f} MB)...", flush=True)
        api.upload_file(path_or_fileobj=str(s), path_in_repo=f"data/{s.name}",
                        repo_id=args.repo, repo_type="dataset")

    stats = ids_dir / "mine_stats.json"
    if stats.exists():
        api.upload_file(path_or_fileobj=str(stats), path_in_repo="mine_stats.json",
                        repo_id=args.repo, repo_type="dataset")

    api.upload_file(path_or_fileobj=README.encode(), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset")
    print(f"\ndone: https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
