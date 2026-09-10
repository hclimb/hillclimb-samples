"""
Prepare an MSA document corpus with stable integer IDs.

Reads an existing vm2825/msa-*-docs dataset, preserves row order, adds an `id`
column, and pushes the result to the caller's HF namespace. This makes the
corpus usable by GenLargeMemEvaluator's ID-based doc hit metrics.

Usage:
    uv run python data/utils/prepare_msa_docs.py --dataset hotpotqa
"""

import argparse
import os

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

from data.msa_prep_common import DATASET_SPECS


def prepare(dataset: str) -> str:
    spec = DATASET_SPECS[dataset]
    hf_token = os.environ["HF_TOKEN"]
    hf_username = os.environ["HF_USERNAME"]
    repo_id = f"{hf_username}/msa-{dataset.replace('_', '-')}-docs-with-ids"

    api = HfApi(token=hf_token)
    if api.repo_exists(repo_id, repo_type="dataset"):
        print(f"Dataset {repo_id} already exists. Skipping.")
        return repo_id

    print(f"Loading {spec['doc_repo']}...")
    src = load_dataset(spec["doc_repo"], split="train", streaming=True, token=hf_token)

    rows = []
    for item in tqdm(src, desc=f"Adding IDs to {dataset} docs"):
        text = (item.get("text") or "").strip()
        if not text:
            continue
        rows.append({"text": text, "id": len(rows)})

    print(f"Prepared {len(rows):,} rows. Pushing to {repo_id}...")
    Dataset.from_list(rows).push_to_hub(repo_id, token=hf_token)
    print(f"Done. Dataset at: https://huggingface.co/datasets/{repo_id}")
    return repo_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), required=True)
    args = parser.parse_args()
    prepare(args.dataset)
