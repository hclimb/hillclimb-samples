"""
Prepare a flat HuggingFace dataset of all unique supporting passages
from mihir-1999/musique-supporting-list.

Each output row has a single 'text' column containing one passage.

Usage:
    uv run python data/utils/prepare_musique_docs.py \
        --split validation
"""

import argparse
import os
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from tqdm import tqdm


def prepare(split="validation"):
    hf_token = os.environ["HF_TOKEN"]
    hf_username = os.environ["HF_USERNAME"]
    repo_id = f"{hf_username}/musique-docs-{split}"

    api = HfApi(token=hf_token)
    if api.repo_exists(repo_id, repo_type="dataset"):
        print(f"Dataset {repo_id} already exists. Skipping.")
        return repo_id

    print(f"Loading mihir-1999/musique-supporting-list (split={split})...")
    src = load_dataset("mihir-1999/musique-supporting", split=split, token=hf_token)

    seen, rows = set(), []
    for item in tqdm(src, desc="Extracting passages"):
        pos_doc = item.get("pos_doc") or ""
        passages = [p.strip() for p in pos_doc.split("\n\n") if p.strip().startswith("**")]
        for passage in passages:
            if not passage or passage in seen:
                continue
            seen.add(passage)
            rows.append({"text": passage, "id": len(rows)})

    print(f"Extracted {len(rows):,} unique passages. Pushing to {repo_id}...")
    flat_ds = Dataset.from_list(rows)
    flat_ds.push_to_hub(repo_id, token=hf_token)
    print(f"Done. Dataset at: https://huggingface.co/datasets/{repo_id}")
    return repo_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()
    prepare(args.split)
