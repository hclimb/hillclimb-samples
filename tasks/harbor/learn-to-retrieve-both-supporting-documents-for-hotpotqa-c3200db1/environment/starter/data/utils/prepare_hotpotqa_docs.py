"""
Prepare a flat HuggingFace dataset of all unique context paragraphs
from hotpotqa/hotpot_qa (distractor, validation split).

Each output row has a single 'text' column: "Title: sentence1 sentence2 ..."

Usage:
    uv run python data/utils/prepare_hotpotqa_docs.py \
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
    repo_id = f"{hf_username}/hotpotqa-distractor-docs-{split}"

    api = HfApi(token=hf_token)
    if api.repo_exists(repo_id, repo_type="dataset"):
        print(f"Dataset {repo_id} already exists. Skipping.")
        return repo_id

    print(f"Loading hotpotqa/hotpot_qa (distractor, split={split})...")
    src = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, streaming=True, token=hf_token)

    seen, rows = set(), []
    for item in tqdm(src, desc="Extracting context paragraphs"):
        context = item.get("context") or {}
        for title, sents in zip(context.get("title", []), context.get("sentences", [])):
            if title in seen:
                continue
            seen.add(title)
            text = title + ": " + " ".join(sents) if isinstance(sents, list) else title
            rows.append({"text": text, "id": len(rows)})

    print(f"Extracted {len(rows):,} unique paragraphs. Pushing to {repo_id}...")
    flat_ds = Dataset.from_list(rows)
    flat_ds.push_to_hub(repo_id, token=hf_token)
    print(f"Done. Dataset at: https://huggingface.co/datasets/{repo_id}")
    return repo_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()
    prepare(args.split)
