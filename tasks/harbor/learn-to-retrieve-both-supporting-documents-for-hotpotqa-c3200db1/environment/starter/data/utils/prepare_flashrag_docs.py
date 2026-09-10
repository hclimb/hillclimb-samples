"""
Prepare a flat HuggingFace dataset of positive passages from any FlashRAG
dataset that uses the metadata.passages structure (passage_text + is_selected).

Each output row has a single 'text' column containing one is_selected=1 passage.

Usage:
    uv run python -m data.utils.prepare_flashrag_docs \
        --hf_config msmarco-qa \
        --split dev

    uv run python -m data.utils.prepare_flashrag_docs \
        --hf_name RUC-NLPIR/FlashRAG_datasets \
        --hf_config hotpotqa \
        --split test \
        --repo_suffix my-custom-name
"""

import argparse
import os
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from tqdm import tqdm


def prepare(
    hf_name: str = "RUC-NLPIR/FlashRAG_datasets",
    hf_config: str = "msmarco-qa",
    split: str = "dev",
    repo_suffix: str = None,
) -> str:
    hf_token = os.environ["HF_TOKEN"]
    hf_username = os.environ["HF_USERNAME"]

    if repo_suffix is None:
        config_slug = hf_config.replace("/", "-").replace("_", "-")
        repo_suffix = f"{config_slug}-positive-passages-{split}"
    repo_id = f"{hf_username}/{repo_suffix}"

    api = HfApi(token=hf_token)
    if api.repo_exists(repo_id, repo_type="dataset"):
        print(f"Dataset {repo_id} already exists. Skipping.")
        return repo_id

    print(f"Loading {hf_name} (config={hf_config}, split={split})...")
    src = load_dataset(hf_name, hf_config, split=split, streaming=True, token=hf_token)

    passages = []
    for item in tqdm(src, desc="Extracting positive passages"):
        passages_data = (item.get("metadata") or {}).get("passages") or {}
        texts = passages_data.get("passage_text", [])
        selected = passages_data.get("is_selected", [])
        for text, sel in zip(texts, selected):
            if sel == 1 and text:
                passages.append({"text": text, "id": len(passages)})

    print(f"Extracted {len(passages):,} positive passages. Pushing to {repo_id}...")
    flat_ds = Dataset.from_list(passages)
    flat_ds.push_to_hub(repo_id, token=hf_token)
    print(f"Done. Dataset at: https://huggingface.co/datasets/{repo_id}")
    return repo_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_name", default="RUC-NLPIR/FlashRAG_datasets")
    parser.add_argument("--hf_config", default="msmarco-qa")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--repo_suffix", default=None,
                        help="Output repo name suffix (default: {hf_config}-positive-passages-{split})")
    args = parser.parse_args()
    prepare(args.hf_name, args.hf_config, args.split, args.repo_suffix)
