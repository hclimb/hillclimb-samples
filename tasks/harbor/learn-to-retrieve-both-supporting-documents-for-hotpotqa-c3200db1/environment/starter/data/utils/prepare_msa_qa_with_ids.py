"""
Augment an MSA QA dataset with `pos_doc_ids` that reference the prepared MSA doc
corpus built by prepare_msa_docs.py.

Usage:
    uv run python data/utils/prepare_msa_qa_with_ids.py --dataset hotpotqa
"""

import argparse
import os

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

from data.msa_prep_common import DATASET_SPECS, split_pos_doc


def prepare(dataset: str) -> str:
    spec = DATASET_SPECS[dataset]
    hf_token = os.environ["HF_TOKEN"]
    hf_username = os.environ["HF_USERNAME"]
    repo_id = f"{hf_username}/msa-{dataset.replace('_', '-')}-qa-with-ids"
    corpus_repo = f"{hf_username}/msa-{dataset.replace('_', '-')}-docs-with-ids"

    api = HfApi(token=hf_token)
    if api.repo_exists(repo_id, repo_type="dataset"):
        print(f"Dataset {repo_id} already exists. Skipping.")
        return repo_id

    print(f"Loading corpus from {corpus_repo} to build text->ids mapping...")
    corpus = load_dataset(corpus_repo, split="train", token=hf_token)
    text_to_ids = {}
    for idx, row in enumerate(tqdm(corpus, desc="Building text->ids")):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        doc_id = int(row["id"]) if "id" in row else idx
        text_to_ids.setdefault(text, []).append(doc_id)
    print(f"  {len(text_to_ids):,} unique doc texts indexed.")

    print(f"Loading QA dataset from {spec['qa_repo']}...")
    qa = load_dataset(spec["qa_repo"], split="train", streaming=True, token=hf_token)

    rows = []
    separator = spec["doc_separator"]
    for item in tqdm(qa, desc=f"Augmenting {dataset} QA with pos_doc_ids"):
        docs = split_pos_doc(item.get("pos_doc") or "", separator)
        pos_doc_ids = []
        for doc in docs:
            pos_doc_ids.extend(text_to_ids.get(doc, [-1]))

        # Preserve order while removing duplicates.
        seen = set()
        ordered_ids = []
        for doc_id in pos_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            ordered_ids.append(doc_id)

        row = dict(item)
        row["pos_doc_ids"] = ordered_ids
        rows.append(row)

    missing = sum(1 for r in rows for pid in r["pos_doc_ids"] if pid == -1)
    total = sum(len(r["pos_doc_ids"]) for r in rows)
    print(f"  {missing}/{total} doc lookups missed (id=-1).")

    print(f"Pushing {len(rows):,} rows to {repo_id}...")
    Dataset.from_list(rows).push_to_hub(repo_id, token=hf_token)
    print(f"Done. Dataset at: https://huggingface.co/datasets/{repo_id}")
    return repo_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), required=True)
    args = parser.parse_args()
    prepare(args.dataset)
