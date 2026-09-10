"""
Augment mihir-1999/musique-supporting with pos_doc_ids: integer IDs that reference
the matching rows in the musique-docs-validation corpus (built by prepare_musique_docs.py).

Each output row keeps all original musique-supporting fields and adds:
  pos_doc_ids: list[int]  — corpus IDs of the supporting passages for this question.
                            -1 if a passage was not found in the corpus.

Must be run AFTER prepare_musique_docs.py (the corpus must exist with its "id" column).

Usage:
    uv run python data/utils/prepare_musique_qa_with_ids.py --split validation
"""

import argparse
import os
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from tqdm import tqdm


def prepare(split: str = "validation") -> str:
    hf_token = os.environ["HF_TOKEN"]
    hf_username = os.environ["HF_USERNAME"]
    repo_id = f"{hf_username}/musique-supporting-with-ids-{split}"

    api = HfApi(token=hf_token)
    if api.repo_exists(repo_id, repo_type="dataset"):
        print(f"Dataset {repo_id} already exists. Skipping.")
        return repo_id

    corpus_repo = "mihir-1999/musique-docs-validation"
    print(f"Loading corpus from {corpus_repo} to build text→id mapping...")
    corpus = load_dataset(corpus_repo, split="train", token=hf_token)
    text_to_id = {}
    for idx, row in enumerate(tqdm(corpus, desc="Building text→id")):
        text_to_id[row["text"]] = int(row["id"]) if "id" in row else idx
    print(f"  {len(text_to_id):,} corpus entries indexed.")

    print(f"Loading mihir-1999/musique-supporting (split={split})...")
    qa = load_dataset("mihir-1999/musique-supporting", split=split, token=hf_token)

    rows = []
    for item in tqdm(qa, desc="Augmenting with pos_doc_ids"):
        pos_doc = item.get("pos_doc") or ""
        passages = [p.strip() for p in pos_doc.split("\n\n") if p.strip().startswith("**")]
        pos_doc_ids = [text_to_id.get(p, -1) for p in passages]

        row = dict(item)
        row["pos_doc_ids"] = pos_doc_ids
        rows.append(row)

    missing = sum(1 for r in rows for pid in r["pos_doc_ids"] if pid == -1)
    total = sum(len(r["pos_doc_ids"]) for r in rows)
    print(f"  {missing}/{total} passage lookups missed (id=-1).")

    print(f"Pushing {len(rows):,} rows to {repo_id}...")
    Dataset.from_list(rows).push_to_hub(repo_id, token=hf_token)
    print(f"Done. Dataset at: https://huggingface.co/datasets/{repo_id}")
    return repo_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()
    prepare(args.split)
