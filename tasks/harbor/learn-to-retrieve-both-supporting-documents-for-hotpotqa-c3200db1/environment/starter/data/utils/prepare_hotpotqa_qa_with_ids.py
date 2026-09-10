"""
Augment hotpotqa/hotpot_qa (distractor) with pos_doc_ids: integer IDs that reference
the matching rows in the hotpotqa-distractor-docs corpus (built by prepare_hotpotqa_docs.py).

Each output row keeps all original HotpotQA fields and adds:
  pos_doc_ids: list[int]  — corpus IDs of the gold supporting paragraphs.
                            -1 if a paragraph was not found in the corpus.

Must be run AFTER prepare_hotpotqa_docs.py (the corpus must exist with its "id" column).

Usage:
    uv run python data/utils/prepare_hotpotqa_qa_with_ids.py --split validation
"""

import argparse
import os
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from tqdm import tqdm


def prepare(split: str = "validation") -> str:
    hf_token = os.environ["HF_TOKEN"]
    hf_username = os.environ["HF_USERNAME"]
    repo_id = f"{hf_username}/hotpotqa-distractor-qa-with-ids-{split}"

    api = HfApi(token=hf_token)
    if api.repo_exists(repo_id, repo_type="dataset"):
        print(f"Dataset {repo_id} already exists. Skipping.")
        return repo_id

    corpus_repo = f"{hf_username}/hotpotqa-distractor-docs-{split}"
    print(f"Loading corpus from {corpus_repo} to build text→id mapping...")
    corpus = load_dataset(corpus_repo, split="train", token=hf_token)
    text_to_id = {}
    for idx, row in enumerate(tqdm(corpus, desc="Building text→id")):
        text_to_id[row["text"]] = int(row["id"]) if "id" in row else idx
    print(f"  {len(text_to_id):,} corpus entries indexed.")

    print(f"Loading hotpotqa/hotpot_qa (distractor, split={split})...")
    qa = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, streaming=True, token=hf_token)

    rows = []
    for item in tqdm(qa, desc="Augmenting with pos_doc_ids"):
        context = item.get("context") or {}
        titles = context.get("title", [])
        sentences_list = context.get("sentences", [])
        sf = item.get("supporting_facts") or {}
        gold_titles = set(sf.get("title", []))

        pos_doc_ids = []
        for title, sents in zip(titles, sentences_list):
            if title in gold_titles:
                text = title + ": " + " ".join(sents) if isinstance(sents, list) else title
                pos_doc_ids.append(text_to_id.get(text, -1))

        row = dict(item)
        row["pos_doc_ids"] = pos_doc_ids
        rows.append(row)

    missing = sum(1 for r in rows for pid in r["pos_doc_ids"] if pid == -1)
    total = sum(len(r["pos_doc_ids"]) for r in rows)
    print(f"  {missing}/{total} paragraph lookups missed (id=-1).")

    print(f"Pushing {len(rows):,} rows to {repo_id}...")
    Dataset.from_list(rows).push_to_hub(repo_id, token=hf_token)
    print(f"Done. Dataset at: https://huggingface.co/datasets/{repo_id}")
    return repo_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()
    prepare(args.split)
