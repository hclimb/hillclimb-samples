"""
Prepare MuSiQue dataset with supporting paragraphs stored as a list in pos_doc.

Same source as prepare_musique_supporting.py but instead of concatenating the
supporting paragraphs into a single string, stores them as a Python list so that
each paragraph is embedded independently by the embedding model.

Output HF dataset: {HF_USERNAME}/musique-supporting-list
Columns: id, question, answer, answer_aliases, pos_doc (list of strings)
Splits: train, validation — answerable=True only

Usage:
    uv run python datagen/musique/prepare_musique_supporting_list.py
    uv run python datagen/musique/prepare_musique_supporting_list.py --hf-upload mihir-1999/musique-supporting-list
"""

import argparse
import json
import os
import zipfile

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

GDRIVE_URL = "https://drive.google.com/uc?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"


def download_musique(output_dir: str):
    zip_path = os.path.join(output_dir, "musique.zip")
    if not os.path.exists(zip_path):
        try:
            import gdown
        except ImportError:
            raise ImportError("gdown is required: uv add gdown")
        print(f"Downloading MuSiQue from Google Drive...")
        gdown.download(GDRIVE_URL, zip_path, quiet=False)
    else:
        print(f"[cache] {zip_path} already exists, skipping download.")

    extract_dir = os.path.join(output_dir, "extracted")
    if not os.path.exists(extract_dir):
        print(f"Extracting to {extract_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    else:
        print(f"[cache] {extract_dir} already exists, skipping extraction.")

    return extract_dir


def build_pos_doc_list(paragraphs: list) -> list[str]:
    """Return a list of paragraph strings, one per supporting paragraph."""
    return [
        f"**{p['title']}**\n{p['paragraph_text']}"
        for p in paragraphs
        if p.get("is_supporting", False)
    ]


def load_jsonl(path: str) -> list:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_jsonl(extract_dir: str, filename: str) -> str:
    for root, _, files in os.walk(extract_dir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"{filename} not found under {extract_dir}")


def process_rows(rows: list) -> list:
    processed = []
    for row in rows:
        if not row.get("answerable", False):
            continue
        pos_doc_list = build_pos_doc_list(row["paragraphs"])
        if not pos_doc_list:
            # Fallback: use all paragraphs if none marked supporting
            pos_doc_list = [
                f"**{p['title']}**\n{p['paragraph_text']}"
                for p in row["paragraphs"]
            ]
        processed.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "answer_aliases": row.get("answer_aliases", []),
            "pos_doc": pos_doc_list,
            "num_supporting": len(pos_doc_list),
        })
    return processed


def prepare(output_dir: str, hf_upload: str | None, hf_token: str | None):
    os.makedirs(output_dir, exist_ok=True)
    extract_dir = download_musique(output_dir)

    print("\nLoading train split...")
    train_path = find_jsonl(extract_dir, "musique_ans_v1.0_train.jsonl")
    train_rows = process_rows(load_jsonl(train_path))
    print(f"  {len(train_rows)} answerable train rows")

    print("Loading validation split...")
    dev_path = find_jsonl(extract_dir, "musique_ans_v1.0_dev.jsonl")
    dev_rows = process_rows(load_jsonl(dev_path))
    print(f"  {len(dev_rows)} answerable validation rows")

    from collections import Counter
    counts = Counter(r["num_supporting"] for r in dev_rows)
    print(f"\nSupporting paragraph counts in validation: {dict(sorted(counts.items()))}")

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(dev_rows)
    dataset_dict = DatasetDict({"train": train_ds, "validation": val_ds})

    print(f"\nDataset summary:")
    print(f"  train:      {len(train_ds)} rows")
    print(f"  validation: {len(val_ds)} rows")
    print(f"  columns:    {train_ds.column_names}")

    sample = dev_rows[0]
    print(f"\nSample row:")
    print(f"  id:            {sample['id']}")
    print(f"  question:      {sample['question']}")
    print(f"  answer:        {sample['answer']}")
    print(f"  num_supporting: {sample['num_supporting']}")
    print(f"  pos_doc (list of {len(sample['pos_doc'])} paragraphs):")
    for i, para in enumerate(sample["pos_doc"]):
        print(f"    [{i}] {para[:120]}...")

    if hf_upload:
        print(f"\nUploading to {hf_upload}...")
        dataset_dict.push_to_hub(hf_upload, token=hf_token, private=False)
        print(f"Done → https://huggingface.co/datasets/{hf_upload}")
    else:
        local_path = os.path.join(output_dir, "dataset_supporting_list")
        dataset_dict.save_to_disk(local_path)
        print(f"Saved locally to {local_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="datagen/musique/musique_data")
    hf_username = os.environ.get("HF_USERNAME", "").strip()
    default_repo = f"{hf_username}/musique-supporting-list" if hf_username else None
    parser.add_argument("--hf-upload", default=default_repo)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    prepare(args.output_dir, args.hf_upload, token)
