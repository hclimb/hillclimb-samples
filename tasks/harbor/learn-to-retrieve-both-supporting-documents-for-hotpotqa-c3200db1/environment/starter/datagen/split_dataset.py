#!/usr/bin/env python3
"""
Split an existing HuggingFace dataset into train/validation/test splits
and push back to the same repo.

Usage:
    python datagen/split_dataset.py --repo mihir-1999/tinystories-qa-3m
    python datagen/split_dataset.py --repo mihir-1999/nemotron-qa --val-size 0.04 --test-size 0.01
"""

import argparse
import logging
import os

import dotenv
dotenv.load_dotenv()

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

VAL_SIZE  = 0.04
TEST_SIZE = 0.01


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo",      required=True,            help="HF dataset repo to split")
    parser.add_argument("--val-size",  type=float, default=VAL_SIZE,  help="Fraction for validation (default: 0.04)")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="Fraction for test (default: 0.01)")
    parser.add_argument("--private",   action="store_true",      help="Keep repo private")
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    logging.info(f"Loading {args.repo}...")
    ds = load_dataset(args.repo, split="train", token=hf_token)
    logging.info(f"Loaded {len(ds)} rows.")

    # First split off test
    split1 = ds.train_test_split(test_size=args.test_size, seed=42)
    train_val = split1["train"]
    test_ds   = split1["test"]

    # Then split val from remaining
    val_fraction = args.val_size / (1.0 - args.test_size)
    split2 = train_val.train_test_split(test_size=val_fraction, seed=42)
    train_ds = split2["train"]
    val_ds   = split2["test"]

    logging.info(f"train: {len(train_ds)} | validation: {len(val_ds)} | test: {len(test_ds)}")

    logging.info(f"Pushing splits to {args.repo}...")
    train_ds.push_to_hub(args.repo, split="train",      token=hf_token, private=args.private)
    val_ds.push_to_hub(  args.repo, split="validation", token=hf_token, private=args.private)
    test_ds.push_to_hub( args.repo, split="test",       token=hf_token, private=args.private)

    logging.info("Done.")


if __name__ == "__main__":
    main()
