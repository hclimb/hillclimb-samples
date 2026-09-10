"""
NovelHopQA dataset for memory-layers evaluation.

Loads QA rows from abhaygupta1266/novelhopqa and book texts from
{HF_USERNAME}/novelhopqa-books, joining them on the `book` title field
to produce `pos_doc` = full book text per row.

The books dataset is a small separate upload (~50MB, 44 rows) created by:
    python data/utils/prepare_novelhopqa.py
"""

import os

from datasets import load_dataset

from .base import BaseDataset
from .qa import (
    QADataset,
    _StreamingSource,
    _StreamingQAFilter,
    _StreamingQATransform,
)
from .utils import make_normalizer


class NovelHopQADataset(QADataset):
    def __init__(
        self,
        tokenizer=None,
        books_hf_name: str = "",
        split: str = "hop_1",
        prompt_path: str = None,
        **kwargs,
    ):
        # Initialize BaseDataset (sets tokenizer, seq_len, hf_token, etc.)
        # Skip QADataset.__init__ since we build self.dataset ourselves below.
        BaseDataset.__init__(self, tokenizer=tokenizer, split=split, **kwargs)

        # Store QADataset-specific attrs that _build_pipeline needs
        self.provide_docs = kwargs.get("provide_docs", True)
        self.chat_template = kwargs.get("chat_template", True)
        self.force_thinking = kwargs.get("force_thinking", False)
        self.num_chunks_per_doc = kwargs.get("num_chunks_per_doc", 4)
        doc_chunk_seq_len = kwargs.get("doc_chunk_seq_len", self.seq_len)
        self.doc_chunk_seq_len = doc_chunk_seq_len
        self.doc_length = self.doc_chunk_seq_len * self.num_chunks_per_doc
        self.min_doc_length = kwargs.get("min_doc_length", 64)
        self.distill = False

        # Load books dict (small, fits in memory)
        print(f"Loading books from {books_hf_name}...")
        books_ds = load_dataset(books_hf_name, split="train", token=self.hf_token)
        books = {row["title"]: row["text"] for row in books_ds}
        print(f"  Loaded {len(books)} books.")

        # Load QA dataset
        ds = load_dataset(
            "abhaygupta1266/novelhopqa",
            split=split,
            token=self.hf_token,
            streaming=True,
        )
        if self.shuffle:
            ds = ds.shuffle(seed=42, buffer_size=100000)

        # Load prompt template and bake book name in per-item during normalization
        base_tmpl = open(prompt_path or "data/prompts/novelhopqa.txt").read()

        def normalize(item):
            book_name = item.get("book", "")
            book_text = books.get(book_name, "")
            # Bake book name into the prompt template so it appears explicitly
            prompt_template = base_tmpl.replace("{book_name}", book_name)
            return {
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "pos_doc": book_text if book_text else item.get("context", ""),
                "neg_doc": [],
                "prompt_template": prompt_template,
            }

        ds = ds.map(normalize, remove_columns=ds.column_names)

        self.dataset = ds
