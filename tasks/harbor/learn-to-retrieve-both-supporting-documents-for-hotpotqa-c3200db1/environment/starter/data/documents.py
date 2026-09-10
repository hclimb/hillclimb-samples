"""Dataset for iterating over raw document chunks from a HuggingFace dataset."""

import numpy as np
from datasets import load_dataset
import os


class DocumentsDataset:
    def __init__(
        self,
        tokenizer,
        hf_name,
        hf_config=None,
        split="train",
        column="pos_doc",
        normalizer_type=None,
        batch_size=64,
        max_docs=None,
        chunk_size=256,
        max_chunks_per_doc=4,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.hf_name = hf_name
        self.split = split
        self.column = column
        self.normalizer_type = normalizer_type
        self.batch_size = batch_size
        self.max_docs = max_docs
        self.chunk_size = chunk_size
        self.max_chunks_per_doc = max_chunks_per_doc
        hf_token = os.environ.get("HF_TOKEN", None)
        self._ds = load_dataset(hf_name, hf_config, split=split, streaming=True, token=hf_token)

    def generator(self):
        """Yield (ids_batch, masks_batch, doc_id_batch) tuples of shape (B, chunk_size).

        Tokenizes each doc on the fly into chunk_size-token chunks (right-padded),
        buffers them, yields in batch_size increments, stops at max_docs total chunks.

        doc_id_batch contains the integer document ID for each chunk. When the dataset
        has an "id" column (added by prepare_*.py scripts), that value is used; otherwise
        a sequential counter is assigned. All chunks from the same source document share
        the same doc_id, enabling document-level retrieval accuracy tracking.

        When normalizer_type="flashrag", expands each row's is_selected=1 passages
        into individual documents (one passage → one or more chunks).
        """
        tokenizer = self.tokenizer
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        chunk_size = self.chunk_size
        max_chunks_per_doc = self.max_chunks_per_doc
        batch_size = self.batch_size
        max_docs = self.max_docs
        max_chunks = self.max_docs * self.max_chunks_per_doc if max_docs is not None else None

        buf_ids = []
        buf_masks = []
        buf_doc_ids = []
        total = 0
        doc_counter = 0

        for item in self._ds:
            if self.normalizer_type == "flashrag":
                passages_data = (item.get("metadata") or {}).get("passages") or {}
                texts = passages_data.get("passage_text", [])
                selected = passages_data.get("is_selected", [])
                doc_texts = [t for t, s in zip(texts, selected) if s == 1]
            elif self.normalizer_type == "hotpotqa":
                context = item.get("context") or {}
                titles = context.get("title", [])
                sentences_list = context.get("sentences", [])
                doc_texts = [
                    t + ": " + " ".join(s) if isinstance(s, list) else t
                    for t, s in zip(titles, sentences_list)
                ]
            else:
                text = item.get(self.column, "")
                doc_texts = [text] if text else []

            for text in doc_texts:
                if not text:
                    continue

                # Use the dataset's "id" field when available (single-text-per-item
                # normalizer only); fall back to a running counter for multi-text
                # normalizers or datasets that pre-date the "id" column.
                if self.normalizer_type is None and "id" in item:
                    doc_id = int(item["id"])
                else:
                    doc_id = doc_counter
                doc_counter += 1

                # Keep chunk tokenization aligned with data.utils.chunk_text so the
                # fallback doc-hit matching path can compare chunk bytes directly.
                full_text = text + (tokenizer.eos_token or "<|im_end|>")
                ids = tokenizer(full_text, return_tensors="np", truncation=False)["input_ids"][0]
                chunks = [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]
                chunks = chunks[:max_chunks_per_doc]

                for chunk in chunks:
                    pad_len = chunk_size - len(chunk)
                    buf_ids.append(np.concatenate([chunk, np.full(pad_len, pad_id, dtype=chunk.dtype)]))
                    buf_masks.append(np.concatenate([
                        np.ones(len(chunk), dtype=np.bool_),
                        np.zeros(pad_len, dtype=np.bool_),
                    ]))
                    buf_doc_ids.append(doc_id)
                    total += 1

                    if len(buf_ids) >= batch_size:
                        yield np.stack(buf_ids), np.stack(buf_masks), np.array(buf_doc_ids, dtype=np.int64)
                        buf_ids = []
                        buf_masks = []
                        buf_doc_ids = []

                    if max_chunks is not None and total >= max_chunks:
                        if buf_ids:
                            yield np.stack(buf_ids), np.stack(buf_masks), np.array(buf_doc_ids, dtype=np.int64)
                        return

        if buf_ids:
            yield np.stack(buf_ids), np.stack(buf_masks), np.array(buf_doc_ids, dtype=np.int64)
