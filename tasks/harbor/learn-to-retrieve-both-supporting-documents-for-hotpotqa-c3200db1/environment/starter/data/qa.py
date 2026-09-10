import os
import glob
import pickle
import json
import random
import time

import jax.numpy as jnp
import numpy as np
import grain.python as grain
from grain._src.python.dataset.dataset import DatasetIterator
from datasets import load_dataset, interleave_datasets, load_from_disk
from typing import Optional, Dict, Any
from .base import BaseDataset
from .utils import build_prefix_text, chunk_text, make_attn_mask, pack_docs, make_normalizer, make_flashrag_normalizer, make_hotpotqa_normalizer, make_docid_normalizer


# TODO: Currently each worker is downloading the entire data stream; it could be optimised to only take specific shard!

class _StreamingIterator(DatasetIterator):
    def __init__(self, iterable, sl=None):
        super().__init__()
        self._iterable = iterable
        self._iter = None
        self._sl = sl
        self._count = 0
        self._target_count = 0

    def __next__(self):
        if self._iter is None:
            self._iter = iter(self._iterable)
        # Fast-forward after set_state: the underlying HF stream is deterministic
        # (fixed shuffle/interleave seeds), so consuming `count` raw items replays
        # the pipeline to the exact position of the saved iterator.
        if self._count < self._target_count:
            print(f"Dataloader fast-forward: skipping {self._target_count - self._count} consumed source items...")
            while self._count < self._target_count:
                next(self._iter)
                self._count += 1
            print("Dataloader fast-forward done.")
        while True:
            item = next(self._iter)
            if self._sl is not None:
                start = self._sl.start or 0
                step = self._sl.step or 1
                if self._count % step != start % step:
                    self._count += 1
                    continue
            self._count += 1
            return item

    def get_state(self):
        return {"count": self._count}

    def set_state(self, state):
        # Don't touch self._count: the fresh HF stream is at position 0, so we
        # record a target and fast-forward lazily on first __next__.
        self._target_count = state.get("count", 0)


class _StreamingSource(grain.IterDataset):
    def __init__(self, iterable):
        super().__init__()
        self._iterable = iterable
        self._slice = None

    def set_slice(self, sl, sequential_slice=False):
        self._slice = sl

    def __iter__(self):
        return _StreamingIterator(self._iterable, self._slice)


# ── Module-level filter + transform (called by both streaming and preprocessing) ─────────
#
# These were previously the .filter() / .map() methods of the two Grain transform classes
# below. Extracted so data/preprocess_arrayrecord.py can call them directly on raw parquet
# rows during the one-time tokenize+filter+shard pass. The Grain transform classes now
# delegate to them for backward compat with the streaming pipeline (QADataset).

def qa_filter_predicate(
    raw_item, tokenizer, seq_len, chat_template, doc_length,
    force_thinking=False, min_doc_length=64, filter_doc_length=True,
):
    """Returns True if the raw item survives all training-sample filters.

    Kept a pure function so preprocessing can compute post-filter N deterministically.
    Any change to the filter logic invalidates preprocessed shards (config-hash gate).
    """
    # Filter: drop rows where pos_doc is < min_doc_length or > doc_length tokens
    # (skip upper-bound check when filter_doc_length=False, e.g. for gen_large_mem)
    pos_doc = raw_item.get("pos_doc", "")
    if isinstance(pos_doc, list):
        pos_doc = pos_doc[0] if pos_doc else ""
    if pos_doc:
        doc_tok_len = len(tokenizer(pos_doc, truncation=False, return_tensors="np")["input_ids"][0])
        if doc_tok_len < min_doc_length:
            return False
        if filter_doc_length and doc_tok_len > doc_length:
            return False

    # Per-source min_neg_docs filter (attached to row by make_normalizer).
    # Missing key (e.g. from hotpotqa/flashrag normalizers) defaults to 0 → no filter.
    min_neg_docs = raw_item.get("_min_neg_docs", 0)
    if min_neg_docs > 0:
        neg_doc = raw_item.get("neg_doc", [])
        if not isinstance(neg_doc, list):
            neg_doc = [neg_doc] if neg_doc else []
        if len(neg_doc) > 0 and len(neg_doc) < min_neg_docs:
            return False

    # Filter: drop rows where formatted question+answer exceeds seq_len
    question = raw_item.get("question", "")
    answer = raw_item.get("answer", "")
    if answer and question and answer.lower() in question.lower():
        if not any(opt in answer for opt in ["A)", "B)", "C)", "D)"]):
            return False
    if question and chat_template:
        has_thinking = force_thinking or answer.startswith("<think>")
        prefix_text = build_prefix_text(
            tokenizer, question, raw_item.get("prompt_template"), chat_template, has_thinking
        )
        full_text = prefix_text + answer + (tokenizer.eos_token or "<|im_end|>")
    else:
        full_text = f"Question: {question}\nAnswer: {answer}" + (tokenizer.eos_token or "<|im_end|>") if question else ""
    if full_text:
        full_tok_len = len(tokenizer(full_text, truncation=False, return_tensors="np")["input_ids"][0])
        if full_tok_len > seq_len:
            return False
    return True


def qa_transform_item(
    item, tokenizer, seq_len, doc_chunk_seq_len, num_chunks_per_doc,
    mask_prefix, chat_template, force_thinking=False,
):
    """Tokenize + build masks + pack_docs. Returns the training-sample dict.

    Called by both the streaming pipeline (via _StreamingQATransform.map wrapper)
    and by preprocess_arrayrecord.py (directly, at shard-write time).
    """
    question = item.get("question", "")
    answer = item.get("answer", "")
    pos_docs_raw = item.get("pos_doc", [])
    neg_docs_raw = item.get("neg_doc", [])
    if isinstance(pos_docs_raw, str): pos_docs_raw = [pos_docs_raw]
    if isinstance(neg_docs_raw, str): neg_docs_raw = [neg_docs_raw]

    # Ablation hook (see wiki/experiments/2026-08-13-doc-access-per-query-loss-investigation.md,
    # "Ablation" section): cot_doc, if the source's cot_field config forwarded one (see
    # data/utils.py::_docid_normalize), gets appended as one more entry in the positives list.
    # pack_docs marks every entry in pos_docs_raw with pos_mask=1 uniformly, so this alone makes
    # the CoT text a retrievable, mask-correct positive document -- no separate pos_doc_mask
    # edit needed. Tests whether doc_access_per_query_loss can collapse toward zero when the
    # answer is trivially present in a memory-bank doc, vs. plateauing on genuinely-hard
    # negatives regardless.
    cot_doc = item.get("cot_doc")
    if cot_doc:
        pos_docs_raw = list(pos_docs_raw) + [cot_doc]

    has_thinking = force_thinking or answer.startswith("<think>")
    prefix_text = build_prefix_text(
        tokenizer, question, item.get("prompt_template"), chat_template, has_thinking
    )
    text = prefix_text + answer + (tokenizer.eos_token or "<|im_end|>")

    tokens = tokenizer(
        text, truncation=True, max_length=seq_len,
        padding="max_length", return_tensors='np'
    )['input_ids'][0]

    attention_mask = make_attn_mask(tokens, tokenizer.pad_token_id)
    loss_mask = attention_mask.copy()

    if mask_prefix:
        prefix_tokens = tokenizer(
            prefix_text, truncation=True, max_length=seq_len, return_tensors='np'
        )['input_ids'][0]
        loss_mask[:min(len(prefix_tokens), len(tokens))] = 0.0

    final_doc_chunks, final_doc_masks, final_pos_mask = pack_docs(
        tokenizer, pos_docs_raw, neg_docs_raw, num_chunks_per_doc, doc_chunk_seq_len
    )

    # pos_doc_ids: corpus-level integer IDs for each positive document, padded
    # to num_chunks_per_doc with -1.  Present only when the dataset was augmented
    # by a prepare_*_with_ids.py script; otherwise all -1s (acc not computed).
    raw_ids = item.get("pos_doc_ids") or []
    pos_doc_ids = np.full(num_chunks_per_doc, -1, dtype=np.int64)
    for j, doc_id in enumerate(raw_ids[:num_chunks_per_doc]):
        pos_doc_ids[j] = int(doc_id)

    return {
        "inputs": tokens, "attn_mask": attention_mask, "loss_mask": loss_mask,
        "docs": final_doc_chunks, "docs_masks": final_doc_masks,
        "pos_doc_mask": final_pos_mask, "pos_doc_ids": pos_doc_ids,
        # Per-row CE gate: 0.0 masks cross-entropy for this row (retrieval-only).
        "ce_enable": np.float32(item.get("_ce_enable", 1.0)),
    }


class _StreamingQAFilter(grain.FilterTransform):
    """Thin Grain-transform wrapper around qa_filter_predicate. Preserves the
    streaming pipeline (QADataset) — new preprocessing calls the function directly."""
    def __init__(self, tokenizer, seq_len, chat_template, doc_length, force_thinking=False, min_doc_length=64, filter_doc_length=True):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.chat_template = chat_template
        self.doc_length = doc_length
        self.force_thinking = force_thinking
        self.min_doc_length = min_doc_length
        self.filter_doc_length = filter_doc_length

    def filter(self, raw_item):
        return qa_filter_predicate(
            raw_item, self.tokenizer, self.seq_len, self.chat_template, self.doc_length,
            force_thinking=self.force_thinking, min_doc_length=self.min_doc_length,
            filter_doc_length=self.filter_doc_length,
        )


class _StreamingQATransform(grain.MapTransform):
    """Thin Grain-transform wrapper around qa_transform_item."""
    def __init__(self, tokenizer, seq_len, doc_chunk_seq_len, num_chunks_per_doc, mask_prefix, chat_template, force_thinking=False):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.doc_chunk_seq_len = doc_chunk_seq_len
        self.num_chunks_per_doc = num_chunks_per_doc
        self.mask_prefix = mask_prefix
        self.chat_template = chat_template
        self.force_thinking = force_thinking

    def map(self, item):
        return qa_transform_item(
            item, self.tokenizer, self.seq_len, self.doc_chunk_seq_len,
            self.num_chunks_per_doc, self.mask_prefix, self.chat_template,
            force_thinking=self.force_thinking,
        )


class _DistillQATransform(grain.MapTransform):
    """Produces student inputs (batch, docs, masks) and teacher inputs (batch, masks).

    The teacher sees [doc | question | answer] as a flat sequence.
    The student sees [question | answer] with docs routed through the embedding model.
    Guarantees exact answer token alignment for distillation using distill_masks.
    """

    def __init__(self, tokenizer, seq_len, doc_chunk_seq_len, num_chunks_per_doc, mask_prefix, chat_template, force_thinking=False):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.doc_chunk_seq_len = doc_chunk_seq_len
        self.num_chunks_per_doc = num_chunks_per_doc
        self.mask_prefix = mask_prefix
        self.chat_template = chat_template
        self.force_thinking = force_thinking
        self.teacher_seq_len = (doc_chunk_seq_len * num_chunks_per_doc) + seq_len

    def map(self, item):
        question = item.get("question", "")
        answer = item.get("answer", "")
        pos_docs_raw = item.get("pos_doc", [])
        neg_docs_raw = item.get("neg_doc", [])
        if isinstance(pos_docs_raw, str): pos_docs_raw = [pos_docs_raw]
        if isinstance(neg_docs_raw, str): neg_docs_raw = [neg_docs_raw]

        has_thinking = self.force_thinking or answer.startswith("<think>")

        # ---------------- STUDENT TOKENS ----------------
        student_prefix_text = build_prefix_text(
            self.tokenizer, question, item.get("student_prompt_template"), self.chat_template, has_thinking
        )
        student_prefix_tokens = self.tokenizer(student_prefix_text, padding=False, return_tensors='np')['input_ids'][0]

        # Tokenize answer explicitly — same tokens shared by student and teacher
        answer_text = answer + (self.tokenizer.eos_token or "<|im_end|>")
        answer_tokens = self.tokenizer(answer_text, add_special_tokens=False, return_tensors='np')['input_ids'][0]

        # Truncate answer to fit student sequence
        answer_tokens_for_student = answer_tokens[:max(0, self.seq_len - len(student_prefix_tokens))]

        pad_val = self.tokenizer.pad_token_id or 0
        student_tokens_unpadded = np.concatenate([student_prefix_tokens, answer_tokens_for_student])
        student_tokens = np.full((self.seq_len,), pad_val, dtype=np.int32)
        student_tokens[:len(student_tokens_unpadded)] = student_tokens_unpadded

        student_attn_mask = make_attn_mask(student_tokens, self.tokenizer.pad_token_id)
        student_loss_mask = student_attn_mask.copy()
        if self.mask_prefix:
            student_loss_mask[:len(student_prefix_tokens)] = 0.0
        student_distill_mask = student_loss_mask.copy()

        # ---------------- TEACHER TOKENS ----------------
        pos_doc_text = "\n\n".join(d for d in pos_docs_raw if d)
        teacher_tmpl = item.get("teacher_prompt_template")
        if teacher_tmpl:
            teacher_content = teacher_tmpl.format(document=pos_doc_text, question=question).rstrip("\n")
        else:
            teacher_content = f"Document:\n{pos_doc_text}\n\nQuestion:\n{question}" if pos_doc_text else question

        if self.chat_template:
            teacher_prefix_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": teacher_content}],
                tokenize=False, add_generation_prompt=True, enable_thinking=has_thinking
            )
        elif teacher_tmpl:
            teacher_prefix_text = teacher_content
        else:
            teacher_prefix_text = f"Context: {pos_doc_text}\nQuestion: {question}\nAnswer: "

        teacher_prefix_tokens = self.tokenizer(teacher_prefix_text, padding=False, return_tensors='np')['input_ids'][0]

        # Left-truncate teacher prefix so the question at the end remains perfectly intact
        teacher_prefix_tokens = teacher_prefix_tokens[-max(0, self.teacher_seq_len - len(answer_tokens_for_student)):]

        teacher_tokens_unpadded = np.concatenate([teacher_prefix_tokens, answer_tokens_for_student])
        teacher_tokens = np.full((self.teacher_seq_len,), pad_val, dtype=np.int32)
        teacher_tokens[:len(teacher_tokens_unpadded)] = teacher_tokens_unpadded

        teacher_attn_mask = make_attn_mask(teacher_tokens, self.tokenizer.pad_token_id)
        teacher_distill_mask = np.zeros_like(teacher_tokens, dtype=np.float32)
        teacher_distill_mask[len(teacher_prefix_tokens):len(teacher_tokens_unpadded)] = 1.0

        # ---------------- DOCS ----------------
        final_doc_chunks, final_doc_masks, final_pos_mask = pack_docs(
            self.tokenizer, pos_docs_raw, neg_docs_raw, self.num_chunks_per_doc, self.doc_chunk_seq_len
        )

        return {
            "inputs": student_tokens,
            "attn_mask": student_attn_mask,
            "loss_mask": student_loss_mask,
            "docs": final_doc_chunks,
            "docs_masks": final_doc_masks,
            "pos_doc_mask": final_pos_mask,
            "teacher_inputs": teacher_tokens,
            "teacher_attn_mask": teacher_attn_mask,
            "student_distill_mask": student_distill_mask,
            "teacher_distill_mask": teacher_distill_mask,
        }


def _hf_retry(what, fn, max_attempts=12):
    """Run fn() with exponential backoff on HF 429 rate limits.

    Many streaming sources resolved concurrently (multiple VMs × grain workers)
    can exceed the HF API quota (1000 req / 5 min). Any dataset-building step can
    trip it (load_dataset, .map feature resolution, interleave_datasets); waiting
    out the window succeeds, so a hard crash would needlessly kill training.
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            if "429" not in str(e) or attempt == max_attempts - 1:
                raise
            delay = min(60 * (attempt + 1), 330) + random.uniform(0, 30)
            print(f"HF 429 during {what}; retrying in {delay:.0f}s (attempt {attempt + 1}/{max_attempts})", flush=True)
            time.sleep(delay)


def _load_dataset_with_backoff(name, hf_config, split, token):
    # Offline mode (HF_HUB_OFFLINE=1, datasets pre-cached into e.g. /dev/shm/hf): streaming=True
    # can't be served from a local snapshot, so build the dataset from the cached files
    # (streaming=False, no network) and re-expose the SAME IterableDataset streaming interface via
    # to_iterable_dataset — downstream .shuffle(buffer_size=...)/interleave_datasets are unchanged.
    # This makes the RAM-cache route work end-to-end and issues zero HF API calls (no 429).
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        # Stream the pre-downloaded local parquet shards (scripts/misc/precache_hf.sh put a subset
        # under $GROUND_HF_PARQUET/<repo-with-slashes-as-__>/). load_dataset("parquet",
        # data_files=<local files>, streaming=True) reads the files directly — no
        # dataset_module_factory, no Hub call, no arrow-build doubling — and returns the same
        # IterableDataset the pipeline expects. Falls through if the local subset is absent.
        base = os.environ.get("GROUND_HF_PARQUET", os.path.expanduser("~/hf_parquet"))
        d = os.path.join(base, name.replace("/", "__"))
        if os.path.isdir(d):
            files = sorted(glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True))
            if files:
                return _hf_retry(f"load {name} (local parquet x{len(files)})",
                                 lambda: load_dataset("parquet", data_files=files, split=split, streaming=True))
    return _hf_retry(f"load {name}", lambda: load_dataset(name, hf_config, split=split, token=token, streaming=True))


# ── Main dataset class ──────────────────────────────────────────────

class QADataset(BaseDataset):
    def __init__(
        self,
        tokenizer=None,
        provide_docs: bool = True,
        chat_template: bool = True,
        num_chunks_per_doc: int = 16,
        split: str = "train",
        seq_len: int = 256,
        doc_chunk_seq_len: Optional[int] = None,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 0,
        mask_prefix: bool = True,
        min_doc_length: int = 64,
        hf_name: str = "",               # single-dataset mode
        field_map=None,                  # single-dataset mode
        think_field=None,                # single-dataset mode
        doc_separator=None,              # single-dataset mode: split pos_doc/neg_doc/neg_scores on this string
        force_thinking: bool = False,    # supported in all modes
        prompt_path=None,                # single-dataset mode (was student_prompt_path)
        sources: Dict = {},              # interleave mode (takes precedence)
        distill: bool = False,           # True → yield teacher+student with distill masks
        shuffle_seed: int = 42,          # shuffle/interleave seed (override on preemption-relaunch)
        **kwargs
    ):
        super().__init__(
            tokenizer=tokenizer,
            split=split,
            seq_len=seq_len,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            mask_prefix=mask_prefix,
            **kwargs
        )
        self.provide_docs = provide_docs
        self.chat_template = chat_template
        self.force_thinking = force_thinking
        self.doc_separator = doc_separator
        self.num_chunks_per_doc = num_chunks_per_doc
        self.min_doc_length = min_doc_length
        self.doc_chunk_seq_len = doc_chunk_seq_len if doc_chunk_seq_len is not None else seq_len
        self.doc_length = self.doc_chunk_seq_len * self.num_chunks_per_doc
        self.distill = distill
        self.shuffle_seed = int(shuffle_seed)

        if sources:
            # Interleave mode
            datasets = []
            weights = []
            for source in sources.values():
                name = source["hf_name"]
                hf_config = source.get("hf_config")
                field_map_s = source.get("field_map")
                think_field_s = source.get("think_field")
                doc_separator_s = source.get("doc_separator")
                normalizer_type = source.get("normalizer_type")
                neg_score_threshold_s = source.get("neg_score_threshold")
                min_neg_docs_s = source.get("min_neg_docs", 0)
                mask_ce_s = source.get("mask_ce", False)
                cot_field_s = source.get("cot_field")

                weights.append(float(source.get("weight", 1.0)))

                ds = _load_dataset_with_backoff(name, hf_config, split, self.hf_token)
                if self.shuffle:
                    ds = ds.shuffle(seed=self.shuffle_seed, buffer_size=100000)
                if normalizer_type == "flashrag":
                    normalizer = make_flashrag_normalizer(source.get("flashrag_context_key"))
                elif normalizer_type == "hotpotqa":
                    normalizer = make_hotpotqa_normalizer()
                elif normalizer_type == "docid":
                    normalizer = make_docid_normalizer(
                        source["corpus_path"], think_field=think_field_s,
                        max_neg_docs=source.get("max_neg_docs"),
                        min_neg_docs=min_neg_docs_s, mask_ce=mask_ce_s,
                        field_map=field_map_s, cot_field=cot_field_s)
                else:
                    normalizer = make_normalizer(field_map_s, think_field_s, doc_separator_s,
                                                 neg_score_threshold=neg_score_threshold_s,
                                                 min_neg_docs=min_neg_docs_s, mask_ce=mask_ce_s)
                ds = _hf_retry(f"map {name}", lambda ds=ds, n=normalizer: ds.map(n, remove_columns=ds.column_names))

                if distill:
                    teacher_prompt_path = source.get("teacher_prompt_path")
                    student_prompt_path = source.get("student_prompt_path")
                    teacher_tmpl = open(teacher_prompt_path or "data/prompts/teacher_default.txt").read()
                    student_tmpl = open(student_prompt_path or "data/prompts/default.txt").read()
                    def add_templates(item, _t=teacher_tmpl, _s=student_tmpl):
                        item["teacher_prompt_template"] = _t
                        item["student_prompt_template"] = _s
                        return item
                    ds = ds.map(add_templates)
                else:
                    prompt_path_s = source.get("prompt_path")
                    tmpl = open(prompt_path_s or "data/prompts/default.txt").read()
                    def add_template(item, _s=tmpl):
                        item["prompt_template"] = _s
                        return item
                    ds = ds.map(add_template)

                datasets.append(ds)

            # Per-source sampling weights: uniform unless any source sets `weight`.
            if any(w != 1.0 for w in weights):
                total = sum(weights)
                probabilities = [w / total for w in weights]
                print(f"Interleaving {len(datasets)} sources with probabilities {probabilities}")
                self.dataset = _hf_retry("interleave", lambda: interleave_datasets(
                    datasets, probabilities=probabilities, seed=self.shuffle_seed,
                    stopping_strategy="all_exhausted",
                ))
            else:
                self.dataset = _hf_retry("interleave", lambda: interleave_datasets(datasets, seed=self.shuffle_seed, stopping_strategy="all_exhausted"))
        elif hf_name:
            # Single-dataset mode
            hf_config = kwargs.pop("hf_config", None)
            normalizer_type = kwargs.pop("normalizer_type", None)
            flashrag_context_key = kwargs.pop("flashrag_context_key", None)
            neg_score_threshold = kwargs.pop("neg_score_threshold", None)
            min_neg_docs_single = kwargs.pop("min_neg_docs", 0)
            mask_ce_single = kwargs.pop("mask_ce", False)
            corpus_path_single = kwargs.pop("corpus_path", None)
            max_neg_docs_single = kwargs.pop("max_neg_docs", None)
            cot_field_single = kwargs.pop("cot_field", None)
            ds = _load_dataset_with_backoff(hf_name, hf_config, split, self.hf_token)
            if self.shuffle:
                ds = ds.shuffle(seed=self.shuffle_seed, buffer_size=100000)
            if normalizer_type == "flashrag":
                normalizer = make_flashrag_normalizer(flashrag_context_key)
            elif normalizer_type == "hotpotqa":
                normalizer = make_hotpotqa_normalizer()
            elif normalizer_type == "docid":
                normalizer = make_docid_normalizer(
                    corpus_path_single, think_field=think_field,
                    max_neg_docs=max_neg_docs_single,
                    min_neg_docs=min_neg_docs_single, mask_ce=mask_ce_single,
                    field_map=field_map, cot_field=cot_field_single)
            else:
                normalizer = make_normalizer(field_map, think_field, self.doc_separator,
                                             neg_score_threshold=neg_score_threshold,
                                             min_neg_docs=min_neg_docs_single, mask_ce=mask_ce_single)
            ds = _hf_retry("map single", lambda ds=ds: ds.map(normalizer, remove_columns=ds.column_names))
            tmpl = open(prompt_path or "data/prompts/default.txt").read()
            def add_template(item, _s=tmpl):
                item["prompt_template"] = _s
                return item
            ds = ds.map(add_template)
            self.dataset = ds
        else:
            raise ValueError("Either hf_name or sources must be provided")

    def _build_pipeline(self):
        """Build the grain IterDataset pipeline with filter → map → mp_prefetch → batch."""
        pipeline = _StreamingSource(self.dataset)
        pipeline = pipeline.filter(_StreamingQAFilter(
            self.tokenizer, self.seq_len, self.chat_template, self.doc_length, self.force_thinking,
            min_doc_length=self.min_doc_length,
            filter_doc_length=self.provide_docs,
        ))
        if self.distill:
            pipeline = pipeline.map(_DistillQATransform(
                tokenizer=self.tokenizer,
                seq_len=self.seq_len,
                doc_chunk_seq_len=self.doc_chunk_seq_len,
                num_chunks_per_doc=self.num_chunks_per_doc,
                mask_prefix=self.mask_prefix,
                chat_template=self.chat_template,
                force_thinking=self.force_thinking,
            ))
        else:
            pipeline = pipeline.map(_StreamingQATransform(
                tokenizer=self.tokenizer,
                seq_len=self.seq_len,
                doc_chunk_seq_len=self.doc_chunk_seq_len,
                num_chunks_per_doc=self.num_chunks_per_doc,
                mask_prefix=self.mask_prefix,
                chat_template=self.chat_template,
                force_thinking=self.force_thinking,
            ))
        if self.num_workers > 0:
            pipeline = pipeline.mp_prefetch(
                grain.MultiprocessingOptions(
                    num_workers=self.num_workers,
                    per_worker_buffer_size=4,
                )
            )
        pipeline = pipeline.batch(self.batch_size, drop_remainder=True)
        return pipeline

    def get_loader_state(self):
        """Serializable state of the live training iterator (None if unavailable)."""
        it = getattr(self, "current_iterator", None)
        if it is None:
            return None
        try:
            return it.get_state()
        except Exception as e:
            import warnings
            warnings.warn(f"Could not capture dataloader state: {e}")
            return None

    def set_loader_state(self, state):
        """Request that the next generator() pipeline resumes from `state`.

        Exactness requires the same dataset config AND the same shuffle_seed as the
        run that saved the state (the stream position is a raw-item count into the
        deterministic shuffled/interleaved stream).
        """
        self._pending_loader_state = state

    def generator(self, num_epochs=None):
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for training generator")

        epochs = num_epochs if num_epochs is not None else 1000000

        for _ in range(epochs):
            pipeline = self._build_pipeline()
            iterator = iter(pipeline)
            pending = getattr(self, "_pending_loader_state", None)
            if pending is not None:
                try:
                    iterator.set_state(pending)
                    print("Dataloader state restored; stream will fast-forward on first batch.")
                except Exception as e:
                    import warnings
                    warnings.warn(f"Could not restore dataloader state ({e}); starting stream from 0.")
                self._pending_loader_state = None
            self.current_iterator = iterator

            while True:
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                except Exception as e:
                    import warnings
                    warnings.warn(f"Dataset iterator crashed (likely due to worker timeout/OOM during JIT compilation). "
                                f"Rebuilding pipeline automatically. Error: {e}")
                    if "429" in str(e):
                        # Rebuilding re-resolves every source across all workers; without a
                        # pause this tight-loops against the HF rate limit.
                        delay = 60 + random.uniform(0, 120)
                        print(f"429 during pipeline build; sleeping {delay:.0f}s before rebuild", flush=True)
                        time.sleep(delay)
                    del iterator
                    del pipeline
                    pipeline = self._build_pipeline()
                    iterator = iter(pipeline)
                    self.current_iterator = iterator
                    continue

                inputs = jnp.array(batch['inputs'])
                attn_mask = jnp.array(batch['attn_mask'])
                loss_mask = jnp.array(batch['loss_mask'])

                docs = jnp.array(batch['docs'])
                docs_masks = jnp.array(batch['docs_masks'])
                pos_doc_mask = jnp.array(batch['pos_doc_mask'])
                pos_doc_ids = jnp.array(batch['pos_doc_ids'])   # (B, num_chunks_per_doc)
                ce_enable = jnp.array(batch['ce_enable']) if 'ce_enable' in batch else jnp.ones(inputs.shape[0], dtype=jnp.float32)  # (B,)

                B, M, d_seq_len = docs.shape
                flat_docs = docs.reshape(B * M, d_seq_len)
                flat_docs_masks = docs_masks.reshape(B * M, d_seq_len)

                if self.distill:
                    teacher_inputs = jnp.array(batch['teacher_inputs'])
                    teacher_attn_mask = jnp.array(batch['teacher_attn_mask'])
                    student_distill_mask = jnp.array(batch['student_distill_mask'])
                    teacher_distill_mask = jnp.array(batch['teacher_distill_mask'])
                    yield {"batch": inputs, "docs": flat_docs, "teacher_batch": teacher_inputs}, {
                        "batch_mask": attn_mask, "docs_mask": flat_docs_masks, "loss_mask": loss_mask,
                        "pos_doc_mask": pos_doc_mask, "pos_doc_ids": pos_doc_ids,
                        "teacher_mask": teacher_attn_mask,
                        "student_distill_mask": student_distill_mask, "teacher_distill_mask": teacher_distill_mask,
                        "ce_enable": ce_enable,
                    }
                elif self.provide_docs:
                    yield {"batch": inputs, "docs": flat_docs}, {
                        "batch_mask": attn_mask, "docs_mask": flat_docs_masks,
                        "loss_mask": loss_mask, "pos_doc_mask": pos_doc_mask,
                        "pos_doc_ids": pos_doc_ids, "ce_enable": ce_enable,
                    }
                else:
                    yield inputs, {
                        "batch_mask": attn_mask, "loss_mask": loss_mask,
                        "pos_doc_ids": pos_doc_ids, "ce_enable": ce_enable,
                    }

            # Explicit cleanup to avoid shared memory file leaks
            del iterator
            del pipeline

    def eval_qa_completion_generator(self):
        """Yields (question, answer) for QA Completion Eval."""
        for item in self.dataset:
            yield item.get('question', ''), item.get('answer', '')


# ── QADatasetIndexed: random-access-by-index path (data_resume rework) ─────────

class QADatasetIndexed(BaseDataset):
    """Reads pre-tokenized samples from ArrayRecord shards written by
    data/preprocess_arrayrecord.py, sampled via grain.IndexSampler.

    Compared to QADataset (streaming path):
      - No filter+tokenize at train time: those ran once during preprocessing.
      - No shuffle_buffer, no interleave: IndexSampler produces a globally-uniform
        permutation over the full sample pool.
      - Resume is O(1) via IndexSampler's iterator state — no fast-forward.
      - `is_indexed = True` sentinel is retained as a dataset-type marker (some
        callers still branch on it), but the trainer no longer skips save/restore
        on it: the position at step N is `step * batch_size`, either read from
        the sibling `dataloader_state.json` (post-fix ckpt) or synthesized in
        closed form here when the file is missing/stale (pre-fix ckpt or
        overwritten). See `set_loader_state` + `generator` below.

    Config keys (in dataset yaml):
      storage: arrayrecord      # dispatch marker in data/__init__.py::get_dataset
      indexed_uri: gs://memory-layers-training/indexed/<config-hash>
      num_epochs: 10            # sampler's upper bound, NOT training length
      shuffle_seed: 42          # sampler seed
      batch_size, num_workers   # inherited from BaseDataset

    Reads metadata.json from indexed_uri to discover shard list + N.
    """

    is_indexed = True  # trainer guards on this sentinel

    def __init__(
        self,
        tokenizer=None,
        indexed_uri: str = "",
        num_epochs: int = 10,
        shuffle_seed: int = 42,
        provide_docs: bool = True,
        distill: bool = False,
        # BaseDataset kwargs
        split: str = "train",
        seq_len: int = 256,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 16,
        mask_prefix: bool = True,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer, split=split, seq_len=seq_len,
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            mask_prefix=mask_prefix, **kwargs,
        )
        if not indexed_uri:
            raise ValueError(
                "QADatasetIndexed requires cfg.dataset.indexed_uri (e.g. "
                "gs://memory-layers-training/indexed/<config-hash>). Run "
                "data/preprocess_arrayrecord.py first."
            )
        self.indexed_uri = indexed_uri.rstrip("/")
        self.num_epochs = int(num_epochs)
        self.shuffle_seed = int(shuffle_seed)
        self.provide_docs = provide_docs
        self.distill = distill

        # Load metadata (completion marker). If missing → preprocessing didn't
        # finish, refuse to run.
        import fsspec, json
        meta_path = f"{self.indexed_uri}/metadata.json"
        try:
            with fsspec.open(meta_path, "r") as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(
                f"metadata.json not found at {meta_path}. Either preprocessing "
                f"didn't complete or the indexed_uri is wrong. Run "
                f"`uv run python data/preprocess_arrayrecord.py --dataset "
                f"<name> --tokenizer <id>` to generate it."
            )
        self.N = int(self.metadata["N"])
        self.n_shards = int(self.metadata["n_shards"])
        self.config_hash = self.metadata["config_hash"]
        print(f"[QADatasetIndexed] loaded metadata: N={self.N}, "
              f"n_shards={self.n_shards}, hash={self.config_hash}", flush=True)

        # Sanity check: sequence-length / doc-shape config in cfg must match
        # what the shards were preprocessed with. Loud fail on mismatch.
        for k in ("seq_len", "num_chunks_per_doc"):
            if k in self.metadata and getattr(self, k, None) is not None:
                if int(self.metadata[k]) != int(getattr(self, k)):
                    raise RuntimeError(
                        f"cfg.dataset.{k}={getattr(self, k)!r} does not match "
                        f"metadata.json['{k}']={self.metadata[k]!r} at "
                        f"{self.indexed_uri}. Either point at a different "
                        f"indexed_uri or re-preprocess with the right config."
                    )

        # Build shard URI list (direct gs:// reads via ArrayRecordDataSource)
        self.shard_uris = [
            f"{self.indexed_uri}/samples-{i:05d}.arrayrecord"
            for i in range(self.n_shards)
        ]

        # State passed in by trainer if resuming. Either a raw Grain state
        # (bytes, restore verbatim) or an int K (synthesize state for global
        # position K on first iterator construction). A fresh launch sets neither.
        self._pending_loader_state = None
        self._pending_synth_K = None
        self.current_iterator = None

    def get_loader_state(self):
        """Return the current Grain iterator state as bytes (or None if the
        iterator hasn't been constructed yet). Called by the trainer at ckpt
        save time.
        """
        it = getattr(self, "current_iterator", None)
        if it is None:
            return None
        try:
            return it.get_state()
        except Exception:
            return None

    def set_loader_state(self, state_or_K):
        """Queue a state to apply on the next `generator()` call.

        Accepts either:
          - bytes / bytearray / dict: a real Grain iterator state (produced by
            `iterator.get_state()`); applied verbatim via `iterator.set_state`.
          - int K (K >= 0, K % worker_count == 0): synthesize the exact state
            for global position K (K samples consumed) and apply that. Uses the
            fresh iterator's own get_state() as a template so the sampler /
            data_source repr strings match byte-for-byte (Grain's _validate_state
            requires exact match).

        Two callers exist, both in trainer._restore_loader_state:
          1. Post-fix ckpt with a sibling dataloader_state.json → pass the bytes.
          2. Pre-fix ckpt (or stale/missing state file) → pass int K =
             step*batch_size; the synthesis path recovers the correct position
             without any per-run save history.
        """
        if isinstance(state_or_K, int):
            self._pending_synth_K = int(state_or_K)
            self._pending_loader_state = None
        else:
            self._pending_loader_state = state_or_K
            self._pending_synth_K = None

    def _build_pipeline(self):
        """Construct the Grain DataLoader + IndexSampler + ArrayRecordDataSource
        pipeline. Deserialize records back into the training-sample dict, then
        batch."""
        source = grain.ArrayRecordDataSource(self.shard_uris)
        sampler = grain.IndexSampler(
            num_records=self.N,
            shuffle=bool(self.shuffle),
            seed=self.shuffle_seed,
            num_epochs=self.num_epochs,
            shard_options=grain.NoSharding(),
        )

        class _Deserialize(grain.MapTransform):
            def map(self, record_bytes):
                return pickle.loads(record_bytes)

        operations = [
            _Deserialize(),
            grain.Batch(batch_size=self.batch_size, drop_remainder=True),
        ]
        return grain.DataLoader(
            data_source=source,
            sampler=sampler,
            operations=operations,
            worker_count=self.num_workers,
            worker_buffer_size=4,
        )

    def generator(self, num_epochs=None):
        """Yield batches in the (inputs_dict, masks_dict) shape the trainer expects.

        The `num_epochs` arg to this method is IGNORED — epoch count is baked
        into the sampler at construction. This mirrors the streaming QADataset's
        signature so the trainer doesn't need to branch on dataset type.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for training generator")

        pipeline = self._build_pipeline()
        iterator = iter(pipeline)

        # Resolve pending state (either raw bytes queued by trainer, or a synth
        # target position K that we materialize here using this iterator's own
        # fresh get_state() as a template — that guarantees the sampler /
        # data_source repr strings match Grain's canonical form, which
        # _validate_state requires byte-for-byte on set_state).
        pending = getattr(self, "_pending_loader_state", None)
        synth_K = getattr(self, "_pending_synth_K", None)

        if synth_K is not None:
            template = json.loads(iterator.get_state().decode())
            W = int(template["worker_count"])
            if synth_K < 0 or synth_K % W != 0:
                raise ValueError(
                    f"synth_K={synth_K} must be a non-negative multiple of "
                    f"worker_count={W}"
                )
            synthesized = dict(template)
            synthesized["last_seen_indices"] = {
                str(i): i + synth_K - W for i in range(W)
            }
            synthesized["last_worker_index"] = W - 1
            pending = json.dumps(synthesized, indent=4).encode()
            self._pending_synth_K = None
            print(
                f"[QADatasetIndexed] synthesized Grain state at global position "
                f"K={synth_K} (W={W}); no fast-forward needed.",
                flush=True,
            )

        if pending is not None:
            # Do NOT swallow set_state failures with a warn — silent index-0
            # replay is the exact regression the sanity-check + synthesis
            # design closes. If validation fails, that's a bug (config drift,
            # sampler-arg mismatch) we want raised, not hidden.
            iterator.set_state(pending)
            print(
                f"[QADatasetIndexed] restored Grain iterator state "
                f"(byte-len {len(pending)})",
                flush=True,
            )
            self._pending_loader_state = None
        self.current_iterator = iterator

        while True:
            try:
                batch = next(iterator)
            except StopIteration:
                # num_epochs exhausted. Trainer's outer while-step-loop will hit
                # its own step cap; if we get here it means the run tried to
                # consume more samples than the sampler allows. Fatal — do NOT
                # silently restart, that hides an under-provisioned num_epochs.
                raise RuntimeError(
                    f"IndexSampler exhausted after num_epochs={self.num_epochs} × "
                    f"N={self.N} = {self.num_epochs * self.N} samples. Increase "
                    f"cfg.dataset.num_epochs and re-launch (Grain will validate "
                    f"the change against any existing checkpoint — fresh run "
                    f"required if you have prior ckpts)."
                )

            # Convert the Grain-yielded batch (a dict of numpy arrays stacked
            # along axis 0) into the training-sample shape the trainer expects.
            inputs = jnp.array(batch['inputs'])
            attn_mask = jnp.array(batch['attn_mask'])
            loss_mask = jnp.array(batch['loss_mask'])
            docs = jnp.array(batch['docs'])
            docs_masks = jnp.array(batch['docs_masks'])
            pos_doc_mask = jnp.array(batch['pos_doc_mask'])
            pos_doc_ids = jnp.array(batch['pos_doc_ids'])
            ce_enable = jnp.array(batch['ce_enable']) if 'ce_enable' in batch else jnp.ones(inputs.shape[0], dtype=jnp.float32)

            B, M, d_seq_len = docs.shape
            flat_docs = docs.reshape(B * M, d_seq_len)
            flat_docs_masks = docs_masks.reshape(B * M, d_seq_len)

            if self.provide_docs:
                yield {"batch": inputs, "docs": flat_docs}, {
                    "batch_mask": attn_mask, "docs_mask": flat_docs_masks,
                    "loss_mask": loss_mask, "pos_doc_mask": pos_doc_mask,
                    "pos_doc_ids": pos_doc_ids, "ce_enable": ce_enable,
                }
            else:
                yield inputs, {
                    "batch_mask": attn_mask, "loss_mask": loss_mask,
                    "pos_doc_ids": pos_doc_ids, "ce_enable": ce_enable,
                }
