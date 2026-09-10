"""Document-copy (reconstruction) dataset — a SELF-CONTAINED, REMOVABLE grounding experiment.

Objective: given a question in the text context, REPRODUCE the relevant document, whose chunks
sit in the memory bank exactly like normal QA training. The answer/CoT is ignored entirely; the
target is the positive document. This forces the memory read to be the source for the document's
high-information tokens (names/numbers), which a QA answer never demanded (those were reachable
parametrically). See the grounding discussion / plan.

INDEPENDENCE / REMOVAL: this file only *reuses* QADataset's streaming source-loading + resume and
*overrides the item transform*. It adds nothing to the trainer or loss (it emits the exact same
batch keys as `_StreamingQATransform`, so the existing CE-on-loss_mask path handles it). To remove
this experiment completely:
  1. delete this file,
  2. delete the `doc_copy` branch in data/__init__.py,
  3. delete configs/dataset/doc_copy_hard_neg.yaml, configs/model/qwen3_mem_embed_copy_topk4.yaml,
     scripts/embed/train_ground_copy.sh, data/prompts/doc_copy.txt.
Nothing in the QA / control / s1 / s2 path imports from here, so removal is inert.

CAVEAT (partial context leak): because the target document is teacher-forced in the sequence, a
predictable connective token can be induction-copied from the already-visible document prefix
rather than read from memory. The *informative* tokens (which have low prefix-predictability) still
have to come from memory — and those are the grounding-critical ones. Watch mem_pos_weight_mass /
mem_top1_weight and the gold-vs-hard-neg ablation to confirm the read is actually carrying content.
"""
import numpy as np
import grain.python as grain

from .qa import QADataset, _StreamingSource, _StreamingQAFilter
from .utils import build_prefix_text, make_attn_mask, pack_docs


class _StreamingDocCopyTransform(grain.MapTransform):
    """Like `_StreamingQATransform`, but the TARGET is the positive document (reproduce it), not the
    answer. Emits identical batch keys so the trainer/loss are unchanged."""

    def __init__(self, tokenizer, seq_len, doc_chunk_seq_len, num_chunks_per_doc, mask_prefix,
                 chat_template, copy_prompt):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.doc_chunk_seq_len = doc_chunk_seq_len
        self.num_chunks_per_doc = num_chunks_per_doc
        self.mask_prefix = mask_prefix
        self.chat_template = chat_template
        self.copy_prompt = copy_prompt

    def map(self, item):
        question = item.get("question", "")
        pos_docs_raw = item.get("pos_doc", [])
        neg_docs_raw = item.get("neg_doc", [])
        if isinstance(pos_docs_raw, str):
            pos_docs_raw = [pos_docs_raw]
        if isinstance(neg_docs_raw, str):
            neg_docs_raw = [neg_docs_raw]

        # TARGET = first positive document (the one whose chunks are the positives in the bank).
        # The QA answer/thinking is deliberately ignored.
        target_doc = pos_docs_raw[0] if pos_docs_raw else ""

        # Prefix is the question only (framed by the copy prompt). NO CoT (has_thinking=False).
        prefix_text = build_prefix_text(
            self.tokenizer, question, self.copy_prompt, self.chat_template, has_thinking=False
        )
        text = prefix_text + target_doc + (self.tokenizer.eos_token or "<|im_end|>")

        tokens = self.tokenizer(
            text, truncation=True, max_length=self.seq_len,
            padding="max_length", return_tensors="np"
        )["input_ids"][0]

        attention_mask = make_attn_mask(tokens, self.tokenizer.pad_token_id)
        loss_mask = attention_mask.copy()
        if self.mask_prefix:
            prefix_tokens = self.tokenizer(
                prefix_text, truncation=True, max_length=self.seq_len, return_tensors="np"
            )["input_ids"][0]
            loss_mask[:min(len(prefix_tokens), len(tokens))] = 0.0

        # Memory bank = positive + hard-negative docs, exactly like QA training.
        final_doc_chunks, final_doc_masks, final_pos_mask = pack_docs(
            self.tokenizer, pos_docs_raw, neg_docs_raw, self.num_chunks_per_doc, self.doc_chunk_seq_len
        )

        pos_doc_ids = np.full(self.num_chunks_per_doc, -1, dtype=np.int64)
        raw_ids = item.get("pos_doc_ids") or []
        for j, doc_id in enumerate(raw_ids[:self.num_chunks_per_doc]):
            pos_doc_ids[j] = int(doc_id)

        # Nothing to copy (no positive doc) -> gate CE off for the row (retrieval-only).
        ce_enable = np.float32(1.0 if target_doc.strip() else 0.0)

        return {
            "inputs": tokens, "attn_mask": attention_mask, "loss_mask": loss_mask,
            "docs": final_doc_chunks, "docs_masks": final_doc_masks,
            "pos_doc_mask": final_pos_mask, "pos_doc_ids": pos_doc_ids,
            "ce_enable": ce_enable,
        }


class DocCopyDataset(QADataset):
    """Reuses QADataset source-loading / interleave / streaming-resume; swaps only the item
    transform to the document-copy objective. Distillation and force_thinking are forced off."""

    def __init__(self, tokenizer=None, copy_prompt_path="data/prompts/doc_copy.txt", **kwargs):
        kwargs["distill"] = False
        kwargs["force_thinking"] = False
        super().__init__(tokenizer=tokenizer, **kwargs)
        with open(copy_prompt_path) as f:
            self.copy_prompt = f.read()

    def _build_pipeline(self):
        pipeline = _StreamingSource(self.dataset)
        pipeline = pipeline.filter(_StreamingQAFilter(
            self.tokenizer, self.seq_len, self.chat_template, self.doc_length, self.force_thinking,
            min_doc_length=self.min_doc_length, filter_doc_length=self.provide_docs,
        ))
        pipeline = pipeline.map(_StreamingDocCopyTransform(
            tokenizer=self.tokenizer, seq_len=self.seq_len,
            doc_chunk_seq_len=self.doc_chunk_seq_len, num_chunks_per_doc=self.num_chunks_per_doc,
            mask_prefix=self.mask_prefix, chat_template=self.chat_template, copy_prompt=self.copy_prompt,
        ))
        if self.num_workers > 0:
            pipeline = pipeline.mp_prefetch(grain.MultiprocessingOptions(
                num_workers=self.num_workers, per_worker_buffer_size=4,
            ))
        pipeline = pipeline.batch(self.batch_size, drop_remainder=True)
        return pipeline
