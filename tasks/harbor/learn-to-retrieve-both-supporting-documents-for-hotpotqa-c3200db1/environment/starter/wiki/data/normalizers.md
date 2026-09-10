# Normalizers

Each source has its own HF schema; a **normalizer** maps a raw row to the common shape the
pipeline expects. In `data/utils.py`, chosen per-source by `normalizer_type`.

## Output shape
```python
{"question", "answer", "pos_doc": [...], "neg_doc": [...],
 "_min_neg_docs", "_ce_enable", "pos_doc_ids"?}
```
`pos_doc`/`neg_doc` are lists of strings; `_ce_enable` / `_min_neg_docs` are read downstream by
the filter and the CE gate (see [batch-format.md](batch-format.md)).

## The three normalizers
- **`make_normalizer(field_map, think_field, doc_separator, neg_score_threshold, min_neg_docs,
  mask_ce)`** (default): renames columns via `field_map`; splits `pos_doc`/`neg_doc` on
  `doc_separator` (or passes lists); wraps the answer in `<think>…</think>` when `think_field`
  is set; drops negatives above `neg_score_threshold` (hard-neg sources); `mask_ce=true` sets
  `_ce_enable=0` (retrieval-only text-similarity rows).
- **`make_flashrag_normalizer(context_key)`** — FlashRAG datasets: `golden_answers[0]` → answer;
  `context_key="passages"` → `is_selected==1` passages become `pos_doc`, rest `neg_doc`;
  `context_key="context"` → title+sentences paragraphs.
- **`make_hotpotqa_normalizer()`** — HotpotQA distractor: paragraphs whose title is in
  `supporting_facts` → `pos_doc`; the rest → `neg_doc` (distractors).

All three forward `pos_doc_ids` when present (added by the `*_with_ids` prep scripts).

## Source config keys (`configs/dataset/sources/*.yaml`)
`hf_name`, `hf_config`, `field_map`, `think_field`, `doc_separator`, `normalizer_type`
(`flashrag`/`hotpotqa`), `flashrag_context_key`, `neg_score_threshold`, `min_neg_docs`,
`mask_ce`, `weight` (interleave probability), `prompt_path` / `teacher_prompt_path` /
`student_prompt_path`. To add a source with a novel schema, prefer a `field_map`; only write a
new normalizer for nested/derived structures.
