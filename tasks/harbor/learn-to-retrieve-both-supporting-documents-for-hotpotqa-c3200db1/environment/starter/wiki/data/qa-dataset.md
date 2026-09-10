# `QADataset` (the primary dataset)

`data/qa.py` — Q/A + supporting documents → memory batches. Streams HF sources through a
`grain` pipeline. Emits the batch contract in [batch-format.md](batch-format.md).

## Source modes
- **Interleave** (`sources:` — takes precedence): each source loaded + normalized, then
  `interleave_datasets` with per-source `weight` (else uniform), `stopping_strategy=
  "all_exhausted"`, `shuffle(buffer=100000, seed=shuffle_seed)`.
- **Single** (`hf_name:`): one dataset + one normalizer.
See [normalizers.md](normalizers.md) for how raw rows map to `question/answer/pos_doc/neg_doc`,
and [dataset-configs.md](dataset-configs.md) for the config shape.

## Pipeline (`_build_pipeline`)
`_StreamingSource → filter(_StreamingQAFilter) → map(transform) → mp_prefetch(num_workers) →
batch(batch_size, drop_remainder=True)`.
- **Filter** drops rows where the positive doc is `< min_doc_length` or `> doc_length`
  (`= doc_chunk_seq_len · num_chunks_per_doc`), fails `min_neg_docs`, leaks the answer inside
  the question, or whose formatted Q+A exceeds `seq_len`.
- **Transform** (`_StreamingQATransform`, or `_DistillQATransform` when `distill`):
  - `build_prefix_text` — chat template (`apply_chat_template`, `enable_thinking`) / prompt
    template / bare `Question:/Answer:`.
  - `text = prefix + answer + eos`, tokenized to `seq_len` (pad to max). `loss_mask` = attn
    mask with the **prefix zeroed** when `mask_prefix` (loss only on the answer).
  - `pack_docs` (`data/utils.py`) chunks pos-then-neg docs into `num_chunks_per_doc` chunks of
    `doc_chunk_seq_len` (`chunk_text`); `pos_doc_mask` marks positives.
  - Carries `pos_doc_ids` (corpus IDs, −1-padded) and `ce_enable` (0 = `mask_ce` retrieval-only
    rows).

## Offline & rate limits
`_load_dataset_with_backoff`: under `HF_HUB_OFFLINE=1` reads pre-cached local parquet from
`$GROUND_HF_PARQUET/<repo__>` (no Hub call); otherwise `load_dataset(..., streaming=True)`
wrapped in `_hf_retry` exponential backoff on HF 429s (pre-cache with `scripts/misc/precache_hf.sh`).

## Resume

**Streaming path (default).** `get_loader_state`/`set_loader_state` persist a raw-item
`count`; `_StreamingIterator` **fast-forwards** the deterministic stream to that position on
next `generator()`. Exact only with the **same dataset config + `shuffle_seed`**. The
generator also auto-rebuilds the pipeline on worker crashes (extra 429 sleep). Fast-forward
is expensive (~hours at 34k+, host-OOM risk beyond 38k) — see
[../infrastructure/experiment-launch-instructions.md](../infrastructure/experiment-launch-instructions.md) §5.

**Indexed path (`storage: arrayrecord`).** `QADatasetIndexed` reads pre-tokenized
`ArrayRecord` shards via `grain.DataLoader + IndexSampler`, seeded by `shuffle_seed`. Sample
at step *N* is a pure function of `(step, config)`, so resume is O(1) — no persisted cursor,
no fast-forward. Trainer skips `_save_loader_state` / `_restore_loader_state` under
`is_indexed=True`. See [storage-modes.md](storage-modes.md) for how to preprocess + point a
config at shards.
