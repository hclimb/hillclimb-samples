# Random-access-by-index data pipeline (ArrayRecord + Grain IndexSampler)

**Date:** 2026-08-08 · **Author:** john · **Status:** in progress (V3 TPU
resume-invariant test pending) · **Branch:** `data_resume`

## What changed

New opt-in data path (`dataset.storage: arrayrecord`) that reads pre-tokenized
`ArrayRecord` shards via `grain.DataLoader + IndexSampler`. Sample content at
`step` becomes a pure function of `(step, config)`: resume is O(1), the
step-28k data-heterogeneity clustering is killed, and the batch-0 replay
artifact from `SKIP_LOADER_RESTORE=1` is retired. Streaming path
(`QADataset`) is untouched; opt-in per recipe.

## Motivation

Two independent failures of the streaming pipeline, both documented:

1. **Fast-forward wall on resume** — `dataloader_state.json` stores a raw-item
   `count` per Grain worker; restore re-reads/tokenizes every prior item. At
   step 34k this was ~90 min end-to-end and can host-OOM at 38k+
   (`wiki/infrastructure/experiment-launch-instructions.md §5`).
2. **Local heterogeneity clustering** — `interleave_datasets(seed=42,
   stopping_strategy="all_exhausted")` + per-source `.shuffle(buffer=100k)`
   preserves local ordering, producing the step-28k `doc_access_acc` jump
   (Δ=−0.117 at frozen weights, `wiki/experiments/2026-08-07-doc-access-jump-is-data-not-model.md`).

The `SKIP_LOADER_RESTORE=1` workaround trades (1) for a batch-0 replay
artifact that corrupts CE curves every resume
(`wiki/implementations/2026-08-06-skip-loader-restore-ce-artifacts.md`).
Neither is acceptable long-term.

## Options weighed

- **Feistel permutation over a manifest.** Simple, but still needs a
  preprocess pass to determine post-filter surviving rows, and then a Grain
  reader anyway. Loses its advantage once we're persisting tokenized samples.
- **Megatron `.bin`/`.idx`.** Battle-tested but overkill at one-row-per-sample.
  Would earn its keep if we ever pack multi-doc.
- **Grain `MapDataset.shuffle().repeat()` on ArrayRecord.** No `sampler=`
  argument — different resume semantics (index-into-permutation vs.
  index-of-sampler). Kept as fallback.
- **Chosen: `grain.DataLoader + IndexSampler + ArrayRecordDataSource`.**
  Native "index-as-function-of-step". Phase-0 spike
  (`scripts/debug/probe_grain_indexsampler.py`) confirmed all 4 questions
  green on grain 0.2.18: DataLoader path works, `iterator.get_state()` /
  `set_state()` seeks correctly, sampler repr encodes
  `num_records`/`num_epochs`/`seed` for ckpt validation.

## How it was built

**Preprocessing** — `data/preprocess_arrayrecord.py`. One-time; reads config
sources, applies `qa_filter_predicate` + `qa_transform_item` on every raw
row, writes ArrayRecord shards in per-source insertion order (no on-disk
shuffle). Emits `manifest.json` (per-sample provenance:
`shard_idx, offset, source, raw_row_id`) and `metadata.json` (config hash +
per-source counts). `metadata.json` is written **last** as the completion
marker; shards are written to `.tmp` and renamed. Idempotent per config hash;
hard-fails on hash mismatch and on any per-source post-filter proportion
drop >30%.

**Runtime** — `data/qa.py::QADatasetIndexed` (new class alongside
`QADataset`). Advertises `is_indexed = True`. Reads directly from
`gs://` via `ArrayRecordDataSource(shard_uris)` — no local copy;
`mp_prefetch` hides GCS latency at our per-worker throughput (~1.6 MB/s).
Deserialize (pickle) + `grain.Batch` operations. `StopIteration` is fatal,
not silent restart.

**Trainer** — `trainer/trainer.py::_save_loader_state` and
`_restore_loader_state` early-return under `is_indexed=True`. Sample index
is fully derivable from `step + config`; a persisted cursor would be a
second source of truth.

**Dispatch** — `data/__init__.py::get_dataset` routes to
`QADatasetIndexed` when `cfg.dataset.storage == "arrayrecord"`. Default is
`streaming` — this rework doesn't change existing recipes.

**Config** — `configs/dataset/qa_hard_neg_think_sft4b.yaml` gains
`storage`, `indexed_uri`, `num_epochs` (sampler upper bound — see the
mandatory inline warning in the yaml about why not to shrink it).

## Reference pages updated

- `wiki/data/README.md` — factory table now names both classes.
- `wiki/data/qa-dataset.md` — `Resume` section split into streaming vs.
  indexed subsections.
- `wiki/infrastructure/experiment-launch-instructions.md §5` —
  `SKIP_LOADER_RESTORE` and the fast-forward wall marked as
  streaming-only; retired on the indexed path.

## Tests

Written but not yet all executed against the final preprocess output:

- **Phase-0 API spike:** `scripts/debug/probe_grain_indexsampler.py` — 4/4
  green on grain 0.2.18.
- **V1 heterogeneity check (numpy-only):**
  `scripts/debug/verify_v1_heterogeneity.py` — asserts pool-uniform ordering
  passes analytic-binomial band while the current interleave control
  fails it (mandatory dual test). **Pending full preprocess output.**
- **V2 byte-equality:** `scripts/debug/verify_v2_byte_equality.py`.
  Executed against smoke preprocess (N=1571): **50/50 sampled rows
  byte-equal** between preprocess-time write and runtime re-transform.
- **V3 resume invariant** (on-TPU, short training): asserts
  `sha256(sorted(batch.raw_row_ids))` per step is bit-identical between
  uninterrupted and mid-run-resumed runs, steps 2001..2100. **Pending.**

## Follow-ups & risks

- V3 must pass before the first real indexed training run — it's the direct
  test of the invariant this rework promises.
- Loss curves on the first indexed run will sit **higher** than the current
  bf16 baseline because we're no longer silently re-showing early batches
  (which flattered CE). State this in the run's wandb description before
  comparing.
- Preprocessing output is per-config-hash — any change to seq_len,
  tokenizer_id, per-source normalizer settings, etc. invalidates. Design
  intent; don't work around it.
- Only `data/qa.py` is on this path. `documents.py`, `doc_copy.py`,
  `novelhopqa.py` stay on streaming until they hit their own scaling wall.
- Does NOT fix bf16 ULP freeze or the LR-schedule reset on within-stage
  resume — those are independent (see the plan file).
