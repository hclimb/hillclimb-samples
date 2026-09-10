# QA storage modes: streaming vs. indexed

`QADataset` supports two backends, selected by `configs/dataset/*.yaml`
`storage:`. Default is streaming — the indexed path is opt-in per recipe.

| Key | Class | When to use |
|-----|-------|-------------|
| `streaming` (default) | `QADataset` | Live-HF or offline-parquet; no preprocess step needed |
| `arrayrecord` | `QADatasetIndexed` | Long runs that resume: O(1) resume, kills local-heterogeneity clustering |

Both classes share the filter + transform (`data/qa.py::qa_filter_predicate`,
`qa_transform_item`) so tokens are identical across modes on the same rows.

## Indexed path — one-time preprocessing

Run once per config hash. `metadata.json` is written last as the completion
marker; re-runs with a matching hash are no-ops. Any change to `seq_len`,
`num_chunks_per_doc`, `min_doc_length`, `tokenizer_id`, or per-source
normalizer settings invalidates the hash — you get a fresh output dir.

```
HF_HUB_OFFLINE=1 GROUND_HF_PARQUET=$HOME/hf_parquet \
    uv run python data/preprocess_arrayrecord.py \
        --dataset qa_hard_neg_think_sft4b \
        --tokenizer Qwen/Qwen3-4B \
        --out gs://memory-layers-training/indexed/
```

Output layout at `gs://<out>/<config-hash>/`:

- `samples-{i:05d}.arrayrecord` — pickled sample dicts, raw insertion order.
- `manifest.json` — per-sample `(shard_idx, offset, source, raw_row_id)`.
- `metadata.json` — config hash, `N`, per-source counts. **Written LAST.**

Add `--max-samples-per-source 500` for a smoke test (output goes to a
`SMOKE-<hash>/` path that can't be reused for training).

## Indexed path — enabling in a config

```yaml
storage: arrayrecord
indexed_uri: gs://memory-layers-training/indexed/<config-hash>
num_epochs: 10   # SAMPLER upper bound, NOT training-length target
shuffle_seed: 42
```

`num_epochs` is pinned into the Grain `IndexSampler` ckpt-validation
signature — bumping it mid-run breaks resume. Set generously.

## Verification

- `scripts/debug/verify_v1_heterogeneity.py` — pool-uniform ordering vs.
  current interleave control (numpy-only, seconds).
- `scripts/debug/verify_v2_byte_equality.py` — sampled manifest rows
  byte-equal to a runtime re-transform (CPU, minutes).
