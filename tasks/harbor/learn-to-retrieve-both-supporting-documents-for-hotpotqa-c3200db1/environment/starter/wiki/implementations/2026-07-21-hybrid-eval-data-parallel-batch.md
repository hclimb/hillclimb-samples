# Data-parallel hybrid eval: per-example [B, M] mem_mask (B=8 → ~7× query throughput)

**Date:** 2026-07-21 · **Author:** rohunagrawal (with Claude) · **Status:** done ·
**Branch:** `msa-hybrid-sweep`

**What changed:** the RAG→memory hybrid evaluator
([2026-07-20 note](2026-07-20-rag-hybrid-evaluator.md)) no longer requires
`dataset.batch_size: 1`. At `batch_size: B > 1` it packs **B distinct queries per forward**
with a **per-example `[B, M]` mem_mask**, one query per mesh data row. Decode is
bandwidth-bound (the weights + bank scan are read per step regardless of rows), so a B-row
step costs ≈ a B=1 step: measured **8 queries / 70 s including compile** (~3.7 s/query
marginal) vs ~26 s/query tiled — ~7× throughput at B=8. The B=1 path is preserved
bit-for-bit (tiling + `row_divergence_rate`), so all MuSiQue hybrid numbers stay comparable.

## Motivation

The evaluator's original design spent the whole mesh on redundancy: B=1, the single query
tiled to all 8 data rows purely as a determinism check. At c10000 × `max_new_tokens` 512
that made the n=128 MS-MARCO arm ~55 min of decode for work the hardware could do in ~7.
Requested for the MS-MARCO c10000 campaign ("add data-parallelism as much as possible").

## Options weighed

- **Per-example `(B, M)` mask (chosen)** vs the alternatives re-weighed from the 07-20 note:
  per-query bank **gather** still remaps flat indices (breaks doc-id↔slot telemetry);
  **TP=4** needs weight + bank resharding and a top-k merge for ~2.5–3× at best;
  **query-sharding across boxes** is ~4× with per-box fixed costs and result merging. The
  `(B, M)` mask wins on ceiling (~8×) and on locality: the only kernel change is the mask
  broadcast rank.
- **Blast-radius containment.** The masking contract lives in kernels shared with training
  (`models/retrieval_ops.py`). The change is shape-polymorphic — `mask.ndim == 1` keeps the
  exact old broadcast, so every existing caller (training included) is untouched; only a
  2-D mask takes the new `mask[:, None, None, :]` branch. Paths that *can't* honor a 2-D
  mask raise `NotImplementedError` instead of mis-broadcasting: the `shard_map` path
  (`_sharded_top_k` reshards the mask over the bank axis) and the non-chunked lookup
  variants (`mem_lookup`, `mem_lookup_gqa`, `product_key_lookup`, `span_readout`). The
  hybrid eval always runs `mem_lookup_chunked` → `_replicated_top_k` (model axis 1), the
  one supported path.

## How it was built & integrated

- `models/retrieval_ops.py::_matmul_top_k` / `_scan_chunks` — mask broadcast is now
  rank-aware; `_replicated_top_k` pads a 2-D mask along its last axis and reshapes to
  `[n_chunks, B, chunk]` so the chunk scan slices along chunks unchanged.
- `evals/gen_large_mem_rag_hybrid.py` — `tiled_mode = (B == 1)` branches: legacy path
  identical (shared `[M]` mask, tile to data axis, divergence check); B>1 builds per-row
  prompts/masks, ships the `(B, M)` mask `P('data', None)`, records per-row samples.
  B must be a multiple of the mesh data-axis size (hard error otherwise).
  `row_divergence_rate` is reported only when tiled batches ran; `generated_count` =
  recorded samples; a tail batch overshooting `num_samples` is trimmed after the loop.
  The query-matching pre-pass now consumes whole batches so both generator passes stay in
  lockstep at any B.
- `configs/eval/tasks/gen_large_mem_msmarco_hybrid.yaml` → `batch_size: 8` (default);
  `…musique_hybrid.yaml` stays `batch_size: 1` (comment corrected — no behavior change).
  `scripts/embed/eval_msmarco_hybrid.sh` gained `EXTRA_OVERRIDES` (space-separated Hydra
  overrides, e.g. `evals.msmarco_hybrid.dataset.batch_size=1` for tiled-mode comparisons).
- `_generate_tokens` needed no changes — it already treats rows independently.

## Reference pages updated

[evaluator-types](../evaluation/evaluator-types.md) (hybrid evaluator section: batching
modes). Kernel behavior is internal; retrieval-modes policy unaffected.

## Tests

`uv run python tests/test_rag_hybrid_mask.py` on `tpu-v6e-slice-mig-1wjb`
(`JAX_PLATFORMS=cpu`, via `scripts/embed/test_rag_hybrid_mask.sh`):

```
PASS rank_unique_doc_ids
PASS build_doc_slot_mask
PASS gold_coverage
PASS masked_slots_never_retrieved
PASS matmul_top_k_per_example_mask
PASS per_example_mask_batched_equals_single
PASS non_chunked_lookups_reject_2d_mask
ALL PASS
```

New cases: `matmul_top_k_per_example_mask` (disjoint per-row masks → per-row containment),
`per_example_mask_batched_equals_single` (**the load-bearing claim**: through the production
`_replicated_top_k`/`_scan_chunks` path — chunk padding included — a `[B, M]` batched lookup
returns per row exactly the indices and scores of that row run alone with its `[M]` mask),
`non_chunked_lookups_reject_2d_mask` (guards fire).

End-to-end smoke (n=8, c10000, k=50, B=8, v6e slice, both workers, `__RUN_EXIT__=0`):
8/8 gold coverage, all generations well-formed, judge 6/8, `mean_active_bank_slots` 4322,
`row_divergence_rate` correctly absent, aux-OOM self-disable unchanged. Generation
8 queries / 70 s incl. compile. On the 2 queries shared with the earlier B=1 smoke, the
generated answers match the B=1 outputs (verbatim prefixes) despite a marginally different
distractor tail. A full B=1-vs-B=8 same-corpus equivalence run (`EXTRA_OVERRIDES` tiled
mode) was started and then **skipped at user request** in favor of launching the real arm;
the batched-equals-solo property is covered at the kernel level by the unit test.

## Follow-ups & risks

- The end-to-end tiled-vs-batched text-equality check remains unrun (kernel-level
  equivalence + spot-check stand in for it). One runner command reproduces it:
  `NUM_SAMPLES=8 NAME_SUFFIX=_smokeb1 EXTRA_OVERRIDES=evals.msmarco_hybrid.dataset.batch_size=1 …`.
- In batched mode there is no within-run determinism monitor (`row_divergence_rate` needs
  redundant rows). Approx top-k remains policy-on.
- The sharded (`shard_map`) path and non-chunked lookups deliberately reject `[B, M]`
  masks; supporting them needs a per-axis reshard design, not just a broadcast.
