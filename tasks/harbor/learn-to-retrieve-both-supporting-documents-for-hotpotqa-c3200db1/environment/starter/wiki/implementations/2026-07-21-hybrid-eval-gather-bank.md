# Hybrid eval `gather_bank`: per-query small banks (host-side corpus, ~free scan, telemetry back)

**Date:** 2026-07-21 · **Author:** rohunagrawal (with Claude) · **Status:** done ·
**Branch:** `msa-hybrid-sweep`

**What changed:** opt-in `eval.gather_bank: true` for the RAG→memory hybrid evaluator.
Instead of replicating the full bank to HBM and masking it per query
([mask design](2026-07-20-rag-hybrid-evaluator.md), [batched masks](2026-07-21-hybrid-eval-data-parallel-batch.md)),
the full bank stays in **host numpy** and each batch ships only its rows' top-k docs' slots —
one fixed-size block per row (`docs_per_row × eff_doc_len`), per-example mask confining each
row to its own block. Fixed shapes across batches → one compile. Three wins, all confirmed
on-box: **(1)** the per-token scan covers `B×k` docs instead of the corpus (~free at k≤50);
**(2)** the 4M-slot replicated-bank HBM guard no longer applies — corpus size becomes
host-RAM bound; **(3)** the aux telemetry forward fits again, so `doc_hit_rate` /
`mem_pos_weight_mass` return at c10000 (first-ever dilution reading at that scale: mass
0.188 on the k=50 smoke).

## Options weighed

- **Union bank vs per-row blocks.** Chosen: per-row blocks with duplicates allowed. A
  deduped union has a batch-dependent size → recompile every batch; padding it to worst-case
  equals the block layout anyway. Duplicated docs across rows waste ≤B× of a tiny bank and
  keep the row→doc mapping trivial (`block_pos // eff_doc_len` indexes the row's ranked list).
- **Why not per-example `mem_k [B, M, D]`?** Would touch model code (`'bntd,bmd->bntm'`).
  The block layout reuses the shared-bank `[M, D]` contract plus the already-shipped
  per-example `[B, M]` mask — evaluator-only, as requested.
- **Equivalence, precisely stated.** Masking applies pre-top-k, so gather selects the same
  candidate slots (verified: `mean_active_bank_slots` bitwise-equal across modes). Generated
  *text* is NOT bitwise stable: bf16 logits tie often (8-bit mantissa) and `top_k`
  tie-breaking follows candidate arrival order, which differs between a 1,250-chunk and
  50-chunk scan — one flipped boundary slot in 512 greedy steps cascades textually. Measured
  on the n=8 A/B: 2/8 identical texts, 6 drift at char 213–977, **8/8 identical per-query
  judge verdicts**, identical 0.75 accuracy. Same divergence class the approx-topk policy
  already accepts.

## How it was built & integrated

- `evals/gen_large_mem_rag_hybrid.py::build_gather_plan` (pure helper) — one row's global
  flat indices, ranked order, unknown ids skipped *without* wasting block slots (the unit
  test caught truncate-before-skip), short rows padded with slot 0 for the mask to kill.
- Evaluator: in gather mode the full-bank replication and `_MAX_BANK_SLOTS` guard are
  skipped; per batch, `mem_k/mem_v` are gathered from host numpy and shipped bf16-replicated
  (~420 MB/batch at k=50, ~42 MB at k=5 — noise); `pos_sets` are translated to local bank
  positions for the telemetry helpers (an uncovered gold with no reachable local position
  reports hit/mass 0.0, a true retrieval miss, not None). Requires `max_chunks_per_doc: 1`
  (hard error otherwise). Both batching modes (tiled B=1 / data-parallel B>1) work over it.
- Config: `gather_bank: false` default in `configs/eval/generation_large_mem_rag_hybrid.yaml`
  (legacy behavior, how every pre-2026-07-21 number was measured);
  `gen_large_mem_msmarco_hybrid.yaml` sets `true`.

## Reference pages updated

[evaluator-types](../evaluation/evaluator-types.md) (hybrid section: gather_bank paragraph).

## Tests

`scripts/embed/test_rag_hybrid_mask.sh` on `tpu-v6e-slice-mig-1wjb` (JAX_PLATFORMS=cpu):

```
PASS rank_unique_doc_ids / build_doc_slot_mask / gold_coverage / masked_slots_never_retrieved
PASS matmul_top_k_per_example_mask / per_example_mask_batched_equals_single
PASS build_gather_plan
PASS non_chunked_lookups_reject_2d_mask
ALL PASS
```

End-to-end A/B (n=8, c10000, k=50, B=8, `_smokegather` vs `_smokeb8`): metrics table above —
judge 0.75 = 0.75 with 8/8 per-query agreement, `mean_active_bank_slots` 4322.375 exactly
equal, `aux_telemetry_oom` False (vs True masked), `doc_hit_rate` 1.0,
`mem_pos_weight_mass` 0.188. Batch wall clock 54 s vs 70 s including compile.

## Follow-ups & risks

- Duplicate gold docs in other rows' blocks are masked off per row; the local-position
  translation deliberately searches all blocks (harmless — validity gates hits).
- Multi-chunk docs (`max_chunks_per_doc > 1`) unsupported by design; needs per-doc variable
  block sizes.
- The serving-cost story (host→HBM gather per query) is still not a throughput measurement;
  it removes the scan term but adds an H2D term a real system would pipeline.
