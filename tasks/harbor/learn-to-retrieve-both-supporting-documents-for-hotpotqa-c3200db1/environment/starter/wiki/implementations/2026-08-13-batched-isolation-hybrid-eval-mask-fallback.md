# `mem_lookup_batched` falls back to `mem_lookup_chunked` for a per-example `[B,M]` mem_mask

**Date:** 2026-08-13 · **Author:** rohunagrawal (with Claude) · **Status:** done · **Commit/PR:**
none yet (working tree on `multihop-finetuning`).

## What changed

`mem_lookup_batched` (`models/memory.py`) no longer raises `NotImplementedError` when handed a
genuinely per-example `[B,M]` `mem_mask` (validity differs per row, not just by block ownership).
It now delegates to `mem_lookup_chunked`, which is mask-shape-agnostic. The `None` / shared-`[M]`
mask case — every existing training run — is untouched: same reshape, same einsum, byte-identical.

## Motivation & context

Asked to eval `multihop_ground4layer_s1warmstart_no_multihop_stage2only_batched_iso_topk128_bs8-2026-08-12-05-42-27`
(a `mem_batched_isolation: true` checkpoint — see its name and
[retrieval-modes.md](../architecture/retrieval-modes.md)) on the hybrid hotpotqa full-corpus task
(`gen_large_mem_msa_hotpotqa_hybrid`, `gather_bank: true`). It crashed at first generation step:

```
NotImplementedError: mem_lookup_batched assumes a shared [M] mask (reshaped to [B, m_per_query]);
per-example [B, M] mem_mask is not supported here.
```

Root cause, traced through `models/memory.py::memory_layer` dispatch and
`evals/gen_large_mem_rag_hybrid.py`'s `gather_bank` branch:

- `gather_bank` mode builds a genuinely **per-row** mask (`masks_np[r, ...]`, gen_large_mem_rag_hybrid.py:496-511):
  each query retrieves its *own* set of docs, so row `r`'s validity pattern differs from every
  other row's — a true `[B,M]` array, not a `[M]` array broadcast over the batch.
- `mem_lookup_chunked` was *already* built to accept exactly this (its docstring / the other
  lookup variants' error messages all point to it: "per-example `[B,M]` mem_mask is only
  supported by `mem_lookup_chunked`"), and the hybrid eval task sets `eval.lookup_chunk_size:
  2048` (`configs/eval/generation_large_mem_rag_hybrid.yaml`) precisely so `memory_layer` routes
  there.
- But `memory_layer`'s dispatch checks `mem_batched_isolation` **before** the
  `mem_lookup_chunk_size` branch (models/memory.py:638 vs 651) — so a checkpoint trained with
  `mem_batched_isolation: true` never reaches `mem_lookup_chunked` at all; it hits
  `mem_lookup_batched`'s hard guard first. Every checkpoint the hybrid gather-bank eval had been
  run against before this one predates `mem_batched_isolation` (introduced
  [2026-08-02](2026-08-02-hard-neg-full-efficient-retrieval.md) for training throughput only), so
  this combination was simply never exercised until now.

## Options weighed

1. **Force `model.memory.mem_batched_isolation=false` as an eval-time override.** Rejected: with
   `per_query_isolation: true` still on (it is, for this checkpoint), dispatch falls through to
   plain `mem_lookup`, which has the *identical* 1D-only guard (models/memory.py:31) — so this
   just trades one `NotImplementedError` for another, no further along.
2. **Reorder `memory_layer`'s dispatch** so `mem_lookup_chunk_size` is checked before
   `mem_batched_isolation`. Rejected: changes behavior for every `mem_batched_isolation: true`
   training run that also happens to set a chunk size for some other reason, and moving a
   priority branch is a bigger blast radius than it needs to be for what is really a narrow,
   local gap in one function.
3. **Fall back inside `mem_lookup_batched` itself, gated on `mem_mask.ndim`** (chosen). Purely
   additive: the existing guard already special-cased "not 1D" as an error; swapping the error
   for a delegate call changes behavior *only* in a case that previously always crashed. No
   training run is affected (training never passes a 2D mask into this path).

## How it was built & integrated

- `models/memory.py::mem_lookup_batched`: the `mem_mask.ndim != 1` guard now returns
  `mem_lookup_chunked(q, w, cfg, collect_aux)` instead of raising. `mem_lookup_chunked` reads the
  same flat `w['mem_k']`/`w['mem_v']`/`w['mem_mask']` convention `mem_lookup_batched` uses (no
  reshaping needed for the fallback), and is itself agnostic to `per_query_isolation` /
  `isolation_group_size` — the mask alone already encodes each row's restriction, matching
  `gather_bank`'s block-per-row construction.
- The `isolation_group_size != 1` guard (a separate, unrelated restriction) still raises as
  before — the mask-shape check now short-circuits ahead of it, which is correct: a genuine 2D
  mask never needs the group-size restriction (chunked lookup doesn't care about groups at all).
- `tests/test_mem_lookup_batched.py`: added `run_2d_mask_fallback_case` — builds a real
  per-row-distinct `[B,M]` mask (block-diagonal layout matching `gather_bank`'s, but with a
  different few slots zeroed per row so rows aren't identical), asserts `mem_lookup_batched` no
  longer raises and matches `mem_lookup_chunked` called directly on the same inputs (indices,
  scores, values). 3 seeds, wired into `main()` alongside the existing equivalence/mismatch cases.

## Reference pages updated

- [architecture/retrieval-modes.md](../architecture/retrieval-modes.md): `mem_lookup_batched`
  table row and its `mem_batched_isolation` section now describe the 2D-mask fallback instead of
  claiming incompatibility.

## Tests

`JAX_PLATFORMS=cpu uv run python tests/test_mem_lookup_batched.py` (CPU-only, run via the launcher
on `rohun-v6e-8-0` per the no-local-accelerator rule — see
[experiment-launch-instructions.md](../infrastructure/experiment-launch-instructions.md)):

```
[PASS] seed=0 activation=softmax: idx_match=True scores_close=True values_close=True read_close=True
[PASS] seed=0 activation=relu: idx_match=True scores_close=True values_close=True read_close=True
[PASS] seed=0 activation=sigmoid: idx_match=True scores_close=True values_close=True read_close=True
[PASS] seed=1 activation=softmax: idx_match=True scores_close=True values_close=True read_close=True
[PASS] seed=1 activation=relu: idx_match=True scores_close=True values_close=True read_close=True
[PASS] seed=1 activation=sigmoid: idx_match=True scores_close=True values_close=True read_close=True
[PASS] seed=2 activation=softmax: idx_match=True scores_close=True values_close=True read_close=True
[PASS] seed=2 activation=relu: idx_match=True scores_close=True values_close=True read_close=True
[PASS] seed=2 activation=sigmoid: idx_match=True scores_close=True values_close=True read_close=True
[PASS] seed=3 activation=softmax: idx_match=True scores_close=True values_close=True read_close=True
[PASS] seed=3 activation=relu: idx_match=True scores_close=True values_close=True read_close=True
[PASS] seed=3 activation=sigmoid: idx_match=True scores_close=True values_close=True read_close=True
[PASS] seed=4 activation=softmax: idx_match=True scores_close=True values_close=True read_close=True
[PASS] seed=4 activation=relu: idx_match=True scores_close=True values_close=True read_close=True
[PASS] seed=4 activation=sigmoid: idx_match=True scores_close=True values_close=True read_close=True
[PASS] isolation_group_size=4 raises NotImplementedError: True
[PASS] seed=0 2D mem_mask falls back to mem_lookup_chunked (matches exactly): True
[PASS] seed=1 2D mem_mask falls back to mem_lookup_chunked (matches exactly): True
[PASS] seed=2 2D mem_mask falls back to mem_lookup_chunked (matches exactly): True

ALL TESTS PASSED
```

Real-hardware confirmation: re-ran the hybrid hotpotqa eval that surfaced the bug
(`multihop_ground4layer_s1warmstart_no_multihop_stage2only_batched_iso_topk128_bs8-2026-08-12-05-42-27`
step 20200, `rohun-v6e-8-0`) with the fix — see
[wiki/experiments/2026-08-13-hotpotqa-hybrid-batched-iso-checkpoint.md](../experiments/2026-08-13-hotpotqa-hybrid-batched-iso-checkpoint.md)
for the actual eval result.

## Follow-ups & risks

- The `mem_lookup_chunked` fallback is the general/slower path (streams via `sharded_top_k_ip`
  rather than the free block-diagonal reshape); fine for eval (one-off), but if a future training
  recipe ever needs a genuine per-example mask *under* `mem_batched_isolation` at training scale,
  this path's throughput hasn't been benchmarked against the batched fast path — check before
  relying on it in a training loop.
- None of `mem_lookup`, `mem_lookup_gqa`, `product_key_lookup` got the same fallback — they still
  hard-raise on a 2D mask. Only `mem_lookup_batched` needed it here (that's the dispatch this
  checkpoint's saved config actually hits); worth doing the same if another variant hits the same
  wall.
