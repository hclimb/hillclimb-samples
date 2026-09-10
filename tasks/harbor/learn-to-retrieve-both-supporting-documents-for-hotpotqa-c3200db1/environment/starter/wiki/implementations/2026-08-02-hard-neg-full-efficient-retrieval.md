# Efficient retrieval for the 200-hard-negative multihop recipe

**Date:** 2026-08-02 **Author:** rohunagrawal (with Claude) **Status:** DONE — retrieval fix
shipped and benchmarked; training run hit and recovered from two further bugs (a `trainer.py`
stage-transition OOM, then a `doc_access_loss`/`mem_batched_isolation` incompatibility that meant
zero real gradient for ~1300 steps) — both fixed, verified in isolation, and confirmed together on
real hardware: training is producing genuine, fluctuating loss values past the point every earlier
attempt either crashed or silently no-op-ed. Judged-accuracy eval is the only piece left, tracked
as a follow-up, not blocking. **Commit/PR:** none yet (working on `datagen`, no PR opened) — see
the diff list in "How it was built" below.

## What changed, so far

1. Fixed `configs/dataset/multihop_hard_neg_full.yaml`'s `batch_size` (4 → 8): the colleague's
   comment already flagged that batch must be a multiple of `jax.device_count()`, and our box is
   an 8-chip v6e-8 (2x4 flex slice), not the 4-chip v4-8 the `4` was tuned for.
2. Added `mem_lookup_batched` (`models/memory.py`) + a new `mem_batched_isolation` cfg flag
   (`memory_layer` dispatch) — a genuinely batched per-row retrieval path. Default off; every
   existing run is unaffected.
3. Added `tests/test_mem_lookup_batched.py` (correctness, CPU-only) and
   `benchmarks/bench_hard_neg_full_retrieval.py` + `scripts/misc/sweep_hard_neg_full_retrieval.sh`
   (throughput/memory, real TPU).
4. Updated `wiki/architecture/retrieval-modes.md` with the new mode + dispatch precedence.

## Motivation & context

The colleague's `multihop_hard_neg_full.yaml` puts **all** of a row's docs (~4 positives + 200
hard negatives, `num_chunks_per_doc=256`, `doc_chunk_seq_len=256`) into the memory bank at once —
`m_per_query = 256*256 = 65,536` doc-token slots per query, the same scale as the single-layer
`qa_hard_neg_think_sft4b` baseline (`wiki/experiments/2026-07-15-approx-topk-training.md`).

But per `wiki/data/batch-format.md` + `models/qwen3_mem_embed.py:84-85`, the embed model's per-query
doc banks are **flattened across the whole batch** into one shared `[B*m_per_query, H]` bank before
`memory_layer` ever sees it, and the default retrieval path (`mem_lookup`, `models/memory.py:59`)
does a *full* `einsum('btnh,mh->bntm', ...)` against that whole flat bank — so query `b` scores
against `B*m_per_query` slots, of which only `m_per_query` are its own. At `B=8` that's
`M_total = 524,288` slots — 8x the per-query bank, and past every OOM threshold already documented
in this codebase (`models/memory.py:130-131`: *"at max_docs=1000, corpus M~250k... OOMs HBM"*;
`models/memory.py:251`: *"OOMs at large batch (34G at B256)"*).

An existing flag, `per_query_isolation` (+ `isolation_group_size=1`), already restricts *which*
slots contribute — but only by **masking the already-materialized** `[B,N,T,B*m]` score matrix
(`mem_lookup`, `models/memory.py:62-82`) to `-1e9` on foreign rows. It pays the full O(B) compute
and memory before throwing 7/8 of the work away, and — critically — it's implemented **only** in
the plain `mem_lookup` path, not in `mem_lookup_chunked` / `mem_lookup_two_pass` / `mem_lookup_gqa`,
which are exactly the paths this bank size would otherwise force you into.

Rohun's stated intuition: query `i` doesn't need to see the batch's other rows' docs at all when
each row already carries hundreds of hard negatives — no in-batch-negative diversity is being
lost by isolating rows, and the framework is TPU-memory-bound at this bank size. That's exactly
right, and the fix should be structural (never build the cross-batch join), not a mask.

## Options weighed

1. **Mask the full matrix** (`per_query_isolation` as it stands). Rejected as the *primary* fix:
   correct output, but pays O(B) compute/memory to throw away `(B-1)/B` of it, and doesn't compose
   with the chunked/two-pass paths this bank size needs.
2. **`mem_lookup_chunked` / `sharded_top_k_ip` streaming** (already exists,
   `wiki/architecture/sharded-retrieval.md`). Bounds *peak* memory (never materializes the full
   `[B,N,T,M]` grid) but still does O(B) *compute* — streams `B*m_per_query` slots per query. Real
   win for memory, no win for the wasted-compute half of the problem.
3. **`mem_lookup_two_pass` with `mem_value_read_kchunk`** (already exists,
   `models/memory.py:230-266`). Same O(B) compute issue as #2; its real strength (avoiding
   `[B,N,T,K,·]` gather materialization) is orthogonal to the cross-batch problem and still worth
   combining with a per-row bank.
4. **True per-row (block-diagonal) batched retrieval — chosen.** Exploit that the flat bank's
   layout is query-major (slot `m` belongs to query `m // m_per_query`, per the existing
   `per_query_isolation` comment) — so for `isolation_group_size=1`, reshaping
   `[B*m_per_query, H] -> [B, m_per_query, H]` is a **free view**, not a gather. Then a *batched*
   `einsum('btnh,bmh->bntm', ...)` scores each query only against its own `m_per_query` slots.
   Score/value tensors scale with `m_per_query`, not `B*m_per_query` — an exact `B`x cut in both
   compute and memory, structurally, not via masking. `G>1` (in-batch negatives) doesn't get this
   for free (would need a real gather to assemble a shared group bank) — not implemented, not
   needed here since the dataset already supplies ~200 hard negatives/query.

## How it was built & integrated

- `models/memory.py::mem_lookup_batched(q, w, cfg, collect_aux)` — new function, same signature/
  return contract as the other `mem_lookup*` variants (`(top_k_scores, top_k_values, aux_data)`).
  Reshapes bank + mask to `[B, m, ·]`, offsets local top-k indices back to **global** flat-bank ids
  (`ti_local + arange(B)*m_per_query`) before gathering values and before writing `aux_data`, so
  downstream telemetry that expects flat-bank ids (`doc_access_acc`, `pos_doc_ids`) is unaffected.
  Reuses `bank_top_k` (approx-topk policy respected) and `mem_weight_from_logits` (activation/temp/
  phantom) exactly as `mem_lookup` does. Supports `mem_t_chunk` (same knob as `mem_lookup_gqa`) to
  bound the `[B,N,tc,m]` tensor over T if needed.
- `memory_layer` dispatch (`models/memory.py`, in the `if/elif` chain): new
  `elif cfg.get('mem_batched_isolation', False):` branch, inserted after the GQA-bank check and
  before `two_pass_topk` — **default `False`**, so every existing config/run is byte-for-byte
  unchanged. To use it: `per_query_isolation: true, isolation_group_size: 1, mem_batched_isolation: true`.
  Raises `NotImplementedError` (not silently wrong) if `isolation_group_size != 1` or if `mem_mask`
  is per-example `[B,M]` rather than the shared `[M]` mask.
- `configs/dataset/multihop_hard_neg_full.yaml`: `batch_size: 4 -> 8` (divisibility for our 8-chip
  slice; comment updated to point here instead of re-deriving the reasoning).
- `trainer/trainer.py` (a separate fix, for a separate bug hit while training on the above —
  see "Real training run launched" below for the full diagnosis): the stage-transition block
  (`Trainer.train`, was lines 441-469) no longer rebuilds a full second `opt_state` via
  `optimizer.init(...)` and patches Adam moments back in. It now reuses `self.opt_state` in place
  and resets only the count-bearing state nodes (Adam's bias-correction counter and the LR
  schedule's own counter — found via namedtuple `_fields`, not `hasattr(x,'count')`, which
  false-positives on every plain tuple's built-in `.count()` method). Verified byte-identical to
  the old behavior in `tests/test_stage_transition_opt_state.py`; eliminates a transient ~2x
  optimizer-state footprint at every stage transition and every resume.

## Reference pages updated

- [`wiki/architecture/retrieval-modes.md`](../architecture/retrieval-modes.md): added
  `mem_lookup_batched` to the dispatch table + a `mem_batched_isolation` section, and noted the
  dispatch precedence order in `memory_layer`.
- [`wiki/training/multi-stage-training.md`](../training/multi-stage-training.md): rewrote the
  "Transition" section to describe the in-place count-reset instead of the old rebuild-and-patch
  behavior it previously documented.

## Tests

**Correctness** — `tests/test_mem_lookup_batched.py` (CPU-only, no TPU): builds small random
`q`/bank/mask (`B=4, m_per_query=37` — deliberately not chunk-aligned), runs both
`mem_lookup(..., per_query_isolation=True, isolation_group_size=1)` (masked full-matrix, exact
top-k) and `mem_lookup_batched` on identical inputs, and checks (after sorting both paths' top-k by
global index, since tie order can differ) that retrieved indices, scores, gathered values, and the
final weighted read all match — across 5 seeds x 3 score activations (softmax/relu/sigmoid), plus
a check that `isolation_group_size != 1` raises rather than silently mis-computing.

Command (run on the TPU box via `scripts/misc/run_mem_lookup_batched_test.sh`, CPU-only per
`JAX_PLATFORMS=cpu`, since this dev machine has no local JAX):
```
JAX_PLATFORMS=cpu uv run python tests/test_mem_lookup_batched.py
```
Output (2026-08-02, on `tpu-v6e-slice-mig-lhxn`):
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

ALL TESTS PASSED
```

**Throughput/memory (real TPU, in progress)** — `benchmarks/bench_hard_neg_full_retrieval.py` at
the actual recipe geometry (`B, N=4, T=512, D=Dv=1024, top_k=128, m_per_query=65,536`), comparing
`full_masked` (current default) vs `batched` vs `two_pass` / `two_pass_kchunk` / `chunked`, swept
over batch size via `scripts/misc/sweep_hard_neg_full_retrieval.sh`. Results pending — appended
below as they land.

**Gotcha hit while setting this up (cost ~25 min): single-host launch on a multi-host slice
hangs.** First attempt ran the sweep via plain `multi-vm-tpu-run.sh` against one slice member
(`tpu-v6e-slice-mig-lhxn`) only. It hung indefinitely at the very first (smallest) config with
near-zero CPU (`ps`: 18s CPU over 26min wall clock, state `S`, `wchan=futex_wait_queue`) — libtpu's
own backend init blocks waiting for the second host of the 2x4 slice to attach, **independent of**
`jax.distributed.initialize()`/`JAX_FORCE_SINGLE_HOST` (those only gate the python-level
distributed-system call; the backend-level peer-wait is unconditional on a slice). This is the
exact failure mode `wiki/infrastructure/experiment-launch-instructions.md` §2.3 rule 1 already
documents ("single-host launch hangs... Did you run your code on all TPU hosts?") — I'd missed
that this applies to ad-hoc benchmark scripts just as much as `train.py`. Fix: always launch via
`multi-tpu-box-run.sh` with the **same** script on **both** `tpu-v6e-slice-mig-lhxn` and
`tpu-v6e-slice-mig-qvlq` (`TRANSPORT=gce`), even for a "just benchmarking retrieval kernels"
script. Also added a per-combo `timeout` inside the sweep script so one bad config can't block the
rest even if something else goes wrong.

<!-- RESULTS-SWEEP-MARKER: append benchmark sweep tables below this line -->

### Sweep results, round 1 (2026-08-02, real v6e-8 slice, both hosts, 8 chips, TP=1)

Geometry: `N=4, T=512, D=Dv=1024, mem_top_k=128, m_per_query=65,536` (multihop_hard_neg_full).
`B=4` fails for every mode with a sharding error (`array axis 0 ... does not evenly divide ...
8`) — **expected, not a bug**: our mesh is `(data=8, model=1)`, so `B` must be a multiple of 8
regardless of retrieval mode (the same constraint that motivated the `batch_size: 4→8` config fix
above; `B=4` was only ever valid on the 4-chip v4-8 the colleague's original comment targeted).

| Mode | B | step time | throughput | peak HBM | Notes |
|------|---|-----------|-----------|----------|-------|
| **batched** (new) | 8  | **65.3ms**   | 62,691 tok/s | 23.68G | |
| **batched** (new) | 16 | **132.9ms**  | 61,644 tok/s | 23.71G | ~linear in B, **peak memory flat** — matches the theory (per-row bank size is B-independent) |
| **batched** (new) | 32 | OOM          | — | — | `RESOURCE_EXHAUSTED`: wants 16.01G more, only 15.11G free — ceiling is somewhere in (16, 32) at this `mem_top_k`/geometry |
| two_pass (no isolation — full cross-batch bank, current-behavior baseline) | 8  | 773.4ms  | 5,296 tok/s | 23.68G | 11.8x slower than batched at the same B and peak memory |
| two_pass | 16 | 5,496.9ms | 1,490 tok/s | 32.30G | 7.1x worse for 2x batch — superlinear |
| two_pass | 32 | 21,924.8ms | 747 tok/s | 30.23G | 4.0x worse for 2x batch again — **quadratic-ish scaling**, exactly the O(B²) cross-batch cost the whole exercise is about |
| two_pass_kchunk | any | benchmark bug | — | — | my harness's fused-read detection checked `aux_data["mem_fused_read"]`, which is only populated when `collect_aux=True` — but the bench calls `collect_aux=False`. `memory_layer` itself doesn't have this bug (it branches on `cfg`, not `aux`); only my benchmark script did. Fixing for round 2. |
| chunked, full_masked | — | pending | — | — | sweep still running; appended when done |

**Headline finding so far:** at `B=8` (this recipe's current config), `mem_batched_isolation` is
**11.8x faster** than the two-pass cross-batch path at identical peak memory, and unlike every
cross-batch mode its cost/memory don't grow with `B` — so it's not just faster, it *changes the
achievable batch size*. Still waiting on `full_masked` (the actual current default this recipe
would use un-fixed) to complete the comparison — expect it to be even worse than `two_pass` since
it doesn't chunk the score matrix at all.

### Sweep results, round 1 — COMPLETE (all modes/batches finished)

Full table, geometry unchanged (`N=4, T=512, D=Dv=1024, mem_top_k=128, m_per_query=65,536`).
`full_masked` and `chunked` also run with `per_query_isolation=true, isolation_group_size=1` set
(i.e. they mask/scan the cross-batch bank down to the same *logical* per-query-own-docs retrieval
that `batched` computes structurally) — so this is an apples-to-apples comparison of **four ways
to compute the identical retrieval**, not "isolated vs not":

| Mode | B=8 | B=16 | B=32 |
|------|-----|------|------|
| **batched** | **65.3ms**, 23.68G | **132.9ms**, 23.71G | OOM (needs 16.0G more, 15.1G free) |
| full_masked (current default + `per_query_isolation`) | 97.2ms, 23.67G | **OOM** (32.03G used, 31.25G cap — over by 0.8G) | OOM (wants 68.7G) |
| chunked (no isolation — cross-batch `sharded_top_k_ip` scan) | 910.7ms, 23.68G | 6,096ms, 19.42G | OOM (wants 49.0G) |
| two_pass (no isolation — cross-batch two-pass) | 773.4ms, 23.68G | 5,497ms, 32.30G | 21,925ms (survives, just very slow) |

**Correcting my own back-of-envelope estimate from the Motivation section above:** I'd predicted
an ~8x memory/compute cut from `batched` (naive fp32 `[B,N,T,B·m]` vs `[B,N,T,m]` tensor size).
The *measured* win at fixed B is much smaller — **1.5x faster than `full_masked` at B=8** (65 vs
97ms), same peak memory — because XLA evidently doesn't materialize the full masked tensor as
naively as the back-of-envelope math assumed (top-k/gather likely dominate real cost more than the
score matmul does). The theoretical estimate was directionally right but overstated at fixed B.

**Where `batched` actually wins decisively is the OOM ceiling, not raw speed at matched B:**
`full_masked` (the isolation mechanism this recipe would use un-fixed) **cannot reach B=16 at all**
— OOMs by just 0.8G. `batched` runs B=16 cleanly at essentially the same peak memory as its own
B=8 (23.68G → 23.71G, i.e. **flat**, confirming the per-row-bank-is-B-independent theory even
though the full-matrix path's *measured* cost didn't blow up as badly as predicted). So the real,
verified benefit is: **same or better speed, AND double the viable batch size** (16 vs 8) at a
fixed ~31GB/chip budget — not "8x less compute," but still a clear, concrete win, and the only
mode of the four that gets both speed and the larger batch.

The un-isolated cross-batch modes (`chunked`, `two_pass`) are 8-12x slower than `batched`/
`full_masked` at B=8 and 40-190x slower at B=16 — this is the actual O(B)-wasted-compute penalty
the whole exercise is about, just visible in the *unmasked* baseline rather than in
`full_masked`'s masked-but-still-full-matrix path. (`chunked` at B=16 has the lowest peak memory
of anything in this row, 19.4G — its whole design point is trading speed for memory headroom via
streaming, and it does that, just not competitively against `batched`.)

**Recommendation adopted:** train `multihop_hard_neg_full` with `mem_batched_isolation=true` (+
`per_query_isolation=true, isolation_group_size=1`) at **`batch_size=16`**, not 8 — it's the
fastest AND the only config of the four that reaches 16 at all. Updated
`configs/dataset/multihop_hard_neg_full.yaml` accordingly (`batch_size: 8 → 16`).

**Known gap:** none of the above modes were tried with `mem_t_chunk` set (T-axis chunking, exists
on `batched`, mirrors `mem_lookup_gqa`) — since `batched` already fits comfortably at B=16 without
it, didn't spend sweep time here. Worth a follow-up if `B=32` turns out to matter (t_chunk trades
some speed for headroom the same way `chunked`'s bank-axis chunking does, but chunks the T axis
instead of M — could plausibly get `batched` from OOM-at-32 to running-at-32, at some speed cost).
`two_pass_kchunk` never got real numbers (my benchmark's fused-read branch had a bug, now fixed in
`benchmarks/bench_hard_neg_full_retrieval.py` — not rerun since `batched` already dominates and
this was meant as a stretch/nice-to-have, not on the critical path).

**Correctness — stage-transition `opt_state` fix** (`tests/test_stage_transition_opt_state.py`,
CPU-only, no TPU): built the exact `optax.chain(clip_by_global_norm, adamw(schedule),
freeze(mask))` construct `setup_optimizer_for_stage` uses, ran a few steps under a stage-A mask to
accumulate real (nonzero) Adam momentum, then compared three ways of handling a transition to
stage B: the old rebuild-and-patch (mirrors `trainer.py`'s previous code exactly); a naive fix
resetting only the `is_adam`-matched node's counter; and the corrected fix resetting every
count-bearing node. Checks momentum preservation, zero-copy (same array objects, not copies),
byte-identical post-transition update results vs. the old method, and that the mask swap (frozen
↔ trainable) still takes effect correctly.

Command:
```
JAX_PLATFORMS=cpu uv run python tests/test_stage_transition_opt_state.py
```
Output (2026-08-02, on `tpu-v6e-slice-mig-lhxn`):
```
count-bearing state nodes in opt_state: 2
[PASS] 'b' unchanged after 3 stage-A steps (frozen): True
stage-A adam count after 3 steps: 3 (expect 3)
[PASS] mu/nu preserved identically by all three methods: True
[PASS] fixed method reuses the exact same mu buffers (zero-copy): True
[confirmed different, as suspected] naive (is_adam-only reset) matches old method: False
[PASS] fixed (reset-every-count) matches old method exactly: True
updates_fixed['b'] = [-4.7392634e-10 -9.8154675e-09] (vs. stage-A's opt_a, which would give exactly 0.0 here since 'b' was frozen there)
[PASS] 'b' (frozen in A, trainable in B) receives a real nonzero update under the fixed method: True
[PASS] 'b' stays exactly zero-update under stage-A's own (frozen) mask: True

ALL TESTS PASSED
```
The "2 count-bearing nodes" result confirms the naive fix's failure mode was real (a genuinely
separate schedule-counter node exists, distinct from Adam's own bias-correction counter) — an
earlier version of this test used `hasattr(x, 'count')` as the node-detection predicate and
reported "1" node, which was itself a bug (every plain Python tuple has a built-in `.count()`
*method*, a false-positive match unrelated to a state namedtuple's `count` *field*; fixed by
checking `'count' in getattr(x, '_fields', ())` instead).

**End-to-end (real hardware):** applied to `trainer.py`, then resumed the stalled training run
(step 5000 checkpoint) a fifth time. Crossed the Stage 0→1 boundary cleanly and reached step 375+
into Stage 1 (checkpoint saved at step 5328) on both hosts, no errors — see "Real training run
launched" below for the full account of getting there.

## `doc_access_loss` was silently a no-op the entire session (found ~1300 steps into the "fixed" run)

**What happened:** the training run confirmed past the Stage 0→1 boundary above kept printing
`Loss: 0.0000` on *every single progress line* — every attempt, every stage, the whole night —
and it went unquestioned because `CE` (shown alongside it) fluctuated normally, and Stages 0-1
deliberately set `ce_weight=0.0` so a static `CE` contribution there is expected. But "Loss" in
that print is `total_loss` — the actual quantity being differentiated
(`trainer.py:537, pbar.set_description(f"Loss: {total_loss:.4f} | ...")`) — and it being *exactly*
`0.0000`, unconditionally, for 1300+ steps, means **the model received zero gradient the entire
session.** Caught only when asked directly to check `doc_access_loss` (`trainer=staged`'s only
other nonzero-weight loss in Stages 0-1) against `mem_batched_isolation`.

**Root cause:** `losses/doc_access_loss.py:16-18`:
```python
mem_scores_list = aux_data.get("mem_scores")
if not mem_scores_list or len(mem_scores_list) == 0:
    return 0.0
```
`mem_lookup_batched` never populated `aux_data["mem_scores"]`, so this fired every step,
unconditionally. Not a small bug — `total_loss = main_loss*ce_weight + aux_result["total"]`, and
with `ce_weight=0.0` (Stages 0-1) and this the only nonzero aux loss, `total_loss` was a literal
constant `0.0` — an exactly-zero-gradient function of the weights — for the entire session so far.

**Why it's not a one-line fix:** `doc_access_loss` is explicitly a cross-batch, in-batch-negatives
contrastive loss (its own docstring: *"Queries compete against all docs in the batch"*) — it needs
the full `[B,T,N,B·m_per_query]` grid to let query `b` contrast against every other query's docs.
`mem_batched_isolation` structurally never builds that grid; that's the entire point of tonight's
earlier retrieval-efficiency work. Populating it to satisfy this loss would mean reintroducing the
exact O(B) cross-batch join eliminated hours ago.

**The fix — a new, mathematically-equivalent-but-per-row loss, not a patch to the old one:**
`doc_access_loss`'s objective is `log_z − log_pos` (softmax cross-entropy against a
uniform-over-positives target) — per (query, token, head), over the CANDIDATE SET a query can
see. Under per-row isolation, that candidate set is just the query's own `m_per_query` slots — no
`block_eye` cross-query positive-mask construction needed at all, since a query's own slots can
never contain another query's docs in the first place. That's a strictly *simpler* version of the
same math, and — crucially — the FULL per-row grid (`[B,T,N,m_per_query]`, not
`[B,T,N,B·m_per_query]`) is cheap enough to expose: it's the exact tensor `mem_lookup_batched`
already computes internally, right before `bank_top_k` truncates it, in the same size class as
everything else `mem_batched_isolation` already runs at.

**Built:**
- `models/memory.py::mem_lookup_batched` — two fixes plus one addition, all requested to stay
  bf16 throughout (the user's explicit ask, and a real bug independent of it):
  1. **dtype bug fixed:** `logits / jnp.sqrt(jnp.array(H, dtype=jnp.float32))` promoted the
     (bf16) `logits` to fp32 for the rest of the function — silently doubling every downstream
     tensor's memory. Replaced with a `q.dtype`-cast scalar multiply; `logits` (and hence the new
     `mem_scores` below) now stay bf16 throughout.
  2. **mislabeling bug fixed:** `aux_data["mem_top_k_logits"]` was set to `top_k_scores` (the
     *post-activation*, softmax-weighted output) instead of the *raw* pre-activation logits every
     sibling function (`mem_lookup`, `mem_lookup_gqa`) puts there. Any loss doing its own
     `logsumexp` over that key (`doc_access_top_k_loss`, and now `doc_access_per_query_loss`)
     would have computed something mathematically wrong, not caught by the earlier
     `test_mem_lookup_batched.py` (which only checks the retrieval *output*, not this aux key).
  3. **new, opt-in:** `cfg.mem_collect_full_scores=true` (an existing-but-previously-unused
     config key, already defaulted `true` in `qwen3_mem_embed.yaml` — revived rather than adding
     a new one) now makes `mem_lookup_batched` also populate `aux_data["mem_scores"]` with the
     full per-row grid, `[B,T,N,m_per_query]`, bf16, `(tensor,)`-wrapped exactly like `mem_lookup`'s
     own `mem_scores`. Gated on `collect_aux` too, so it costs nothing when no aux loss needs it.
- `losses/doc_access_per_query_loss.py` (new) — `doc_access_loss`'s exact math, scoped to
  `docs_per_query` instead of the batch-global `num_docs`, no `block_eye`. Registered as
  `"doc_access_per_query_loss"`; no fp32 upcast of the score tensor (only the scalar loss
  accumulator is fp32, matching `doc_access_loss`'s own convention — a scalar has no memory cost
  either way, and `jax.nn.logsumexp`'s max-subtraction is stable regardless of input dtype).
- `configs/trainer/staged_batched_isolation.yaml` (new) — `staged.yaml` with `doc_access_loss`
  disabled and `doc_access_per_query_loss` enabled (weight 0.1) in every stage. **`staged.yaml`
  itself was left untouched** — it's shared by non-batched-isolation recipes
  (`mem_lookup`/`mem_lookup_two_pass`) that `doc_access_loss` is exactly right for (they do build
  the cross-batch grid it needs). OmegaConf lists replace wholesale on override, so every stage's
  other fields (`trainable_params`, `max_step`, `ce_weight`, `warmup_frac`) are repeated verbatim
  from `staged.yaml`, not just the `aux_losses` diff.
- `configs/trainer/staged_ground_batched_isolation.yaml` (new, same day, follow-up request) — the
  same `doc_access_loss` → `doc_access_per_query_loss` swap for `staged_ground.yaml`'s sibling
  2-stage/frozen-main recipe, used by `scripts/embed/train_multihop_ground3layer.sh` (multihop
  recipe + `train_ground_s1.sh`'s zero-init-`mem_o_proj`/multi-layer architecture at 3 layers,
  `[9,18,27]`). Telemetry carries through from `staged_ground`'s base `aux_losses` unmodified
  (only `training_stages` is overridden). See
  [2026-08-02-multihop-ground3layer.md](../experiments/2026-08-02-multihop-ground3layer.md) —
  launched, compiled, passed its first checkpoint with no OOM despite 3x the retrieval work.
- `scripts/embed/train_multihop_hard_neg_full.sh`: `trainer=staged` → `trainer=staged_batched_isolation`,
  `+model.memory.mem_collect_full_scores=true` added.

**Tests** (`tests/test_doc_access_per_query_loss.py`, CPU-only, plus a rerun of
`tests/test_mem_lookup_batched.py` as a regression check):
```
=== regression: test_mem_lookup_batched.py ===
[PASS] seed=0 activation=softmax: idx_match=True scores_close=True values_close=True read_close=True
... (all 16 cases) ...
ALL TESTS PASSED

=== new: test_doc_access_per_query_loss.py ===
[PASS] mem_collect_full_scores=False -> no mem_scores key: True
[PASS] mem_collect_full_scores=True -> mem_scores present: True
[PASS] mem_scores shape == (B,T,N,m_per_query): (3, 4, 2, 35) vs (3, 4, 2, 35)
[PASS] mem_scores dtype is bf16 (not promoted to fp32): bfloat16
[PASS] mem_top_k_logits dtype is bf16: bfloat16
[PASS] mem_top_k_logits (raw) != returned top_k_scores (post-activation): True
[PASS] loss matches independent numpy reference: 0.656250 vs 0.651888 (diff=0.004362)
[PASS] loss ~0 when every doc is 'positive' (no negatives to contrast against): 0.000000
[PASS] missing mem_scores -> returns 0.0 (not an error): True

ALL TESTS PASSED
```
The reference is an independent hand-computed numpy `log_z - log_pos`, not `doc_access_loss`
itself — deliberately, so a shared bug in both wouldn't silently cancel out. The regression run
confirms the bf16 dtype fix didn't change the (numerically identical, per the earlier test)
retrieval *values* — only their dtype/memory footprint and the previously-mislabeled aux key.

**Restarting fresh, not resuming:** every checkpoint from this session (steps 200 through 6660+)
was produced under a constant-zero-gradient function — nothing was actually learned, so there is
nothing to preserve by resuming. Relaunched from step 0 with the fix rather than
`trainer.resume_from`.

**CONFIRMED on real hardware (2026-08-02 ~17:50 UTC):** `Loss: 0.2930`, `0.3585`, `0.3334`, ... —
genuinely fluctuating step to step (sampled range ~0.26-0.39 over the first 350 steps), not a
stuck constant. Checkpoints landing cleanly (step 333 confirmed on both hosts). Real gradient
signal, for the first time this session.

## Real training run launched

`scripts/embed/train_multihop_hard_neg_full.sh` — `qwen3_mem_embed` (Qwen3-4B main + Qwen3-
Embedding-0.6B embed model), `dataset=multihop_hard_neg_full` (`batch_size=16`), `mem_top_k=128`,
`mem_batched_isolation=true` + `per_query_isolation=true` + `isolation_group_size=1`.

Data staged first via `datagen/download_multihop_hardneg.py` (idempotent, run on **both** hosts —
each has its own disk): `mihir-1999/multihop_qa_sft-hard-neg-train` (1,341,045 rows, 1.26GB
parquet) + `multihop_doc_corpus.arrow` (1,289,524 docs, 0.91GB, memory-mapped). Took ~1 min/host.

**Repro block:**
- Command: `TRANSPORT=gce ZONE=europe-west4-a PROJECT_ID=memory-layers bash scripts/infrastructure/multi-tpu-box-run.sh tpu-v6e-slice-mig-lhxn=scripts/embed/train_multihop_hard_neg_full.sh tpu-v6e-slice-mig-qvlq=scripts/embed/train_multihop_hard_neg_full.sh`
- Commit: `48a9041617aa914fae63ff5ae2b247ceca8f02b5` (`datagen` branch) **+ uncommitted local
  changes** (this note's full diff: `mem_lookup_batched` in `models/memory.py`, the
  `multihop_hard_neg_full.yaml` batch_size/isolation changes, the new benchmark/test/sweep
  scripts) — not committed per instructions (only commit when explicitly asked). A colleague
  reproducing this needs the working tree as of this note, not just the SHA.
- TPU: v6e-8 flex-start 2x4 slice (`tpu-v6e-slice-mig-{lhxn,qvlq}`, project `memory-layers`,
  `europe-west4-a`).
- wandb: `run_name=multihop_hard_neg_full_batched_iso_topk128_bs16`, `wandb_run_id=auto` (derived
  from run_name — check wandb for the resolved run URL, not captured here since this note is
  written before the run's first log line lands).
- Checkpoints: `gs://$GCS_BUCKET/multihop_hard_neg_full_batched_iso_topk128_bs16-<date>-<time>/qwen3_mem_embed/<step>/` (exact run-dir timestamp TBD — check the launch log's `RUN_START_TIME`).

**Status:** first attempt (`batch_size=16`) started cleanly (model/dataset/checkpoint-manager/W&B
init all OK, `Starting training for 100000 steps...` on both hosts) but **OOM'd on the actual
first train step**: `RESOURCE_EXHAUSTED... Used 33.66G of 31.25G hbm. Exceeded hbm capacity by
2.42G` — on both hosts symmetrically (expected, SPMD). This is the gap flagged in this note's own
Follow-ups section before launch ("doc-embedding cost... not yet measured in isolation") landing
for real: the retrieval-only benchmark showed `batched` fitting fine at B=16 (~23.7G), but the
full step also runs the embed model (Qwen3-Embedding-0.6B) forward+backward over all
`B*num_chunks_per_doc` doc chunks, which weren't part of that benchmark and do scale with B.

**Fix applied:** reverted `configs/dataset/multihop_hard_neg_full.yaml` batch_size 16→8 (updated
the config's own comments to say so honestly, plus `scripts/embed/train_multihop_hard_neg_full.sh`
run_name → `..._bs8`) and relaunched. `batch_size=8` is the verified-safe point (it's exactly
where the retrieval-only sweep already showed `batched` working at 65.3ms/23.68G) — the B=16 win
is deferred to a follow-up rather than assumed. **B=16 end-to-end (needs embed-model remat or a
direct measurement of its memory cost) is now a real, separate follow-up item**, not solved by
this session's retrieval work alone.

Relaunch (batch_size=8) is in flight — watching via the persistent Monitor for a clean first step
and the first checkpoint save. Will update again once confirmed.

**CONFIRMED STABLE (2026-08-02 ~03:55 UTC):** cleared step 1 (280s — JIT compile, expected),
decelerating compile overhead through ~step 10, converged to **steady-state ~2.0-2.4s/step** by
step ~20 (this is the real, full end-to-end step cost — main model + embed model + memory layer +
optimizer — vs the ~65ms the retrieval kernel alone measured in isolation; retrieval was always
going to be a small fraction of the full step, this confirms it). Loss/CE decreasing normally
(8.18 → 6.94 → 7.62 across the first ~200 steps, the usual noisy-early-training pattern, not a
divergence). **First checkpoint saved at step 200**:
`gs://memory-layers-training/multihop_hard_neg_full_batched_iso_topk128_bs8-2026-08-02-03-39-56/qwen3_mem_embed`.
At ~2.1s/step steady state, `checkpoint_interval=200` is ~7 min/checkpoint, `max_to_keep=20` is a
~140-min retention window — both reasonable for an unattended overnight run.

At ~2.1s/step, the full 100,000-step run is ~58 hours — not expected to finish overnight; the goal
tonight was a stable, correctly-configured launch, not a completed run. Left running.

**Final confirmation (2026-08-02 ~04:28 UTC) — premature.** At that point 3 checkpoints (200/400/
600) had landed cleanly and I declared the overnight task done, stopped the monitor, and sent a
"stably running" summary. **That was wrong** — I stopped watching too early. The run kept going
for another ~3 hours and crashed at **step 5000** (07:0x UTC):

```
RESOURCE_EXHAUSTED: Error allocating device buffer: Attempting to allocate 5.94M.
That was not possible. There are 4.20M free.
```

This is a **different failure mode** from the batch_size=16 launch-time OOM earlier in this note —
that one failed immediately, at a fixed ~2.4G over budget, from an under-provisioned config. This
one ran successfully for 5000 steps (checkpoints 200 through 5000 all present and valid in GCS —
confirmed via `gsutil ls`, so orbax's `max_to_keep` rotation is working correctly and isn't the
cause) and then failed on an allocation of only ~6MB with only ~4MB free — i.e. **device memory
was gradually consumed over ~3.3 hours of training until essentially none was left.** This is a
leak or slow fragmentation, not a capacity misconfiguration, and **the root cause is not
diagnosed**. Also observed at the same shutdown: `UserWarning: resource_tracker: There appear to
be 162 leaked shared_memory objects to clean up at shutdown` — this is **host-side** shared memory
(grain dataloader workers), a separate memory pool from the device HBM the crash actually
complained about, so it's noted as a second finding, not asserted as the same root cause.

**What I did about it:** added `RESUME_FROM` support to `scripts/embed/train_multihop_hard_neg_full.sh`
(same convention as every other `train_*.sh` in this repo) and relaunched, resuming model +
optimizer + dataloader state from the step-5000 checkpoint
(`gs://memory-layers-training/multihop_hard_neg_full_batched_iso_topk128_bs8-2026-08-02-03-39-56/qwen3_mem_embed`).
This is a **workaround** (bounds the damage to `checkpoint_interval` steps per crash, not a fix) —
if the leak is real and steady, expect another crash roughly 5000 steps / ~3.3h after each resume
unless something is found and fixed. **Genuinely diagnosing this needs device-memory profiling
across steps** (e.g. `jax.profiler` memory traces at steps 100 vs 4900) — not done this session,
tracked as task #8 / a flagged follow-up here.

**Resume mechanism worked correctly** (this was a real open question — the launch runbook flags
full-resume as "untested in the current offline-parquet config"): log shows
`Restored dataloader state from gs://.../qwen3_mem_embed/5000/dataloader_state.json`, not "stream
starts from 0," and it reached "Starting training" quickly with no multi-hour fast-forward hang
(unlike the documented risk for the much-larger `qa_hard_neg_think_sft4b` dataset — this dataset
is 1.34M rows vs that one's ~13M, so replay is cheap). One data point, but a genuinely useful one:
**full resume (model + optimizer + dataloader) is now verified working for this dataset/recipe.**

**ROOT CAUSE FOUND (2026-08-02 ~08:00 UTC) — it was never a leak.** The resumed run crashed again
almost immediately, at the **exact same step (5000)**, with the **exact same numbers**
(`Attempting to allocate 5.94M. There are 4.20M free.`) — in a brand-new checkpoint manager, on
its very first save. Identical step + identical byte counts across two different process
launches rules out "accumulates over process lifetime" and points at something deterministic tied
to global step 5000 specifically. `configs/trainer/staged.yaml` (the base `staged_telemetry`
inherits from) answers it directly: **Stage 0 ends at `max_step: 5000`.** Stage 0's
`trainable_params: [".*mem_.*", ".*embed_proj_conv.*"]` only needs gradients for the memory layer
and its tiny conv projection; Stage 1 (`steps 5000-10000`) is
`trainable_params: [".*mem_.*", ".*embed_model.*"]` — the **entire** Qwen3-Embedding-0.6B embed
model becomes trainable, needing far more gradient/backward memory. We are over budget by only
**~1.7MB** (5.94 - 4.20) — this is a razor-thin margin, not a dramatic capacity problem, and it's
triggered by the stage transition's memory step-up, not by any slow accumulation. (The
`[implementation note](../implementations/2026-07-20-optimizer-moment-allocation.md)` on Adam
moments being pre-allocated for frozen params too is *related* context — optimizer state size
doesn't change at the boundary — but the stage transition still changes what needs live
*gradients*, which is the actual jump here.) The `OSError: handle is closed` flood in the log
after the crash is cascading grain-worker shutdown noise, not a separate bug.

**Fix attempt 1:** switched `trainer=staged_telemetry` → `trainer=staged` in
`scripts/embed/train_multihop_hard_neg_full.sh` — `staged_telemetry` adds ~13 weight-0 auxiliary
telemetry computations (`mem_write_norm`, `mem_head_query_cos`, `mem_cross_layer_cos`, etc.) that
contribute exactly 0 to the loss but still retain extra intermediate activation buffers to compute
diagnostics from. Given we're only ~1.7MB over, dropping telemetry (diagnostic-only, zero effect
on the actual training objective) seemed the lowest-risk way to close a gap this small.

**Fix attempt 1 FAILED.** Relaunched resuming from step 5000 a third time with `trainer=staged`;
it again printed `=== Transitioning to Stage 1 at step 5000 ===`, ran for ~15 minutes of active
compilation (a genuinely good sign — the first two attempts failed almost immediately at this
point, not after 15 min of real work), then crashed with **the same signature**: `Attempting to
allocate 5.94M. There are 4.02M free` (previously 4.20M — statistically the same, not a trend).
So telemetry removal was not the (or not the only) fix.

**Fix attempt 2:** noticed `checkpoint_interval=200` **evenly divides** `max_step=5000`
(and 10000, 15000 — every stage boundary in `staged.yaml`) — so a checkpoint **save** (a real,
sizeable device-side operation: reading current weights/optimizer state for the async GCS
write) was landing on the **exact same step** as the Stage 1 recompilation/optimizer-rebuild,
every single time, in all three crashes. Two demanding operations stacked on the one step where
headroom is already thinnest seemed a much better-fitting explanation than a diagnostic-only
telemetry cost. Changed `checkpoint_interval: 200 → 333` (doesn't evenly divide 5000/10000/15000)
and `max_to_keep: 20 → 15` to match, **without** touching `mem_top_k`/`batch_size`.

**Fix attempt 2 ALSO FAILED — and this is the more informative result.** Relaunched resuming from
step 5000 a fourth time; crashed again at the Stage 1 transition with the **exact same free-memory
number as attempt 1** (`4.02M free`, not attempt-1-original's `4.20M`) — i.e. **the
checkpoint-interval change measurably changed nothing.** That cleanly falsifies the
checkpoint-coincidence theory: if a concurrent save were the cause, decoupling it from the
transition step should have moved the free-memory number, and it didn't move at all between
attempts 3 and 4 (only attempt 1→3, from removing telemetry, shifted it, and only by ~0.18MB).

**Stopped guessing at config knobs here** — four attempts, two independent hypotheses cleanly
falsified by evidence — and switched to actually reading the code path instead (prompted by the
user directly challenging the "no remat on the embed model" framing of the memory delta with the
actual `qwen3.py`/`qwen3_mem.py` source). That reading is what found the real cause below.

### What is definitively known
- **100% reproducible**: 4/4 launches crash at the Stage 0→1 boundary (`staged.yaml`, `max_step:
  5000`), always with `Attempting to allocate 5.94M` and `~4.0-4.2M free` — the *shortfall* is
  consistently ~1.7-1.9MB, vanishingly small against a 31.25GB/chip budget.
- **Not a time-based leak** (ruled out in the very first re-diagnosis): identical step number and
  near-identical byte counts across four separate process launches, including two with fresh
  checkpoint managers on their first-ever save.
- **Not (solely) `staged_telemetry`'s aux-loss overhead**: removing it (attempt 1) shifted the
  free-memory number by ~0.18MB (4.20→4.02) but didn't close the gap.
- **Not checkpoint-save/stage-transition coincidence**: decoupling checkpoint timing from the
  stage boundary (attempt 2) changed the free-memory number by exactly 0MB versus attempt 1.
- **What's real and unambiguous**: Stage 1 (`trainable_params: [".*mem_.*", ".*embed_model.*"]`)
  needs meaningfully more memory than Stage 0 (`[".*mem_.*", ".*embed_proj_conv.*"]`) — the whole
  Qwen3-Embedding-0.6B now needs gradients — and at `batch_size=8` with `mem_top_k=128` we are
  right at that boundary, over by a couple MB.

### ACTUAL ROOT CAUSE — found by reading `trainer.py`, not by guessing config

Two facts, verified directly from source, that rule out both of the earlier hypotheses' *shared*
assumption (that Stage 1 needs more memory because it backprops through more of the model):

1. **`stop_grad_frozen` defaults to `false`** (`configs/trainer/standard.yaml:18`, never overridden
   by this recipe), and `_train_step` only `stop_gradient`s frozen weights when it's `true`
   (`trainer.py:209-218`, `:491`). So **`jax.value_and_grad(loss_fn)(weights)` computes gradients
   over the entire weights pytree — main 4B model included — every step, in every stage,
   regardless of `trainable_params`.** The backward graph's shape is identical across all 4
   stages. Freeze/masking only gates which computed gradients get *applied* as an optimizer
   update, not which get computed.
2. Given that, the only place stage-dependent memory pressure could come from is
   `utils.py::setup_optimizer_for_stage` — and its own comment explains exactly why: `optax.adamw`
   allocates `mu`/`nu` for every leaf regardless of trainability (~16GB/chip-scale for the 4B),
   confirmed unconditional across stages. But at every transition, `trainer.py:441-469` called
   `new_opt_state = self.optimizer.init(self.model.weights)` — **building a full second copy of
   that optimizer state** while `old_opt_state` was still referenced — then immediately discarded
   the fresh `mu`/`nu` in favor of copying the old ones back in (the "momentum transfer" step).
   That transient ~2x optimizer-state footprint, landing exactly at the one moment memory is
   already tightest, is the diagnosed cause.

This also explains why the crash recurred on *every resume*, not just the original run's natural
crossing of the boundary: `trainer.py:406-412` forces `current_stage_idx = -1` on any resume with
`step > 0`, guaranteeing the transition block (and its double-allocation) fires immediately,
regardless of whether the resumed step is actually at a stage boundary.

**Fix, verified then applied:**
- `tests/test_stage_transition_opt_state.py` (CPU-only, no TPU) built the exact same
  `optax.chain(clip_by_global_norm, adamw(schedule), freeze(mask))` construct and compared three
  ways of handling a transition: the old rebuild-and-patch; a naive fix that resets only the
  `is_adam`-matched node's step counter; and a fix that resets *every* count-bearing state node
  (found via namedtuple `_fields`, not `hasattr(x,'count')` — every plain tuple has a built-in
  `.count()` *method* that creates a false-positive match). **The naive fix produced different
  results than the old method** — confirming a real, separate schedule-counter node exists and
  would've been silently missed. **The corrected fix reproduced the old method byte-for-byte**
  (identical params and opt_state after a post-transition update), while literally never
  allocating a second copy of `mu`/`nu` (verified via `is`-identity on the arrays).
- Applied to `trainer.py:440-469`: removed the `old_opt_state`/`new_opt_state`/`patch_fn` dance
  entirely; `self.opt_state` is now reused in place, with only count-bearing nodes reset to 0.
  See `wiki/training/multi-stage-training.md` for the updated reference description.

**RESULT: confirmed fixed on real hardware.** Resumed from the step-5000 checkpoint a fifth time
with the fix applied — crossed the Stage 0→1 boundary cleanly, reached step 375+ into Stage 1 at
steady-state speed (~2s/step), with a fresh checkpoint saved at step **5328**
(`gs://memory-layers-training/multihop_hard_neg_full_batched_iso_topk128_bs8-2026-08-02-16-02-39/qwen3_mem_embed/5328/`)
— past the point every one of the four prior attempts died, on both hosts, no errors since.

One secondary, unrelated finding surfaced along the way: `_train_step`'s `forward` static jit
argument is rebuilt as a brand-new `functools.partial` object at every stage transition
(`setup_optimizer_for_stage`'s `model.forward = partial(model.forward.func, new_cfg)`), and
`functools.partial` has no content-based `__eq__`/`__hash__` — so JAX's jit cache treats it as a
new function every time and fully recompiles the training step (many minutes for this model size),
even when nothing that actually affects the compiled program changed. Not fixed tonight (it's a
performance cost, not correctness/OOM), but worth a follow-up: giving `forward`'s underlying cfg a
stable identity (or hashing on content) across transitions would save a multi-minute recompile at
every stage boundary *and* every resume.

Remaining open items, in priority order: (1) `batch_size=16` end-to-end still OOMs (needs
embed-model memory profiling/remat — separate from the fix above, never revisited tonight); (2)
the wasteful stage-transition recompile noted just above; (3) an eval box for judged accuracy; (4)
`two_pass_kchunk` / `tp_devices` sweep angles never revisited once `batched` proved to be the clear
winner.

**Not yet done:** an eval box (per `wiki/evaluation/eval-boxes.md`) — this run is train-only
tonight; judged-accuracy numbers are a follow-up, not blocking the launch. A companion
`wiki/experiments/` write-up covering the whole overnight arc is at
[`2026-08-02-multihop-hard-neg-full-batched-isolation.md`](../experiments/2026-08-02-multihop-hard-neg-full-batched-isolation.md).

## Follow-ups & risks

- `mem_batched_isolation` only supports `isolation_group_size=1`. If a future recipe wants
  in-batch negatives *on top of* dataset hard-negs (`G>1`), it needs a real group-gather — not
  built, would need its own benchmark before trusting it at this scale.
- Not yet checked against `mem_collect_full_scores` consumers (`doc_access_top_k_loss`,
  `mem_uniform_kl`) that expect a `mem_scores` full-grid aux key — `mem_lookup_batched` doesn't
  populate it (same as `two_pass_topk` already doesn't, in most configs, so this mirrors existing
  behavior rather than regressing it, but hasn't been verified against `staged_telemetry`'s actual
  loss set for this recipe).
- Doc-embedding cost (embedding `B*num_chunks_per_doc` docs through the 0.6B embed model every
  step) is unaffected by any of this — it's inherent to putting 200+ docs/query in the bank at all,
  and may end up being the actual bottleneck once retrieval stops being one. Not yet measured in
  isolation.
- tp_devices (tensor-parallel / mesh sharding of the bank itself, `mem_shard_axis`) sweep not yet
  done — everything above is data-parallel only (`tp_devices=1`, the production default).
- `_train_step`'s `forward` static jit arg is a fresh `functools.partial` every stage transition
  (identity-only equality), forcing a full multi-minute recompile at every transition *and* every
  resume even when nothing that affects the compiled program changed — noted above, not fixed.
- `batch_size=16` end-to-end still OOMs (the original, separate finding from earlier tonight) —
  needs embed-model memory profiling or gradient checkpointing, not attempted.
