# Streaming SnapKV baseline (JAX): full-corpus context stuffing with KV compression

**Date:** 2026-07-21 · **Author:** rohunagrawal (with Claude, overnight autonomous run) ·
**Status:** done · **Branch:** pending PR (same change-set as the hybrid evaluator)

**What changed:** `scripts/embed/snapkv_stream.py` — a standalone baseline that streams an
entire corpus into an off-the-shelf Qwen3-4B context under a ≤32k position budget,
compressing the KV cache SnapKV-style per segment — plus one additive function,
`models/qwen3.py::forward_window_scores` (a normal forward of a window chunk that also emits
per-layer, per-KV-head softmax mass over the cache; `jax.nn.dot_product_attention` hides
scores, so the layer loop is duplicated). Runners: `snapkv_musique.sh` (CORPUS/COMP env),
`snapkv_qasper.sh`, `snapkv_debug.sh` (two-arm isolation harness). Results feed the Pareto
plots ([experiment page](../experiments/2026-07-21-snapkv-baseline.md)).

## Options weighed

- **JAX reimplementation vs upstream repo.** Upstream SnapKV is CUDA + HF-transformers
  monkey-patching; running it would mean torch-xla bring-up on the slice. The algorithm is
  ~40 lines against our stack, so reimplementation won.
- **Streaming segments vs long-context RoPE.** `models/qwen3.py` has plain RoPE (no YaRN), so
  a 131k+ single-window prefill is out of reach. Corpus is prefilled in 4,096-token segments:
  prefill → probe → keep top-C per (layer, KV head) → compact → next segment starts at the
  compacted frontier. Positions are "compressed-space", so the budget never exceeds native
  32,768.
- **Question-conditioned vs query-agnostic probes.** MuSiQue runs faithful SnapKV (probe =
  the question; per-query corpus re-streaming — the honest throughput cost of the method).
  QASPER cannot afford 8.24M tokens/query, so it uses `--probe self` (segment scored by its
  own last tokens, TOVA-flavored): ONE query-independent cache, built once, shared — labeled
  SnapKV-inspired, not faithful.
- **Gather mechanics.** Explicit-sharding mode rejects `take_along_axis` on the sharded cache;
  the compaction gathers via a one-hot einsum with pinned PartitionSpecs (tp=1 → the reshards
  are metadata-only).

## The correctness-critical piece: position repacking

Kept keys carry RoPE phases up to `seg_end`, while the next content restarts at `cur+C` —
without correction, queries end up at LOWER phases than cached keys and greedy decoding
degenerates (observed: endless newline runs). Fixed in `compact()`: gathered keys are
**re-rotated by exactly (dst_phase − src_phase)** — RoPE composes, so the delta rotation is
exact and the cache keeps phase ≡ slot everywhere. Isolated via `snapkv_debug.sh`
(identity-compaction and no-compaction arms both coherent; real compression broken → the
positions were the only remaining variable).

## Other bring-up findings (test record is the smoke ladder in the experiment page)

- `create_mask` returns a **4-dim** mask (`[B/1, 1, T, S]`); assuming 3-dim let `jnp.where`
  silently rank-upgrade the probe logits to 6-dim and corrupt score aggregation. Fixed, and
  the reduction was made label-driven (`einsum('bkgts->bks')`) so a future shape drift errors
  loudly instead of mis-summing.
- Multi-host: jitted wrappers must take weights as ARGUMENTS (closure over global arrays is
  illegal); `s[0]` on a data-sharded batch axis is an illegal slice (use a batch mean of the
  identical tiled rows); one host's transient HF 504 kills the whole slice run (retry/backoff
  around dataset loads).
- Decode runs in on-device blocks of 128 with a host EOS check per block (a per-token
  `process_allgather` costs ~50 ms/token). Padding tokens must never follow the assistant
  tag (greedy continues the pad run); tails are front-padded.
- allenai/qasper is a script dataset (unsupported by modern `datasets`); read the Hub's
  `refs/convert/parquet` branch. The parquet schema nests answers as
  `qas.answers[i].answer[j].{free_form_answer,extractive_spans}`.

## Reference pages updated

None needed: this is a standalone baseline script, not part of the eval framework
(`get_evaluator` untouched), and `forward_window_scores` is additive with no behavior change
to existing paths. The experiment page carries the results; the
[hybrid implementation note](2026-07-20-rag-hybrid-evaluator.md) carries the multi-host
groundwork this reuses.

## Follow-ups & risks

- Compression ladder (8×/16× at c512) unmeasured; only the minimum-that-fits was run per
  corpus (c512→4×, c2048→16×, QASPER→256–320×).
- The QASPER indexing loop is compaction-dispatch-bound (36 `compact` calls × 2,011 segments
  ≈ 59 min); batching compaction across layers would cut it several-fold.
- Cross-segment RoPE geometry after repacking is approximate by construction (positions are
  compressed-space, not original) — inherent to all streaming compression, worth stating
  wherever numbers are quoted.
- `judge_one.py`/`snapkv_judge.sh` (box-side standalone judge) are session helpers, not repo
  files; fold into `scripts/misc/` if SnapKV becomes a recurring baseline.
