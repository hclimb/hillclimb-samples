# Retrieval Modes

The `mem_lookup*` family in `models/memory.py`. Each takes the normed query and bank, returns
`(top_k_scores [B,N,T,K], top_k_values [B,N,T,K,Dv], aux)`. `memory_layer` selects one (see
[memory-layer.md](memory-layer.md) dispatch table).

| Fn | Selected when | Notes |
|----|---------------|-------|
| `mem_lookup` | default | Sharded (`sharded_top_k_ip`) if the bank is device-sharded, else full `einsum` score matrix + `bank_top_k`. Masks before top-k. Supports `per_query_isolation`. |
| `mem_lookup_batched` | `mem_batched_isolation=true` (+ `per_query_isolation`, `isolation_group_size=1`) | **True** per-row retrieval: reshapes the flat `[B*m,H]` bank to `[B,m,H]` (free — bank layout is query-major) and runs a *batched* `einsum('btnh,bmh->bntm', ...)`, so the score/value tensors scale with `m` (docs-per-query), not `B*m`. Not a mask on the full matrix — the cross-batch join is never built. `mem_t_chunk` bounds the `[B,N,tc,m]` tensor over T, same knob as `mem_lookup_gqa`. A genuinely per-example `[B,M]` `mem_mask` (not just a shared `[M]` one) falls back to `mem_lookup_chunked` — see below. |
| `mem_lookup_gqa` | `mem_k.ndim==3` | Per-kv-head bank `[M,Nkv,H]`; `Nq` query heads grouped into `Nkv` groups, each attends only its bank slice. Chunks over tokens (`mem_t_chunk`) to bound the `[B,Nq,T,M]` score matrix. Plain softmax. |
| `mem_lookup_chunked` | `mem_lookup_chunk_size` set | Streams a large bank in chunks via `sharded_top_k_ip`; `return_all_scores` decoupled from `collect_aux` so two-pass pass-1 stays O(K). |
| `product_key_lookup (DEPRECATED)` | `mem_use_product_keys` | Product keys: split query, score two `[sqrt_M, H/2]` half-tables, Cartesian-combine top-k → **O(√M)**. Requires `mem_size` a perfect square. |
| `mem_lookup_two_pass` | `two_pass_topk` | Gradient-efficient large-bank read; see [staged-readout.md](staged-readout.md). |

Dispatch precedence in `memory_layer` (`models/memory.py`): GQA bank shape > `mem_batched_isolation`
> `two_pass_topk` > `mem_use_product_keys` > `mem_lookup_chunk_size` set > default `mem_lookup`.

## `per_query_isolation` (in `mem_lookup`)
Flat bank layout: slot `m` belongs to query `m // (M//B)`. With group size `G`
(`isolation_group_size`), query `b` attends only to docs whose owning query is in `b`'s group
of `G` consecutive queries. `G=1` = full isolation (own docs only); `G=4` = self + 3 in-batch
negatives (no dataset hard-negs needed). Masked before top-k — the full `[B,N,T,B*m]` matrix is
still materialized; use `mem_lookup_batched` (below) when `G=1` and the bank is large enough that
the mask's wasted `(B-1)/B` compute/memory matters (see
[implementation note](../implementations/2026-08-02-hard-neg-full-efficient-retrieval.md)).

## `mem_batched_isolation` (true per-row retrieval, `mem_lookup_batched`)
Default `false` (every existing run keeps the masked full-matrix path byte-for-byte). Set
`mem_batched_isolation: true` alongside `per_query_isolation: true, isolation_group_size: 1` to
switch to the batched path: same retrieval semantics (query `b` sees only its own `m` docs), but
the score/gather tensors are `[B,N,T,m]` instead of `[B,N,T,B*m]` — an exact `B`x cut in both
compute and peak memory for the retrieval step, verified numerically identical to the masked path
(`tests/test_mem_lookup_batched.py`). Only supports `isolation_group_size=1` — raises
`NotImplementedError` otherwise (a `G>1` shared-group bank isn't a free reshape).

A genuinely per-example `[B,M]` `mem_mask` (validity differs per row, not just by block
ownership — e.g. `evals/gen_large_mem_rag_hybrid.py`'s `gather_bank` mode, where each query
retrieves its own docs) can't use the block-diagonal reshape trick, so `mem_lookup_batched`
**falls back to `mem_lookup_chunked`** (mask-shape-agnostic, via `sharded_top_k_ip`'s replicated
path) instead of raising. This only triggers on a 2D mask — the fast reshape path above is
unaffected and stays byte-identical for the `None`/shared-`[M]`-mask case every training run
uses. See
[implementation note](../implementations/2026-08-13-batched-isolation-hybrid-eval-mask-fallback.md).

Score scaling (`/ sqrt(H)`) is done with a `q.dtype`-cast scalar, not `/ jnp.array(H,
dtype=float32)` — dividing by an fp32 array promotes the (bf16) logits to fp32 for the rest of the
function, silently doubling every downstream tensor's memory. `aux_data["mem_top_k_logits"]` is
the **raw, pre-activation** logits (matching `mem_lookup`'s own convention), not the post-
activation `top_k_scores` returned as the function's actual retrieval output.

`model.memory.mem_collect_full_scores: true` (opt-in, gated on `collect_aux`) additionally
populates `aux_data["mem_scores"]` with the **full per-row** score grid `[B,T,N,m]` — pre-top-k,
same `(tensor,)` 1-tuple convention as `mem_lookup`'s own `mem_scores`. This is what
`doc_access_loss` needs, but that loss needs the *cross-batch* `[B,T,N,B*m]` grid, which this mode
never builds — use `doc_access_per_query_loss` instead (see
[auxiliary-losses.md](../training/auxiliary-losses.md)), which this per-row grid is sized for.

## Score → weight (`retrieval_ops.py::mem_weight_from_logits`)
`softmax` (corpus-size *sensitive*; `phantom_log_n>0` adds background mass as a long-corpus
proxy) · `relu` / `sigmoid` (corpus-size *invariant*, unnormalized/bounded) · `temp` divides
logits first. Resolved by `_mem_score_opts` (eval env overrides).

## `bank_top_k` (`memory_utils.py`)
Exact `jax.lax.top_k`, or TPU-fast `approx_max_k`. Which one is chosen by
`resolve_approx_topk(cfg)` with precedence **env > model cfg > built-in default (exact)**:

- **cfg keys:** `mem_approx_topk` (bool) and `mem_approx_recall` (float, default 0.95).
  `qwen3_mem_embed` sets `mem_approx_topk: true`, `mem_approx_recall: 0.99` — so approx is the
  **default for that model, train and eval** (set `mem_approx_topk: false` for exact-retrieval
  eval).
- **env overrides:** `MEM_APPROX_TOPK=0|1`, `MEM_APPROX_RECALL=<f>` win over cfg (one-off runs).

Why: at M≈65k the exact top-k is ~40% of the whole 4B train step; approx cuts the step ~1.6–1.7×
at ~98% recall@64 — see [experiment](../experiments/2026-07-15-approx-topk-training.md). `cfg` is
threaded to every call site (`mem_lookup`, `mem_lookup_gqa`, and the `sharded_top_k_ip` chain).

**Policy (2026-07-20): approx top-k is ALWAYS preferred — train and eval — for speed** (1.6–2.1×
step cuts, up to 2.4× at 2M-slot banks; see the
[2026-07-20 4-layer result](../experiments/2026-07-20-ground4layer-approx-topk-v6e-slice.md)).
Do not flip evals to exact by default. Known tradeoff, accepted: greedy decoding is not
bit-reproducible across runs whose compiled shapes differ
([2026-07-18 threats](../experiments/2026-07-18-musique-midtraining-vs-rag.md)) — approx recovers
~98.7% of exact top-64 at recall 0.99, well under the ~0.02 judge noise floor at n=128.
