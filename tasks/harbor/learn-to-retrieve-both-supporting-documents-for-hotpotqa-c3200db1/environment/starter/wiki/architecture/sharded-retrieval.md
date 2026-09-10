# Sharded Retrieval (`retrieval_ops.py`)

Distributed/chunked top-k inner-product retrieval over the bank. Public entry:
`sharded_top_k_ip(query, mem_k, mem_v, top_k, …)` → `(scores, values, indices, all_scores)`.

## Dispatch
Mesh resolved as: `concrete_mesh` arg → `_global_mesh` → none. If the bank is sharded on
`shard_axis` (`n_shards>1`) → **sharded path**, else → **replicated path**.

## Sharded path (`_sharded_top_k`)
`shard_map` runs a local matmul + local `bank_top_k` per device; only `n_shards·K` candidates
cross device boundaries (no full-bank all-gather), then a final `top_k` merges them.
- **Fast path** (score bytes ≤ `SCORE_BUDGET`=2 GB): one matmul per device.
- **Scan path** (large T/prefill): per-device chunked scan (`_scan_chunks`) so the local
  `[B,N,T,local_M]` score tensor stays bounded.
- **`keys_only`**: skip `mem_v` (two-pass pass-1 discards values).
- **CPU value offload**: `set_cpu_mem_v(np_array)` keeps `mem_v` in host RAM; values fetched
  for the top-K indices via `pure_callback` (`_lookup_cpu_mem_v`) — bank never on device.

## Replicated path (`_replicated_top_k` → `_scan_chunks`)
`lax.scan` over `CHUNK_SIZE`=8192 chunks maintaining a running top-k buffer. Only path that
can `return_all_scores` (the full `[B,N,T,M]` grid for `doc_access`/`mem_uniform_kl`).

## Must-know
- **`set_global_mesh(mesh)`** must be called before any JIT call using the sharded path —
  inside JIT the weights are abstract, so `mem_k.sharding.mesh` isn't reachable at trace time.
- `mem_weight_from_logits` (activation/temp/`phantom_log_n`) turns final logits into weights
  here — see [retrieval-modes.md](retrieval-modes.md).
- `chunked_memory_top_k_retrieval` (`memory_utils.py`) is a standalone single-device variant
  of the scan.

**Related:** [memory-layer.md](memory-layer.md) · [memory-bank-construction.md](memory-bank-construction.md).
