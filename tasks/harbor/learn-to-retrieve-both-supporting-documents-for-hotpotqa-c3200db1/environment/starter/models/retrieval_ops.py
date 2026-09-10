"""
Sharded and chunked top-k inner-product retrieval for large memory banks.

Two strategies, selected based on whether mem_k is sharded across devices:

1. **Sharded** — shard_map computes local matmul + local top-k per device, then
   only the tiny K candidates cross device boundaries.  Avoids all-gathering the
   full memory bank.  For large T (prefill), automatically falls back to a
   per-device chunked scan so the local score tensor [B,N,T,local_M] stays within
   SCORE_BUDGET.

2. **Replicated / single-device** — lax.scan through memory in fixed-size chunks,
   maintaining a running top-k buffer.  Optionally returns all raw scores for
   auxiliary losses.

The concrete Mesh must be registered via set_global_mesh() before any JIT-compiled
call that uses the sharded path.  This is necessary because inside JIT the weights
are traced abstract values and their .sharding.mesh is not accessible.
"""

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from utils import is_jax_mesh_active
from .memory_utils import bank_top_k


CHUNK_SIZE = 8192
# Max bytes for the per-device score tensor [B, N, T, local_M] (float32)
# before falling back to a per-device chunked scan inside shard_map.
# 2 GB is conservative for TPU v4/v5.
SCORE_BUDGET = 2 * (1024 ** 3)

def mem_weight_from_logits(logits, activation="softmax", temp=1.0, phantom_log_n=0.0, axis=-1):
    """Convert top-k retrieval logits to slot weights for the value read.

    activation:
      "softmax"  — cross-slot normalized (corpus-size SENSITIVE: denominator grows with the
                   number of competitive slots). Optional `phantom_log_n > 0` adds a constant
                   background mass exp(phantom_log_n) to the denominator, emulating competition
                   against ~exp(phantom_log_n) extra distractors (train-time long-corpus proxy).
      "relu"     — element-wise max(0, .), UNnormalized. Corpus-size INVARIANT: a slot's weight
                   depends only on its own score, not on how many others compete.
      "sigmoid"  — element-wise logistic in [0,1], also corpus-invariant, bounded.
    `temp` divides the logits first (lower = sharper) for every activation.
    """
    z = logits / temp
    if activation == "relu":
        return jax.nn.relu(z)
    if activation == "sigmoid":
        return jax.nn.sigmoid(z)
    # softmax (default), optionally with a phantom background-mass term in the denominator
    if phantom_log_n and phantom_log_n > 0.0:
        m = jax.lax.stop_gradient(jnp.max(z, axis=axis, keepdims=True))
        e = jnp.exp(z - m)
        denom = jnp.sum(e, axis=axis, keepdims=True) + jnp.exp(phantom_log_n - m)
        return e / denom
    return jax.nn.softmax(z, axis=axis)


# Module-level concrete Mesh.  Set this before JIT compilation via
# set_global_mesh(mesh) so the sharded path can use it at trace time.
_global_mesh = None

# When set, mem_v lives on CPU (numpy float32). Retrieval uses io_callback
# to fetch only the top-K rows after index selection, avoiding placing the
# full mem_v on device.  Set via set_cpu_mem_v() before inference.
_cpu_mem_v = None


def set_global_mesh(mesh):
    global _global_mesh
    _global_mesh = mesh


def set_cpu_mem_v(mem_v_np):
    """Store mem_v on CPU (numpy float32) for callback-based lookup.
    Pass None to disable CPU path and revert to device mem_v.
    """
    global _cpu_mem_v
    _cpu_mem_v = mem_v_np


def _lookup_cpu_mem_v(flat_indices, Dv, out_dtype):
    """Fetch rows from CPU mem_v by flat indices, return on device."""
    import numpy as np
    B_N_T_K = flat_indices.shape[0]
    result_shape = jax.ShapeDtypeStruct((B_N_T_K, Dv), jnp.float32)

    def _cb(idx):
        return _cpu_mem_v[idx].astype(np.float32)

    values_f32 = jax.pure_callback(_cb, result_shape, flat_indices)
    return values_f32.astype(out_dtype)


def _pad_to_multiple(mem, multiple: int):
    M = mem.shape[0]
    remainder = M % multiple
    if remainder == 0:
        return mem, M
    pad_len = multiple - remainder
    mem = jnp.concatenate([mem, jnp.zeros((pad_len,) + mem.shape[1:], dtype=mem.dtype)], axis=0)
    return mem, M


def _matmul_top_k(query, keys, values, mask, top_k, D, index_offset, cfg=None):
    """Full matmul + top-k + local value gather on a single shard.

    query  : [B, N, T, D]
    keys   : [local_M, D]
    values : [local_M, Dv]
    mask   : [local_M] (shared) or [B, local_M] (per-example) bool, or None
    Returns (scores [B,N,T,K], values [B,N,T,K,Dv], indices [B,N,T,K]).
    """
    logits = jnp.einsum('bntd,md->bntm', query, keys,
                        preferred_element_type=query.dtype)
    logits = logits / jnp.sqrt(jnp.array(D, dtype=jnp.float32))
    if mask is not None:
        mbc = mask[None, None, None, :] if mask.ndim == 1 else mask[:, None, None, :]
        logits = jnp.where(mbc, logits, jnp.finfo(jnp.float32).min)
    top_scores, top_sel = bank_top_k(logits, top_k, cfg)
    top_indices = top_sel + index_offset
    top_values = values[top_sel]  # [B, N, T, K, Dv]
    return top_scores, top_values, top_indices


def _scan_chunks(query, mem_k_chunks, mask_chunks, real_M, top_k, chunk_size,
                 D, carry_sharding, use_mesh, index_offset=0, return_all_scores=False):
    """Scan over memory chunks, maintaining a running top-k buffer.

    Returns (final_scores, final_indices, all_scores).
    all_scores is [B, N, T, real_M] when return_all_scores=True, else None.
    """
    B, N, T, _ = query.shape
    n_chunks = mem_k_chunks.shape[0]

    def scan_fn(carry, chunk_idx):
        top_scores, top_indices = carry
        k_chunk = jax.lax.dynamic_slice_in_dim(mem_k_chunks, chunk_idx, 1, axis=0)[0]

        if use_mesh:
            scores = jnp.einsum('bntd,cd->bntc', query, k_chunk,
                                preferred_element_type=query.dtype,
                                out_sharding=P('data', 'model', None, None))
        else:
            scores = jnp.einsum('bntd,cd->bntc', query, k_chunk,
                                preferred_element_type=query.dtype)

        scores = scores / jnp.sqrt(jnp.array(D, dtype=jnp.float32))

        if mask_chunks is not None:
            # mask_chunks [n_chunks, chunk] (shared) or [n_chunks, B, chunk] (per-example)
            m = jax.lax.dynamic_slice_in_dim(mask_chunks, chunk_idx, 1, axis=0)[0]
            mbc = m[None, None, None, :] if m.ndim == 1 else m[:, None, None, :]
            scores = jnp.where(mbc, scores, jnp.finfo(jnp.float32).min)

        offsets = chunk_idx * chunk_size + jnp.arange(chunk_size)
        scores = jnp.where((offsets < real_M)[None, None, None, :], scores,
                           jnp.finfo(jnp.float32).min)

        chunk_indices = index_offset + chunk_idx * chunk_size + jnp.arange(chunk_size)
        chunk_indices = jnp.broadcast_to(chunk_indices[None, None, None, :], scores.shape)

        if carry_sharding is not None:
            scores        = jax.sharding.reshard(scores,        carry_sharding)
            chunk_indices = jax.sharding.reshard(chunk_indices, carry_sharding)

        combined_scores  = jnp.concatenate([top_scores,  scores],       axis=-1)
        combined_indices = jnp.concatenate([top_indices, chunk_indices], axis=-1)
        new_top_scores, sel = jax.lax.top_k(combined_scores, top_k)
        new_top_indices = jnp.take_along_axis(combined_indices, sel, axis=-1)

        if carry_sharding is not None:
            new_top_scores  = jax.sharding.reshard(new_top_scores,  carry_sharding)
            new_top_indices = jax.sharding.reshard(new_top_indices, carry_sharding)

        per_step_out = scores if return_all_scores else None
        return (new_top_scores, new_top_indices), per_step_out

    init_scores  = jnp.full((B, N, T, top_k), jnp.finfo(jnp.float32).min, dtype=jnp.float32)
    init_indices = jnp.zeros((B, N, T, top_k), dtype=jnp.int32)
    if carry_sharding is not None:
        init_scores  = jax.sharding.reshard(init_scores,  carry_sharding)
        init_indices = jax.sharding.reshard(init_indices, carry_sharding)

    (final_scores, final_indices), chunk_scores_out = jax.lax.scan(
        scan_fn, (init_scores, init_indices), jnp.arange(n_chunks))

    all_scores = None
    if return_all_scores and chunk_scores_out is not None:
        # [n_chunks, B, N, T, chunk_size] → [B, N, T, real_M]
        all_scores = (jnp.transpose(chunk_scores_out, (1, 2, 3, 0, 4))
                      .reshape(B, N, T, -1)[:, :, :, :real_M])

    return final_scores, final_indices, all_scores


def _replicated_top_k(query, mem_k, mem_v, top_k, mem_mask, chunk_size, D, Dv, B, N, T,
                       use_mesh, return_all_scores=False, keys_only=False, temp=1.0,
                       activation="softmax", phantom_log_n=0.0):
    mem_k_padded, real_M = _pad_to_multiple(mem_k, chunk_size)
    if not keys_only:
        mem_v_padded, _  = _pad_to_multiple(mem_v, chunk_size)
    M_padded = mem_k_padded.shape[0]

    if mem_mask is not None:
        pad_len = M_padded - mem_mask.shape[-1]
        if pad_len > 0:
            pad_shape = mem_mask.shape[:-1] + (pad_len,)
            mem_mask = jnp.concatenate(
                [mem_mask, jnp.zeros(pad_shape, dtype=mem_mask.dtype)], axis=-1)

    mem_k_chunks = mem_k_padded.reshape(M_padded // chunk_size, chunk_size, D)
    if mem_mask is None:
        mask_chunks = None
    elif mem_mask.ndim == 1:
        mask_chunks = mem_mask.reshape(M_padded // chunk_size, chunk_size)
    else:
        # per-example [B, M] -> [n_chunks, B, chunk] so the scan still slices along chunks
        mask_chunks = mem_mask.reshape(
            mem_mask.shape[0], M_padded // chunk_size, chunk_size).transpose(1, 0, 2)
    carry_sharding = P('data', 'model', None, None) if use_mesh else None

    final_scores, final_indices, all_scores = _scan_chunks(
        query, mem_k_chunks, mask_chunks, real_M, top_k, chunk_size,
        D, carry_sharding, use_mesh, index_offset=0,
        return_all_scores=return_all_scores)

    top_k_scores = mem_weight_from_logits(
        final_scores, activation, temp, phantom_log_n).astype(query.dtype)
    if keys_only:
        return top_k_scores, None, final_indices, all_scores
    if use_mesh:
        top_k_values = mem_v_padded.at[final_indices].get(
            out_sharding=P('data', 'model', None, None, None))
    else:
        top_k_values = mem_v_padded[final_indices]

    return top_k_scores, top_k_values, final_indices, all_scores


def _sharded_top_k(query, mem_k, mem_v, top_k, mem_mask, chunk_size, D, Dv, B, N, T,
                    n_shards, shard_axis, concrete_mesh, keys_only=False, temp=1.0,
                    activation="softmax", phantom_log_n=0.0, cfg=None):
    """shard_map-based retrieval: local matmul+top_k per device, cross-device reduce.

    Each device operates on its local mem_k slice.  Only n_shards*K candidates
    cross device boundaries.  When _cpu_mem_v is set, mem_v is fetched from CPU
    via pure_callback after index selection — mem_v never lives on device.
    When keys_only=True, skip mem_v entirely and return None for values.
    """
    from jax.experimental.shard_map import shard_map

    if mem_mask is not None and mem_mask.ndim != 1:
        raise NotImplementedError(
            "per-example [B, M] mem_mask is only supported in the replicated path "
            "(shard axis of size 1): the sharded path reshards the mask over the bank axis."
        )

    use_cpu_v = _cpu_mem_v is not None
    skip_v = use_cpu_v or keys_only

    mem_k = jax.sharding.reshard(mem_k, P(shard_axis, None))
    if not skip_v:
        mem_v = jax.sharding.reshard(mem_v, P(shard_axis, None))
    if mem_mask is not None:
        mem_mask = jax.sharding.reshard(mem_mask, P(shard_axis))

    query_rep = jax.sharding.reshard(query, P(None, None, None, None))
    local_M = mem_k.shape[0] // n_shards
    device_offsets = jnp.arange(n_shards, dtype=jnp.int32) * local_M
    device_offsets = jax.sharding.reshard(device_offsets, P(shard_axis))

    sa = shard_axis
    has_mask = mem_mask is not None
    score_bytes = B * N * T * local_M * 4  # float32

    if score_bytes <= SCORE_BUDGET:
        # ── Fast path: single matmul per device ──────────────────────────
        if skip_v:
            def _per_shard(q, k_local, offset, *mask_args):
                m = mask_args[0] if mask_args else None
                logits = jnp.einsum('bntd,md->bntm', q, k_local,
                                    preferred_element_type=q.dtype)
                logits = logits / jnp.sqrt(jnp.array(D, dtype=jnp.float32))
                if m is not None:
                    logits = jnp.where(m[None, None, None, :], logits,
                                       jnp.finfo(jnp.float32).min)
                top_scores, top_sel = bank_top_k(logits, top_k, cfg)
                top_indices = top_sel + offset
                return top_scores[None], top_indices[None]

            in_specs = (P(None, None, None, None), P(sa, None), P(sa))
            if has_mask:
                in_specs = in_specs + (P(sa),)
            out_specs = (P(sa, None, None, None, None),
                         P(sa, None, None, None, None))
            mapped_fn = shard_map(_per_shard, mesh=concrete_mesh,
                                  in_specs=in_specs, out_specs=out_specs,
                                  check_rep=False)
            args = (query_rep, mem_k, device_offsets)
            if has_mask:
                args = args + (mem_mask,)
            per_dev_scores, per_dev_indices = mapped_fn(*args)
            per_dev_values = None
        else:
            def _per_shard(q, k_local, v_local, offset, *mask_args):
                m = mask_args[0] if mask_args else None
                scores, values, indices = _matmul_top_k(q, k_local, v_local, m, top_k, D, offset, cfg)
                return scores[None], values[None], indices[None]

            in_specs = (P(None, None, None, None), P(sa, None), P(sa, None), P(sa))
            if has_mask:
                in_specs = in_specs + (P(sa),)
            out_specs = (P(sa, None, None, None, None),
                         P(sa, None, None, None, None, None),
                         P(sa, None, None, None, None))
            mapped_fn = shard_map(_per_shard, mesh=concrete_mesh,
                                  in_specs=in_specs, out_specs=out_specs,
                                  check_rep=False)
            args = (query_rep, mem_k, mem_v, device_offsets)
            if has_mask:
                args = args + (mem_mask,)
            per_dev_scores, per_dev_values, per_dev_indices = mapped_fn(*args)

    else:
        # ── Scan path: chunked scan per device (large T / prefill) ───────
        # Use chunk_size if provided (controls peak per-step memory, hence XLA
        # compilation memory).  Fall back to SCORE_BUDGET-derived size only if
        # chunk_size was not specified (chunk_size defaults to CHUNK_SIZE=8192).
        if chunk_size < local_M:
            scan_chunk = chunk_size
        else:
            scan_chunk = max((SCORE_BUDGET // (B * N * T * 4) // 128) * 128, 128)
        scan_chunk = min(scan_chunk, local_M)
        pad_to = ((local_M + scan_chunk - 1) // scan_chunk) * scan_chunk
        n_local_chunks = pad_to // scan_chunk

        if skip_v:
            def _per_shard_scan(q, k_local, offset, *mask_args):
                m = mask_args[0] if mask_args else None
                pad_len = pad_to - local_M
                k_pad = jnp.concatenate([k_local,
                                          jnp.zeros((pad_len, D), dtype=k_local.dtype)], axis=0)
                k_chunks = k_pad.reshape(n_local_chunks, scan_chunk, D)
                if m is not None:
                    m_pad = jnp.concatenate([m, jnp.zeros(pad_len, dtype=m.dtype)])
                    m_chunks = m_pad.reshape(n_local_chunks, scan_chunk)
                else:
                    m_chunks = None
                scores, indices, _ = _scan_chunks(
                    q, k_chunks, m_chunks,
                    real_M=local_M,
                    top_k=top_k, chunk_size=scan_chunk, D=D,
                    carry_sharding=None, use_mesh=False,
                    index_offset=offset)
                return scores[None], indices[None]

            in_specs = (P(None, None, None, None), P(sa, None), P(sa))
            if has_mask:
                in_specs = in_specs + (P(sa),)
            out_specs = (P(sa, None, None, None, None),
                         P(sa, None, None, None, None))
            mapped_fn = shard_map(_per_shard_scan, mesh=concrete_mesh,
                                  in_specs=in_specs, out_specs=out_specs,
                                  check_rep=False)
            args = (query_rep, mem_k, device_offsets)
            if has_mask:
                args = args + (mem_mask,)
            per_dev_scores, per_dev_indices = mapped_fn(*args)
            per_dev_values = None
        else:
            def _per_shard_scan(q, k_local, v_local, offset, *mask_args):
                m = mask_args[0] if mask_args else None
                pad_len = pad_to - local_M
                k_pad = jnp.concatenate([k_local,
                                          jnp.zeros((pad_len, D), dtype=k_local.dtype)], axis=0)
                k_chunks = k_pad.reshape(n_local_chunks, scan_chunk, D)
                if m is not None:
                    m_pad = jnp.concatenate([m, jnp.zeros(pad_len, dtype=m.dtype)])
                    m_chunks = m_pad.reshape(n_local_chunks, scan_chunk)
                else:
                    m_chunks = None
                scores, indices, _ = _scan_chunks(
                    q, k_chunks, m_chunks,
                    real_M=local_M,
                    top_k=top_k, chunk_size=scan_chunk, D=D,
                    carry_sharding=None, use_mesh=False,
                    index_offset=offset)
                local_indices = indices - offset
                values = v_local[local_indices]
                return scores[None], values[None], indices[None]

            in_specs = (P(None, None, None, None), P(sa, None), P(sa, None), P(sa))
            if has_mask:
                in_specs = in_specs + (P(sa),)
            out_specs = (P(sa, None, None, None, None),
                         P(sa, None, None, None, None, None),
                         P(sa, None, None, None, None))
            mapped_fn = shard_map(_per_shard_scan, mesh=concrete_mesh,
                                  in_specs=in_specs, out_specs=out_specs,
                                  check_rep=False)
            args = (query_rep, mem_k, mem_v, device_offsets)
            if has_mask:
                args = args + (mem_mask,)
            per_dev_scores, per_dev_values, per_dev_indices = mapped_fn(*args)

    # ── Cross-device reduction (only K * n_shards tiny candidates) ───────
    per_dev_scores  = jax.sharding.reshard(per_dev_scores,  P(None, None, None, None, None))
    per_dev_indices = jax.sharding.reshard(per_dev_indices, P(None, None, None, None, None))

    merged_scores  = per_dev_scores.transpose(1, 2, 3, 0, 4).reshape(B, N, T, n_shards * top_k)
    merged_indices = per_dev_indices.transpose(1, 2, 3, 0, 4).reshape(B, N, T, n_shards * top_k)

    final_scores, sel = jax.lax.top_k(merged_scores, top_k)
    final_indices = jnp.take_along_axis(merged_indices, sel, axis=-1)  # [B, N, T, K]

    top_k_scores = mem_weight_from_logits(
        final_scores, activation, temp, phantom_log_n).astype(query.dtype)

    if keys_only:
        return top_k_scores, None, final_indices
    elif use_cpu_v:
        # Fetch only the K rows we need from CPU mem_v — never materialises
        # the full mem_v on device.
        final_values = _lookup_cpu_mem_v(
            final_indices.reshape(-1), Dv, query.dtype
        ).reshape(B, N, T, top_k, Dv)
    else:
        per_dev_values = jax.sharding.reshard(per_dev_values,
                                               P(None, None, None, None, None, None))
        merged_values = per_dev_values.transpose(1, 2, 3, 0, 4, 5).reshape(
            B, N, T, n_shards * top_k, Dv)
        final_values = jnp.take_along_axis(merged_values, sel[..., None], axis=-2)

    return top_k_scores, final_values, final_indices


def sharded_top_k_ip(query, mem_k, mem_v, top_k, mem_mask=None, chunk_size=CHUNK_SIZE,
                      concrete_mesh=None, shard_axis='model', return_all_scores=False,
                      keys_only=False, temp=1.0, activation="softmax", phantom_log_n=0.0,
                      cfg=None):
    """Top-k inner-product retrieval, dispatching to sharded or replicated-chunked path.

    The concrete Mesh is resolved in this order:
      1. ``concrete_mesh`` argument (explicit override)
      2. ``_global_mesh`` module variable (set via set_global_mesh() before JIT)
      3. Fallback to replicated path

    Args:
        query:             [B, N, T, D]
        mem_k:             [M, D]
        mem_v:             [M, Dv]
        top_k:             int
        mem_mask:          [M] bool (True = valid), optional. A per-example [B, M] mask is
                           also accepted, replicated path only (the sharded path raises).
        chunk_size:        int, memory entries per scan step (replicated path only)
        concrete_mesh:     jax.sharding.Mesh or None.
        shard_axis:        mesh axis name memory is sharded on (default 'model')
        return_all_scores: if True, also return [B, N, T, M] raw scores.
                           Only available in the replicated path; None otherwise.
        keys_only:         if True, skip mem_v entirely and return None for values.
                           Used in two-pass Pass 1 where values are discarded.

    Returns:
        top_k_scores:  [B, N, T, K]   softmax-normalised
        top_k_values:  [B, N, T, K, Dv] or None (when keys_only=True)
        top_k_indices: [B, N, T, K]
        all_scores:    [B, N, T, M] or None
    """
    B, N, T, D = query.shape
    Dv = mem_v.shape[-1] if not keys_only else 0
    use_mesh = is_jax_mesh_active()

    if concrete_mesh is None:
        concrete_mesh = _global_mesh

    if concrete_mesh is not None:
        n_shards = concrete_mesh.shape.get(shard_axis, 1)
        if n_shards > 1:
            scores, values, indices = _sharded_top_k(
                query, mem_k, mem_v, top_k, mem_mask, chunk_size,
                D, Dv, B, N, T, n_shards, shard_axis, concrete_mesh,
                keys_only=keys_only, temp=temp,
                activation=activation, phantom_log_n=phantom_log_n, cfg=cfg)
            return scores, values, indices, None  # all_scores unavailable in sharded path

    scores, values, indices, all_scores = _replicated_top_k(
        query, mem_k, mem_v, top_k, mem_mask, chunk_size, D, Dv, B, N, T,
        use_mesh, return_all_scores=return_all_scores, keys_only=keys_only, temp=temp,
        activation=activation, phantom_log_n=phantom_log_n)
    return scores, values, indices, all_scores
