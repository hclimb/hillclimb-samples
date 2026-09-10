import os
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from .qwen3 import rms_norm
from .memory_utils import get_indices_product_keys, bank_top_k
from .retrieval_ops import sharded_top_k_ip, CHUNK_SIZE, mem_weight_from_logits
from utils import is_jax_mesh_active


def _mem_score_opts(cfg):
    """Resolve (activation, temp, phantom_log_n) from cfg with env overrides (eval-time)."""
    activation = os.environ.get("MEM_SCORE_ACTIVATION", cfg.get("mem_score_activation", "softmax"))
    temp = float(os.environ.get("MEM_SOFTMAX_TEMP", cfg.get("mem_softmax_temp", 1.0)))
    phantom = float(os.environ.get("MEM_PHANTOM_LOG_N", cfg.get("mem_phantom_log_n", 0.0)))
    return activation, temp, phantom


def _get_concrete_mesh(w):
    """Extract the concrete Mesh from mem_k's sharding, if available."""
    mem_k = w.get('mem_k')
    if mem_k is None:
        return None
    sharding = getattr(mem_k, 'sharding', None)
    return getattr(sharding, 'mesh', None)


def mem_lookup(q, w, cfg, collect_aux=False):
    mem_mask = w.get('mem_mask', None)
    if mem_mask is not None and getattr(mem_mask, 'ndim', 1) != 1:
        raise NotImplementedError(
            "per-example [B, M] mem_mask is only supported by mem_lookup_chunked "
            "(set mem_lookup_chunk_size); this lookup variant assumes a shared [M] mask.")
    if cfg.get('mem_k_prenormed', False):
        mem_k_normed = w['mem_k']
    else:
        mem_k_normed = rms_norm(w['mem_k'], w['mem_k_norm'], cfg['rms_norm_eps'])
    concrete_mesh = _get_concrete_mesh(w)
    shard_axis = cfg.get('mem_shard_axis', 'model')
    activation, temp, phantom = _mem_score_opts(cfg)

    # Use shard_map path when memory is distributed across devices
    if concrete_mesh is not None and concrete_mesh.shape.get(shard_axis, 1) > 1:
        q_t = jnp.transpose(q, (0, 2, 1, 3))  # [B, N, T, H]
        top_k_scores, top_k_values, top_k_indices, _ = sharded_top_k_ip(
            q_t, mem_k_normed, w['mem_v'], cfg['mem_top_k'],
            mem_mask=mem_mask, chunk_size=CHUNK_SIZE,
            concrete_mesh=concrete_mesh, shard_axis=shard_axis,
            return_all_scores=False, temp=temp,
            activation=activation, phantom_log_n=phantom, cfg=cfg,
        )
        aux_data = None
        if collect_aux:
            aux_data = {"mem_top_k_indices": top_k_indices, "mem_top_k_logits": top_k_scores}
        return top_k_scores, top_k_values, aux_data

    # Replicated / single-device: full score matrix (fine for small M)
    logits = jnp.einsum('btnh,mh->bntm', q, mem_k_normed, preferred_element_type=q.dtype, out_sharding=P('data', 'model', None, None))
    logits = (logits / jnp.sqrt(jnp.array(q.shape[-1], dtype=jnp.float32)))

    if cfg.get('per_query_isolation', False):
        # Per-query doc isolation with a GROUP SIZE. Flat bank layout: position m belongs to query
        # m // (M // B). query_of_pos[m] is that owning query. A query b attends to every position
        # whose owning query is in the SAME group of `isolation_group_size` (G) consecutive queries.
        #   G = 1  -> b sees ONLY its own docs (full isolation).
        #   G = 4  -> b sees its own docs + the 3 batch-mates in its group -> exactly 4 docs/query
        #            (1 pos + 3 STANDARD in-batch negs), no dataset hard-negs needed.
        # Masking happens BEFORE top_k so telemetry aux reflects the masked read. Off -> identical.
        B = q.shape[0]
        M = mem_k_normed.shape[0]
        m_per_query = M // B
        G = cfg.get('isolation_group_size', 1)
        query_of_pos = jnp.arange(M) // m_per_query                                     # [M]
        query_doc_mask = (query_of_pos[None, :] // G) == (jnp.arange(B)[:, None] // G)   # [B, M]
        if mem_mask is not None:
            query_doc_mask = query_doc_mask & (jnp.reshape(mem_mask, (-1,)) != 0)[None, :]
        if is_jax_mesh_active():
            query_doc_mask = jax.sharding.reshard(query_doc_mask, P('data', None))
        logits = jnp.where(query_doc_mask[:, None, None, :], logits, -1e9)
    elif mem_mask is not None:
        logits = jnp.where(mem_mask, logits, -1e9)

    top_k_logits, top_k_indices = bank_top_k(logits, cfg['mem_top_k'], cfg)
    aux_data = None
    if collect_aux:
        aux_data = {"mem_scores": (jnp.transpose(logits, (0, 2, 1, 3)),), "mem_top_k_indices": top_k_indices, "mem_top_k_logits": top_k_logits}

    top_k_scores = mem_weight_from_logits(top_k_logits, activation, temp, phantom)
    top_k_values = w['mem_v'].at[top_k_indices].get(out_sharding=P('data', 'model', None, None, None))
    return top_k_scores, top_k_values, aux_data


def mem_lookup_batched(q, w, cfg, collect_aux=False):
    """Per-row (true block-diagonal) memory retrieval: query row b attends ONLY to the
    m_per_query bank slots that belong to it, via a BATCHED einsum against a [B, m, H] bank —
    not a cross-batch join against the flat [B*m, H] bank followed by a mask. Requires
    per_query_isolation with isolation_group_size=1 (own docs only): the flat bank's layout
    (position m belongs to query m // m_per_query, see `mem_lookup`'s per_query_isolation
    comment) makes the [B*m, H] -> [B, m, H] reshape a pure view — no gather, no data movement.

    Score/memory tensors scale with m_per_query, not B*m_per_query: an 8x cut for the
    multihop_hard_neg_full recipe (B=8) vs the masked full-matrix path. See
    wiki/implementations/2026-08-02-hard-neg-full-efficient-retrieval.md for the numbers and
    why `mem_lookup`'s existing per_query_isolation mask doesn't get this for free (it masks
    the already-materialized [B,N,T,B*m] matrix instead of never building it).

    isolation_group_size > 1 (self + G-1 in-batch negatives) is NOT supported here — that needs
    a group-gather (bank shared across G rows isn't a free reshape) and isn't needed when the
    dataset already supplies hundreds of hard negatives per query. Falls back to `mem_lookup`'s
    masked path in that case (see dispatch in `memory_layer`).

    A genuinely per-example [B, M] mask (each row's own distinct validity, not just this row's
    block boundary) falls back to `mem_lookup_chunked`, which is mask-shape-agnostic — this is
    the eval-time case: hybrid gather-bank eval builds one retrieved-doc set per query, so the
    mask differs per row, not just by ownership. See
    wiki/implementations/2026-08-13-batched-isolation-hybrid-eval-mask-fallback.md.
    """
    mem_mask = w.get('mem_mask', None)
    if mem_mask is not None and getattr(mem_mask, 'ndim', 1) != 1:
        return mem_lookup_chunked(q, w, cfg, collect_aux)
    if int(cfg.get('isolation_group_size', 1) or 1) != 1:
        raise NotImplementedError(
            "mem_lookup_batched only supports isolation_group_size=1 (full per-query "
            "isolation); G>1 needs a group-gather not implemented here — use mem_lookup's "
            "masked per_query_isolation path instead (mem_batched_isolation=False).")

    if cfg.get('mem_k_prenormed', False):
        mem_k_normed = w['mem_k']
    else:
        mem_k_normed = rms_norm(w['mem_k'], w['mem_k_norm'], cfg['rms_norm_eps'])
    mem_v = w['mem_v']
    activation, temp, phantom = _mem_score_opts(cfg)

    B, T = q.shape[0], q.shape[1]
    M, H = mem_k_normed.shape
    Dv = mem_v.shape[-1]
    if M % B != 0:
        raise ValueError(f"mem_lookup_batched: bank size {M} not divisible by batch {B}")
    m_per_query = M // B

    mem_k_b = mem_k_normed.reshape(B, m_per_query, H)   # free reshape, see docstring
    mem_v_b = mem_v.reshape(B, m_per_query, Dv)
    mask_b = jnp.reshape(mem_mask, (B, m_per_query)) != 0 if mem_mask is not None else None
    global_offset = (jnp.arange(B, dtype=jnp.int32) * m_per_query)[:, None, None, None]  # [B,1,1,1]

    # Scale by a dtype-matched Python-float scalar, not `/ jnp.array(H, dtype=jnp.float32)` —
    # dividing by an fp32 array promotes the (bf16) logits to fp32 for the rest of the function,
    # silently doubling every downstream tensor's memory. Everything memory-layer-related is
    # meant to stay bf16 (weights/activations already are); this keeps `logits` — and hence
    # `mem_scores` below, the tensor this matters most for — in q.dtype throughout.
    inv_sqrt_h = jnp.asarray(1.0 / (float(H) ** 0.5), dtype=q.dtype)

    # Optionally also expose the full per-row score grid (pre-top-k) for aux losses that want
    # exact (not top-k-approximated) per-query statistics — e.g. doc_access_per_query_loss. Cheap
    # here specifically because it's per-ROW ([B,N,T,m_per_query]), not the old cross-batch
    # [B,N,T,B*m_per_query] grid `doc_access_loss` needs and this mode structurally never builds.
    # Mirrors the existing (unused before this) `mem_collect_full_scores` cfg key.
    want_full_scores = collect_aux and bool(cfg.get('mem_collect_full_scores', False))

    # Chunk over the query-token (T) axis, mirroring mem_lookup_gqa's mem_t_chunk: bounds the
    # [B,N,tc,m_per_query] score tensor when T is large. 0/unset -> single chunk (whole T).
    TC = int(cfg.get('mem_t_chunk', 0) or 0)
    chunk_bounds = [(0, T)] if TC <= 0 or TC >= T else [(t0, min(t0 + TC, T)) for t0 in range(0, T, TC)]

    sc_parts, val_parts, idx_parts, logit_parts, full_score_parts = [], [], [], [], []
    for t0, t1 in chunk_bounds:
        q_c = q[:, t0:t1]  # [B, tc, N, H]
        logits = jnp.einsum('btnh,bmh->bntm', q_c, mem_k_b, preferred_element_type=q.dtype,
                             out_sharding=P('data', 'model', None, None))
        logits = logits * inv_sqrt_h  # stays q.dtype (bf16), see comment above
        if mask_b is not None:
            logits = jnp.where(mask_b[:, None, None, :], logits, jnp.asarray(-1e9, dtype=logits.dtype))
        if want_full_scores:
            full_score_parts.append(logits)
        tl, ti_local = bank_top_k(logits, cfg['mem_top_k'], cfg)                      # [B,N,tc,K]
        ts = mem_weight_from_logits(tl, activation, temp, phantom)
        ti_global = ti_local + global_offset                                          # -> flat bank ids
        tv = mem_v.at[ti_global].get(out_sharding=P('data', 'model', None, None, None))  # [B,N,tc,K,Dv]
        sc_parts.append(ts); val_parts.append(tv)
        if collect_aux:
            idx_parts.append(ti_global)
            logit_parts.append(tl)  # RAW pre-activation logits — matches mem_lookup's convention

    top_k_scores = sc_parts[0] if len(sc_parts) == 1 else jnp.concatenate(sc_parts, axis=2)
    top_k_values = val_parts[0] if len(val_parts) == 1 else jnp.concatenate(val_parts, axis=2)
    aux_data = None
    if collect_aux:
        idx = idx_parts[0] if len(idx_parts) == 1 else jnp.concatenate(idx_parts, axis=2)
        tl_full = logit_parts[0] if len(logit_parts) == 1 else jnp.concatenate(logit_parts, axis=2)
        aux_data = {"mem_top_k_indices": idx, "mem_top_k_logits": tl_full}
        if want_full_scores:
            full = full_score_parts[0] if len(full_score_parts) == 1 else jnp.concatenate(full_score_parts, axis=2)
            # [B,N,T,m_per_query] -> [B,T,N,m_per_query], (tensor,) 1-tuple: matches mem_lookup's
            # own "mem_scores" convention exactly (see mem_lookup, models/memory.py), so
            # doc_access_per_query_loss's aggregation/unpacking is identical to every sibling loss.
            aux_data["mem_scores"] = (jnp.transpose(full, (0, 2, 1, 3)),)
    return top_k_scores, top_k_values, aux_data


def mem_lookup_gqa(q, w, cfg, collect_aux=False):
    """GQA memory retrieval over a PER-KV-HEAD bank — a faithful port of the MaxText
    `qwen3_mem.py::_memory_op` retrieval (mem_num_kv_heads banks, GQA query grouping).

    q          [B, T, Nq, H]              (already mem_q_proj'd + mem_q_norm'd)
    w['mem_k'] [M, Nkv, H]                per-kv-head bank keys
    w['mem_v'] [M, Nkv, Dv]              per-kv-head bank values
    Nq query heads are split into Nkv groups of G = Nq // Nkv; group kv attends ONLY to
    bank slice kv. Scoring matches the single-bank path exactly (scale 1/sqrt(H), then a
    plain softmax over the K retrieved slots), so activation/temp knobs don't apply here.

    Returns top_k_scores [B,Nq,T,K], top_k_values [B,Nq,T,K,Dv] (per-head gathered so the
    caller's dense read einsum('bntk,bntkv->btnv') is unchanged), and aux with
    mem_top_k_indices [B,Nq,T,K] (flat bank indices, shared M axis -> doc_access_acc works).
    """
    mem_mask = w.get('mem_mask', None)
    if mem_mask is not None and getattr(mem_mask, 'ndim', 1) != 1:
        raise NotImplementedError(
            "per-example [B, M] mem_mask is only supported by mem_lookup_chunked "
            "(set mem_lookup_chunk_size); this lookup variant assumes a shared [M] mask.")
    mem_k = w['mem_k']                                   # [M, Nkv, H]
    mem_v = w['mem_v']                                   # [M, Nkv, Dv]
    if cfg.get('mem_k_prenormed', False):
        k = mem_k
    else:
        k = rms_norm(mem_k, w['mem_k_norm'], cfg['rms_norm_eps'])   # normed over the last (H) axis
    B, T, Nq, H = q.shape
    M, Nkv = mem_k.shape[0], mem_k.shape[1]
    G = Nq // Nkv
    K = cfg['mem_top_k']
    Dv = mem_v.shape[-1]

    scale = (1.0 / jnp.sqrt(jnp.array(H, dtype=jnp.float32))).astype(q.dtype)
    valid = (jnp.reshape(mem_mask, (-1,)) != 0)[None, None, None, :] if mem_mask is not None else None  # [1,1,1,M]

    # Chunk over the query-token (T) axis so the [B,Nq,TC,M] score matrix is never materialized at
    # full T. At max_docs=1000 the corpus M ~ 250k, and a full-T [B,Nq,512,M] prefill matrix OOMs
    # HBM; TC-chunking bounds it (decode has T=1 -> a single chunk, no overhead).
    TC = min(int(cfg.get('mem_t_chunk', 32) or 32), T)
    n_chunks = (T + TC - 1) // TC
    sc_parts, val_parts, idx_parts = [], [], []
    for c in range(n_chunks):
        t0, t1 = c * TC, min(c * TC + TC, T)
        tc = t1 - t0
        qg = (q[:, t0:t1] * scale).reshape(B, tc, Nkv, G, H)
        # per-kv-head scores: [B,Nkv,G,tc,M] -> [B,Nq,tc,M]. bf16 score (half the HBM of fp32 — the
        # M~250k score dominates memory at max_docs=1000; the H-contraction still accumulates in
        # fp32 on the MXU). out_sharding pins batch on 'data' (bank replicated) so the gather/reshape
        # aren't sharding-ambiguous under the eval mesh.
        logits = jnp.einsum('btkgh,mkh->bkgtm', qg, k.astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16,
                            out_sharding=P('data', None, None, None, None))
        logits = logits.reshape(B, Nq, tc, M, out_sharding=P('data', None, None, None))
        if valid is not None:
            logits = jnp.where(valid, logits, jnp.asarray(-1e9, jnp.bfloat16))
        tl, ti = bank_top_k(logits, K, cfg)                               # [B,Nq,tc,K]
        ts = jax.nn.softmax(tl.astype(jnp.float32), axis=-1).astype(q.dtype)
        ti_g = ti.reshape(B, Nkv, G, tc, K, out_sharding=P('data', None, None, None, None))
        pv = [mem_v[:, kv, :].at[ti_g[:, kv]].get(out_sharding=P('data', None, None, None, None))
              for kv in range(Nkv)]                                        # each [B,G,tc,K,Dv]
        tv = jnp.stack(pv, axis=1).reshape(B, Nq, tc, K, Dv, out_sharding=P('data', None, None, None, None))
        sc_parts.append(ts); val_parts.append(tv)
        if collect_aux:
            idx_parts.append(ti)

    top_k_scores = sc_parts[0] if n_chunks == 1 else jnp.concatenate(sc_parts, axis=2)   # [B,Nq,T,K]
    top_k_values = val_parts[0] if n_chunks == 1 else jnp.concatenate(val_parts, axis=2)  # [B,Nq,T,K,Dv]
    aux_data = None
    if collect_aux:
        idx = idx_parts[0] if n_chunks == 1 else jnp.concatenate(idx_parts, axis=2)
        aux_data = {"mem_top_k_indices": idx}
    return top_k_scores, top_k_values, aux_data


def mem_lookup_chunked(q, w, cfg, collect_aux=False, keys_only=False, return_all_scores=None):
    # return_all_scores decoupled from collect_aux: the full [B,N,T,M] score matrix ("mem_scores")
    # is O(M) and only needed for the full-grid doc_access telemetry. Two-pass pass-1 wants only the
    # top-k INDICES, so it passes return_all_scores=False and stays O(K). Default None -> collect_aux
    # (preserves old behavior for callers that relied on mem_scores being present).
    if return_all_scores is None:
        return_all_scores = collect_aux
    mem_mask = w.get('mem_mask', None)
    lookup_chunk_size = cfg.get('mem_lookup_chunk_size', CHUNK_SIZE)
    if cfg.get('mem_k_prenormed', False):
        mem_k_normed = w['mem_k']
    else:
        mem_k_normed = rms_norm(w['mem_k'], w['mem_k_norm'], cfg['rms_norm_eps'])
    concrete_mesh = _get_concrete_mesh(w)
    shard_axis = cfg.get('mem_shard_axis', 'model')

    q_t = jnp.transpose(q, (0, 2, 1, 3))  # [B, N, T, H]
    top_k_scores, top_k_values, top_k_indices, all_scores = sharded_top_k_ip(
        q_t, mem_k_normed, w['mem_v'], cfg['mem_top_k'],
        mem_mask=mem_mask, chunk_size=lookup_chunk_size,
        concrete_mesh=concrete_mesh, shard_axis=shard_axis,
        return_all_scores=return_all_scores,
        keys_only=keys_only,
        temp=_mem_score_opts(cfg)[1],
        activation=_mem_score_opts(cfg)[0], phantom_log_n=_mem_score_opts(cfg)[2], cfg=cfg,
    )
    aux_data = None
    if collect_aux:
        aux_data = {
            "mem_top_k_indices": top_k_indices,
            "mem_top_k_logits": top_k_scores,
        }
        if all_scores is not None:
            # all_scores [B, N, T, M] → transpose to [B, T, N, M] to match full-matrix path
            aux_data["mem_scores"] = (jnp.transpose(all_scores, (0, 2, 1, 3)),)
    return top_k_scores, top_k_values, aux_data


def product_key_lookup(q, w, cfg, collect_aux=False):
    mem_mask = w.get('mem_mask', None)
    if mem_mask is not None and getattr(mem_mask, 'ndim', 1) != 1:
        raise NotImplementedError(
            "per-example [B, M] mem_mask is only supported by mem_lookup_chunked "
            "(set mem_lookup_chunk_size); this lookup variant assumes a shared [M] mask.")
    mem_k_normed = rms_norm(w['mem_k'], w['mem_k_norm'], cfg['rms_norm_eps'])
    top_k_logits, top_k_indices, mem_scores = get_indices_product_keys(q, mem_k_normed, cfg['mem_top_k'], mem_mask)
    top_k_scores = jax.nn.softmax(top_k_logits, axis=-1)
    top_k_values = w['mem_v'].at[top_k_indices].get(out_sharding=P('data', 'model', None, None, None))
    aux_data = None
    if collect_aux:
        aux_data = {"mem_scores": (mem_scores,), "mem_top_k_indices": top_k_indices, "mem_top_k_logits": top_k_logits}
    return top_k_scores, top_k_values, aux_data


def input_gate(x_norm, y, w, cfg):
    gate_flat = jnp.einsum('btd,vd->btv', x_norm, w['mem_gate_proj'], preferred_element_type=jnp.float32, out_sharding=P('data', None, 'model', None))
    gate = jax.nn.silu(gate_flat)  # [B, T, V]
    y_flat = y.reshape(y.shape[0], y.shape[1], -1)  # [B, T, N*V]
    gated_y = y_flat * gate  # [B, T, N*V]    
    o = jnp.einsum('btv,dv->btd', gated_y, w['mem_up_proj'], preferred_element_type=x_norm.dtype, out_sharding=P('data', None, None, None))
    return o


def _fused_topk_logits(q_t, top_k_indices, mem_k_normed, kchunk, gather_sharding):
    """logits_k [B,N,T,K] via a jax.remat chunk-scan over K: gathers KCHUNK keys per step and
    contracts with q, so the [B,N,T,K,H] key tensor never materializes (fwd or bwd). Lets the
    pass-2 logits stay O([B,N,T,K]) at large batch."""
    B, N, T, K = top_k_indices.shape
    D = q_t.shape[-1]
    nkc = K // kchunk
    idx_s = jnp.moveaxis(top_k_indices.reshape(B, N, T, nkc, kchunk), 3, 0)  # [nkc,B,N,T,kchunk]

    @jax.remat
    def step(_, ii):
        k_c = mem_k_normed.at[ii].get(out_sharding=gather_sharding)          # [B,N,T,kchunk,H]
        lg = jnp.einsum('bntd,bntcd->bntc', q_t, k_c, preferred_element_type=jnp.float32)
        return None, lg
    _, lg_s = jax.lax.scan(step, None, idx_s)                                # [nkc,B,N,T,kchunk]
    return jnp.moveaxis(lg_s, 0, 3).reshape(B, N, T, K) / jnp.sqrt(jnp.array(D, jnp.float32))


def _fused_value_read(top_k_indices, top_k_scores, mem_v, kchunk, gather_sharding):
    """Weighted value read y [B,N,T,V] via a jax.remat chunk-scan over K: gathers KCHUNK values
    per step and accumulates, so the [B,N,T,K,V] value tensor never materializes. Replaces the
    dense gather + einsum('bntk,bntkv->bntv') that OOMs at large batch (34G at B256)."""
    B, N, T, K = top_k_indices.shape
    V = mem_v.shape[-1]
    nkc = K // kchunk
    idx_s = jnp.moveaxis(top_k_indices.reshape(B, N, T, nkc, kchunk), 3, 0)  # [nkc,B,N,T,kchunk]
    w_s = jnp.moveaxis(top_k_scores.reshape(B, N, T, nkc, kchunk), 3, 0)     # [nkc,B,N,T,kchunk]

    @jax.remat
    def step(acc, iw):
        ii, ww = iw
        v_c = mem_v.at[ii].get(out_sharding=gather_sharding)                 # [B,N,T,kchunk,V]
        return acc + jnp.einsum('bntc,bntcv->bntv', ww.astype(v_c.dtype), v_c), None
    init = jnp.zeros((B, N, T, V), dtype=top_k_scores.dtype)
    y, _ = jax.lax.scan(step, init, (idx_s, w_s))
    return y                                                                 # [B,N,T,V]


def mem_lookup_two_pass(q, w, cfg, collect_aux=False, pos_slot_indices=None):
    """Two-pass memory lookup for gradient-efficient training with large memory banks.

    Pass 1 (no-grad): chunked scan over full M to get top-k indices.
    Pass 2 (with-grad): gather K vectors, compute [B, N, T, K] scores only.

    Gradients are exact: non-selected entries contribute zero to the loss.

    Args:
        pos_slot_indices: optional [B, P] int array of flat memory slot indices for
            all docs in the batch (positive and negative). When provided, pass-2 also
            computes logits for these slots and stores them in aux_data under
            "mem_pos_indices" / "mem_pos_logits". This guarantees the
            doc_access_top_k_loss always has a positive in its comparison pool even
            when pass-1 top-k misses all positive doc tokens.
    """
    # Pass 1: stop-gradient on q so XLA skips backward through the full score matrix.
    # mem_k from embed_forward may carry @data sharding from batch processing;
    # the chunked scan needs to reshape [M] → [n_chunks, chunk_size], which requires
    # mem_k to be fully replicated first.
    # keys_only=True skips mem_v entirely — values are discarded in Pass 1.
    q_sg = jax.lax.stop_gradient(q)
    if is_jax_mesh_active():
        mem_k_pass1 = jax.lax.stop_gradient(jax.sharding.reshard(w['mem_k'], P(None, None)))
    else:
        mem_k_pass1 = jax.lax.stop_gradient(w['mem_k'])
    w_pass1 = {**w, 'mem_k': mem_k_pass1}

    # collect_aux=True to get mem_top_k_indices out, but return_all_scores=False so pass-1 never
    # materializes the O(M) [B,N,T,M] mem_scores grid (two-pass only needs the indices). This is
    # what keeps the two-pass forward O(K) instead of O(M).
    _, _, pass1_aux = mem_lookup_chunked(q_sg, w_pass1, cfg, collect_aux=True, keys_only=True,
                                         return_all_scores=False)
    top_k_indices = jax.lax.stop_gradient(pass1_aux["mem_top_k_indices"])  # [B, N, T, K]

    # Pass 2: gather only K vectors, compute gradients over [B, N, T, K] scores.
    # Use .at[].get(out_sharding=...) so XLA knows how to shard the gather output,
    # matching the sharding convention used by mem_lookup / mem_lookup_chunked.

    # out_sharding hints are only valid inside a mesh context
    if is_jax_mesh_active():
        gather_sharding  = P('data', 'model', None, None, None)
        logits_sharding  = P('data', 'model', None, None)
    else:
        gather_sharding  = None
        logits_sharding  = None

    # Default to norm-then-gather: at production sizes B·N·T·K ≫ M, so rms_norm
    # backward on the full [M, H] bank is cheaper than on the gathered [B,N,T,K,H]
    # tensor. Both paths now flow gradients to w['mem_k']/w['mem_v']; the split is
    # purely a memory tradeoff. Override with cfg.sparse_grads=True only when K·B·N·T < M.
    sparse_grads = cfg.get('sparse_grads', False)
    # kchunk>0: gather+contract the K keys/values in remat chunk-scans so the [B,N,T,K,H]/[B,N,T,K,V]
    # tensors never materialize -> pass-2 stays O([B,N,T,·]) at large batch. Incompatible with
    # sparse_grads (which needs the dense mem_k_k/mem_v_k) and span_window>0.
    kchunk = int(cfg.get('mem_value_read_kchunk', 0))
    if sparse_grads:
        kchunk = 0

    if sparse_grads:
        # Gather-then-norm (instead of norm-then-gather): rms_norm backward touches
        # only the K gathered rows → O(K·H) instead of O(M·H). Gradient still flows
        # through the gather into w['mem_k'] / w['mem_v'] as a sparse [M,H] scatter-add
        # during autodiff, which is what lets the embed model receive gradients.
        mem_k_k_raw = w['mem_k'].at[top_k_indices].get(out_sharding=gather_sharding)  # [B, N, T, K, H]
        if cfg.get('mem_k_prenormed', False):
            mem_k_k = mem_k_k_raw
        else:
            mem_k_k = rms_norm(mem_k_k_raw, w['mem_k_norm'], cfg['rms_norm_eps'])
        mem_v_k = w['mem_v'].at[top_k_indices].get(out_sharding=gather_sharding)      # [B, N, T, K, Dv]
        mem_k_normed = None  # not needed in sparse path (pos_slot gather handled below)
    else:
        if cfg.get('mem_k_prenormed', False):
            mem_k_normed = w['mem_k']
        else:
            mem_k_normed = rms_norm(w['mem_k'], w['mem_k_norm'], cfg['rms_norm_eps'])
        if kchunk > 0:
            mem_k_k = None  # keys gathered per-chunk in _fused_topk_logits (never [B,N,T,K,H])
            mem_v_k = None  # values gathered per-chunk in _fused_value_read
        else:
            mem_k_k = mem_k_normed.at[top_k_indices].get(out_sharding=gather_sharding)  # [B, N, T, K, H]
            mem_v_k = w['mem_v'].at[top_k_indices].get(out_sharding=gather_sharding)    # [B, N, T, K, Dv]

    q_t = jnp.transpose(q, (0, 2, 1, 3))  # [B, N, T, H]
    if kchunk > 0:
        logits_k = _fused_topk_logits(q_t, top_k_indices, mem_k_normed, kchunk, gather_sharding)
    else:
        logits_k = jnp.einsum('bntd,bntkd->bntk', q_t, mem_k_k,
                               preferred_element_type=q.dtype,
                               out_sharding=logits_sharding) / jnp.sqrt(
            jnp.array(q.shape[-1], dtype=jnp.float32))
    top_k_scores = jax.nn.softmax(logits_k, axis=-1).astype(q.dtype)
    # Fused value read (kchunk>0): y [B,N,T,V] via chunk-scan; returned in place of mem_v_k so the
    # caller (memory_layer) uses it directly instead of the dense einsum('bntk,bntkv->btnv').
    fused_y = _fused_value_read(top_k_indices, top_k_scores, w['mem_v'], kchunk,
                                gather_sharding) if kchunk > 0 else None

    # Compute positive-slot logits so doc_access_top_k_loss always has a positive
    # in its comparison pool, even when pass-1 top-k missed all positive doc tokens.
    #
    # pos_slot_indices is derived from jnp.arange(B) so it is fully-replicated even
    # though q_t is data-sharded.  We must constrain it to P('data', None) before
    # the gather so that mem_k_pos and pos_logits share q_t's batch sharding.
    # We also pre-expand pos_slot_indices to [B, N, T, P] (matching top_k_indices
    # layout) so the loss can concatenate without any extra sharding work.
    pos_logits = None
    pos_indices_expanded = None
    if collect_aux and pos_slot_indices is not None:
        _, N_k, T_k, _ = logits_k.shape
        P_count = pos_slot_indices.shape[-1]

        if is_jax_mesh_active():
            # pos_slot_indices is derived from static ranges so it is fully-replicated.
            # reshard (not with_sharding_constraint) is required for Explicit mesh axes.
            pos_slot_indices = jax.sharding.reshard(pos_slot_indices, P('data', None))
            pos_gather_sharding = P('data', None, None)
            pos_logits_sharding = logits_sharding  # P('data', 'model', None, None)
        else:
            pos_gather_sharding = None
            pos_logits_sharding = None

        if sparse_grads:
            mem_k_pos_raw = w['mem_k'].at[pos_slot_indices].get(out_sharding=pos_gather_sharding)
            mem_k_pos = (mem_k_pos_raw if cfg.get('mem_k_prenormed', False)
                         else rms_norm(mem_k_pos_raw, w['mem_k_norm'], cfg['rms_norm_eps']))
        else:
            mem_k_pos = mem_k_normed.at[pos_slot_indices].get(
                out_sharding=pos_gather_sharding)  # [B, P, H]
        pos_logits = jnp.einsum(
            'bntd,bpd->bntp', q_t, mem_k_pos,
            preferred_element_type=q.dtype,
            out_sharding=pos_logits_sharding,
        ) / jnp.sqrt(jnp.array(q.shape[-1], dtype=jnp.float32))  # [B, N, T, P]

        # Expand flat [B, P] indices to [B, N, T, P] with the same sharding as
        # top_k_indices so the concatenation in the loss is sharding-compatible.
        pos_indices_expanded = jnp.broadcast_to(
            pos_slot_indices[:, None, None, :],
            (pos_slot_indices.shape[0], N_k, T_k, P_count),
        )
        if is_jax_mesh_active():
            pos_indices_expanded = jax.sharding.reshard(
                pos_indices_expanded, logits_sharding)  # P('data', 'model', None, None)

    aux_data = None
    if collect_aux:
        aux_data = {"mem_top_k_indices": top_k_indices, "mem_top_k_logits": logits_k}
        if sparse_grads:
            # K-sized tensors for caller to compute sparse gradients:
            #   jax.grad(inner)(mem_k_k, mem_v_k) → [B,N,T,K,H], [B,N,T,K,Dv]
            # then: mem_k = mem_k.at[top_k_indices].add(-lr * grad_mem_k_k)
            aux_data["mem_k_k"] = mem_k_k
            aux_data["mem_v_k"] = mem_v_k
        if pos_logits is not None:
            aux_data["mem_pos_indices"] = pos_indices_expanded  # [B, N, T, P]
            aux_data["mem_pos_logits"] = pos_logits              # [B, N, T, P]
        # Expose full-M scores from pass 1 under "mem_scores" so doc_access_loss and
        # mem_uniform_kl can find them.  These are stop-gradient (pass 1 uses sg on
        # both q and mem_k/v), so they provide correct loss *values* for logging but
        # carry no gradient — an inherent tradeoff of two-pass top-k.
        if pass1_aux is not None and "dmem_scores" in pass1_aux:
            aux_data["mem_scores"] = pass1_aux["dmem_scores"]
        if kchunk > 0:
            aux_data["mem_fused_read"] = True   # signals memory_layer to skip the dense read einsum
    # kchunk>0: return the already-reduced read y [B,N,T,V] in the values slot; memory_layer
    # transposes it to [B,T,N,V] and skips einsum('bntk,bntkv->btnv').
    return top_k_scores, (fused_y if kchunk > 0 else mem_v_k), aux_data


def span_readout(w, cfg, top_k_indices, span_w):
    """Stage 3: replace each retrieved value with a boundary-respecting mean over the
    t±span_w window. Keys are position-aligned with values, so a match on a context/cue
    token ("born") still drags in the adjacent answer token ("1863"). Respects document
    boundaries (a window may not span across docs) via mem_mask + effective_doc_len.

    top_k_indices: [B,N,T,K] slot ids into the mem_v bank. Returns (pooled_values, straddle_frac).
    """
    mem_v = w['mem_v']
    M = mem_v.shape[0]
    offsets = jnp.arange(-span_w, span_w + 1)              # [W]
    nbr = top_k_indices[..., None] + offsets               # [B,N,T,K,W]
    nbr_clamped = jnp.clip(nbr, 0, M - 1)
    win_v = mem_v.at[nbr_clamped].get(out_sharding=P('data', 'model', None, None, None, None))  # [B,N,T,K,W,Dv]

    # valid neighbor = in range, a real doc token (mem_mask), and same document as center
    valid = (nbr >= 0) & (nbr < M)
    mem_mask = w.get('mem_mask', None)
    if mem_mask is not None and getattr(mem_mask, 'ndim', 1) != 1:
        raise NotImplementedError(
            "per-example [B, M] mem_mask is only supported by mem_lookup_chunked "
            "(set mem_lookup_chunk_size); span_readout assumes a shared [M] mask.")
    if mem_mask is not None:
        tok_valid = mem_mask.at[nbr_clamped].get(out_sharding=P('data', 'model', None, None, None)).astype(bool)
        valid = valid & tok_valid
    doc_len = cfg.get('effective_doc_len', None)
    if doc_len:
        same_doc = (nbr_clamped // doc_len) == (top_k_indices[..., None] // doc_len)
        valid = valid & same_doc

    vf = valid.astype(win_v.dtype)[..., None]              # [B,N,T,K,W,1]
    pooled = (win_v * vf).sum(axis=-2) / (vf.sum(axis=-2) + 1e-6)   # [B,N,T,K,Dv]
    straddle = 1.0 - valid.astype(jnp.float32).mean()      # share of window slots that fell out-of-doc
    return pooled, straddle


def _memory_telemetry(q, top_k_scores, o, x, w):
    """Weight-0 read-channel telemetry from the actual mixture weights (path-agnostic;
    top_k_scores is post-softmax in every lookup path). All reductions are over the last
    axis, the head axis (via a sum that all-reduces), or the whole tensor, so they don't
    materialize a [N,N] over the sharded head axis."""
    f32 = jnp.float32
    p = top_k_scores.astype(f32)                           # [B,N,T,K] softmax weights over K slots
    p = p / (p.sum(axis=-1, keepdims=True) + 1e-9)
    entropy = -(p * jnp.log(p + 1e-9)).sum(axis=-1)        # [B,N,T]
    eff_slots = 1.0 / (jnp.square(p).sum(axis=-1) + 1e-9)  # participation ratio ~ how many slots contribute
    top1 = p.max(axis=-1)
    o32 = o.astype(f32)
    o_norm = jnp.mean(jnp.linalg.norm(o32, axis=-1))
    x_norm = jnp.mean(jnp.linalg.norm(x.astype(f32), axis=-1))

    # Cross-head query cosine: heads here specialize only their query direction (K/V shared).
    # Mean pairwise cosine over head pairs = (‖Σ_n q̂_n‖² − N)/(N(N−1)), a reduction over the
    # head axis — avoids the risky einsum 'btnh,btmh->nm' that contracts the sharded head axis.
    qn = q.astype(f32)
    qn = qn / (jnp.linalg.norm(qn, axis=-1, keepdims=True) + 1e-9)   # [B,T,N,H]
    N = qn.shape[2]
    s = qn.sum(axis=2)                                     # [B,T,H] (sums over the sharded head axis)
    head_cos = jnp.mean((jnp.square(s).sum(axis=-1) - N) / (N * (N - 1) + 1e-9))

    return {
        "mem_write_norm": o_norm,                          # ‖o‖ into the residual stream (zero-init growth)
        "mem_write_ratio": o_norm / (x_norm + 1e-6),       # write relative to stream norm (dilution)
        "mem_topk_entropy": jnp.mean(entropy),             # 128-way averaging blur
        "mem_effective_slots": jnp.mean(eff_slots),        # exp-entropy-like count of contributing slots
        "mem_top1_weight": jnp.mean(top1),                 # sharpness of the mixture
        "mem_o_proj_norm": jnp.linalg.norm(w['mem_o_proj'].astype(f32)),  # dead-layer check
        "mem_head_query_cos": head_cos,                    # head redundancy (→1 => heads collapse)
        "mem_o_meanvec": o32.mean(axis=(0, 1)),            # per-layer write direction (cross-layer cos, registry side)
    }


def memory_layer(cfg, x, w, collect_aux=False, pos_slot_indices=None):

    # MEM_TOP_K env override (inference-time knob; avoids fragile Hydra model.memory.* overrides)
    _tk = os.environ.get("MEM_TOP_K")
    if _tk:
        cfg = {**cfg, "mem_top_k": int(_tk)}

    if cfg.get("mem_placement", "replace_mlp") == "after_attention":
        x_norm = rms_norm(x, w['mem_layernorm'], cfg['rms_norm_eps'])
    else:
        x_norm = rms_norm(x, w['post_attention_layernorm'], cfg['rms_norm_eps'])

    # query proj
    q = jnp.einsum('btd,nhd->btnh', x_norm, w['mem_q_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, 'model', None))
    q = rms_norm(q, w['mem_q_norm'], cfg['rms_norm_eps'])

    # Stage 3 span readout needs the retrieved indices even when aux isn't otherwise
    # collected (it's an architecture change, live at train and eval time).
    span_w = int(cfg.get('span_window', 0) or 0)
    need_aux = collect_aux or span_w > 0

    aux_data = None
    if w.get('mem_k') is not None and getattr(w['mem_k'], 'ndim', 0) == 3:
        # GQA per-kv-head bank (mem_k [M,Nkv,H]) — ported from MaxText. Takes priority over the
        # single-bank paths; span/two-pass/product-key/chunk knobs don't apply to this path.
        top_k_scores, top_k_values, aux_data = mem_lookup_gqa(q, w, cfg, need_aux)
    elif cfg.get('mem_batched_isolation', False):
        # True per-row retrieval (no cross-batch join at all — see mem_lookup_batched
        # docstring). Requires per_query_isolation + isolation_group_size=1; default off so
        # every existing run (which relies on the masked full-matrix per_query_isolation, or no
        # isolation at all) is byte-for-byte unchanged.
        top_k_scores, top_k_values, aux_data = mem_lookup_batched(q, w, cfg, need_aux)
    elif cfg.get('two_pass_topk', False):
        # two-pass lookup: pass 1 no-grad for indices, pass 2 sparse grad
        top_k_scores, top_k_values, aux_data = mem_lookup_two_pass(q, w, cfg, need_aux, pos_slot_indices)
    elif cfg.get('mem_use_product_keys', False):
        # product key lookup
        top_k_scores, top_k_values, aux_data = product_key_lookup(q, w, cfg, need_aux)
    else:
        if cfg.get('mem_lookup_chunk_size', None) is not None:
            # chunked lookup
            top_k_scores, top_k_values, aux_data = mem_lookup_chunked(q, w, cfg, need_aux)
        else:
            # full lookup
            top_k_scores, top_k_values, aux_data = mem_lookup(q, w, cfg, need_aux)

    # Stage 3: neighbor-window value gather (default span_window=0 => no-op, so control /
    # Stage 1 / Stage 2 are byte-identical to before).
    # NOTE: incompatible with two_pass_topk sparse-grad mode — span pools densely from the
    # bank while the sparse path grads against the un-pooled mem_v_k. Grounding runs use
    # two_pass_topk=false, so this doesn't trigger; don't combine span_window>0 with two-pass.
    if span_w > 0 and aux_data is not None and "mem_top_k_indices" in aux_data:
        top_k_values, straddle_frac = span_readout(w, cfg, aux_data["mem_top_k_indices"], span_w)
        if collect_aux:
            aux_data = {**aux_data, "mem_boundary_straddle": straddle_frac}

    fused_read = (cfg.get('two_pass_topk', False)
                  and int(cfg.get('mem_value_read_kchunk', 0)) > 0
                  and span_w == 0)
    if fused_read:
        # two_pass returned the already-reduced read y [B,N,T,V]; just reorder to [B,T,N,V].
        y = jnp.transpose(top_k_values, (0, 2, 1, 3))
    else:
        y = jnp.einsum('bntk,bntkv->btnv', top_k_scores, top_k_values, preferred_element_type=top_k_scores.dtype, out_sharding=P('data', None, 'model', None))

    if cfg.get('mem_use_gating', False) and 'mem_gate_proj' in w:
        # input dependent gating
        o = input_gate(x_norm, y, w, cfg)
    else:
        o = jnp.einsum('btnv,dnv->btd', y, w['mem_o_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, None, None))

    # DLA ablation hook (eval-time): zero this layer's memory write so the change in the
    # answer-token logit measures the write's relevance. MEM_ABLATE_LAYER=<layer idx> or "all".
    _abl = os.environ.get("MEM_ABLATE_LAYER")
    if _abl is not None and (_abl == "all" or _abl == str(cfg.get('_mem_layer_idx'))):
        o = o * 0.0

    if collect_aux:
        aux_data = {} if aux_data is None else dict(aux_data)
        aux_data.update(_memory_telemetry(q, top_k_scores, o, x, w))
        # per-slot softmax weights for the positive-slot-mass diagnostic (registry side)
        aux_data["mem_top_k_probs"] = top_k_scores

    x += o.astype(x.dtype)

    return x, aux_data