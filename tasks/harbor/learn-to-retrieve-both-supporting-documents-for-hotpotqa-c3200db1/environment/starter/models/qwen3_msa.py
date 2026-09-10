'''
Memory Sparse Attention (MSA) — JAX implementation for inference/eval.

Paper: "MSA: Memory Sparse Attention for Efficient End-to-End Memory Model
Scaling to 100M Tokens" (arXiv:2603.23516). Reference torch impl:
github.com/EverMind-AI/MSA. This is a faithful-as-practical port of the MSA
*inference* path (NO generative-retrieval / memory-interleave) adapted to this
repo's static-memory eval harness.

Architecture (MSA-4B = Qwen3-4B-Instruct-2507 backbone):
  - 36 layers, hidden 2560, 32 q-heads / 8 kv-heads, head_dim 128, rope_theta 5e6.
  - Router projectors (router_q_proj [32,128,2560], router_k_proj [8,128,2560]) on
    the latter-half layers 18..35 only. Scalar `temperature` (unused at inference
    for INFONCE_DECOUPLE).

Two phases (see claude/msa-implementation-plan.md "EXACT ALGORITHM SPEC"):
  1. encode_docs: run docs through the backbone (each doc independent, doc-local
     RoPE), capture per-router-layer pooled K̄, V̄ (post-RoPE/qk-norm) and router
     K̄ᴿ (no RoPE). Mean-pool over `pooling_kernel_size` (64) token chunks.
  2. query: layers 0-17 attend locally; layers 18-35 route top-k docs (cosine-ish
     dot, head-mean, query-max, chunk-amax), concat selected pooled K̄,V̄ before the
     query's local KV, attend.
'''
import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.sharding import PartitionSpec as P

from .qwen3 import (
    Model, load as qwen3_load, rms_norm, apply_rope, mlp, ModelOutput,
    forward_layer, create_mask, get_sharding_safe,
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _msa_cfg_from_hf(hf_cfg, yaml_msa=None):
    """Build the runtime MSA hyperparam dict.

    Two sources, in priority order:
      1. HF config.json `msa_config` block  (EverMind-AI/MSA-4B — pretrained routers).
      2. The model YAML `msa:` block        (stock Qwen/Qwen3-4B — fresh routers; no
         `msa_config` exists, so router_layers default to the latter half of layers).
    """
    yaml_msa = yaml_msa or {}
    m = hf_cfg.get('msa_config', None)
    L = hf_cfg['num_hidden_layers']
    if m:
        router = m.get('router_layer_idx', '')
        if router == 'all':
            router_layers = list(range(L))
        else:
            router_layers = [int(i) for i in str(router).split(',') if i != '']
        return {
            'router_layers': router_layers,
            'pooling_kernel_size': int(m.get('pooling_kernel_size', 64)),
            'top_k_docs': int(m.get('top_k_docs', 16)),
            'head_reduce_method': m.get('head_reduce_method', 'mean'),
            'query_reduce_method': m.get('query_reduce_method', 'max'),
            'chunk_reduce_method': m.get('chunk_reduce_method', 'max'),
            'decouple_router': bool(m.get('decouple_router', True)),
            'aux_loss_method': m.get('aux_loss_method', 'INFONCE_DECOUPLE'),
            'infonce_loss_temp': float(m.get('infonce_loss_temp', 0.1)),
            'router_init': 'pretrained',
            'force_pos_docs': bool(yaml_msa.get('force_pos_docs', True)),
        }
    # Stock Qwen3-4B path: derive everything from the YAML block.
    rl = yaml_msa.get('router_layers', 'half')
    if rl is None or rl == 'half':
        router_layers = list(range(L // 2, L))      # latter half (paper §3.2.1)
    else:
        router_layers = [int(i) for i in rl]
    return {
        'router_layers': router_layers,
        'pooling_kernel_size': int(yaml_msa.get('pooling_kernel_size', 64)),
        'top_k_docs': int(yaml_msa.get('top_k_docs', 16)),
        'head_reduce_method': 'mean',
        'query_reduce_method': 'max',
        'chunk_reduce_method': 'max',
        'decouple_router': True,
        'aux_loss_method': 'INFONCE_DECOUPLE',
        'infonce_loss_temp': float(yaml_msa.get('infonce_loss_temp', 0.1)),
        'router_init': str(yaml_msa.get('router_init', 'copy')),
        'force_pos_docs': bool(yaml_msa.get('force_pos_docs', True)),
    }


def _init_routers(weights, cfg, tp_devices, mode='copy', seed=0):
    """Add `layers.{L}.router_{q,k}_proj` for router layers when absent (stock Qwen3-4B).

    router_q_proj: [N, H, D] (mirrors q_proj),  router_k_proj: [K, H, D] (mirrors k_proj).
      - mode='copy':   warm-start from the layer's own q_proj / k_proj weights (same space).
      - mode='normal': small random normal, std = 1/sqrt(D).
    No-op for keys already present (MSA-4B path).
    """
    N, K = cfg['num_attention_heads'], cfg['num_key_value_heads']
    H, D = cfg['head_dim'], cfg['hidden_size']
    rng = jax.random.PRNGKey(seed)
    for L in cfg['msa']['router_layers']:
        qk, kk = f'layers.{L}.router_q_proj', f'layers.{L}.router_k_proj'
        if qk in weights and kk in weights:
            continue
        if mode == 'copy':
            weights[qk] = weights[f'layers.{L}.q_proj']    # [N,H,D]
            weights[kk] = weights[f'layers.{L}.k_proj']    # [K,H,D]
        else:
            rng, s1, s2 = jax.random.split(rng, 3)
            std = 1.0 / (D ** 0.5)
            wq = jax.random.normal(s1, (N * H, D), jnp.bfloat16) * std
            wk = jax.random.normal(s2, (K * H, D), jnp.bfloat16) * std
            weights[qk] = jax.device_put(wq, get_sharding_safe(qk, (N * H, D), tp_devices)).reshape(N, H, D)
            weights[kk] = jax.device_put(wk, get_sharding_safe(kk, (K * H, D), tp_devices)).reshape(K, H, D)
    return weights


def load(cfg, tp_devices=1, hf_ckpt_dir='~/weights/huggingface'):
    """Load the MSA backbone (MSA-4B *or* stock Qwen3-4B) and ensure routers exist.

    - MSA-4B: routers ship in the checkpoint; qwen3_load reshaped `router_{q,k}_proj.0`
      by head — we just drop the trailing `.0`.
    - Qwen3-4B: no routers in the checkpoint → initialize fresh ones (copy/normal).
    """
    model_id = cfg.main_model.model_id
    load_weights = cfg.main_model.get('load_weights', True)
    model = qwen3_load(model_id, tp_devices=tp_devices, load_weights=load_weights,
                       hf_ckpt_dir=hf_ckpt_dir, mask_type='causal')

    new_w = {}
    for k, v in model.weights.items():
        nk = k.replace('router_q_proj.0', 'router_q_proj').replace('router_k_proj.0', 'router_k_proj')
        new_w[nk] = v
    model.weights = new_w

    # Attach MSA config (HF msa_config if present, else the YAML msa block).
    yaml_msa = cfg.get('msa', {}) if hasattr(cfg, 'get') else {}
    msa = _msa_cfg_from_hf(model.cfg, yaml_msa)
    model.cfg['msa'] = msa
    model.cfg['router_layers_set'] = set(msa['router_layers'])

    # Initialize fresh routers when the checkpoint had none (stock Qwen3-4B).
    model.weights = _init_routers(model.weights, model.cfg, tp_devices,
                                  mode=msa['router_init'])

    # rebind forward: dispatches plain backbone vs. training forward (see forward()).
    model.forward = partial(forward, model.cfg)
    return model


def init(cfg, tp_devices):
    return load(cfg, tp_devices=tp_devices)


# ---------------------------------------------------------------------------
# Plain forward (no memory) — sanity check the backbone reproduces Qwen3.
# Router layers behave as ordinary Qwen3 layers here (router projs unused).
# ---------------------------------------------------------------------------
def forward(cfg, x, weights, pad_mask=None, kv=None, pos=0, return_hidden=False,
            collect_aux=False, **kw):
    # Training/eval path: a QA batch carries {"batch","docs"} + dict pad_mask.
    if isinstance(x, dict) and "docs" in x:
        return train_forward(cfg, x, weights, pad_mask=pad_mask, collect_aux=collect_aux)
    from .qwen3 import forward as qwen3_forward
    # router_q/k_proj weights are simply ignored by the base forward_layer.
    return qwen3_forward(cfg, x, weights, pad_mask=pad_mask, kv=kv, pos=pos,
                         return_hidden=return_hidden)


# ---------------------------------------------------------------------------
# Helpers shared by encode + query
# ---------------------------------------------------------------------------
def apply_rope_pos(x, theta, positions):
    """RoPE with explicit positions. x:[B,T,N,H], positions:[B,T] (int/float)."""
    H = x.shape[-1]
    freq = 1.0 / (theta ** (jnp.arange(0, H, 2, dtype=jnp.float32) / H))
    inp = positions[:, :, None].astype(jnp.float32) * freq[None, None, :]   # [B,T,H/2]
    sin, cos = jnp.sin(inp).astype(x.dtype), jnp.cos(inp).astype(x.dtype)
    sin, cos = sin[:, :, None, :], cos[:, :, None, :]                       # [B,T,1,H/2]
    x1, x2 = x[:, :, :, :H // 2], x[:, :, :, H // 2:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def _qkv(cfg, x_norm, w):
    """Standard Qwen3 q/k/v projections + q/k norm. Returns [B,T,N,H]/[B,T,K,H]."""
    q = jnp.einsum('btd,nhd->btnh', x_norm, w['q_proj'], preferred_element_type=x_norm.dtype, out_sharding=P('data', None, None, None))
    k = jnp.einsum('btd,khd->btkh', x_norm, w['k_proj'], preferred_element_type=x_norm.dtype, out_sharding=P('data', None, None, None))
    v = jnp.einsum('btd,khd->btkh', x_norm, w['v_proj'], preferred_element_type=x_norm.dtype, out_sharding=P('data', None, None, None))
    q = rms_norm(q, w['q_norm'], cfg['rms_norm_eps'])
    k = rms_norm(k, w['k_norm'], cfg['rms_norm_eps'])
    return q, k, v


def _embed(weights, ids):
    """Embedding lookup with explicit out_sharding (embed table is sharded)."""
    return weights['embed_tokens'].at[ids, :].get(out_sharding=P('data', None, None)).astype(jnp.bfloat16)


def _layer_weights(weights, i):
    prefix = f'layers.{i}.'
    return {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}


def _masked_pool_chunks(x, mask, kernel):
    """Mean-pool [B,T,H,D] over `kernel`-token chunks, averaging ONLY valid tokens
    (matches reference: pooled = sum(states)/num_valid_tokens_in_chunk).

    x: [B,T,H,D]  mask: [B,T] bool. Returns ([B,T//kernel,H,D], chunk_valid[B,T//k]).
    """
    B, T = mask.shape
    nch = T // kernel
    xr = x[:, :nch * kernel].reshape(B, nch, kernel, *x.shape[2:])      # [B,nch,k,H,D]
    mr = mask[:, :nch * kernel].reshape(B, nch, kernel).astype(jnp.float32)  # [B,nch,k]
    w = mr[:, :, :, None, None]                                          # [B,nch,k,1,1]
    s = (xr.astype(jnp.float32) * w).sum(axis=2)                         # [B,nch,H,D]
    cnt = mr.sum(axis=2)                                                 # [B,nch]
    pooled = (s / jnp.clip(cnt, 1.0)[:, :, None, None]).astype(x.dtype)
    return pooled, (cnt > 0)


# ---------------------------------------------------------------------------
# Phase 1 — encode documents into per-router-layer pooled banks
# ---------------------------------------------------------------------------
def encode_docs(cfg, doc_ids, pad_mask, weights):
    """Encode a batch of document chunks into pooled per-router-layer banks.

    Args:
      doc_ids:  [B, T] int token ids (each row = one fixed-length doc chunk).
      pad_mask: [B, T] bool, True for real tokens.
      weights:  model weights dict (layers.* + embed_tokens + norm).

    Returns dict:
      'kbar':  {layer: [B, T//P, 8, 128]}  pooled K (rope'd, doc-local positions)
      'vbar':  {layer: [B, T//P, 8, 128]}  pooled V
      'krbar': {layer: [B, T//P, 8, 128]}  pooled router K (no rope)
      'chunk_valid': [B, T//P] bool  (chunk has >=1 valid token)
    Each row's chunks are doc-local; caller tracks row -> source doc id.
    """
    kernel = cfg['msa']['pooling_kernel_size']
    rope_theta = cfg['rope_theta']
    eps = cfg['rms_norm_eps']
    router_set = cfg['router_layers_set']

    B, T = doc_ids.shape
    # doc-local positions per row: 0..T-1 (pos arg=0 -> apply_rope uses arange(T))
    # causal mask within each doc + pad
    rows = jnp.arange(T)[:, None]
    cols = jnp.arange(T)[None, :]
    causal = (cols <= rows)[None]                      # [1,T,T]
    attn_mask = (causal & pad_mask[:, None, :])[:, None]  # [B,1,T,T]

    x = _embed(weights, doc_ids)

    kbar, vbar, krbar = {}, {}, {}
    for i in range(cfg['num_hidden_layers']):
        w = _layer_weights(weights, i)
        x_norm = rms_norm(x, w['input_layernorm'], eps)
        q, k, v = _qkv(cfg, x_norm, w)             # [B,T,N,H], [B,T,K,H], [B,T,K,H]
        q = apply_rope(q, rope_theta, 0)
        k = apply_rope(k, rope_theta, 0)
        attn = jax.nn.dot_product_attention(q, k, v, mask=attn_mask)  # GQA [B,T,N,H]
        o = jnp.einsum('btnh,dnh->btd', attn, w['o_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, None))
        x = x + o

        if i in router_set:
            # standard K,V pooled (rope'd k, raw v) — 8 kv heads, valid-token mean
            kbar[i], _ = _masked_pool_chunks(k, pad_mask, kernel)
            vbar[i], _ = _masked_pool_chunks(v, pad_mask, kernel)
            # router K: router_k_proj on the *input-normed* hidden, no rope, no norm
            rk = jnp.einsum('btd,khd->btkh', x_norm, w['router_k_proj'],
                            preferred_element_type=x_norm.dtype, out_sharding=P('data', None, None, None))   # [B,T,8,128]
            krbar[i], _ = _masked_pool_chunks(rk, pad_mask, kernel)

        x = mlp(cfg, x, w)

    nch = T // kernel
    chunk_valid = (pad_mask[:, :nch * kernel].reshape(B, nch, kernel).sum(axis=2) > 0)
    return {'kbar': kbar, 'vbar': vbar, 'krbar': krbar, 'chunk_valid': chunk_valid}


# ---------------------------------------------------------------------------
# Routing: score a query's router-Q against the pooled router-K bank.
# ---------------------------------------------------------------------------
def route_topk(cfg, rq, bank_krbar, bank_chunk_doc, bank_chunk_valid, q_mask,
               num_docs, top_k):
    """Per-router-layer top-k document selection.

    Args:
      rq:               [B, Sq, 32, 128] router query (no rope, no norm).
      bank_krbar:       [C, 8, 128] pooled router keys for the whole corpus.
      bank_chunk_doc:   [C] int, source doc id per chunk.
      bank_chunk_valid: [C] bool.
      q_mask:           [B, Sq] bool, True for real query tokens.
      num_docs:         int, total number of distinct docs (doc ids in [0,num_docs)).
      top_k:            int.

    Returns sel_doc_ids [B, top_k] int (doc ids; -1 if fewer docs than top_k).
    """
    head_reduce = cfg['msa']['head_reduce_method']
    top_k = min(top_k, num_docs)                  # can't select more docs than exist
    B, Sq = rq.shape[0], rq.shape[1]
    C = bank_krbar.shape[0]
    G = cfg['num_attention_heads'] // cfg['num_key_value_heads']   # groups (4)
    Kh = cfg['num_key_value_heads']                                # 8

    # rq [B,Sq,32,128] -> [B,Sq,Kh,G,128]
    rqg = rq.reshape(B, Sq, Kh, G, rq.shape[-1])
    # COSINE similarity (paper Eq 2 / official EverMind-AI/MSA: F.normalize(q) · F.normalize(k),
    # scale 1.0 for decouple_router + INFONCE_DECOUPLE). L2-normalize router-Q and router-K over
    # the head dim before the dot — NOT a raw dot product (which biases toward high-norm chunks).
    rqg = rqg.astype(jnp.float32)
    rqg = rqg * jax.lax.rsqrt(jnp.maximum(jnp.sum(rqg * rqg, axis=-1, keepdims=True), 1e-6))
    bank_krbar = bank_krbar.astype(jnp.float32)
    bank_krbar = bank_krbar * jax.lax.rsqrt(jnp.maximum(jnp.sum(bank_krbar * bank_krbar, axis=-1, keepdims=True), 1e-6))
    # scores[b,sq,kh,g,c] = sum_d rqg[b,sq,kh,g,d] * krbar[c,kh,d]
    scores = jnp.einsum('bskgd,ckd->bskgc', rqg, bank_krbar,
                        preferred_element_type=jnp.float32, out_sharding=P('data', None, None, None, None))         # [B,Sq,Kh,G,C]
    scores = scores.reshape(B, Sq, Kh * G, C)                       # [B,Sq,32,C]

    neg = jnp.finfo(jnp.float32).min
    valid = q_mask[:, :, None, None] & bank_chunk_valid[None, None, None, :]
    scores = jnp.where(valid, scores, neg)

    if head_reduce == 'mean':
        scores = scores.mean(axis=2)        # [B,Sq,C]
    else:
        scores = scores.max(axis=2)
    # query reduce = max over Sq
    scores = scores.max(axis=1)             # [B,C]

    # chunk -> doc reduce = amax. scatter via segment_max over doc ids.
    # doc_scores[b, d] = max over chunks c with bank_chunk_doc[c]==d
    onehot_neg = jnp.zeros((B, num_docs), dtype=jnp.float32, out_sharding=P('data', None)) + neg
    # use jnp at[].max scatter
    doc_idx = jnp.where(bank_chunk_valid, bank_chunk_doc, 0)        # [C]
    doc_scores = onehot_neg.at[:, doc_idx].max(scores)   # [B, num_docs]

    # top_k doc ids
    _, sel = jax.lax.top_k(doc_scores, top_k)                       # [B, top_k]
    # mark invalid (-inf score) selections as -1
    sel_score = jnp.take_along_axis(doc_scores, sel, axis=1)
    sel = jnp.where(sel_score > -1e30, sel, -1)
    return sel


# ---------------------------------------------------------------------------
# Phase 2 — query forward with per-layer routing + concat-KV sparse attention.
#
# Implemented as a prefill that (a) selects top-k docs per router layer, and
# (b) gathers their pooled K̄/V̄ into a fixed-size doc-context that is concatenated
# ahead of the query's local KV for attention. Returns logits + a kv cache that
# carries the (constant) doc-context so decode can reuse it.
# ---------------------------------------------------------------------------
def _gather_ctx(bank_xbar, sel_chunk_idx, sel_chunk_valid):
    """Gather [B, max_ctx, 8, 128] from bank [C,8,128] given chunk indices.

    sel_chunk_idx:   [B, max_ctx] int (clamped >=0; invalid masked separately).
    sel_chunk_valid: [B, max_ctx] bool.
    """
    g = bank_xbar.at[sel_chunk_idx].get(out_sharding=P('data', None, None, None))   # [B, max_ctx, 8, 128]
    g = jnp.where(sel_chunk_valid[:, :, None, None], g, 0.0)
    return g


def select_doc_context(cfg, query_hidden_by_layer, banks, doc_chunk_table,
                       doc_chunk_table_valid, bank_chunk_doc, bank_chunk_valid,
                       q_mask, weights, num_docs):
    """For each router layer, route top-k docs and gather their pooled K̄/V̄.

    Args:
      query_hidden_by_layer: dict {layer: input-normed hidden [B,Sq,D]} for router
                             layers (the hidden states feeding router_q_proj).
      banks:  {'kbar','vbar','krbar': {layer: [C,8,128]}}
      doc_chunk_table:       [num_docs, P] int, pooled-chunk indices per doc (pad=0)
      doc_chunk_table_valid: [num_docs, P] bool
    Returns dict {layer: (ctx_k [B,maxctx,8,128], ctx_v, ctx_valid [B,maxctx])}.
    """
    top_k = cfg['msa']['top_k_docs']
    out = {}
    for L in cfg['msa']['router_layers']:
        rq = jnp.einsum('btd,nhd->btnh', query_hidden_by_layer[L], weights[f'layers.{L}.router_q_proj'],
                        preferred_element_type=jnp.float32, out_sharding=P('data', None, None, None))        # [B,Sq,32,128]
        sel = route_topk(cfg, rq, banks['krbar'][L], bank_chunk_doc, bank_chunk_valid,
                         q_mask, num_docs, top_k)                  # [B, top_k]
        sel_clamped = jnp.clip(sel, 0, num_docs - 1)
        # gather this layer's selected chunk indices: [B, top_k, P] -> [B, maxctx]
        ci = doc_chunk_table[sel_clamped]                          # [B, top_k, P]
        cv = doc_chunk_table_valid[sel_clamped] & (sel[:, :, None] >= 0)
        B = rq.shape[0]
        ci = ci.reshape(B, -1)
        cv = cv.reshape(B, -1)
        ctx_k = _gather_ctx(banks['kbar'][L], ci, cv)
        ctx_v = _gather_ctx(banks['vbar'][L], ci, cv)
        out[L] = (ctx_k, ctx_v, cv)
    return out


def _attn(cfg, q, k_local, v_local, local_mask, ctx=None):
    """GQA attention; optionally prepend a doc context (ctx_k, ctx_v, ctx_valid).

    q:[B,T,N,H]  k_local/v_local:[B,S,Kh,H]  local_mask:[B,1,T,S] bool
    ctx: (ctx_k [B,C,Kh,H], ctx_v, ctx_valid [B,C]) or None.
    """
    if ctx is None:
        return jax.nn.dot_product_attention(q, k_local, v_local, mask=local_mask)
    ctx_k, ctx_v, ctx_valid = ctx
    K = jnp.concatenate([ctx_k, k_local], axis=1)
    V = jnp.concatenate([ctx_v, v_local], axis=1)
    T = q.shape[1]
    ctx_mask = jnp.broadcast_to(ctx_valid[:, None, None, :], (q.shape[0], 1, T, ctx_k.shape[1]))
    mask = jnp.concatenate([ctx_mask, local_mask], axis=-1)
    return jax.nn.dot_product_attention(q, K, V, mask=mask)


def _lm_head(cfg, x, weights):
    x = rms_norm(x, weights['norm'], cfg['rms_norm_eps'])
    out_embed = weights['embed_tokens'] if cfg['tie_word_embeddings'] else weights['lm_head']
    return jnp.einsum('btd,vd->btv', x, out_embed, preferred_element_type=x.dtype, out_sharding=P('data', None, 'model'))


def prefill(cfg, query_ids, q_pad_mask, weights, banks, doc_chunk_table,
            doc_chunk_table_valid, bank_chunk_doc, bank_chunk_valid, num_docs, maxlen):
    """Run the query prompt; route per router layer; return last-token logits +
    decode cache with preallocated maxlen local KV.

    query_ids/q_pad_mask: [B, Sq] (Sq <= maxlen). Returns (logits[B,Sq,V], cache).
    cache: {'local_k':{l:[B,maxlen,Kh,H]}, 'local_v':{...}, 'local_valid':[B,maxlen],
            'doc_ctx':{l:(ctx_k,ctx_v,ctx_valid)}, 'pos':Sq}
    """
    eps = cfg['rms_norm_eps']
    rope_theta = cfg['rope_theta']
    router_set = cfg['router_layers_set']
    Kh, H = cfg['num_key_value_heads'], cfg['head_dim']
    B, T = query_ids.shape

    rows = jnp.arange(T)[:, None]
    cols = jnp.arange(T)[None, :]
    causal = (cols <= rows)[None]
    local_mask = (causal & q_pad_mask[:, None, :])[:, None]     # [B,1,T,T]
    # left-padded prompts: real tokens numbered 0.. via cumsum of the pad mask.
    positions = jnp.clip(jnp.cumsum(q_pad_mask.astype(jnp.int32), axis=1) - 1, 0, None)

    x = _embed(weights, query_ids)

    local_k, local_v, doc_ctx = {}, {}, {}
    for i in range(cfg['num_hidden_layers']):
        w = _layer_weights(weights, i)
        x_norm = rms_norm(x, w['input_layernorm'], eps)
        q, k, v = _qkv(cfg, x_norm, w)
        q = apply_rope_pos(q, rope_theta, positions)
        k = apply_rope_pos(k, rope_theta, positions)
        # preallocate maxlen cache and write prefill range
        kc = jnp.zeros((B, maxlen, Kh, H), dtype=k.dtype, out_sharding=P('data', None, None, None)).at[:, :T].set(k, out_sharding=P('data', None, None, None))
        vc = jnp.zeros((B, maxlen, Kh, H), dtype=v.dtype, out_sharding=P('data', None, None, None)).at[:, :T].set(v, out_sharding=P('data', None, None, None))
        local_k[i], local_v[i] = kc, vc
        if i in router_set:
            rq = jnp.einsum('btd,nhd->btnh', x_norm, w['router_q_proj'],
                            preferred_element_type=jnp.float32, out_sharding=P('data', None, None, None))
            sel = route_topk(cfg, rq, banks['krbar'][i], bank_chunk_doc,
                             bank_chunk_valid, q_pad_mask, num_docs, cfg['msa']['top_k_docs'])
            sel_clamped = jnp.clip(sel, 0, num_docs - 1)
            ci = doc_chunk_table.at[sel_clamped].get(out_sharding=P('data', None, None)).reshape(B, -1)
            cv = (doc_chunk_table_valid.at[sel_clamped].get(out_sharding=P('data', None, None))
                  & (sel[:, :, None] >= 0)).reshape(B, -1)
            ctx = (_gather_ctx(banks['kbar'][i], ci, cv),
                   _gather_ctx(banks['vbar'][i], ci, cv), cv)
            doc_ctx[i] = ctx
            attn = _attn(cfg, q, k, v, local_mask, ctx=ctx)
        else:
            attn = _attn(cfg, q, k, v, local_mask, ctx=None)
        o = jnp.einsum('btnh,dnh->btd', attn, w['o_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, None))
        x = x + o
        x = mlp(cfg, x, w)

    logits = _lm_head(cfg, x, weights)
    local_valid = jnp.zeros((B, maxlen), dtype=bool, out_sharding=P('data', None)).at[:, :T].set(q_pad_mask, out_sharding=P('data', None))
    cache = {'local_k': local_k, 'local_v': local_v, 'local_valid': local_valid,
             'doc_ctx': doc_ctx}
    return logits, cache


def decode_step(cfg, token_ids, weights, cache, write_idx, positions):
    """One decode step.

    token_ids: [B,1] int. write_idx: scalar int array (cache array index, same for
    all rows since prompts are left-padded). positions: [B,1] per-row RoPE position.
    Returns (logits_last [B,V], cache).
    """
    eps = cfg['rms_norm_eps']
    rope_theta = cfg['rope_theta']
    router_set = cfg['router_layers_set']
    B = token_ids.shape[0]
    maxlen = cache['local_valid'].shape[1]

    x = _embed(weights, token_ids)   # [B,1,D]
    # local mask: attend to all currently-valid local positions (incl. the new one)
    new_valid = cache['local_valid'].at[:, write_idx].set(True, out_sharding=P('data', None))
    cache['local_valid'] = new_valid

    for i in range(cfg['num_hidden_layers']):
        w = _layer_weights(weights, i)
        x_norm = rms_norm(x, w['input_layernorm'], eps)
        q, k, v = _qkv(cfg, x_norm, w)
        q = apply_rope_pos(q, rope_theta, positions)
        k = apply_rope_pos(k, rope_theta, positions)
        kc = jax.lax.dynamic_update_slice(cache['local_k'][i], k, (0, write_idx, 0, 0))
        vc = jax.lax.dynamic_update_slice(cache['local_v'][i], v, (0, write_idx, 0, 0))
        cache['local_k'][i], cache['local_v'][i] = kc, vc
        local_mask = new_valid[:, None, None, :]                  # [B,1,1,maxlen]
        ctx = cache['doc_ctx'].get(i) if i in router_set else None
        attn = _attn(cfg, q, kc, vc, local_mask, ctx=ctx)
        o = jnp.einsum('btnh,dnh->btd', attn, w['o_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, None))
        x = x + o
        x = mlp(cfg, x, w)

    logits = _lm_head(cfg, x, weights)
    return logits[:, -1], cache


# ---------------------------------------------------------------------------
# Training forward — paper §3.3 recipe (msa.pdf).
#
# Each query owns M=num_chunks_per_doc doc-chunk slots (positives + hard negatives,
# from data/qa.py:pack_docs). Unlike the inference corpus-routing path, routing is
# per-query over its OWN M slots — faithful to the paper's aux loss where the
# negative set N = D \ P is drawn from the query's associated document set D.
#
# Layers 0..mid: plain causal self-attn (local context only).
# Router layers (latter half): query attends to its slots' pooled K̄/V̄ concatenated
#   before its local KV (train-on-short ⇒ all valid slots attended, top_k≈M), and we
#   emit per-layer doc-level cosine routing scores (Eq 2) for the aux loss (Eq 5).
# ---------------------------------------------------------------------------
def _enc_router_layer(cfg, kernel, x, w, attn_mask, pad_mask):
    """One doc-encode layer that also returns pooled K̄/V̄/K̄ᴿ (router layer)."""
    eps, rope_theta = cfg['rms_norm_eps'], cfg['rope_theta']
    x_norm = rms_norm(x, w['input_layernorm'], eps)
    q, k, v = _qkv(cfg, x_norm, w)
    q = apply_rope(q, rope_theta, 0)
    k = apply_rope(k, rope_theta, 0)
    attn = jax.nn.dot_product_attention(q, k, v, mask=attn_mask)
    o = jnp.einsum('btnh,dnh->btd', attn, w['o_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, None))
    x = x + o
    kbar, _ = _masked_pool_chunks(k, pad_mask, kernel)
    vbar, _ = _masked_pool_chunks(v, pad_mask, kernel)
    rk = jnp.einsum('btd,khd->btkh', x_norm, w['router_k_proj'],
                    preferred_element_type=x_norm.dtype, out_sharding=P('data', None, None, None))
    krbar, _ = _masked_pool_chunks(rk, pad_mask, kernel)
    x = mlp(cfg, x, w)
    return x, kbar, vbar, krbar


def _enc_plain_layer(cfg, x, w, attn_mask):
    """One doc-encode layer with no pooling (non-router layer)."""
    eps, rope_theta = cfg['rms_norm_eps'], cfg['rope_theta']
    x_norm = rms_norm(x, w['input_layernorm'], eps)
    q, k, v = _qkv(cfg, x_norm, w)
    q = apply_rope(q, rope_theta, 0)
    k = apply_rope(k, rope_theta, 0)
    attn = jax.nn.dot_product_attention(q, k, v, mask=attn_mask)
    o = jnp.einsum('btnh,dnh->btd', attn, w['o_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, None))
    x = x + o
    x = mlp(cfg, x, w)
    return x


def _encode_docs_banks(cfg, doc_ids, doc_mask, weights):
    """Differentiable doc encode → per-router-layer pooled banks (remat per layer).

    doc_ids/doc_mask: [ND, Ld]. Returns ({L: kbar/vbar/krbar [ND, nch, Kh, Hd]},
    chunk_valid [ND, nch]).
    """
    kernel = cfg['msa']['pooling_kernel_size']
    router_set = set(cfg['msa']['router_layers'])
    ND, Ld = doc_ids.shape
    rows = jnp.arange(Ld)[:, None]
    cols = jnp.arange(Ld)[None, :]
    attn_mask = ((cols <= rows)[None] & doc_mask[:, None, :])[:, None]   # [ND,1,Ld,Ld]

    x = _embed(weights, doc_ids)
    kbar, vbar, krbar = {}, {}, {}
    for i in range(cfg['num_hidden_layers']):
        w = _layer_weights(weights, i)
        if i in router_set:
            x, kbar[i], vbar[i], krbar[i] = jax.remat(partial(_enc_router_layer, cfg, kernel))(
                x, w, attn_mask, doc_mask)
        else:
            x = jax.remat(partial(_enc_plain_layer, cfg))(x, w, attn_mask)

    nch = Ld // kernel
    chunk_valid = (doc_mask[:, :nch * kernel].reshape(ND, nch, kernel).sum(axis=2) > 0)
    return {'kbar': kbar, 'vbar': vbar, 'krbar': krbar}, chunk_valid


def _route_doc_scores(cfg, rq, krbar_q, q_mask, chunk_valid_q):
    """Per-query doc-level cosine routing score (paper Eq 2).

    rq:            [B, T, N, Hd]   router query (no rope/norm).
    krbar_q:       [B, M, nch, Kh, Hd]  pooled router keys per slot.
    q_mask:        [B, T] bool.    chunk_valid_q: [B, M, nch] bool.
    Returns doc_scores [B, M] = max_token mean_head max_subchunk cos(rq, k̄ᴿ).
    """
    Kh = cfg['num_key_value_heads']
    G = cfg['num_attention_heads'] // Kh
    B, T = rq.shape[0], rq.shape[1]
    M, nch = krbar_q.shape[1], krbar_q.shape[2]
    # L2-normalize with a gradient-bounding eps INSIDE the sqrt. The cosine-normalize
    # gradient scales as ~1/||x||, so a tiny eps (1e-12) lets it reach ~1e6 for small
    # router vectors -> bf16 overflow -> NaN grad (worsens as trained routers shrink).
    # eps=1e-6 bounds the gradient to ~1e3 (safe) and is weight-independent. The +eps is
    # inside the sqrt so the grad is finite even at x=0 (NOT sqrt(ssq)+eps, whose grad
    # diverges at 0). Also mask invalid query tokens / doc chunks to 0.
    EPS = 1e-6
    def _safe_norm(x):
        return x * jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + EPS)
    rqn = _safe_norm(rq.astype(jnp.float32)) * q_mask.astype(jnp.float32)[:, :, None, None]       # [B,T,N,Hd]
    krn = _safe_norm(krbar_q.astype(jnp.float32)) * chunk_valid_q.astype(jnp.float32)[:, :, :, None, None]  # [B,M,nch,Kh,Hd]
    rqg = rqn.reshape(B, T, Kh, G, rqn.shape[-1])
    # cos score per (query token, q-head, slot, subchunk)
    sc = jnp.einsum('btkgd,bmckd->btkgmc', rqg, krn,
                    preferred_element_type=jnp.float32)        # [B,T,Kh,G,M,nch]
    sc = sc.reshape(B, T, Kh * G, M, nch).mean(axis=2)         # mean over heads -> [B,T,M,nch]
    # finite sentinel (not finfo.min): cosine ∈ [-1,1], so -1e4 is "never selected"
    # yet stays finite after the loss's /τ (finfo.min/τ overflows to -inf → NaN grad).
    neg = jnp.float32(-1e4)
    valid = q_mask[:, :, None, None].astype(bool) & chunk_valid_q[:, None, :, :]
    sc = jnp.where(valid, sc, neg)
    sc = sc.max(axis=1)                                        # max over query tokens -> [B,M,nch]
    return sc.max(axis=2)                                      # max over subchunks -> [B,M]


def _query_router_layer(cfg, x, w, local_mask, ctx_k, ctx_v, ctx_valid, krbar_q,
                        q_mask, chunk_valid_q):
    """Router query layer: doc-context attention + per-layer routing scores."""
    eps, rope_theta = cfg['rms_norm_eps'], cfg['rope_theta']
    x_norm = rms_norm(x, w['input_layernorm'], eps)
    q, k, v = _qkv(cfg, x_norm, w)
    q = apply_rope(q, rope_theta, 0)
    k = apply_rope(k, rope_theta, 0)
    rq = jnp.einsum('btd,nhd->btnh', x_norm, w['router_q_proj'],
                    preferred_element_type=jnp.float32, out_sharding=P('data', None, None, None))
    doc_scores = _route_doc_scores(cfg, rq, krbar_q, q_mask, chunk_valid_q)
    attn = _attn(cfg, q, k, v, local_mask, ctx=(ctx_k, ctx_v, ctx_valid))
    o = jnp.einsum('btnh,dnh->btd', attn, w['o_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, None))
    x = x + o
    x = mlp(cfg, x, w)
    return x, doc_scores


def train_forward(cfg, x, weights, pad_mask=None, collect_aux=False):
    """Teacher-forced forward over a QA batch with per-query MSA routing.

    x:        {"batch": [B,T] query+answer ids, "docs": [B*M, Ld] doc-chunk ids}
    pad_mask: {"batch_mask":[B,T], "docs_mask":[B*M,Ld], "pos_doc_mask":[B,M]}
    Returns ModelOutput(logits=[B,T,V], aux={route_scores, slot_valid, pos_doc_mask}).
    """
    q_ids = x["batch"]
    docs = x["docs"]
    batch_mask = pad_mask["batch_mask"]
    docs_mask = pad_mask["docs_mask"]
    B, T = q_ids.shape
    ND = docs.shape[0]
    M = ND // B
    router_set = set(cfg['msa']['router_layers'])
    Kh, Hd = cfg['num_key_value_heads'], cfg['head_dim']

    # Phase 1 — encode this batch's docs into per-router-layer pooled banks.
    banks, chunk_valid = _encode_docs_banks(cfg, docs, docs_mask, weights)   # [ND,nch,...], [ND,nch]
    nch = chunk_valid.shape[1]
    # Reshapes split the data-sharded leading axis ND=B*M -> (B, M); each query's
    # M rows are contiguous and on one device, so this is shard-free, but JAX needs
    # the output sharding spelled out explicitly.
    sh4 = P('data', None, None, None)
    sh5 = P('data', None, None, None, None)
    chunk_valid_q = chunk_valid.reshape(B, M, nch, out_sharding=P('data', None, None))   # [B,M,nch]
    slot_valid = chunk_valid_q.any(axis=2)                                   # [B,M]

    # Per-query doc context (all valid slots) and per-query router-key banks.
    ctx_k = {L: banks['kbar'][L].reshape(B, M * nch, Kh, Hd, out_sharding=sh4) for L in router_set}
    ctx_v = {L: banks['vbar'][L].reshape(B, M * nch, Kh, Hd, out_sharding=sh4) for L in router_set}
    krbar_q = {L: banks['krbar'][L].reshape(B, M, nch, Kh, Hd, out_sharding=sh5) for L in router_set}
    ctx_valid = chunk_valid.reshape(B, M * nch, out_sharding=P('data', None))            # [B,M*nch]

    # Phase 2 — query forward (right-padded, standard causal positions).
    local_mask = create_mask(cfg, jnp.zeros((B, T, 1)), None, batch_mask, 0)  # [B,1,T,T]
    # Align with the data-sharded doc-context mask so _attn's mask concat matches.
    local_mask = jax.sharding.reshard(local_mask, P('data', None, None, None))
    x_h = _embed(weights, q_ids)
    route_scores = []
    for i in range(cfg['num_hidden_layers']):
        w = _layer_weights(weights, i)
        if i in router_set:
            x_h, ds = jax.remat(partial(_query_router_layer, cfg))(
                x_h, w, local_mask, ctx_k[i], ctx_v[i], ctx_valid,
                krbar_q[i], batch_mask, chunk_valid_q)
            route_scores.append(ds)
        else:
            x_h, _ = jax.remat(partial(forward_layer, cfg))(x_h, w, local_mask, None, 0)

    logits = _lm_head(cfg, x_h, weights)
    aux = None
    if collect_aux:
        aux = {'route_scores': route_scores, 'slot_valid': slot_valid,
               'pos_doc_mask': pad_mask.get('pos_doc_mask')}
    return ModelOutput(logits=logits, kv=None, aux=aux)

