import os
import jax
from jax.sharding import PartitionSpec as P
import jax.numpy as jnp
from omegaconf import OmegaConf

from .qwen3 import rms_norm
from utils import is_jax_mesh_active

# TPU-native approximate top-k for the memory-bank search. Exact jax.lax.top_k over a large bank
# is sorting-network-bound on TPU: at M=65,536 it is ~40% of the whole 4B train step, and
# approx_max_k cuts the step ~1.6-1.7x at ~98% recall@64 (see
# wiki/experiments/2026-07-15-approx-topk-training.md). Controlled per-model by the cfg keys
# mem_approx_topk / mem_approx_recall; the MEM_APPROX_TOPK / MEM_APPROX_RECALL env vars override
# cfg for one-off experiments. Built-in fallback is exact, so a config that sets neither is
# unchanged.
def resolve_approx_topk(cfg=None):
    """(use_approx, recall_target). Precedence: env var > model cfg > built-in default (exact)."""
    env = os.environ.get("MEM_APPROX_TOPK")
    if env is not None:
        use_approx = env == "1"
    else:
        cfg_val = cfg.get("mem_approx_topk") if cfg is not None else None
        use_approx = bool(cfg_val) if cfg_val is not None else False

    env_recall = os.environ.get("MEM_APPROX_RECALL")
    if env_recall is not None:
        recall = float(env_recall)
    else:
        cfg_recall = cfg.get("mem_approx_recall") if cfg is not None else None
        recall = float(cfg_recall) if cfg_recall is not None else 0.95
    return use_approx, recall


def bank_top_k(x, k, cfg=None):
    """top-k over the memory bank; approximate (TPU-fast) per resolve_approx_topk(cfg)."""
    use_approx, recall = resolve_approx_topk(cfg)
    if use_approx:
        return jax.lax.approx_max_k(x, k, recall_target=recall)
    return jax.lax.top_k(x, k)

def get_indices_product_keys(query, keys, knn, mem_mask=None):
    """
    Product key lookup for efficient memory attention.
    
    Args:
        query: [B, T, N, H] query tensor
        keys: [2, sqrt_mem_size, H//2] product keys
        knn: number of top-k neighbors
    
    Returns:
        top_k_scores: [B, N, T, knn] scores
        top_k_indices: [B, N, T, knn] indices into full memory
        pk_scores: tuple (scores1, scores2) raw scores for aux losses, each [B, T, N, n_keys]
    """
    B, T, N, H = query.shape
    half = H // 2
    n_keys = keys.shape[1]  # sqrt(mem_size)
    
    # Split query for product quantization
    q1 = query[..., :half]  # [B, T, N, half]
    q2 = query[..., half:]  # [B, T, N, half] 
    
    # keys: [2, n_keys, half]
    keys1 = keys[0]  # [n_keys, half]
    keys2 = keys[1]  # [n_keys, half]
    
    # Compute scores for each half - these are the expensive ops, shard them
    scores1 = jnp.einsum('btnh,kh->btnk', q1, keys1, preferred_element_type=query.dtype, out_sharding=P('data', None, 'model', None))
    scores2 = jnp.einsum('btnh,kh->btnk', q2, keys2, preferred_element_type=query.dtype, out_sharding=P('data', None, 'model', None))
    
    scores1 = (scores1 / jnp.sqrt(jnp.array(half, dtype=jnp.float32))).astype(scores1.dtype)
    scores2 = (scores2 / jnp.sqrt(jnp.array(half, dtype=jnp.float32))).astype(scores2.dtype)

    if mem_mask is not None:
        scores1 = jnp.where(mem_mask, scores1, -1e9)
        scores2 = jnp.where(mem_mask, scores2, -1e9)

    # Get top-k for each half - small knn, sharding inherited from inputs
    top_scores1, top_indices1 = jax.lax.top_k(scores1, knn)
    top_scores2, top_indices2 = jax.lax.top_k(scores2, knn)
    
    # Cartesian product - small tensors [B, T, N, knn^2], no explicit sharding needed
    scores1_exp = top_scores1[..., :, None]
    scores2_exp = top_scores2[..., None, :]
    all_scores = (scores1_exp + scores2_exp).reshape(B, T, N, -1)
    
    indices1_exp = top_indices1[..., :, None]
    indices2_exp = top_indices2[..., None, :]
    all_indices = (indices1_exp * n_keys + indices2_exp).reshape(B, T, N, -1)
    
    # Select overall best scores and indices
    top_k_scores, best_indices = jax.lax.top_k(all_scores, knn)
    top_k_indices = jnp.take_along_axis(all_indices, best_indices, axis=-1)
    
    # Transpose and constrain output sharding to match forward_layer expectations
    top_k_scores = jnp.transpose(top_k_scores, (0, 2, 1, 3))
    top_k_indices = jnp.transpose(top_k_indices, (0, 2, 1, 3))
    
    # Return raw scores for auxiliary losses
    pk_scores = (scores1, scores2)
    
    return top_k_scores, top_k_indices, pk_scores


def get_memory_sharding(name):
    if "embed_proj_conv_k" in name or "embed_proj_conv_v" in name: return P()
    if "mem_q_proj" in name: return P('model', 'data')
    if "mem_o_proj" in name: return P('data', 'model')
    if "mem_gate_proj" in name: return P('model', 'data')
    if "mem_up_proj" in name: return P('data', 'model')
    if "mem_k_proj" in name: return P('model', None)
    if "mem_v_proj" in name: return P('model', None)
    if "mem_k_prod_proj" in name: return P('model', 'data')
    if "mem_q_norm" in name or "mem_k_norm" in name or "mem_o_norm" in name: return P()
    if "mem_layer_scale" in name: return P()
    if "mem_k" in name or "mem_v" in name: return P('model', None)
    return P()


def add_memory_layer(cfg, model_cfg, weights, init_empty=False):

    mem_cfg = cfg.memory if hasattr(cfg, 'memory') else cfg

    layers = mem_cfg.get("mem_layers", [])
    mem_size = mem_cfg.get("mem_size")
    num_heads = mem_cfg.get("mem_num_heads")
    k_dim = mem_cfg.get("mem_k_dim")
    v_dim = mem_cfg.get("mem_v_dim")
    use_product_keys = mem_cfg.get("mem_use_product_keys", False)
    placement = mem_cfg.get("mem_placement", "replace_mlp")
    # Zero-init the memory output projection: the memory branch starts as identity
    # (contributes nothing) and its write grows only as it earns CE reduction, so any
    # CE drop is provably memory-sourced. Also subsumes the old routing warmup.
    # See grounding_experiments_plan.md Stage 1.
    zero_init_o = mem_cfg.get("mem_o_proj_zero_init", False)
    hidden_size = model_cfg["hidden_size"]

    if layers == []:
        return weights, model_cfg

    # Add memory settings from Hydra config to model config
    model_cfg.update(OmegaConf.to_container(mem_cfg, resolve=True))

    key = jax.random.PRNGKey(42)

    # layers is an array of integers specifying which MLP layers will be replaced with memory layers
    for i in layers:
        prefix = f'layers.{i}.'

        if placement == "replace_mlp":
            # delete the mlp weights for that layer
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                if (k := prefix + name) in weights:
                    del weights[k]
        
        if placement == "after_attention":
            key, subkey = jax.random.split(key)
            weights[prefix + "mem_layernorm"] = jax.nn.initializers.ones(subkey, shape=(hidden_size,), dtype=jnp.bfloat16)
            
            

        # Doc-code: the bank key is [token_key (k_dim) ; doc_code (code_dim)], so the query
        # projection and both retrieval norms are sized to the CONCATENATED retrieval dim.
        # The dot product then decomposes additively: token-similarity + doc-affinity.
        retr_dim = k_dim + int(mem_cfg.get("mem_doc_code_dim", 0) or 0)
        for name, shape, target in [("mem_q_proj", (num_heads * retr_dim, hidden_size), (num_heads, retr_dim, hidden_size)),
                                    ("mem_o_proj", (hidden_size, num_heads * v_dim), (hidden_size, num_heads, v_dim))]:
            key, subkey = jax.random.split(key)
            if name == "mem_o_proj" and zero_init_o:
                weights[prefix + name] = jax.jit(lambda: jnp.zeros(shape, dtype=jnp.bfloat16), out_shardings=get_memory_sharding(prefix + name))()
            else:
                weights[prefix + name] = jax.jit(lambda k: jax.random.normal(k, shape, dtype=jnp.bfloat16) * 0.02, out_shardings=get_memory_sharding(prefix + name))(subkey)
            weights[prefix + name] = weights[prefix + name].reshape(target)
        
        # RMS norm weights for query, key, and output (initialized to ones, like Qwen3 Q/K norms).
        # With doc-code these span the concatenated retrieval dim (joint RMS over both parts —
        # the simple choice; separate per-part norms would decouple the scales, not done here).
        weights[prefix + "mem_q_norm"] = jnp.ones((retr_dim,), dtype=jnp.bfloat16)
        weights[prefix + "mem_k_norm"] = jnp.ones((retr_dim,), dtype=jnp.bfloat16)
        weights[prefix + "mem_o_norm"] = jnp.ones((hidden_size,), dtype=jnp.bfloat16)

        # LayerScale: scalar initialized to 0.1 for stable residual addition
        weights[prefix + "mem_layer_scale"] = jnp.array(0.1, dtype=jnp.bfloat16)

        # Add gating weights if enabled
        if mem_cfg.get("mem_use_gating", False):
            # mem_gate_proj: projects from hidden_size to num_heads * v_dim (gate logits)
            key, subkey = jax.random.split(key)
            gate_shape = (num_heads * v_dim, hidden_size)
            weights[prefix + "mem_gate_proj"] = jax.random.normal(subkey, gate_shape, dtype=jnp.bfloat16) * 0.02
            weights[prefix + "mem_gate_proj"] = jax.device_put(weights[prefix + "mem_gate_proj"], get_memory_sharding(prefix + "mem_gate_proj"))# sharding later
            
            # mem_up_proj: projects from num_heads * v_dim to hidden_size (final output)
            key, subkey = jax.random.split(key)
            up_shape = (hidden_size, num_heads * v_dim)
            weights[prefix + "mem_up_proj"] = jax.random.normal(subkey, up_shape, dtype=jnp.bfloat16) * 0.02
            weights[prefix + "mem_up_proj"] = jax.device_put(weights[prefix + "mem_up_proj"], get_memory_sharding(prefix + "mem_up_proj"))

        
    if init_empty:
        weights["mem_k"] = jnp.array([])
        weights["mem_v"] = jnp.array([])
    else:
        key, k1, k2 = jax.random.split(key, 3)

        if use_product_keys:
            # Product keys: shape [2, sqrt_mem_size, mem_k_dim // 2]
            sqrt_mem_size = int(jnp.sqrt(mem_size))
            assert sqrt_mem_size * sqrt_mem_size == mem_size, f"mem_size must be a perfect square for product keys, got {mem_size}"
            mem_k = jax.random.normal(k1, (2, sqrt_mem_size, k_dim // 2), dtype=jnp.bfloat16) * 0.02
            weights["mem_k"] = jax.device_put(mem_k, P(None, 'model', 'data'))
        else:
            # Standard keys: shape [mem_size, mem_k_dim]
            mem_k = jax.random.normal(k1, (mem_size, k_dim), dtype=jnp.bfloat16) * 0.02
            weights["mem_k"] = jax.device_put(mem_k, get_memory_sharding("mem_k"))
        
        mem_v = jax.random.normal(k2, (mem_size, v_dim), dtype=jnp.bfloat16) * 0.02
        weights["mem_v"] = jax.device_put(mem_v, get_memory_sharding("mem_v"))

    return weights, model_cfg
 

def _pad_memory_to_chunk_multiple(mem, chunk_size: int):
    """Pad a memory bank along axis-0 to be divisible by chunk_size.

    Returns (padded_mem, original_size).
    """
    M = mem.shape[0]
    remainder = M % chunk_size
    if remainder == 0:
        return mem, M
    pad_len = chunk_size - remainder
    pad_shape = (pad_len,) + mem.shape[1:]
    mem = jnp.concatenate([mem, jnp.zeros(pad_shape, dtype=mem.dtype)], axis=0)
    return mem, M


def chunked_memory_top_k_retrieval(query, mem_k, mem_v, top_k, mem_mask=None,
                                    chunk_size=8192):
    """Scan-based chunked top-k inner-product retrieval over a large memory bank.

    Streams through the memory bank in fixed-size chunks so the full
    [B, N, T, M] score tensor is never materialised.

    Args:
        query:    [B, N, T, D]  — batch, heads, tokens, head_dim
        mem_k:    [M, D]        — memory keys
        mem_v:    [M, Dv]       — memory values
        top_k:    int           — number of top results to return
        mem_mask: [M] bool      — True = valid entry (optional)
        chunk_size: int         — memory entries processed per scan step

    Returns:
        top_k_scores:  [B, N, T, K]       softmax-normalised scores
        top_k_values:  [B, N, T, K, Dv]   gathered values
        top_k_indices: [B, N, T, K]       indices into the original memory bank
        all_scores:    [B, N, T, real_M]  raw (pre-softmax) scores for every valid memory entry
    """
    B, N, T, D = query.shape
    use_mesh = is_jax_mesh_active()

    mem_k_padded, real_M = _pad_memory_to_chunk_multiple(mem_k, chunk_size)
    mem_v_padded, _      = _pad_memory_to_chunk_multiple(mem_v, chunk_size)
    M_padded = mem_k_padded.shape[0]
    n_chunks = M_padded // chunk_size

    if mem_mask is not None:
        pad_len = M_padded - mem_mask.shape[0]
        if pad_len > 0:
            mem_mask = jnp.concatenate([mem_mask, jnp.zeros(pad_len, dtype=mem_mask.dtype)])

    mem_k_chunks = mem_k_padded.reshape(n_chunks, chunk_size, D)
    mask_chunks  = mem_mask.reshape(n_chunks, chunk_size) if mem_mask is not None else None

    carry_sharding = P('data', 'model', None, None) if use_mesh else None

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
            m = jax.lax.dynamic_slice_in_dim(mask_chunks, chunk_idx, 1, axis=0)[0]
            scores = jnp.where(m[None, None, None, :], scores, jnp.finfo(jnp.float32).min)

        offsets = chunk_idx * chunk_size + jnp.arange(chunk_size)
        scores = jnp.where((offsets < real_M)[None, None, None, :], scores,
                           jnp.finfo(jnp.float32).min)

        chunk_indices = jnp.broadcast_to(
            (chunk_idx * chunk_size + jnp.arange(chunk_size))[None, None, None, :],
            scores.shape,
        )

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

        return (new_top_scores, new_top_indices), scores

    init_scores  = jnp.full((B, N, T, top_k), jnp.finfo(jnp.float32).min, dtype=jnp.float32)
    init_indices = jnp.zeros((B, N, T, top_k), dtype=jnp.int32)
    if carry_sharding is not None:
        init_scores  = jax.sharding.reshard(init_scores,  carry_sharding)
        init_indices = jax.sharding.reshard(init_indices, carry_sharding)

    (final_scores, final_indices), all_chunk_scores = jax.lax.scan(
        scan_fn, (init_scores, init_indices), jnp.arange(n_chunks),
    )

    # all_chunk_scores: [n_chunks, B, N, T, chunk_size] → [B, N, T, real_M]
    all_scores = jnp.transpose(all_chunk_scores, (1, 2, 3, 0, 4)).reshape(B, N, T, -1)[:, :, :, :real_M]

    top_k_scores = jax.nn.softmax(final_scores, axis=-1).astype(query.dtype)
    if use_mesh:
        top_k_values = mem_v_padded.at[final_indices].get(
            out_sharding=P('data', 'model', None, None, None)
        )
    else:
        top_k_values = mem_v_padded[final_indices]

    return top_k_scores, top_k_values, final_indices, all_scores


def add_kv_head(weights, cfg):

    d_embed = weights['norm'].shape[0]
    k_dim = cfg['mem_k_dim']
    v_dim = cfg['mem_v_dim']

    nkv = int(cfg.get('mem_num_kv_heads', 0) or 0)
    key = jax.random.PRNGKey(44)
    key, k1, k2 = jax.random.split(key, 3)

    if nkv > 0:
        # GQA per-kv-head bank: project encoder hidden -> [d_embed, Nkv, dim] (matches the MaxText
        # mem_proj kernels). Left replicated (tiny; eval runs tp_devices=1).
        weights['mem_k_proj'] = jax.random.normal(k1, (d_embed, nkv, k_dim), dtype=jnp.bfloat16) * 0.02
        weights['mem_v_proj'] = jax.random.normal(k2, (d_embed, nkv, v_dim), dtype=jnp.bfloat16) * 0.02
        return weights

    # Single shared bank (original path)
    weights['mem_k_proj'] = jax.random.normal(k1, (d_embed, k_dim), dtype=jnp.bfloat16) * 0.02
    weights['mem_v_proj'] = jax.random.normal(k2, (d_embed, v_dim), dtype=jnp.bfloat16) * 0.02

    # Doc-code: per-chunk pooled-identity projection, concatenated onto every token key of
    # the chunk in embed_forward. Single-bank path only (GQA banks unsupported).
    code_dim = int(cfg.get('mem_doc_code_dim', 0) or 0)
    if code_dim > 0:
        key, k3 = jax.random.split(key)
        weights['doc_code_proj'] = jax.random.normal(k3, (d_embed, code_dim), dtype=jnp.bfloat16) * 0.02
        weights['doc_code_proj'] = jax.device_put(weights['doc_code_proj'], get_memory_sharding('doc_code_proj'))

    weights['mem_k_proj'] = jax.device_put(weights['mem_k_proj'], get_memory_sharding('mem_k_proj'))
    weights['mem_v_proj'] = jax.device_put(weights['mem_v_proj'], get_memory_sharding('mem_v_proj'))

    return weights


def add_embed_conv(cfg, model_cfg, weights):
    """Initialize 1D conv weights for the embed model (applied before k/v projection).
    
    Follows the same pattern as add_memory_layer: propagates config and returns (weights, model_cfg).
    """
    embed_cfg = cfg.embed_model if hasattr(cfg, 'embed_model') else cfg

    # Propagate embed conv settings into model config
    for key in ('embed_conv', 'embed_conv_kernel_size', 'embed_conv_stride'):
        val = embed_cfg.get(key, None)
        if val is not None:
            model_cfg[key] = val

    if not model_cfg.get('embed_conv', False):
        return weights, model_cfg

    d_embed = weights['norm'].shape[0]
    kernel_size = model_cfg['embed_conv_kernel_size']

    key = jax.random.PRNGKey(45)
    key, k1, k2 = jax.random.split(key, 3)

    # Separate conv for keys
    weights['embed_proj_conv_k_weight'] = jax.random.normal(k1, (d_embed, d_embed, kernel_size), dtype=jnp.bfloat16) * 0.02
    weights['embed_proj_conv_k_bias'] = jnp.zeros((d_embed,), dtype=jnp.bfloat16)
    weights['embed_proj_conv_k_weight'] = jax.device_put(weights['embed_proj_conv_k_weight'], get_memory_sharding('embed_proj_conv_k'))
    weights['embed_proj_conv_k_bias'] = jax.device_put(weights['embed_proj_conv_k_bias'], get_memory_sharding('embed_proj_conv_k'))

    # Separate conv for values
    weights['embed_proj_conv_v_weight'] = jax.random.normal(k2, (d_embed, d_embed, kernel_size), dtype=jnp.bfloat16) * 0.02
    weights['embed_proj_conv_v_bias'] = jnp.zeros((d_embed,), dtype=jnp.bfloat16)
    weights['embed_proj_conv_v_weight'] = jax.device_put(weights['embed_proj_conv_v_weight'], get_memory_sharding('embed_proj_conv_v'))
    weights['embed_proj_conv_v_bias'] = jax.device_put(weights['embed_proj_conv_v_bias'], get_memory_sharding('embed_proj_conv_v'))

    return weights, model_cfg