"""
Weight-0 read-channel telemetry for the grounding experiments.

Registered as auxiliary "losses" but meant to run at weight 0: each is computed and logged
every step while contributing nothing to the gradient. See grounding_experiments_plan.md
"Training-time telemetry".

Most values are emitted per memory layer by models/memory.py::memory_layer and aggregated by
models/qwen3_mem_embed.py::main_forward into a per-layer LIST (one entry per memory-bearing
layer, in mem_layers order). Metrics that return a dict {"l0":..., "mean":...} are logged as
train/<name>/l0 ... and train/<name>/mean (see losses/registry.py). List index i corresponds
to mem_layers[i].
"""
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from .registry import register_aux_loss


def _per_layer(aux_data, key):
    """{l0..ln, mean} over the per-layer list at aux_data[key]; 0.0 if absent."""
    vals = aux_data.get(key)
    if not vals or len(vals) == 0:
        return 0.0
    out = {f"l{i}": jnp.asarray(v, dtype=jnp.float32) for i, v in enumerate(vals)}
    out["mean"] = jnp.mean(jnp.stack([out[f"l{i}"] for i in range(len(vals))]))
    return out


# --- signals emitted directly by memory_layer (per-layer scalars) ---
@register_aux_loss("mem_write_norm")
def compute_mem_write_norm(aux_data, mask, input_mask, inputs, **kwargs):
    """‖o‖ of the memory write into the residual stream. With zero-init mem_o_proj this
    starts at 0 and should climb as the branch earns CE (the 'channel is alive' check)."""
    return _per_layer(aux_data, "mem_write_norm")


@register_aux_loss("mem_write_ratio")
def compute_mem_write_ratio(aux_data, mask, input_mask, inputs, **kwargs):
    """‖o‖/‖x‖: write relative to the residual-stream norm (dilution; watch it shrink with depth)."""
    return _per_layer(aux_data, "mem_write_ratio")


@register_aux_loss("mem_topk_entropy")
def compute_mem_topk_entropy(aux_data, mask, input_mask, inputs, **kwargs):
    """Entropy of the top-k softmax mixture — the 128-way averaging blur. Stays high => mush."""
    return _per_layer(aux_data, "mem_topk_entropy")


@register_aux_loss("mem_effective_slots")
def compute_mem_effective_slots(aux_data, mask, input_mask, inputs, **kwargs):
    """Participation ratio 1/Σwᵢ²: how many slots actually contribute to the mixture."""
    return _per_layer(aux_data, "mem_effective_slots")


@register_aux_loss("mem_top1_weight")
def compute_mem_top1_weight(aux_data, mask, input_mask, inputs, **kwargs):
    """Max softmax weight — complementary sharpness read on the mixture."""
    return _per_layer(aux_data, "mem_top1_weight")


@register_aux_loss("mem_o_proj_norm")
def compute_mem_o_proj_norm(aux_data, mask, input_mask, inputs, **kwargs):
    """‖W_o‖ per layer — zero-init growth / dead-layer check (a layer stuck ~0 is collapsed)."""
    return _per_layer(aux_data, "mem_o_proj_norm")


@register_aux_loss("mem_boundary_straddle")
def compute_mem_boundary_straddle(aux_data, mask, input_mask, inputs, **kwargs):
    """Stage 3: share of span windows that fell out-of-document (should be ~0; correctness probe)."""
    return _per_layer(aux_data, "mem_boundary_straddle")


@register_aux_loss("mem_head_query_cos")
def compute_mem_head_query_cos(aux_data, mask, input_mask, inputs, **kwargs):
    """Mean pairwise cross-head query cosine per layer — heads can only specialize their query
    direction (K/V shared); → 1 means the heads collapse to one probe and extra heads buy nothing."""
    return _per_layer(aux_data, "mem_head_query_cos")


@register_aux_loss("mem_cross_layer_cos")
def compute_mem_cross_layer_cos(aux_data, mask, input_mask, inputs, **kwargs):
    """Mean pairwise cosine of the per-layer write directions (mean o vector). High => layers
    write redundant info (multi-layer buys little); low => complementary writes. 0 if <2 layers."""
    vecs = aux_data.get("mem_o_meanvec")
    if not vecs or len(vecs) < 2:
        return 0.0
    V = jnp.stack([jnp.asarray(v, dtype=jnp.float32) for v in vecs])   # [L, D]
    Vn = V / (jnp.linalg.norm(V, axis=-1, keepdims=True) + 1e-9)
    gram = Vn @ Vn.T                                                    # [L, L]
    L = gram.shape[0]
    return (gram.sum() - jnp.trace(gram)) / (L * (L - 1) + 1e-9)


@register_aux_loss("mem_kv_cos")
def compute_mem_kv_cos(aux_data, mask, input_mask, inputs, **kwargs):
    """cos(K_repr, V_repr) for the same doc token — the Stage-2 decouple metric; should DROP as
    the key/value models specialize into different objects."""
    v = aux_data.get("mem_kv_cos")
    return v if v is not None else 0.0


@register_aux_loss("mem_value_anisotropy")
def compute_mem_value_anisotropy(aux_data, mask, input_mask, inputs, **kwargs):
    """Mean pairwise cosine of the value bank — embedding trunks collapse values (high cosine);
    base-LM values should be more spread (lower). Quantifies the fidelity win."""
    v = aux_data.get("mem_value_anisotropy")
    return v if v is not None else 0.0


# --- positive-slot diagnostics: join retrieved slots with the positive-doc mask ---
def _positive_slot_stats(aux_data, loss_mask, input_mask):
    """Yield (weight_mass, hit_rate) per layer. weight_mass = softmax weight landing on
    positive-doc slots (the 'right answer' number, vs doc_access_acc's 'right doc' argmax
    count). hit_rate = fraction of active queries whose top-k contains any positive slot.
    Mirrors losses/doc_access_acc.py for the positive-doc lookup."""
    idx_list = aux_data.get("mem_top_k_indices")
    prob_list = aux_data.get("mem_top_k_probs")
    if not idx_list:
        return None

    docs_mask = input_mask["docs_mask"]
    num_docs_total, raw_doc_len = docs_mask.shape
    doc_len = aux_data.get("effective_doc_len", raw_doc_len)
    eff_mask = aux_data.get("effective_mem_mask", None)
    validity_mask = eff_mask if eff_mask is not None else docs_mask.flatten()

    mass_by_layer, hit_by_layer = [], []
    for li, indices in enumerate(idx_list):
        B, H, S, K = indices.shape
        retrieved_doc_ids = indices // doc_len

        if input_mask is not None and "pos_doc_mask" in input_mask:
            pos_mask_local = input_mask["pos_doc_mask"]
        else:
            pos_mask_local = jnp.ones((B, 1), dtype=jnp.float32)
        B_local, M_local = pos_mask_local.shape
        block_eye = jnp.eye(B_local, dtype=pos_mask_local.dtype)
        pos_doc_mask_global = (block_eye[:, :, None] * pos_mask_local[None, :, :]).reshape(B_local, num_docs_total)

        batch_idx = jnp.arange(B)[:, None, None, None]
        is_pos = pos_doc_mask_global.at[batch_idx, retrieved_doc_ids].get(out_sharding=P('data', 'model', None, None)) > 0
        is_valid_token = validity_mask.at[indices].get(out_sharding=P('data', 'model', None, None)) == 1
        is_active = loss_mask[:, None, :, None].astype(jnp.bool_)
        is_valid = is_valid_token & is_active
        pos_valid = (is_pos & is_valid).astype(jnp.float32)

        # hit rate: any positive slot in the top-k, per active query
        any_hit = jnp.max(pos_valid, axis=-1)                      # [B,H,S]
        active_q = jnp.broadcast_to(is_active[..., 0], any_hit.shape).astype(jnp.float32)
        hit_by_layer.append(jnp.sum(any_hit) / (jnp.sum(active_q) + 1e-6))

        # weight mass: softmax weight on positive slots / weight on valid slots
        if prob_list is not None and li < len(prob_list):
            probs = prob_list[li].astype(jnp.float32)              # [B,H,S,K]
            valid_f = is_valid.astype(jnp.float32)
            mass_by_layer.append(jnp.sum(probs * pos_valid) / (jnp.sum(probs * valid_f) + 1e-6))
        else:
            mass_by_layer.append(jnp.asarray(0.0))
    return mass_by_layer, hit_by_layer


def collect_eval_telemetry(aux_data, loss_mask, input_mask):
    """Flat dict of memory-attention telemetry scalars (layer-mean) for eval-time logging:
    mem_topk_entropy, mem_effective_slots, mem_top1_weight (from aux_data), plus
    mem_pos_weight_mass + mem_hit_rate (softmax mass / top-k hit on the positive-doc slots).
    Returns {} if the aux data isn't present. Safe to call once per batch and average."""
    out = {}
    for key in ("mem_topk_entropy", "mem_effective_slots", "mem_top1_weight"):
        v = _per_layer(aux_data, key)
        if isinstance(v, dict):
            if "mean" in v:
                out[key] = float(v["mean"])
        elif v is not None:
            out[key] = float(v)
    try:
        stats = _positive_slot_stats(aux_data, loss_mask, input_mask) if input_mask else None
    except Exception:
        stats = None  # corpus eval may lack the batch pos-doc mask; entropy/slots/top1 still returned
    if stats is not None:
        mass_by_layer, hit_by_layer = stats
        if mass_by_layer:
            out["mem_pos_weight_mass"] = float(jnp.mean(jnp.stack([jnp.asarray(x) for x in mass_by_layer])))
        if hit_by_layer:
            out["mem_hit_rate"] = float(jnp.mean(jnp.stack([jnp.asarray(x) for x in hit_by_layer])))
    return out


@register_aux_loss("mem_pos_weight_mass")
def compute_mem_pos_weight_mass(aux_data, mask, input_mask, inputs, **kwargs):
    """Softmax weight mass on positive-doc slots — the number that separates 'right doc'
    from 'right answer' (train-time analog of the eval answer-slot-weight diagnostic)."""
    stats = _positive_slot_stats(aux_data, mask, input_mask)
    if stats is None:
        return 0.0
    mass, _ = stats
    out = {f"l{i}": v for i, v in enumerate(mass)}
    out["mean"] = jnp.mean(jnp.stack(mass))
    return out


@register_aux_loss("mem_hit_rate")
def compute_mem_hit_rate(aux_data, mask, input_mask, inputs, **kwargs):
    """Fraction of active queries whose top-k contains a positive-doc slot (train analog of
    doc_token_hit_rate)."""
    stats = _positive_slot_stats(aux_data, mask, input_mask)
    if stats is None:
        return 0.0
    _, hit = stats
    out = {f"l{i}": v for i, v in enumerate(hit)}
    out["mean"] = jnp.mean(jnp.stack(hit))
    return out
