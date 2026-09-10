import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from .registry import register_aux_loss


@register_aux_loss("doc_access_top_k_loss")
def compute_doc_access_top_k_loss(aux_data, mask, input_mask, inputs, temperature=1.0, **kwargs):
    """
    Contrastive doc-access loss using pass-2 top-k scores (with gradients).

    Same contrastive objective as doc_access_loss but operates only over the K
    selected memory vectors from the two-pass top-k lookup:

        loss = logsumexp(logits_k[valid]) - logsumexp(logits_k[positive & valid])

    The partition function over K is a tight approximation to the full-M partition
    because non-top-k entries contribute ~0 to the softmax.

    Tradeoff vs. full doc_access_loss:
    - Gradients flow through pass-2 (exact, not stop-gradient).
    - If a positive doc is not in top-k, that sample contributes 0 loss / gradient.
    """
    top_k_indices_list = aux_data.get("mem_top_k_indices")
    top_k_logits_list = aux_data.get("mem_top_k_logits")

    if not top_k_indices_list or not top_k_logits_list:
        return 0.0

    loss_mask = mask  # (B, T)
    docs_mask = input_mask["docs_mask"]  # (num_docs, raw_doc_len)
    num_docs, raw_doc_len = docs_mask.shape
    doc_len = aux_data.get("effective_doc_len", raw_doc_len)

    # Validity: which flat memory slots are non-padding
    effective_mem_mask = aux_data.get("effective_mem_mask", None)
    slot_valid = effective_mem_mask if effective_mem_mask is not None else docs_mask.reshape(-1)

    B = loss_mask.shape[0]

    # Positive doc setup.
    # In grad-accum mode, _mini_step slices pos_doc_mask to [mini_bs, docs_per_query] and
    # passes mini_batch_offset=i so we can compute global query indices correctly.
    if input_mask is not None and "pos_doc_mask" in input_mask:
        pos_mask_local = input_mask["pos_doc_mask"]  # (B, docs_per_query)
        docs_per_query = pos_mask_local.shape[1]
        mini_batch_offset = input_mask.get("mini_batch_offset", 0)
        # global_query_idx[b] = global index of this mini-batch query
        global_query_idx = mini_batch_offset * B + jnp.arange(B)  # [B]
    else:
        pos_mask_local = None
        docs_per_query = 1
        global_query_idx = jnp.arange(B)

    # Per-layer positive-slot augmentation data (only present for two-pass lookup).
    mem_pos_indices_list = aux_data.get("mem_pos_indices")   # list of [B, P] or None
    mem_pos_logits_list = aux_data.get("mem_pos_logits")     # list of [B, N, T, P] or None

    total_loss = 0.0

    for layer_idx, (top_k_indices, top_k_logits) in enumerate(zip(top_k_indices_list, top_k_logits_list)):
        # Augment the top-k pool with logits for all doc slots computed in pass-2.
        # This ensures the contrastive loss always has a positive in the comparison set,
        # even when pass-1 retrieval missed all positive doc tokens.
        if (mem_pos_logits_list and mem_pos_indices_list
                and layer_idx < len(mem_pos_logits_list)):
            pos_logits_i  = mem_pos_logits_list[layer_idx]   # [B, N, T, P]
            pos_indices_i = mem_pos_indices_list[layer_idx]  # [B, N, T, P] — pre-expanded
            # Both arrays already carry the same sharding as top_k_* thanks to
            # with_sharding_constraint applied in mem_lookup_two_pass.
            top_k_indices = jnp.concatenate([top_k_indices, pos_indices_i],  axis=-1)
            top_k_logits  = jnp.concatenate([top_k_logits,  pos_logits_i],  axis=-1)
        # Shape [B, N, T, K] where K may be enlarged by P positive-slot entries above.
        _, N, T, K = top_k_logits.shape

        top_k_doc_ids = top_k_indices // doc_len  # [B, N, T, K]

        # --- Validity gather ---
        # Use .at[].get() with explicit out_sharding to satisfy JAX's mesh constraints.
        validity_k = slot_valid.at[top_k_indices].get(
            out_sharding=P('data', 'model', None, None))  # [B, N, T, K]

        # --- Positive mask (no 2D gather — pure arithmetic + take_along_axis) ---
        # (1) Does this doc belong to the global query for batch item b?
        top_k_doc_owner = top_k_doc_ids // docs_per_query  # [B, N, T, K]
        is_mine = (top_k_doc_owner == global_query_idx[:, None, None, None])  # [B, N, T, K]

        if pos_mask_local is not None:
            # (2) Is this sub-doc marked positive in pos_mask_local?
            # top_k_sub ∈ [0, docs_per_query) — small second dim, safe to broadcast+gather.
            top_k_sub = (top_k_doc_ids % docs_per_query).astype(jnp.int32)  # [B, N, T, K]
            # Expand pos_mask_local to [B, N, T, docs_per_query] then gather along last axis.
            pos_mask_exp = jnp.broadcast_to(
                pos_mask_local[:, None, None, :],
                (B, N, T, docs_per_query),
            )  # [B, N, T, docs_per_query] — zero-copy broadcast
            pos_sub_k = jnp.take_along_axis(pos_mask_exp, top_k_sub, axis=-1) > 0  # [B, N, T, K]
            pos_k = is_mine & pos_sub_k
        else:
            pos_k = is_mine

        scaled_logits = top_k_logits / temperature  # [B, N, T, K]

        # Scan over T to keep activation footprint small.
        logits_T    = jnp.moveaxis(scaled_logits,      2, 0)  # [T, B, N, K]
        validity_T  = jnp.moveaxis(validity_k,         2, 0)  # [T, B, N, K]
        pos_T       = jnp.moveaxis(pos_k & validity_k, 2, 0)  # [T, B, N, K]
        loss_mask_T = jnp.moveaxis(loss_mask,          1, 0)  # [T, B]

        def scan_body(carry, x):
            logits_t, validity_t, pos_t, loss_mask_t = x
            log_z_t   = jax.nn.logsumexp(logits_t, axis=-1, where=validity_t)  # [B, N]
            log_pos_t = jax.nn.logsumexp(logits_t, axis=-1, where=pos_t)       # [B, N]
            log_z_t   = jnp.where(jnp.isfinite(log_z_t),   log_z_t,   0.0)
            log_pos_t = jnp.where(jnp.isfinite(log_pos_t), log_pos_t, 0.0)
            layer_loss_t = log_z_t - log_pos_t  # [B, N], >= 0
            carry = carry + jnp.sum(layer_loss_t * loss_mask_t[:, None])
            return carry, None

        loss_sum, _ = jax.lax.scan(
            scan_body,
            init=jnp.zeros((), dtype=jnp.float32),
            xs=(logits_T, validity_T, pos_T, loss_mask_T),
        )

        total_loss = total_loss + loss_sum / (jnp.sum(loss_mask) * N + 1e-9)

    return total_loss / len(top_k_indices_list)
