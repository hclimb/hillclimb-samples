import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from .registry import register_aux_loss

@register_aux_loss("doc_access_loss")
def compute_doc_access_loss(aux_data, mask, input_mask, inputs, temperature=1.0, **kwargs):
    """
    Computes documentation access loss with Global In-Batch Negatives.
    
    Guarantees:
    1. Loss >= 0 (Numerically stable log-ratio).
    2. No cross-query contamination (Each query has its own specific positives).
    3. In-batch negatives (Queries compete against all docs in the batch).
    """
    mem_scores_list = aux_data.get("mem_scores")
    if not mem_scores_list or len(mem_scores_list) == 0:
        return 0.0
    
    loss_mask = mask # (B, T)
    docs_mask = input_mask["docs_mask"]   # (num_docs, raw_doc_len)
    num_docs, raw_doc_len = docs_mask.shape
    doc_len = aux_data.get("effective_doc_len", raw_doc_len)
    
    # 1. Build per-token validity mask (Padding Mask)
    effective_mem_mask = aux_data.get("effective_mem_mask", None)
    if effective_mem_mask is not None:
        doc_token_mask = effective_mem_mask.reshape(num_docs, doc_len)
    else:
        doc_token_mask = docs_mask

    # mask_expanded: (1, 1, 1, num_docs, doc_len) - True for non-padding tokens
    mask_expanded = (doc_token_mask[None, None, None, :, :] > 0)

    total_loss = 0.0
    
    for scores_item in mem_scores_list:
        # Handle shape unpacking
        if isinstance(scores_item, tuple) and len(scores_item) == 2:
            s1, s2 = scores_item
            B, T, N, _ = s1.shape
            logits = (s1[..., :, None] + s2[..., None, :]).reshape(B, T, N, -1)
        else:
            logits = scores_item[0] if isinstance(scores_item, tuple) else scores_item
            B, T, N, M = logits.shape

        # 2. Scale and Reshape: (B, T, N, num_docs, doc_len)
        scaled_logits = (logits / temperature).reshape(B, T, N, num_docs, doc_len)
        
        # 3. Create the Global Positive Mask (B, num_docs)
        # Each query i only treats docs in its own block as potential positives
        if input_mask is not None and "pos_doc_mask" in input_mask:
            pos_mask_local = input_mask["pos_doc_mask"] # (B, docs_per_query)
            B_local, M_local = pos_mask_local.shape
            block_eye = jnp.eye(B, dtype=pos_mask_local.dtype)
            # Diagonal gating: Query i only sees Positives in Block i
            pos_doc_mask_global = (block_eye[:, :, None] * pos_mask_local[None, :, :]).reshape(B, num_docs)
        else:
            # Fallback: Assume doc i is positive for query i (Standard Diagonal)
            pos_doc_mask_global = jnp.eye(B, num_docs, dtype=jnp.bool_)

        # pos_mask_broad: (B, 1, 1, num_docs, 1)
        pos_mask_broad = pos_doc_mask_global[:, None, None, :, None] > 0

        # 4. Precompute T-independent masks at (B, N, num_docs*doc_len)
        # mask_expanded: (1,1,1,num_docs,doc_len) and pos_mask_broad: (B,1,1,num_docs,1) have no T axis.
        # Broadcasting them to scaled_logits.shape was the source of the 16 GB OOM.
        total_slots = num_docs * doc_len

        flat_token_mask_2d = jnp.broadcast_to(
            mask_expanded.reshape(1, 1, total_slots),
            (B, N, total_slots),
        )  # (B, N, total_slots) — zero-copy

        flat_pos_mask_2d = jnp.broadcast_to(
            (pos_mask_broad & mask_expanded).reshape(B, 1, total_slots),
            (B, N, total_slots),
        )  # (B, N, total_slots) — zero-copy

        # 5. Prepare T-first layout for scan
        logits_T_first = jnp.moveaxis(
            scaled_logits.reshape(B, T, N, total_slots), 1, 0
        )  # (T, B, N, total_slots)
        loss_mask_T_first = jnp.moveaxis(loss_mask, 1, 0)  # (T, B)

        # 6. Scan over T: one time step at a time — avoids materializing (B,T,N,slots)
        def scan_body(carry, x):
            logits_t, loss_mask_t = x  # (B, N, total_slots), (B,)

            log_z_t = jax.nn.logsumexp(
                logits_t, axis=-1, where=flat_token_mask_2d,
            )  # (B, N), -inf where mask is all-False
            log_pos_t = jax.nn.logsumexp(
                logits_t, axis=-1, where=flat_pos_mask_2d,
            )  # (B, N), -inf where mask is all-False

            # Guard against (-inf)-(-inf)=nan; loss_mask_t zeros out these positions anyway.
            log_z_t = jnp.where(jnp.isfinite(log_z_t), log_z_t, 0.0)
            log_pos_t = jnp.where(jnp.isfinite(log_pos_t), log_pos_t, 0.0)

            layer_loss_t = log_z_t - log_pos_t  # (B, N), >= 0
            carry = carry + jnp.sum(layer_loss_t * loss_mask_t[:, None])
            return carry, None

        loss_sum, _ = jax.lax.scan(
            scan_body,
            init=jnp.zeros((), dtype=jnp.float32),
            xs=(logits_T_first, loss_mask_T_first),
        )

        # 7. Normalize
        total_loss += loss_sum / (jnp.sum(loss_mask) * N + 1e-9)
        
    return total_loss / len(mem_scores_list)