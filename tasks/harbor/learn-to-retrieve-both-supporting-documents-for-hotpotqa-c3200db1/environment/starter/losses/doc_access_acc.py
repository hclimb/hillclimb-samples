import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from .registry import register_aux_loss


@register_aux_loss("doc_access_acc")
def compute_doc_access_acc(aux_data, mask, input_mask, inputs, **kwargs):
    """
    Computes documentation access accuracy.
    
    For each retrieved memory index (top_k), checks if it belongs to a
    *positive* document for that batch item and is not a padding token.

    Supports multiple docs per batch item via pos_doc_mask.
    """
    mem_top_k_indices_list = aux_data.get("mem_top_k_indices")
    if not mem_top_k_indices_list or len(mem_top_k_indices_list) == 0:
        return 0.0
    
    loss_mask = mask
    docs_mask = input_mask["docs_mask"]   # (num_docs_total, raw_doc_len)
    num_docs_total, raw_doc_len = docs_mask.shape
    
    # Use effective doc length after conv (if conv was applied), otherwise raw doc_len
    doc_len = aux_data.get("effective_doc_len", raw_doc_len)
    
    # Flatten docs_mask for efficient indexing
    effective_mem_mask = aux_data.get("effective_mem_mask", None)
    if effective_mem_mask is not None:
        validity_mask = effective_mem_mask  # already flat: (total_mem_vectors,)
    else:
        validity_mask = docs_mask.flatten()
    
    total_corr_keys = 0.0
    total_keys = 0.0
    
    for indices in mem_top_k_indices_list:
        # indices shape: (batch_size, num_heads, seq_len, top_k)
        B, H, S, K = indices.shape

        # 1. Map each retrieved index to its document id
        retrieved_doc_ids = indices // doc_len  # (B, H, S, K) values in [0, num_docs_total)

        # 2. Build positive doc mask per batch item → (B, num_docs_total)
        if input_mask is not None and "pos_doc_mask" in input_mask:
            pos_mask_local = input_mask["pos_doc_mask"]  # (B, docs_per_query)
        else:
            pos_mask_local = jnp.ones((B, 1), dtype=jnp.float32)

        B_local, M_local = pos_mask_local.shape
        # block_eye maps batch item i to its own doc block
        block_eye = jnp.eye(B_local, dtype=pos_mask_local.dtype)  # (B, B)
        pos_doc_mask_global = (block_eye[:, :, None] * pos_mask_local[None, :, :]).reshape(B_local, num_docs_total)
        # pos_doc_mask_global: (B, num_docs_total) — 1 for positive docs, 0 otherwise

        # 3. Look up whether retrieved doc_id is positive for this batch item
        # Use advanced indexing: for each (b, h, s, k), check pos_doc_mask_global[b, retrieved_doc_ids[b,h,s,k]]
        batch_idx = jnp.arange(B)[:, None, None, None]  # (B, 1, 1, 1)
        is_correct_doc = pos_doc_mask_global.at[batch_idx, retrieved_doc_ids].get(out_sharding=P('data', 'model', None, None)) > 0  # (B, H, S, K)

        # 4. Check if retrieved index is a valid doc token (not padding)
        is_valid_token = validity_mask.at[indices].get(out_sharding=P('data', 'model', None, None)) == 1
        
        # 5. Check if the query itself is not padding
        is_active_query = loss_mask[:, None, :, None].astype(jnp.bool_)
        
        # Combine conditions
        is_valid = is_valid_token & is_active_query
        correct = (is_correct_doc & is_valid).astype(jnp.float32)
        
        total_corr_keys += jnp.sum(correct)
        total_keys += jnp.sum(is_valid)
        
    return total_corr_keys / total_keys