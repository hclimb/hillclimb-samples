import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from .registry import register_aux_loss

@register_aux_loss("doc_access_consistency")
def compute_doc_access_consistency(aux_data, mask, input_mask, inputs, **kwargs):
    """
    Measures how consistently the top-k retrieved indices point to the
    same *positive* document for each query token.

    Supports multiple docs per batch item via pos_doc_mask.
    Agreement is computed over valid (non-padding, active-query) retrieval pairs.
    """
    mem_top_k_indices_list = aux_data.get("mem_top_k_indices")
    if not mem_top_k_indices_list or len(mem_top_k_indices_list) == 0:
        return 0.0
    
    loss_mask = mask
    docs_mask = input_mask["docs_mask"]   # (num_docs_total, raw_doc_len)
    num_docs_total, raw_doc_len = docs_mask.shape
    
    # Use effective doc length after conv (if conv was applied), otherwise raw_doc_len
    doc_len = aux_data.get("effective_doc_len", raw_doc_len)
    
    # When conv is active, use the post-conv mem_mask for validity checks
    effective_mem_mask = aux_data.get("effective_mem_mask", None)
    if effective_mem_mask is not None:
        validity_mask = effective_mem_mask  # already flat: (total_mem_vectors,)
    else:
        validity_mask = docs_mask.flatten()
    
    total_var = 0.0
    
    for indices in mem_top_k_indices_list:
        # indices shape: (batch_size, num_heads, seq_len, top_k)
        B, H, S, K = indices.shape

        doc_ids = indices // doc_len  # (B, H, S, K)
        
        # Check validity (not padding doc token)
        is_valid_token = validity_mask.at[indices].get(out_sharding=P('data', 'model', None, None)) == 1
        
        # Check if query is active
        is_active_query = loss_mask[:, None, :, None].astype(jnp.bool_)
        
        # Combined validity: valid token AND active query
        is_valid = is_valid_token & is_active_query  # (B, H, S, K)
        
        # --- ID-INVARIANT CONSISTENCY LOGIC ---
        # 1. Compute pairwise equality: (B, H, S, K, K)
        # Does index i point to the same document as index j?
        same_doc = (doc_ids[..., :, None] == doc_ids[..., None, :])
        
        # 2. Compute pairwise validity mask: (B, H, S, K, K)
        # Only count pairs where BOTH indices are valid tokens
        valid_pairs = (is_valid[..., :, None] & is_valid[..., None, :])
        
        # 3. Calculate Agreement Score
        # Proportion of valid pairs that agree on the document ID
        num_valid_pairs = jnp.sum(valid_pairs, axis=(-1, -2))
        num_same_doc = jnp.sum(same_doc.astype(jnp.float32) * valid_pairs, axis=(-1, -2))
        
        # agreement is 1.0 if all agree, 1/K if all different, 0 if no valid pairs
        # Use 1.0 as the default for empty/invalid queries to avoid penalizing them
        agreement = jnp.where(num_valid_pairs > 0, num_same_doc / num_valid_pairs, 1.0)

        total_var += jnp.nan_to_num(jnp.mean(agreement, where=is_active_query.squeeze(-1)), nan=0.0)
        
    return total_var / len(mem_top_k_indices_list)