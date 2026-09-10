"""
Memory Uniform KL Loss

Encourages uniform access to memory keys in memory layers by computing
KL divergence between the access distribution and a uniform distribution.
Works with both product keys and standard keys.
"""

import jax
import jax.numpy as jnp
from .registry import register_aux_loss


@register_aux_loss("mem_uniform_kl")
def compute_mem_uniform_kl_loss(aux_data, mask, input_mask, inputs, **kwargs):
    """
    Compute KL divergence between memory key access distribution and uniform distribution.
    
    Args:
        aux_data: Dict containing "mem_scores" - list of score tuples from memory layers.
                  Each element is either:
                  - A tuple (scores1, scores2) for product keys
                  - A single scores tensor for non-product-key layers
                  Each scores tensor has shape [B, T, N, n_keys]
        mask: [B, T] loss mask tensor (1 for valid positions, 0 for padding)
        **kwargs: Additional config options (unused for now)
    
    Returns:
        KL divergence loss (scalar) encouraging uniform access to memory keys
    """
    mem_scores_list = aux_data.get("mem_scores")
    
    if mem_scores_list is None or len(mem_scores_list) == 0:
        return 0.0
    
    total_kl = 0.0
    num_layers = 0
    
    for scores_item in mem_scores_list:
        # Handle both product keys (tuple) and standard keys (single tensor)
        if isinstance(scores_item, tuple):
            # Product keys: (scores1, scores2)
            scores_list = list(scores_item)
        else:
            # Single scores tensor (non-product-key)
            scores_list = [scores_item]
        
        for scores in scores_list:
            # scores: [B, T, N, n_keys]
            n_keys = scores.shape[-1]
            
            # Expand mask to match scores shape: [B, T] -> [B, T, 1, 1]
            mask_expanded = mask[:, :, None, None]
            
            # Apply softmax to get probabilities
            probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)  # [B, T, N, n_keys]
            
            # Masked average across B, T, N dimensions to get access distribution
            masked_probs = probs * mask_expanded
            
            # Sum across B, T, N and normalize to get average access distribution
            # Shape: [n_keys]
            sum_count = (mask_expanded.sum() * probs.shape[2]) + 1e-9  # B*T*N valid positions
            avg_probs = masked_probs.sum(axis=(0, 1, 2)) / sum_count
            
            # Renormalize to ensure they sum to 1
            avg_probs = avg_probs / (avg_probs.sum() + 1e-9)
            
            # Uniform distribution
            uniform = jnp.ones(n_keys) / n_keys
            
            # KL divergence: KL(avg_probs || uniform)
            eps = 1e-9
            kl = jnp.sum(avg_probs * (jnp.log(avg_probs + eps) - jnp.log(uniform + eps)))
            
            total_kl = total_kl + kl
            num_layers += 1
    
    if num_layers == 0:
        return 0.0
    
    return total_kl / num_layers
