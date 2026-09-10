"""
Model Output Classes

Provides a consistent output format for all model forward passes.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import jax.numpy as jnp


@dataclass
class ModelOutput:
    """
    Standard output format for all model forward passes.
    
    Attributes:
        logits: Output logits [B, T, V]
        kv: KV cache dict (optional, None if not requested)
        aux: Auxiliary data dict for aux losses (optional, None if not collected)
    """
    logits: jnp.ndarray
    kv: Optional[Dict] = None
    aux: Optional[Dict[str, Any]] = None
    
    def __iter__(self):
        """Allow unpacking for backward compatibility."""
        yield self.logits
        if self.kv is not None:
            yield self.kv
    
    def get_logits(self) -> jnp.ndarray:
        """Get logits tensor."""
        return self.logits
    
    def get_kv(self) -> Optional[Dict]:
        """Get KV cache if available."""
        return self.kv
    
    def get_aux(self) -> Optional[Dict[str, Any]]:
        """Get auxiliary data if available."""
        return self.aux
    
    def has_aux(self) -> bool:
        """Check if auxiliary data is present."""
        return self.aux is not None and len(self.aux) > 0
