"""
Auxiliary Loss Registry

Provides a scalable pattern for registering and computing multiple auxiliary losses.
"""

from typing import Dict, Any, Callable, Optional
import jax
import jax.numpy as jnp

# Global registry for auxiliary loss functions
_AUX_LOSS_REGISTRY: Dict[str, Callable] = {}


def register_aux_loss(name: str):
    """
    Decorator to register an auxiliary loss function.
    
    The registered function should have signature:
        fn(aux_data: Dict, mask: Array, **kwargs) -> scalar loss
    
    Args:
        name: Unique name for the loss (used in config)
    
    Example:
        @register_aux_loss("my_loss")
        def compute_my_loss(aux_data, mask, **kwargs):
            ...
            return loss_value
    """
    def decorator(fn: Callable) -> Callable:
        if name in _AUX_LOSS_REGISTRY:
            raise ValueError(f"Auxiliary loss '{name}' is already registered")
        _AUX_LOSS_REGISTRY[name] = fn
        return fn
    return decorator


def get_aux_loss_names() -> list:
    """Return list of all registered auxiliary loss names."""
    return list(_AUX_LOSS_REGISTRY.keys())


class AuxLossRegistry:
    """
    Registry for computing auxiliary losses based on config.
    
    Provides methods to:
    - Check which losses are enabled
    - Compute individual or all enabled losses
    - Get weighted sum of all losses
    """
    
    def __init__(self, aux_loss_config: Optional[Dict] = None):
        """
        Initialize the registry with config.
        
        Args:
            aux_loss_config: Dict mapping loss names to their config:
                {
                    "loss_name": {
                        "enabled": bool,
                        "weight": float,
                        ...extra kwargs for the loss fn
                    }
                }
        """
        self.config = aux_loss_config or {}
    
    def is_enabled(self, name: str) -> bool:
        """Check if a specific loss is enabled."""
        if name not in self.config:
            return False
        return self.config[name].get("enabled", False)
    
    def get_weight(self, name: str) -> float:
        """Get the weight for a specific loss."""
        if name not in self.config:
            return 0.0
        return self.config[name].get("weight", 1.0)
    
    def get_enabled_losses(self) -> list:
        """Return list of enabled loss names."""
        return [name for name in self.config if self.is_enabled(name)]
    
    def has_enabled_losses(self) -> bool:
        """Check if any auxiliary loss is enabled."""
        return len(self.get_enabled_losses()) > 0


def compute_aux_losses(aux_data, mask, input_mask, inputs, aux_loss_config):
    """
    Compute all enabled auxiliary losses.
    
    Args:
        aux_data: Dictionary containing auxiliary data from model forward pass.
                  Keys should match what each registered loss expects.
                  Can be None if no aux data is collected.
        mask: Loss mask tensor [B, T] (1 for valid, 0 for padding)
        aux_loss_config: Config dict for aux losses (same format as AuxLossRegistry)
    
    Returns:
        Dictionary with:
            - "total": weighted sum of all losses
            - "losses": dict mapping loss name -> unweighted loss value
            - "weighted_losses": dict mapping loss name -> weighted loss value
    """
    result = {
        "total": 0.0,
        "losses": {},
        "weighted_losses": {},
    }
    
    if aux_loss_config is None or aux_data is None:
        return result
    
    registry = AuxLossRegistry(aux_loss_config)
    
    if not registry.has_enabled_losses():
        return result
    
    total_loss = 0.0
    
    for name in registry.get_enabled_losses():
        if name not in _AUX_LOSS_REGISTRY:
            # Skip unknown losses (could add warning here)
            continue
        
        loss_fn = _AUX_LOSS_REGISTRY[name]
        weight = registry.get_weight(name)
        
        # Get extra kwargs for this loss from config (excluding 'enabled' and 'weight')
        loss_cfg = aux_loss_config.get(name, {})
        extra_kwargs = {k: v for k, v in loss_cfg.items() if k not in ("enabled", "weight")}
        
        # Compute the loss. A metric may return a scalar (a normal loss/metric) or a dict of
        # named scalars (e.g. per-layer diagnostic series) — the latter are logged as
        # train/<name>/<subkey>. Static keys keep this JIT-safe.
        loss_value = loss_fn(aux_data, mask, input_mask, inputs, **extra_kwargs)

        items = loss_value.items() if isinstance(loss_value, dict) else [(None, loss_value)]
        for sub, sv in items:
            # Handle potential NaNs in a JIT-compatible way
            is_nan = jnp.isnan(sv).any()
            sv = jnp.where(is_nan, jnp.zeros_like(sv), sv)
            key = name if sub is None else f"{name}/{sub}"
            result["losses"][key] = sv
            result["weighted_losses"][key] = sv * weight
            # weight-0 metrics are pure diagnostics — stop_gradient so they contribute exactly
            # zero to the gradient, not 0*grad. Otherwise a diagnostic whose value is finite but
            # whose GRADIENT is non-finite (e.g. ‖v‖ at v=0 under zero-init mem_o_proj) makes
            # 0 * NaN = NaN and poisons the whole grad (every optimizer step then gets skipped).
            contrib = jax.lax.stop_gradient(sv) if weight == 0 else sv
            total_loss = total_loss + contrib * weight
    
    result["total"] = total_loss
    return result
