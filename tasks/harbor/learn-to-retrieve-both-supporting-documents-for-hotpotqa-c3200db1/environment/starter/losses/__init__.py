"""
Auxiliary Losses Module

This module provides a scalable registry-based system for auxiliary losses.
Each loss function is registered with a name and can be enabled/weighted via config.

Usage in config (trainer.yaml):
    aux_losses:
        mem_uniform_kl:
            enabled: true
            weight: 0.1

To add a new loss:
1. Create a new file in losses/ (e.g., my_loss.py)
2. Use @register_aux_loss("my_loss") decorator
3. Import it here
"""

from .registry import (
    AuxLossRegistry,
    register_aux_loss,
    compute_aux_losses,
    get_aux_loss_names,
)
from .mem_uniform_kl import compute_mem_uniform_kl_loss
from .doc_access_acc import compute_doc_access_acc
from .doc_access_consistency import compute_doc_access_consistency
from .doc_access_loss import compute_doc_access_loss
from .distillation_loss import compute_distillation_loss
from .doc_access_top_k_loss import compute_doc_access_top_k_loss
from .doc_access_per_query_loss import compute_doc_access_per_query_loss
from .msa_route_loss import compute_msa_route_loss
from .msa_route_acc import compute_msa_route_acc
from .mem_telemetry import (
    compute_mem_write_norm,
    compute_mem_write_ratio,
    compute_mem_topk_entropy,
    compute_mem_effective_slots,
    compute_mem_top1_weight,
    compute_mem_o_proj_norm,
    compute_mem_boundary_straddle,
    compute_mem_pos_weight_mass,
    compute_mem_hit_rate,
    compute_mem_head_query_cos,
    compute_mem_cross_layer_cos,
    compute_mem_kv_cos,
    compute_mem_value_anisotropy,
)

__all__ = [
    "AuxLossRegistry",
    "register_aux_loss",
    "compute_aux_losses",
    "get_aux_loss_names",
    "compute_mem_uniform_kl_loss",
    "compute_doc_access_acc",
    "compute_doc_access_consistency",
    "compute_doc_access_loss",
    "compute_distillation_loss",
    "compute_doc_access_top_k_loss",
    "compute_doc_access_per_query_loss",
    "compute_msa_route_loss",
    "compute_msa_route_acc",
    "compute_mem_write_norm",
    "compute_mem_write_ratio",
    "compute_mem_topk_entropy",
    "compute_mem_effective_slots",
    "compute_mem_top1_weight",
    "compute_mem_o_proj_norm",
    "compute_mem_boundary_straddle",
    "compute_mem_pos_weight_mass",
    "compute_mem_hit_rate",
    "compute_mem_head_query_cos",
    "compute_mem_cross_layer_cos",
    "compute_mem_kv_cos",
    "compute_mem_value_anisotropy",
]
