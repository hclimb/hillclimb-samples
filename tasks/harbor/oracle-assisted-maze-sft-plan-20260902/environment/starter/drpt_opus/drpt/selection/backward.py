"""
Autograd functions for gradient-based data curation and compression.

This module provides three distinct autograd Functions for Linear layers:
- CompressedLinearBackward: Pure gradient compression (no curation)
- LayerWiseSubsetLinearBackward: Per-layer curation, single-pass
- GlobalSubsetLinearBackward: Score accumulation, two-pass

And one for non-linear layers in merged-batch one-pass mode:
- TrainOnlyRMSNormBackward: RMSNorm that computes grad_weight from train slice only

Each function routes to specific handlers based on CompressionMode:
- NONE: Full gradients for scoring and updates
- SCORE_ONLY: Compressed scoring, full gradient updates
- FULL: Compressed scoring and gradient updates (MeSO)
"""

from __future__ import annotations

import math
import weakref
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Tuple
    from torch import Tensor
    from .state import LayerWiseSubsetState, GlobalSubsetState
    from ..hook import GradientHook
    from ..compressor import Compressor

import torch
import torch.nn.functional as F
from torch.autograd import Function

from ..compression_mode import CompressionMode
from .utils import (
    augment_input_for_bias,
    split_train_val_batch,
    compute_scores_and_similarity,
    compute_scores_full_ghost,
    compute_scores_direct_materialization,
    compute_selected_gradients,
    compute_total_gradient,
)


# =============================================================================
# Helper functions for backward passes
# =============================================================================

def _get_val_components(
    hook_manager: GradientHook,
    layer_idx: int,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """Get cached target gradients, optionally staging one layer to a device."""
    val_cache = hook_manager._val_cache

    val_grad_output, val_input = val_cache.get_factorized(
        layer_idx, device=device
    )
    if val_grad_output is not None and val_input is not None:
        return val_grad_output, val_input, None, None

    val_grad_total = val_cache.get_full(layer_idx, device=device)
    val_bias_grad = val_cache.get_bias_grad(layer_idx, device=device)
    return None, None, val_grad_total, val_bias_grad


def _compute_scale_factor(state: LayerWiseSubsetState, selected_indices: Tensor) -> Tensor:
    """Compute token-based gradient scale factor for selected samples."""
    return state._compute_scale_factor(selected_indices)


def _target_standalone_scale(state: "LayerWiseSubsetState") -> Tensor:
    """Restore a merged-batch target gradient to standalone normalization."""
    if getattr(state, "_use_stored_val", False):
        return torch.ones((), device=state.device, dtype=torch.float32)
    if state.batch_total_tokens_tensor is None or state.train_total_tokens_tensor is None:
        raise RuntimeError("Token counts must be set before advanced solver scoring")
    val_tokens = state.batch_total_tokens_tensor - state.train_total_tokens_tensor
    if not bool((val_tokens > 0).detach().cpu()):
        raise RuntimeError("No valid target tokens are available for selection")
    return (
        state.batch_total_tokens_tensor.float() / val_tokens.float()
    )


def _candidate_base_tokens(state: "LayerWiseSubsetState") -> Tensor:
    if state.batch_total_tokens_tensor is None or state.tokens_per_sample is None:
        raise RuntimeError("Token counts must be set before advanced solver scoring")
    return state.batch_total_tokens_tensor.float()


def _linear_target_gradients(
    _train_dtype: torch.dtype,
    state: "LayerWiseSubsetState",
    val_grad_output: Optional[Tensor],
    val_input: Optional[Tensor],
    val_grad_total: Optional[Tensor],
    val_bias_grad: Optional[Tensor],
    has_bias: bool,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Build raw (unmapped) target weight/bias gradients in standalone scale."""
    if val_grad_total is None:
        if val_grad_output is None or val_input is None:
            raise ValueError("Advanced solver requires a target gradient")
        # Promote retained factors before contraction.  The default SFT model
        # is bf16, and casting only after the einsum can erase small target
        # modes that the soft objective and spectral SVD are meant to see.
        val_grad_total = compute_total_gradient(
            val_grad_output.detach().float(), val_input.detach().float()
        )
    target_scale = _target_standalone_scale(state).to(val_grad_total.device)
    target_weight = val_grad_total.detach().float() * target_scale

    target_bias = None
    if has_bias:
        if val_bias_grad is not None:
            target_bias = val_bias_grad.detach().float()
        elif val_grad_output is not None:
            dims = tuple(range(val_grad_output.dim() - 1))
            target_bias = val_grad_output.detach().float().sum(dim=dims)
        if target_bias is not None:
            target_bias = target_bias * target_scale
    return target_weight, target_bias


def _parameter_optimizer_kind(
    hook_manager: "GradientHook", param: Tensor, default: str = "adamw"
) -> str:
    optimizer = getattr(hook_manager, "optimizer", None)
    if optimizer is not None and hasattr(optimizer, "get_param_optimizer_kind"):
        return str(optimizer.get_param_optimizer_kind(param)).lower()
    optimizer_type = str(
        getattr(hook_manager, "optimizer_aware_config", {}).get(
            "optimizer_type", default
        )
    ).lower()
    return "muon" if optimizer_type in ("hybrid", "both", "muon") else "adamw"


def _muon_surrogate_includes_adamw_scores(state: "LayerWiseSubsetState") -> bool:
    """Whether AdamW-managed parameters participate in the spectral criterion."""
    config = getattr(state, "solver_config", {}) or {}
    return bool(config.get("muon_surrogate_include_adamw_scores", True))


def _muon_surrogate_is_saturated(state: "LayerWiseSubsetState") -> bool:
    """Whether the spectral criterion uses concave mode saturation."""
    config = getattr(state, "solver_config", {}) or {}
    return bool(config.get("muon_surrogate_saturation", False))


def _muon_surrogate_mode_weights(
    state: "LayerWiseSubsetState", beta: Tensor
) -> Tensor:
    """Return alpha_r for the configured uniform or singular-value variant."""
    config = getattr(state, "solver_config", {}) or {}
    weighting = str(
        config.get("muon_surrogate_mode_weighting", "uniform")
    ).lower()
    scale = float(config.get("muon_surrogate_alpha", 1.0))
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("muon_surrogate_alpha must be finite and non-negative")
    if weighting in ("uniform", "one", "ones"):
        return beta.new_full(beta.shape, scale)
    if weighting in ("singular_value", "singular_values", "beta"):
        return beta * scale
    raise ValueError(
        "muon_surrogate_mode_weighting must be 'uniform' or 'singular_value'"
    )


def _adamw_candidate_transform(
    hook_manager: "GradientHook", param: Tensor, candidate: Tensor
) -> Tensor:
    """Apply the same frozen AdamW linear mapper used by OptA scoring."""
    from .optimizer_aware import _get_adamw_inv_rms, _group_lr

    config = getattr(hook_manager, "optimizer_aware_config", {}) or {}
    inv_rms = _get_adamw_inv_rms(
        hook_manager, param, float(config.get("adam_eps", 1e-8))
    )
    if inv_rms is None:
        # Keep the update-map scale consistent before AdamW has initialized its
        # second-moment state.  In particular, Hybrid soft objectives combine
        # this fallback with Muon maps that already include their group LR.
        return candidate * _group_lr(hook_manager, param)
    return (
        candidate
        * inv_rms.detach().to(device=candidate.device, dtype=candidate.dtype)
        * _group_lr(hook_manager, param)
    )


def _adamw_bias_scores_standalone(
    hook_manager: "GradientHook",
    state: "LayerWiseSubsetState",
    module,
    train_grad_output: Tensor,
    target_bias: Optional[Tensor],
) -> Tensor:
    if module.bias is None or target_bias is None:
        return torch.zeros(
            train_grad_output.shape[0],
            device=train_grad_output.device,
            dtype=torch.float32,
        )
    train_bias = train_grad_output.detach().float()
    if train_bias.dim() == 3:
        train_bias = train_bias.sum(dim=1)
    probe = _adamw_candidate_transform(
        hook_manager, module.bias, target_bias.detach().float()
    )
    scores = train_bias @ probe
    if not getattr(state, "_use_stored_val", False):
        scores = scores * (
            state.batch_total_tokens_tensor.float()
            / state.train_total_tokens_tensor.float()
        )
    return scores


def _muon_candidate_transform(
    hook_manager: "GradientHook",
    state: "LayerWiseSubsetState",
    param: Tensor,
    candidate: Tensor,
) -> Tensor:
    from .advanced_solvers import muon_live_candidate_transform

    optimizer = getattr(hook_manager, "optimizer", None)
    if optimizer is not None and hasattr(optimizer, "get_param_state"):
        opt_state = optimizer.get_param_state(param) or {}
    else:
        opt_state = optimizer.state.get(param, {}) if optimizer is not None else {}
    group = getattr(hook_manager, "_optimizer_group_by_param_id", {}).get(id(param), {})
    config = getattr(hook_manager, "optimizer_aware_config", {}) or {}
    solver_config = getattr(state, "solver_config", {}) or {}
    use_live = bool(solver_config.get("soft_weighting_use_optimizer_state", True))
    momentum_buffer = opt_state.get("momentum_buffer") if use_live else None
    return muon_live_candidate_transform(
        candidate,
        momentum_buffer=momentum_buffer,
        optimizer_dtype=param.dtype,
        momentum=float(group.get("momentum", config.get("muon_momentum", 0.95))),
        nesterov=bool(group.get("nesterov", config.get("muon_nesterov", True))),
        ns_steps=int(group.get("muon_ns_steps", config.get("muon_steps", 5))),
        eps=float(group.get("muon_eps", config.get("muon_eps", 1e-7))),
        lr=float(group.get("lr", 1.0)),
        shape_lr_scale=bool(group.get(
            "muon_lr_shape_scale", config.get("muon_lr_shape_scale", True)
        )),
        adjust_lr_fn=group.get(
            "adjust_lr_fn", config.get("muon_adjust_lr_fn", "original")
        ),
    )


def _make_soft_linear_objective(
    hook_manager: "GradientHook",
    state: "LayerWiseSubsetState",
    layer_idx: int,
    train_grad_output: Tensor,
    train_input: Tensor,
    target_weight: Tensor,
    target_bias: Optional[Tensor],
    has_bias: bool,
):
    """Create a detached factor objective whose only differentiable input is w."""
    from .advanced_solvers import weighted_linear_gradients, weighted_token_scale

    module = hook_manager._get_module_from_idx(layer_idx)
    go = train_grad_output.detach()
    inp = train_input.detach()
    target_weight = target_weight.detach().float()
    target_bias = target_bias.detach().float() if target_bias is not None else None
    tokens = state.tokens_per_sample.detach().float()
    base_tokens = _candidate_base_tokens(state).detach()
    weight_kind = _parameter_optimizer_kind(hook_manager, module.weight)
    gamma = float(state.solver_config.get("soft_weighting_gamma", 0.0))
    contraction_precision = str(
        state.solver_config.get("soft_replay_precision", "fp32")
    )

    # Identity/AdamW are fixed linear maps.  With the default gamma=0 their
    # exact objective reduces to a scalar score vector, so global soft runs do
    # not retain activation factors or a model-sized target matrix per layer.
    if weight_kind != "muon" and gamma == 0.0:
        target_probe = _adamw_candidate_transform(
            hook_manager, module.weight, target_weight
        )
        per_sample_scores, _ = compute_scores_and_similarity(
            go.float(), inp.float(), None, None, target_probe, False
        )
        if has_bias and target_bias is not None and module.bias is not None:
            bias_probe = _adamw_candidate_transform(
                hook_manager, module.bias, target_bias
            )
            train_bias = go.float().sum(dim=1) if go.dim() == 3 else go.float()
            per_sample_scores = per_sample_scores + (
                train_bias @ bias_probe
            )
        per_sample_scores = per_sample_scores.detach().float()

        def linear_fractional_objective(weights: Tensor):
            scale = weighted_token_scale(weights, tokens, base_tokens)
            return scale * torch.dot(weights, per_sample_scores)

        return linear_fractional_objective

    def objective(weights: Tensor):
        scale = weighted_token_scale(weights, tokens, base_tokens)
        candidate_weight, candidate_bias = weighted_linear_gradients(
            go,
            inp,
            weights,
            scale=scale,
            has_bias=has_bias,
            replay_precision=contraction_precision,
        )
        if weight_kind == "muon":
            update_weight = _muon_candidate_transform(
                hook_manager, state, module.weight, candidate_weight
            )
        else:
            update_weight = _adamw_candidate_transform(
                hook_manager, module.weight, candidate_weight
            )
        alignment = (update_weight * target_weight).sum()
        update_norm_sq = update_weight.square().sum()

        if candidate_bias is not None and target_bias is not None and module.bias is not None:
            update_bias = _adamw_candidate_transform(
                hook_manager, module.bias, candidate_bias
            )
            alignment = alignment + (update_bias * target_bias).sum()
            update_norm_sq = update_norm_sq + update_bias.square().sum()
        return (alignment, update_norm_sq) if gamma != 0.0 else alignment

    return objective


def _soft_reference_linear_scores(
    hook_manager: "GradientHook",
    state: "LayerWiseSubsetState",
    layer_idx: int,
    train_grad_output: Tensor,
    train_input: Tensor,
    val_grad_output: Optional[Tensor],
    val_input: Optional[Tensor],
    val_grad_total: Optional[Tensor],
    val_bias_grad: Optional[Tensor],
    has_bias: bool,
) -> Tensor:
    from .optimizer_aware import compute_optimizer_aware_linear_scores

    scores, _, _, _ = compute_optimizer_aware_linear_scores(
        hook_manager, state, layer_idx,
        train_grad_output, train_input,
        val_grad_output, val_input, val_grad_total, False,
        collect_raw_scores=False,
    )
    scores, _ = _add_bias_scores(
        scores, None, train_grad_output,
        val_grad_output, val_bias_grad, has_bias,
    )
    if not getattr(state, "_use_stored_val", False):
        scores = scores * state.score_correction
    return scores.detach().float()


def _optimize_layer_soft_weights(
    state: "LayerWiseSubsetState",
    layer_idx: int,
    objective,
    reference_scores: Optional[Tensor],
) -> Tensor:
    from .advanced_solvers import optimize_soft_weights

    cfg = state.solver_config
    weights, diagnostics = optimize_soft_weights(
        objective,
        state.train_batch_size,
        state.num_selected,
        device=torch.device(state.device),
        steps=int(cfg.get("soft_weighting_steps", 20)),
        lr=float(cfg.get("soft_weighting_lr", 0.1)),
        tolerance=float(cfg.get("soft_weighting_tol", 1e-5)),
        patience=int(cfg.get("soft_weighting_patience", 3)),
        gamma=float(cfg.get("soft_weighting_gamma", 0.0)),
        constraint=str(
            cfg.get("soft_weighting_constraint", "capped_simplex")
        ),
    )
    return _record_layer_soft_weights(
        state, layer_idx, weights, diagnostics, reference_scores
    )


def _record_layer_soft_weights(
    state: "LayerWiseSubsetState",
    layer_idx: int,
    weights: Tensor,
    diagnostics: dict,
    reference_scores: Optional[Tensor],
) -> Tensor:
    """Record one already-solved Soft row with the scalar path's semantics."""
    state._record_soft_diagnostics(
        f"layer_{layer_idx}", weights, diagnostics, reference_scores,
        layer_idx=layer_idx,
    )
    state._layer_selections.append((layer_idx, state.num_selected))
    if state._record_selections:
        record = {
            "layer_idx": layer_idx,
            "weights": weights.detach().float().cpu().tolist(),
            "num_selected": int(state.num_selected),
            "selection_seed": int(state.derived_seed(layer_idx)),
            "global_step": int(state.global_step),
        }
        if reference_scores is not None:
            record["reference_scores"] = (
                reference_scores.detach().float().cpu().tolist()
            )
        if state.tokens_per_sample is not None:
            record["valid_token_counts"] = (
                state.tokens_per_sample.detach().float().cpu().tolist()
            )
        state._selection_records.append(record)
    return weights.detach()


def _compute_muon_spectral_linear_support(
    hook_manager: "GradientHook",
    state: "LayerWiseSubsetState",
    layer_idx: int,
    train_grad_output: Tensor,
    train_input: Tensor,
    target_weight: Optional[Tensor],
) -> Tuple[Tensor, Tensor, int]:
    """Return projected-ghost support, retained singular values, and rank."""
    from .advanced_solvers import (
        compute_spectral_modes_with_values,
        spectral_linear_mode_support,
    )

    cached_modes = None
    if bool(getattr(state, "windowed_execution", False)):
        cached_modes = state.get_window_spectral_modes(layer_idx)
    if cached_modes is None:
        if target_weight is None:
            raise RuntimeError(
                f"Layer {layer_idx} has no target gradient or cached spectral modes"
            )
        cfg = state.solver_config
        u, v, beta, active_rank = compute_spectral_modes_with_values(
            target_weight.detach().float(),
            rank=int(cfg.get("muon_surrogate_rank", 32)),
            full_svd_max_dim=int(cfg.get("muon_surrogate_full_svd_max_dim", 256)),
            rtol=float(cfg.get("muon_surrogate_rtol", 1e-6)),
            oversample=int(cfg.get("muon_surrogate_oversample", 8)),
            power_iters=int(cfg.get("muon_surrogate_power_iters", 2)),
            seed=state.derived_seed(layer_idx),
        )
        if bool(getattr(state, "windowed_execution", False)):
            state.store_window_spectral_modes(
                layer_idx, u, v, beta, active_rank
            )
    else:
        u, v, beta, active_rank = cached_modes
    state._append_diagnostic(f"spectral/layer_{layer_idx}/active_rank", active_rank)
    if active_rank == 0:
        state._append_diagnostic(f"spectral/layer_{layer_idx}/zero_target", 1.0)
        return torch.zeros(
            (train_grad_output.shape[0], 0),
            device=train_grad_output.device,
            dtype=torch.float32,
        ), beta.to(device=train_grad_output.device), 0
    support = spectral_linear_mode_support(
        train_grad_output.detach().float(),
        train_input.detach().float(),
        u, v,
    )
    # Candidate factors in a merged batch carry the total-batch denominator.
    # Restore the candidate side only; target singular vectors are scale-free.
    if not getattr(state, "_use_stored_val", False):
        support = support * (
            state.batch_total_tokens_tensor.float()
            / state.train_total_tokens_tensor.float()
        )
    return support, beta.to(device=support.device, dtype=support.dtype), active_rank


def _compute_muon_spectral_linear_scores(
    hook_manager: "GradientHook",
    state: "LayerWiseSubsetState",
    layer_idx: int,
    train_grad_output: Tensor,
    train_input: Tensor,
    target_weight: Optional[Tensor],
) -> Tuple[Tensor, int]:
    """Return modular spectral scores with uniform or beta mode weights."""
    support, beta, active_rank = _compute_muon_spectral_linear_support(
        hook_manager,
        state,
        layer_idx,
        train_grad_output,
        train_input,
        target_weight,
    )
    if active_rank == 0:
        return support.new_zeros((support.shape[0],)), 0
    scores = support @ _muon_surrogate_mode_weights(state, beta)
    return scores, active_rank


def _select_muon_saturated_support(
    state: "LayerWiseSubsetState",
    layer_idx: int,
    support: Tensor,
    beta: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Greedily select a layer-local subset under log1p mode saturation."""
    from .advanced_solvers import (
        seeded_spectral_saturation_greedy,
        spectral_saturated_objective,
    )

    alpha = _muon_surrogate_mode_weights(state, beta)
    selected = seeded_spectral_saturation_greedy(
        support,
        state.num_selected,
        alpha=alpha,
        seed=state.derived_seed(layer_idx),
    )
    objective = spectral_saturated_objective(support, selected, alpha=alpha)
    state._append_diagnostic(
        f"spectral/layer_{layer_idx}/saturated_objective", objective
    )
    # Singleton values are useful for diagnostics and selection records, but are
    # not used as an additive replacement for the greedy set objective.
    singleton_scores = torch.log1p(support) @ alpha
    return selected, singleton_scores


def _make_soft_embedding_objective(
    hook_manager: "GradientHook",
    state: "LayerWiseSubsetState",
    layer_idx: int,
    train_grad_output: Tensor,
    train_ids: Tensor,
    target_weight: Tensor,
    num_embeddings: int,
    padding_idx: int,
):
    """Create the AdamW/identity soft objective for an embedding parameter."""
    from .advanced_solvers import weighted_embedding_gradient, weighted_token_scale
    from .utils import compute_embedding_scores

    module = hook_manager._get_module_from_idx(layer_idx)
    target_probe = _adamw_candidate_transform(
        hook_manager, module.weight, target_weight.detach().float()
    )
    go = train_grad_output.detach()
    ids = train_ids.detach()
    per_sample_scores = compute_embedding_scores(
        go.float(), ids, target_probe
    ).detach().float()
    tokens = state.tokens_per_sample.detach().float()
    base_tokens = _candidate_base_tokens(state).detach()
    gamma = float(state.solver_config.get("soft_weighting_gamma", 0.0))

    if gamma == 0.0:
        def linear_fractional_objective(weights: Tensor):
            scale = weighted_token_scale(weights, tokens, base_tokens)
            return scale * torch.dot(weights, per_sample_scores)
        return linear_fractional_objective, per_sample_scores

    def objective(weights: Tensor):
        scale = weighted_token_scale(weights, tokens, base_tokens)
        alignment = scale * torch.dot(weights, per_sample_scores)
        candidate = weighted_embedding_gradient(
            go.float(), ids, weights,
            num_embeddings=num_embeddings,
            padding_idx=None if padding_idx < 0 else padding_idx,
            scale=scale,
        )
        update = _adamw_candidate_transform(hook_manager, module.weight, candidate)
        return alignment, update.square().sum()

    return objective, per_sample_scores


# =============================================================================
# Shared helpers for LayerWiseSubset backward paths
# =============================================================================

def _dispatch_scoring(
    scoring_method: str,
    train_grad_output: "Tensor",
    train_input: "Tensor",
    val_grad_output: "Optional[Tensor]",
    val_input: "Optional[Tensor]",
    val_grad_total: "Optional[Tensor]",
    use_second_order: bool,
    direct_batch_size: int = 0,
    return_materialized: bool = False,
) -> "Tuple[Tensor, Optional[Tensor], Optional[Tensor]]":
    """Dispatch to the appropriate scoring function based on scoring_method.

    Returns:
        (scores, similarity, materialized_grads) where materialized_grads is
        the [B, O*I] per-sample weight gradient tensor when return_materialized=True
        and scoring_method="direct", otherwise None.
    """
    if scoring_method == "direct":
        scores, similarity, G_train = compute_scores_direct_materialization(
            train_grad_output, train_input, val_grad_output, val_input,
            val_grad_total, use_second_order, batch_size=direct_batch_size,
            return_materialized=return_materialized,
        )
        return scores, similarity, G_train
    elif scoring_method == "full_ghost":
        return (*compute_scores_full_ghost(
            train_grad_output, train_input, val_grad_output, val_input,
            val_grad_total, use_second_order
        ), None)
    else:  # "reduced_ghost" (default)
        return (*compute_scores_and_similarity(
            train_grad_output, train_input, val_grad_output, val_input,
            val_grad_total, use_second_order
        ), None)


def _add_bias_scores(
    scores: "Tensor",
    similarity: "Optional[Tensor]",
    train_grad_output: "Tensor",
    val_grad_output: "Optional[Tensor]",
    val_bias_grad: "Optional[Tensor]",
    has_bias: bool,
) -> "Tuple[Tensor, Optional[Tensor]]":
    """
    Add bias gradient contribution to influence scores and similarity.

    The standard scoring functions compute scores from weight gradients only
    (go ⊗ inp). This adds the bias gradient term: go_i · val_go.

    Args:
        scores: Weight-only influence scores [B]
        similarity: Weight-only similarity matrix [B, B] or None
        train_grad_output: Training grad_output [B, S, O] or [B, O]
        val_grad_output: Validation grad_output [V, S, O] or None
        val_bias_grad: Precomputed validation bias gradient [O] or None
        has_bias: Whether the layer has bias

    Returns:
        (scores, similarity) with bias contribution added
    """
    if not has_bias:
        return scores, similarity

    # Compute train bias gradients: sum over sequence dim
    if train_grad_output.dim() == 3:
        train_bias = train_grad_output.sum(dim=1)  # [B, O]
    else:
        train_bias = train_grad_output  # [B, O]

    # Compute val bias gradient from factorized components or cache
    if val_grad_output is not None:
        if val_grad_output.dim() == 3:
            val_bias = val_grad_output.sum(dim=(0, 1))  # [O]
        else:
            val_bias = val_grad_output.sum(dim=0)  # [O]
    elif val_bias_grad is not None:
        val_bias = val_bias_grad  # [O] from cache
    else:
        return scores, similarity

    # Add bias score: train_bias_i · val_bias
    scores = scores + train_bias @ val_bias.to(train_bias.dtype)

    # Add bias similarity: train_bias_i · train_bias_j
    if similarity is not None:
        similarity = similarity + train_bias @ train_bias.T

    return scores, similarity


def _do_selection(
    state: "LayerWiseSubsetState",
    layer_idx: int,
    scores: Tensor,
    similarity: Optional[Tensor],
    group_key: Optional[str] = None,
) -> Tensor:
    """Run per-layer curation: pick indices, record stats."""
    selected_indices = state._select_indices(scores, similarity, layer_idx=layer_idx)
    return _record_selection(
        state, layer_idx, scores, selected_indices, group_key=group_key
    )


def _record_selection(
    state: "LayerWiseSubsetState",
    layer_idx: int,
    scores: Tensor,
    selected_indices: Tensor,
    group_key: Optional[str] = None,
) -> Tensor:
    """Record an externally solved per-layer subset and its diagnostic scores."""
    selected_indices = selected_indices.sort()[0]
    state._last_selected_indices = selected_indices
    state.num_selected = selected_indices.shape[0]

    if hasattr(state, '_layer_selections'):
        state._layer_selections.append((layer_idx, state.num_selected))
    if hasattr(state, 'add_selection_score_diagnostics'):
        state.add_selection_score_diagnostics(group_key or f"layer_{layer_idx}", scores, selected_indices)
    if state._record_selections:
        record = {
            'layer_idx': layer_idx,
            'selected_indices': selected_indices.tolist(),
            'scores': scores.detach().float().cpu().tolist(),
        }
        geometry = getattr(state, '_current_optimizer_geometry', None)
        if geometry is not None:
            record['optimizer_geometry'] = geometry
            state._current_optimizer_geometry = None
        state._selection_records.append(record)
    return selected_indices


def _produce_gradient_update(
    hook_manager: "GradientHook",
    update_compressor: Optional["Compressor"],
    state: "LayerWiseSubsetState",
    layer_idx: int,
    train_grad_output: Tensor,
    train_input: Tensor,
    selected_indices: Tensor,
    has_bias: bool,
    materialized_grads: Optional[Tensor] = None,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """After curation, produce the gradient update.

    If update_compressor exists: compress selected gradients → store for MeSO → return (None, None).
    Otherwise: compute full gradients for selected samples → return (grad_weight, grad_bias).

    When materialized_grads (G_train [B, O*I]) is provided from direct scoring,
    reuses it for w.grad instead of recomputing from go/inp — avoids holding both
    the raw activations and the gradient simultaneously.
    """
    scale_factor = _compute_scale_factor(state, selected_indices)

    if update_compressor is not None:
        sel_go = train_grad_output[selected_indices]
        sel_inp = augment_input_for_bias(train_input[selected_indices], has_bias)
        update_compressed = update_compressor.forward((sel_go, sel_inp))
        reduced_grad = update_compressed.mean(dim=0, keepdim=True) * scale_factor
        hook_manager._store_compressed_grad(layer_idx, reduced_grad)
        return None, None

    if materialized_grads is not None:
        # Reuse G_train from direct scoring: G_train[i] = go_i^T @ inp_i flattened
        # Note: materialized_grads contains weight grads only (no bias augmentation).
        O = train_grad_output.shape[-1]
        grad_weight = materialized_grads[selected_indices].sum(dim=0).reshape(O, -1) * scale_factor
        grad_bias = None
        if has_bias:
            sel_go = train_grad_output[selected_indices]
            if sel_go.dim() == 3:
                grad_bias = sel_go.sum(dim=(0, 1)) * scale_factor
            else:
                grad_bias = sel_go.sum(dim=0) * scale_factor
        return grad_weight, grad_bias

    grad_weight, grad_bias = compute_selected_gradients(
        train_grad_output, train_input, selected_indices, has_bias, scale_factor
    )
    return grad_weight, grad_bias


def _window_replay_linear_gradient(
    hook_manager: "GradientHook",
    state: "LayerWiseSubsetState",
    layer_idx: int,
    train_grad_output: Tensor,
    train_input: Tensor,
    has_bias: bool,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Assemble this compute chunk using a finalized logical-window decision."""
    from .advanced_solvers import weighted_linear_gradients, weighted_token_scale

    kind, decision = state.get_window_decision(layer_idx)
    start = state.window_chunk_start
    end = state.window_chunk_end
    if kind == "weights":
        global_weights = decision.to(
            device=train_grad_output.device, dtype=torch.float32
        )
        local_weights = global_weights[start:end]
        scale = weighted_token_scale(
            global_weights,
            state.tokens_per_sample.detach().float().to(global_weights.device),
            state.batch_total_tokens_tensor.detach().float().to(
                global_weights.device
            ),
        )
        return weighted_linear_gradients(
            train_grad_output,
            train_input,
            local_weights,
            scale=scale,
            has_bias=has_bias,
            replay_precision=str(
                state.solver_config.get("soft_replay_precision", "fp32")
            ),
        )

    global_indices = decision.to(
        device=train_grad_output.device, dtype=torch.long
    )
    in_chunk = (global_indices >= start) & (global_indices < end)
    local_indices = global_indices[in_chunk] - start
    scale = state._compute_scale_factor(global_indices).to(
        train_grad_output.device
    )
    return compute_selected_gradients(
        train_grad_output,
        train_input,
        local_indices,
        has_bias,
        scale,
    )


def finalize_windowed_layerwise_selection(
    hook_manager: "GradientHook",
) -> "LayerWiseSubsetState":
    """Finalize exactly one layerwise decision over the logical candidate window."""
    from .advanced_solvers import (
        optimize_batched_linear_soft_weights,
        weighted_token_scale,
    )

    state = hook_manager.selection_state
    if state is None or not getattr(state, "windowed_execution", False):
        raise RuntimeError("No windowed layerwise selection state is active")
    if state.window_phase != "score":
        raise RuntimeError("Window selection can be finalized only after scoring")

    variant = getattr(state, "selection_variant", "score")
    saturated = (
        variant == "muon_spectral" and _muon_surrogate_is_saturated(state)
    )
    batched_soft_results = {}
    if variant == "soft":
        cfg = state.solver_config
        if float(cfg.get("soft_weighting_gamma", 0.0)) != 0.0:
            raise RuntimeError(
                "Windowed soft weighting supports only the exact gamma=0 "
                "linear AdamW objective"
            )
        eligible_layers = []
        for candidate_layer_idx in range(state.num_layers):
            if (
                candidate_layer_idx in state._window_full_batch_layers
                or candidate_layer_idx in state._window_zero_layers
                or candidate_layer_idx in state._window_factors
            ):
                continue
            state.require_complete_window(candidate_layer_idx)
            eligible_layers.append(candidate_layer_idx)

        if eligible_layers:
            device = torch.device(state.device)
            score_rows = torch.stack(
                [
                    state._window_scores[candidate_layer_idx].to(
                        device=device, dtype=torch.float32
                    )
                    for candidate_layer_idx in eligible_layers
                ],
                dim=0,
            )
            weight_rows, diagnostic_rows = (
                optimize_batched_linear_soft_weights(
                    score_rows,
                    state.tokens_per_sample.detach().float().to(device),
                    state.batch_total_tokens_tensor.detach().float().to(device),
                    state.num_selected,
                    steps=int(cfg.get("soft_weighting_steps", 20)),
                    lr=float(cfg.get("soft_weighting_lr", 0.1)),
                    tolerance=float(cfg.get("soft_weighting_tol", 1e-5)),
                    patience=int(cfg.get("soft_weighting_patience", 3)),
                    constraint=str(
                        cfg.get("soft_weighting_constraint", "capped_simplex")
                    ),
                )
            )
            batched_soft_results = {
                candidate_layer_idx: (
                    weight_rows[row_idx],
                    diagnostic_rows[row_idx],
                )
                for row_idx, candidate_layer_idx in enumerate(eligible_layers)
            }

    for layer_idx in range(state.num_layers):
        group_key = hook_manager.get_optimizer_group_keys()[layer_idx]
        if layer_idx in state._window_full_batch_layers:
            scores = torch.zeros(
                state.train_batch_size,
                device=state.device,
                dtype=torch.float32,
            )
            selected = torch.arange(
                state.train_batch_size,
                device=state.device,
                dtype=torch.long,
            )
            selected = _record_selection(
                state, layer_idx, scores, selected, group_key
            )
            state.set_window_decision(layer_idx, "indices", selected)
            continue

        if layer_idx in state._window_zero_layers:
            selected = torch.empty(0, device=state.device, dtype=torch.long)
            state._layer_selections.append((layer_idx, 0))
            state.set_window_decision(layer_idx, "indices", selected)
            continue

        if saturated:
            state.require_complete_window(layer_idx, support=True)
            support = state._window_support[layer_idx]
            beta = state._window_beta[layer_idx]
            selected, scores = _select_muon_saturated_support(
                state, layer_idx, support, beta
            )
            selected = _record_selection(
                state, layer_idx, scores, selected, group_key
            )
            state.set_window_decision(layer_idx, "indices", selected)
            continue

        scores: Optional[Tensor]
        if variant == "soft" and layer_idx in state._window_factors:
            # Muon Soft solves the nonlinear objective from retained factors.
            # Optimizer-aware per-sample reference scores were diagnostic-only
            # and are deliberately absent from the hot path.
            scores = state._window_scores.get(layer_idx)
        else:
            state.require_complete_window(layer_idx)
            scores = state._window_scores[layer_idx]
        if variant == "soft":
            batched_result = batched_soft_results.get(layer_idx)
            if batched_result is not None:
                weights, diagnostics = batched_result
                weights = _record_layer_soft_weights(
                    state,
                    layer_idx,
                    weights,
                    diagnostics,
                    scores,
                )
                state.set_window_decision(layer_idx, "weights", weights)
                continue
            if layer_idx in state._window_factors:
                # Muon's update map is nonlinear in the weighted candidate
                # gradient. Rebuild its objective from the contiguous CPU bf16
                # factor window. In bf16_fp32 mode factors remain bf16 on the
                # accelerator while weights, target, reductions, and GEMM output
                # stay fp32; fp32 mode preserves the original exact staging.
                cpu_go, cpu_input = state.get_window_factors(layer_idx)
                device = torch.device(state.device)
                contraction_precision = str(
                    state.solver_config.get("soft_replay_precision", "fp32")
                ).strip().lower().replace("-", "_")
                factor_dtype = (
                    torch.bfloat16
                    if contraction_precision == "bf16_fp32"
                    else torch.float32
                )
                train_go = cpu_go.to(device=device, dtype=factor_dtype)
                train_input = cpu_input.to(device=device, dtype=factor_dtype)
                val_go, val_input, val_total, val_bias = _get_val_components(
                    hook_manager, layer_idx, device=device
                )
                target_weight, target_bias = _linear_target_gradients(
                    torch.float32,
                    state,
                    val_go,
                    val_input,
                    val_total,
                    val_bias,
                    hook_manager._get_module_from_idx(layer_idx).bias is not None,
                )
                objective = _make_soft_linear_objective(
                    hook_manager,
                    state,
                    layer_idx,
                    train_go,
                    train_input,
                    target_weight,
                    target_bias,
                    hook_manager._get_module_from_idx(layer_idx).bias is not None,
                )
                if scores is not None:
                    scores = scores.to(device=device, dtype=torch.float32)
            else:
                if scores is None:
                    raise RuntimeError(
                        f"Layer {layer_idx} has neither Soft factors nor scores"
                    )
                scores = scores.to(device=state.device, dtype=torch.float32)
                tokens = state.tokens_per_sample.detach().float().to(scores.device)
                base_tokens = state.batch_total_tokens_tensor.detach().float().to(
                    scores.device
                )

                def objective(weights: Tensor, layer_scores: Tensor = scores):
                    scale = weighted_token_scale(weights, tokens, base_tokens)
                    return scale * torch.dot(weights, layer_scores)

            weights = _optimize_layer_soft_weights(state, layer_idx, objective, scores)
            state.set_window_decision(layer_idx, "weights", weights)
            state._window_factors.pop(layer_idx, None)
            continue

        if scores is None:
            raise RuntimeError(f"Layer {layer_idx} has no selection scores")
        selected = _do_selection(
            state, layer_idx, scores, None, group_key
        )
        raw_scores = state._window_raw_scores.get(layer_idx)
        if raw_scores is not None and getattr(state, "optimizer_aware", False):
            state.add_raw_opt_score_diagnostics(
                group_key, raw_scores, scores, selected
            )
        state.set_window_decision(layer_idx, "indices", selected)

    state.window_phase = "finalized"
    return state


def finalize_windowed_global_selection(
    hook_manager: "GradientHook",
) -> "LayerWiseSubsetState":
    """Finalize one global decision shared by every layer over the logical window.

    ``GlobalSubsetState`` scores a candidate by the sum of its per-layer
    alignments (see ``GlobalSubsetState.process_layer_gradients``), which is
    exactly what the windowed score tables already hold once every compute chunk
    has been scored. Pooling those tables and deciding once therefore reproduces
    global selection for an arbitrary chunk size, using the same state, replay
    path, and token scaling as the layer-wise window.
    """
    state = hook_manager.selection_state
    if state is None or not getattr(state, "windowed_execution", False):
        raise RuntimeError("No windowed global selection state is active")
    if state.window_phase != "score":
        raise RuntimeError("Window selection can be finalized only after scoring")

    variant = getattr(state, "selection_variant", "score")
    if variant != "score":
        raise RuntimeError(
            "Windowed global selection supports only hard score selection, got "
            f"selection_variant={variant!r}"
        )

    group_keys = hook_manager.get_optimizer_group_keys()
    device = torch.device(state.device)
    pooled = torch.zeros(
        state.train_batch_size, device=device, dtype=torch.float32
    )
    pooled_raw = torch.zeros_like(pooled)
    has_raw_scores = False
    scored_layers = []

    for layer_idx in range(state.num_layers):
        if layer_idx in state._window_full_batch_layers:
            continue
        if layer_idx in state._window_zero_layers:
            continue
        state.require_complete_window(layer_idx)
        pooled += state._window_scores[layer_idx].to(
            device=device, dtype=torch.float32
        )
        raw_scores = state._window_raw_scores.get(layer_idx)
        if raw_scores is not None:
            pooled_raw += raw_scores.to(device=device, dtype=torch.float32)
            has_raw_scores = True
        scored_layers.append(layer_idx)

    full_batch = torch.arange(
        state.train_batch_size, device=device, dtype=torch.long
    )
    if scored_layers:
        selected = state._select_indices(pooled, None, layer_idx=-1).sort()[0]
    else:
        # No hooked layer produced a score for this window. Keeping the whole
        # window is the only unbiased fallback, and it matches how the per-layer
        # path treats a layer it could not score.
        selected = full_batch
    state._last_selected_indices = selected
    state.num_selected = int(selected.numel())

    state.add_selection_score_diagnostics("global", pooled, selected)
    if has_raw_scores and getattr(state, "optimizer_aware", False):
        state.add_raw_opt_score_diagnostics("global", pooled_raw, pooled, selected)
    if state._record_selections:
        state._selection_records.append({
            "layer_idx": -1,
            "selected_indices": selected.tolist(),
            "scores": pooled.detach().float().cpu().tolist(),
        })

    empty = torch.empty(0, device=device, dtype=torch.long)
    for layer_idx in range(state.num_layers):
        if layer_idx in state._window_zero_layers:
            state._layer_selections.append((layer_idx, 0))
            state.set_window_decision(layer_idx, "indices", empty)
            continue
        if layer_idx in state._window_full_batch_layers:
            state._layer_selections.append((layer_idx, state.train_batch_size))
            state.set_window_decision(layer_idx, "indices", full_batch)
            continue
        # Per-group alignment of the shared subset, recorded under the same
        # metric names the layer-wise runs use so the two stay comparable.
        state.add_selection_score_diagnostics(
            group_keys[layer_idx], state._window_scores[layer_idx], selected
        )
        state._layer_selections.append((layer_idx, int(selected.numel())))
        state.set_window_decision(layer_idx, "indices", selected)

    state.window_phase = "finalized"
    return state


def _store_update_grad(
    hook_manager: "GradientHook",
    update_compressor: "Optional[Compressor]",
    score_compressor: "Optional[Compressor]",
    layer_idx: int,
    grad_output: Tensor,
    input_aug: Tensor,
    has_bias: bool,
    score_compressed_reduced: Tensor,
) -> None:
    """Store compressed gradient for MeSO when no curation is active.

    Reuses score-compressed grad if compressors are shared; otherwise re-compresses.
    """
    if update_compressor is None:
        return
    if update_compressor is score_compressor:
        hook_manager._store_compressed_grad(layer_idx, score_compressed_reduced)
    else:
        update_compressed = update_compressor.forward((grad_output, input_aug))
        hook_manager._store_compressed_grad(layer_idx, update_compressed.sum(dim=0, keepdim=True))


# =============================================================================
# Autograd Functions
# =============================================================================

class CompressedLinearBackward(Function):
    """
    Autograd Function for pure gradient compression (no data curation).

    Used when compression is enabled (MeSO optimizer) but no data curation is active.
    """

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        hook_manager: "GradientHook",
        layer_idx: int
    ) -> Tensor:
        """Forward pass: standard linear transformation."""
        input_compute = input.to(weight.dtype) if input.dtype != weight.dtype else input
        ctx.save_for_backward(input_compute, weight, bias)
        ctx.hook_manager_ref = weakref.ref(hook_manager)
        ctx.layer_idx = layer_idx
        return F.linear(input_compute, weight, bias)

    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor
    ) -> Tuple[Tensor, None, None, None, None]:
        """Backward pass: compress gradients and store for MeSO optimizer."""
        input, weight, bias = ctx.saved_tensors
        layer_idx = ctx.layer_idx

        hook_manager = ctx.hook_manager_ref()
        if hook_manager is None:
            raise RuntimeError("Hook manager was garbage collected before backward pass")

        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)

        grad_input = grad_output @ weight.to(grad_output.dtype)

        compressor = hook_manager.update_compressors[layer_idx]

        with torch.no_grad():
            input_aug = augment_input_for_bias(input, bias is not None)
            compressed_grad = compressor.forward((grad_output, input_aug))
            compressed_grad = compressed_grad.sum(dim=0, keepdim=True)
            hook_manager._store_compressed_grad(layer_idx, compressed_grad)

        return grad_input, None, None, None, None


class LayerWiseSubsetLinearBackward(Function):
    """
    Autograd Function for layer_wise_subset descent (per-layer curation).

    Single-pass: At each layer, computes scores, selects samples,
    and aggregates gradients immediately.
    """

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        hook_manager: "GradientHook",
        layer_idx: int
    ) -> Tensor:
        """Forward pass: standard linear transformation."""
        input_compute = input.to(weight.dtype) if input.dtype != weight.dtype else input
        ctx.save_for_backward(input_compute, weight, bias)
        ctx.hook_manager_ref = weakref.ref(hook_manager)
        ctx.layer_idx = layer_idx
        return F.linear(input_compute, weight, bias)

    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], None, None]:
        """Backward pass with per-layer curation."""
        input, weight, bias = ctx.saved_tensors
        layer_idx = ctx.layer_idx

        hook_manager = ctx.hook_manager_ref()
        if hook_manager is None:
            raise RuntimeError("Hook manager was garbage collected before backward pass")

        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)

        grad_input = grad_output @ weight.to(grad_output.dtype)

        score_compressor = hook_manager.score_compressors[layer_idx]
        update_compressor = hook_manager.update_compressors[layer_idx]
        state: Optional[LayerWiseSubsetState] = hook_manager.selection_state
        capture_val_mode = hook_manager.capture_val_mode
        use_stored_val = (
            state is not None and
            getattr(state, '_use_stored_val', False)
        )

        # Determine scoring path: use compressed only when scoring_method="compress"
        # (or during val capture when val cache is in compressed mode)
        use_compressed_scoring = False
        if score_compressor is not None:
            if capture_val_mode:
                # Val capture: use compression only if val cache is in compressed mode
                use_compressed_scoring = hook_manager._val_cache.is_compressed
            elif state is not None:
                use_compressed_scoring = getattr(state, 'scoring_method', 'reduced_ghost') == 'compress'

        with torch.no_grad():
            if use_compressed_scoring:
                grad_weight, grad_bias = LayerWiseSubsetLinearBackward._backward_compressed(
                    hook_manager, score_compressor, update_compressor, state, layer_idx,
                    input, grad_output, bias, capture_val_mode, use_stored_val
                )
            else:
                grad_weight, grad_bias = LayerWiseSubsetLinearBackward._backward_full(
                    hook_manager, update_compressor, state, layer_idx,
                    input, bias, grad_output,
                    capture_val_mode, use_stored_val
                )

        return grad_input, grad_weight, grad_bias, None, None

    @staticmethod
    def _backward_compressed(
        hook_manager: "GradientHook",
        score_compressor: "Compressor",
        update_compressor: Optional["Compressor"],
        state: Optional["LayerWiseSubsetState"],
        layer_idx: int,
        input: Tensor,
        grad_output: Tensor,
        bias: Optional[Tensor],
        capture_val_mode: bool,
        use_stored_val: bool
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """LayerWiseSubset backward with score compression.

        Score computation uses score_compressor. Gradient updates are either:
        - Compressed via update_compressor (MeSO): returns (None, None)
        - Full gradients (no MeSO): returns (grad_weight, grad_bias)
        When update_compressor is score_compressor, reuses already-compressed grads.
        """
        has_bias = bias is not None
        input_aug = augment_input_for_bias(input, has_bias)

        # --- Compress ---

        score_compressed = score_compressor.forward((grad_output, input_aug))


        # --- Validation capture: store compressed val gradient ---
        if capture_val_mode:
            total_grad = score_compressed.sum(dim=0)
            val_cache = hook_manager.val_cache
            if val_cache._compressed[layer_idx] is None:
                val_cache._compressed[layer_idx] = total_grad
            else:
                val_cache._compressed[layer_idx] = val_cache._compressed[layer_idx] + total_grad
            return None, None

        # --- No curation state: just store MeSO update if needed ---
        if state is None:
            _store_update_grad(hook_manager, update_compressor, score_compressor,
                               layer_idx, grad_output, input_aug, has_bias,
                               score_compressed.sum(dim=0, keepdim=True))
            return None, None

        # --- Compute scores from score-compressed gradients ---

        if use_stored_val:
            train_grads = score_compressed
            val_grad = hook_manager._val_cache.get_compressed(layer_idx)
            score_correction = None
        else:
            train_grads, val_grads = split_train_val_batch(score_compressed, state.train_batch_size)
            val_grad = val_grads.sum(dim=0)
            score_correction = state.score_correction

        if val_grad is None:
    
            _store_update_grad(hook_manager, update_compressor, score_compressor,
                               layer_idx, grad_output, input_aug, has_bias,
                               score_compressed.mean(dim=0, keepdim=True))
            return None, None

        scores = train_grads @ val_grad
        if score_correction is not None:
            scores = scores * score_correction

        similarity = None
        if state.use_second_order:
            similarity = train_grads @ train_grads.T
            if score_correction is not None:
                similarity = similarity * (score_correction ** 2)


        # --- Shared compressors: delegate to state for efficient select+reduce ---
        if update_compressor is not None and update_compressor is score_compressor:

            reduced_grad, _ = state.process_layer_gradients(
                train_grads, val_grad, layer_idx, score_correction
            )
            hook_manager._store_compressed_grad(layer_idx, reduced_grad)

            return None, None

        # --- Select ---

        selected_indices = _do_selection(state, layer_idx, scores, similarity)


        # --- w.grad ---

        if use_stored_val:
            train_grad_output, train_input = grad_output, input
        else:
            train_grad_output, _ = split_train_val_batch(grad_output, state.train_batch_size)
            train_input, _ = split_train_val_batch(input, state.train_batch_size)

        return _produce_gradient_update(
            hook_manager, update_compressor, state, layer_idx,
            train_grad_output, train_input, selected_indices, has_bias
        )

    @staticmethod
    def _backward_full(
        hook_manager: "GradientHook",
        update_compressor: Optional["Compressor"],
        state: Optional["LayerWiseSubsetState"],
        layer_idx: int,
        input: Tensor,
        bias: Optional[Tensor],
        grad_output: Tensor,
        capture_val_mode: bool,
        use_stored_val: bool
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """LayerWiseSubset backward without score compression (full gradient scoring).

        If update_compressor is set, compresses selected gradients for MeSO.
        """
        has_bias = bias is not None

        # --- Validation capture: store full gradients ---
        if capture_val_mode:
            hook_manager.val_cache.store_layer(
                layer_idx=layer_idx,
                grad_output=grad_output.detach(),
                input=input.detach(),
                compressor=None
            )
            return None, None

        # --- No curation state: just store MeSO update if needed ---
        if state is None:
            if update_compressor is not None:
                input_aug = augment_input_for_bias(input, has_bias)
                update_compressed = update_compressor.forward((grad_output, input_aug))
                hook_manager._store_compressed_grad(layer_idx, update_compressed.sum(dim=0, keepdim=True))
            return None, None

        windowed = bool(getattr(state, "windowed_execution", False))
        if windowed:
            if not use_stored_val:
                raise RuntimeError("Windowed execution requires separate_batch target gradients")
            if state.window_phase == "replay":
                return _window_replay_linear_gradient(
                    hook_manager,
                    state,
                    layer_idx,
                    grad_output,
                    input,
                    has_bias,
                )
            if state.window_phase != "score":
                raise RuntimeError(f"Invalid window phase: {state.window_phase!r}")

        if getattr(state, "selection_variant", "score") == "random":
            if use_stored_val:
                train_grad_output, train_input = grad_output, input
            else:
                train_grad_output, _ = split_train_val_batch(
                    grad_output, state.train_batch_size
                )
                train_input, _ = split_train_val_batch(input, state.train_batch_size)
            scores = torch.zeros(
                train_grad_output.shape[0] if windowed else state.train_batch_size,
                device=train_grad_output.device,
                dtype=torch.float32,
            )
            if windowed:
                state.store_window_scores(layer_idx, scores)
                return None, None
            selected_indices = _do_selection(
                state, layer_idx, scores, None,
                hook_manager.get_optimizer_group_keys()[layer_idx],
            )
            return _produce_gradient_update(
                hook_manager, update_compressor, state, layer_idx,
                train_grad_output, train_input, selected_indices, has_bias,
            )

        variant = getattr(state, "selection_variant", "score")
        if windowed and variant in ("soft", "muon_spectral") and (
            layer_idx in state._window_zero_layers
        ):
            # The target is invariant throughout the logical window. Once the
            # first chunk proves it is zero, later chunks cannot add a score.
            return None, None

        linear_probe_cached = False
        cached_spectral_modes = None
        collect_optimizer_diagnostics = False
        if windowed and getattr(state, "optimizer_aware", False):
            from .optimizer_aware import has_cached_optimizer_aware_linear_probe

            linear_probe_cached = has_cached_optimizer_aware_linear_probe(
                state, layer_idx
            )
            diagnostic_predicate = getattr(
                state, "should_collect_optimizer_diagnostics", None
            )
            collect_optimizer_diagnostics = bool(
                callable(diagnostic_predicate)
                and diagnostic_predicate()
                and variant != "soft"
            )
        if (
            variant == "muon_spectral"
            and not _muon_surrogate_includes_adamw_scores(state)
        ):
            module = hook_manager._get_module_from_idx(layer_idx)
            weight_is_muon = (
                _parameter_optimizer_kind(hook_manager, module.weight) == "muon"
            )
            if not weight_is_muon:
                # No target gradient is needed: this ablation deliberately keeps
                # AdamW-only layers on their full candidate-batch gradient.
                if use_stored_val:
                    train_grad_output, train_input = grad_output, input
                else:
                    train_grad_output, _ = split_train_val_batch(
                        grad_output, state.train_batch_size
                    )
                    train_input, _ = split_train_val_batch(
                        input, state.train_batch_size
                    )
                all_indices = torch.arange(
                    state.train_batch_size,
                    device=train_grad_output.device,
                    dtype=torch.long,
                )
                state._append_diagnostic(
                    f"spectral/layer_{layer_idx}/adamw_full_batch", 1.0
                )
                if windowed:
                    state.mark_window_full_batch(layer_idx)
                    return None, None
                state._last_selected_indices = all_indices
                state._layer_selections.append(
                    (layer_idx, state.train_batch_size)
                )
                return _produce_gradient_update(
                    hook_manager, update_compressor, state, layer_idx,
                    train_grad_output, train_input, all_indices, has_bias,
                )
            if windowed:
                # Matrix-only Muon surrogates need the raw target only until the
                # first candidate chunk has frozen its spectral modes. Check
                # this before staging validation tensors so chunks 2..N do not
                # copy or rescale a model-sized target that cannot affect them.
                cached_spectral_modes = state.get_window_spectral_modes(layer_idx)

        # --- Compute scores from full gradients ---
        if use_stored_val:
            train_grad_output, train_input = grad_output, input
            if cached_spectral_modes is not None:
                val_grad_output = val_input = val_grad_total = val_bias_grad = None
            elif linear_probe_cached and not collect_optimizer_diagnostics:
                val_grad_output = val_input = val_grad_total = None
                val_bias_grad = hook_manager._val_cache.get_bias_grad(
                    layer_idx, device=train_grad_output.device
                )
            else:
                val_grad_output, val_input, val_grad_total, val_bias_grad = (
                    _get_val_components(
                        hook_manager, layer_idx, device=train_grad_output.device
                    )
                )
            if (
                val_grad_output is None
                and val_grad_total is None
                and not linear_probe_cached
                and cached_spectral_modes is None
            ):
                return None, None
            score_correction = None
        else:
            train_grad_output, val_grad_output = split_train_val_batch(grad_output, state.train_batch_size)
            train_input, val_input = split_train_val_batch(input, state.train_batch_size)
            val_grad_total = None
            val_bias_grad = None
            score_correction = state.score_correction

        if variant == "soft":
            if update_compressor is not None:
                raise ValueError("Soft weighting does not support update compression")
            module = hook_manager._get_module_from_idx(layer_idx)
            weight_is_muon = (
                _parameter_optimizer_kind(hook_manager, module.weight) == "muon"
            )
            adamw_window_fastpath = bool(
                windowed
                and float(state.solver_config.get("soft_weighting_gamma", 0.0))
                == 0.0
                and not weight_is_muon
            )
            muon_window_seen = bool(
                windowed
                and weight_is_muon
                and layer_idx in state._window_factors
            )

            # A Muon Soft logical window needs the raw target only on its first
            # chunk to establish the zero-target invariant. Later chunks retain
            # candidate factors only; finalization rebuilds the objective once.
            target_weight = target_bias = None
            if not (
                (adamw_window_fastpath and linear_probe_cached)
                or muon_window_seen
            ):
                target_weight, target_bias = _linear_target_gradients(
                    train_grad_output.dtype, state,
                    val_grad_output, val_input, val_grad_total, val_bias_grad,
                    has_bias,
                )
                if bool((target_weight.square().sum() == 0).detach().cpu()) and (
                    target_bias is None
                    or bool((target_bias.square().sum() == 0).detach().cpu())
                ):
                    state._append_diagnostic(
                        f"soft/layer_{layer_idx}/zero_target", 1.0
                    )
                    if windowed:
                        state.mark_window_zero(layer_idx)
                    return None, None

            reference_scores = None
            if not (windowed and weight_is_muon):
                reference_scores = _soft_reference_linear_scores(
                    hook_manager, state, layer_idx,
                    train_grad_output, train_input,
                    val_grad_output, val_input, val_grad_total, val_bias_grad, has_bias,
                )
            if adamw_window_fastpath:
                if reference_scores is None:
                    raise RuntimeError("AdamW Soft requires linear reference scores")
                state.store_window_scores(layer_idx, reference_scores)
                return None, None

            if windowed:
                if reference_scores is not None:
                    state.store_window_scores(layer_idx, reference_scores)
                if weight_is_muon:
                    state.store_window_factors(
                        layer_idx, train_grad_output, train_input
                    )
                return None, None
            objective = _make_soft_linear_objective(
                hook_manager, state, layer_idx,
                train_grad_output, train_input,
                target_weight, target_bias, has_bias,
            )
            weights = _optimize_layer_soft_weights(
                state, layer_idx, objective, reference_scores
            )

            from .advanced_solvers import weighted_linear_gradients, weighted_token_scale
            scale = weighted_token_scale(
                weights,
                state.tokens_per_sample.detach().float(),
                _candidate_base_tokens(state),
            )
            grad_weight, grad_bias = weighted_linear_gradients(
                train_grad_output, train_input,
                weights.to(train_grad_output.device),
                scale=scale,
                has_bias=has_bias,
                replay_precision=str(
                    state.solver_config.get("soft_replay_precision", "fp32")
                ),
            )
            module = hook_manager._get_module_from_idx(layer_idx)
            grad_weight = grad_weight.to(module.weight.dtype)
            if grad_bias is not None and module.bias is not None:
                grad_bias = grad_bias.to(module.bias.dtype)
            return grad_weight, grad_bias

        scoring_method = getattr(state, 'scoring_method', 'reduced_ghost')
        materialized_grads = None
        scores_already_scaled = False
        preselected_indices = None
        if variant == "muon_spectral":
            module = hook_manager._get_module_from_idx(layer_idx)
            weight_is_muon = (
                _parameter_optimizer_kind(hook_manager, module.weight) == "muon"
            )
            include_adamw_scores = _muon_surrogate_includes_adamw_scores(state)
            cached_modes = (
                cached_spectral_modes
                if cached_spectral_modes is not None
                else (
                    state.get_window_spectral_modes(layer_idx)
                    if windowed and weight_is_muon
                    else None
                )
            )
            target_weight = target_bias = None
            if cached_modes is None or include_adamw_scores:
                target_weight, target_bias = _linear_target_gradients(
                    train_grad_output.dtype, state,
                    val_grad_output, val_input, val_grad_total, val_bias_grad, has_bias,
                )
            if target_weight is not None and bool(
                (target_weight.square().sum() == 0).detach().cpu()
            ) and (
                not include_adamw_scores
                or target_bias is None
                or bool((target_bias.square().sum() == 0).detach().cpu())
            ):
                state._append_diagnostic(f"spectral/layer_{layer_idx}/zero_target", 1.0)
                if windowed:
                    # No target signal means this layer intentionally receives
                    # no candidate update. Mark that decision explicitly so
                    # logical-window finalization does not mistake the absent
                    # score vector for an incomplete score phase.
                    state.mark_window_zero(layer_idx)
                    if (
                        use_stored_val
                        and weight_is_muon
                        and not include_adamw_scores
                    ):
                        hook_manager._val_cache.discard_weight_gradient(layer_idx)
                return None, None
            if weight_is_muon:
                if _muon_surrogate_is_saturated(state):
                    if include_adamw_scores:
                        raise ValueError(
                            "Saturated Muon surrogate is defined for the "
                            "matrix-only criterion"
                        )
                    support, beta, _ = _compute_muon_spectral_linear_support(
                        hook_manager, state, layer_idx,
                        train_grad_output, train_input, target_weight,
                    )
                    if windowed:
                        state.store_window_support(layer_idx, support, beta)
                        scores = support.sum(dim=1)
                    else:
                        preselected_indices, scores = _select_muon_saturated_support(
                            state, layer_idx, support, beta
                        )
                else:
                    scores, _ = _compute_muon_spectral_linear_scores(
                        hook_manager, state, layer_idx,
                        train_grad_output, train_input, target_weight,
                    )
                if (
                    use_stored_val
                    and windowed
                    and not include_adamw_scores
                    and cached_modes is None
                ):
                    # (U, V, beta, rank) is now the complete immutable target
                    # representation for this logical window. Release the raw
                    # validation weight gradient immediately; later chunks use
                    # the cached modes and never fetch it again.
                    hook_manager._val_cache.discard_weight_gradient(layer_idx)
                if include_adamw_scores:
                    # In the mixed Hybrid criterion, biases remain AdamW OptA
                    # parameters.  The matrix-only ablation intentionally omits
                    # this score, although the bias follows the weight's subset.
                    scores = scores + _adamw_bias_scores_standalone(
                        hook_manager, state, module, train_grad_output, target_bias,
                    )
                similarity = None
                scores_already_scaled = True
            else:
                from .optimizer_aware import compute_optimizer_aware_linear_scores

                scores, similarity, geometry, raw_scores = compute_optimizer_aware_linear_scores(
                    hook_manager, state, layer_idx,
                    train_grad_output, train_input,
                    val_grad_output, val_input, val_grad_total, False,
                )
                if score_correction is not None:
                    scores = scores * score_correction
                scores = scores + _adamw_bias_scores_standalone(
                    hook_manager, state, module, train_grad_output, target_bias,
                )
                scores_already_scaled = True
                if state._record_selections:
                    state._current_optimizer_geometry = geometry
        elif getattr(state, 'optimizer_aware', False):
            from .optimizer_aware import compute_optimizer_aware_linear_scores

            scores, similarity, geometry, raw_scores = compute_optimizer_aware_linear_scores(
                hook_manager, state, layer_idx,
                train_grad_output, train_input,
                val_grad_output, val_input, val_grad_total,
                state.use_second_order,
            )
            if state._record_selections:
                state._current_optimizer_geometry = geometry

            # Bias terms are not matrix-shaped, so keep their raw AdamW-style
            # contribution. LoRA adapters normally have no bias.
            scores, similarity = _add_bias_scores(
                scores, similarity, train_grad_output,
                val_grad_output, val_bias_grad, has_bias
            )
            if raw_scores is not None:
                raw_scores, _ = _add_bias_scores(
                    raw_scores, None, train_grad_output,
                    val_grad_output, val_bias_grad, has_bias
                )
        else:
            # Request materialized G_train for direct scoring so we can reuse it
            # for w.grad without keeping train_input alive.
            want_materialized = (scoring_method == "direct"
                                 and update_compressor is None)
            scores, similarity, materialized_grads = _dispatch_scoring(
                scoring_method, train_grad_output, train_input,
                val_grad_output, val_input, val_grad_total,
                state.use_second_order,
                direct_batch_size=getattr(state, 'direct_batch_size', 0),
                return_materialized=want_materialized,
            )

            # Add bias gradient contribution to scores
            scores, similarity = _add_bias_scores(
                scores, similarity, train_grad_output,
                val_grad_output, val_bias_grad, has_bias
            )

        # Free raw activations early when we have materialized grads for w.grad.
        # Use None assignment (not del) since train_input is still passed by name below.
        if materialized_grads is not None:
            train_input = val_input = val_grad_total = val_grad_output = None

        if score_correction is not None and not scores_already_scaled:
            scores = scores * score_correction
            if similarity is not None:
                similarity = similarity * (score_correction ** 2)

        # --- Select, then produce gradient update ---
        metric_group_key = hook_manager.get_optimizer_group_keys()[layer_idx]
        if windowed:
            state.store_window_scores(
                layer_idx,
                scores,
                raw_scores=(
                    raw_scores
                    if getattr(state, "optimizer_aware", False)
                    and variant != "muon_spectral"
                    else None
                ),
                geometry=(
                    geometry
                    if getattr(state, "optimizer_aware", False)
                    and variant != "muon_spectral"
                    else None
                ),
            )
            return None, None
        if preselected_indices is None:
            selected_indices = _do_selection(
                state, layer_idx, scores, similarity, metric_group_key
            )
        else:
            selected_indices = _record_selection(
                state, layer_idx, scores, preselected_indices, metric_group_key
            )
        if (
            getattr(state, 'optimizer_aware', False)
            and variant != "muon_spectral"
            and raw_scores is not None
        ):
            state.add_raw_opt_score_diagnostics(
                metric_group_key,
                raw_scores,
                scores,
                selected_indices,
            )

        return _produce_gradient_update(
            hook_manager, update_compressor, state, layer_idx,
            train_grad_output, train_input, selected_indices, has_bias,
            materialized_grads=materialized_grads,
        )


class GlobalSubsetLinearBackward(Function):
    """
    Autograd Function for GlobalSubset method (global curation).

    Pass 1: Accumulates scores across all layers (no gradient output).
    Pass 2: Forward/backward on selected samples only.
            Without MeSO: hooks disabled, standard autograd gradients.
            With MeSO: hooks stay enabled, CompressedLinearBackward stores
            compressed gradients for the optimizer.
    """

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        hook_manager: "GradientHook",
        layer_idx: int
    ) -> Tensor:
        """Forward pass: standard linear transformation."""
        input_compute = input.to(weight.dtype) if input.dtype != weight.dtype else input
        ctx.save_for_backward(input_compute, weight, bias)
        ctx.hook_manager_ref = weakref.ref(hook_manager)
        ctx.layer_idx = layer_idx
        return F.linear(input_compute, weight, bias)

    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor
    ) -> Tuple[Tensor, None, None, None, None]:
        """Backward pass: accumulate scores only."""
        input, weight, bias = ctx.saved_tensors
        layer_idx = ctx.layer_idx

        hook_manager = ctx.hook_manager_ref()
        if hook_manager is None:
            raise RuntimeError("Hook manager was garbage collected before backward pass")

        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)

        grad_input = grad_output @ weight.to(grad_output.dtype)

        score_compressor = hook_manager.score_compressors[layer_idx]
        state: Optional[GlobalSubsetState] = hook_manager.selection_state

        if state is None:
            return grad_input, None, None, None, None

        use_stored_val = getattr(state, '_use_stored_val', False)

        # Use compressed scoring only when scoring_method="compress"
        use_compressed_scoring = (
            score_compressor is not None
            and getattr(state, 'scoring_method', 'reduced_ghost') == 'compress'
        )

        with torch.no_grad():
            if use_compressed_scoring:
                GlobalSubsetLinearBackward._accumulate_compressed(
                    hook_manager, score_compressor, state, layer_idx,
                    input, grad_output, bias, use_stored_val
                )
            else:
                GlobalSubsetLinearBackward._accumulate_full(
                    hook_manager, state, layer_idx,
                    input, grad_output, bias is not None, use_stored_val
                )

            # One-pass mode: retain (grad_output, input) for post-hoc gradient assembly
            if state.one_pass:
                if use_stored_val:
                    # SeparateBatch: entire batch is train
                    hook_manager.retain_layer_data(layer_idx, grad_output, input)
                else:
                    # MergedBatch: extract train portion only
                    train_go, _ = split_train_val_batch(grad_output, state.train_batch_size)
                    train_inp, _ = split_train_val_batch(input, state.train_batch_size)
                    hook_manager.retain_layer_data(layer_idx, train_go, train_inp)

        return grad_input, None, None, None, None

    @staticmethod
    def _accumulate_compressed(
        hook_manager: "GradientHook",
        compressor: "Compressor",
        state: "GlobalSubsetState",
        layer_idx: int,
        input: Tensor,
        grad_output: Tensor,
        bias: Optional[Tensor],
        use_stored_val: bool
    ) -> None:
        """Accumulate scores from compressed gradients."""
        input_aug = augment_input_for_bias(input, bias is not None)
        compressed_grad = compressor.forward((grad_output, input_aug))

        if use_stored_val:
            train_grads = compressed_grad
            val_grad = hook_manager._val_cache.get_compressed(layer_idx)
            # Cached mode: gradients already correctly scaled
            score_correction = None
        else:
            train_grads, val_grads = split_train_val_batch(compressed_grad, state.train_batch_size)
            val_grad = val_grads.sum(dim=0)  # Sum, not mean, for token-weighted semantics
            # Joint batch needs correction: T_total²/(T_train × T_val)
            score_correction = state.score_correction  # Tensor

        if val_grad is not None:
            state.process_layer_gradients(train_grads, val_grad, layer_idx, score_correction)

    @staticmethod
    def _accumulate_full(
        hook_manager: "GradientHook",
        state: "GlobalSubsetState",
        layer_idx: int,
        input: Tensor,
        grad_output: Tensor,
        has_bias: bool,
        use_stored_val: bool
    ) -> None:
        """Accumulate scores from full gradients."""
        variant = getattr(state, "selection_variant", "score")
        if variant == "random":
            return
        if (
            variant == "muon_spectral"
            and not _muon_surrogate_includes_adamw_scores(state)
        ):
            module = hook_manager._get_module_from_idx(layer_idx)
            if _parameter_optimizer_kind(hook_manager, module.weight) != "muon":
                # The outer backward retains this layer for shared-subset update
                # assembly; only its score contribution is omitted here.
                state._append_diagnostic(
                    f"spectral/layer_{layer_idx}/adamw_score_omitted", 1.0
                )
                return
        if use_stored_val:
            train_grad_output, train_input = grad_output, input
            val_go, val_inp, val_grad_total, val_bias_grad = (
                _get_val_components(
                    hook_manager, layer_idx, device=train_grad_output.device
                )
            )
            if val_go is None and val_grad_total is None:
                return
            # Cached mode: gradients already correctly scaled, no correction needed
            score_correction = None
        else:
            # Joint batch mode: split merged batch
            train_grad_output, val_go = split_train_val_batch(grad_output, state.train_batch_size)
            train_input, val_inp = split_train_val_batch(input, state.train_batch_size)
            val_grad_total = None
            val_bias_grad = None
            # Joint batch needs correction: T_total²/(T_train × T_val)
            score_correction = state.score_correction  # Tensor

        if variant == "soft":
            target_weight, target_bias = _linear_target_gradients(
                train_grad_output.dtype, state,
                val_go, val_inp, val_grad_total, val_bias_grad, has_bias,
            )
            if bool((target_weight.square().sum() == 0).detach().cpu()) and (
                target_bias is None
                or bool((target_bias.square().sum() == 0).detach().cpu())
            ):
                state._append_diagnostic(f"soft/layer_{layer_idx}/zero_target", 1.0)
                return
            objective = _make_soft_linear_objective(
                hook_manager, state, layer_idx,
                train_grad_output, train_input,
                target_weight, target_bias, has_bias,
            )
            reference_scores = _soft_reference_linear_scores(
                hook_manager, state, layer_idx,
                train_grad_output, train_input,
                val_go, val_inp, val_grad_total, val_bias_grad, has_bias,
            )
            state.add_soft_objective(objective, reference_scores)
            return

        scores_already_scaled = False
        if variant == "muon_spectral":
            module = hook_manager._get_module_from_idx(layer_idx)
            weight_is_muon = (
                _parameter_optimizer_kind(hook_manager, module.weight) == "muon"
            )
            include_adamw_scores = _muon_surrogate_includes_adamw_scores(state)
            target_weight, target_bias = _linear_target_gradients(
                train_grad_output.dtype, state,
                val_go, val_inp, val_grad_total, val_bias_grad, has_bias,
            )
            if bool((target_weight.square().sum() == 0).detach().cpu()) and (
                not include_adamw_scores
                or target_bias is None
                or bool((target_bias.square().sum() == 0).detach().cpu())
            ):
                state._append_diagnostic(f"spectral/layer_{layer_idx}/zero_target", 1.0)
                return
            if weight_is_muon:
                scores, _ = _compute_muon_spectral_linear_scores(
                    hook_manager, state, layer_idx,
                    train_grad_output, train_input, target_weight,
                )
                if include_adamw_scores:
                    scores = scores + _adamw_bias_scores_standalone(
                        hook_manager, state, module, train_grad_output, target_bias,
                    )
                similarity = None
                scores_already_scaled = True
                raw_scores = None
            else:
                from .optimizer_aware import compute_optimizer_aware_linear_scores

                scores, similarity, geometry, raw_scores = compute_optimizer_aware_linear_scores(
                    hook_manager, state, layer_idx,
                    train_grad_output, train_input,
                    val_go, val_inp, val_grad_total, False,
                )
                if score_correction is not None:
                    scores = scores * score_correction
                scores = scores + _adamw_bias_scores_standalone(
                    hook_manager, state, module, train_grad_output, target_bias,
                )
                scores_already_scaled = True
                raw_scores = None
        elif getattr(state, 'optimizer_aware', False):
            from .optimizer_aware import compute_optimizer_aware_linear_scores

            scores, similarity, geometry, raw_scores = compute_optimizer_aware_linear_scores(
                hook_manager, state, layer_idx,
                train_grad_output, train_input,
                val_go, val_inp, val_grad_total,
                state.use_second_order,
            )
            if state._record_selections:
                state._current_optimizer_geometry = geometry
            scores, similarity = _add_bias_scores(
                scores, similarity, train_grad_output,
                val_go, val_bias_grad, has_bias
            )
            if raw_scores is not None:
                raw_scores, _ = _add_bias_scores(
                    raw_scores, None, train_grad_output,
                    val_go, val_bias_grad, has_bias
                )
        else:
            # Route to scoring method based on state configuration
            scores, similarity, _ = _dispatch_scoring(
                state.scoring_method, train_grad_output, train_input,
                val_go, val_inp, val_grad_total, state.use_second_order,
                direct_batch_size=getattr(state, 'direct_batch_size', 0)
            )

            # Add bias gradient contribution to scores
            scores, similarity = _add_bias_scores(
                scores, similarity, train_grad_output,
                val_go, val_bias_grad, has_bias
            )

        # Accumulate scores using the state method (handles correction internally)
        if scores_already_scaled:
            score_correction = None
        state.accumulate_precomputed_scores(
            scores,
            similarity,
            score_correction,
            layer_idx=layer_idx,
            raw_scores_for_diag=(
                raw_scores
                if getattr(state, 'optimizer_aware', False) and raw_scores is not None
                else None
            ),
        )


# =============================================================================
# Non-linear layer backward for merged-batch one-pass mode
# =============================================================================

class TrainOnlyRMSNormBackward(Function):
    """
    Custom autograd Function for RMSNorm in merged-batch one-pass mode.

    Computes grad_input for the full merged batch (needed for chain rule
    propagation to earlier layers), but computes grad_weight from only the
    train slice of the batch. This prevents validation gradient leakage
    into non-linear layer parameter updates.

    Matches the HuggingFace RMSNorm forward:
        hidden = hidden.to(float32)
        variance = hidden.pow(2).mean(-1, keepdim=True)
        hidden = hidden * rsqrt(variance + eps)
        return weight * hidden.to(input_dtype)
    """

    @staticmethod
    def forward(
        ctx,
        input: "Tensor",
        weight: "Tensor",
        eps: float,
        hook_manager: "GradientHook",
    ) -> "Tensor":
        """Forward pass: standard RMSNorm."""
        input_dtype = input.dtype
        input_f32 = input.to(torch.float32)
        variance = input_f32.pow(2).mean(-1, keepdim=True)
        rsqrt_var = torch.rsqrt(variance + eps)
        normalized = input_f32 * rsqrt_var
        output = weight * normalized.to(input_dtype)

        ctx.save_for_backward(input_f32, weight.to(torch.float32), rsqrt_var, normalized)
        ctx.hook_manager_ref = weakref.ref(hook_manager)
        ctx.input_dtype = input_dtype
        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: "Tensor",
    ) -> "Tuple[Tensor, Tensor, None, None]":
        """
        Backward pass: full grad_input, train-only grad_weight.

        grad_input must cover the full merged batch for correct chain rule
        propagation. grad_weight is computed from the train slice only.
        """
        input_f32, weight_f32, rsqrt_var, normalized = ctx.saved_tensors

        hook_manager = ctx.hook_manager_ref()
        state = hook_manager.selection_state if hook_manager is not None else None
        train_bs = state.train_batch_size if state is not None else None

        D = input_f32.shape[-1]
        grad_output_f32 = grad_output.to(torch.float32)

        # grad_input: FULL batch (chain rule to earlier layers)
        # d(RMSNorm)/dx = rsqrt * (I - x_hat * x_hat^T / D) * diag(weight)
        grad_normalized = grad_output_f32 * weight_f32
        grad_input = rsqrt_var * (
            grad_normalized
            - normalized * (grad_normalized * normalized).sum(-1, keepdim=True) / D
        )
        grad_input = grad_input.to(ctx.input_dtype)

        # grad_weight: TRAIN-ONLY slice with normalization correction.
        # grad_output carries 1/batch_total_tokens from the merged-batch loss.
        # RMSNorm effectively "selects all train samples", so we rescale by
        # batch_total_tokens / train_total_tokens to match a train-only loss
        # (i.e., what you'd get from forward/backward on just the train samples).
        if train_bs is not None and train_bs < grad_output.shape[0]:
            train_go = grad_output_f32[:train_bs]
            train_norm = normalized[:train_bs]
            grad_weight = (train_go * train_norm).flatten(0, -2).sum(0)
            if (state is not None
                    and state.batch_total_tokens_tensor is not None
                    and state.train_total_tokens_tensor is not None):
                scale = state.batch_total_tokens_tensor / state.train_total_tokens_tensor
                grad_weight = grad_weight * scale
        else:
            # Fallback: full batch (separate batch mode or no state)
            grad_weight = (grad_output_f32 * normalized).flatten(0, -2).sum(0)

        return grad_input, grad_weight, None, None


class GlobalSubsetEmbeddingBackward(Function):
    """
    Autograd Function for Embedding in GlobalSubset method (global curation).

    Accumulates per-sample influence scores using gather-dot-sum
    (the embedding analogue of the reduced ghost inner product).
    In one-pass mode, retains (grad_output, input_ids) for post-hoc assembly.
    """

    @staticmethod
    def forward(
        ctx,
        input_ids: "Tensor",
        weight: "Tensor",
        hook_manager: "GradientHook",
        layer_idx: int,
        padding_idx: int,
    ) -> "Tensor":
        """Forward pass: standard embedding lookup."""
        ctx.save_for_backward(input_ids, weight)
        ctx.hook_manager_ref = weakref.ref(hook_manager)
        ctx.layer_idx = layer_idx
        ctx.padding_idx = padding_idx
        return F.embedding(input_ids, weight, padding_idx=padding_idx if padding_idx >= 0 else None)

    @staticmethod
    def backward(
        ctx,
        grad_output: "Tensor",
    ) -> "Tuple[None, None, None, None, None]":
        """Backward pass: accumulate scores, retain data. No weight gradient."""
        from .utils import (
            compute_embedding_scores,
            compute_embedding_val_gradient,
            split_train_val_batch,
        )

        input_ids, weight = ctx.saved_tensors
        layer_idx = ctx.layer_idx

        hook_manager = ctx.hook_manager_ref()
        if hook_manager is None:
            return None, None, None, None, None

        state = hook_manager.selection_state
        if state is None:
            return None, None, None, None, None

        use_stored_val = getattr(state, '_use_stored_val', False)

        with torch.no_grad():
            variant = getattr(state, "selection_variant", "score")
            collect_optimizer_diagnostics = False
            if getattr(state, "optimizer_aware", False):
                diagnostic_predicate = getattr(
                    state, "should_collect_optimizer_diagnostics", None
                )
                collect_optimizer_diagnostics = bool(
                    callable(diagnostic_predicate)
                    and diagnostic_predicate()
                    and variant != "soft"
                )
            if variant == "random":
                if state.one_pass:
                    if use_stored_val:
                        hook_manager.retain_layer_data(layer_idx, grad_output, input_ids)
                    else:
                        train_go, _ = split_train_val_batch(
                            grad_output, state.train_batch_size
                        )
                        train_ids, _ = split_train_val_batch(
                            input_ids, state.train_batch_size
                        )
                        hook_manager.retain_layer_data(layer_idx, train_go, train_ids)
                return None, None, None, None, None

            if (
                variant == "muon_spectral"
                and not _muon_surrogate_includes_adamw_scores(state)
            ):
                # Embeddings are AdamW-managed.  They do not contribute to the
                # matrix-only global criterion, but retain their train factors
                # so the shared Muon-derived subset still updates them.
                if use_stored_val:
                    train_go, train_ids = grad_output, input_ids
                else:
                    train_go, _ = split_train_val_batch(
                        grad_output, state.train_batch_size
                    )
                    train_ids, _ = split_train_val_batch(
                        input_ids, state.train_batch_size
                    )
                state._append_diagnostic(
                    f"spectral/layer_{layer_idx}/adamw_score_omitted", 1.0
                )
                if state.one_pass:
                    hook_manager.retain_layer_data(
                        layer_idx, train_go, train_ids
                    )
                return None, None, None, None, None

            if use_stored_val:
                # Separate batch: entire batch is train, val grad from cache
                train_go = grad_output
                train_ids = input_ids
                val_grad_weight = hook_manager.val_cache.get_full(layer_idx)
                if val_grad_weight is None:
                    # No val data for this layer — skip scoring
                    if state.one_pass:
                        hook_manager.retain_layer_data(layer_idx, grad_output, input_ids)
                    return None, None, None, None, None
                score_correction = None
            else:
                # Merged batch: split into train/val
                train_go, val_go = split_train_val_batch(grad_output, state.train_batch_size)
                train_ids, val_ids = split_train_val_batch(input_ids, state.train_batch_size)
                # Compute val gradient by scattering val grad_output
                V, D = weight.shape
                val_accum_go = (
                    val_go.float()
                    if variant in ("soft", "muon_spectral")
                    else val_go
                )
                val_grad_weight = compute_embedding_val_gradient(
                    val_accum_go, val_ids, V, D, ctx.padding_idx
                )
                score_correction = state.score_correction

            if (
                variant == "muon_spectral"
                and bool((val_grad_weight.detach().float().square().sum() == 0).cpu())
            ):
                state._append_diagnostic(f"spectral/layer_{layer_idx}/zero_target", 1.0)
                if state.one_pass:
                    hook_manager.retain_layer_data(layer_idx, train_go, train_ids)
                return None, None, None, None, None

            if variant == "soft":
                target_scale = _target_standalone_scale(state).to(val_grad_weight.device)
                target_weight = val_grad_weight.detach().float() * target_scale
                if bool((target_weight.square().sum() == 0).detach().cpu()):
                    state._append_diagnostic(f"soft/layer_{layer_idx}/zero_target", 1.0)
                else:
                    objective, reference_scores = _make_soft_embedding_objective(
                        hook_manager, state, layer_idx,
                        train_go, train_ids, target_weight,
                        weight.shape[0], ctx.padding_idx,
                    )
                    state.add_soft_objective(objective, reference_scores)
                if state.one_pass:
                    hook_manager.retain_layer_data(layer_idx, train_go, train_ids)
                return None, None, None, None, None

            raw_scores = None
            if getattr(state, 'optimizer_aware', False):
                from .optimizer_aware import maybe_precondition_embedding_val

                raw_val_grad_weight = (
                    val_grad_weight if collect_optimizer_diagnostics else None
                )
                val_grad_weight, geometry = maybe_precondition_embedding_val(
                    hook_manager, state, layer_idx, val_grad_weight
                )
                if state._record_selections:
                    state._current_optimizer_geometry = geometry
                if raw_val_grad_weight is not None:
                    raw_scores = compute_embedding_scores(
                        train_go, train_ids, raw_val_grad_weight
                    )

            # Compute per-sample scores
            scores = compute_embedding_scores(train_go, train_ids, val_grad_weight)

            if score_correction is not None:
                scores = scores * score_correction
                if raw_scores is not None:
                    raw_scores = raw_scores * score_correction

            # Accumulate into GlobalSubsetState (no similarity for embedding)
            state.accumulate_precomputed_scores(
                scores,
                None,
                None,
                layer_idx=layer_idx,
                raw_scores_for_diag=raw_scores,
            )

            # One-pass: retain data for post-hoc gradient assembly
            if state.one_pass:
                if use_stored_val:
                    hook_manager.retain_layer_data(layer_idx, grad_output, input_ids)
                else:
                    # Merged batch: retain train portion only
                    hook_manager.retain_layer_data(layer_idx, train_go, train_ids)

        return None, None, None, None, None


class LayerWiseSubsetEmbeddingBackward(Function):
    """
    Autograd Function for Embedding in LayerWiseSubset method (per-layer curation).

    Computes scores, selects samples, and produces the curated gradient
    for the embedding layer in a single pass.
    """

    @staticmethod
    def forward(
        ctx,
        input_ids: "Tensor",
        weight: "Tensor",
        hook_manager: "GradientHook",
        layer_idx: int,
        padding_idx: int,
    ) -> "Tensor":
        """Forward pass: standard embedding lookup."""
        ctx.save_for_backward(input_ids, weight)
        ctx.hook_manager_ref = weakref.ref(hook_manager)
        ctx.layer_idx = layer_idx
        ctx.padding_idx = padding_idx
        return F.embedding(input_ids, weight, padding_idx=padding_idx if padding_idx >= 0 else None)

    @staticmethod
    def backward(
        ctx,
        grad_output: "Tensor",
    ) -> "Tuple[None, Optional[Tensor], None, None, None]":
        """Backward pass: score, select, produce curated grad_weight."""
        from .utils import (
            compute_embedding_scores,
            compute_embedding_val_gradient,
            compute_embedding_selected_gradients,
            split_train_val_batch,
        )

        input_ids, weight = ctx.saved_tensors
        layer_idx = ctx.layer_idx
        padding_idx = ctx.padding_idx
        V, D = weight.shape

        hook_manager = ctx.hook_manager_ref()
        if hook_manager is None:
            return None, None, None, None, None

        state = hook_manager.selection_state
        capture_val_mode = hook_manager.capture_val_mode
        use_stored_val = (
            state is not None and
            getattr(state, '_use_stored_val', False)
        )

        with torch.no_grad():
            # --- Val capture mode: store embedding val gradient ---
            if capture_val_mode:
                capture_go = grad_output
                accumulation_dtype = hook_manager.val_cache.accumulation_dtype
                if accumulation_dtype is not None:
                    capture_go = capture_go.to(accumulation_dtype)
                val_grad = compute_embedding_val_gradient(
                    capture_go, input_ids, V, D, padding_idx
                )
                hook_manager.val_cache.store_precomputed(layer_idx, val_grad)
                return None, None, None, None, None

            # --- No state: no curation ---
            if state is None:
                return None, None, None, None, None

            windowed = bool(getattr(state, "windowed_execution", False))
            if windowed:
                if not use_stored_val:
                    raise RuntimeError(
                        "Windowed embedding execution requires separate_batch"
                    )
                if state.window_phase == "replay":
                    from .advanced_solvers import (
                        weighted_embedding_gradient,
                        weighted_token_scale,
                    )

                    kind, decision = state.get_window_decision(layer_idx)
                    start, end = state.window_chunk_start, state.window_chunk_end
                    if kind == "weights":
                        global_weights = decision.to(
                            device=grad_output.device, dtype=torch.float32
                        )
                        scale = weighted_token_scale(
                            global_weights,
                            state.tokens_per_sample.detach().float().to(
                                global_weights.device
                            ),
                            state.batch_total_tokens_tensor.detach().float().to(
                                global_weights.device
                            ),
                        )
                        grad_weight = weighted_embedding_gradient(
                            grad_output,
                            input_ids,
                            global_weights[start:end],
                            num_embeddings=V,
                            padding_idx=None if padding_idx < 0 else padding_idx,
                            scale=scale,
                        )
                    else:
                        global_indices = decision.to(
                            device=grad_output.device, dtype=torch.long
                        )
                        local = global_indices[
                            (global_indices >= start) & (global_indices < end)
                        ] - start
                        scale = state._compute_scale_factor(global_indices).to(
                            grad_output.device
                        )
                        grad_weight = compute_embedding_selected_gradients(
                            grad_output,
                            input_ids,
                            local,
                            scale,
                            V,
                            D,
                            padding_idx,
                        )
                    return None, grad_weight.to(weight.dtype), None, None, None
                if state.window_phase != "score":
                    raise RuntimeError(f"Invalid window phase: {state.window_phase!r}")

            variant = getattr(state, "selection_variant", "score")
            if windowed and variant == "soft" and (
                layer_idx in state._window_zero_layers
            ):
                return None, None, None, None, None

            embedding_probe_cached = False
            collect_optimizer_diagnostics = False
            if getattr(state, "optimizer_aware", False):
                diagnostic_predicate = getattr(
                    state, "should_collect_optimizer_diagnostics", None
                )
                collect_optimizer_diagnostics = bool(
                    callable(diagnostic_predicate)
                    and diagnostic_predicate()
                    and variant != "soft"
                )
                if windowed:
                    from .optimizer_aware import (
                        has_cached_optimizer_aware_embedding_probe,
                    )

                    embedding_probe_cached = (
                        has_cached_optimizer_aware_embedding_probe(
                            state, layer_idx
                        )
                    )
            if variant == "random":
                if use_stored_val:
                    train_go, train_ids = grad_output, input_ids
                else:
                    train_go, _ = split_train_val_batch(
                        grad_output, state.train_batch_size
                    )
                    train_ids, _ = split_train_val_batch(
                        input_ids, state.train_batch_size
                    )
                scores = torch.zeros(
                    train_go.shape[0] if windowed else state.train_batch_size,
                    device=train_go.device,
                    dtype=torch.float32,
                )
                if windowed:
                    state.store_window_scores(layer_idx, scores)
                    return None, None, None, None, None
                selected_indices = _do_selection(
                    state, layer_idx, scores, None,
                    hook_manager.get_optimizer_group_keys()[layer_idx],
                )
                scale_factor = state._compute_scale_factor(selected_indices)
                grad_weight = compute_embedding_selected_gradients(
                    train_go, train_ids, selected_indices, scale_factor,
                    V, D, padding_idx,
                )
                return None, grad_weight.to(weight.dtype), None, None, None

            if (
                variant == "muon_spectral"
                and not _muon_surrogate_includes_adamw_scores(state)
            ):
                # There is no layer-local Muon score for an AdamW embedding.
                # Use its full candidate gradient rather than an arbitrary
                # zero-score top-k subset.
                if use_stored_val:
                    train_go, train_ids = grad_output, input_ids
                else:
                    train_go, _ = split_train_val_batch(
                        grad_output, state.train_batch_size
                    )
                    train_ids, _ = split_train_val_batch(
                        input_ids, state.train_batch_size
                    )
                state._append_diagnostic(
                    f"spectral/layer_{layer_idx}/adamw_full_batch", 1.0
                )
                # Under windowed execution train_go holds only the current
                # chunk, while the full-batch decision is over the logical
                # window. finalize_windowed_layerwise_selection rebuilds that
                # index set itself for layers marked here, and the return below
                # discards grad_weight anyway -- so building it from
                # logical-window indices against a chunk-sized tensor is both
                # dead work and an out-of-bounds gather. Mark and leave.
                if windowed:
                    state.mark_window_full_batch(layer_idx)
                    return None, None, None, None, None
                all_indices = torch.arange(
                    state.train_batch_size,
                    device=train_go.device,
                    dtype=torch.long,
                )
                scale_factor = state._compute_scale_factor(all_indices)
                grad_weight = compute_embedding_selected_gradients(
                    train_go, train_ids, all_indices, scale_factor,
                    V, D, padding_idx,
                )
                state._last_selected_indices = all_indices
                state._layer_selections.append(
                    (layer_idx, state.train_batch_size)
                )
                return None, grad_weight.to(weight.dtype), None, None, None

            # --- Get val gradient and train data ---
            if use_stored_val:
                train_go = grad_output
                train_ids = input_ids
                if embedding_probe_cached and not collect_optimizer_diagnostics:
                    val_grad_weight = None
                else:
                    val_grad_weight = hook_manager.val_cache.get_full(
                        layer_idx, device=train_go.device
                    )
                if val_grad_weight is None and not embedding_probe_cached:
                    return None, None, None, None, None
                score_correction = None
            else:
                train_go, val_go = split_train_val_batch(grad_output, state.train_batch_size)
                train_ids, val_ids = split_train_val_batch(input_ids, state.train_batch_size)
                val_accum_go = (
                    val_go.float()
                    if variant in ("soft", "muon_spectral")
                    else val_go
                )
                val_grad_weight = compute_embedding_val_gradient(
                    val_accum_go, val_ids, V, D, padding_idx
                )
                score_correction = state.score_correction

            if variant == "soft":
                target_weight = None
                if not embedding_probe_cached:
                    target_weight = (
                        val_grad_weight.detach().float()
                        * _target_standalone_scale(state).to(val_grad_weight.device)
                    )
                    if bool((target_weight.square().sum() == 0).detach().cpu()):
                        state._append_diagnostic(
                            f"soft/layer_{layer_idx}/zero_target", 1.0
                        )
                        if windowed:
                            state.mark_window_zero(layer_idx)
                        return None, None, None, None, None

                if windowed and float(
                    state.solver_config.get("soft_weighting_gamma", 0.0)
                ) == 0.0:
                    from .optimizer_aware import maybe_precondition_embedding_val
                    from .utils import compute_embedding_scores

                    target_probe, _ = maybe_precondition_embedding_val(
                        hook_manager,
                        state,
                        layer_idx,
                        target_weight,
                        discard_raw_target=True,
                        soft_candidate_map=True,
                    )
                    reference_scores = compute_embedding_scores(
                        train_go.float(), train_ids, target_probe
                    ).detach().float()
                    state.store_window_scores(layer_idx, reference_scores)
                    return None, None, None, None, None

                objective, reference_scores = _make_soft_embedding_objective(
                    hook_manager, state, layer_idx,
                    train_go, train_ids, target_weight, V, padding_idx,
                )
                if windowed:
                    state.store_window_scores(layer_idx, reference_scores)
                    return None, None, None, None, None
                weights = _optimize_layer_soft_weights(
                    state, layer_idx, objective, reference_scores
                )
                from .advanced_solvers import weighted_embedding_gradient, weighted_token_scale
                scale = weighted_token_scale(
                    weights,
                    state.tokens_per_sample.detach().float(),
                    _candidate_base_tokens(state),
                )
                grad_weight = weighted_embedding_gradient(
                    train_go, train_ids, weights.to(train_go.device),
                    num_embeddings=V,
                    padding_idx=None if padding_idx < 0 else padding_idx,
                    scale=scale,
                )
                return None, grad_weight.to(weight.dtype), None, None, None

            if (
                variant == "muon_spectral"
                and bool((val_grad_weight.detach().float().square().sum() == 0).cpu())
            ):
                state._append_diagnostic(f"spectral/layer_{layer_idx}/zero_target", 1.0)
                if windowed:
                    state.mark_window_zero(layer_idx)
                return None, None, None, None, None

            # --- Compute scores ---
            raw_scores = None
            if getattr(state, 'optimizer_aware', False):
                from .optimizer_aware import maybe_precondition_embedding_val

                raw_val_grad_weight = (
                    val_grad_weight if collect_optimizer_diagnostics else None
                )
                val_grad_weight, geometry = maybe_precondition_embedding_val(
                    hook_manager, state, layer_idx, val_grad_weight
                )
                if state._record_selections:
                    state._current_optimizer_geometry = geometry
                if raw_val_grad_weight is not None:
                    raw_scores = compute_embedding_scores(
                        train_go, train_ids, raw_val_grad_weight
                    )
            scores = compute_embedding_scores(train_go, train_ids, val_grad_weight)
            if score_correction is not None:
                scores = scores * score_correction
                if raw_scores is not None:
                    raw_scores = raw_scores * score_correction

            # --- Selection ---
            metric_group_key = hook_manager.get_optimizer_group_keys()[layer_idx]
            if windowed:
                state.store_window_scores(
                    layer_idx,
                    scores,
                    raw_scores=raw_scores,
                    geometry=(
                        geometry if getattr(state, "optimizer_aware", False) else None
                    ),
                )
                return None, None, None, None, None
            selected_indices = _do_selection(state, layer_idx, scores, None, metric_group_key)
            if raw_scores is not None:
                state.add_raw_opt_score_diagnostics(
                    metric_group_key,
                    raw_scores,
                    scores,
                    selected_indices,
                )

            # --- Gradient update for selected samples ---
            scale_factor = state._compute_scale_factor(selected_indices)
            grad_weight = compute_embedding_selected_gradients(
                train_go, train_ids, selected_indices, scale_factor,
                V, D, padding_idx,
            )
            grad_weight = grad_weight.to(weight.dtype)

        return None, grad_weight, None, None, None
