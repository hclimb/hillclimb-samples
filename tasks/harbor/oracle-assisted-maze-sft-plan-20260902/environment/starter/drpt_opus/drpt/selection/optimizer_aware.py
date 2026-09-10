"""
Optimizer-aware scoring utilities for group-wise data regularization.

The core selection update still uses raw gradients after samples are chosen.
These helpers only change the scoring geometry:
    Method A: score_i = <P_t g_i, g_val>
    Method B: score_i = <P_t g_i, P_t g_val>
where P_t is an optimizer-induced transform for the current parameter group.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .utils import compute_scores_and_similarity, compute_total_gradient, _materialize_gradients

if TYPE_CHECKING:
    from ..hook import GradientHook
    from .state import LayerWiseSubsetState


def _as_step_int(step) -> int:
    if torch.is_tensor(step):
        if step.numel() == 0:
            return 0
        return int(step.detach().cpu().item())
    return int(step or 0)


def _get_adamw_inv_rms(
    hook_manager: "GradientHook",
    param: Tensor,
    fallback_eps: float,
) -> Optional[Tensor]:
    """Return AdamW's diagonal inverse-RMS preconditioner for ``param``.

    Returns ``None`` before the optimizer has initialized state, which makes the
    caller fall back to identity geometry for the warmup/first step.
    """
    optimizer = getattr(hook_manager, "optimizer", None)
    if optimizer is None:
        return None

    if hasattr(optimizer, "get_param_state"):
        state = optimizer.get_param_state(param)
    else:
        state = optimizer.state.get(param)
    if not state:
        return None

    exp_avg_sq = state.get("exp_avg_sq")
    if exp_avg_sq is None or exp_avg_sq.shape != param.shape:
        return None

    step = _as_step_int(state.get("step", 0))
    if step <= 0:
        return None

    group = _optimizer_group_for_param(hook_manager, param)
    beta2 = group.get("betas", (0.9, 0.999))[1]
    eps = group.get("eps", fallback_eps)

    bias_correction2 = 1.0 - beta2 ** step
    if bias_correction2 <= 0:
        return None

    denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
    denom = denom.add(eps)
    return denom.reciprocal()


def _optimizer_group_for_param(
    hook_manager: "GradientHook",
    param: Tensor,
) -> dict:
    return getattr(hook_manager, "_optimizer_group_by_param_id", {}).get(id(param), {})


def _group_lr(
    hook_manager: "GradientHook",
    param: Tensor,
    default: float = 1.0,
) -> float:
    group = _optimizer_group_for_param(hook_manager, param)
    return float(group.get("lr", default))


def _spectral_concentration_penalty(matrices: Tensor, eps: float) -> Tensor:
    mats = matrices.to(torch.float32)
    fro_sq = mats.square().sum(dim=(-2, -1)).clamp_min(eps)
    spec_sq = torch.linalg.matrix_norm(mats, ord=2, dim=(-2, -1)).square()
    return (spec_sq / fro_sq).to(matrices.dtype)


def _muon_adjust_lr_scale(
    shape: torch.Size,
    adjust_lr_fn: Optional[str] = "original",
    enabled: bool = True,
) -> float:
    if not enabled or len(shape) != 2 or shape[1] == 0:
        return 1.0
    mode = "original" if adjust_lr_fn is None else str(adjust_lr_fn).lower()
    if mode in ("", "none", "false", "off", "disabled"):
        return 1.0
    if mode == "original":
        return math.sqrt(max(1.0, shape[0] / shape[1]))
    if mode == "match_rms_adamw":
        return 0.2 * math.sqrt(max(shape[0], shape[1]))
    raise ValueError(
        "muon_adjust_lr_fn must be one of: original, match_rms_adamw, none"
    )


def _muon_preconditioner_scale(
    param: Tensor,
    hook_manager: "GradientHook",
    config: dict,
) -> float:
    """Return OPUS Muon scalar kappa_t = eta_t * shape_scale * d q_t / d g_t."""
    group = _optimizer_group_for_param(hook_manager, param)
    lr = float(group.get("lr", 1.0))
    mu = float(group.get("momentum", config.get("muon_momentum", 0.95)))
    nesterov = bool(group.get("nesterov", config.get("muon_nesterov", True)))
    grad_coeff = (1.0 - mu * mu) if nesterov else (1.0 - mu)
    use_shape_scale = bool(group.get(
        "muon_lr_shape_scale",
        config.get("muon_lr_shape_scale", True),
    ))
    shape_scale = _muon_adjust_lr_scale(
        param.shape,
        group.get("adjust_lr_fn", config.get("muon_adjust_lr_fn", "original")),
        use_shape_scale,
    )
    return lr * grad_coeff * shape_scale


def _target_mode(config: dict) -> str:
    mode = str(config.get("target_mode", "opta")).lower().replace("-", "_")
    aliases = {
        "a": "opta",
        "method_a": "opta",
        "target_loss_majorization": "opta",
        "optimizer_induced": "opta",
        "optimizer_induced_feasible_set": "opta",
        "b": "optb",
        "method_b": "optb",
        "target_update_matching": "optb",
        "target_update_trajectory_matching": "optb",
        "optimizer_step_matching": "optb",
    }
    mode = aliases.get(mode, mode)
    if mode not in ("opta", "optb"):
        raise ValueError("optimizer-aware target_mode must be either 'opta' or 'optb'")
    return mode


def _resolve_linear_geometry(
    module: nn.Module,
    hook_manager: "GradientHook",
    config: dict,
) -> str:
    optimizer_type = str(config.get("optimizer_type", "adamw")).lower()
    if optimizer_type in ("adamw", "adamw_only", "adamw-only"):
        return "adamw"

    optimizer = getattr(hook_manager, "optimizer", None)
    if optimizer is not None and hasattr(optimizer, "get_param_optimizer_kind"):
        return optimizer.get_param_optimizer_kind(module.weight)

    geometry = str(config.get("matrix_geometry", "muon")).lower()
    if geometry == "auto":
        geometry = "muon" if isinstance(module, nn.Linear) and module.weight.ndim == 2 else "adamw"
    return geometry


def _get_muon_reference(
    hook_manager: "GradientHook",
    param: Tensor,
    proxy_grad: Tensor,
    config: dict,
) -> Tuple[Tensor, str]:
    """Construct OPUS-style frozen Muon reference direction for this step."""
    optimizer = getattr(hook_manager, "optimizer", None)
    if optimizer is not None and hasattr(optimizer, "get_param_state"):
        state = optimizer.get_param_state(param)
    else:
        state = optimizer.state.get(param, {}) if optimizer is not None else {}
    momentum_buffer = state.get("momentum_buffer")

    group = _optimizer_group_for_param(hook_manager, param)
    mu = float(group.get("momentum", config.get("muon_momentum", 0.95)))
    nesterov = bool(group.get("nesterov", config.get("muon_nesterov", True)))
    reference_mode = str(config.get("muon_reference", "momentum_proxy")).lower()

    if momentum_buffer is None or not torch.is_tensor(momentum_buffer):
        return proxy_grad.detach(), "proxy"

    momentum_buffer = momentum_buffer.to(device=proxy_grad.device, dtype=proxy_grad.dtype)
    if reference_mode == "momentum":
        return momentum_buffer.detach(), "momentum"
    if reference_mode == "proxy":
        return proxy_grad.detach(), "proxy"

    # Default: frozen lookahead reference using PyTorch Muon's source rule:
    #   buf_t = mu * buf_{t-1} + (1 - mu) * g_proxy
    #   update_ref = (1 - mu) * g_proxy + mu * buf_t   if nesterov
    #              = buf_t                             otherwise
    buf_t = mu * momentum_buffer + (1.0 - mu) * proxy_grad
    reference = (1.0 - mu) * proxy_grad + mu * buf_t if nesterov else buf_t
    return reference.detach(), "momentum_proxy"


def _apply_frozen_muon_left_preconditioner(
    matrices: Tensor,
    reference: Tensor,
    config: dict,
) -> Tensor:
    """
    Apply OPUS frozen Muon linearized left preconditioner:
        S_t G = a G + b A G + c A^2 G, A = q q^T, q = ref / ||ref||_F.

    This is implicit and does not materialize S_t. It still materializes A
    through products with q, so configs cap dimensions for dense full-rank runs.
    """
    eps = float(config.get("muon_eps", 1e-7))
    ref = reference.to(device=matrices.device, dtype=matrices.dtype)
    norm = ref.norm().clamp_min(eps)
    q = ref / norm

    a = float(config.get("muon_ns_a", 3.4445))
    b = float(config.get("muon_ns_b", -4.7750))
    c = float(config.get("muon_ns_c", 2.0315))

    q_t_g = torch.matmul(q.transpose(0, 1), matrices)
    a_g = torch.matmul(q, q_t_g)
    q_t_ag = torch.matmul(q.transpose(0, 1), a_g)
    a2_g = torch.matmul(q, q_t_ag)
    return a * matrices + b * a_g + c * a2_g


def _frozen_muon_step_matrix(work_ref: Tensor, config: dict) -> Tensor:
    """Return one frozen Newton-Schulz left factor for ``work_ref``."""
    a = float(config.get("muon_ns_a", 3.4445))
    b = float(config.get("muon_ns_b", -4.7750))
    c = float(config.get("muon_ns_c", 2.0315))
    gram = work_ref @ work_ref.transpose(0, 1)
    gram2 = gram @ gram
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return a * eye + b * gram + c * gram2


def _apply_frozen_muon_adjoint_to_target(
    target: Tensor,
    reference: Tensor,
    param: Tensor,
    hook_manager: "GradientHook",
    config: dict,
) -> Tensor:
    """
    Build the dual probe P*_t target for OPUS-style Muon scoring.

    The actual Muon code transposes tall matrices before applying the
    Newton-Schulz left polynomial. We mirror that orientation here. If the
    frozen operator on candidate gradients is P(G)=LGR, this returns
    P*(Y)=L^T Y R^T, so reduced-ghost scoring can compute <G_i, P*Y> without
    materializing per-sample candidate matrices.
    """
    steps = int(config.get("muon_steps", 5))
    if steps <= 0:
        return target

    orig_dtype = target.dtype
    eps = float(config.get("muon_eps", 1e-7))
    ref = reference.detach().to(device=target.device, dtype=torch.float32)
    work_ref = ref / ref.norm().clamp_min(eps)

    transposed = work_ref.shape[0] > work_ref.shape[1]
    if transposed:
        work_ref = work_ref.transpose(0, 1)

    probe = target.to(torch.float32)

    if transposed:
        # Candidate operator:
        #   H_0 = G^T, H_{k+1} = S_k H_k, P(G)=H_T^T.
        # Therefore P*(Y)=Y S_{T-1} ... S_0.
        right_adjoint = torch.eye(work_ref.shape[0], device=work_ref.device, dtype=work_ref.dtype)
        for _ in range(steps):
            factor = _frozen_muon_step_matrix(work_ref, config)
            right_adjoint = factor @ right_adjoint
            work_ref = factor @ work_ref
        probe = probe @ right_adjoint
    else:
        # Candidate operator:
        #   P(G)=S_{T-1} ... S_0 G.
        # Therefore P*(Y)=S_0^T ... S_{T-1}^T Y.
        left_operator = torch.eye(work_ref.shape[0], device=work_ref.device, dtype=work_ref.dtype)
        for _ in range(steps):
            factor = _frozen_muon_step_matrix(work_ref, config)
            left_operator = factor @ left_operator
            work_ref = factor @ work_ref
        probe = left_operator.transpose(0, 1) @ probe

    probe = probe * _muon_preconditioner_scale(param, hook_manager, config)

    return probe.to(orig_dtype)


def _apply_frozen_muon_preconditioner_to_target(
    target: Tensor,
    reference: Tensor,
    param: Tensor,
    hook_manager: "GradientHook",
    config: dict,
) -> Tensor:
    """
    Apply the same frozen Muon linearized operator P_t used for candidates.

    This is used by Method B, where the target is the hypothetical target-only
    optimizer step P_t g_target. The returned tensor is still only a scoring
    probe; the eventual selected update is assembled from raw gradients.
    """
    steps = int(config.get("muon_steps", 5))
    if steps <= 0:
        return target

    orig_dtype = target.dtype
    eps = float(config.get("muon_eps", 1e-7))
    ref = reference.detach().to(device=target.device, dtype=torch.float32)
    work_ref = ref / ref.norm().clamp_min(eps)

    transposed = work_ref.shape[0] > work_ref.shape[1]
    probe = target.to(torch.float32)
    if transposed:
        work_ref = work_ref.transpose(0, 1)
        probe = probe.transpose(0, 1)

    for _ in range(steps):
        factor = _frozen_muon_step_matrix(work_ref, config)
        probe = factor @ probe
        work_ref = factor @ work_ref

    if transposed:
        probe = probe.transpose(0, 1)

    probe = probe * _muon_preconditioner_scale(param, hook_manager, config)

    return probe.to(orig_dtype)


def _collect_raw_score_diagnostics(state: "LayerWiseSubsetState") -> bool:
    """Whether this outer step requested the optional raw-vs-OptA comparison."""
    predicate = getattr(state, "should_collect_optimizer_diagnostics", None)
    return bool(predicate()) if callable(predicate) else False


def _linear_probe_cache(state: "LayerWiseSubsetState") -> dict:
    cache = getattr(state, "_optimizer_linear_probe_cache", None)
    if cache is None:
        cache = {}
        setattr(state, "_optimizer_linear_probe_cache", cache)
    return cache


def has_cached_optimizer_aware_linear_probe(
    state: "LayerWiseSubsetState",
    layer_idx: int,
) -> bool:
    """Return whether this outer step already froze this layer's AdamW probe."""
    cache = getattr(state, "_optimizer_linear_probe_cache", None)
    return cache is not None and int(layer_idx) in cache


def _embedding_probe_cache(state: "LayerWiseSubsetState") -> dict:
    cache = getattr(state, "_optimizer_embedding_probe_cache", None)
    if cache is None:
        cache = {}
        setattr(state, "_optimizer_embedding_probe_cache", cache)
    return cache


def has_cached_optimizer_aware_embedding_probe(
    state: "LayerWiseSubsetState",
    layer_idx: int,
) -> bool:
    cache = getattr(state, "_optimizer_embedding_probe_cache", None)
    return cache is not None and int(layer_idx) in cache


def _window_probe_cache_enabled(
    state: "LayerWiseSubsetState",
    use_second_order: bool = False,
) -> bool:
    return bool(
        getattr(state, "windowed_execution", False)
        and getattr(state, "window_phase", None) == "score"
        and not use_second_order
    )


def _discard_raw_weight_target(
    hook_manager: "GradientHook",
    layer_idx: int,
) -> None:
    cache = getattr(hook_manager, "_val_cache", None)
    if cache is not None and hasattr(cache, "discard_weight_gradient"):
        cache.discard_weight_gradient(layer_idx)


def compute_optimizer_aware_linear_scores(
    hook_manager: "GradientHook",
    state: "LayerWiseSubsetState",
    layer_idx: int,
    train_grad_output: Tensor,
    train_input: Tensor,
    val_grad_output: Optional[Tensor],
    val_input: Optional[Tensor],
    val_grad_total: Optional[Tensor],
    use_second_order: bool,
    *,
    collect_raw_scores: Optional[bool] = None,
) -> Tuple[Tensor, Optional[Tensor], str, Optional[Tensor]]:
    """Compute optimizer-aware scores for a Linear weight matrix."""
    module = hook_manager._get_module_from_idx(layer_idx)
    config = getattr(state, "optimizer_aware_config", {}) or {}
    geometry = _resolve_linear_geometry(module, hook_manager, config)
    mode = _target_mode(config)

    if collect_raw_scores is None:
        collect_raw_scores = _collect_raw_score_diagnostics(state)
    collect_raw_scores = bool(collect_raw_scores)
    cache_probe = geometry == "adamw" and _window_probe_cache_enabled(
        state, use_second_order
    )
    cached_probe = None
    if cache_probe:
        cached_probe = _linear_probe_cache(state).get(int(layer_idx))

    # A cached first-order probe is the complete scoring target. The raw
    # target was intentionally released on non-diagnostic steps, so later
    # candidate chunks neither rebuild the AdamW map nor stage that target.
    if cached_probe is not None:
        target_probe, geometry_name = cached_probe
        target_probe = target_probe.to(
            device=train_grad_output.device, dtype=train_grad_output.dtype
        )
        scores, similarity = compute_scores_and_similarity(
            train_grad_output, train_input, None, None, target_probe, False,
        )
        raw_scores = None
        if collect_raw_scores:
            if val_grad_total is None:
                if val_grad_output is None or val_input is None:
                    raise ValueError(
                        "Raw OptA diagnostics require validation gradients"
                    )
                val_grad_total = compute_total_gradient(
                    val_grad_output.to(train_grad_output.dtype),
                    val_input.to(train_input.dtype),
                )
            else:
                val_grad_total = val_grad_total.to(train_grad_output.dtype)
            raw_scores, _ = compute_scores_and_similarity(
                train_grad_output, train_input, None, None,
                val_grad_total, False,
            )
        return scores, similarity, geometry_name, raw_scores

    if val_grad_total is None:
        if val_grad_output is None or val_input is None:
            raise ValueError("Optimizer-aware scoring requires validation gradients")
        val_grad_total = compute_total_gradient(
            val_grad_output.to(train_grad_output.dtype),
            val_input.to(train_input.dtype),
        )
    else:
        val_grad_total = val_grad_total.to(train_grad_output.dtype)

    if geometry == "identity":
        scores, similarity = compute_scores_and_similarity(
            train_grad_output, train_input, None, None,
            val_grad_total, use_second_order,
        )
        raw_scores = scores.detach() if collect_raw_scores else None
        return scores, similarity, f"{geometry}_{mode}", raw_scores

    if geometry == "adamw":
        raw_scores = None
        if collect_raw_scores:
            raw_scores, _ = compute_scores_and_similarity(
                train_grad_output, train_input, None, None,
                val_grad_total, False,
            )
        inv_rms = _get_adamw_inv_rms(
            hook_manager,
            module.weight,
            float(config.get("adam_eps", 1e-8)),
        )
        if inv_rms is None:
            scores, similarity = compute_scores_and_similarity(
                train_grad_output, train_input, None, None,
                val_grad_total, use_second_order,
            )
            geometry_name = f"adamw_identity_{mode}"
            if cache_probe:
                _linear_probe_cache(state)[int(layer_idx)] = (
                    val_grad_total.detach(), geometry_name
                )
                if not collect_raw_scores:
                    _discard_raw_weight_target(hook_manager, layer_idx)
            return scores, similarity, geometry_name, (
                raw_scores.detach() if raw_scores is not None else None
            )

        precond = inv_rms.to(device=val_grad_total.device, dtype=val_grad_total.dtype)
        precond = precond * _group_lr(hook_manager, module.weight)
        target_probe = val_grad_total * precond
        if mode == "optb":
            target_probe = target_probe * precond

        if not use_second_order:
            scores, similarity = compute_scores_and_similarity(
                train_grad_output, train_input, None, None, target_probe, False,
            )
            geometry_name = f"{geometry}_{mode}"
            if cache_probe:
                _linear_probe_cache(state)[int(layer_idx)] = (
                    target_probe.detach(), geometry_name
                )
                if not collect_raw_scores:
                    _discard_raw_weight_target(hook_manager, layer_idx)
            return scores, similarity, geometry_name, (
                raw_scores.detach() if raw_scores is not None else None
            )

        train_grads = _materialize_gradients(train_grad_output, train_input).reshape(
            train_grad_output.shape[0], *val_grad_total.shape
        )
        precond_train = train_grads * precond
        score_target = val_grad_total if mode == "opta" else val_grad_total * precond
        scores = torch.einsum("boi,oi->b", precond_train, score_target)
        flat = precond_train.flatten(start_dim=1)
        similarity = flat @ flat.T
        return scores, similarity, f"{geometry}_{mode}", (
            raw_scores.detach() if raw_scores is not None else None
        )

    if geometry != "muon":
        raise ValueError(
            f"Unknown optimizer-aware matrix geometry '{geometry}'. "
            "Use one of: muon, adamw, identity, auto."
        )

    reference, ref_source = _get_muon_reference(hook_manager, module.weight, val_grad_total, config)
    target = val_grad_total
    if mode == "optb":
        target = _apply_frozen_muon_preconditioner_to_target(
            val_grad_total,
            reference,
            module.weight,
            hook_manager,
            config,
        )
    dual_probe = _apply_frozen_muon_adjoint_to_target(
        target,
        reference,
        module.weight,
        hook_manager,
        config,
    )
    raw_scores = None
    if collect_raw_scores:
        raw_scores, _ = compute_scores_and_similarity(
            train_grad_output, train_input, None, None,
            val_grad_total, False,
        )
    
    if use_second_order:
        raise ValueError(
            "OptimizerAwareGroupWise with Muon geometry currently supports "
            "first-order reduced-ghost scoring only. Disable use_second_order "
            "or add a transformed-feature similarity implementation."
        )
    scores, similarity = compute_scores_and_similarity(
        train_grad_output, train_input, None, None,
        dual_probe, False,
    )

    spectral_lambda = float(config.get("spectral_lambda", 0.0))
    if spectral_lambda != 0.0:
        train_grads = _materialize_gradients(train_grad_output, train_input).reshape(
            train_grad_output.shape[0], *val_grad_total.shape
        )
        precond_train = _apply_frozen_muon_left_preconditioner(
            train_grads, reference, config
        )
        penalty = _spectral_concentration_penalty(
            precond_train,
            eps=float(config.get("spectral_eps", 1e-12)),
        )
        scores = scores - spectral_lambda * penalty

    return scores, similarity, f"{geometry}_{mode}_adjoint_{ref_source}", (
        raw_scores.detach() if raw_scores is not None else None
    )


def maybe_precondition_embedding_val(
    hook_manager: "GradientHook",
    state: "LayerWiseSubsetState",
    layer_idx: int,
    val_grad_weight: Optional[Tensor],
    *,
    discard_raw_target: Optional[bool] = None,
    soft_candidate_map: bool = False,
) -> Tuple[Tensor, str]:
    """Apply AdamW diagonal geometry to embedding validation gradients if available."""
    config = getattr(state, "optimizer_aware_config", {}) or {}
    geometry = (
        "adamw"
        if soft_candidate_map
        else str(config.get("vector_geometry", "adamw")).lower()
    )
    mode = _target_mode(config)
    if discard_raw_target is None:
        discard_raw_target = not _collect_raw_score_diagnostics(state)
    cache_probe = _window_probe_cache_enabled(state)
    cached_probe = None
    if cache_probe:
        cached_probe = _embedding_probe_cache(state).get(int(layer_idx))
    if cached_probe is not None:
        probe, geometry_name = cached_probe
        return probe, geometry_name
    if val_grad_weight is None:
        raise ValueError("Optimizer-aware embedding scoring requires a target gradient")

    if geometry in ("identity", "none"):
        probe, geometry_name = val_grad_weight, f"identity_{mode}"
        if cache_probe:
            _embedding_probe_cache(state)[int(layer_idx)] = (
                probe.detach(), geometry_name
            )
            if discard_raw_target:
                _discard_raw_weight_target(hook_manager, layer_idx)
        return probe, geometry_name
    if geometry != "adamw":
        return val_grad_weight, f"identity_{mode}"

    module = hook_manager._get_module_from_idx(layer_idx)
    inv_rms = _get_adamw_inv_rms(
        hook_manager,
        module.weight,
        float(config.get("adam_eps", 1e-8)),
    )
    if inv_rms is None:
        probe = val_grad_weight
        if soft_candidate_map:
            # Match _adamw_candidate_transform exactly. Soft's finite-step
            # solver is sensitive even to a layer-uniform objective scale.
            probe = probe * _group_lr(hook_manager, module.weight)
        geometry_name = f"adamw_identity_{mode}"
        if cache_probe:
            _embedding_probe_cache(state)[int(layer_idx)] = (
                probe.detach(), geometry_name
            )
            if discard_raw_target:
                _discard_raw_weight_target(hook_manager, layer_idx)
        return probe, geometry_name

    inv_rms = inv_rms.to(val_grad_weight.device, val_grad_weight.dtype)
    precond = inv_rms * _group_lr(hook_manager, module.weight)
    probe = val_grad_weight * precond
    if mode == "optb":
        probe = probe * precond
    geometry_name = f"adamw_{mode}"
    if cache_probe:
        _embedding_probe_cache(state)[int(layer_idx)] = (
            probe.detach(), geometry_name
        )
        if discard_raw_target:
            _discard_raw_weight_target(hook_manager, layer_idx)
    return probe, geometry_name
