"""Numerical building blocks for soft weighting and Muon spectral selection.

This module intentionally has no dependency on selection state or autograd hooks.  It
contains the small, independently testable pieces shared by layer-wise and global
selection:

* projection and projected-Adam optimization on the capped simplex;
* exact cardinality-k selection for token-normalized hard objectives;
* the live-state Muon candidate update used by the differentiable objective;
* target singular-mode extraction and projected-ghost spectral scores; and
* weighted gradient assembly from retained Linear/Embedding factors.

The target gradient is never transformed here.  Callers form objectives by pairing a
raw target gradient with one of the candidate transforms below.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor


SoftObjective = Callable[[Tensor], Union[Tensor, Tuple[Tensor, Tensor]]]
_TORCH_MM_HAS_OUT_DTYPE = "out_dtype" in (torch.mm.__doc__ or "")
_CUDA_NATIVE_BF16_BY_DEVICE: Dict[int, bool] = {}


def _validate_mass(num_items: int, mass: float) -> None:
    if num_items <= 0:
        raise ValueError("the capped simplex requires at least one item")
    if not math.isfinite(float(mass)):
        raise ValueError("capped-simplex mass must be finite")
    if mass < 0 or mass > num_items:
        raise ValueError(
            f"capped-simplex mass must lie in [0, {num_items}], got {mass}"
        )


def project_capped_simplex(values: Tensor, k: float) -> Tensor:
    """Euclidean projection onto ``{0 <= w <= 1, sum(w) = k}``.

    The projection has the form ``clamp(values - tau, 0, 1)``.  A monotone
    bisection determines ``tau``; a final active-set correction removes the few
    ulps of residual left by finite precision.  The operation is used outside the
    optimizer's autograd graph and therefore deliberately does not define a custom
    backward.
    """

    if values.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got shape {values.shape}")
    _validate_mass(values.numel(), float(k))
    if not torch.isfinite(values).all():
        raise ValueError("cannot project non-finite capped-simplex values")

    if k == 0:
        return torch.zeros_like(values)
    if k == values.numel():
        return torch.ones_like(values)

    # The vector is only one training batch wide (typically 8).  Project it on
    # CPU so the scalar bisection below does not turn every ``.item()`` into a
    # CUDA synchronization.  This costs one tiny D2H/H2D transfer per projected
    # Adam step instead of ~80 device synchronizations.
    work = values.detach().to(device="cpu", dtype=torch.float64)
    lower = (work.min() - 1.0).item()
    upper = work.max().item()
    target = float(k)
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        current = torch.clamp(work - midpoint, min=0.0, max=1.0).sum().item()
        if current > target:
            lower = midpoint
        else:
            upper = midpoint

    projected = torch.clamp(work - 0.5 * (lower + upper), min=0.0, max=1.0)

    # Correct any cast/bisection residual without violating the box.  Usually one
    # iteration is enough, but the loop also handles an exactly saturated face.
    for _ in range(4):
        residual = target - projected.sum().item()
        if abs(residual) <= 4.0 * torch.finfo(torch.float64).eps * max(1.0, target):
            break
        if residual > 0:
            free = projected < 1.0
            capacity = 1.0 - projected[free]
        else:
            free = projected > 0.0
            capacity = projected[free]
        if not bool(free.any()):
            break
        share = abs(residual) / int(free.sum().item())
        delta = torch.minimum(capacity, torch.full_like(capacity, share))
        if residual > 0:
            projected[free] += delta
        else:
            projected[free] -= delta

    return projected.to(device=values.device, dtype=values.dtype)


def project_capped_simplex_batched(values: Tensor, k: float) -> Tensor:
    """Project every row onto ``{0 <= w <= 1, sum(w) = k}``.

    This is the vectorized counterpart of :func:`project_capped_simplex`.  It
    deliberately uses the same float64 bisection and active-set correction, but
    transfers all rows together.  CUDA callers therefore pay one tiny D2H/H2D
    transfer pair per projected-Adam iteration instead of one pair per layer.
    """

    if values.ndim != 2:
        raise ValueError(f"values must be two-dimensional, got shape {values.shape}")
    num_rows, num_items = values.shape
    if num_rows <= 0:
        raise ValueError("batched projection requires at least one row")
    _validate_mass(num_items, float(k))

    work = values.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(work).all()):
        raise ValueError("cannot project non-finite capped-simplex values")
    if k == 0:
        return torch.zeros_like(values)
    if k == num_items:
        return torch.ones_like(values)

    lower = work.amin(dim=1) - 1.0
    upper = work.amax(dim=1)
    target = float(k)
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        current = torch.clamp(
            work - midpoint.unsqueeze(1), min=0.0, max=1.0
        ).sum(dim=1)
        move_lower = current > target
        lower = torch.where(move_lower, midpoint, lower)
        upper = torch.where(move_lower, upper, midpoint)

    projected = torch.clamp(
        work - (0.5 * (lower + upper)).unsqueeze(1), min=0.0, max=1.0
    )

    residual_tolerance = (
        4.0 * torch.finfo(torch.float64).eps * max(1.0, target)
    )
    for _ in range(4):
        residual = target - projected.sum(dim=1)
        active = residual.abs() > residual_tolerance
        if not bool(active.any()):
            break
        add_mass = residual > 0
        free = torch.where(
            add_mass.unsqueeze(1), projected < 1.0, projected > 0.0
        ) & active.unsqueeze(1)
        capacity = torch.where(
            add_mass.unsqueeze(1), 1.0 - projected, projected
        )
        free_count = free.sum(dim=1).clamp_min(1)
        share = residual.abs() / free_count
        delta = torch.minimum(capacity, share.unsqueeze(1))
        signed_delta = torch.where(add_mass.unsqueeze(1), delta, -delta)
        projected = torch.where(free, projected + signed_delta, projected)

    return projected.to(device=values.device, dtype=values.dtype)


def project_probability_simplex(values: Tensor) -> Tensor:
    """Euclidean projection onto ``{p >= 0, sum(p) = 1}``.

    Unlike :func:`project_capped_simplex`, this projector does not impose an
    explicit coordinate-wise upper cap.  (``p_i <= 1`` follows implicitly from
    non-negativity and unit mass.)  The sorted-threshold algorithm is evaluated
    on CPU in float64 because the vector is only one candidate batch wide.
    """

    if values.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got shape {values.shape}")
    if values.numel() == 0:
        raise ValueError("the probability simplex requires at least one item")
    if not torch.isfinite(values).all():
        raise ValueError("cannot project non-finite probability-simplex values")

    work = values.detach().to(device="cpu", dtype=torch.float64)
    sorted_values, _ = torch.sort(work, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=0) - 1.0
    positions = torch.arange(
        1, work.numel() + 1, device="cpu", dtype=torch.float64
    )
    active = sorted_values - cumulative / positions > 0
    rho = int(torch.nonzero(active, as_tuple=False)[-1].item())
    threshold = cumulative[rho] / float(rho + 1)
    projected = torch.clamp(work - threshold, min=0.0)

    # Remove the tiny sum residual left by the threshold arithmetic.
    projected = projected / projected.sum()
    return projected.to(device=values.device, dtype=values.dtype)


def project_probability_simplex_batched(values: Tensor) -> Tensor:
    """Project each row onto the probability simplex in one batched transfer."""

    if values.ndim != 2:
        raise ValueError(f"values must be two-dimensional, got shape {values.shape}")
    num_rows, num_items = values.shape
    if num_rows <= 0 or num_items <= 0:
        raise ValueError("batched probability projection requires non-empty rows")
    work = values.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(work).all()):
        raise ValueError("cannot project non-finite probability-simplex values")

    sorted_values, _ = torch.sort(work, dim=1, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=1) - 1.0
    positions = torch.arange(
        1, num_items + 1, device="cpu", dtype=torch.float64
    ).unsqueeze(0)
    active = sorted_values - cumulative / positions > 0
    rho = active.sum(dim=1) - 1
    threshold = cumulative.gather(1, rho.unsqueeze(1)).squeeze(1) / (
        rho.to(torch.float64) + 1.0
    )
    projected = torch.clamp(work - threshold.unsqueeze(1), min=0.0)
    projected = projected / projected.sum(dim=1, keepdim=True)
    return projected.to(device=values.device, dtype=values.dtype)


def _resolve_soft_constraint(n: int, k: int, constraint: str) -> Tuple[str, float]:
    """Validate a soft feasible-set name and return its optimization mass."""

    _validate_mass(n, float(k))
    if isinstance(k, bool) or int(k) != k or k < 1:
        raise ValueError("soft weighting requires integer k >= 1")
    normalized = str(constraint).strip().lower().replace("-", "_")
    if normalized == "capped_simplex":
        return normalized, float(k)
    if normalized == "probability_simplex":
        return normalized, 1.0
    raise ValueError(
        "soft weighting constraint must be 'capped_simplex' or "
        "'probability_simplex'"
    )


def _project_soft_constraint(
    values: Tensor, constraint: str, mass: float
) -> Tensor:
    if constraint == "capped_simplex":
        return project_capped_simplex(values, mass)
    if constraint == "probability_simplex":
        if mass != 1.0:
            raise ValueError("probability-simplex mass must equal one")
        return project_probability_simplex(values)
    raise AssertionError(f"unresolved soft constraint: {constraint}")


def _project_soft_constraint_batched(
    values: Tensor, constraint: str, mass: float
) -> Tensor:
    if constraint == "capped_simplex":
        return project_capped_simplex_batched(values, mass)
    if constraint == "probability_simplex":
        if mass != 1.0:
            raise ValueError("probability-simplex mass must equal one")
        return project_probability_simplex_batched(values)
    raise AssertionError(f"unresolved soft constraint: {constraint}")


def _split_soft_objective(value: Any, gamma: float) -> Tuple[Tensor, Tensor, Tensor]:
    """Return penalized objective, alignment, and update norm squared."""

    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(
                "soft objective tuples must be (alignment, update_norm_squared)"
            )
        alignment, update_norm_sq = value
    else:
        alignment = value
        if gamma != 0.0:
            raise ValueError(
                "gamma is nonzero, but objective_fn did not return "
                "(alignment, update_norm_squared)"
            )
        update_norm_sq = torch.zeros((), device=alignment.device, dtype=alignment.dtype)

    if not torch.is_tensor(alignment) or alignment.numel() != 1:
        raise ValueError("objective_fn alignment must be a scalar tensor")
    if not torch.is_tensor(update_norm_sq) or update_norm_sq.numel() != 1:
        raise ValueError("objective_fn update norm must be a scalar tensor")
    objective = alignment - 0.5 * gamma * update_norm_sq
    return objective.reshape(()), alignment.reshape(()), update_norm_sq.reshape(())


def _objective_and_gradient(
    objective_fn: SoftObjective,
    weights: Tensor,
    gamma: float,
    *,
    need_gradient: bool,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Evaluate one soft objective, optionally releasing its graph via a VJP."""

    context = torch.enable_grad() if need_gradient else torch.no_grad()
    with context:
        raw_value = objective_fn(weights)
        objective, _, _ = _split_soft_objective(raw_value, gamma)
        if need_gradient:
            if objective.requires_grad:
                gradient = torch.autograd.grad(
                    objective,
                    weights,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )[0]
            else:
                gradient = torch.zeros_like(weights)
        else:
            gradient = None
    return objective.detach(), None if gradient is None else gradient.detach()


def _soft_weight_diagnostics(
    weights: Tensor,
    *,
    initial_objective: float,
    best_objective: float,
    iterations: int,
    converged: bool,
    boundary_tolerance: float,
    reference_topk: Optional[Tensor],
    feasible_set: str = "capped_simplex",
    constraint_mass: Optional[float] = None,
    reference_k: Optional[int] = None,
) -> Dict[str, Any]:
    work = weights.detach().to(torch.float64)
    mass = work.sum().item()
    probabilities = work / max(mass, torch.finfo(work.dtype).tiny)
    positive = probabilities > 0
    entropy = float(-(probabilities[positive] * probabilities[positive].log()).sum().item())
    ess = float(1.0 / probabilities.square().sum().clamp_min(torch.finfo(work.dtype).tiny).item())
    near_zero = work <= boundary_tolerance
    near_one = work >= 1.0 - boundary_tolerance
    expected_mass = mass if constraint_mass is None else float(constraint_mass)
    diagnostics: Dict[str, Any] = {
        "initial_objective": initial_objective,
        "final_objective": best_objective,
        "objective_improvement": best_objective - initial_objective,
        "iterations": int(iterations),
        "converged": bool(converged),
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(weights.numel()) if weights.numel() > 1 else 0.0,
        "ess": ess,
        "near_zero_fraction": float(near_zero.to(torch.float32).mean().item()),
        "near_one_fraction": float(near_one.to(torch.float32).mean().item()),
        "boundary_fraction": float((near_zero | near_one).to(torch.float32).mean().item()),
        "feasible_set": feasible_set,
        "constraint_mass": expected_mass,
        "mass_residual": abs(mass - expected_mass),
        "probability_simplex": float(feasible_set == "probability_simplex"),
        "explicit_upper_cap": float(feasible_set == "capped_simplex"),
    }
    if reference_topk is not None:
        k = (
            int(round(mass)) if reference_k is None else int(reference_k)
        )
        k = min(k, work.numel())
        soft_topk = torch.topk(work, k=min(k, work.numel())).indices.cpu()
        reference = reference_topk.detach().flatten().cpu()
        overlap = len(set(soft_topk.tolist()).intersection(reference.tolist()))
        diagnostics["topk_overlap"] = overlap / max(1, min(k, reference.numel()))
    return diagnostics


def optimize_soft_weights(
    objective_fn: SoftObjective,
    n: int,
    k: int,
    *,
    device: Union[torch.device, str],
    steps: int = 20,
    lr: float = 0.1,
    tolerance: float = 1e-5,
    patience: int = 3,
    gamma: float = 0.0,
    adam_betas: Tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    boundary_tolerance: float = 1e-4,
    reference_topk: Optional[Tensor] = None,
    constraint: str = "capped_simplex",
) -> Tuple[Tensor, Dict[str, Any]]:
    """Maximize a differentiable objective on a configured simplex.

    Only ``w`` participates in autograd: callers should close over detached
    train/target factors.  The function always optimizes float32 weights, starts
    at the uniform feasible point, performs projected Adam ascent, and returns the
    best finite iterate.  A tuple-valued objective enables the optional trust
    region term: ``(alignment, update_norm_squared)``.
    """

    constraint, constraint_mass = _resolve_soft_constraint(n, k, constraint)
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if lr <= 0 or not math.isfinite(lr):
        raise ValueError("lr must be positive and finite")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be non-negative and finite")
    if patience < 1:
        raise ValueError("patience must be at least one")
    if gamma < 0 or not math.isfinite(gamma):
        raise ValueError("gamma must be non-negative and finite")

    weights = torch.full(
        (n,), constraint_mass / n, device=torch.device(device), dtype=torch.float32,
        requires_grad=True,
    )
    current_objective, current_gradient = _objective_and_gradient(
        objective_fn, weights, gamma, need_gradient=True
    )
    first = current_objective
    if not torch.isfinite(first):
        raise FloatingPointError("initial soft-weighting objective is non-finite")

    initial_objective = float(first.detach().item())
    best_objective = initial_objective
    best_weights = weights.detach().clone()
    previous_finite_objective = initial_objective
    beta1, beta2 = adam_betas
    first_moment = torch.zeros_like(weights)
    second_moment = torch.zeros_like(weights)
    stale_steps = 0
    converged = False
    completed_steps = 0

    for step in range(1, steps + 1):
        if (
            not torch.isfinite(current_objective)
            or current_gradient is None
            or not torch.isfinite(current_gradient).all()
        ):
            break

        with torch.no_grad():
            first_moment.mul_(beta1).add_(current_gradient, alpha=1.0 - beta1)
            second_moment.mul_(beta2).addcmul_(
                current_gradient, current_gradient, value=1.0 - beta2
            )
            first_hat = first_moment / (1.0 - beta1**step)
            second_hat = second_moment / (1.0 - beta2**step)
            proposal = weights + lr * first_hat / (second_hat.sqrt() + adam_eps)
            weights.copy_(
                _project_soft_constraint(proposal, constraint, constraint_mass)
            )

        candidate, candidate_gradient = _objective_and_gradient(
            objective_fn, weights, gamma, need_gradient=step < steps
        )
        completed_steps = step
        if not torch.isfinite(candidate):
            break
        candidate_value = float(candidate.detach().item())
        if candidate_value > best_objective:
            best_objective = candidate_value
            best_weights = weights.detach().clone()

        relative_improvement = max(0.0, candidate_value - previous_finite_objective)
        relative_improvement /= max(1.0, abs(previous_finite_objective))
        if relative_improvement <= tolerance:
            stale_steps += 1
        else:
            stale_steps = 0
        previous_finite_objective = candidate_value
        if stale_steps >= patience:
            converged = True
            break
        current_objective = candidate
        current_gradient = candidate_gradient

    diagnostics = _soft_weight_diagnostics(
        best_weights,
        initial_objective=initial_objective,
        best_objective=best_objective,
        iterations=completed_steps,
        converged=converged,
        boundary_tolerance=boundary_tolerance,
        reference_topk=reference_topk,
        feasible_set=constraint,
        constraint_mass=constraint_mass,
        reference_k=int(k),
    )
    return best_weights, diagnostics


def optimize_batched_linear_soft_weights(
    scores: Tensor,
    tokens_per_sample: Tensor,
    base_tokens: Union[Tensor, float, int],
    k: int,
    *,
    steps: int = 20,
    lr: float = 0.1,
    tolerance: float = 1e-5,
    patience: int = 3,
    adam_betas: Tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    boundary_tolerance: float = 1e-4,
    reference_topk: Optional[Tensor] = None,
    constraint: str = "capped_simplex",
) -> Tuple[Tensor, List[Dict[str, Any]]]:
    """Solve independent gamma=0 linear Soft objectives in one layer batch.

    Row ``l`` maximizes

    ``base_tokens * dot(w_l, scores_l) / dot(w_l, tokens_per_sample)``

    under the configured simplex.  This is the exact windowed AdamW/identity
    objective represented by a stored score vector.  Initialization, Adam state,
    best-iterate tracking, patience, and early stopping remain independent for
    every layer; only the small tensor operations and projection transfers are
    batched.  The objective's analytic gradient is mathematically identical to
    the autograd derivative used by :func:`optimize_soft_weights`.
    """

    if scores.ndim != 2:
        raise ValueError(
            f"scores must have shape [layers, candidates], got {scores.shape}"
        )
    num_layers, n = scores.shape
    if num_layers <= 0:
        raise ValueError("scores must contain at least one layer")
    constraint, constraint_mass = _resolve_soft_constraint(n, k, constraint)
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if lr <= 0 or not math.isfinite(lr):
        raise ValueError("lr must be positive and finite")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be non-negative and finite")
    if patience < 1:
        raise ValueError("patience must be at least one")
    if tokens_per_sample.ndim != 1 or tokens_per_sample.numel() != n:
        raise ValueError("tokens_per_sample must have one entry per candidate")
    if reference_topk is not None and (
        reference_topk.ndim != 2 or reference_topk.shape[0] != num_layers
    ):
        raise ValueError("reference_topk must have shape [layers, selected]")

    device = scores.device
    layer_scores = scores.detach().to(device=device, dtype=torch.float32)
    tokens = tokens_per_sample.detach().to(device=device, dtype=torch.float32)
    base = torch.as_tensor(base_tokens, device=device, dtype=torch.float32)
    if base.numel() != 1:
        raise ValueError("base_tokens must be a scalar")
    if device.type == "cpu":
        if not bool(torch.isfinite(tokens).all()) or bool((tokens < 0).any()):
            raise ValueError("token counts must be finite and non-negative")
        if not bool(torch.isfinite(base)) or float(base.detach().item()) <= 0:
            raise ValueError("base_tokens must be a positive finite scalar")

    def evaluate(current: Tensor) -> Tuple[Tensor, Tensor]:
        denominator = (current * tokens.unsqueeze(0)).sum(dim=1)
        if device.type == "cpu" and (
            not bool(torch.isfinite(denominator).all())
            or bool((denominator <= 0).any())
        ):
            raise ValueError("soft weights select no valid tokens")
        scale = base.reshape(()) / denominator
        numerator = (current * layer_scores).sum(dim=1)
        objective = scale * numerator
        gradient = (
            scale.unsqueeze(1) * layer_scores
            - (objective / denominator).unsqueeze(1) * tokens.unsqueeze(0)
        )
        return objective, gradient

    weights = torch.full(
        (num_layers, n),
        constraint_mass / n,
        device=device,
        dtype=torch.float32,
    )
    current_objective, current_gradient = evaluate(weights)
    initial_values = current_objective.detach().to(
        device="cpu", dtype=torch.float64
    )
    if not bool(torch.isfinite(initial_values).all()):
        raise FloatingPointError(
            "initial batched soft-weighting objective is non-finite"
        )

    best_values = initial_values.clone()
    previous_values = initial_values.clone()
    best_weights = weights.clone()
    first_moment = torch.zeros_like(weights)
    second_moment = torch.zeros_like(weights)
    stale_steps = torch.zeros(num_layers, dtype=torch.int64)
    completed_steps = torch.zeros(num_layers, dtype=torch.int64)
    converged = torch.zeros(num_layers, dtype=torch.bool)
    active = torch.isfinite(current_gradient).all(dim=1).detach().cpu()
    beta1, beta2 = adam_betas

    for step in range(1, steps + 1):
        if not bool(active.any()):
            break
        active_device = active.to(device=device).unsqueeze(1)
        next_first = first_moment.clone()
        next_first.mul_(beta1).add_(current_gradient, alpha=1.0 - beta1)
        next_second = second_moment.clone()
        next_second.mul_(beta2).addcmul_(
            current_gradient, current_gradient, value=1.0 - beta2
        )
        first_moment = torch.where(active_device, next_first, first_moment)
        second_moment = torch.where(active_device, next_second, second_moment)
        first_hat = first_moment / (1.0 - beta1**step)
        second_hat = second_moment / (1.0 - beta2**step)
        proposal = weights + lr * first_hat / (second_hat.sqrt() + adam_eps)
        projected = _project_soft_constraint_batched(
            proposal, constraint, constraint_mass
        )
        weights = torch.where(active_device, projected, weights)

        candidate, candidate_gradient = evaluate(weights)
        completed_steps[active] = step
        candidate_values = candidate.detach().to(
            device="cpu", dtype=torch.float64
        )
        finite_candidate = torch.isfinite(candidate_values)
        evaluated = active & finite_candidate
        better = evaluated & (candidate_values > best_values)
        if bool(better.any()):
            best_values[better] = candidate_values[better]
            best_mask = better.to(device=device).unsqueeze(1)
            best_weights = torch.where(best_mask, weights, best_weights)

        improvement = torch.clamp(
            candidate_values - previous_values, min=0.0
        )
        relative_improvement = improvement / torch.maximum(
            torch.ones_like(previous_values), previous_values.abs()
        )
        next_stale = torch.where(
            relative_improvement <= tolerance,
            stale_steps + 1,
            torch.zeros_like(stale_steps),
        )
        stale_steps[evaluated] = next_stale[evaluated]
        previous_values[evaluated] = candidate_values[evaluated]
        newly_converged = evaluated & (stale_steps >= patience)
        converged |= newly_converged
        gradient_finite = (
            torch.isfinite(candidate_gradient).all(dim=1).detach().cpu()
        )
        active = evaluated & ~newly_converged & gradient_finite
        current_objective = candidate
        current_gradient = candidate_gradient

    best_weights_cpu = best_weights.detach().cpu()
    references_cpu = (
        None if reference_topk is None else reference_topk.detach().cpu()
    )
    diagnostics: List[Dict[str, Any]] = []
    for layer_idx in range(num_layers):
        diagnostics.append(
            _soft_weight_diagnostics(
                best_weights_cpu[layer_idx],
                initial_objective=float(initial_values[layer_idx]),
                best_objective=float(best_values[layer_idx]),
                iterations=int(completed_steps[layer_idx]),
                converged=bool(converged[layer_idx]),
                boundary_tolerance=boundary_tolerance,
                reference_topk=(
                    None
                    if references_cpu is None
                    else references_cpu[layer_idx]
                ),
                feasible_set=constraint,
                constraint_mass=constraint_mass,
                reference_k=int(k),
            )
        )
    return best_weights, diagnostics


def _streaming_objective_and_gradient(
    objective_fns: Tuple[SoftObjective, ...],
    weights: Tensor,
    gamma: float,
    *,
    need_gradient: bool,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Evaluate layer closures one at a time and immediately release each graph."""

    total_value = torch.zeros((), device=weights.device, dtype=torch.float32)
    total_gradient = torch.zeros_like(weights) if need_gradient else None
    for objective_fn in objective_fns:
        context = torch.enable_grad() if need_gradient else torch.no_grad()
        with context:
            raw_value = objective_fn(weights)
            objective, _, _ = _split_soft_objective(raw_value, gamma)
            if need_gradient:
                if objective.requires_grad:
                    layer_gradient = torch.autograd.grad(
                        objective, weights, retain_graph=False, create_graph=False,
                        allow_unused=True,
                    )[0]
                else:
                    layer_gradient = torch.zeros_like(weights)
            else:
                layer_gradient = None
        if need_gradient:
            if layer_gradient is not None:
                total_gradient.add_(layer_gradient.detach().to(total_gradient.dtype))
        total_value.add_(objective.detach().to(total_value.dtype))
    return total_value, total_gradient


def optimize_soft_weights_streaming(
    objective_fns: Iterable[SoftObjective],
    n: int,
    k: int,
    *,
    device: Union[torch.device, str],
    steps: int = 20,
    lr: float = 0.1,
    tolerance: float = 1e-5,
    patience: int = 3,
    gamma: float = 0.0,
    adam_betas: Tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    boundary_tolerance: float = 1e-4,
    reference_topk: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, Any]]:
    """Streaming counterpart of :func:`optimize_soft_weights`.

    Every closure represents one layer/objective contribution.  During each
    iteration its gradient with respect to the shared ``w`` is accumulated and
    the layer graph is released before the next closure runs.  This preserves the
    global shared-weight objective without retaining a model-sized graph.
    """

    closures = tuple(objective_fns)
    if not closures:
        raise ValueError("objective_fns must contain at least one layer closure")
    _validate_mass(n, float(k))
    if int(k) != k or k < 1:
        raise ValueError("soft weighting requires integer k >= 1")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if lr <= 0 or not math.isfinite(lr):
        raise ValueError("lr must be positive and finite")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be non-negative and finite")
    if patience < 1:
        raise ValueError("patience must be at least one")
    if gamma < 0 or not math.isfinite(gamma):
        raise ValueError("gamma must be non-negative and finite")

    weights = torch.full(
        (n,), float(k) / n, device=torch.device(device), dtype=torch.float32,
        requires_grad=True,
    )
    # Compute the initial value and gradient together.  The old implementation
    # evaluated every projected point once without grad (to update best/early
    # stopping) and then evaluated that exact same point again with grad at the
    # beginning of the next Adam iteration.  Carrying the already-computed
    # gradient forward produces identical projected-Adam iterates while nearly
    # halving expensive nonlinear objective forwards (notably Muon's NS map).
    current_objective, current_gradient = _streaming_objective_and_gradient(
        closures, weights, gamma, need_gradient=True
    )
    first = current_objective
    if not torch.isfinite(first):
        raise FloatingPointError("initial soft-weighting objective is non-finite")

    initial_objective = float(first.item())
    best_objective = initial_objective
    best_weights = weights.detach().clone()
    previous_finite_objective = initial_objective
    first_moment = torch.zeros_like(weights)
    second_moment = torch.zeros_like(weights)
    beta1, beta2 = adam_betas
    stale_steps = 0
    converged = False
    completed_steps = 0

    for step in range(1, steps + 1):
        if (
            not torch.isfinite(current_objective)
            or current_gradient is None
            or not torch.isfinite(current_gradient).all()
        ):
            break
        with torch.no_grad():
            first_moment.mul_(beta1).add_(current_gradient, alpha=1.0 - beta1)
            second_moment.mul_(beta2).addcmul_(
                current_gradient, current_gradient, value=1.0 - beta2
            )
            first_hat = first_moment / (1.0 - beta1**step)
            second_hat = second_moment / (1.0 - beta2**step)
            proposal = weights + lr * first_hat / (second_hat.sqrt() + adam_eps)
            weights.copy_(project_capped_simplex(proposal, k))

        # The final projected point needs only its objective.  All earlier
        # points also compute the gradient that the next iteration consumes.
        candidate, candidate_gradient = _streaming_objective_and_gradient(
            closures, weights, gamma, need_gradient=step < steps
        )
        completed_steps = step
        if not torch.isfinite(candidate):
            break
        candidate_value = float(candidate.item())
        if candidate_value > best_objective:
            best_objective = candidate_value
            best_weights = weights.detach().clone()

        relative_improvement = max(0.0, candidate_value - previous_finite_objective)
        relative_improvement /= max(1.0, abs(previous_finite_objective))
        if relative_improvement <= tolerance:
            stale_steps += 1
        else:
            stale_steps = 0
        previous_finite_objective = candidate_value
        if stale_steps >= patience:
            converged = True
            break
        current_objective = candidate
        current_gradient = candidate_gradient

    diagnostics = _soft_weight_diagnostics(
        best_weights,
        initial_objective=initial_objective,
        best_objective=best_objective,
        iterations=completed_steps,
        converged=converged,
        boundary_tolerance=boundary_tolerance,
        reference_topk=reference_topk,
    )
    return best_weights, diagnostics


ExplicitSoftObjective = Callable[
    [Tensor],
    Union[
        Tuple[Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor, Tensor],
    ],
]


def optimize_soft_weights_with_grad(
    objective_and_grad_fn: ExplicitSoftObjective,
    n: int,
    k: int,
    *,
    device: Union[torch.device, str],
    steps: int = 20,
    lr: float = 0.1,
    tolerance: float = 1e-5,
    patience: int = 3,
    gamma: float = 0.0,
    adam_betas: Tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    boundary_tolerance: float = 1e-4,
    reference_topk: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, Any]]:
    """Projected Adam driven by an explicit objective-and-gradient callback.

    With ``gamma=0`` the callback returns ``(objective, gradient)``.  To apply the
    update-norm penalty it returns
    ``(alignment, update_norm_sq, alignment_grad, update_norm_sq_grad)``.
    This interface is useful when the caller already streams and reduces layer
    derivatives itself.
    """

    _validate_mass(n, float(k))
    if int(k) != k or k < 1:
        raise ValueError("soft weighting requires integer k >= 1")
    if steps < 0 or patience < 1:
        raise ValueError("steps must be non-negative and patience at least one")
    if lr <= 0 or not math.isfinite(lr):
        raise ValueError("lr must be positive and finite")
    if tolerance < 0 or gamma < 0 or not math.isfinite(tolerance + gamma):
        raise ValueError("tolerance and gamma must be finite and non-negative")

    def evaluate(current: Tensor) -> Tuple[Tensor, Tensor]:
        result = objective_and_grad_fn(current.detach())
        if len(result) == 2:
            if gamma != 0.0:
                raise ValueError(
                    "nonzero gamma requires explicit callback output "
                    "(alignment, norm_sq, alignment_grad, norm_sq_grad)"
                )
            value, gradient = result
        elif len(result) == 4:
            alignment, norm_sq, alignment_grad, norm_grad = result
            value = alignment - 0.5 * gamma * norm_sq
            gradient = alignment_grad - 0.5 * gamma * norm_grad
        else:
            raise ValueError("explicit soft callback must return two or four tensors")
        return (
            torch.as_tensor(value, device=current.device, dtype=torch.float32).reshape(()),
            torch.as_tensor(gradient, device=current.device, dtype=torch.float32),
        )

    weights = torch.full(
        (n,), float(k) / n, device=torch.device(device), dtype=torch.float32
    )
    current_objective, current_gradient = evaluate(weights)
    if (
        not torch.isfinite(current_objective)
        or current_gradient.shape != weights.shape
        or not torch.isfinite(current_gradient).all()
    ):
        raise FloatingPointError("initial explicit soft objective/gradient is non-finite")
    initial_objective = float(current_objective.item())
    best_objective = initial_objective
    best_weights = weights.clone()
    previous_finite_objective = initial_objective
    first_moment = torch.zeros_like(weights)
    second_moment = torch.zeros_like(weights)
    beta1, beta2 = adam_betas
    stale_steps = 0
    converged = False
    completed_steps = 0

    for step in range(1, steps + 1):
        if (
            not torch.isfinite(current_objective)
            or current_gradient.shape != weights.shape
            or not torch.isfinite(current_gradient).all()
        ):
            break
        first_moment.mul_(beta1).add_(current_gradient, alpha=1.0 - beta1)
        second_moment.mul_(beta2).addcmul_(
            current_gradient, current_gradient, value=1.0 - beta2
        )
        first_hat = first_moment / (1.0 - beta1**step)
        second_hat = second_moment / (1.0 - beta2**step)
        proposal = weights + lr * first_hat / (second_hat.sqrt() + adam_eps)
        weights = project_capped_simplex(proposal, k)

        candidate, candidate_gradient = evaluate(weights)
        completed_steps = step
        if not torch.isfinite(candidate):
            break
        candidate_value = float(candidate.item())
        if candidate_value > best_objective:
            best_objective = candidate_value
            best_weights = weights.clone()
        relative_improvement = max(0.0, candidate_value - previous_finite_objective)
        relative_improvement /= max(1.0, abs(previous_finite_objective))
        stale_steps = stale_steps + 1 if relative_improvement <= tolerance else 0
        previous_finite_objective = candidate_value
        if stale_steps >= patience:
            converged = True
            break
        current_objective = candidate
        current_gradient = candidate_gradient

    diagnostics = _soft_weight_diagnostics(
        best_weights,
        initial_objective=initial_objective,
        best_objective=best_objective,
        iterations=completed_steps,
        converged=converged,
        boundary_tolerance=boundary_tolerance,
        reference_topk=reference_topk,
    )
    return best_weights, diagnostics


def weighted_token_scale(
    weights: Tensor,
    tokens_per_sample: Tensor,
    base_tokens: Union[Tensor, float, int],
) -> Tensor:
    """Return ``T_base / sum_i(w_i T_i)`` for weighted-token normalization."""

    if weights.ndim != 1 or tokens_per_sample.ndim != 1:
        raise ValueError("weights and tokens_per_sample must be one-dimensional")
    if weights.numel() != tokens_per_sample.numel():
        raise ValueError("weights and tokens_per_sample must have equal length")
    tokens = tokens_per_sample.to(device=weights.device, dtype=weights.dtype)
    denominator = torch.dot(weights, tokens)
    base = torch.as_tensor(base_tokens, device=weights.device, dtype=weights.dtype)
    if base.numel() != 1:
        raise ValueError("base_tokens must be a scalar")

    # These validations are useful for the public CPU helper and unit tests, but
    # Python bool/.item checks would force a GPU synchronization on every layer
    # and every soft-solver iterate.  Runtime selection constructs token counts
    # from a boolean label mask and validates that their total is positive once
    # in SelectionState.set_token_counts; on CUDA, invalid denominators therefore
    # simply propagate non-finite values to the solver's once-per-iterate guard.
    if weights.device.type == "cpu" and (
        not bool(torch.isfinite(tokens).all()) or bool((tokens < 0).any())
    ):
        raise ValueError("token counts must be finite and non-negative")
    if weights.device.type == "cpu" and (
        not bool(torch.isfinite(base)) or float(base.detach().item()) <= 0
    ):
        raise ValueError("base_tokens must be a positive finite scalar")
    if weights.device.type == "cpu" and (
        not bool(torch.isfinite(denominator))
        or float(denominator.detach().item()) <= 0
    ):
        raise ValueError("soft weights select no valid tokens")
    return base.reshape(()) / denominator


def muon_live_candidate_transform(
    aggregate_gradient: Tensor,
    momentum_buffer: Optional[Tensor] = None,
    *,
    optimizer_dtype: Optional[torch.dtype] = None,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    eps: float = 1e-7,
    lr: float = 1.0,
    shape_lr_scale: bool = True,
    adjust_lr_fn: Optional[str] = "original",
) -> Tensor:
    """Apply the live-state Muon candidate map used by the actual optimizer.

    If ``m`` is the previous momentum buffer, the matrix sent to Newton--Schulz
    is ``mu^2 m + (1-mu^2)G`` with Nesterov and
    ``mu m + (1-mu)G`` otherwise.  The returned update also includes the optimizer
    learning rate and matrix-shape multiplier.
    """

    if aggregate_gradient.ndim != 2:
        raise ValueError("Muon candidate transform requires a matrix gradient")
    if not 0.0 <= momentum < 1.0:
        raise ValueError("Muon momentum must lie in [0, 1)")
    if ns_steps < 0:
        raise ValueError("ns_steps must be non-negative")
    if eps <= 0 or not math.isfinite(eps):
        raise ValueError("eps must be positive and finite")
    if aggregate_gradient.device.type == "cpu" and not bool(
        torch.isfinite(aggregate_gradient).all()
    ):
        raise FloatingPointError("Muon aggregate gradient is non-finite")

    # Match the dtype boundary of the live optimizer exactly.  The SFT model is
    # normally bf16, so its momentum update and Nesterov lookahead are rounded
    # in bf16 before Newton--Schulz promotes internally to float32.  Keeping w
    # itself float32 still gives a differentiable objective, while this cast
    # makes the surrogate see the same q_t that optimizer.step() will see.
    if optimizer_dtype is None:
        optimizer_dtype = (
            momentum_buffer.dtype
            if momentum_buffer is not None
            else aggregate_gradient.dtype
        )
    if not optimizer_dtype.is_floating_point:
        raise ValueError("optimizer_dtype must be floating point")
    work = aggregate_gradient.to(optimizer_dtype)
    if momentum_buffer is None:
        previous = torch.zeros_like(work)
    else:
        if momentum_buffer.shape != aggregate_gradient.shape:
            raise ValueError("Muon momentum buffer shape does not match gradient")
        if momentum_buffer.device.type == "cpu" and not bool(
            torch.isfinite(momentum_buffer).all()
        ):
            raise FloatingPointError("Muon momentum buffer is non-finite")
        previous = momentum_buffer.detach().to(device=work.device, dtype=work.dtype)

    # Use the optimizer's two operations rather than only their algebraically
    # simplified form.  They are equal in real arithmetic, but bf16 rounds once
    # when forming m_t and again when forming the Nesterov lookahead q_t.
    updated_buffer = previous.mul(momentum).add(
        work, alpha=1.0 - momentum
    )
    if nesterov:
        candidate = work.mul(1.0 - momentum).add(
            updated_buffer, alpha=momentum
        )
    else:
        candidate = updated_buffer
    # Lazy import avoids a hook -> advanced_solvers -> optimizer -> hook cycle at
    # module import time while still reusing the optimizer's exact implementation.
    from ..optimizer import _muon_adjust_lr_scale, zeropower_via_newton_schulz

    update = zeropower_via_newton_schulz(candidate, steps=ns_steps, eps=eps)
    shape_scale = _muon_adjust_lr_scale(
        aggregate_gradient.shape, adjust_lr_fn, shape_lr_scale
    )
    return update * (float(lr) * shape_scale)


def _randomized_svd(
    matrix: Tensor,
    rank: int,
    oversample: int,
    power_iters: int,
    seed: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    m, n = matrix.shape
    sketch_rank = min(min(m, n), rank + oversample)
    generator = torch.Generator(device=matrix.device)
    generator.manual_seed(int(seed))
    omega = torch.randn(
        n, sketch_rank, generator=generator, device=matrix.device, dtype=matrix.dtype
    )
    q, _ = torch.linalg.qr(matrix @ omega, mode="reduced")
    for _ in range(power_iters):
        q, _ = torch.linalg.qr(matrix @ (matrix.T @ q), mode="reduced")
    small = q.T @ matrix
    small_u, singular_values, vh = torch.linalg.svd(small, full_matrices=False)
    u = q @ small_u
    keep = min(rank, singular_values.numel())
    return u[:, :keep], singular_values[:keep], vh[:keep]


def compute_spectral_modes_with_values(
    target: Tensor,
    *,
    rank: int = 32,
    full_svd_max_dim: int = 256,
    rtol: float = 1e-6,
    oversample: int = 8,
    power_iters: int = 2,
    seed: int = 0,
) -> Tuple[Tensor, Tensor, Tensor, int]:
    """Extract active target modes and singular values.

    Small matrices use a full SVD.  Larger matrices use a deterministic
    randomized rank-``rank`` approximation.  Singular values at or below
    ``rtol * beta_max`` are discarded.  The return value is
    ``(U, V, beta, active_rank)``; singular values are kept separate from the
    unit singular vectors so callers can choose either ``alpha_r = 1`` or
    ``alpha_r = beta_r`` without changing the projected-ghost calculation.  A
    finite all-zero target returns empty mode matrices and an empty beta vector.
    """

    if target.ndim != 2:
        raise ValueError("spectral target must be a matrix")
    if rank < 1 or full_svd_max_dim < 1:
        raise ValueError("rank and full_svd_max_dim must be positive")
    if oversample < 0 or power_iters < 0:
        raise ValueError("oversample and power_iters must be non-negative")
    if rtol < 0 or not math.isfinite(rtol):
        raise ValueError("rtol must be finite and non-negative")
    if not torch.isfinite(target).all():
        raise FloatingPointError("target contains non-finite values before SVD")

    work = target.detach().to(torch.float32)
    m, n = work.shape
    min_dim = min(m, n)
    if min_dim == 0:
        return (
            work.new_empty((m, 0)),
            work.new_empty((n, 0)),
            work.new_empty((0,)),
            0,
        )

    if min_dim <= full_svd_max_dim:
        u, singular_values, vh = torch.linalg.svd(work, full_matrices=False)
    else:
        u, singular_values, vh = _randomized_svd(
            work, rank=rank, oversample=oversample,
            power_iters=power_iters, seed=seed,
        )
    if not (
        torch.isfinite(u).all()
        and torch.isfinite(singular_values).all()
        and torch.isfinite(vh).all()
    ):
        raise FloatingPointError("SVD produced a non-finite value")
    if singular_values.numel() == 0 or float(singular_values[0].item()) == 0.0:
        return (
            work.new_empty((m, 0)),
            work.new_empty((n, 0)),
            work.new_empty((0,)),
            0,
        )

    active = singular_values > rtol * singular_values[0]
    u = u[:, active]
    v = vh[active].T
    beta = singular_values[active]
    active_rank = int(active.sum().item())
    return u, v, beta, active_rank


def compute_spectral_modes(
    target: Tensor,
    *,
    rank: int = 32,
    full_svd_max_dim: int = 256,
    rtol: float = 1e-6,
    oversample: int = 8,
    power_iters: int = 2,
    seed: int = 0,
) -> Tuple[Tensor, Tensor, int]:
    """Extract active target singular directions as ``U[:, r], V[:, r]``.

    This backward-compatible wrapper intentionally discards the singular values;
    it implements the original ``alpha_r = 1`` surrogate.
    """

    u, v, _beta, active_rank = compute_spectral_modes_with_values(
        target,
        rank=rank,
        full_svd_max_dim=full_svd_max_dim,
        rtol=rtol,
        oversample=oversample,
        power_iters=power_iters,
        seed=seed,
    )
    return u, v, active_rank


def spectral_linear_mode_support(
    train_grad_output: Tensor,
    train_input: Tensor,
    u: Tensor,
    v: Tensor,
) -> Tensor:
    """Compute ``relu(u_r^T G_i v_r)`` without materializing ``G_i``."""

    if train_grad_output.ndim not in (2, 3) or train_input.ndim != train_grad_output.ndim:
        raise ValueError("linear factors must both be rank two or both be rank three")
    if train_grad_output.shape[:-1] != train_input.shape[:-1]:
        raise ValueError("linear factor batch/token dimensions do not match")
    if u.ndim != 2 or v.ndim != 2 or u.shape[1] != v.shape[1]:
        raise ValueError("U and V must be mode matrices with the same rank")
    if train_grad_output.shape[-1] != u.shape[0] or train_input.shape[-1] != v.shape[0]:
        raise ValueError("mode dimensions do not match linear factors")

    dtype = torch.float32
    left = train_grad_output.to(dtype) @ u.to(device=train_grad_output.device, dtype=dtype)
    right = train_input.to(dtype) @ v.to(device=train_input.device, dtype=dtype)
    support = left * right
    if support.ndim == 3:
        support = support.sum(dim=1)
    return support.clamp_min(0.0)


def spectral_linear_scores(
    train_grad_output: Tensor,
    train_input: Tensor,
    u: Tensor,
    v: Tensor,
    alpha: Union[float, Tensor] = 1.0,
) -> Tensor:
    """Return the modular Muon spectral score for every candidate sample."""

    support = spectral_linear_mode_support(train_grad_output, train_input, u, v)
    if torch.is_tensor(alpha):
        mode_weights = alpha.to(device=support.device, dtype=support.dtype)
        if mode_weights.ndim == 0:
            return support.sum(dim=1) * mode_weights
        if mode_weights.ndim != 1 or mode_weights.numel() != support.shape[1]:
            raise ValueError("alpha must be scalar or have one entry per mode")
        return support @ mode_weights
    return support.sum(dim=1) * float(alpha)


def _spectral_mode_weights(
    mode_support: Tensor,
    alpha: Union[float, Tensor],
) -> Tensor:
    """Validate and materialize non-negative per-mode surrogate weights."""

    if torch.is_tensor(alpha):
        weights = alpha.to(device=mode_support.device, dtype=mode_support.dtype)
        if weights.ndim == 0:
            weights = weights.expand(mode_support.shape[1])
        elif weights.ndim != 1 or weights.numel() != mode_support.shape[1]:
            raise ValueError("alpha must be scalar or have one entry per mode")
    else:
        if not math.isfinite(float(alpha)):
            raise ValueError("alpha must be finite")
        weights = mode_support.new_full((mode_support.shape[1],), float(alpha))
    if not bool(torch.isfinite(weights).all()):
        raise FloatingPointError("spectral mode weights contain non-finite values")
    if bool((weights < 0).any()):
        raise ValueError("spectral mode weights must be non-negative")
    return weights


def spectral_saturated_objective(
    mode_support: Tensor,
    selected_indices: Tensor,
    alpha: Union[float, Tensor] = 1.0,
) -> Tensor:
    r"""Evaluate the concave-over-modular spectral objective.

    The value is

    ``sum_r alpha_r * log1p(sum_{i in S} mode_support[i, r])``.

    Non-negative support and mode weights make this set function normalized,
    monotone, and submodular.
    """

    if mode_support.ndim != 2:
        raise ValueError("mode_support must have shape [num_samples, num_modes]")
    if not bool(torch.isfinite(mode_support).all()):
        raise FloatingPointError("spectral mode support contains non-finite values")
    if bool((mode_support < 0).any()):
        raise ValueError("spectral mode support must be non-negative")
    if selected_indices.ndim != 1:
        raise ValueError("selected_indices must be one-dimensional")
    if selected_indices.numel() and (
        bool((selected_indices < 0).any())
        or bool((selected_indices >= mode_support.shape[0]).any())
    ):
        raise ValueError("selected index is out of range")
    weights = _spectral_mode_weights(mode_support, alpha)
    if selected_indices.numel() == 0:
        coverage = mode_support.new_zeros((mode_support.shape[1],))
    else:
        coverage = mode_support[selected_indices.to(mode_support.device)].sum(dim=0)
    return torch.dot(weights, torch.log1p(coverage))


def seeded_spectral_saturation_greedy(
    mode_support: Tensor,
    k: int,
    *,
    alpha: Union[float, Tensor] = 1.0,
    seed: int,
) -> Tensor:
    r"""Greedily maximize the monotone saturated spectral objective.

    This is the standard cardinality-constrained greedy solver for

    ``F(S) = sum_r alpha_r log1p(sum_{i in S} a[i, r])``.

    It never materializes sample gradient matrices: its input is only the
    projected-ghost support table ``[num_samples, active_rank]``.  A seeded,
    fixed priority permutation resolves exact marginal-gain ties at every step.
    """

    if mode_support.ndim != 2:
        raise ValueError("mode_support must have shape [num_samples, num_modes]")
    n = mode_support.shape[0]
    if isinstance(k, bool) or not isinstance(k, int) or k < 0 or k > n:
        raise ValueError("k must lie between zero and the number of samples")
    if not bool(torch.isfinite(mode_support).all()):
        raise FloatingPointError("spectral mode support contains non-finite values")
    if bool((mode_support < 0).any()):
        raise ValueError("spectral mode support must be non-negative")
    weights = _spectral_mode_weights(mode_support, alpha)
    if k == 0:
        return torch.empty(0, device=mode_support.device, dtype=torch.long)

    generator = torch.Generator(device=mode_support.device)
    generator.manual_seed(int(seed))
    priority = torch.randperm(n, generator=generator, device=mode_support.device)
    selected = torch.empty(k, device=mode_support.device, dtype=torch.long)
    unavailable = torch.zeros(n, device=mode_support.device, dtype=torch.bool)
    coverage = mode_support.new_zeros((mode_support.shape[1],))

    for step in range(k):
        before = torch.log1p(coverage)
        gains = (
            torch.log1p(coverage.unsqueeze(0) + mode_support) - before.unsqueeze(0)
        ) @ weights
        gains = gains.masked_fill(unavailable, -torch.inf)
        # Stable sorting of the fixed seeded permutation gives reproducible tie
        # handling without perturbing genuinely distinct floating-point gains.
        order = torch.argsort(gains[priority], descending=True, stable=True)
        chosen = priority[order[0]]
        selected[step] = chosen
        unavailable[chosen] = True
        coverage = coverage + mode_support[chosen]

    return selected


def seeded_topk(scores: Tensor, k: int, seed: int) -> Tensor:
    """Exact top-k with a deterministic seeded ordering for tied scores."""

    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if k < 0 or k > scores.numel():
        raise ValueError("k must lie between zero and the number of scores")
    if not torch.isfinite(scores).all():
        raise FloatingPointError("cannot select from non-finite scores")
    if k == 0:
        return torch.empty(0, device=scores.device, dtype=torch.long)
    generator = torch.Generator(device=scores.device)
    generator.manual_seed(int(seed))
    permutation = torch.randperm(scores.numel(), generator=generator, device=scores.device)
    # Stable sorting retains the seeded permutation exactly within equal-score runs,
    # without perturbing close but genuinely different scores with artificial noise.
    order = torch.argsort(scores[permutation], descending=True, stable=True)
    return permutation[order[:k]]


def exact_token_normalized_topk(
    scores: Tensor,
    tokens_per_sample: Tensor,
    k: int,
    seed: int,
) -> Tensor:
    """Solve the exact cardinality-``k`` token-normalized selection problem.

    The returned set maximizes

    ``sum(scores[i] for i in S) / sum(tokens_per_sample[i] for i in S)``

    over all ``|S| = k``.  This objective is not modular when token counts vary,
    so ordinary score top-k is generally not exact.  Candidate batches are small
    (normally eight items), making exhaustive enumeration both simple and exact.

    Individual zero-token items are allowed, but a subset with zero total tokens
    is infeasible.  If no positive-token subset exists, the function raises rather
    than silently producing an update with an undefined normalization.  Equal
    objective values are resolved by a seeded permutation of the exhaustive
    combinations; returned indices themselves are in ascending order.
    """

    if scores.ndim != 1 or tokens_per_sample.ndim != 1:
        raise ValueError("scores and tokens_per_sample must be one-dimensional")
    if scores.numel() != tokens_per_sample.numel():
        raise ValueError("scores and tokens_per_sample must have equal length")
    n = scores.numel()
    if n == 0:
        raise ValueError("token-normalized selection requires at least one item")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1 or k > n:
        raise ValueError("k must be an integer between one and the number of scores")
    if scores.is_complex() or tokens_per_sample.is_complex():
        raise ValueError("scores and token counts must be real-valued")
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("cannot select from non-finite scores")
    if (
        not bool(torch.isfinite(tokens_per_sample).all())
        or bool((tokens_per_sample < 0).any())
    ):
        raise ValueError("token counts must be finite and non-negative")

    # Enumerate canonical, sorted index tuples once, then permute the rows to make
    # torch.argmax's first-maximum rule a deterministic seeded tie-break.  The
    # objective itself is evaluated in float64 to avoid changing the exact winner
    # merely because the input scores or counts were stored in bf16.
    combinations = torch.tensor(
        list(itertools.combinations(range(n), k)), dtype=torch.long
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    combinations = combinations[
        torch.randperm(combinations.shape[0], generator=generator)
    ].to(device=scores.device)

    work_scores = scores.detach().to(dtype=torch.float64)
    work_tokens = tokens_per_sample.detach().to(
        device=scores.device, dtype=torch.float64
    )
    numerators = work_scores[combinations].sum(dim=1)
    denominators = work_tokens[combinations].sum(dim=1)
    feasible = denominators > 0
    if not bool(feasible.any()):
        raise ValueError("no cardinality-k subset contains any valid tokens")

    safe_denominators = torch.where(
        feasible, denominators, torch.ones_like(denominators)
    )
    objectives = numerators / safe_denominators
    if not bool(torch.isfinite(objectives[feasible]).all()):
        raise FloatingPointError("token-normalized subset objective is non-finite")
    objectives = objectives.masked_fill(~feasible, -torch.inf)
    return combinations[torch.argmax(objectives)]


def _broadcast_sample_weights(values: Tensor, weights: Tensor) -> Tensor:
    shape = (weights.numel(),) + (1,) * (values.ndim - 1)
    return weights.to(device=values.device, dtype=values.dtype).reshape(shape)


def _weighted_accumulation_dtype(*values: Tensor) -> torch.dtype:
    """Choose a contraction dtype without rounding continuous weights early."""
    dtype = values[0].dtype
    for value in values[1:]:
        dtype = torch.promote_types(dtype, value.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _normalize_replay_precision(replay_precision: str) -> str:
    mode = str(replay_precision).strip().lower().replace("-", "_")
    if mode not in ("fp32", "bf16_fp32"):
        raise ValueError(
            "replay_precision must be 'fp32' or 'bf16_fp32'"
        )
    return mode


def _can_use_bf16_fp32_mm(grad_output: Tensor, input: Tensor) -> bool:
    """Whether CUDA can contract bf16 operands into an fp32 output."""

    if not (
        grad_output.device.type == "cuda"
        and input.device == grad_output.device
        and grad_output.dtype == torch.bfloat16
        and input.dtype == torch.bfloat16
        and _TORCH_MM_HAS_OUT_DTYPE
    ):
        return False
    device_index = grad_output.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    if device_index not in _CUDA_NATIVE_BF16_BY_DEVICE:
        major, _minor = torch.cuda.get_device_capability(device_index)
        _CUDA_NATIVE_BF16_BY_DEVICE[device_index] = major >= 8
    return _CUDA_NATIVE_BF16_BY_DEVICE[device_index]


class _BF16FP32WeightedLinearContraction(torch.autograd.Function):
    """Differentiable weighted bf16 GEMM with an fp32 output.

    ``torch.mm(..., out_dtype=torch.float32)`` currently has no autograd
    derivative. Muon Soft nevertheless needs a gradient only with respect to
    the continuous sample weights; its retained activation factors are detached.
    This custom VJP therefore keeps the fast mixed-precision forward while
    computing the weight-only derivative explicitly in float32.

    The bf16 cast in the forward uses the usual straight-through dtype-cast
    derivative. The backward contracts the saved bf16 input values with the
    fp32 upstream matrix, then performs the sample reduction in fp32. No
    gradients are constructed for the potentially large retained factors.
    """

    @staticmethod
    def forward(
        ctx,
        grad_output: Tensor,
        input: Tensor,
        weights: Tensor,
    ) -> Tensor:
        output_dim = grad_output.shape[-1]
        input_dim = input.shape[-1]
        work_go = grad_output.float()
        weighted_go_bf16 = (
            work_go * _broadcast_sample_weights(work_go, weights)
        ).to(torch.bfloat16)
        ctx.save_for_backward(grad_output, input)
        ctx.weights_dtype = weights.dtype
        return torch.mm(
            weighted_go_bf16.reshape(-1, output_dim).transpose(0, 1),
            input.reshape(-1, input_dim),
            out_dtype=torch.float32,
        )

    @staticmethod
    def backward(ctx, grad_matrix: Tensor):
        grad_output, input = ctx.saved_tensors
        output_dim = grad_output.shape[-1]
        input_dim = input.shape[-1]

        # For C[o, i] = sum_p weighted_go[p, o] * input[p, i],
        # dL/dweighted_go[p, o] = sum_i input[p, i] * dL/dC[o, i].
        # Use fp32 here: casting the upstream derivative to bf16 would add a
        # second, avoidable approximation to the solver trajectory.
        grad_weighted_go = torch.mm(
            input.reshape(-1, input_dim).float(),
            grad_matrix.reshape(output_dim, input_dim).float().transpose(0, 1),
        )
        per_position = (
            grad_weighted_go
            * grad_output.reshape(-1, output_dim).float()
        )
        if grad_output.ndim == 3:
            grad_weights = per_position.reshape(
                grad_output.shape[0], grad_output.shape[1], output_dim
            ).sum(dim=(1, 2))
        else:
            grad_weights = per_position.sum(dim=1)
        return None, None, grad_weights.to(dtype=ctx.weights_dtype)


def weighted_linear_gradients(
    grad_output: Tensor,
    input: Tensor,
    weights: Tensor,
    *,
    scale: Union[Tensor, float] = 1.0,
    has_bias: bool = True,
    replay_precision: str = "fp32",
) -> Tuple[Tensor, Optional[Tensor]]:
    """Assemble weighted Linear weight/bias gradients from retained factors.

    bf16_fp32 keeps continuous weights, normalization, bias reduction, and the
    contraction output in fp32, while using bf16 operands only for the large
    Linear contraction. This can introduce the intended bf16-scale numerical
    perturbation to a Soft solver trajectory; unsupported devices/dtypes use
    the exact fp32 path.
    """

    if grad_output.ndim not in (2, 3) or input.ndim != grad_output.ndim:
        raise ValueError("linear factors must both be rank two or both be rank three")
    if grad_output.shape[:-1] != input.shape[:-1]:
        raise ValueError("linear factor batch/token dimensions do not match")
    if weights.ndim != 1 or weights.numel() != grad_output.shape[0]:
        raise ValueError("weights must have one entry per sample")
    replay_precision = _normalize_replay_precision(replay_precision)

    if replay_precision == "bf16_fp32" and _can_use_bf16_fp32_mm(
        grad_output, input
    ):
        scalar = torch.as_tensor(
            scale, device=grad_output.device, dtype=torch.float32
        )
        if torch.is_grad_enabled() and weights.requires_grad:
            # Solver path: out_dtype GEMM has no native derivative, so use the
            # weight-only custom VJP above. Bias remains an ordinary fp32
            # reduction, and multiplying by scalar outside the custom Function
            # preserves the derivative of token normalization.
            grad_weight = _BF16FP32WeightedLinearContraction.apply(
                grad_output, input, weights
            ) * scalar
            if has_bias:
                work_go = grad_output.float()
                weighted_go_fp32 = work_go * _broadcast_sample_weights(
                    work_go, weights
                )
                if grad_output.ndim == 3:
                    grad_bias = weighted_go_fp32.sum(dim=(0, 1)) * scalar
                else:
                    grad_bias = weighted_go_fp32.sum(dim=0) * scalar
            else:
                grad_bias = None
            return grad_weight, grad_bias

        # Apply the continuous weights before the sole lossy cast. Bias and
        # token normalization therefore remain identical to the fp32 path.
        weighted_go_fp32 = grad_output.float()
        weighted_go_fp32.mul_(
            _broadcast_sample_weights(weighted_go_fp32, weights)
        )
        if grad_output.ndim == 3:
            grad_bias = (
                weighted_go_fp32.sum(dim=(0, 1)) * scalar
                if has_bias
                else None
            )
        else:
            grad_bias = (
                weighted_go_fp32.sum(dim=0) * scalar if has_bias else None
            )

        output_dim = grad_output.shape[-1]
        input_dim = input.shape[-1]
        weighted_go_bf16 = weighted_go_fp32.to(torch.bfloat16)
        del weighted_go_fp32
        grad_weight = torch.mm(
            weighted_go_bf16.reshape(-1, output_dim).transpose(0, 1),
            input.reshape(-1, input_dim),
            out_dtype=torch.float32,
        )
        grad_weight.mul_(scalar)
        return grad_weight, grad_bias

    compute_dtype = _weighted_accumulation_dtype(grad_output, input, weights)
    work_go = grad_output.to(compute_dtype)
    work_input = input.to(compute_dtype)
    weighted_go = work_go * _broadcast_sample_weights(work_go, weights)
    scalar = torch.as_tensor(scale, device=grad_output.device, dtype=compute_dtype)
    if grad_output.ndim == 3:
        grad_weight = torch.einsum("bso,bsi->oi", weighted_go, work_input) * scalar
        grad_bias = weighted_go.sum(dim=(0, 1)) * scalar if has_bias else None
    else:
        grad_weight = torch.einsum("bo,bi->oi", weighted_go, work_input) * scalar
        grad_bias = weighted_go.sum(dim=0) * scalar if has_bias else None
    return grad_weight, grad_bias


def weighted_embedding_gradient(
    grad_output: Tensor,
    input_ids: Tensor,
    weights: Tensor,
    *,
    num_embeddings: int,
    padding_idx: Optional[int] = None,
    scale: Union[Tensor, float] = 1.0,
) -> Tensor:
    """Assemble a dense weighted Embedding gradient via projected scatter-add."""

    if input_ids.shape != grad_output.shape[:-1]:
        raise ValueError("input_ids shape must match grad_output without its last axis")
    if weights.ndim != 1 or weights.numel() != input_ids.shape[0]:
        raise ValueError("weights must have one entry per sample")
    if num_embeddings <= 0:
        raise ValueError("num_embeddings must be positive")

    compute_dtype = _weighted_accumulation_dtype(grad_output, weights)
    work_go = grad_output.to(compute_dtype)
    weighted_go = work_go * _broadcast_sample_weights(work_go, weights)
    flat_ids = input_ids.reshape(-1).to(torch.long)
    flat_grad = weighted_go.reshape(-1, grad_output.shape[-1])
    valid = (flat_ids >= 0) & (flat_ids < num_embeddings)
    if padding_idx is not None and padding_idx >= 0:
        valid &= flat_ids != int(padding_idx)
    if not bool(valid.all()):
        flat_ids = flat_ids[valid]
        flat_grad = flat_grad[valid]
    result = torch.zeros(
        num_embeddings, grad_output.shape[-1],
        device=grad_output.device, dtype=compute_dtype,
    )
    result.index_add_(0, flat_ids, flat_grad)
    scalar = torch.as_tensor(scale, device=result.device, dtype=result.dtype)
    return result * scalar


__all__ = [
    "compute_spectral_modes",
    "compute_spectral_modes_with_values",
    "exact_token_normalized_topk",
    "muon_live_candidate_transform",
    "optimize_batched_linear_soft_weights",
    "optimize_soft_weights",
    "optimize_soft_weights_streaming",
    "optimize_soft_weights_with_grad",
    "project_capped_simplex",
    "project_capped_simplex_batched",
    "project_probability_simplex",
    "project_probability_simplex_batched",
    "seeded_topk",
    "seeded_spectral_saturation_greedy",
    "spectral_linear_mode_support",
    "spectral_linear_scores",
    "spectral_saturated_objective",
    "weighted_embedding_gradient",
    "weighted_linear_gradients",
    "weighted_token_scale",
]
