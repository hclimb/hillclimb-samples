"""
Curation strategies for gradient-based data curation in trainers.

This module provides two families of strategies based on how validation
gradients are obtained:

1. **MergedBatch Strategies**:
   - Train and val samples are merged into a single batch
   - Val gradients computed during the same forward/backward pass
   - Factory: create_merged_batch_strategy()
   - Note: Has padding overhead when val/train have different sequence lengths

2. **SeparateBatch Strategies**:
   - Val gradients are pre-captured and cached before training
   - Training uses cached val gradients for curation scoring
   - Factory: create_separate_batch_strategy()
   - Val storage mode is derived from scoring_method in start_val_capture():
       * reduced_ghost/direct: Stores total gradient [O, I] per layer.
       Better when validation batch is large (e.g., self-reference validation in RLHF).
       * full_ghost: Stores [V, S, O] and [V, S, I] components (for pairwise scoring).
       More memory-efficient during training as it avoids materializing [B_train, O, I].
       Better when validation batch is small (e.g., external validation set in SFT).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, Optional, Callable, Tuple
    from ..hook import GradientHook
    from torch import Tensor

import torch
import torch.nn as nn
from .state import LayerWiseSubsetState, GlobalSubsetState, OptimizerGroupWiseSubsetState

logger = logging.getLogger(__name__)


# ============================================================
# JOINT BATCH STRATEGIES
# Val gradients computed from merged train+val batch
# ============================================================

class MergedBatchStrategy(ABC):
    """
    Abstract strategy for merged-batch data curation.

    Used when train and val samples are merged into a single batch,
    and val gradients are computed during the same forward/backward pass.

    Note: Has padding overhead when val/train have different sequence lengths.
    """

    def __init__(
        self,
        grad_hook: Optional[GradientHook],
        frac: float,
        use_second_order: bool = False,
        selection_mode: str = "topk",
        record_selections: bool = False,
        scoring_method: str = "reduced_ghost",
        seed: int = 42,
        **solver_config,
    ):
        """
        Initialize curation strategy.

        Args:
            grad_hook: GradientHook instance (can be None for NoSelection)
            frac: Curation fraction / filter fraction
            use_second_order: Use greedy curation with second-order
            selection_mode: "topk" (select top frac) or "filtering" (drop bottom frac of negative)
            record_selections: If True, record curation data for case study analysis
            scoring_method: Scoring method for influence scores ("reduced_ghost", "full_ghost", "direct")
        """
        self.grad_hook = grad_hook
        self.frac = frac
        self.use_second_order = use_second_order
        self.selection_mode = selection_mode
        self.record_selections = record_selections
        self.scoring_method = scoring_method
        self.seed = int(seed)
        self.solver_config = dict(solver_config)
        self.last_selection_record = None
        self.last_diagnostic_metrics = {}

    @property
    def has_update_compression(self) -> bool:
        """Check if update compression (MeSO) is enabled.

        When True, hooks stay enabled during GlobalSubset pass 2 so that
        CompressedLinearBackward stores compressed gradients for MeSO.
        """
        if self.grad_hook is None:
            return False
        return self.grad_hook.compression_mode.uses_compressed_updates

    def _configure_current_state_diagnostics(self, global_step: int) -> None:
        if self.grad_hook is None or self.grad_hook.selection_state is None:
            return
        self.grad_hook.selection_state.configure_optimizer_diagnostics(
            interval=int(
                self.solver_config.get(
                    "optimizer_aware_diagnostic_interval", 0
                )
            ),
            global_step=int(global_step),
        )

    def _extract_selection_records(self):
        """Extract curation records from state before cleanup."""
        if self.grad_hook is None or self.grad_hook.selection_state is None:
            self.last_selection_record = None
            self.last_diagnostic_metrics = {}
            return
        state = self.grad_hook.selection_state
        if hasattr(state, 'get_diagnostic_metrics'):
            self.last_diagnostic_metrics = state.get_diagnostic_metrics()
        else:
            self.last_diagnostic_metrics = {}
        if state._record_selections and state._selection_records:
            self.last_selection_record = list(state._selection_records)
        else:
            self.last_selection_record = None

    @abstractmethod
    def execute_training_step(
        self,
        model: nn.Module,
        merged_batch: Dict[str, Tensor],
        train_batch_size: int,
        compute_loss_fn: Callable[[nn.Module, Dict[str, Tensor]], Tensor],
        **kwargs
    ) -> Tensor:
        """
        Execute a complete training step with curation.

        Args:
            model: The model to train
            merged_batch: Merged train+val batch
            train_batch_size: Number of train samples in merged batch
            compute_loss_fn: Function to compute loss
            **kwargs: Additional arguments (lr, batch_train, etc.)

        Returns:
            Detached loss tensor
        """
        pass


class MergedBatchNoSelectionStrategy(MergedBatchStrategy):
    """
    Baseline strategy: no data curation, standard training.

    Uses all training samples without curation.
    """

    def execute_training_step(
        self,
        model: nn.Module,
        merged_batch: Dict[str, Tensor],
        train_batch_size: int,
        compute_loss_fn: Callable[[nn.Module, Dict[str, Tensor]], Tensor],
        **kwargs
    ) -> Tensor:
        """Standard training step without curation."""
        # Disable hooks if present (use standard gradient computation)
        if self.grad_hook is not None and not self.has_update_compression:
            self.grad_hook.disable_hooks()

        # Extract train-only portion (no need for val in baseline)
        train_batch = {k: v[:train_batch_size] for k, v in merged_batch.items()}

        model.zero_grad()
        loss = compute_loss_fn(model, train_batch)
        loss.backward()

        # Re-enable hooks
        if self.grad_hook is not None and not self.has_update_compression:
            self.grad_hook.enable_hooks()

        return loss.detach()


class MergedBatchLayerWiseSubsetStrategy(MergedBatchStrategy):
    """
    Layer-Wise Subset strategy with merged batch: single-pass, per-layer curation.

    Curation and gradient aggregation happen layer-by-layer
    during the backward pass.
    """

    selection_method = "LayerWiseSubset"

    def execute_training_step(
        self,
        model: nn.Module,
        merged_batch: Dict[str, Tensor],
        train_batch_size: int,
        compute_loss_fn: Callable[[nn.Module, Dict[str, Tensor]], Tensor],
        **kwargs
    ) -> Tensor:
        """Training step with per-layer curation."""
        lr = kwargs.get('lr', 1e-4)

        # Set up streaming state
        self._setup_state(train_batch_size, lr)
        self._configure_current_state_diagnostics(kwargs.get('global_step', 0))

        # Set token counts for proper gradient scaling
        if 'labels' in merged_batch:
            self.grad_hook.set_token_counts(merged_batch['labels'], train_batch_size)

        model.zero_grad()
        loss = compute_loss_fn(model, merged_batch)
        loss.backward()  # Per-layer curation happens in backward hooks

        # Extract curation records before cleanup
        self._extract_selection_records()

        # Cleanup
        self._cleanup()

        return loss.detach()

    def _setup_state(self, train_batch_size: int, lr: float) -> None:
        """Set up LayerWiseSubsetState for this step."""
        dtype = next(self.grad_hook.model.parameters()).dtype

        optimizer_aware = self.selection_method == "LayerWiseOptimizerAwareSubset"
        state = LayerWiseSubsetState(
            train_batch_size=train_batch_size,
            num_layers=len(self.grad_hook.layer_names),
            frac=self.frac,
            lr=lr,
            device=self.grad_hook.device,
            dtype=dtype,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            scoring_method=self.scoring_method,
            optimizer_aware=optimizer_aware,
            optimizer_aware_config=getattr(self.grad_hook, "optimizer_aware_config", {}),
        )

        self.grad_hook.selection_state = state

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()
        self.grad_hook.clear_token_counts()


class MergedBatchGlobalSubsetStrategy(MergedBatchStrategy):
    """
    GlobalSubset strategy with merged batch: two-pass, global curation.

    Pass 1: Compute curation scores across all layers
    Pass 2: Forward/backward only on globally selected samples
    """

    def execute_training_step(
        self,
        model: nn.Module,
        merged_batch: Dict[str, Tensor],
        train_batch_size: int,
        compute_loss_fn: Callable[[nn.Module, Dict[str, Tensor]], Tensor],
        **kwargs
    ) -> Tensor:
        """Training step with global curation."""
        lr = kwargs.get('lr', 1e-4)
        batch_train = kwargs.get('batch_train')  # Original train batch for pass 2

        # === PASS 1: Score Accumulation ===
        self._setup_state(train_batch_size, lr)
        self._configure_current_state_diagnostics(kwargs.get('global_step', 0))

        if 'labels' in merged_batch:
            self.grad_hook.set_token_counts(merged_batch['labels'], train_batch_size)

        model.zero_grad()
        loss_for_scoring = compute_loss_fn(model, merged_batch)
        loss_for_scoring.backward()

        # Get globally selected indices
        state: GlobalSubsetState = self.grad_hook.selection_state
        selected_indices = state.get_final_selection()
        # Sort indices for sequential memory access (better cache locality)
        selected_indices = selected_indices.sort()[0]
        n_selected = len(selected_indices)

        # Extract curation records before cleanup
        self._extract_selection_records()

        self._cleanup()

        # Handle empty curation: skip pass 2 and return zero loss
        if n_selected == 0:
            # Re-enable hooks for next step
            if not self.has_update_compression:
                self.grad_hook.enable_hooks()
            else:
                self.grad_hook.clear_token_counts()

            import torch
            zero_loss = torch.tensor(0.0, device=next(model.parameters()).device)
            return zero_loss

        # === PASS 2: Gradient Computation on Selected ===
        if batch_train is None:
            # Fall back to extracting from merged batch
            batch_train = {k: v[:train_batch_size] for k, v in merged_batch.items()}

        filtered_inputs = {
            'input_ids': batch_train['input_ids'][selected_indices],
            'attention_mask': batch_train['attention_mask'][selected_indices],
            'labels': batch_train['labels'][selected_indices]
        }

        # For pass 2, disable hooks if no compression (we want full gradients for selected samples)
        if not self.has_update_compression:
            self.grad_hook.disable_hooks()
        else:
            # For MeSO, set token counts for selected batch
            self.grad_hook.set_token_counts(filtered_inputs['labels'])

        model.zero_grad()
        loss = compute_loss_fn(model, filtered_inputs)
        loss.backward()

        # Re-enable hooks / clear token counts
        if not self.has_update_compression:
            self.grad_hook.enable_hooks()
        else:
            self.grad_hook.clear_token_counts()

        return loss.detach()

    def _setup_state(self, train_batch_size: int, lr: float) -> None:
        """Set up GlobalSubsetState for this step."""
        dtype = next(self.grad_hook.model.parameters()).dtype

        state = GlobalSubsetState(
            train_batch_size=train_batch_size,
            num_layers=len(self.grad_hook.layer_names),
            frac=self.frac,
            lr=lr,
            device=self.grad_hook.device,
            dtype=dtype,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            scoring_method=self.scoring_method,
        )

        self.grad_hook.selection_state = state

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()
        self.grad_hook.clear_token_counts()


class MergedBatchOptimizerAwareGroupWiseStrategy(MergedBatchLayerWiseSubsetStrategy):
    """
    Backward-compatible layer-wise optimizer-aware strategy.

    Execution is layer-wise/group-wise like LayerWiseSubset, but scoring is
    computed in the optimizer-induced geometry carried by the selection state.
    """

    selection_method = "LayerWiseOptimizerAwareSubset"


class MergedBatchOptimizerGroupWiseStrategy(MergedBatchStrategy):
    """One-pass optimizer-group-wise subset strategy."""

    selection_method = "OptimizerGroupWise"
    optimizer_aware = False

    def execute_training_step(
        self,
        model: nn.Module,
        merged_batch: Dict[str, Tensor],
        train_batch_size: int,
        compute_loss_fn: Callable[[nn.Module, Dict[str, Tensor]], Tensor],
        **kwargs
    ) -> Tensor:
        lr = kwargs.get('lr', 1e-4)
        self._setup_state(train_batch_size, lr)
        self._configure_current_state_diagnostics(kwargs.get('global_step', 0))

        if 'labels' in merged_batch:
            self.grad_hook.set_token_counts(merged_batch['labels'], train_batch_size)

        model.zero_grad()
        loss = compute_loss_fn(model, merged_batch)
        loss.backward()

        state: OptimizerGroupWiseSubsetState = self.grad_hook.selection_state
        group_selections = state.finalize_group_selections()

        self._extract_selection_records()

        if not group_selections or all(sel.numel() == 0 for sel in group_selections.values()):
            self.grad_hook.clear_retained_data()
            self._cleanup()
            import torch
            return torch.tensor(0.0, device=next(model.parameters()).device)

        self.grad_hook.assemble_groupwise_gradients_from_retained(state)
        self._cleanup()
        return loss.detach()

    def _setup_state(self, train_batch_size: int, lr: float) -> None:
        dtype = next(self.grad_hook.model.parameters()).dtype

        state = OptimizerGroupWiseSubsetState(
            train_batch_size=train_batch_size,
            num_layers=len(self.grad_hook.layer_names),
            frac=self.frac,
            lr=lr,
            device=self.grad_hook.device,
            dtype=dtype,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            scoring_method=self.scoring_method,
            one_pass=True,
            group_keys=self.grad_hook.get_optimizer_group_keys(),
            optimizer_aware=self.optimizer_aware,
            optimizer_aware_config=getattr(self.grad_hook, "optimizer_aware_config", {}),
        )

        self.grad_hook.selection_state = state

    def _cleanup(self) -> None:
        self.grad_hook.clear_selection()
        self.grad_hook.clear_token_counts()
        self.grad_hook.clear_retained_data()


class MergedBatchOptimizerAwareGroupSubsetStrategy(MergedBatchOptimizerGroupWiseStrategy):
    """Optimizer-group-wise subset strategy with optimizer-induced scoring."""

    selection_method = "OptimizerAwareGroupWise"
    optimizer_aware = True


class MergedBatchGlobalSubsetOnePassStrategy(MergedBatchStrategy):
    """
    One-pass subset strategy with merged batch (Algorithm 4.2).

    Single forward+backward pass: scoring and data retention happen during backward,
    then post-hoc gradient assembly from retained (grad_output, input) per layer.
    Saves the second forward+backward at the cost of higher peak memory.

    Non-linear layers (RMSNorm) are wrapped with TrainOnlyRMSNormBackward so
    their grad_weight is computed from the train slice only, preventing
    validation gradient leakage.
    """

    def execute_training_step(
        self,
        model: nn.Module,
        merged_batch: Dict[str, Tensor],
        train_batch_size: int,
        compute_loss_fn: Callable[[nn.Module, Dict[str, Tensor]], Tensor],
        **kwargs
    ) -> Tensor:
        """Training step with one-pass global curation."""
        lr = kwargs.get('lr', 1e-4)

        # Set up GlobalSubsetState with one_pass=True
        self._setup_state(train_batch_size, lr)
        self._configure_current_state_diagnostics(kwargs.get('global_step', 0))

        if 'labels' in merged_batch:
            self.grad_hook.set_token_counts(merged_batch['labels'], train_batch_size)

        model.zero_grad()
        loss = compute_loss_fn(model, merged_batch)
        loss.backward()
        # After backward:
        # - Scores accumulated in state, layer data retained in hook
        # - Hooked linear layers: .grad is None (GlobalSubsetLinearBackward returns None)
        # - RMSNorm layers: .grad contains train-only gradient (TrainOnlyRMSNormBackward)

        # Get globally selected indices
        state: GlobalSubsetState = self.grad_hook.selection_state
        selected_indices = state.get_final_selection()
        selected_indices = selected_indices.sort()[0]
        n_selected = len(selected_indices)

        # Extract curation records before cleanup
        self._extract_selection_records()

        if n_selected == 0:
            self.grad_hook.clear_retained_data()
            self._cleanup()
            import torch
            return torch.tensor(0.0, device=next(model.parameters()).device)

        # Compute scale factor for exact parity with two-pass
        scale_factor = state._compute_scale_factor_for_assembly(selected_indices)

        # Post-hoc gradient assembly for linear layers.
        # No model.zero_grad() here — non-linear layers retain their train-only
        # gradients from backward, and hooked linear layers have None grad
        # (GlobalSubsetLinearBackward suppresses weight/bias grads).
        self.grad_hook.assemble_gradients_from_retained(selected_indices, scale_factor)

        self._cleanup()
        return loss.detach()

    def _setup_state(self, train_batch_size: int, lr: float) -> None:
        """Set up GlobalSubsetState with one_pass=True."""
        dtype = next(self.grad_hook.model.parameters()).dtype

        state = GlobalSubsetState(
            train_batch_size=train_batch_size,
            num_layers=len(self.grad_hook.layer_names),
            frac=self.frac,
            lr=lr,
            device=self.grad_hook.device,
            dtype=dtype,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            scoring_method=self.scoring_method,
            one_pass=True,
        )

        self.grad_hook.selection_state = state

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()
        self.grad_hook.clear_token_counts()
        self.grad_hook.clear_retained_data()  # Safety net


class MergedBatchOptimizerAwareGlobalSubsetStrategy(MergedBatchGlobalSubsetOnePassStrategy):
    """Global subset strategy with optimizer-induced scoring."""

    selection_method = "OptimizerAwareGlobalSubset"

    def _setup_state(self, train_batch_size: int, lr: float) -> None:
        dtype = next(self.grad_hook.model.parameters()).dtype

        state = GlobalSubsetState(
            train_batch_size=train_batch_size,
            num_layers=len(self.grad_hook.layer_names),
            frac=self.frac,
            lr=lr,
            device=self.grad_hook.device,
            dtype=dtype,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            scoring_method=self.scoring_method,
            one_pass=True,
            optimizer_aware=True,
            optimizer_aware_config=getattr(self.grad_hook, "optimizer_aware_config", {}),
        )

        self.grad_hook.selection_state = state


def _validate_advanced_strategy(strategy: MergedBatchStrategy, variant: str) -> None:
    if strategy.selection_mode != "topk":
        raise ValueError(f"{variant} requires selection_mode='topk'")
    if strategy.use_second_order:
        raise ValueError(f"{variant} does not support use_second_order")
    if strategy.scoring_method != "reduced_ghost":
        raise ValueError(f"{variant} v1 requires scoring_method='reduced_ghost'")
    if strategy.grad_hook is None:
        raise ValueError(f"{variant} requires a GradientHook")
    if strategy.grad_hook.compression_mode is not None and (
        strategy.grad_hook.compression_mode.uses_compressed_scoring
        or strategy.grad_hook.compression_mode.uses_compressed_updates
    ):
        raise ValueError(f"{variant} v1 does not support score/update compression")
    if variant == "muon_spectral":
        optimizer = getattr(strategy.grad_hook, "optimizer", None)
        has_muon = False
        if optimizer is not None and hasattr(optimizer, "get_param_optimizer_kind"):
            for module in strategy.grad_hook.layer_name_to_module.values():
                param = getattr(module, "weight", None)
                if param is not None and optimizer.get_param_optimizer_kind(param) == "muon":
                    has_muon = True
                    break
        if not has_muon:
            raise ValueError(
                "Muon spectral surrogate requires a Muon-capable optimizer with Muon parameters"
            )
        config = _advanced_solver_config(strategy)
        if bool(config.get("muon_surrogate_saturation", False)) and not bool(
            getattr(strategy, "supports_muon_surrogate_saturation", False)
        ):
            raise ValueError(
                "Saturated Muon surrogate is currently implemented only for "
                "layerwise selection"
            )
        if bool(config.get("muon_surrogate_saturation", False)) and bool(
            config.get("muon_surrogate_include_adamw_scores", True)
        ):
            raise ValueError(
                "Saturated Muon surrogate requires matrix-only scoring"
            )


def _advanced_solver_config(strategy) -> dict:
    """Apply method-level spectral semantics without mutating shared CLI config."""
    config = dict(strategy.solver_config)
    mode_weighting = getattr(strategy, "muon_surrogate_mode_weighting", None)
    saturation = getattr(strategy, "muon_surrogate_saturation", None)
    include_adamw = getattr(
        strategy, "muon_surrogate_include_adamw_scores", None
    )
    soft_constraint = getattr(strategy, "soft_weighting_constraint", None)
    if mode_weighting is not None:
        config["muon_surrogate_mode_weighting"] = mode_weighting
    if saturation is not None:
        config["muon_surrogate_saturation"] = bool(saturation)
    if include_adamw is not None:
        config["muon_surrogate_include_adamw_scores"] = bool(include_adamw)
    if soft_constraint is not None:
        config["soft_weighting_constraint"] = str(soft_constraint)
    return config


class MergedBatchAdvancedLayerWiseStrategy(MergedBatchLayerWiseSubsetStrategy):
    """Shared execution for random, soft, and Muon-surrogate layerwise runs."""

    selection_variant = "score"
    supports_muon_surrogate_saturation = True

    def execute_training_step(self, *args, **kwargs):
        self._advanced_global_step = int(kwargs.get("global_step", 0))
        _validate_advanced_strategy(self, self.selection_variant)
        return super().execute_training_step(*args, **kwargs)

    def _setup_state(self, train_batch_size: int, lr: float) -> None:
        # Solver weights, token totals, and accumulated scores stay float32
        # even when model parameters and retained factors use bf16.
        dtype = torch.float32
        self.grad_hook.selection_state = LayerWiseSubsetState(
            train_batch_size=train_batch_size,
            num_layers=len(self.grad_hook.layer_names),
            frac=self.frac,
            lr=lr,
            device=self.grad_hook.device,
            dtype=dtype,
            use_second_order=False,
            selection_mode="topk",
            record_selections=self.record_selections,
            scoring_method="reduced_ghost",
            optimizer_aware=self.selection_variant in ("soft", "muon_spectral"),
            optimizer_aware_config=getattr(self.grad_hook, "optimizer_aware_config", {}),
            selection_variant=self.selection_variant,
            solver_config=_advanced_solver_config(self),
            seed=self.seed,
            global_step=getattr(self, "_advanced_global_step", 0),
        )


class MergedBatchLayerWiseRandomSubsetStrategy(MergedBatchAdvancedLayerWiseStrategy):
    selection_variant = "random"


class MergedBatchLayerWiseSoftWeightingStrategy(MergedBatchAdvancedLayerWiseStrategy):
    selection_variant = "soft"
    soft_weighting_constraint = "capped_simplex"


class MergedBatchLayerWiseSoftProbabilityStrategy(
    MergedBatchAdvancedLayerWiseStrategy
):
    """Layerwise soft weighting on the uncapped unit probability simplex."""

    selection_variant = "soft"
    soft_weighting_constraint = "probability_simplex"


class MergedBatchLayerWiseMuonSpectralStrategy(MergedBatchAdvancedLayerWiseStrategy):
    selection_variant = "muon_spectral"


class MergedBatchLayerWiseMuonMatrixSpectralStrategy(
    MergedBatchAdvancedLayerWiseStrategy
):
    """Layerwise Muon-matrix scorer with AdamW-only layers left full-batch."""

    selection_variant = "muon_spectral"


class MergedBatchLayerWiseMuonMatrixSpectralPStrategy(
    MergedBatchAdvancedLayerWiseStrategy
):
    """Matrix-only modular spectral scorer with ``alpha_r = beta_r``."""

    selection_variant = "muon_spectral"
    muon_surrogate_include_adamw_scores = False
    muon_surrogate_mode_weighting = "singular_value"
    muon_surrogate_saturation = False


class MergedBatchLayerWiseMuonMatrixSpectralSatStrategy(
    MergedBatchAdvancedLayerWiseStrategy
):
    """Matrix-only log1p-saturated scorer with uniform mode weights."""

    selection_variant = "muon_spectral"
    muon_surrogate_include_adamw_scores = False
    muon_surrogate_mode_weighting = "uniform"
    muon_surrogate_saturation = True


class MergedBatchLayerWiseMuonMatrixSpectralSatPStrategy(
    MergedBatchAdvancedLayerWiseStrategy
):
    """Matrix-only log1p-saturated scorer with ``alpha_r = beta_r``."""

    selection_variant = "muon_spectral"
    muon_surrogate_include_adamw_scores = False
    muon_surrogate_mode_weighting = "singular_value"
    muon_surrogate_saturation = True


class MergedBatchAdvancedGlobalStrategy(MergedBatchGlobalSubsetOnePassStrategy):
    """One-pass global execution for deterministic random/spectral subsets."""

    selection_variant = "score"

    def execute_training_step(self, *args, **kwargs):
        self._advanced_global_step = int(kwargs.get("global_step", 0))
        _validate_advanced_strategy(self, self.selection_variant)
        return super().execute_training_step(*args, **kwargs)

    def _setup_state(self, train_batch_size: int, lr: float) -> None:
        dtype = torch.float32
        self.grad_hook.selection_state = GlobalSubsetState(
            train_batch_size=train_batch_size,
            num_layers=len(self.grad_hook.layer_names),
            frac=self.frac,
            lr=lr,
            device=self.grad_hook.device,
            dtype=dtype,
            use_second_order=False,
            selection_mode="topk",
            record_selections=self.record_selections,
            scoring_method="reduced_ghost",
            one_pass=True,
            optimizer_aware=self.selection_variant == "muon_spectral",
            optimizer_aware_config=getattr(self.grad_hook, "optimizer_aware_config", {}),
            selection_variant=self.selection_variant,
            solver_config=_advanced_solver_config(self),
            seed=self.seed,
            global_step=getattr(self, "_advanced_global_step", 0),
        )


class MergedBatchGlobalRandomSubsetStrategy(MergedBatchAdvancedGlobalStrategy):
    selection_variant = "random"


class MergedBatchGlobalMuonSpectralStrategy(MergedBatchAdvancedGlobalStrategy):
    selection_variant = "muon_spectral"


class MergedBatchGlobalMuonMatrixSpectralStrategy(MergedBatchAdvancedGlobalStrategy):
    """Global subset scored only by Muon-managed matrix parameters."""

    selection_variant = "muon_spectral"


class MergedBatchGlobalSoftWeightingStrategy(MergedBatchAdvancedGlobalStrategy):
    selection_variant = "soft"

    def execute_training_step(
        self, model, merged_batch, train_batch_size, compute_loss_fn, **kwargs
    ):
        self._advanced_global_step = int(kwargs.get("global_step", 0))
        _validate_advanced_strategy(self, self.selection_variant)
        lr = kwargs.get("lr", 1e-4)
        self._setup_state(train_batch_size, lr)
        self._configure_current_state_diagnostics(kwargs.get("global_step", 0))
        if "labels" not in merged_batch:
            raise RuntimeError("Soft weighting requires labels for token normalization")
        self.grad_hook.set_token_counts(merged_batch["labels"], train_batch_size)

        model.zero_grad()
        loss = compute_loss_fn(model, merged_batch)
        loss.backward()

        state = self.grad_hook.selection_state
        weights = state.optimize_global_soft_weights()
        self._extract_selection_records()
        self.grad_hook.assemble_weighted_gradients_from_retained(weights, state)
        result_loss = loss.detach()
        # A global Muon objective closes over factors from every matrix layer.
        # Drop those references before returning and release now-unused CUDA
        # blocks; otherwise the allocator becomes fragmented over successive
        # steps and the 1B model eventually OOMs even on an 80GB A100.
        state._soft_objectives.clear()
        self._cleanup()
        del state, weights, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result_loss


def create_merged_batch_strategy(
    method: str,
    grad_hook: Optional[GradientHook],
    frac: float = 0.5,
    use_second_order: bool = False,
    selection_mode: str = "topk",
    record_selections: bool = False,
    scoring_method: str = "reduced_ghost",
    subset_mode: str = "one_pass",
    seed: int = 42,
    **solver_config,
) -> MergedBatchStrategy:
    """
    Factory function to create merged-batch curation strategy.

    Note: Has padding overhead when val/train have different sequence lengths.

    Args:
        method: Curation method ("NA", "LayerWiseSubset", "GlobalSubset")
        grad_hook: GradientHook instance
        frac: Curation/filter fraction
        use_second_order: Use greedy curation with second-order
        selection_mode: "topk" (select top frac) or "filtering" (drop bottom frac of negative)
        record_selections: If True, record curation data for case study analysis
        scoring_method: Scoring method ("reduced_ghost", "full_ghost", "direct", "compress")
        subset_mode: For GlobalSubset method: "one_pass" (Algorithm 4.2) or "two_pass" (Algorithm 4.3)

    Returns:
        Appropriate MergedBatchStrategy instance
    """
    kwargs = dict(grad_hook=grad_hook, frac=frac, use_second_order=use_second_order,
                  selection_mode=selection_mode, record_selections=record_selections,
                  scoring_method=scoring_method, seed=seed, **solver_config)

    if method == "NA":
        return MergedBatchNoSelectionStrategy(**kwargs)

    advanced_methods = {
        "LayerWiseRandomSubset": MergedBatchLayerWiseRandomSubsetStrategy,
        "LayerWiseSoftWeighting": MergedBatchLayerWiseSoftWeightingStrategy,
        "LayerWiseSoftProbability": MergedBatchLayerWiseSoftProbabilityStrategy,
        "LayerWiseMuonSpectral": MergedBatchLayerWiseMuonSpectralStrategy,
        "LayerWiseMuonMatrixSpectral": MergedBatchLayerWiseMuonMatrixSpectralStrategy,
        "LayerWiseMuonMatrixSpectralP": MergedBatchLayerWiseMuonMatrixSpectralPStrategy,
        "LayerWiseMuonMatrixSpectralSat": MergedBatchLayerWiseMuonMatrixSpectralSatStrategy,
        "LayerWiseMuonMatrixSpectralSatP": MergedBatchLayerWiseMuonMatrixSpectralSatPStrategy,
        "GlobalRandomSubset": MergedBatchGlobalRandomSubsetStrategy,
        "GlobalSoftWeighting": MergedBatchGlobalSoftWeightingStrategy,
        "GlobalMuonSpectral": MergedBatchGlobalMuonSpectralStrategy,
        "GlobalMuonMatrixSpectral": MergedBatchGlobalMuonMatrixSpectralStrategy,
    }
    if method in advanced_methods:
        strategy = advanced_methods[method](**kwargs)
        grad_hook.wrap_nonlinear_layers()
        return strategy

    if method == "LayerWiseSubset":
        strategy = MergedBatchLayerWiseSubsetStrategy(**kwargs)
        # Wrap non-linear layers so their grad_weight comes from train slice only.
        grad_hook.wrap_nonlinear_layers()
        return strategy

    if method == "LayerWiseOptimizerAwareSubset":
        strategy = MergedBatchOptimizerAwareGroupWiseStrategy(**kwargs)
        grad_hook.wrap_nonlinear_layers()
        return strategy

    if method == "OptimizerGroupWise":
        strategy = MergedBatchOptimizerGroupWiseStrategy(**kwargs)
        grad_hook.wrap_nonlinear_layers()
        return strategy

    if method == "OptimizerAwareGroupWise":
        strategy = MergedBatchOptimizerAwareGroupSubsetStrategy(**kwargs)
        grad_hook.wrap_nonlinear_layers()
        return strategy

    if method == "OptimizerAwareGlobalSubset":
        strategy = MergedBatchOptimizerAwareGlobalSubsetStrategy(**kwargs)
        grad_hook.wrap_nonlinear_layers()
        return strategy

    if method == "GlobalSubset":
        if subset_mode == "one_pass":
            strategy = MergedBatchGlobalSubsetOnePassStrategy(**kwargs)
            grad_hook.wrap_nonlinear_layers()
            return strategy
        else:
            return MergedBatchGlobalSubsetStrategy(**kwargs)

    raise ValueError(f"Unknown curation method: {method}")




# ============================================================
# CACHED VAL STRATEGIES
# Val gradients pre-captured and cached before training
# Avoids padding overhead when val/train have different seq lengths
# ============================================================

class SeparateBatchStrategy(ABC):
    """
    Abstract strategy for separate-batch data curation.

    Used when val gradients are pre-captured and cached before training,
    rather than computed from a merged batch during the same forward pass.

    Val storage mode is derived from scoring_method in start_val_capture():
    - reduced_ghost/direct: Stores total gradient [O, I] per layer.
      Better when validation batch is large (e.g., self-reference validation in RLHF).
    - full_ghost: Stores [V, S, O] and [V, S, I] components (for pairwise scoring).
      More memory-efficient during training. Better when validation batch is small.
    """

    def __init__(
        self,
        grad_hook: Optional[GradientHook],
        frac: float,
        use_second_order: bool = False,
        selection_mode: str = "topk",
        record_selections: bool = False,
        scoring_method: str = "reduced_ghost",
        seed: int = 42,
        **solver_config,
    ):
        """
        Initialize stored-val curation strategy.

        Args:
            grad_hook: GradientHook instance (can be None for NoSelection)
            frac: Fraction parameter. Meaning depends on selection_mode:
                  - "topk": Fraction of samples to select (top frac by score)
                  - "filtering": Fraction of negative-influence samples to DROP
            use_second_order: Use greedy curation with second-order
            selection_mode: "topk" (select top frac) or "filtering" (drop bottom frac of negative)
            record_selections: If True, record curation data for case study analysis
            scoring_method: Scoring method ("reduced_ghost", "full_ghost", "direct", "compress")
        """
        self.grad_hook = grad_hook
        self.frac = frac
        self.use_second_order = use_second_order
        self.selection_mode = selection_mode
        self.record_selections = record_selections
        self.scoring_method = scoring_method
        self.seed = int(seed)
        self.solver_config = dict(solver_config)
        self.last_selection_record = None
        self.last_diagnostic_metrics = {}

    @property
    def has_update_compression(self) -> bool:
        """Check if update compression (MeSO) is enabled.

        When True, hooks stay enabled during GlobalSubset pass 2 so that
        CompressedLinearBackward stores compressed gradients for MeSO.
        """
        if self.grad_hook is None:
            return False
        return self.grad_hook.compression_mode.uses_compressed_updates

    def _configure_current_state_diagnostics(self, global_step: int) -> None:
        if self.grad_hook is None or self.grad_hook.selection_state is None:
            return
        self.grad_hook.selection_state.configure_optimizer_diagnostics(
            interval=int(
                self.solver_config.get(
                    "optimizer_aware_diagnostic_interval", 0
                )
            ),
            global_step=int(global_step),
        )

    def _extract_selection_records(self):
        """Extract curation records from state before cleanup."""
        if self.grad_hook is None or self.grad_hook.selection_state is None:
            self.last_selection_record = None
            self.last_diagnostic_metrics = {}
            return
        state = self.grad_hook.selection_state
        if hasattr(state, 'get_diagnostic_metrics'):
            self.last_diagnostic_metrics = state.get_diagnostic_metrics()
        else:
            self.last_diagnostic_metrics = {}
        if state._record_selections and state._selection_records:
            self.last_selection_record = list(state._selection_records)
        else:
            self.last_selection_record = None

    @abstractmethod
    def execute_training_step(
        self,
        model: nn.Module,
        batch_size: int,
        compute_loss_fn: Callable[[], Tuple[Tensor, Dict]],
        lr: float,
        **kwargs
    ) -> Tuple[Tensor, Dict]:
        """
        Execute a complete training step with curation.

        Args:
            model: The model to train
            batch_size: Number of samples in the batch
            compute_loss_fn: Zero-arg function that computes loss and returns (loss, stats)
            lr: Learning rate for score scaling
            **kwargs: Additional arguments (filter_batch_fn for GlobalSubset)

        Returns:
            Tuple of (loss, stats_dict) where stats includes curation metrics
        """
        pass


class SeparateBatchNoSelectionStrategy(SeparateBatchStrategy):
    """
    Baseline strategy: no data curation, standard training.
    """

    def execute_training_step(
        self,
        model: nn.Module,
        batch_size: int,
        compute_loss_fn: Callable[[], Tuple[Tensor, Dict]],
        lr: float,
        **kwargs
    ) -> Tuple[Tensor, Dict]:
        """Standard training step without curation."""
        # Disable hooks for baseline (use standard gradient computation)
        if self.grad_hook is not None:
            self.grad_hook.disable_hooks()

        model.zero_grad()
        loss, stats = compute_loss_fn()
        loss.backward()

        # Re-enable hooks
        if self.grad_hook is not None:
            self.grad_hook.enable_hooks()

        return loss.detach(), stats


class SeparateBatchLayerWiseSubsetStrategy(SeparateBatchStrategy):
    """
    Layer-Wise Subset strategy with cached val: per-layer curation.

    Curation and gradient aggregation happen layer-by-layer during backward,
    using pre-captured validation gradients.
    """

    selection_method = "LayerWiseSubset"

    def execute_training_step(
        self,
        model: nn.Module,
        batch_size: int,
        compute_loss_fn: Callable[[], Tuple[Tensor, Dict]],
        lr: float,
        **kwargs
    ) -> Tuple[Tensor, Dict]:
        """Training step with per-layer curation using stored val grads.

        Args:
            model: The model to train
            batch_size: Number of samples in the batch
            compute_loss_fn: Zero-arg function that computes loss and returns (loss, stats)
            lr: Learning rate for score scaling
            **kwargs:
                labels: Optional label tensor for token-based gradient scaling.
                        If provided, enables proper score scaling across modes.
        """
        # Set up streaming state with stored validation gradients
        self._setup_state(batch_size, lr)
        self._configure_current_state_diagnostics(kwargs.get('global_step', 0))

        # Set token counts for gradient scaling (if labels provided)
        # In SeparateBatch mode, entire batch is train, so pass batch_size as train_batch_size
        labels = kwargs.get('labels')
        if labels is not None:
            self.grad_hook.set_token_counts(labels, batch_size)

        model.zero_grad()
        loss, stats = compute_loss_fn()
        loss.backward()  # Per-layer curation happens in backward hooks

        # Add curation stats from streaming state
        sel_state = self.grad_hook.selection_state
        if hasattr(sel_state, '_layer_selections') and sel_state._layer_selections:
            n_selected_list = [n for _, n in sel_state._layer_selections]
            stats["selection/mean_selected"] = sum(n_selected_list) / len(n_selected_list)
            stats["selection/min_selected"] = min(n_selected_list)
            # Use min for n_selected check - if any layer had 0, flag it
            stats["selection/n_selected"] = min(n_selected_list)

        # Extract curation records before cleanup
        self._extract_selection_records()

        # Cleanup
        self._cleanup()

        return loss.detach(), stats

    def execute_windowed_training_step(
        self,
        model: nn.Module,
        *,
        batch_size: int,
        microbatch_size: int,
        labels: Tensor,
        compute_chunk_loss_fn: Callable[[int, int], Tensor],
        lr: float,
        global_step: int = 0,
    ) -> Tuple[Tensor, Dict]:
        """Score chunks, decide once over N, then replay weighted chunks.

        Both score and replay losses are normalized by the valid assistant-token
        count of the complete logical window. Thus C=1/C=2 produce the same
        per-example factors and final parameter gradients as an unsplit N batch.
        """
        if not 0 < microbatch_size <= batch_size:
            raise ValueError("microbatch_size must be in [1, batch_size]")
        self._advanced_global_step = int(global_step)
        advanced_variant = getattr(self, "selection_variant", None)
        if advanced_variant is not None:
            _validate_advanced_strategy(self, advanced_variant)
        self._setup_state(batch_size, lr)
        self._configure_current_state_diagnostics(global_step)
        state = self.grad_hook.selection_state
        if not isinstance(state, LayerWiseSubsetState):
            raise RuntimeError("Windowed execution requires a layerwise selection state")
        state.enable_windowed_execution()
        self.grad_hook.set_token_counts(labels, batch_size)
        tokens = (labels[:, 1:] != -100).sum(dim=1)
        total_tokens = tokens.sum()
        if not bool((total_tokens > 0).detach().cpu()):
            raise RuntimeError("Logical candidate window contains no assistant tokens")

        score_loss = torch.zeros((), device=labels.device, dtype=torch.float32)
        for start in range(0, batch_size, microbatch_size):
            end = min(start + microbatch_size, batch_size)
            state.set_window_chunk("score", start, end)
            model.zero_grad(set_to_none=True)
            chunk_loss = compute_chunk_loss_fn(start, end)
            fraction = tokens[start:end].sum().to(chunk_loss.dtype) / total_tokens
            scaled_loss = chunk_loss * fraction
            score_loss = score_loss + scaled_loss.detach().float()
            scaled_loss.backward()

        self._finalize_window()
        self._extract_selection_records()

        # Finalization has copied the complete global decision into the
        # selection state.  Replay needs only those indices/weights, so release
        # both the CPU bf16 Muon factors and the CPU FP32 target cache before
        # candidate activations are rebuilt.
        state._window_factors.clear()
        state._window_spectral_modes.clear()
        self.grad_hook.clear_val_buffer()

        # Only replay gradients reach the optimizer. Scoring gradients (and full
        # gradients of unwrapped nonlinear parameters) are discarded here.
        model.zero_grad(set_to_none=True)
        replay_loss = torch.zeros((), device=labels.device, dtype=torch.float32)
        for start in range(0, batch_size, microbatch_size):
            end = min(start + microbatch_size, batch_size)
            state.set_window_chunk("replay", start, end)
            chunk_loss = compute_chunk_loss_fn(start, end)
            fraction = tokens[start:end].sum().to(chunk_loss.dtype) / total_tokens
            scaled_loss = chunk_loss * fraction
            replay_loss = replay_loss + scaled_loss.detach().float()
            scaled_loss.backward()

        stats: Dict[str, float] = {
            "selection/window_score_loss": float(score_loss.detach().cpu()),
            "selection/logical_candidates": float(batch_size),
            "selection/candidate_microbatch": float(microbatch_size),
        }
        if state._layer_selections:
            selected = [count for _, count in state._layer_selections]
            stats.update(
                {
                    "selection/mean_selected": sum(selected) / len(selected),
                    "selection/min_selected": min(selected),
                    "selection/n_selected": min(selected),
                }
            )
        self._cleanup()
        return replay_loss.detach(), stats

    def _finalize_window(self) -> None:
        """Turn the scored window into one decision per layer.

        Overridden by the windowed global strategies, which pool the same score
        tables into a single subset shared by every layer.
        """
        from .backward import finalize_windowed_layerwise_selection

        finalize_windowed_layerwise_selection(self.grad_hook)

    def _setup_state(self, batch_size: int, lr: float) -> None:
        """Set up LayerWiseSubsetState with stored val gradients."""
        self.grad_hook.setup_selection_with_stored_val(
            train_batch_size=batch_size,
            selection_method=self.selection_method,
            frac=self.frac,
            lr=lr,
            compute_scores_only=False,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            scoring_method=self.scoring_method,
        )

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()


class SeparateBatchOptimizerAwareGroupWiseStrategy(SeparateBatchLayerWiseSubsetStrategy):
    """
    Backward-compatible layer-wise optimizer-aware strategy with cached validation gradients.
    """

    selection_method = "LayerWiseOptimizerAwareSubset"


class SeparateBatchOptimizerGroupWiseStrategy(SeparateBatchStrategy):
    """One-pass optimizer-group-wise subset strategy with cached validation gradients."""

    selection_method = "OptimizerGroupWise"
    optimizer_aware = False

    def execute_training_step(
        self,
        model: nn.Module,
        batch_size: int,
        compute_loss_fn: Callable[[], Tuple[Tensor, Dict]],
        lr: float,
        **kwargs
    ) -> Tuple[Tensor, Dict]:
        labels = kwargs.get('labels')

        self._setup_state(batch_size, lr)
        self._configure_current_state_diagnostics(kwargs.get('global_step', 0))

        if labels is not None:
            self.grad_hook.set_token_counts(labels, batch_size)

        model.zero_grad()
        loss, stats = compute_loss_fn()
        loss.backward()

        state: OptimizerGroupWiseSubsetState = self.grad_hook.selection_state
        group_selections = state.finalize_group_selections()
        self._extract_selection_records()

        if not group_selections or all(sel.numel() == 0 for sel in group_selections.values()):
            self.grad_hook.clear_retained_data()
            self._cleanup()
            import torch
            zero_loss = torch.tensor(0.0, device=next(model.parameters()).device)
            stats["selection/n_selected"] = 0
            return zero_loss, stats

        self.grad_hook.assemble_groupwise_gradients_from_retained(state)
        stats["selection/n_selected"] = min(sel.numel() for sel in group_selections.values())
        self._cleanup()
        return loss.detach(), stats

    def _setup_state(self, batch_size: int, lr: float) -> None:
        self.grad_hook.setup_selection_with_stored_val(
            train_batch_size=batch_size,
            selection_method=self.selection_method,
            frac=self.frac,
            lr=lr,
            compute_scores_only=True,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            scoring_method=self.scoring_method,
            one_pass=True,
        )

    def _cleanup(self) -> None:
        self.grad_hook.clear_selection()
        self.grad_hook.clear_retained_data()


class SeparateBatchOptimizerAwareGroupSubsetStrategy(SeparateBatchOptimizerGroupWiseStrategy):
    """Optimizer-group-wise subset strategy with optimizer-induced scoring."""

    selection_method = "OptimizerAwareGroupWise"
    optimizer_aware = True


class SeparateBatchGlobalSubsetStrategy(SeparateBatchStrategy):
    """
    GlobalSubset strategy with cached val: global curation.

    Pass 1: Compute curation scores across all layers
    Pass 2: Forward/backward only on globally selected samples
    """

    def execute_training_step(
        self,
        model: nn.Module,
        batch_size: int,
        compute_loss_fn: Callable[[], Tuple[Tensor, Dict]],
        lr: float,
        **kwargs
    ) -> Tuple[Tensor, Dict]:
        """Training step with global curation using stored val grads."""
        # filter_batch_fn: Callable[[Tensor], Callable] that takes selected_indices
        # and returns a new compute_loss_fn for the filtered batch
        filter_batch_fn = kwargs.get('filter_batch_fn')
        if filter_batch_fn is None:
            raise ValueError("GlobalSubset strategy requires 'filter_batch_fn' in kwargs")

        # === PASS 1: Score Accumulation ===
        self._setup_state(batch_size, lr)
        self._configure_current_state_diagnostics(kwargs.get('global_step', 0))

        model.zero_grad()
        loss_for_scoring, _ = compute_loss_fn()
        loss_for_scoring.backward()

        # Get globally selected indices
        selected_indices = self.grad_hook.selection_state.get_final_selection()
        # Sort indices for sequential memory access (better cache locality)
        selected_indices = selected_indices.sort()[0]
        n_selected = len(selected_indices)

        # Extract curation records before cleanup
        self._extract_selection_records()

        self._cleanup()

        # Handle empty curation: skip pass 2 and return zero loss
        if n_selected == 0:
            # Re-enable hooks for next step
            self.grad_hook.enable_hooks()

            # Return zero loss and stats indicating batch was skipped
            # Include placeholder loss stats for consistent logging
            import torch
            zero_loss = torch.tensor(0.0, device=next(model.parameters()).device)
            stats = {
                "loss/total": 0.0,
                "loss/policy": 0.0,
                "loss/value": 0.0,
                "policy/approx_kl": 0.0,
                "policy/clipfrac": 0.0,
                "policy/ratio_mean": 1.0,
                "values/mean": 0.0,
                "selection/n_selected": 0,
            }
            return zero_loss, stats

        # === PASS 2: Gradient Computation on Selected ===
        # Disable hooks only if no update compression (standard optimizer for selected samples).
        # With MeSO, keep hooks enabled so CompressedLinearBackward stores
        # compressed gradients for the optimizer.
        if not self.has_update_compression:
            self.grad_hook.disable_hooks()

        model.zero_grad()

        # Get filtered compute_loss_fn for selected samples
        filtered_compute_loss_fn = filter_batch_fn(selected_indices)
        loss, stats = filtered_compute_loss_fn()
        loss.backward()

        # Re-enable hooks if we disabled them
        if not self.has_update_compression:
            self.grad_hook.enable_hooks()

        # Add curation stats
        stats["selection/n_selected"] = n_selected

        return loss.detach(), stats

    def _setup_state(self, batch_size: int, lr: float) -> None:
        """Set up GlobalSubsetState with stored val gradients."""
        self.grad_hook.setup_selection_with_stored_val(
            train_batch_size=batch_size,
            selection_method="GlobalSubset",
            frac=self.frac,
            lr=lr,
            compute_scores_only=True,  # Only accumulate scores in pass 1
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            scoring_method=self.scoring_method,
        )

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()


class SeparateBatchGlobalSubsetOnePassStrategy(SeparateBatchStrategy):
    """
    One-pass subset strategy with separate val batch (Algorithm 4.2).

    Recommended one-pass mode: exact scale factor parity with two-pass since
    batch_total_tokens == train_total_tokens in separate batch mode.
    """

    def execute_training_step(
        self,
        model: nn.Module,
        batch_size: int,
        compute_loss_fn: Callable[[], Tuple[Tensor, Dict]],
        lr: float,
        **kwargs
    ) -> Tuple[Tensor, Dict]:
        """Training step with one-pass global curation using stored val grads."""
        labels = kwargs.get('labels')

        # Set up GlobalSubsetState with one_pass=True and stored val
        self._setup_state(batch_size, lr)
        self._configure_current_state_diagnostics(kwargs.get('global_step', 0))

        if labels is not None:
            self.grad_hook.set_token_counts(labels, batch_size)

        model.zero_grad()
        loss, stats = compute_loss_fn()
        loss.backward()
        # After backward: scores accumulated, layer data retained

        # Get globally selected indices
        selected_indices = self.grad_hook.selection_state.get_final_selection()
        selected_indices = selected_indices.sort()[0]
        n_selected = len(selected_indices)

        # Extract curation records before cleanup
        self._extract_selection_records()

        if n_selected == 0:
            self.grad_hook.clear_retained_data()
            self._cleanup()
            import torch
            zero_loss = torch.tensor(0.0, device=next(model.parameters()).device)
            stats["selection/n_selected"] = 0
            return zero_loss, stats

        # Compute scale factor for exact parity with two-pass
        state = self.grad_hook.selection_state
        scale_factor = state._compute_scale_factor_for_assembly(selected_indices)

        # Post-hoc gradient assembly for linear layers.
        # Non-linear layers (LayerNorm, embeddings, etc.) retain their autograd
        # gradients from backward — GlobalSubsetLinearBackward returns None for
        # weight/bias, so hooked linear layers have no stale grad to clear.
        self.grad_hook.assemble_gradients_from_retained(selected_indices, scale_factor)

        stats["selection/n_selected"] = n_selected
        self._cleanup()
        return loss.detach(), stats

    def _setup_state(self, batch_size: int, lr: float) -> None:
        """Set up GlobalSubsetState with one_pass=True and stored val gradients."""
        self.grad_hook.setup_selection_with_stored_val(
            train_batch_size=batch_size,
            selection_method="GlobalSubset",
            frac=self.frac,
            lr=lr,
            compute_scores_only=True,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            scoring_method=self.scoring_method,
            one_pass=True,
        )

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()
        self.grad_hook.clear_retained_data()  # Safety net


class SeparateBatchOptimizerAwareGlobalSubsetStrategy(SeparateBatchGlobalSubsetOnePassStrategy):
    """Global subset strategy with optimizer-induced scoring and cached validation gradients."""

    def _setup_state(self, batch_size: int, lr: float) -> None:
        self.grad_hook.setup_selection_with_stored_val(
            train_batch_size=batch_size,
            selection_method="OptimizerAwareGlobalSubset",
            frac=self.frac,
            lr=lr,
            compute_scores_only=True,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            scoring_method=self.scoring_method,
            one_pass=True,
        )


class SeparateBatchWindowedGlobalSubsetStrategy(SeparateBatchLayerWiseSubsetStrategy):
    """GlobalSubset under the exact logical-candidate window.

    The one-pass global strategy holds every layer's per-candidate factors until
    the backward finishes, which the microbatched window is designed to avoid.
    This variant instead reuses the layer-wise window — score chunks, decide
    once, replay — and only changes the decision rule: the per-layer score
    tables are pooled into a single subset shared by every layer, which is the
    quantity ``GlobalSubsetState`` accumulates in the unwindowed path.
    """

    selection_method = "LayerWiseSubset"

    def execute_training_step(self, *args, **kwargs):
        raise RuntimeError(
            "Windowed global selection requires execute_windowed_training_step; "
            "the unwindowed path uses SeparateBatchGlobalSubsetOnePassStrategy"
        )

    def _finalize_window(self) -> None:
        from .backward import finalize_windowed_global_selection

        finalize_windowed_global_selection(self.grad_hook)


class SeparateBatchWindowedOptimizerAwareGlobalSubsetStrategy(
    SeparateBatchWindowedGlobalSubsetStrategy
):
    """Windowed global subset with optimizer-induced (OptA) scoring geometry."""

    selection_method = "LayerWiseOptimizerAwareSubset"


class SeparateBatchAdvancedLayerWiseStrategy(SeparateBatchLayerWiseSubsetStrategy):
    supports_muon_surrogate_saturation = True
    selection_variant = "score"

    def execute_training_step(self, *args, **kwargs):
        self._advanced_global_step = int(kwargs.get("global_step", 0))
        _validate_advanced_strategy(self, self.selection_variant)
        return super().execute_training_step(*args, **kwargs)

    def _setup_state(self, batch_size: int, lr: float) -> None:
        if self.grad_hook.val_cache.get_num_captured() == 0:
            raise RuntimeError("Advanced solver requires captured target gradients")
        if self.selection_variant in ("soft", "muon_spectral") and (
            self.grad_hook.val_total_tokens is None
            or self.grad_hook.val_total_tokens <= 0
        ):
            raise RuntimeError(
                "Advanced solver requires at least one valid target token"
            )
        dtype = torch.float32
        state = LayerWiseSubsetState(
            train_batch_size=batch_size,
            num_layers=len(self.grad_hook.layer_names),
            frac=self.frac,
            lr=lr,
            device=self.grad_hook.device,
            dtype=dtype,
            use_second_order=False,
            selection_mode="topk",
            record_selections=self.record_selections,
            scoring_method="reduced_ghost",
            optimizer_aware=self.selection_variant in ("soft", "muon_spectral"),
            optimizer_aware_config=getattr(self.grad_hook, "optimizer_aware_config", {}),
            selection_variant=self.selection_variant,
            solver_config=_advanced_solver_config(self),
            seed=self.seed,
            global_step=getattr(self, "_advanced_global_step", 0),
        )
        state._use_stored_val = True
        self.grad_hook.selection_state = state


class SeparateBatchLayerWiseRandomSubsetStrategy(SeparateBatchAdvancedLayerWiseStrategy):
    selection_variant = "random"


class SeparateBatchLayerWiseSoftWeightingStrategy(SeparateBatchAdvancedLayerWiseStrategy):
    selection_variant = "soft"
    soft_weighting_constraint = "capped_simplex"


class SeparateBatchLayerWiseSoftProbabilityStrategy(
    SeparateBatchAdvancedLayerWiseStrategy
):
    """Layerwise soft weighting on the uncapped unit probability simplex."""

    selection_variant = "soft"
    soft_weighting_constraint = "probability_simplex"


class SeparateBatchLayerWiseMuonSpectralStrategy(SeparateBatchAdvancedLayerWiseStrategy):
    selection_variant = "muon_spectral"


class SeparateBatchLayerWiseMuonMatrixSpectralStrategy(
    SeparateBatchAdvancedLayerWiseStrategy
):
    """Layerwise Muon-matrix scorer with AdamW-only layers left full-batch."""

    selection_variant = "muon_spectral"


class SeparateBatchLayerWiseMuonMatrixSpectralPStrategy(
    SeparateBatchAdvancedLayerWiseStrategy
):
    """Matrix-only modular spectral scorer with ``alpha_r = beta_r``."""

    selection_variant = "muon_spectral"
    muon_surrogate_include_adamw_scores = False
    muon_surrogate_mode_weighting = "singular_value"
    muon_surrogate_saturation = False


class SeparateBatchLayerWiseMuonMatrixSpectralSatStrategy(
    SeparateBatchAdvancedLayerWiseStrategy
):
    """Matrix-only log1p-saturated scorer with uniform mode weights."""

    selection_variant = "muon_spectral"
    muon_surrogate_include_adamw_scores = False
    muon_surrogate_mode_weighting = "uniform"
    muon_surrogate_saturation = True


class SeparateBatchLayerWiseMuonMatrixSpectralSatPStrategy(
    SeparateBatchAdvancedLayerWiseStrategy
):
    """Matrix-only log1p-saturated scorer with ``alpha_r = beta_r``."""

    selection_variant = "muon_spectral"
    muon_surrogate_include_adamw_scores = False
    muon_surrogate_mode_weighting = "singular_value"
    muon_surrogate_saturation = True


class SeparateBatchAdvancedGlobalStrategy(SeparateBatchGlobalSubsetOnePassStrategy):
    selection_variant = "score"

    def execute_training_step(self, *args, **kwargs):
        self._advanced_global_step = int(kwargs.get("global_step", 0))
        _validate_advanced_strategy(self, self.selection_variant)
        return super().execute_training_step(*args, **kwargs)

    def _setup_state(self, batch_size: int, lr: float) -> None:
        if self.grad_hook.val_cache.get_num_captured() == 0:
            raise RuntimeError("Advanced solver requires captured target gradients")
        if self.selection_variant in ("soft", "muon_spectral") and (
            self.grad_hook.val_total_tokens is None
            or self.grad_hook.val_total_tokens <= 0
        ):
            raise RuntimeError(
                "Advanced solver requires at least one valid target token"
            )
        dtype = torch.float32
        state = GlobalSubsetState(
            train_batch_size=batch_size,
            num_layers=len(self.grad_hook.layer_names),
            frac=self.frac,
            lr=lr,
            device=self.grad_hook.device,
            dtype=dtype,
            use_second_order=False,
            selection_mode="topk",
            record_selections=self.record_selections,
            scoring_method="reduced_ghost",
            one_pass=True,
            optimizer_aware=self.selection_variant == "muon_spectral",
            optimizer_aware_config=getattr(self.grad_hook, "optimizer_aware_config", {}),
            selection_variant=self.selection_variant,
            solver_config=self.solver_config,
            seed=self.seed,
            global_step=getattr(self, "_advanced_global_step", 0),
        )
        state._use_stored_val = True
        self.grad_hook.selection_state = state


class SeparateBatchGlobalRandomSubsetStrategy(SeparateBatchAdvancedGlobalStrategy):
    selection_variant = "random"


class SeparateBatchGlobalMuonSpectralStrategy(SeparateBatchAdvancedGlobalStrategy):
    selection_variant = "muon_spectral"


class SeparateBatchGlobalMuonMatrixSpectralStrategy(SeparateBatchAdvancedGlobalStrategy):
    """Global subset scored only by Muon-managed matrix parameters."""

    selection_variant = "muon_spectral"


class SeparateBatchGlobalSoftWeightingStrategy(SeparateBatchAdvancedGlobalStrategy):
    selection_variant = "soft"

    def execute_training_step(
        self, model, batch_size, compute_loss_fn, lr, **kwargs
    ):
        self._advanced_global_step = int(kwargs.get("global_step", 0))
        _validate_advanced_strategy(self, self.selection_variant)
        self._setup_state(batch_size, lr)
        self._configure_current_state_diagnostics(kwargs.get("global_step", 0))
        labels = kwargs.get("labels")
        if labels is None:
            raise RuntimeError("Soft weighting requires labels for token normalization")
        self.grad_hook.set_token_counts(labels, batch_size)

        model.zero_grad()
        loss, stats = compute_loss_fn()
        loss.backward()
        state = self.grad_hook.selection_state
        weights = state.optimize_global_soft_weights()
        self._extract_selection_records()
        self.grad_hook.assemble_weighted_gradients_from_retained(weights, state)
        stats["selection/n_selected"] = state.num_selected
        stats["selection/soft_mass"] = float(weights.sum().detach().cpu())
        result_loss = loss.detach()
        state._soft_objectives.clear()
        self._cleanup()
        del state, weights, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result_loss, stats


def create_separate_batch_strategy(
    method: str,
    grad_hook: Optional[GradientHook],
    frac: float = 0.5,
    use_second_order: bool = False,
    selection_mode: str = "topk",
    record_selections: bool = False,
    scoring_method: str = "reduced_ghost",
    subset_mode: str = "one_pass",
    seed: int = 42,
    windowed: bool = False,
    **solver_config,
) -> SeparateBatchStrategy:
    """
    Factory function to create separate-batch curation strategy.

    Avoids padding overhead when val/train have different sequence lengths.

    Args:
        method: Curation method ("NA", "LayerWiseSubset", "GlobalSubset")
        grad_hook: GradientHook instance
        frac: Fraction parameter. Meaning depends on selection_mode:
              - "topk": Fraction of samples to select (top frac by score)
              - "filtering": Fraction of negative-influence samples to DROP
        use_second_order: Use greedy curation with second-order
        selection_mode: "topk" (select top frac) or "filtering" (drop bottom frac of negative)
        record_selections: If True, record curation data for case study analysis
        scoring_method: Scoring method ("reduced_ghost", "full_ghost", "direct", "compress")
        subset_mode: For GlobalSubset method: "one_pass" (Algorithm 4.2) or "two_pass" (Algorithm 4.3)
        windowed: True when the trainer drives an exact logical candidate window
            (compute chunk smaller than the logical batch). Only the global
            methods branch on this; every other method uses the same class in
            both modes.

    Returns:
        Appropriate SeparateBatchStrategy instance
    """
    kwargs = dict(grad_hook=grad_hook, frac=frac, use_second_order=use_second_order,
                  selection_mode=selection_mode, record_selections=record_selections,
                  scoring_method=scoring_method, seed=seed, **solver_config)

    if method == "NA":
        return SeparateBatchNoSelectionStrategy(**kwargs)

    if windowed and method in ("GlobalSubset", "OptimizerAwareGlobalSubset"):
        windowed_global = {
            "GlobalSubset": SeparateBatchWindowedGlobalSubsetStrategy,
            "OptimizerAwareGlobalSubset": (
                SeparateBatchWindowedOptimizerAwareGlobalSubsetStrategy
            ),
        }
        strategy = windowed_global[method](**kwargs)
        grad_hook.check_unhooked_trainable_params()
        return strategy

    advanced_methods = {
        "LayerWiseRandomSubset": SeparateBatchLayerWiseRandomSubsetStrategy,
        "LayerWiseSoftWeighting": SeparateBatchLayerWiseSoftWeightingStrategy,
        "LayerWiseSoftProbability": SeparateBatchLayerWiseSoftProbabilityStrategy,
        "LayerWiseMuonSpectral": SeparateBatchLayerWiseMuonSpectralStrategy,
        "LayerWiseMuonMatrixSpectral": SeparateBatchLayerWiseMuonMatrixSpectralStrategy,
        "LayerWiseMuonMatrixSpectralP": SeparateBatchLayerWiseMuonMatrixSpectralPStrategy,
        "LayerWiseMuonMatrixSpectralSat": SeparateBatchLayerWiseMuonMatrixSpectralSatStrategy,
        "LayerWiseMuonMatrixSpectralSatP": SeparateBatchLayerWiseMuonMatrixSpectralSatPStrategy,
        "GlobalRandomSubset": SeparateBatchGlobalRandomSubsetStrategy,
        "GlobalSoftWeighting": SeparateBatchGlobalSoftWeightingStrategy,
        "GlobalMuonSpectral": SeparateBatchGlobalMuonSpectralStrategy,
        "GlobalMuonMatrixSpectral": SeparateBatchGlobalMuonMatrixSpectralStrategy,
    }
    if method in advanced_methods:
        strategy = advanced_methods[method](**kwargs)
        grad_hook.check_unhooked_trainable_params()
        return strategy

    if method == "LayerWiseSubset":
        return SeparateBatchLayerWiseSubsetStrategy(**kwargs)

    if method == "LayerWiseOptimizerAwareSubset":
        return SeparateBatchOptimizerAwareGroupWiseStrategy(**kwargs)

    if method == "OptimizerGroupWise":
        strategy = SeparateBatchOptimizerGroupWiseStrategy(**kwargs)
        grad_hook.check_unhooked_trainable_params()
        return strategy

    if method == "OptimizerAwareGroupWise":
        strategy = SeparateBatchOptimizerAwareGroupSubsetStrategy(**kwargs)
        grad_hook.check_unhooked_trainable_params()
        return strategy

    if method == "OptimizerAwareGlobalSubset":
        strategy = SeparateBatchOptimizerAwareGlobalSubsetStrategy(**kwargs)
        grad_hook.check_unhooked_trainable_params()
        return strategy

    if method == "GlobalSubset":
        if subset_mode == "one_pass":
            strategy = SeparateBatchGlobalSubsetOnePassStrategy(**kwargs)
            # Safety check: warn about trainable params not covered by hooks.
            # No non-linear wrapping needed (no val contamination in separate batch),
            # but unhooked params get full-batch train grad instead of curated.
            grad_hook.check_unhooked_trainable_params()
            return strategy
        else:
            return SeparateBatchGlobalSubsetStrategy(**kwargs)

    raise ValueError(f"Unknown curation method: {method}")
