"""
Curation state classes for gradient-based data curation.

This module provides two distinct state classes:
- LayerWiseSubsetState: Per-layer curation (layer_wise_subset descent), single-pass
- GlobalSubsetState: Global curation (subset descent), two-pass score accumulation
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable
if TYPE_CHECKING:
    from typing import Optional, Tuple
    from torch import Tensor

import torch

from ..utils import greedy_selection, topk_selection, negative_filtering


def _sanitize_metric_group(group_key: str) -> str:
    """Make optimizer/layer group names safe for wandb metric paths."""
    return "".join(
        ch if (ch.isalnum() or ch in ("_", "-")) else "_"
        for ch in str(group_key)
    )


def _rank_positions_desc(scores: torch.Tensor) -> torch.Tensor:
    """Return descending rank positions, where 0 is highest score."""
    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(scores.numel(), device=scores.device, dtype=torch.float32)
    return ranks


def _safe_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x.float().flatten()
    y = y.float().flatten()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    return torch.where(denom > eps, (x * y).sum() / denom, torch.zeros((), device=x.device))


class SelectionState(ABC):
    """
    Abstract base class for curation state management during backward pass.

    Subclasses implement different curation strategies:
    - LayerWiseSubsetState: Per-layer curation (layer_wise_subset descent), immediate gradient aggregation
    - GlobalSubsetState: Score accumulation (subset descent), global curation after all layers
    """

    def __init__(
        self,
        train_batch_size: int,
        num_layers: int,
        frac: float,
        lr: float,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        use_second_order: bool = False,
        selection_mode: str = "topk",
        record_selections: bool = False,
        selection_variant: str = "score",
        solver_config: Optional[dict[str, Any]] = None,
        seed: int = 42,
        global_step: int = 0,
        optimizer_aware_diagnostic_interval: int = 0,
    ):
        """
        Initialize curation state.

        Args:
            train_batch_size: Number of training samples
            num_layers: Total number of layers
            frac: Fraction parameter. Meaning depends on selection_mode:
                  - "topk": Fraction of samples to select
                  - "filtering": Fraction of negative-influence samples to DROP
            lr: Learning rate for score scaling
            device: Device for tensors
            dtype: Data type for tensors
            use_second_order: If True, use greedy curation with second-order interactions
            selection_mode: "topk" (select top frac) or "filtering" (drop bottom frac of negative)
            record_selections: If True, record selected indices and scores per layer for case study
        """
        self.train_batch_size = train_batch_size
        self.num_layers = num_layers
        self.frac = frac
        self.lr = lr
        self.device = device
        self.dtype = dtype
        self.use_second_order = use_second_order
        self.selection_mode = selection_mode
        self.selection_variant = str(selection_variant)
        self.solver_config = dict(solver_config or {})
        self.seed = int(seed)
        self.global_step = int(global_step)
        self.optimizer_aware_diagnostic_interval = int(
            optimizer_aware_diagnostic_interval
        )
        if self.optimizer_aware_diagnostic_interval < 0:
            raise ValueError(
                "optimizer_aware_diagnostic_interval must be non-negative"
            )

        if self.selection_variant not in ("score", "random", "soft", "muon_spectral"):
            raise ValueError(f"Unknown selection variant: {self.selection_variant}")

        # Number of samples to select (for top-k mode)
        self.num_selected = max(1, int(train_batch_size * frac))

        # Token-based scaling (set via set_token_counts)
        self.tokens_per_sample: Optional[Tensor] = None
        self.train_total_tokens_tensor: Optional[Tensor] = None

        # Precomputed score correction for joint batch mode (1.0 = no correction)
        # Stored as Tensor to avoid D2H memory copies during backward
        self.score_correction: Optional[Tensor] = None

        # Curation recording for case study analysis
        self._record_selections = record_selections
        self._selection_records: list = []

        # Lightweight per-step diagnostics consumed by the trainer/wandb.
        self._diagnostic_values: dict[str, list[Tensor]] = {}

        # Global soft weighting stores one detached layer closure at a time and
        # differentiates each closure independently during optimization.  This
        # avoids retaining an autograd graph spanning all model layers.
        self._soft_objectives: list[Callable[[Tensor], Tensor]] = []
        self._soft_reference_scores = torch.zeros(
            train_batch_size, device=device, dtype=torch.float32
        )
        self._soft_weights: Optional[Tensor] = None

    def _append_diagnostic(self, name: str, value: Tensor) -> None:
        """Queue one scalar diagnostic without synchronizing the accelerator."""
        if not torch.is_tensor(value):
            value = torch.tensor(float(value), device=self.device)
        value = value.detach().float().reshape(())
        self._diagnostic_values.setdefault(name, []).append(value)

    def configure_optimizer_diagnostics(
        self,
        *,
        interval: int,
        global_step: Optional[int] = None,
    ) -> None:
        interval = int(interval)
        if interval < 0:
            raise ValueError(
                "optimizer_aware_diagnostic_interval must be non-negative"
            )
        self.optimizer_aware_diagnostic_interval = interval
        if global_step is not None:
            self.global_step = int(global_step)

    def should_collect_optimizer_diagnostics(self) -> bool:
        interval = self.optimizer_aware_diagnostic_interval
        return interval > 0 and self.global_step % interval == 0

    def add_raw_opt_score_diagnostics(
        self,
        group_key: str,
        raw_scores: Tensor,
        opt_scores: Tensor,
        selected_indices: Optional[Tensor] = None,
    ) -> None:
        """Track cheap raw-vs-optimizer-aware ranking diagnostics.

        Uses only per-sample score vectors. It does not materialize full
        per-sample gradients or run an additional forward/backward pass.
        """
        if not self.should_collect_optimizer_diagnostics():
            return
        raw = raw_scores.detach().float().flatten()
        opt = opt_scores.detach().float().flatten()
        n = min(raw.numel(), opt.numel())
        if n < 2:
            return
        raw = raw[:n]
        opt = opt[:n]

        safe_group = _sanitize_metric_group(group_key)
        raw_rank = _rank_positions_desc(raw)
        opt_rank = _rank_positions_desc(opt)

        k = self.num_selected
        if selected_indices is not None:
            k = selected_indices.numel()
        k = min(max(int(k), 1), n)

        raw_top = torch.topk(raw, k=k).indices
        opt_top = torch.topk(opt, k=k).indices
        overlap = (raw_top[:, None] == opt_top[None, :]).any(dim=1).float().mean()
        mean_rank_shift = (raw_rank - opt_rank).abs().mean() / max(float(n - 1), 1.0)

        self._append_diagnostic(f"diag/{safe_group}/raw_opt_spearman", _safe_corr(raw_rank, opt_rank))
        self._append_diagnostic(f"diag/{safe_group}/raw_opt_topk_overlap", overlap)
        self._append_diagnostic(f"diag/{safe_group}/raw_opt_pearson", _safe_corr(raw, opt))
        self._append_diagnostic(f"diag/{safe_group}/mean_rank_shift", mean_rank_shift)

    def add_selection_score_diagnostics(
        self,
        group_key: str,
        scores: Tensor,
        selected_indices: Tensor,
    ) -> None:
        """Track selected score alignment diagnostics with no extra gradient work."""
        scores = scores.detach().float().flatten()
        if scores.numel() == 0 or selected_indices is None or selected_indices.numel() == 0:
            return
        selected_indices = selected_indices.to(device=scores.device, dtype=torch.long)
        selected_scores = scores[selected_indices]
        safe_group = _sanitize_metric_group(group_key)

        self._append_diagnostic(
            f"update/{safe_group}/selected_alignment",
            selected_scores.mean(),
        )
        self._append_diagnostic(
            f"update/{safe_group}/selected_alignment_sum",
            selected_scores.sum(),
        )
        self._append_diagnostic(
            f"update/{safe_group}/selected_fraction",
            torch.tensor(
                selected_indices.numel() / max(float(scores.numel()), 1.0),
                device=scores.device,
            ),
        )
        if selected_indices.numel() < scores.numel():
            mask = torch.ones(scores.numel(), device=scores.device, dtype=torch.bool)
            mask[selected_indices] = False
            nonselected = scores[mask]
            if nonselected.numel() > 0:
                self._append_diagnostic(
                    f"update/{safe_group}/selected_score_margin",
                    selected_scores.mean() - nonselected.mean(),
                )

    def get_diagnostic_metrics(self) -> dict[str, float]:
        names: list[str] = []
        means: list[Tensor] = []
        for name, values in self._diagnostic_values.items():
            if not values:
                continue
            stacked = torch.stack(values)
            finite = torch.isfinite(stacked)
            finite_count = finite.sum()
            finite_sum = torch.where(finite, stacked, 0.0).sum()
            means.append(
                torch.where(
                    finite_count > 0,
                    finite_sum / finite_count.clamp_min(1),
                    torch.full((), float("nan"), device=stacked.device),
                )
            )
            names.append(name)
        if not means:
            return {}
        host_values = torch.stack(means).detach().cpu().tolist()
        return {
            name: float(value)
            for name, value in zip(names, host_values)
            if math.isfinite(float(value))
        }

    def clear_diagnostic_metrics(self) -> None:
        self._diagnostic_values.clear()

    def derived_seed(self, layer_idx: int = -1) -> int:
        """Derive a stable per-step/per-layer seed without Python hashing."""
        modulus = 2**63 - 25
        layer_term = int(layer_idx) + 2
        return int(
            (self.seed * 6364136223846793005
             + self.global_step * 1442695040888963407
             + layer_term * 22695477) % modulus
        )

    def _uses_exact_token_normalized_opta(self) -> bool:
        """Whether this state uses the length-matched hard OptA ablation."""
        config = getattr(self, "optimizer_aware_config", {}) or {}
        return bool(
            getattr(self, "optimizer_aware", False)
            and self.selection_variant == "score"
            and config.get("token_normalized_selection", False)
            and str(config.get("target_mode", "opta")).lower() == "opta"
            and self.selection_mode == "topk"
            and not self.use_second_order
        )

    def _select_exact_token_normalized_opta(
        self, scores: Tensor, *, layer_idx: int
    ) -> Tensor:
        """Select the exact cardinality-k maximizer of score/token ratio."""
        if self.tokens_per_sample is None:
            raise RuntimeError(
                "Token counts must be set before token-normalized hard OptA selection"
            )
        from .advanced_solvers import exact_token_normalized_topk

        selected = exact_token_normalized_topk(
            scores.detach().float(),
            self.tokens_per_sample.detach(),
            self.num_selected,
            seed=self.derived_seed(layer_idx),
        )
        selected_tokens = self.tokens_per_sample[selected].sum().float()
        objective = scores[selected].float().sum() / selected_tokens
        group = "global" if layer_idx < 0 else f"layer_{layer_idx}"
        self._append_diagnostic(
            f"selection/{group}/token_normalized_objective", objective
        )
        self._append_diagnostic(
            f"selection/{group}/selected_valid_tokens", selected_tokens
        )
        return selected

    def add_soft_objective(
        self,
        objective: Callable[[Tensor], Tensor],
        reference_scores: Optional[Tensor] = None,
    ) -> None:
        self._soft_objectives.append(objective)
        if reference_scores is not None:
            self._soft_reference_scores.add_(reference_scores.detach().float())

    def _record_soft_diagnostics(
        self,
        group_key: str,
        weights: Tensor,
        diagnostics: dict[str, Any],
        reference_scores: Optional[Tensor] = None,
        *,
        layer_idx: int = -1,
    ) -> None:
        safe_group = _sanitize_metric_group(group_key)
        w = weights.detach().float()
        mass = w.sum().clamp_min(1e-12)
        probabilities = w / mass
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        ess = mass.square() / w.square().sum().clamp_min(1e-12)
        saturation = ((w <= 1e-4) | (w >= 1.0 - 1e-4)).float().mean()
        constraint = str(
            self.solver_config.get(
                "soft_weighting_constraint", "capped_simplex"
            )
        ).lower().replace("-", "_")
        if constraint not in ("capped_simplex", "probability_simplex"):
            raise ValueError(
                "soft_weighting_constraint must be 'capped_simplex' or "
                "'probability_simplex'"
            )
        expected_mass = 1.0 if constraint == "probability_simplex" else float(
            self.num_selected
        )
        self._append_diagnostic(f"soft/{safe_group}/entropy", entropy)
        self._append_diagnostic(f"soft/{safe_group}/ess", ess)
        self._append_diagnostic(f"soft/{safe_group}/boundary_saturation", saturation)
        self._append_diagnostic(f"soft/{safe_group}/constraint_mass", expected_mass)
        self._append_diagnostic(
            f"soft/{safe_group}/mass_residual", (mass - expected_mass).abs()
        )
        self._append_diagnostic(
            f"soft/{safe_group}/probability_simplex",
            float(constraint == "probability_simplex"),
        )
        self._append_diagnostic(
            f"soft/{safe_group}/explicit_upper_cap",
            float(constraint == "capped_simplex"),
        )
        for name, value in diagnostics.items():
            if isinstance(value, (int, float)) or torch.is_tensor(value):
                self._append_diagnostic(f"soft/{safe_group}/{name}", value)

        if reference_scores is not None and reference_scores.numel() == w.numel():
            from .advanced_solvers import seeded_topk

            k = min(self.num_selected, w.numel())
            selection_seed = self.derived_seed(layer_idx)
            soft_top = seeded_topk(w, k, seed=selection_seed)
            scores = reference_scores.detach().to(
                device=w.device, dtype=torch.float32
            )
            ref_top = seeded_topk(
                scores, k, seed=selection_seed
            )
            overlap = (soft_top[:, None] == ref_top[None, :]).any(dim=1).float().mean()
            self._append_diagnostic(f"soft/{safe_group}/opta_topk_overlap", overlap)

            # Do not solve the exhaustive cardinality-constrained OptANorm
            # comparator in the training hot path.  At Dolci N=16, K=8 that
            # diagnostic materializes 12,870 combinations for every layer and
            # step, while it never affects the continuous weights.  Selection
            # records retain weights, scores, token counts, k, and seed so the
            # exact comparator can be reconstructed offline when needed.

    def optimize_global_soft_weights(self) -> Tensor:
        if self.selection_variant != "soft":
            raise RuntimeError("Global soft optimization requested for a non-soft state")
        if not self._soft_objectives:
            # A model may have no target-supported hooked layer. Uniform is the
            # only deterministic, unbiased feasible fallback in that case.
            self._soft_weights = torch.full(
                (self.train_batch_size,),
                self.num_selected / float(self.train_batch_size),
                device=self.device,
                dtype=torch.float32,
            )
            self._append_diagnostic("soft/global/zero_target_layers", 1.0)
            return self._soft_weights

        from .advanced_solvers import optimize_soft_weights_streaming

        cfg = self.solver_config
        weights, diagnostics = optimize_soft_weights_streaming(
            self._soft_objectives,
            self.train_batch_size,
            self.num_selected,
            device=torch.device(self.device),
            steps=int(cfg.get("soft_weighting_steps", 20)),
            lr=float(cfg.get("soft_weighting_lr", 0.1)),
            tolerance=float(cfg.get("soft_weighting_tol", 1e-5)),
            patience=int(cfg.get("soft_weighting_patience", 3)),
            gamma=float(cfg.get("soft_weighting_gamma", 0.0)),
        )
        self._soft_weights = weights.detach()
        self._record_soft_diagnostics(
            "global", weights, diagnostics, self._soft_reference_scores
        )
        if self._record_selections:
            record = {
                "weights": weights.detach().float().cpu().tolist(),
                "reference_scores": self._soft_reference_scores.cpu().tolist(),
                "num_selected": int(self.num_selected),
                "selection_seed": int(self.derived_seed(-1)),
                "layer_idx": -1,
                "global_step": int(self.global_step),
            }
            if self.tokens_per_sample is not None:
                record["valid_token_counts"] = (
                    self.tokens_per_sample.detach().float().cpu().tolist()
                )
            self._selection_records = [record]
        return self._soft_weights

    def set_token_counts(
        self,
        tokens_per_sample: Tensor,
        total_train_tokens: Tensor,
        batch_total_tokens: Tensor,
    ) -> None:
        """
        Set token counts for gradient scaling and score correction.

        Args:
            tokens_per_sample: Response token count per training sample [train_batch_size].
                              Used for gradient scaling (sum over selected samples).
            total_train_tokens: Sum of response tokens in training samples (scalar Tensor)
            batch_total_tokens: Sum of tokens in entire batch (train + val for joint batch, scalar Tensor)
        """
        if not bool((total_train_tokens > 0).detach().cpu()):
            raise RuntimeError("No valid training tokens are available for selection")

        # Store for gradient scaling: train_total_tokens / selected_tokens
        self.tokens_per_sample = tokens_per_sample
        self.train_total_tokens_tensor = total_train_tokens.to(dtype=self.dtype)
        self.batch_total_tokens_tensor = batch_total_tokens.to(dtype=self.dtype)

        # Precompute score correction for joint batch mode
        # All operations kept on device as Tensors to avoid D2H memcpy
        val_tokens = batch_total_tokens - total_train_tokens
        # Use torch.where to handle the conditional without branching on CPU values
        correction = (batch_total_tokens.to(self.dtype) ** 2) / (total_train_tokens.to(self.dtype) * val_tokens.to(self.dtype))
        # Set to 1.0 if val_tokens <= 0 or total_train_tokens <= 0
        valid_mask = (val_tokens > 0) & (total_train_tokens > 0)
        self.score_correction = torch.where(
            valid_mask,
            correction,
            torch.ones((), device=tokens_per_sample.device, dtype=self.dtype)
        )

    def _select_indices(
        self,
        scores: Tensor,
        similarity: Optional[Tensor] = None,
        layer_idx: int = -1,
    ) -> Tensor:
        """
        Select sample indices based on scores.

        Args:
            scores: Per-sample scores [train_batch_size]
            similarity: Optional similarity matrix [train_batch_size, train_batch_size]

        Returns:
            Selected indices tensor
        """
        if self.selection_variant == "random":
            generator = torch.Generator(device=scores.device)
            generator.manual_seed(self.derived_seed(layer_idx))
            return torch.randperm(
                self.train_batch_size, generator=generator, device=scores.device
            )[:self.num_selected]

        if self._uses_exact_token_normalized_opta():
            return self._select_exact_token_normalized_opta(
                scores, layer_idx=layer_idx
            )

        # Apply lr scaling
        scores_scaled = scores * self.lr
        if similarity is not None:
            similarity = similarity * (self.lr ** 2)

        if self.selection_mode == "filtering":
            return negative_filtering(scores_scaled, self.frac)
        elif self.use_second_order and similarity is not None:
            return greedy_selection(scores_scaled, similarity, self.num_selected)
        elif self.selection_variant == "muon_spectral":
            from .advanced_solvers import seeded_topk
            if not torch.isfinite(scores_scaled).all():
                raise FloatingPointError("Muon spectral scores contain non-finite values")
            return seeded_topk(
                scores_scaled, self.num_selected, seed=self.derived_seed(layer_idx)
            )
        else:
            return topk_selection(scores_scaled, self.num_selected)

    def _compute_scale_factor(self, selected_indices: Tensor) -> Tensor:
        """
        Compute token-based gradient scale factor for selected samples.

        Returns batch_total_tokens / selected_tokens so the curated gradient
        matches the magnitude of a forward/backward on only the selected samples.

        Uses batch_total_tokens (not train_total_tokens) because grad_output from
        autograd is normalized by 1/batch_total_tokens. In separate-batch mode,
        batch_total == train_total, so this is equivalent.
        """
        if self.tokens_per_sample is None or self.batch_total_tokens_tensor is None:
            raise RuntimeError(
                "Token counts not set. Call set_token_counts() before curation. "
                "For SeparateBatch strategies, pass 'labels' in kwargs to execute_training_step()."
            )
        # Handle empty curation to avoid division by zero
        if selected_indices.numel() == 0:
            return torch.tensor(1.0, device=self.device, dtype=self.dtype)
        selected_tokens = self.tokens_per_sample[selected_indices].sum()
        scale = self.batch_total_tokens_tensor / selected_tokens
        return torch.where(selected_tokens == 0, torch.ones_like(scale), scale)

    @abstractmethod
    def process_layer_gradients(
        self,
        train_grads: Tensor,
        val_grad: Tensor,
        layer_idx: int,
        score_correction: Optional[Tensor] = None,
    ) -> Optional[Tuple[Tensor, int]]:
        """
        Process gradients for a single layer.

        Args:
            train_grads: Per-sample gradients [train_batch_size, feature_dim]
            val_grad: Total validation gradient [feature_dim] (sum over val samples)
            layer_idx: Index of the current layer
            score_correction: Correction factor for joint batch mode (scalar Tensor).
                For joint batch: T_total²/(T_train × T_val) to convert to standalone scaling.
                For cached mode: None (no correction needed).

        Returns:
            For Streaming: (reduced_grad, num_selected) tuple
            For GlobalSubset: None (scores accumulated internally)
        """
        pass

    @abstractmethod
    def get_final_selection(self) -> Tensor:
        """
        Get final selected indices.

        For Streaming: Raises NotImplementedError (curation is per-layer)
        For GlobalSubset: Returns globally selected indices after all layers
        """
        pass


class LayerWiseSubsetState(SelectionState):
    """
    State for layer_wise_subset descent: per-layer curation, single-pass.

    At each layer, immediately computes scores, selects samples,
    and aggregates gradients. No global accumulation needed.
    """

    def __init__(
        self,
        scoring_method: str = "reduced_ghost",
        direct_batch_size: int = 0,
        optimizer_aware: bool = False,
        optimizer_aware_config: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.scoring_method = scoring_method
        self.direct_batch_size = direct_batch_size
        self.optimizer_aware = optimizer_aware
        self.optimizer_aware_config = optimizer_aware_config or {}
        # Track last selected indices for stats
        self._last_selected_indices: Optional[Tensor] = None

        # Track curation stats across all layers
        self._layer_selections: list = []  # (layer_idx, n_selected) tuples
        # Exact logical-window execution is opt-in. The ordinary one-pass path
        # remains untouched unless enable_windowed_execution() is called.
        self.windowed_execution = False
        self.window_phase: Optional[str] = None
        self.window_chunk_start = 0
        self.window_chunk_end = 0
        self._window_scores: dict[int, Tensor] = {}
        self._window_score_filled: dict[int, Tensor] = {}
        self._window_raw_scores: dict[int, Tensor] = {}
        self._window_raw_filled: dict[int, Tensor] = {}
        self._window_support: dict[int, Tensor] = {}
        self._window_support_filled: dict[int, Tensor] = {}
        self._window_beta: dict[int, Tensor] = {}
        self._window_full_batch_layers: set[int] = set()
        self._window_zero_layers: set[int] = set()
        self._window_decisions: dict[int, tuple[str, Tensor]] = {}
        self._window_geometry: dict[int, str] = {}
        # Non-linear Muon soft objectives cannot be reduced to one score per
        # candidate. Keep one contiguous, preallocated bf16 factor window per
        # layer on CPU. This avoids retaining one allocation per chunk and a
        # second full-window torch.cat copy during finalization.
        self._window_factors: dict[int, tuple[Tensor, Tensor, Tensor]] = {}
        # Muon surrogate target modes depend only on the fixed target gradient,
        # optimizer configuration, layer, and logical window. Candidate chunks
        # reuse them instead of repeating the same SVD C times.
        self._window_spectral_modes: dict[
            int, tuple[Tensor, Tensor, Tensor, int]
        ] = {}

    def enable_windowed_execution(self) -> None:
        if self.use_second_order:
            raise ValueError(
                "Windowed layerwise execution does not support second-order "
                "cross-candidate similarities"
            )
        self.windowed_execution = True
        self.window_phase = "score"
        self.window_chunk_start = 0
        self.window_chunk_end = 0

    def set_window_chunk(self, phase: str, start: int, end: int) -> None:
        if not self.windowed_execution:
            raise RuntimeError("Windowed execution has not been enabled")
        if phase not in ("score", "replay"):
            raise ValueError("Window phase must be 'score' or 'replay'")
        if not (0 <= start < end <= self.train_batch_size):
            raise ValueError(
                f"Invalid window chunk [{start}, {end}) for "
                f"{self.train_batch_size} candidates"
            )
        self.window_phase = phase
        self.window_chunk_start = int(start)
        self.window_chunk_end = int(end)

    def _store_window_vector(
        self,
        table: dict[int, Tensor],
        filled_table: dict[int, Tensor],
        layer_idx: int,
        values: Tensor,
    ) -> None:
        values = values.detach().float().flatten()
        expected = self.window_chunk_end - self.window_chunk_start
        if values.numel() != expected:
            raise RuntimeError(
                f"Layer {layer_idx} produced {values.numel()} scores for a "
                f"window chunk of size {expected}"
            )
        if layer_idx not in table:
            table[layer_idx] = torch.empty(
                self.train_batch_size,
                device=values.device,
                dtype=torch.float32,
            )
            filled_table[layer_idx] = torch.zeros(
                self.train_batch_size,
                device="cpu",
                dtype=torch.bool,
            )
        table[layer_idx][self.window_chunk_start:self.window_chunk_end] = values
        filled_table[layer_idx][
            self.window_chunk_start:self.window_chunk_end
        ] = True

    def store_window_scores(
        self,
        layer_idx: int,
        scores: Tensor,
        *,
        raw_scores: Optional[Tensor] = None,
        geometry: Optional[str] = None,
    ) -> None:
        self._store_window_vector(
            self._window_scores,
            self._window_score_filled,
            layer_idx,
            scores,
        )
        if raw_scores is not None:
            self._store_window_vector(
                self._window_raw_scores,
                self._window_raw_filled,
                layer_idx,
                raw_scores,
            )
        if geometry is not None:
            self._window_geometry.setdefault(layer_idx, str(geometry))

    def store_window_support(
        self,
        layer_idx: int,
        support: Tensor,
        beta: Tensor,
    ) -> None:
        support = support.detach().float()
        expected = self.window_chunk_end - self.window_chunk_start
        if support.ndim != 2 or support.shape[0] != expected:
            raise RuntimeError(
                f"Layer {layer_idx} produced support shape "
                f"{tuple(support.shape)} for chunk size {expected}"
            )
        if layer_idx not in self._window_support:
            self._window_support[layer_idx] = torch.empty(
                (self.train_batch_size, support.shape[1]),
                device=support.device,
                dtype=torch.float32,
            )
            self._window_support_filled[layer_idx] = torch.zeros(
                self.train_batch_size,
                device="cpu",
                dtype=torch.bool,
            )
            self._window_beta[layer_idx] = beta.detach().float().to(
                support.device
            )
        elif self._window_support[layer_idx].shape[1] != support.shape[1]:
            raise RuntimeError("Muon spectral rank changed across candidate chunks")
        self._window_support[layer_idx][
            self.window_chunk_start:self.window_chunk_end
        ] = support
        self._window_support_filled[layer_idx][
            self.window_chunk_start:self.window_chunk_end
        ] = True

    def store_window_factors(
        self,
        layer_idx: int,
        grad_output: Tensor,
        input: Tensor,
    ) -> None:
        expected = self.window_chunk_end - self.window_chunk_start
        if grad_output.shape[0] != expected or input.shape[0] != expected:
            raise RuntimeError(
                f"Layer {layer_idx} factor batch does not match window chunk "
                f"[{self.window_chunk_start}, {self.window_chunk_end})"
            )
        key = int(layer_idx)
        if key not in self._window_factors:
            cpu_grad_output = torch.empty(
                (self.train_batch_size, *grad_output.shape[1:]),
                device="cpu",
                dtype=torch.bfloat16,
            )
            cpu_input = torch.empty(
                (self.train_batch_size, *input.shape[1:]),
                device="cpu",
                dtype=torch.bfloat16,
            )
            filled = torch.zeros(
                self.train_batch_size, device="cpu", dtype=torch.bool
            )
            self._window_factors[key] = (cpu_grad_output, cpu_input, filled)

        cpu_grad_output, cpu_input, filled = self._window_factors[key]
        if cpu_grad_output.shape[1:] != grad_output.shape[1:]:
            raise RuntimeError(
                f"Layer {layer_idx} grad-output factor shape changed across chunks"
            )
        if cpu_input.shape[1:] != input.shape[1:]:
            raise RuntimeError(
                f"Layer {layer_idx} input factor shape changed across chunks"
            )
        start, end = self.window_chunk_start, self.window_chunk_end
        if bool(filled[start:end].any()):
            raise RuntimeError(
                f"Layer {layer_idx} factor window [{start}, {end}) was stored twice"
            )
        cpu_grad_output[start:end].copy_(grad_output.detach())
        cpu_input[start:end].copy_(input.detach())
        filled[start:end] = True

    def get_window_factors(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        factors = self._window_factors.get(int(layer_idx))
        if factors is None:
            raise RuntimeError(f"Incomplete CPU factor window for layer {layer_idx}")
        grad_output, input, filled = factors
        if not bool(filled.all()):
            raise RuntimeError(f"Incomplete CPU factor window for layer {layer_idx}")
        return grad_output, input

    def get_window_spectral_modes(
        self, layer_idx: int
    ) -> Optional[tuple[Tensor, Tensor, Tensor, int]]:
        return self._window_spectral_modes.get(int(layer_idx))

    def store_window_spectral_modes(
        self,
        layer_idx: int,
        u: Tensor,
        v: Tensor,
        beta: Tensor,
        active_rank: int,
    ) -> None:
        key = int(layer_idx)
        if key in self._window_spectral_modes:
            raise RuntimeError(
                f"Muon spectral modes for layer {layer_idx} were stored twice"
            )
        self._window_spectral_modes[key] = (
            u.detach(),
            v.detach(),
            beta.detach(),
            int(active_rank),
        )

    def mark_window_full_batch(self, layer_idx: int) -> None:
        self._window_full_batch_layers.add(int(layer_idx))

    def mark_window_zero(self, layer_idx: int) -> None:
        self._window_zero_layers.add(int(layer_idx))

    def require_complete_window(self, layer_idx: int, *, support: bool = False) -> None:
        table = (
            self._window_support_filled if support else self._window_score_filled
        )
        filled = table.get(layer_idx)
        if filled is None or not bool(filled.all()):
            kind = "support" if support else "scores"
            raise RuntimeError(
                f"Incomplete window {kind} for layer {layer_idx}; every logical "
                "candidate must be scored before finalization"
            )

    def set_window_decision(
        self,
        layer_idx: int,
        kind: str,
        values: Tensor,
    ) -> None:
        if kind not in ("indices", "weights"):
            raise ValueError("Window decision kind must be indices or weights")
        self._window_decisions[int(layer_idx)] = (kind, values.detach())

    def get_window_decision(self, layer_idx: int) -> tuple[str, Tensor]:
        decision = self._window_decisions.get(int(layer_idx))
        if decision is None:
            empty = torch.empty(
                0, device=self.device, dtype=torch.long
            )
            return "indices", empty
        return decision


    def process_layer_gradients(
        self,
        train_grads: Tensor,
        val_grad: Tensor,
        layer_idx: int,
        score_correction: Optional[Tensor] = None,
    ) -> Tuple[Tensor, int]:
        """
        Immediately select and aggregate at this layer.

        Args:
            train_grads: Per-sample gradients [train_batch_size, feature_dim]
            val_grad: Total validation gradient [feature_dim]
            layer_idx: Index of the current layer
            score_correction: Correction factor for joint batch mode (scalar Tensor or None)

        Returns:
            (reduced_grad, num_selected) tuple
        """
        # Step 1: Compute scores (gradient alignment)
        scores = train_grads @ val_grad

        if score_correction is not None:
            scores = scores * score_correction

        # Step 2: Compute similarity if second-order
        similarity = None
        if self.use_second_order:
            similarity = train_grads @ train_grads.T
            if score_correction is not None:
                similarity = similarity * (score_correction ** 2)

        # Step 3: Select indices
        selected_indices = self._select_indices(scores, similarity)
        # Sort indices for sequential memory access (better cache locality)
        selected_indices = selected_indices.sort()[0]
        self._last_selected_indices = selected_indices
        num_selected = selected_indices.shape[0]

        # Track curation for this layer
        self._layer_selections.append((layer_idx, num_selected))

        # Record curation data for case study analysis
        if self._record_selections:
            self._selection_records.append({
                'layer_idx': layer_idx,
                'selected_indices': selected_indices.tolist(),
                'scores': scores.detach().float().cpu().tolist(),
            })

        # Step 4: Aggregate selected gradients
        # Note: empty curation (num_selected=0) naturally produces zero gradients
        # since train_grads[empty_indices].sum() = zeros
        selected_grads = train_grads[selected_indices]
        reduced_grad = selected_grads.sum(dim=0, keepdim=True)

        # Step 5: Apply token-based gradient scaling
        # _compute_scale_factor handles empty curation internally (returns 1.0)
        scale_factor = self._compute_scale_factor(selected_indices)
        reduced_grad = reduced_grad * scale_factor

        self.num_selected = num_selected
        return reduced_grad, num_selected

    def get_final_selection(self) -> Tensor:
        """Layer-Wise Subset descent uses per-layer curation, not global."""
        raise NotImplementedError(
            "LayerWiseSubsetState uses per-layer curation. "
            "Use process_layer_gradients() at each layer instead."
        )


class GlobalSubsetState(SelectionState):
    """
    State for GlobalSubset method: global curation, two-pass.

    Pass 1: Accumulates scores across all layers
    Pass 2: Uses global curation for gradient computation on selected samples

    Args (in addition to SelectionState):
        scoring_method: "reduced_ghost" for ghost/factored inner product (default),
                        "direct" for explicit per-sample gradient materialization
                        (Algorithm 4.4 in the paper).
    """

    def __init__(
        self,
        scoring_method: str = "reduced_ghost",
        one_pass: bool = False,
        direct_batch_size: int = 0,
        optimizer_aware: bool = False,
        optimizer_aware_config: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.scoring_method = scoring_method
        self.direct_batch_size = direct_batch_size
        self.one_pass = one_pass
        self.optimizer_aware = optimizer_aware
        self.optimizer_aware_config = optimizer_aware_config or {}

        # Accumulators for global scoring
        self.grad_dot_scores = torch.zeros(
            self.train_batch_size,
            device=self.device,
            dtype=self.dtype
        )

        self.similarity_matrix: Optional[Tensor] = None
        if self.use_second_order:
            self.similarity_matrix = torch.zeros(
                self.train_batch_size, self.train_batch_size,
                device=self.device,
                dtype=self.dtype
            )

        self._diag_raw_scores: Optional[Tensor] = None
        self._diag_opt_scores: Optional[Tensor] = None

    def process_layer_gradients(
        self,
        train_grads: Tensor,
        val_grad: Tensor,
        layer_idx: int,
        score_correction: Optional[Tensor] = None,
    ) -> None:
        """
        Accumulate scores - no immediate curation.

        Args:
            train_grads: Per-sample gradients [train_batch_size, feature_dim]
            val_grad: Total validation gradient [feature_dim]
            layer_idx: Index of the current layer
            score_correction: Correction factor for joint batch mode (scalar Tensor or None)

        Returns:
            None (scores accumulated internally)
        """
        # Cast to accumulator dtype if needed
        if train_grads.dtype != self.dtype:
            train_grads = train_grads.to(self.dtype)
        if val_grad.dtype != self.dtype:
            val_grad = val_grad.to(self.dtype)

        # Accumulate first-order scores: train_grads @ val_grad
        # Use tensor ops to avoid D2H sync from .item()
        layer_scores = torch.mv(train_grads, val_grad)
        if score_correction is not None:
            layer_scores = layer_scores * score_correction
        self.grad_dot_scores.add_(layer_scores)

        # Accumulate similarity matrix if second-order
        if self.similarity_matrix is not None:
            layer_sim = torch.mm(train_grads, train_grads.t())
            if score_correction is not None:
                layer_sim = layer_sim * (score_correction ** 2)
            self.similarity_matrix.add_(layer_sim)

        return None

    def accumulate_precomputed_scores(
        self,
        scores: Tensor,
        similarity: Optional[Tensor],
        score_correction: Optional[Tensor] = None,
        layer_idx: Optional[int] = None,
        raw_scores_for_diag: Optional[Tensor] = None,
    ) -> None:
        """
        Accumulate pre-computed scores (for full gradient path).

        This method is used when scores are computed externally (e.g., from
        factorized grad_output and input) rather than from flattened gradients.

        Args:
            scores: Pre-computed scores [train_batch_size]
            similarity: Pre-computed similarity matrix [train_batch_size, train_batch_size] or None
            score_correction: Correction factor for joint batch mode (scalar Tensor or None)
        """
        if raw_scores_for_diag is not None and not self.should_collect_optimizer_diagnostics():
            raw_scores_for_diag = None

        # Apply score correction
        if score_correction is not None:
            scores = scores * score_correction
            if similarity is not None:
                similarity = similarity * (score_correction ** 2)
            if raw_scores_for_diag is not None:
                raw_scores_for_diag = raw_scores_for_diag * score_correction

        # Accumulate to state
        self.grad_dot_scores += scores.to(self.dtype)
        if self.similarity_matrix is not None and similarity is not None:
            self.similarity_matrix += similarity.to(self.dtype)

        if raw_scores_for_diag is not None:
            if self._diag_raw_scores is None:
                self._diag_raw_scores = torch.zeros_like(self.grad_dot_scores)
                self._diag_opt_scores = torch.zeros_like(self.grad_dot_scores)
            self._diag_raw_scores.add_(raw_scores_for_diag.to(self.dtype))
            self._diag_opt_scores.add_(scores.to(self.dtype))

    def get_final_selection(self) -> Tensor:
        """
        Compute global curation after all layers processed.

        Returns:
            Tensor of selected indices
        """
        if self.selection_variant == "soft":
            raise RuntimeError(
                "Soft weighting is continuous; call optimize_global_soft_weights()"
            )

        scores = self.grad_dot_scores * self.lr

        similarity = None
        if self.similarity_matrix is not None:
            similarity = self.similarity_matrix * (self.lr ** 2)

        if self.selection_variant == "random":
            generator = torch.Generator(device=scores.device)
            generator.manual_seed(self.derived_seed(-1))
            selected_indices = torch.randperm(
                self.train_batch_size, generator=generator, device=scores.device
            )[:self.num_selected]
        elif self._uses_exact_token_normalized_opta():
            selected_indices = self._select_exact_token_normalized_opta(
                self.grad_dot_scores, layer_idx=-1
            )
        elif self.selection_mode == "filtering":
            selected_indices = negative_filtering(scores, self.frac)
        elif self.use_second_order and similarity is not None:
            selected_indices = greedy_selection(scores, similarity, self.num_selected)
        elif self.selection_variant == "muon_spectral":
            from .advanced_solvers import seeded_topk
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Muon spectral scores contain non-finite values")
            selected_indices = seeded_topk(
                scores, self.num_selected, seed=self.derived_seed(-1)
            )
        else:
            selected_indices = topk_selection(scores, self.num_selected)

        self.num_selected = len(selected_indices)
        self.add_selection_score_diagnostics("global", self.grad_dot_scores, selected_indices)
        if self._diag_raw_scores is not None and self._diag_opt_scores is not None:
            self.add_raw_opt_score_diagnostics(
                "global",
                self._diag_raw_scores,
                self._diag_opt_scores,
                selected_indices,
            )

        # Record curation data for case study analysis
        if self._record_selections:
            self._selection_records = [{
                'selected_indices': selected_indices.tolist(),
                'scores': self.grad_dot_scores.detach().float().cpu().tolist(),
            }]

        return selected_indices

    def _compute_scale_factor_for_assembly(self, selected_indices: Tensor) -> Tensor:
        """
        Compute scale factor for one-pass gradient assembly (exact parity with two-pass).

        Uses batch_total_tokens (not train_total_tokens) to correct for merged-batch
        normalization. In two-pass mode, pass 2 computes loss on selected samples only,
        giving grad = (1/selected_tokens) * raw_grad. In one-pass, grad_output is scaled
        by 1/batch_total_tokens, so we need scale = batch_total_tokens / selected_tokens.

        For SeparateBatch, batch_total_tokens == train_total_tokens, so this is equivalent
        to the standard scale factor.
        """
        if self.tokens_per_sample is None or self.batch_total_tokens_tensor is None:
            raise RuntimeError(
                "Token counts not set. Call set_token_counts() before curation."
            )
        if selected_indices.numel() == 0:
            return torch.tensor(1.0, device=self.device, dtype=self.dtype)
        selected_tokens = self.tokens_per_sample[selected_indices].sum()
        scale = self.batch_total_tokens_tensor / selected_tokens
        return torch.where(selected_tokens == 0, torch.ones_like(scale), scale)

    def reset_accumulators(self) -> None:
        """Reset accumulators for next batch."""
        self.grad_dot_scores.zero_()
        if self.similarity_matrix is not None:
            self.similarity_matrix.zero_()


class OptimizerGroupWiseSubsetState(SelectionState):
    """
    State for optimizer-aware group-wise subset descent.

    Scores are accumulated across layers that share an optimizer-aware group
    key, then each group selects its own subset. This separates grouping from
    scoring geometry:
      - optimizer_aware=False: raw gradient scores over optimizer-aware groups
      - optimizer_aware=True: optimizer-induced scores over optimizer-aware groups
    """

    def __init__(
        self,
        scoring_method: str = "reduced_ghost",
        one_pass: bool = True,
        direct_batch_size: int = 0,
        group_keys: Optional[list] = None,
        optimizer_aware: bool = False,
        optimizer_aware_config: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.scoring_method = scoring_method
        self.direct_batch_size = direct_batch_size
        self.one_pass = one_pass
        self.group_keys = list(group_keys or [f"layer:{i}" for i in range(self.num_layers)])
        if len(self.group_keys) != self.num_layers:
            raise ValueError(
                f"group_keys length {len(self.group_keys)} does not match num_layers {self.num_layers}"
            )
        self.optimizer_aware = optimizer_aware
        self.optimizer_aware_config = optimizer_aware_config or {}

        self.group_scores: dict[str, Tensor] = {}
        self.group_similarity: dict[str, Tensor] = {}
        self.group_selected_indices: dict[str, Tensor] = {}
        self.group_diag_raw_scores: dict[str, Tensor] = {}
        self.group_diag_opt_scores: dict[str, Tensor] = {}

    def _group_for_layer(self, layer_idx: int) -> str:
        return self.group_keys[layer_idx]

    def _ensure_group(self, group_key: str) -> None:
        if group_key not in self.group_scores:
            self.group_scores[group_key] = torch.zeros(
                self.train_batch_size,
                device=self.device,
                dtype=self.dtype,
            )
            if self.use_second_order:
                self.group_similarity[group_key] = torch.zeros(
                    self.train_batch_size,
                    self.train_batch_size,
                    device=self.device,
                    dtype=self.dtype,
                )

    def process_layer_gradients(
        self,
        train_grads: Tensor,
        val_grad: Tensor,
        layer_idx: int,
        score_correction: Optional[Tensor] = None,
    ) -> None:
        if train_grads.dtype != self.dtype:
            train_grads = train_grads.to(self.dtype)
        if val_grad.dtype != self.dtype:
            val_grad = val_grad.to(self.dtype)

        scores = torch.mv(train_grads, val_grad)
        similarity = None
        if self.use_second_order:
            similarity = torch.mm(train_grads, train_grads.t())
        self.accumulate_precomputed_scores(scores, similarity, score_correction, layer_idx=layer_idx)
        return None

    def accumulate_precomputed_scores(
        self,
        scores: Tensor,
        similarity: Optional[Tensor],
        score_correction: Optional[Tensor] = None,
        layer_idx: Optional[int] = None,
        raw_scores_for_diag: Optional[Tensor] = None,
    ) -> None:
        if layer_idx is None:
            raise ValueError("OptimizerGroupWiseSubsetState requires layer_idx for score accumulation")

        if raw_scores_for_diag is not None and not self.should_collect_optimizer_diagnostics():
            raw_scores_for_diag = None

        group_key = self._group_for_layer(layer_idx)
        self._ensure_group(group_key)

        if score_correction is not None:
            scores = scores * score_correction
            if similarity is not None:
                similarity = similarity * (score_correction ** 2)
            if raw_scores_for_diag is not None:
                raw_scores_for_diag = raw_scores_for_diag * score_correction

        self.group_scores[group_key].add_(scores.to(self.dtype))
        if self.use_second_order and similarity is not None:
            self.group_similarity[group_key].add_(similarity.to(self.dtype))
        if raw_scores_for_diag is not None:
            if group_key not in self.group_diag_raw_scores:
                self.group_diag_raw_scores[group_key] = torch.zeros_like(self.group_scores[group_key])
                self.group_diag_opt_scores[group_key] = torch.zeros_like(self.group_scores[group_key])
            self.group_diag_raw_scores[group_key].add_(raw_scores_for_diag.to(self.dtype))
            self.group_diag_opt_scores[group_key].add_(scores.to(self.dtype))

    def finalize_group_selections(self) -> dict[str, Tensor]:
        self.group_selected_indices = {}
        for group_key, raw_scores in self.group_scores.items():
            scores = raw_scores * self.lr
            similarity = None
            if self.use_second_order and group_key in self.group_similarity:
                similarity = self.group_similarity[group_key] * (self.lr ** 2)

            if self.selection_mode == "filtering":
                selected = negative_filtering(scores, self.frac)
            elif self.use_second_order and similarity is not None:
                selected = greedy_selection(scores, similarity, self.num_selected)
            else:
                selected = topk_selection(scores, self.num_selected)
            self.group_selected_indices[group_key] = selected
            self.add_selection_score_diagnostics(group_key, raw_scores, selected)
            if group_key in self.group_diag_raw_scores and group_key in self.group_diag_opt_scores:
                self.add_raw_opt_score_diagnostics(
                    group_key,
                    self.group_diag_raw_scores[group_key],
                    self.group_diag_opt_scores[group_key],
                    selected,
                )

        if self._record_selections:
            self._selection_records = [
                {
                    "group_key": group_key,
                    "selected_indices": selected.detach().cpu().tolist(),
                    "scores": self.group_scores[group_key].detach().float().cpu().tolist(),
                }
                for group_key, selected in self.group_selected_indices.items()
            ]
        return self.group_selected_indices

    def get_selected_indices_for_layer(self, layer_idx: int) -> Tensor:
        if not self.group_selected_indices:
            self.finalize_group_selections()
        group_key = self._group_for_layer(layer_idx)
        selected = self.group_selected_indices.get(group_key)
        if selected is None:
            return torch.empty(0, device=self.device, dtype=torch.long)
        return selected.sort()[0]

    def get_final_selection(self) -> Tensor:
        """Return the union of selected indices across optimizer-aware groups."""
        if not self.group_selected_indices:
            self.finalize_group_selections()
        if not self.group_selected_indices:
            return torch.empty(0, device=self.device, dtype=torch.long)
        return torch.unique(torch.cat(list(self.group_selected_indices.values()))).sort()[0]

    def _compute_scale_factor_for_layer(self, layer_idx: int) -> Tensor:
        selected_indices = self.get_selected_indices_for_layer(layer_idx)
        if self.tokens_per_sample is None or self.batch_total_tokens_tensor is None:
            raise RuntimeError(
                "Token counts not set. Call set_token_counts() before curation."
            )
        if selected_indices.numel() == 0:
            return torch.tensor(1.0, device=self.device, dtype=self.dtype)
        selected_tokens = self.tokens_per_sample[selected_indices].sum()
        scale = self.batch_total_tokens_tensor / selected_tokens
        return torch.where(selected_tokens == 0, torch.ones_like(scale), scale)
