"""
Layer-Wise Subset SFT trainer with unified data curation and model update.
"""

import json
import math
import os
import time
import logging
from collections.abc import MutableMapping
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler, Sampler, SequentialSampler
from typing import Dict
from torch import Tensor

from transformers import Trainer

from drpt.optimizer import MuonWithAuxAdamW, MeSOAdamW
from drpt.hook import GradientHook
from drpt.selection import create_separate_batch_strategy, create_merged_batch_strategy
from SFT.train.collator import META_IDX_KEY
from SFT.train.target_signal import (
    CORRECT_INCORRECT_MARGIN,
    GROUP_IDS_KEY,
    NLL,
    REWARD_WEIGHTED_SFT,
    REWARDS_KEY,
    ROLES_KEY,
    canonicalize_target_signal_mode,
    compute_target_signal_loss_from_batch,
    grouped_slice_normalization_weight,
    iter_grouped_logical_slices,
    iter_logical_slices,
    model_inputs_from_target_batch,
    slice_normalization_weight,
    token_mean_normalization_weights,
)

# Signals whose loss couples several trajectories of one prompt. They need the
# grouped collator, logits-level loss, and group-aware microbatch slicing; the
# other two are ordinary per-token cross entropy and keep the original path.
_GROUPED_TARGET_SIGNALS = frozenset({CORRECT_INCORRECT_MARGIN, REWARD_WEIGHTED_SFT})

logger = logging.getLogger(__name__)

_ADVANCED_TARGET_METHODS = frozenset({
    "GlobalSoftWeighting",
    "LayerWiseSoftWeighting",
    "LayerWiseSoftProbability",
    "GlobalMuonSpectral",
    "LayerWiseMuonSpectral",
    "GlobalMuonMatrixSpectral",
    "LayerWiseMuonMatrixSpectral",
    "LayerWiseMuonMatrixSpectralP",
    "LayerWiseMuonMatrixSpectralSat",
    "LayerWiseMuonMatrixSpectralSatP",
})


class ManifestOrderSampler(Sampler):
    """Yield one immutable, without-replacement candidate traversal.

    The indices are resolved from stable example IDs in the audited artifact,
    rather than generated from model/optimizer RNG state at runtime.
    """

    def __init__(self, indices):
        self.indices = tuple(int(index) for index in indices)
        if any(index < 0 for index in self.indices):
            raise ValueError("ManifestOrderSampler indices must be non-negative")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("ManifestOrderSampler indices must be unique")

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def _compact_selection_metrics(metrics: Dict) -> Dict[str, float]:
    """Collapse per-layer solver diagnostics into one finite metric per field.

    Selection counters and the existing update/alignment diagnostics retain
    their original keys.  Soft-weighting and spectral diagnostics are averaged
    over their layer/group component, e.g.
    ``soft/layer_3/entropy`` becomes ``soft/entropy``.  The full per-layer
    dictionary remains available on the selection strategy, and optional
    detailed selection-record capture is unaffected; this helper only bounds
    the logging payload.
    """
    compact = {}
    diagnostic_groups = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            numeric_value = float(value)
        elif torch.is_tensor(value) and value.numel() == 1:
            numeric_value = float(value.detach().float().cpu().item())
        else:
            continue
        if not math.isfinite(numeric_value):
            continue
        if key.startswith(('soft/', 'spectral/')):
            family = key.split('/', 1)[0]
            metric_name = key.rsplit('/', 1)[-1]
            diagnostic_groups.setdefault(f'{family}/{metric_name}', []).append(
                numeric_value
            )
        elif key.startswith(('selection/', 'update/', 'diag/')):
            compact[key] = numeric_value
    for key, values in diagnostic_groups.items():
        compact[key] = sum(values) / len(values)
    return compact


class LayerWiseSubsetTrainer(Trainer):
    """
    SFT Trainer supporting gradient-based data curation with optional compression.
    """

    def __init__(
        self,
        grad_hook: GradientHook,
        val_dataset=None,
        *args,
        target_grad_dataset=None,
        target_val_dataset=None,
        general_val_dataset=None,
        train_order_indices=None,
        target_collator=None,
        **kwargs,
    ):
        """
        Initialize the trainer.

        Args:
            grad_hook: GradientHook instance for gradient capture and compression.
            val_dataset: Small validation set used for data curation during training.
                Batches from this set are merged with training batches to compute
                curation scores that guide which training samples to use.
            target_collator: Collator for the target-gradient dataloader only.
                Grouped target signals nest several trajectories under one
                prompt, which the model collator cannot batch. When omitted the
                target dataloader keeps using data_collator, so the historical
                NLL path is byte-for-byte unchanged.
            *args, **kwargs: Same as transformers.Trainer.
                Must include eval_dataset: Large held-out set for generalization testing.
                This is separate from val_dataset and used only during evaluation.
        """
        # Set this before Trainer.__init__ so any future eager dataloader
        # construction also observes the immutable order.
        self._manifest_train_order = (
            None if train_order_indices is None else tuple(train_order_indices)
        )

        # Extract eval_dataset from kwargs before passing to parent
        if target_grad_dataset is not None:
            val_dataset = target_grad_dataset
        if general_val_dataset is not None:
            kwargs["eval_dataset"] = general_val_dataset
        eval_dataset = kwargs.get('eval_dataset', None)
        if eval_dataset is None:
            raise ValueError("LayerWiseSubsetTrainer requires an eval_dataset to be passed in kwargs")

        # Pass eval_dataset to parent Trainer
        super().__init__(*args, **kwargs)

        if self._manifest_train_order is not None:
            expected = len(self.train_dataset)
            if len(self._manifest_train_order) != expected:
                raise ValueError(
                    "Stored candidate order must cover the complete train pool: "
                    f"order={len(self._manifest_train_order)}, dataset={expected}"
                )
            if set(self._manifest_train_order) != set(range(expected)):
                raise ValueError(
                    "Stored candidate order must be a permutation of every train row"
                )

        # Store our custom datasets
        self.grad_hook = grad_hook
        self.target_collator = target_collator
        self.target_signal_mode = canonicalize_target_signal_mode(
            getattr(self.args, "target_signal_mode", NLL)
        )
        if self.target_signal_mode in _GROUPED_TARGET_SIGNALS and target_collator is None:
            raise ValueError(
                f"target_signal_mode={self.target_signal_mode!r} nests several "
                "trajectories per prompt and requires a target_collator"
            )
        self.val_dataset = val_dataset
        self.target_grad_dataset = val_dataset
        self.target_val_dataset = (
            target_val_dataset if target_val_dataset is not None else val_dataset
        )
        self.general_val_dataset = eval_dataset
        self.eval_dataset_custom = eval_dataset
        self._step0_general_val_loss = None
        self._step0_target_val_loss = None

        # Set selection_frac to 1.0 when no curation method is specified (consistency)
        # This means we "select" all samples when not doing data curation
        if not hasattr(self.args, 'selection_frac') or self.args.selection_frac is None:
            if self.args.method == 'NA':
                self.args.selection_frac = 1.0

        # Initialize results tracking
        self.evaluation_results = []
        self.selection_diagnostics_history = []

        # Track wall time using CUDA events for accurate GPU timing
        if torch.cuda.is_available():
            self.training_start_event = torch.cuda.Event(enable_timing=True)
            self.training_start_event.record()
        else:
            self.training_start_event = None
            self.training_start_time = time.time()  # Fallback for CPU

        # Track cumulative evaluation time so it can be subtracted from wall time
        self.cumulative_eval_time = 0.0

        # Determine validation batch size for data curation
        # Use val_batch_size_for_selection if provided, otherwise default to per_device_train_batch_size
        self.val_batch_size_for_selection = (
            self.args.val_batch_size_for_selection
            if hasattr(self.args, 'val_batch_size_for_selection') and self.args.val_batch_size_for_selection is not None
            else self.args.per_device_train_batch_size
        )

        # Keep target-gradient sampling independent from model/optimizer RNG use.
        # Every immutable 32k method must see the same deterministic stream of scarce
        # target examples, even when optimizer setup consumes a different RNG path.
        self._target_sampler_generator = torch.Generator()
        self._target_sampler_generator.manual_seed(
            int(getattr(self.args, "data_seed", None) or self.args.seed)
        )

        # Create validation dataloader iterator for efficient batch sampling during training
        # Only needed if we're doing layer_wise_subset data curation
        if val_dataset is not None:
            self.val_dataloader_iter = iter(
                self.get_val_dataloader(self.val_dataset, batch_size=self.val_batch_size_for_selection, shuffle=True)
            )
        else:
            self.val_dataloader_iter = None

        # Determine if we have update compression (which implies MeSO)
        self.has_compression = (
            self.grad_hook is not None and
            self.grad_hook.compression_mode.uses_compressed_updates
        )

        # Curation recording for case study analysis
        self._record_selections = getattr(self.args, 'record_selections', False)
        self._record_selections_freq = max(1, getattr(self.args, 'record_selections_freq', 1))
        self._selection_records = []

        # Per-domain selection tracking. Cheap (integer bookkeeping only) but it
        # forces a per-layer GPU->CPU sync to read the selected indices, so it is
        # sampled on a stride instead of every step. `pool_metadata` is attached
        # by train.py; without it tracking stays off.
        self._track_selection_domains = getattr(self.args, 'track_selection_domains', True)
        self._track_selection_domains_freq = max(
            1, getattr(self.args, 'track_selection_domains_freq', 10)
        )
        self._pool_metadata = None
        self._domain_candidates = {}
        self._domain_selected = {}
        self._source_candidates = {}
        self._source_selected = {}
        self._domain_steps_tracked = 0
        self._domain_timeline = []
        self._last_meta_idx = None

        # Create curation strategy for clean separation of curation methods
        # SFT uses topk mode: select top frac samples by alignment score
        # Strategy is determined by val_strategy argument:
        # - separate_batch_factorized: Separate val pass, store factorized components (default)
        # - separate_batch: Separate val pass, store mean gradient
        # - merged_batch: Merge train+val into single batch
        val_strategy = getattr(self.args, 'val_strategy', 'separate_batch_factorized')
        scoring_method = getattr(self.args, 'scoring_method', 'reduced_ghost')
        subset_mode = getattr(self.args, 'subset_mode', 'one_pass')
        solver_kwargs = {
            'seed': getattr(self.args, 'seed', 42),
            'soft_weighting_steps': getattr(self.args, 'soft_weighting_steps', 20),
            'soft_weighting_lr': getattr(self.args, 'soft_weighting_lr', 0.1),
            'soft_weighting_tol': getattr(self.args, 'soft_weighting_tol', 1e-5),
            'soft_weighting_patience': getattr(self.args, 'soft_weighting_patience', 3),
            'soft_weighting_gamma': getattr(self.args, 'soft_weighting_gamma', 0.0),
            'soft_weighting_use_optimizer_state': getattr(
                self.args, 'soft_weighting_use_optimizer_state', True
            ),
            'soft_weighting_constraint': getattr(
                self.args, 'soft_weighting_constraint', 'capped_simplex'
            ),
            'soft_replay_precision': getattr(
                self.args, 'soft_replay_precision', 'fp32'
            ),
            'muon_surrogate_alpha': getattr(self.args, 'muon_surrogate_alpha', 1.0),
            'muon_surrogate_rank': getattr(self.args, 'muon_surrogate_rank', 32),
            'muon_surrogate_full_svd_max_dim': getattr(
                self.args, 'muon_surrogate_full_svd_max_dim', 256
            ),
            'muon_surrogate_rtol': getattr(self.args, 'muon_surrogate_rtol', 1e-6),
            'muon_surrogate_oversample': getattr(self.args, 'muon_surrogate_oversample', 8),
            'muon_surrogate_power_iters': getattr(self.args, 'muon_surrogate_power_iters', 2),
            'muon_surrogate_include_adamw_scores': getattr(
                self.args, 'muon_surrogate_include_adamw_scores', True
            ),
            'muon_surrogate_mode_weighting': getattr(
                self.args, 'muon_surrogate_mode_weighting', 'uniform'
            ),
            'muon_surrogate_saturation': getattr(
                self.args, 'muon_surrogate_saturation', False
            ),
            'optimizer_aware_diagnostic_interval': getattr(
                self.args, 'optimizer_aware_diagnostic_interval', 0
            ),
        }
        # The strategy-level flag also gates domain tracking, which reads the
        # same per-unit selection records. The expensive text capture in
        # _capture_selection_record stays governed by args.record_selections.
        strategy_records = self._record_selections or self._track_selection_domains
        if val_strategy == 'merged_batch':
            self.selection_strategy = create_merged_batch_strategy(
                method=self.args.method,
                grad_hook=self.grad_hook,
                frac=getattr(self.args, 'selection_frac', 0.5),
                use_second_order=getattr(self.args, 'use_second_order', False),
                selection_mode=getattr(self.args, 'selection_mode', 'topk'),
                record_selections=strategy_records,
                scoring_method=scoring_method,
                subset_mode=subset_mode,
                **solver_kwargs,
            )
        else:
            # separate_batch_factorized or separate_batch
            self.selection_strategy = create_separate_batch_strategy(
                method=self.args.method,
                grad_hook=self.grad_hook,
                frac=getattr(self.args, 'selection_frac', 0.5),
                use_second_order=getattr(self.args, 'use_second_order', False),
                selection_mode=getattr(self.args, 'selection_mode', 'topk'),
                record_selections=strategy_records,
                scoring_method=scoring_method,
                subset_mode=subset_mode,
                windowed=self._windowed_execution_enabled(),
                **solver_kwargs,
            )
        self.val_strategy = val_strategy
        if self._grouped_target_signal() and val_strategy == 'merged_batch':
            # merged_batch concatenates target rows onto the training batch and
            # reads one model loss off the pair. A grouped signal needs its own
            # forward with group ids intact, which that path cannot express.
            raise ValueError(
                f"target_signal_mode={self.target_signal_mode!r} requires the "
                "separate-batch validation strategy"
            )

        logger.info("="*60)
        logger.info("Initialized LayerWiseSubsetTrainer")
        logger.info(f"  Target-gradient signal: {self.target_signal_mode}")
        selection_frac = getattr(self.args, 'selection_frac', None)
        logger.info(f"  Method: {self.args.method} (curation fraction: {selection_frac})")
        logger.info(f"  Validation strategy: {self.val_strategy}")
        logger.info(f"  Compression: {self.has_compression}")
        logger.info(f"  Validation set size: {len(val_dataset) if val_dataset is not None else 0}")
        logger.info(f"  Evaluation set size: {len(eval_dataset) if eval_dataset is not None else 0}")
        logger.info(f"  Training batch size: {self.args.per_device_train_batch_size}")
        logger.info(f"  Validation batch size (for curation): {self.val_batch_size_for_selection}")
        if self._record_selections:
            logger.info(f"  Curation recording: enabled (every {self._record_selections_freq} steps)")

        # Log the training mode based on configuration
        # Naming convention: {curation}-{compression}-{training_type}
        selection_methods = (
            'LayerWiseSubset',
            'LayerWiseOptimizerAwareSubset',
            'GlobalSubset',
            'OptimizerAwareGlobalSubset',
            'OptimizerGroupWise',
            'OptimizerAwareGroupWise',
            'GlobalRandomSubset',
            'LayerWiseRandomSubset',
            'GlobalSoftWeighting',
            'LayerWiseSoftWeighting',
            'LayerWiseSoftProbability',
            'GlobalMuonSpectral',
            'LayerWiseMuonSpectral',
            'GlobalMuonMatrixSpectral',
            'LayerWiseMuonMatrixSpectral',
            'LayerWiseMuonMatrixSpectralP',
            'LayerWiseMuonMatrixSpectralSat',
            'LayerWiseMuonMatrixSpectralSatP',
        )
        if self.args.method in selection_methods:
            if self.has_compression:
                logger.info(f"  Mode: {self.args.method} with compression (MeSO optimizer)")
            else:
                logger.info(f"  Mode: {self.args.method} without compression (standard optimizer)")
        elif self.has_compression:
            logger.info(f"  Mode: MeSO only (compressed gradients, no curation)")
        else:
            logger.info(f"  Mode: Baseline (full gradients, no curation)")

        logger.info("="*60)

    def _maybe_log_selection_metrics(self, stats: Dict = None) -> None:
        """Log cheap curation diagnostics collected during the current step."""
        metrics = {}
        if stats:
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    metrics[key] = value
                elif torch.is_tensor(value) and value.numel() == 1:
                    metrics[key] = value.detach().float().cpu().item()

        diag = getattr(self.selection_strategy, 'last_diagnostic_metrics', None)
        if diag:
            metrics.update(diag)

        if not metrics:
            return
        compact_metrics = _compact_selection_metrics(metrics)
        if compact_metrics:
            snapshot = {'step': int(self.state.global_step), **compact_metrics}
            self.selection_diagnostics_history.append(snapshot)

        logging_steps = max(1, int(getattr(self.args, 'logging_steps', 1) or 1))
        if compact_metrics and self.state.global_step % logging_steps == 0:
            self.log(compact_metrics)

    def _get_unwrapped_optimizer(self):
        """
        Get the underlying optimizer, unwrapping AcceleratedOptimizer if present.

        Returns:
            The underlying optimizer (e.g., MeSOAdamW or AdamW)
        """
        optimizer = self.optimizer
        if hasattr(optimizer, 'optimizer'):
            optimizer = optimizer.optimizer
        return optimizer

    def create_optimizer(self):
        """
        Setup the optimizer based on compression setting.

        - With compression: Use MeSOAdamW (compressed optimizer states)
        - Without compression: Use standard AdamW
        """
        if self.optimizer is not None:
            if self.grad_hook is not None:
                self.grad_hook.set_optimizer(self._get_unwrapped_optimizer())
            return self.optimizer

        optimizer_type = getattr(self.args, 'optimizer_type', 'adamw')
        if self.has_compression and optimizer_type in ("muon", "hybrid"):
            raise ValueError(
                f"optimizer_type={optimizer_type!r} cannot be combined with "
                "MeSO/update compression. Refusing to silently replace the "
                "official-first Muon runtime with MeSOAdamW."
            )
        if self.has_compression:
            logger.info("Using MeSOAdamW optimizer (compression enabled)")

            # Create compressed optimizer
            self.optimizer = MeSOAdamW(
                params=self.model.parameters(),
                grad_hook=self.grad_hook,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
        elif optimizer_type in ("muon", "hybrid"):
            logger.info(
                "Using MuonWithAuxAdamW facade for optimizer_type=%s "
                "(eligible 2D hidden weights=Muon, rest=auxiliary AdamW)",
                optimizer_type,
            )
            self.optimizer = MuonWithAuxAdamW(
                named_params=self.model.named_parameters(),
                model=self.model,
                lr=self.args.learning_rate,
                muon_lr=getattr(self.args, 'muon_learning_rate', None),
                aux_adamw_lr=getattr(
                    self.args, 'aux_adamw_learning_rate', None
                ),
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
                muon_momentum=getattr(self.args, 'optimizer_aware_muon_momentum', 0.95),
                muon_nesterov=getattr(self.args, 'optimizer_aware_muon_nesterov', True),
                muon_ns_steps=getattr(self.args, 'optimizer_aware_muon_steps', 5),
                muon_eps=getattr(self.args, 'optimizer_aware_muon_eps', 1e-7),
                muon_lr_shape_scale=getattr(self.args, 'optimizer_aware_muon_lr_shape_scale', True),
                muon_adjust_lr_fn=getattr(self.args, 'optimizer_aware_muon_adjust_lr_fn', 'original'),
                muon_backend=getattr(self.args, 'optimizer_aware_muon_backend', 'auto'),
                lora_optimizer=getattr(self.args, 'optimizer_aware_lora_optimizer', 'adamw'),
            )
        else:
            # Use standard Hugging Face optimizer creation
            logger.info("Using standard AdamW optimizer (no compression)")
            super().create_optimizer()

        if self.grad_hook is not None:
            unwrapped_optimizer = self._get_unwrapped_optimizer()
            self.grad_hook.set_optimizer(unwrapped_optimizer)
            if hasattr(unwrapped_optimizer, "resolved_muon_backend"):
                # Numerical scoring reads exact live groups/state through
                # set_optimizer; keep the descriptive config in sync as well.
                self.grad_hook.configure_optimizer_aware(
                    muon_backend=unwrapped_optimizer.resolved_muon_backend
                )

        unwrapped_optimizer = self._get_unwrapped_optimizer()
        if hasattr(unwrapped_optimizer, "get_runtime_metadata"):
            self.optimizer_runtime_metadata = (
                unwrapped_optimizer.get_runtime_metadata()
            )
            logger.info(
                "Resolved optimizer runtime: %s",
                self.optimizer_runtime_metadata,
            )
        else:
            self.optimizer_runtime_metadata = {
                "optimizer_runtime_class": (
                    f"{type(unwrapped_optimizer).__module__}."
                    f"{type(unwrapped_optimizer).__qualname__}"
                )
            }

        return self.optimizer

    def _get_train_sampler(self, train_dataset=None):
        if self._manifest_train_order is not None:
            dataset = train_dataset if train_dataset is not None else self.train_dataset
            if len(dataset) != len(self._manifest_train_order):
                raise ValueError(
                    "Cannot apply stored candidate order to a differently sized dataset"
                )
            return ManifestOrderSampler(self._manifest_train_order)
        return super()._get_train_sampler(train_dataset)

    def get_val_dataloader(self, val_dataset, batch_size=4, shuffle=True):
        """Create validation dataloader for data curation."""
        if shuffle:
            sampler = RandomSampler(
                val_dataset,
                generator=self._target_sampler_generator,
            )
        else:
            sampler = SequentialSampler(val_dataset)

        return DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=getattr(self, "target_collator", None) or self.data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _merge_batches(self, batch_train: Dict[str, Tensor], batch_val: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Merge training and validation batches along batch dimension.

        NOTE: This method is kept for backward compatibility but is no longer used
        by default. The trainer now uses separate val/train passes via StoredValStrategy
        to avoid padding overhead when batches have different sequence lengths.

        Handles the case where batches have different sequence lengths by padding
        to the maximum length across both batches.

        Args:
            batch_train: Training batch
            batch_val: Validation batch

        Returns:
            Merged batch with train samples first, then val samples
        """
        merged_batch = {}

        # Find max sequence length for padding
        max_seq_len = max(
            batch_train.get('input_ids', batch_train.get('attention_mask')).shape[1],
            batch_val.get('input_ids', batch_val.get('attention_mask')).shape[1]
        )

        for key in batch_train.keys():
            if key not in batch_val:
                # If key not in val batch, just use train batch value
                merged_batch[key] = batch_train[key]
                continue

            train_tensor = batch_train[key]
            val_tensor = batch_val[key]

            # Pad to max sequence length if needed (for 2D tensors like input_ids, attention_mask, labels)
            if train_tensor.dim() == 2:
                train_seq_len = train_tensor.shape[1]
                val_seq_len = val_tensor.shape[1]

                # Determine padding value based on key
                if key == 'attention_mask':
                    pad_value = 0
                elif key == 'labels':
                    pad_value = -100  # Standard ignore index for CrossEntropyLoss
                else:  # input_ids and other token-based fields
                    # Use processing_class's pad_token_id if available, otherwise 0
                    processing_class = getattr(self, 'processing_class', None)
                    pad_value = processing_class.pad_token_id if processing_class is not None and hasattr(processing_class, 'pad_token_id') and processing_class.pad_token_id is not None else 0

                # Pad train batch if needed
                if train_seq_len < max_seq_len:
                    padding = torch.full(
                        (train_tensor.shape[0], max_seq_len - train_seq_len),
                        pad_value,
                        dtype=train_tensor.dtype,
                        device=train_tensor.device
                    )
                    train_tensor = torch.cat([train_tensor, padding], dim=1)

                # Pad val batch if needed
                if val_seq_len < max_seq_len:
                    padding = torch.full(
                        (val_tensor.shape[0], max_seq_len - val_seq_len),
                        pad_value,
                        dtype=val_tensor.dtype,
                        device=val_tensor.device
                    )
                    val_tensor = torch.cat([val_tensor, padding], dim=1)

            # Concatenate along batch dimension (dim=0)
            merged_batch[key] = torch.cat([train_tensor, val_tensor], dim=0)

        return merged_batch

    @staticmethod
    def _slice_model_batch(batch, start: int, end: int):
        """Slice batch-shaped tensors while preserving scalar/non-tensor values."""
        size = int(batch["input_ids"].shape[0])
        sliced = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == size:
                sliced[key] = value[start:end]
            else:
                sliced[key] = value
        return sliced

    def _windowed_execution_enabled(self) -> bool:
        candidate = getattr(self.args, "candidate_microbatch_size", None)
        logical = getattr(self.args, "logical_candidate_batch_size", None)
        return (
            candidate is not None
            and logical is not None
            and int(candidate) < int(logical)
        )

    def _grouped_target_signal(self) -> bool:
        return self.target_signal_mode in _GROUPED_TARGET_SIGNALS

    def _compute_target_signal_loss(self, model, batch):
        """Target-domain loss under the configured target-gradient signal.

        ``nll`` and ``answer_only_ce`` are the same token-mean cross entropy and
        keep the original code path: the model computes its own loss, so the
        full ``[B, S, V]`` logits are never materialized outside the fused
        kernel.  Only the grouped signals, whose loss couples trajectories the
        model cannot see, need logits at this level.
        """

        if not self._grouped_target_signal():
            # Drop the collator's grouping metadata so model(**batch) still
            # works. A historical NLL batch carries none of these keys, so this
            # is a no-op there and that path stays exactly as it was.
            model_batch = {
                key: value
                for key, value in batch.items()
                if key not in (GROUP_IDS_KEY, ROLES_KEY, REWARDS_KEY)
            }
            return self._compute_loss_for_selection(model, model_batch)

        with self.compute_loss_context_manager():
            outputs = model(**model_inputs_from_target_batch(batch))
            loss = compute_target_signal_loss_from_batch(
                outputs.logits,
                batch,
                mode=self.target_signal_mode,
                beta=float(getattr(self.args, "target_signal_beta", 1.0)),
                margin=float(getattr(self.args, "target_signal_margin", 0.0)),
            )
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        return loss

    def _target_logical_slices(self, batch_val, labels):
        """Yield ``(start, end, fraction)`` chunks of one logical target batch.

        ``fraction`` is the chunk's exact share of the unsliced logical loss, so
        summing the scaled chunk losses reproduces the loss -- and therefore the
        gradient -- that a single unchunked backward would have produced. Each
        signal has its own denominator: token counts for cross entropy, reward
        weighted token counts for reward_weighted_sft, and whole prompt groups
        for the margin loss, which cannot be split mid-pair at all.
        """

        num_rows = int(labels.shape[0])
        if self.target_signal_mode == CORRECT_INCORRECT_MARGIN:
            group_ids = batch_val.get(GROUP_IDS_KEY)
            if group_ids is None:
                raise RuntimeError("margin target batches must contain group_ids")
            groups_per_microbatch = int(
                getattr(self.args, "target_signal_groups_per_microbatch", None)
                or len(torch.unique(group_ids))
            )
            for start, end in iter_grouped_logical_slices(group_ids, groups_per_microbatch):
                yield start, end, grouped_slice_normalization_weight(group_ids, start, end)
            return

        microbatch = int(getattr(self.args, "target_microbatch_size", None) or num_rows)
        if microbatch > num_rows:
            raise ValueError("target_microbatch_size exceeds logical target batch")

        if self.target_signal_mode == REWARD_WEIGHTED_SFT:
            rewards = batch_val.get(REWARDS_KEY)
            if rewards is None:
                raise RuntimeError("reward-weighted target batches must contain rewards")
            weights = token_mean_normalization_weights(labels, rewards.double())
            for start, end in iter_logical_slices(num_rows, microbatch):
                yield start, end, float(slice_normalization_weight(weights, start, end))
            return

        # Plain token fractions, evaluated in double and rounded once by the
        # caller's cast. That is bit-identical to the historical
        # tokens[start:end].sum() / total_tokens this replaced.
        tokens = (labels[:, 1:] != -100).sum(dim=1)
        total_tokens = float(tokens.sum())
        for start, end in iter_logical_slices(num_rows, microbatch):
            yield start, end, float(tokens[start:end].sum()) / total_tokens

    def _accumulate_target_signal_backward(self, model, batch_val, labels):
        """Backward one logical target batch chunk by chunk, returning its loss.

        Each chunk's loss is scaled by its exact share of the logical loss, so
        the accumulated gradient equals the one an unchunked backward would have
        produced. Chunking is what keeps peak memory bounded when the target
        loss is computed from logits.
        """

        logical_loss = torch.zeros((), device=labels.device, dtype=torch.float32)
        for start, end, fraction in self._target_logical_slices(batch_val, labels):
            chunk = self._slice_model_batch(batch_val, start, end)
            loss = self._compute_target_signal_loss(model, chunk)
            scaled = loss * torch.as_tensor(fraction, device=loss.device, dtype=loss.dtype)
            logical_loss = logical_loss + scaled.detach().float()
            scaled.backward()
        return logical_loss

    def _capture_target_window(self, model, batch_val):
        """Capture one logical target gradient over target microbatches."""
        labels = batch_val.get("labels")
        if labels is None:
            raise RuntimeError("target_grad_dataset batches must contain labels")
        # Causal-LM CE predicts labels[:, 1:] from logits[:, :-1].  Count the
        # same supervised positions used by the model loss so logical-window
        # normalization remains exact even for an unusual labelled first token.
        tokens = (labels[:, 1:] != -100).sum(dim=1)
        total_tokens = tokens.sum()
        if not bool((total_tokens > 0).detach().cpu()):
            raise RuntimeError("Target-gradient logical batch contains no valid tokens")

        self.grad_hook.start_val_capture(
            scoring_method=getattr(self.args, "scoring_method", "reduced_ghost"),
            # Dolci32k exact windows keep the complete target cache in FP32 for
            # every method, on the target_cache_device selected by the profile.
            full_precision=True,
        )
        model.zero_grad(set_to_none=True)
        logical_loss = self._accumulate_target_signal_backward(model, batch_val, labels)
        self.grad_hook.end_val_capture(
            val_total_tokens=int(total_tokens.detach().cpu().item())
        )
        if str(getattr(self.args, "target_cache_device", "cuda")).lower() == "cpu":
            self.grad_hook.val_cache.offload_to_cpu(
                pin_memory=bool(
                    getattr(self.args, "target_cache_pin_memory", False)
                )
            )
        model.zero_grad(set_to_none=True)
        return logical_loss.detach()

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Training step using curation strategy pattern.

        The curation strategy handles the difference between:
        - LayerWiseSubset: Single-pass, per-layer curation
        - GlobalSubset: Two-pass, global curation
        - NA: Baseline (no curation)

        With or without compression (MeSO).
        """
        model.train()
        args = self.args

        # `meta_idx` rides along with the batch purely so selected examples can
        # be traced back to their source pool row. Strip it before anything can
        # forward it into model(**batch). The check is on MutableMapping, not
        # dict: HF collators return BatchEncoding, which is a UserDict and would
        # fail an isinstance(..., dict) test.
        self._last_meta_idx = None
        if isinstance(inputs, MutableMapping) and META_IDX_KEY in inputs:
            meta_idx = inputs.pop(META_IDX_KEY)
            self._last_meta_idx = meta_idx.tolist() if torch.is_tensor(meta_idx) else list(meta_idx)

        record_this_step = self._should_record_selections()
        track_domains = self._should_track_domains()
        if self.selection_strategy is not None:
            self.selection_strategy.record_selections = bool(
                record_this_step or track_domains
            )

        # Refresh compressors if needed (only for MeSO)
        if self.grad_hook is not None and self.optimizer is not None:
            self.grad_hook.set_optimizer(self._get_unwrapped_optimizer())

        if self.has_compression:
            unwrapped_optimizer = self._get_unwrapped_optimizer()
            if isinstance(unwrapped_optimizer, MeSOAdamW):
                unwrapped_optimizer.refresh_compressors_if_needed()

        # === DATA CURATION MODE (LayerWiseSubset or GlobalSubset) ===
        if args.method in (
            'LayerWiseSubset',
            'LayerWiseOptimizerAwareSubset',
            'GlobalSubset',
            'OptimizerAwareGlobalSubset',
            'OptimizerGroupWise',
            'OptimizerAwareGroupWise',
            'GlobalRandomSubset',
            'LayerWiseRandomSubset',
            'GlobalSoftWeighting',
            'LayerWiseSoftWeighting',
            'LayerWiseSoftProbability',
            'GlobalMuonSpectral',
            'LayerWiseMuonSpectral',
            'GlobalMuonMatrixSpectral',
            'LayerWiseMuonMatrixSpectral',
            'LayerWiseMuonMatrixSpectralP',
            'LayerWiseMuonMatrixSpectralSat',
            'LayerWiseMuonMatrixSpectralSatP',
        ):
            # Get validation batch for curation
            try:
                val_batch = next(self.val_dataloader_iter)
            except StopIteration:
                self.val_dataloader_iter = iter(
                    self.get_val_dataloader(self.val_dataset, batch_size=self.val_batch_size_for_selection, shuffle=True)
                )
                val_batch = next(self.val_dataloader_iter)

            # Prepare inputs
            batch_train = self._prepare_inputs(inputs)
            batch_val = self._prepare_inputs(val_batch)
            train_batch_size = batch_train['input_ids'].shape[0]

            # Get current learning rate
            lr = self.optimizer.param_groups[0]["lr"] if hasattr(self, 'optimizer') and self.optimizer else args.learning_rate
            if lr is None or lr == 0:
                lr = args.learning_rate

            if self.val_strategy == 'merged_batch':
                # === MERGED BATCH MODE: Merge train+val, single forward/backward ===
                merged_batch = self._merge_batches(batch_train, batch_val)

                def compute_loss(model, batch):
                    return self._compute_loss_for_selection(model, batch)

                loss = self.selection_strategy.execute_training_step(
                    model=model,
                    merged_batch=merged_batch,
                    train_batch_size=train_batch_size,
                    compute_loss_fn=compute_loss,
                    lr=lr,
                    batch_train=batch_train,  # For GlobalSubset pass 2
                    global_step=self.state.global_step,
                )
                stats = {}
            else:
                # === SEPARATE BATCH MODE: Separate val pass, then train with stored grads ===
                # Storage mode (factorized/full/compressed) is derived from scoring_method:
                #   full_ghost   → factorized [V,S,O] + [V,S,I] (for pairwise scoring)
                #   reduced_ghost/direct → full [O,I] (summed gradient, cheaper)
                #   compress     → compressed [k]
                # PASS 1: Capture validation gradients
                val_labels = batch_val.get('labels')
                val_total_tokens = None
                if val_labels is not None:
                    val_total_tokens = int(
                        (val_labels[:, 1:] != -100).sum().detach().cpu().item()
                    )
                if (
                    self.args.method in _ADVANCED_TARGET_METHODS
                    and (val_total_tokens is None or val_total_tokens <= 0)
                ):
                    raise RuntimeError(
                        "Advanced solver target batch contains no valid tokens"
                    )
                if self._windowed_execution_enabled():
                    val_loss = self._capture_target_window(model, batch_val)
                else:
                    self.grad_hook.start_val_capture(
                        scoring_method=getattr(self.args, 'scoring_method', 'reduced_ghost'),
                        full_precision=self.args.method in _ADVANCED_TARGET_METHODS,
                    )
                    model.zero_grad()
                    if self._grouped_target_signal():
                        # A grouped target loss is computed from logits, so even
                        # without windowed execution it must be chunked or the
                        # whole target batch's [B, S, V] lands in memory at once.
                        if val_labels is None:
                            raise RuntimeError(
                                "grouped target signals require labels in the target batch"
                            )
                        val_loss = self._accumulate_target_signal_backward(
                            model, batch_val, val_labels
                        )
                    else:
                        val_loss = self._compute_target_signal_loss(model, batch_val)
                        val_loss.backward()
                    self.grad_hook.end_val_capture(
                        val_total_tokens=val_total_tokens
                    )

                # PASS 2: Train with curation using stored val gradients
                def compute_train_loss():
                    loss = self._compute_loss_for_selection(model, batch_train)
                    return loss, {}  # SeparateBatchStrategy expects (loss, stats) tuple

                if self._windowed_execution_enabled():
                    logical = int(self.args.logical_candidate_batch_size)
                    if train_batch_size != logical:
                        raise RuntimeError(
                            f"logical candidate window requires exactly {logical} "
                            f"candidates, received {train_batch_size}; enable drop_last"
                        )
                    loss, stats = self.selection_strategy.execute_windowed_training_step(
                        model=model,
                        batch_size=train_batch_size,
                        microbatch_size=int(self.args.candidate_microbatch_size),
                        labels=batch_train["labels"],
                        compute_chunk_loss_fn=lambda start, end: (
                            self._compute_loss_for_selection(
                                model,
                                self._slice_model_batch(batch_train, start, end),
                            )
                        ),
                        lr=lr,
                        global_step=self.state.global_step,
                    )
                    stats["selection/target_grad_loss"] = float(
                        val_loss.detach().float().cpu().item()
                    )
                else:
                    loss, stats = self.selection_strategy.execute_training_step(
                        model=model,
                        batch_size=train_batch_size,
                        compute_loss_fn=compute_train_loss,
                        lr=lr,
                        # Pass labels for token-based gradient scaling
                        labels=batch_train.get('labels'),
                        global_step=self.state.global_step,
                        # For GlobalSubset pass 2, provide filter function
                        filter_batch_fn=lambda indices: (
                            lambda: (self._compute_loss_for_selection(model, {
                                'input_ids': batch_train['input_ids'][indices],
                                'attention_mask': batch_train['attention_mask'][indices],
                                'labels': batch_train['labels'][indices],
                            }), {})
                        )
                    )

                # Cleanup val buffer
                self.grad_hook.clear_val_buffer()

            # Record curation data for case study
            if record_this_step:
                self._capture_selection_record(batch_train, batch_val)

            if track_domains:
                self._accumulate_domain_selection(train_batch_size)

            self._maybe_log_selection_metrics(stats)

            return loss

        # === BASELINE MODE (no data curation) ===
        else:
            return self._training_step_baseline(model, inputs)

    def _compute_loss_for_selection(self, model, batch):
        """Compute loss for curation (handles multi-GPU and grad accumulation)."""
        with self.compute_loss_context_manager():
            outputs = model(**batch)
            loss = outputs.loss

        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        return loss

    def _training_step_baseline(self, model, inputs):
        """Baseline training step without curation."""
        args = self.args

        model.zero_grad()

        # Disable hooks if no compression (baseline training)
        if not self.has_compression and self.grad_hook is not None and self.grad_hook.hooks_registered:
            self.grad_hook.disable_hooks()

        inputs = self._prepare_inputs(inputs)
        if self._windowed_execution_enabled():
            labels = inputs.get("labels")
            if labels is None:
                raise RuntimeError("Windowed FullTraining requires labels")
            logical = int(self.args.logical_candidate_batch_size)
            if int(labels.shape[0]) != logical:
                raise RuntimeError(
                    f"Windowed FullTraining expected {logical} candidates, "
                    f"received {labels.shape[0]}"
                )
            tokens = (labels[:, 1:] != -100).sum(dim=1)
            total_tokens = tokens.sum()
            if not bool((total_tokens > 0).detach().cpu()):
                raise RuntimeError("Logical candidate window has no valid tokens")
            loss = torch.zeros((), device=labels.device, dtype=torch.float32)
            microbatch = int(self.args.candidate_microbatch_size)
            for start in range(0, logical, microbatch):
                end = min(start + microbatch, logical)
                chunk = self._slice_model_batch(inputs, start, end)
                with self.compute_loss_context_manager():
                    chunk_loss = self.compute_loss(model, chunk)
                fraction = tokens[start:end].sum().to(chunk_loss.dtype) / total_tokens
                scaled = chunk_loss * fraction
                loss = loss + scaled.detach().float()
                scaled.backward()
        else:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

        # Re-enable hooks if they were disabled
        if not self.has_compression and self.grad_hook is not None and self.grad_hook.hooks_registered:
            self.grad_hook.enable_hooks()

        return loss.detach()

    def set_pool_metadata(self, pool_metadata):
        """Attach the per-row `{id, dataset, domain}` sidecar of the train pool.

        Without it, `meta_idx` cannot be resolved to a domain and per-domain
        selection tracking stays off.
        """
        self._pool_metadata = pool_metadata

    def _should_track_domains(self) -> bool:
        if not self._track_selection_domains or self._pool_metadata is None:
            return False
        if self.args.method == 'NA':
            # Full training selects everything; a selection rate is meaningless.
            return False
        return self.state.global_step % self._track_selection_domains_freq == 0

    def _should_record_selections(self) -> bool:
        return bool(
            self._record_selections
            and self.state.global_step % self._record_selections_freq == 0
        )

    def _domain_of(self, batch_position: int) -> str:
        """Domain of the example at `batch_position` in the current train batch."""
        if not self._last_meta_idx or batch_position >= len(self._last_meta_idx):
            return "unknown"
        row = self._last_meta_idx[batch_position]
        if self._pool_metadata is None or row >= len(self._pool_metadata):
            return "unknown"
        return self._pool_metadata[row].get("domain", "unknown")

    def _source_of(self, batch_position: int) -> str:
        if not self._last_meta_idx or batch_position >= len(self._last_meta_idx):
            return "unknown"
        row = self._last_meta_idx[batch_position]
        if self._pool_metadata is None or row >= len(self._pool_metadata):
            return "unknown"
        metadata = self._pool_metadata[row]
        return metadata.get("source_dataset", metadata.get("dataset", "unknown"))

    def _per_example_selection_weights(self, train_batch_size: int):
        """Mean selection weight in [0, 1] for each example in the current batch.

        Hard methods record `selected_indices` per unit (per layer, per optimizer
        group, or once globally); averaging the 0/1 indicator over units gives the
        fraction of units that kept the example. Soft methods record continuous
        `weights` directly. Returns None when the step produced no usable record.
        """
        records = getattr(self.selection_strategy, 'last_selection_record', None)
        if not records:
            return None

        totals = [0.0] * train_batch_size
        units = 0
        for record in records:
            if not isinstance(record, dict):
                continue
            if 'selected_indices' in record:
                for index in record['selected_indices']:
                    if 0 <= index < train_batch_size:
                        totals[index] += 1.0
                units += 1
            elif 'weights' in record:
                weights = record['weights']
                for index in range(min(train_batch_size, len(weights))):
                    totals[index] += float(weights[index])
                units += 1

        if units == 0:
            return None
        return [value / units for value in totals]

    def _accumulate_domain_selection(self, train_batch_size: int):
        """Fold this step's per-domain candidate/selected mass into the totals."""
        weights = self._per_example_selection_weights(train_batch_size)
        if weights is None:
            return

        step_candidates = {}
        step_selected = {}
        step_source_candidates = {}
        step_source_selected = {}
        for position in range(train_batch_size):
            domain = self._domain_of(position)
            source = self._source_of(position)
            step_candidates[domain] = step_candidates.get(domain, 0.0) + 1.0
            step_selected[domain] = step_selected.get(domain, 0.0) + weights[position]
            step_source_candidates[source] = step_source_candidates.get(source, 0.0) + 1.0
            step_source_selected[source] = step_source_selected.get(source, 0.0) + weights[position]

        for domain, count in step_candidates.items():
            self._domain_candidates[domain] = self._domain_candidates.get(domain, 0.0) + count
        for domain, count in step_selected.items():
            self._domain_selected[domain] = self._domain_selected.get(domain, 0.0) + count
        for source, count in step_source_candidates.items():
            self._source_candidates[source] = self._source_candidates.get(source, 0.0) + count
        for source, count in step_source_selected.items():
            self._source_selected[source] = self._source_selected.get(source, 0.0) + count
        self._domain_steps_tracked += 1

        self._domain_timeline.append({
            'step': self.state.global_step,
            'candidates': step_candidates,
            'selected': {k: round(v, 6) for k, v in step_selected.items()},
            'source_candidates': step_source_candidates,
            'source_selected': {k: round(v, 6) for k, v in step_source_selected.items()},
        })

    def _save_selection_domain_summary(self):
        """Write per-domain selection rates for mixed-domain alignment analysis.

        `selection_rate` is selected mass over candidate mass for that domain.
        `lift` compares it to the run's overall selection rate: > 1 means the
        curation method prefers that domain over a uniform draw.
        """
        if not self._domain_candidates:
            return
        if getattr(self.args, "local_rank", -1) not in (-1, 0):
            return

        total_candidates = sum(self._domain_candidates.values())
        total_selected = sum(self._domain_selected.values())
        overall_rate = (total_selected / total_candidates) if total_candidates else 0.0

        def summarize(candidates_by_key, selected_by_key):
            summary = {}
            for key, candidates in sorted(candidates_by_key.items()):
                selected = selected_by_key.get(key, 0.0)
                rate = (selected / candidates) if candidates else 0.0
                summary[key] = {
                    'candidates': candidates,
                    'selected': round(selected, 6),
                    'candidate_share': round(candidates / total_candidates, 6) if total_candidates else 0.0,
                    'selected_share': round(selected / total_selected, 6) if total_selected else 0.0,
                    'selection_rate': round(rate, 6),
                    'lift': round(rate / overall_rate, 6) if overall_rate else None,
                }
            return summary

        by_domain = summarize(self._domain_candidates, self._domain_selected)
        by_source = summarize(self._source_candidates, self._source_selected)

        payload = {
            'metadata': {
                'method': self.args.method,
                'optimizer_type': getattr(self.args, 'optimizer_type', None),
                'selection_frac': getattr(self.args, 'selection_frac', None),
                'train_batch_size': self.args.per_device_train_batch_size,
                'target_task': getattr(self.args, 'analysis_dataset', None),
                'train_dataset_names': getattr(self.args, 'train_dataset_names', None),
                'steps_tracked': self._domain_steps_tracked,
                'tracking_freq': self._track_selection_domains_freq,
                'overall_selection_rate': round(overall_rate, 6),
            },
            'by_domain': by_domain,
            'by_source_dataset': by_source,
            'timeline': self._domain_timeline,
        }

        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(self.args.output_dir, "selection_domain_summary.json")
        with open(output_file, 'w') as handle:
            json.dump(payload, handle, indent=2)
        logger.info(
            f"Saved per-domain selection summary over {self._domain_steps_tracked} "
            f"tracked steps to {output_file}"
        )

    def _capture_selection_record(
        self,
        batch_train: Dict[str, Tensor],
        batch_val: Dict[str, Tensor],
    ):
        """Capture curation record from the last training step.

        Records decoded text for both training and validation samples,
        along with per-layer (LayerWiseSubset) or global (GlobalSubset) curation data.
        """
        record = getattr(self.selection_strategy, 'last_selection_record', None)
        if record is None:
            return

        # Decode training and validation samples to text
        tokenizer = self.processing_class
        train_texts = tokenizer.batch_decode(batch_train['input_ids'], skip_special_tokens=False)
        val_texts = tokenizer.batch_decode(batch_val['input_ids'], skip_special_tokens=False)

        step_record = {
            'step': self.state.global_step,
            'train_samples': train_texts,
            'val_samples': val_texts,
        }
        if self._last_meta_idx is not None:
            step_record['train_meta_idx'] = [
                int(index) for index in self._last_meta_idx
            ]

        if self.args.method in (
            'LayerWiseSubset',
            'LayerWiseOptimizerAwareSubset',
            'LayerWiseRandomSubset',
            'LayerWiseSoftWeighting',
            'LayerWiseSoftProbability',
            'LayerWiseMuonSpectral',
            'LayerWiseMuonMatrixSpectral',
            'LayerWiseMuonMatrixSpectralP',
            'LayerWiseMuonMatrixSpectralSat',
            'LayerWiseMuonMatrixSpectralSatP',
        ):
            step_record['layers'] = record
        elif self.args.method in ('OptimizerGroupWise', 'OptimizerAwareGroupWise'):
            step_record['groups'] = record
        else:
            # GlobalSubset: single global curation
            step_record['selection'] = record[0] if record else {}

        self._selection_records.append(step_record)

    def _save_selection_records(self):
        """Save accumulated curation records to JSON file."""
        if not self._selection_records:
            return

        output_file = os.path.join(self.args.output_dir, "selection_records.json")
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)

        data = {
            'metadata': {
                'method': self.args.method,
                'selection_frac': self.args.selection_frac,
                'train_batch_size': self.args.per_device_train_batch_size,
                'record_freq': self._record_selections_freq,
                'num_layers': len(self.grad_hook.layer_names) if self.grad_hook else 0,
                'layer_names': self.grad_hook.layer_names if self.grad_hook else [],
            },
            'steps': self._selection_records,
        }

        with open(output_file, 'w') as f:
            json.dump(data, f)

        logger.info(f"Saved {len(self._selection_records)} curation records to {output_file}")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate to compute target- and general-validation perplexity.

        Computes:
        - Target validation: held-out target-domain monitoring data
        - General validation: held-out general-pool monitoring data

        Selection uses ``target_grad_dataset`` through its own dataloader; the
        immutable-profile ``target_val_dataset`` evaluated here is never used to score or
        choose candidates.

        Args:
            eval_dataset: Dataset to evaluate on (defaults to self.eval_dataset)
            ignore_keys: Keys to ignore in the output
            metric_key_prefix: Prefix for metric keys

        Returns:
            Dict of schema-v2 target/general losses, perplexities, and
            step-0 reductions, plus legacy aliases.
        """
        # Disable gradient hooks during evaluation to avoid overhead
        if self.grad_hook is not None:
            self.grad_hook.disable_hooks()

        # Record time before evaluation
        if self.training_start_event is not None:
            eval_start_event = torch.cuda.Event(enable_timing=True)
            eval_start_event.record()
        else:
            eval_start_cpu = time.time()

        # Evaluate on the disjoint target-domain monitoring split.
        target_val_loss, target_val_perplexity = self._evaluate_on_dataset(
            self.target_val_dataset,
            description="Target validation"
        )

        # Evaluate on the held-out general-pool split.
        general_val_loss, general_val_perplexity = self._evaluate_on_dataset(
            self.general_val_dataset,
            description="General validation"
        )

        # Calculate elapsed wall time and subtract cumulative evaluation time
        if self.training_start_event is not None:
            eval_end_event = torch.cuda.Event(enable_timing=True)
            eval_end_event.record()
            torch.cuda.synchronize()
            wall_time = self.training_start_event.elapsed_time(eval_end_event) / 1000.0  # Convert ms to seconds
            eval_duration = eval_start_event.elapsed_time(eval_end_event) / 1000.0
        else:
            eval_end_cpu = time.time()
            wall_time = eval_end_cpu - self.training_start_time
            eval_duration = eval_end_cpu - eval_start_cpu

        self.cumulative_eval_time += eval_duration
        train_wall_time = wall_time - self.cumulative_eval_time

        if self._step0_general_val_loss is None:
            self._step0_general_val_loss = general_val_loss
        if self._step0_target_val_loss is None:
            self._step0_target_val_loss = target_val_loss
        general_reduction = self._step0_general_val_loss - general_val_loss
        target_reduction = self._step0_target_val_loss - target_val_loss
        general_reduction_pct = (
            100.0 * general_reduction / self._step0_general_val_loss
            if self._step0_general_val_loss else 0.0
        )
        target_reduction_pct = (
            100.0 * target_reduction / self._step0_target_val_loss
            if self._step0_target_val_loss else 0.0
        )

        # Schema v2 names the three roles explicitly. The general_loss alias is
        # retained for existing dashboards and plotting notebooks.
        eval_metrics = {
            f"{metric_key_prefix}_loss": general_val_loss,
            f"{metric_key_prefix}_perplexity": general_val_perplexity,
            "eval/schema_version": 2,
            "eval/general_val_loss": general_val_loss,
            "eval/general_val_perplexity": general_val_perplexity,
            "eval/general_val_loss_reduction": general_reduction,
            "eval/general_val_loss_reduction_percent": general_reduction_pct,
            "eval/target_val_loss": target_val_loss,
            "eval/target_val_perplexity": target_val_perplexity,
            "eval/target_val_loss_reduction": target_reduction,
            "eval/target_val_loss_reduction_percent": target_reduction_pct,
            "eval/general_loss": general_val_loss,
            "eval/general_perplexity": general_val_perplexity,
            "eval/target_loss": target_val_loss,
            "eval/target_perplexity": target_val_perplexity,
            "eval/target_general_loss_gap": target_val_loss - general_val_loss,
            "wall_time": wall_time,
            "train_wall_time": train_wall_time,
        }
        self.log(eval_metrics)

        # Save results
        result_entry = {
            "schema_version": 2,
            "step": self.state.global_step,
            "epoch": self.state.epoch,
            "general_val_loss": general_val_loss,
            "general_val_perplexity": general_val_perplexity,
            "general_val_loss_reduction": general_reduction,
            "general_val_loss_reduction_percent": general_reduction_pct,
            "target_val_loss": target_val_loss,
            "target_val_perplexity": target_val_perplexity,
            "target_val_loss_reduction": target_reduction,
            "target_val_loss_reduction_percent": target_reduction_pct,
            # Legacy aliases consumed by the current plotter.
            "val_loss": target_val_loss,
            "val_perplexity": target_val_perplexity,
            "eval_loss": general_val_loss,
            "eval_perplexity": general_val_perplexity,
            "wall_time": wall_time,
            "train_wall_time": train_wall_time,
        }
        self.evaluation_results.append(result_entry)
        self._save_evaluation_results()
        self._save_selection_records()
        self._save_selection_diagnostics()
        self._save_selection_domain_summary()

        logger.info(
            f"Step {self.state.global_step}: "
            f"target_val_perplexity={target_val_perplexity:.4f}, "
            f"general_val_perplexity={general_val_perplexity:.4f}, "
            f"train_wall_time={train_wall_time:.2f}s (eval_time={eval_duration:.2f}s)"
        )

        # Re-enable hooks after evaluation if curation method is active OR compression is used
        # Hooks are needed for both: (1) layer_wise_subset data curation, (2) MeSO compressed gradients
        if self.grad_hook is not None and (self.args.method != "NA" or self.has_compression):
            self.grad_hook.enable_hooks()

        # Reset control.should_evaluate flag (HF Trainer's CallbackHandler.on_evaluate normally
        # does this; without it, target-only runs would double-eval when a step boundary
        # coincides with an epoch boundary — _maybe_log_save_evaluate fires once at on_step_end
        # and again at on_epoch_end with the flag still True).
        if hasattr(self, 'callback_handler') and hasattr(self, 'state') and hasattr(self, 'control'):
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, eval_metrics)

        return eval_metrics

    def _evaluate_on_dataset(self, dataset, description="Evaluation"):
        """
        Evaluate model on a given dataset and compute loss and perplexity.

        Args:
            dataset: Dataset to evaluate on
            description: Description for logging

        Returns:
            tuple: (average_loss, perplexity)
        """
        model = self.model
        model.eval()

        eval_batch_size = (
            int(self.args.candidate_microbatch_size)
            if self._windowed_execution_enabled()
            else int(self.args.per_device_eval_batch_size)
        )
        logger.debug(f"{description}: Dataset size = {len(dataset)}, Batch size = {eval_batch_size}")

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        total_loss = 0.0
        total_valid_tokens = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.args.device) for k, v in batch.items()}

                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss

                # Skip NaN batches (e.g. all labels are -100 after truncation)
                if torch.isnan(loss):
                    logger.debug(f"{description}: Skipping NaN loss batch {num_batches} "
                                 f"(likely all labels masked after truncation)")
                    continue

                # HF causal-LM loss is the mean over non--100 labels. Reweight by
                # that count to obtain one CE over the complete split.
                valid_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
                if valid_tokens == 0:
                    continue
                total_loss += loss.item() * valid_tokens
                total_valid_tokens += valid_tokens
                num_batches += 1

        logger.debug(
            f"{description}: Processed {num_batches} batches, "
            f"{total_valid_tokens} valid assistant tokens"
        )

        # Compute average loss and perplexity
        avg_loss = (
            total_loss / total_valid_tokens
            if total_valid_tokens > 0
            else float("inf")
        )
        perplexity = math.exp(avg_loss) if avg_loss != float("inf") else float("inf")

        logger.debug(f"{description}: avg_loss={avg_loss:.4f}, perplexity={perplexity:.4f}")

        model.train()

        return avg_loss, perplexity

    def _save_evaluation_results(self):
        """Save evaluation results to JSON file."""
        output_file = os.path.join(self.args.output_dir, "evaluation_results.json")

        # Ensure output directory exists
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)

        # Save results
        with open(output_file, "w") as f:
            json.dump(self.evaluation_results, f, indent=2)

    def _save_selection_diagnostics(self):
        """Persist lightweight per-step solver diagnostics for offline reports."""
        if not self.selection_diagnostics_history:
            return
        output_file = os.path.join(self.args.output_dir, "selection_diagnostics.json")
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(self.selection_diagnostics_history, handle, indent=2)

    def on_train_end(self):
        """Called at the end of training to save final results."""
        self._save_evaluation_results()
        self._save_selection_records()
        self._save_selection_diagnostics()
        self._save_selection_domain_summary()
        logger.info(f"Training completed. Final results saved to {self.args.output_dir}/evaluation_results.json")
