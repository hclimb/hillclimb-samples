"""
Training arguments for SFT experiments.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as TA

from SFT.train.target_signal import (
    CORRECT_INCORRECT_MARGIN,
    canonicalize_target_signal_mode,
)


fsdp_config = {
    "mpt7b_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["MPTBlock"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
    },
    "opt125m_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["OPTDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
    },
    "mpt7b_lora": {
        "fsdp_transformer_layer_cls_to_wrap": ["MPTBlock"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
    "llama_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
    "llama2_7b_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
    "llama2_13b_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
    "mistral_7b_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["MistralDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
}


@dataclass
class TrainingArguments(TA):
    analysis_mode: float = field(
        default=False,
        metadata={
            "help": (
                "Whether to run in analysis mode. "
            )
        },
    )
    analysis_dataset: str = field(
        default="bbh",
        metadata={
            "help": (
                "The dataset to use for analysis mode. "
            )
        },
    )
    train_dataset_names: str = field(
        default=None,
        metadata={
            "help": (
                "The dataset to use for training. "
            )
        },
    )


    # Data Curation Arguments
    method: str = field(
        default="NA",
        metadata={
            "help": (
                "Data curation method: 'NA' (no curation), "
                "'LayerWiseSubset' (per-layer raw curation), "
                "'LayerWiseOptimizerAwareSubset' (per-layer optimizer-aware curation), "
                "'GlobalSubset' (global raw curation), "
                "'OptimizerAwareGlobalSubset' (global optimizer-aware curation), "
                "'OptimizerGroupWise' (optimizer-group raw curation), "
                "'OptimizerAwareGroupWise' (optimizer-group optimizer-aware curation), "
                "'GlobalRandomSubset'/'LayerWiseRandomSubset' (deterministic random baselines), "
                "'GlobalSoftWeighting'/'LayerWiseSoftWeighting' (continuous capped-simplex weights), "
                "'LayerWiseSoftProbability' (uncapped probability-simplex weights), "
                "'GlobalMuonSpectral'/'LayerWiseMuonSpectral' (legacy mixed Hybrid selection surrogate), "
                "or 'GlobalMuonMatrixSpectral'/'LayerWiseMuonMatrixSpectral' "
                "and the layerwise P/Sat/SatP variants (Muon-matrix-only scoring)."
            )
        },
    )
    selection_frac: float = field(
        default=0.5,
        metadata={"help": "Fraction of samples to select (0-1)"},
    )
    selection_mode: str = field(
        default="topk",
        metadata={
            "help": (
                "Selection mode: 'topk' (select top frac samples by score) or "
                "'filtering' (drop bottom frac of negative-score samples)."
            )
        },
    )
    n_val: int = field(
        default=8,
        metadata={"help": "Number of validation samples for data curation"},
    )
    n_eval: int = field(
        default=500,
        metadata={"help": "Number of evaluation samples for generalization testing"},
    )
    n_target_val: int = field(
        default=128,
        metadata={
            "help": (
                "Number of disjoint target-domain examples used only for CE "
                "monitoring. Immutable 32k profiles fix this to 128."
            )
        },
    )
    val_batch_size_for_selection: int = field(
        default=1,
        metadata={
            "help": (
                "Batch size for validation data used during training for data curation. "
                "If None, defaults to per_device_train_batch_size. "
                "This allows independent control of batch size for data curation during training."
            )
        },
    )
    logical_candidate_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Logical number of candidate examples ranked by one selection "
                "decision. When set, it must equal per_device_train_batch_size; "
                "candidate_microbatch_size controls only the GPU compute chunk."
            )
        },
    )
    candidate_microbatch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Internal candidate compute chunk for exact logical-window "
                "selection. A value smaller than per_device_train_batch_size "
                "enables score/finalize/replay execution."
            )
        },
    )
    target_microbatch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Internal compute chunk for one logical target-gradient batch. "
                "The target gradient is normalized over all valid tokens in the "
                "logical target batch, not independently per chunk."
            )
        },
    )
    target_signal_mode: str = field(
        default="nll",
        metadata={
            "help": (
                "Objective whose gradient defines the target signal D* is scored "
                "against. 'nll' is the historical token-mean cross entropy; "
                "'answer_only_ce' keeps that loss but supervises only final-answer "
                "tokens; 'correct_incorrect_margin' contrasts the reference against "
                "a verified-wrong trajectory; 'reward_weighted_sft' weights several "
                "pre-generated trajectories by verified correctness. Every mode but "
                "'nll' reads offline artifacts; none adds an online rollout."
            ),
            "choices": ["nll", "answer_only_ce", "correct_incorrect_margin", "reward_weighted_sft"],
        },
    )
    target_signal_beta: float = field(
        default=1.0,
        metadata={
            "help": (
                "Inverse temperature on the correct/incorrect log-probability "
                "margin. Larger values sharpen the softplus penalty."
            )
        },
    )
    target_signal_margin: float = field(
        default=0.0,
        metadata={
            "help": (
                "Required gap between correct and incorrect mean log probabilities "
                "before the margin loss stops penalizing a pair."
            )
        },
    )
    target_signal_incorrect_reward: float = field(
        default=0.0,
        metadata={
            "help": (
                "Reward given to a verified-incorrect trajectory under "
                "reward_weighted_sft. The default 0.0 drops those rows entirely, "
                "which is rejection-sampling fine-tuning; a small positive value "
                "keeps them at reduced weight."
            )
        },
    )
    target_signal_max_candidates: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Cap on trajectories kept per target prompt under "
                "reward_weighted_sft. The reference is always kept first, so 1 "
                "reproduces plain NLL. Unset keeps every verified trajectory."
            )
        },
    )
    target_signal_align_prompts: bool = field(
        default=False,
        metadata={
            "help": (
                "Restrict every signal to the target prompts the margin objective "
                "can use. The margin loss drops a prompt whose generations were "
                "all correct, so by default its D* is a subset of the one its nll "
                "control sees. Enable this on every arm of a comparison to make "
                "the target prompt set identical and isolate the objective."
            )
        },
    )
    target_signal_groups_per_microbatch: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Prompt groups per target compute chunk. Grouped signals slice on "
                "group boundaries so a correct/incorrect pair is never split across "
                "chunks; target_microbatch_size counts rows and cannot express that."
            )
        },
    )
    target_cache_device: str = field(
        default="cuda",
        metadata={
            "help": (
                "Storage device for captured target gradients during windowed "
                "selection: 'cuda' keeps them resident; 'cpu' stages one layer "
                "back to the accelerator when it is scored."
            )
        },
    )
    target_cache_pin_memory: bool = field(
        default=False,
        metadata={
            "help": (
                "Pin CPU-offloaded target-gradient tensors. This can improve "
                "staging throughput but consumes page-locked host memory."
            )
        },
    )
    track_selection_domains: bool = field(
        default=True,
        metadata={
            "help": (
                "Record per-source-domain candidate/selected counts and write "
                "selection_domain_summary.json. This is the artifact that shows "
                "whether a curation method prefers the target-relevant domain "
                "inside a mixed train pool."
            )
        },
    )
    track_selection_domains_freq: int = field(
        default=10,
        metadata={
            "help": (
                "Sample per-domain selection tracking every N steps. Reading the "
                "per-layer selected indices forces a GPU->CPU sync, so the default "
                "stride keeps the overhead negligible while still giving ~100 "
                "samples over a 1000-step run."
            )
        },
    )
    val_seq_length_multiplier: float = field(
        default=1.2,
        metadata={
            "help": (
                "Reject target/validation examples longer than this multiple of the "
                "average training sequence length. Set to 0 (or any non-positive value) "
                "to disable the heuristic and admit any target up to max_seq_length. "
                "dolci32k disables it because long reasoning targets can be "
                "legitimately much longer than the train average."
            )
        },
    )

    # Gradient Compression Arguments
    sparsification: str = field(
        default=None,
        metadata={
            "help": (
                "Sparsification method and dimension in format 'METHOD-DIM' or 'METHOD-DIM*DIM' for factorized. "
                "Examples: 'Rademacher-512', 'Gaussian-256*256'. Set to None to disable sparsification."
            )
        },
    )
    projection: str = field(
        default=None,
        metadata={
            "help": (
                "Projection method and dimension in format 'METHOD-DIM' or 'METHOD-DIM*DIM' for factorized. "
                "Examples: 'Gaussian-256', 'Rademacher-128*128'. Set to None to use identity (no projection)."
            )
        },
    )
    update_compressor_freq: int = field(
        default=200,
        metadata={
            "help": (
                "Number of steps between projector refreshes. "
                "Set to a large value (e.g., 1000000) to effectively disable refresh. Default: 200"
            )
        },
    )
    score_compression: str = field(
        default=None,
        metadata={
            "help": (
                "Score-only compression for influence score computation. "
                "Same format as sparsification: 'METHOD-DIM*DIM'. "
                "Examples: 'normal-64*64' (Gaussian, 64x64 factorized). "
                "Set to None to disable (use exact scoring). Default: None"
            )
        },
    )
    scoring_method: str = field(
        default="reduced_ghost",
        metadata={
            "help": (
                "Scoring method for influence score computation: "
                "'reduced_ghost' (default, our ghost inner product, never materializes per-sample grads), "
                "'full_ghost' (GREATS-style ghost IP, materializes for 3D but true ghost for 2D), "
                "'direct' (explicit per-sample gradient materialization), "
                "'compress' (compressed per-sample gradients, requires score_compression to be set)."
            )
        },
    )
    subset_mode: str = field(
        default="one_pass",
        metadata={
            "help": (
                "GlobalSubset descent mode (only used when method='GlobalSubset'): "
                "'one_pass' (default, Algorithm 4.2): Single forward+backward pass, "
                "retains activations for post-hoc gradient assembly. "
                "'two_pass' (Algorithm 4.3): First pass for scoring, second pass on selected subset."
            )
        },
    )
    use_second_order: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use second-order interactions for data curation. "
                "If True, uses greedy curation considering sample similarities (O(k*n) complexity). "
                "If False (default), uses simple top-k curation based on scores."
            )
        },
    )
    val_strategy: str = field(
        default="separate_batch_factorized",
        metadata={
            "help": (
                "Validation gradient strategy for data curation: "
                "'separate_batch_factorized' (default): Separate val pass, store [V,S,O] and [V,S,I] factors. "
                "'separate_batch': Separate val pass, store mean gradient [O,I] per layer. "
                "'merged_batch': Merge train+val into single batch, compute val grad in same pass. "
                "All modes should produce identical gradients when selection_frac=1.0."
            )
        },
    )

    # Optimizer-Aware Group-Wise Data Regularization
    optimizer_aware_matrix_geometry: str = field(
        default="muon",
        metadata={
            "help": (
                "Matrix parameter scoring geometry for optimizer-aware curation methods: "
                "'muon', 'adamw', 'identity', or 'auto'. Default: muon."
            )
        },
    )
    optimizer_type: str = field(
        default="adamw",
        metadata={
            "help": (
                "Actual optimizer update type: 'adamw' uses the configured "
                "AdamW optimizer for all trainable parameters; 'muon' and "
                "'hybrid' both use official torch.optim.Muon for eligible 2D "
                "hidden weights and official AdamW for the remaining parameters, "
                "with a bundled/local Muon fallback only when the official PyTorch "
                "Muon is unavailable or incompatible. The 'muon' label denotes "
                "matrix-only Muon surrogate scoring, while 'hybrid' preserves "
                "mixed Muon+AdamW scoring."
            )
        },
    )
    muon_learning_rate: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Base learning rate for Muon-managed matrix parameters. "
                "Defaults to the standard learning_rate when omitted."
            )
        },
    )
    aux_adamw_learning_rate: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Learning rate for embedding, output, bias, norm, and other "
                "parameters managed by the auxiliary AdamW optimizer. "
                "Defaults to the standard learning_rate when omitted."
            )
        },
    )
    optimizer_aware_vector_geometry: str = field(
        default="adamw",
        metadata={
            "help": (
                "Non-matrix/embedding scoring geometry for optimizer-aware curation methods: "
                "'adamw' or 'identity'. Default: adamw."
            )
        },
    )
    optimizer_aware_target_mode: str = field(
        default="opta",
        metadata={
            "help": (
                "Optimizer-aware target geometry: 'opta' scores <P_t g_i, g_target> "
                "for optimizer-induced feasible-set/target-loss majorization; "
                "'optb' scores <P_t g_i, P_t g_target> for optimizer-step matching."
            )
        },
    )
    optimizer_aware_adam_eps: float = field(
        default=1e-8,
        metadata={"help": "Fallback epsilon for AdamW diagonal scoring geometry."},
    )
    optimizer_aware_muon_steps: int = field(
        default=5,
        metadata={"help": "Newton-Schulz iterations for Muon-style matrix scoring."},
    )
    optimizer_aware_muon_reference: str = field(
        default="momentum_proxy",
        metadata={
            "help": (
                "Frozen Muon reference for OPUS-style scoring: "
                "'momentum_proxy', 'momentum', or 'proxy'."
            )
        },
    )
    optimizer_aware_muon_momentum: float = field(
        default=0.95,
        metadata={"help": "Muon momentum coefficient for the Muon runtime and scoring reference."},
    )
    optimizer_aware_muon_nesterov: bool = field(
        default=True,
        metadata={
            "help": (
                "Use PyTorch Muon Nesterov momentum. The optimizer-aware dual "
                "probe uses the same frozen momentum rule."
            )
        },
    )
    optimizer_aware_muon_eps: float = field(
        default=1e-7,
        metadata={"help": "Normalization epsilon for Muon-style matrix scoring."},
    )
    optimizer_aware_muon_max_dim: int = field(
        default=256,
        metadata={
            "help": (
                "Legacy maximum min(out_features, in_features) for candidate-side "
                "Muon scoring. The default reduced-ghost adjoint path does not "
                "materialize candidate matrices and does not use this cap."
            )
        },
    )
    optimizer_aware_muon_lr_shape_scale: bool = field(
        default=True,
        metadata={
            "help": (
                "Apply Muon/OPUS matrix-shape learning-rate scaling in the Muon "
                "runtime and the optimizer-aware dual probe."
            )
        },
    )
    optimizer_aware_muon_adjust_lr_fn: str = field(
        default="original",
        metadata={
            "help": (
                "PyTorch Muon LR adjustment rule for matrix shape scaling: "
                "'original', 'match_rms_adamw', or 'none'."
            )
        },
    )
    optimizer_aware_muon_backend: str = field(
        default="auto",
        metadata={
            "help": (
                "Muon backend for optimizer_type='muon' or 'hybrid': 'auto' uses "
                "torch.optim.Muon when available and compatible, with local fallback otherwise; "
                "'torch' requests torch.optim.Muon; legacy 'local' is deprecated "
                "and also tries official Muon first before fallback."
            )
        },
    )
    optimizer_aware_lora_optimizer: str = field(
        default="adamw",
        metadata={
            "help": (
                "Optimizer assignment for LoRA adapter matrices under optimizer_type='muon' or 'hybrid': "
                "'adamw' (default) or 'muon'."
            )
        },
    )
    optimizer_aware_spectral_lambda: float = field(
        default=0.0,
        metadata={
            "help": (
                "Optional spectral concentration penalty weight for Muon-style "
                "per-sample matrix scores. Default: 0.0."
            )
        },
    )
    optimizer_aware_spectral_eps: float = field(
        default=1e-12,
        metadata={"help": "Epsilon for the spectral concentration penalty."},
    )
    optimizer_aware_token_normalized_selection: bool = field(
        default=False,
        metadata={
            "help": (
                "For hard OptA top-k selection, exactly maximize the selected-score "
                "sum divided by selected valid-token count. Disabled by default to "
                "preserve the legacy OPUS/OptA baseline."
            )
        },
    )

    # Continuous soft-weighting solver
    soft_weighting_steps: int = field(
        default=20,
        metadata={"help": "Maximum projected-Adam ascent steps for soft weighting."},
    )
    soft_weighting_lr: float = field(
        default=0.1,
        metadata={"help": "Projected-Adam learning rate for soft weighting."},
    )
    soft_weighting_tol: float = field(
        default=1e-5,
        metadata={"help": "Relative objective-improvement tolerance for soft weighting."},
    )
    soft_weighting_patience: int = field(
        default=3,
        metadata={"help": "Consecutive below-tolerance steps before soft weighting stops."},
    )
    soft_weighting_gamma: float = field(
        default=0.0,
        metadata={"help": "Soft-weighting update-size penalty; default 0 disables it."},
    )
    soft_weighting_use_optimizer_state: bool = field(
        default=True,
        metadata={"help": "Use live optimizer state in the soft-weighting candidate map."},
    )
    soft_weighting_constraint: str = field(
        default="capped_simplex",
        metadata={
            "help": (
                "Soft-weight feasible set: 'capped_simplex' uses 0<=w_i<=1 "
                "and sum_i w_i=k; 'probability_simplex' uses w_i>=0 and "
                "sum_i w_i=1 without the 1/k probability cap."
            )
        },
    )
    soft_replay_precision: str = field(
        default="fp32",
        metadata={
            "help": (
                "Soft Linear contraction precision: 'fp32' preserves the exact "
                "Muon solver and replay contractions; 'bf16_fp32' uses bf16 "
                "CUDA factor operands for the large Muon-solver and replay GEMMs "
                "with fp32 weights, target, bias, normalization, reductions, and output."
            )
        },
    )

    # Muon singular-mode support surrogate (modular, hence submodular)
    muon_surrogate_alpha: float = field(
        default=1.0,
        metadata={"help": "Uniform weight for each retained target singular mode."},
    )
    muon_surrogate_rank: int = field(
        default=32,
        metadata={"help": "Maximum randomized-SVD rank for large target matrices."},
    )
    muon_surrogate_full_svd_max_dim: int = field(
        default=256,
        metadata={"help": "Use full SVD when min(matrix shape) is at most this value."},
    )
    muon_surrogate_rtol: float = field(
        default=1e-6,
        metadata={"help": "Drop target singular modes at or below rtol times the largest value."},
    )
    muon_surrogate_oversample: int = field(
        default=8,
        metadata={"help": "Oversampling dimension for deterministic randomized SVD."},
    )
    muon_surrogate_power_iters: int = field(
        default=2,
        metadata={"help": "Power iterations for deterministic randomized SVD."},
    )
    muon_surrogate_include_adamw_scores: bool = field(
        default=True,
        metadata={
            "help": (
                "Include AdamW-managed parameter scores in the legacy Hybrid Muon "
                "selection criterion. MuonMatrixSpectral methods set this to false."
            )
        },
    )
    muon_surrogate_mode_weighting: str = field(
        default="uniform",
        metadata={
            "help": (
                "Target-mode weighting for the Muon surrogate: 'uniform' uses "
                "alpha_r=1; 'singular_value' uses alpha_r=beta_r."
            )
        },
    )
    muon_surrogate_saturation: bool = field(
        default=False,
        metadata={
            "help": (
                "Use the concave coverage objective sum_r alpha_r*log1p("
                "sum_{i in S} a_{i,r}) and deterministic greedy selection."
            )
        },
    )

    # Curation Recording (Case Study)
    record_selections: bool = field(
        default=False,
        metadata={
            "help": (
                "Record selected sample indices and scores per step for case study analysis. "
                "For LayerWiseSubset: records per-layer curation. For GlobalSubset: records global curation. "
                "Saves to output_dir/selection_records.json."
            )
        },
    )
    record_selections_freq: int = field(
        default=1,
        metadata={
            "help": (
                "Record curation decisions every N steps. Default: 1 (every step). "
                "Increase to reduce file size for long training runs."
            )
        },
    )
    optimizer_aware_diagnostic_interval: int = field(
        default=0,
        metadata={
            "help": (
                "Compute optional same-state Raw-vs-OptA diagnostics every N "
                "optimizer steps. 0 disables the extra Raw score path. "
                "Selection records are configured independently."
            )
        },
    )

    # Experiment Tracking
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases project name when report_to includes 'wandb'."},
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases run name. Defaults to TrainingArguments.run_name."},
    )
    wandb_group: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases group name for comparing baseline sweeps."},
    )
    wandb_tags: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated Weights & Biases tags."},
    )

    # Profiling
    profile: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable PyTorch profiler. Profiles first 10 steps and saves trace to output_dir/profile/. "
                "View with: tensorboard --logdir=output_dir/profile/ or chrome://tracing"
            )
        },
    )
    profile_steps: int = field(
        default=10,
        metadata={"help": "Number of steps to profile (default: 10)"},
    )

    def __post_init__(self):
        if isinstance(self.fsdp_config, str):
            self.fsdp_config = fsdp_config[self.fsdp_config]
        if self.train_dataset_names is not None:
            self.train_dataset_names = self.train_dataset_names.split(" ")
        self.optimizer_type = self.optimizer_type.lower()
        if self.optimizer_type in ("adamw_only", "adamw-only"):
            self.optimizer_type = "adamw"
        if self.optimizer_type not in ("adamw", "muon", "hybrid"):
            raise ValueError("optimizer_type must be one of: 'adamw', 'muon', 'hybrid'")
        disabled_compression_values = (None, "", "none", "None")
        if self.optimizer_type in ("muon", "hybrid") and (
            self.sparsification not in disabled_compression_values
            or self.projection not in disabled_compression_values
        ):
            raise ValueError(
                f"optimizer_type={self.optimizer_type!r} cannot be combined with "
                "MeSO/update compression; Muon-labeled runs must use the "
                "official-first Muon runtime."
            )
        for name in ("muon_learning_rate", "aux_adamw_learning_rate"):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(float(value)) or value < 0
            ):
                raise ValueError(
                    f"{name} must be finite and non-negative when specified"
                )
        for name in (
            "logical_candidate_batch_size",
            "candidate_microbatch_size",
            "target_microbatch_size",
            "target_signal_max_candidates",
            "target_signal_groups_per_microbatch",
        ):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive when specified")
        self.target_signal_mode = canonicalize_target_signal_mode(self.target_signal_mode)
        if not math.isfinite(float(self.target_signal_beta)) or self.target_signal_beta <= 0:
            raise ValueError("target_signal_beta must be finite and positive")
        if not math.isfinite(float(self.target_signal_margin)):
            raise ValueError("target_signal_margin must be finite")
        if (
            not math.isfinite(float(self.target_signal_incorrect_reward))
            or self.target_signal_incorrect_reward < 0
        ):
            raise ValueError("target_signal_incorrect_reward must be finite and non-negative")
        if (
            self.target_signal_mode == CORRECT_INCORRECT_MARGIN
            and self.target_signal_incorrect_reward
        ):
            raise ValueError(
                "target_signal_incorrect_reward applies to reward_weighted_sft only; "
                "the margin objective weights its pair by construction"
            )
        self.optimizer_aware_diagnostic_interval = int(
            self.optimizer_aware_diagnostic_interval
        )
        if self.optimizer_aware_diagnostic_interval < 0:
            raise ValueError(
                "optimizer_aware_diagnostic_interval must be non-negative"
            )
        if (
            self.logical_candidate_batch_size is not None
            and int(self.logical_candidate_batch_size)
            != int(self.per_device_train_batch_size)
        ):
            raise ValueError(
                "logical_candidate_batch_size must equal "
                "per_device_train_batch_size; use candidate_microbatch_size to "
                "reduce GPU memory"
            )
        if (
            self.candidate_microbatch_size is not None
            and int(self.candidate_microbatch_size)
            > int(self.per_device_train_batch_size)
        ):
            raise ValueError(
                "candidate_microbatch_size cannot exceed the logical candidate "
                "batch (per_device_train_batch_size)"
            )
        windowed_execution = (
            self.candidate_microbatch_size is not None
            and int(self.candidate_microbatch_size)
            < int(self.per_device_train_batch_size)
        )
        if windowed_execution and int(self.gradient_accumulation_steps) != 1:
            raise ValueError(
                "Exact candidate-window execution requires "
                "gradient_accumulation_steps=1. GPU chunking is performed inside "
                "one logical Trainer step; HF gradient accumulation would change "
                "selection and optimizer-step semantics."
            )
        self.target_cache_device = str(self.target_cache_device).lower()
        if self.target_cache_device in ("gpu", "device"):
            self.target_cache_device = "cuda"
        if self.target_cache_device not in ("cuda", "cpu"):
            raise ValueError("target_cache_device must be 'cuda' or 'cpu'")
        if self.candidate_microbatch_size is not None and self.gradient_checkpointing:
            checkpoint_kwargs = dict(self.gradient_checkpointing_kwargs or {})
            checkpoint_kwargs.setdefault("use_reentrant", False)
            self.gradient_checkpointing_kwargs = checkpoint_kwargs
        self.optimizer_aware_target_mode = self.optimizer_aware_target_mode.lower().replace("-", "_")
        target_mode_aliases = {
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
        self.optimizer_aware_target_mode = target_mode_aliases.get(
            self.optimizer_aware_target_mode,
            self.optimizer_aware_target_mode,
        )
        if self.optimizer_aware_target_mode not in ("opta", "optb"):
            raise ValueError("optimizer_aware_target_mode must be either 'opta' or 'optb'")
        self.optimizer_aware_lora_optimizer = self.optimizer_aware_lora_optimizer.lower()
        if self.optimizer_aware_lora_optimizer not in ("adamw", "muon"):
            raise ValueError("optimizer_aware_lora_optimizer must be either 'adamw' or 'muon'")
        self.optimizer_aware_muon_backend = self.optimizer_aware_muon_backend.lower()
        if self.optimizer_aware_muon_backend in ("pytorch",):
            self.optimizer_aware_muon_backend = "torch"
        if self.optimizer_aware_muon_backend not in ("auto", "torch", "local"):
            raise ValueError("optimizer_aware_muon_backend must be one of: auto, torch, local")
        self.optimizer_aware_muon_adjust_lr_fn = self.optimizer_aware_muon_adjust_lr_fn.lower()
        if self.optimizer_aware_muon_adjust_lr_fn in ("", "false", "off", "disabled"):
            self.optimizer_aware_muon_adjust_lr_fn = "none"
        if self.optimizer_aware_muon_adjust_lr_fn not in ("original", "match_rms_adamw", "none"):
            raise ValueError(
                "optimizer_aware_muon_adjust_lr_fn must be one of: original, match_rms_adamw, none"
            )

        new_solver_methods = {
            "GlobalRandomSubset",
            "LayerWiseRandomSubset",
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
        }
        if self.method in new_solver_methods:
            if self.scoring_method != "reduced_ghost":
                raise ValueError(
                    f"{self.method} v1 requires scoring_method='reduced_ghost'"
                )
            if self.selection_mode != "topk":
                raise ValueError(f"{self.method} v1 requires selection_mode='topk'")
            if self.use_second_order:
                raise ValueError(f"{self.method} does not support use_second_order")
            if self.subset_mode != "one_pass":
                raise ValueError(f"{self.method} v1 requires subset_mode='one_pass'")
            disabled_values = (None, "", "none", "None")
            if self.score_compression not in disabled_values:
                raise ValueError(f"{self.method} does not support score compression")
            if self.sparsification not in disabled_values or self.projection not in disabled_values:
                raise ValueError(f"{self.method} does not support MeSO/update compression")
            if not 0.0 < self.selection_frac <= 1.0:
                raise ValueError("selection_frac must be in (0, 1] for the new solvers")

        self.soft_weighting_constraint = (
            self.soft_weighting_constraint.lower().replace("-", "_")
        )
        if self.soft_weighting_constraint not in (
            "capped_simplex", "probability_simplex"
        ):
            raise ValueError(
                "soft_weighting_constraint must be 'capped_simplex' or "
                "'probability_simplex'"
            )
        self.soft_replay_precision = (
            self.soft_replay_precision.strip().lower().replace("-", "_")
        )
        if self.soft_replay_precision not in ("fp32", "bf16_fp32"):
            raise ValueError(
                "soft_replay_precision must be 'fp32' or 'bf16_fp32'"
            )
        if self.method == "LayerWiseSoftProbability":
            if self.soft_weighting_constraint != "probability_simplex":
                raise ValueError(
                    "LayerWiseSoftProbability requires "
                    "soft_weighting_constraint='probability_simplex'"
                )
            if self.optimizer_type not in ("adamw", "muon"):
                raise ValueError(
                    "LayerWiseSoftProbability supports optimizer_type='adamw' "
                    "or 'muon' only"
                )
        elif self.method in {"GlobalSoftWeighting", "LayerWiseSoftWeighting"}:
            if self.soft_weighting_constraint != "capped_simplex":
                raise ValueError(
                    f"{self.method} requires "
                    "soft_weighting_constraint='capped_simplex'"
                )
        elif self.soft_weighting_constraint != "capped_simplex":
            raise ValueError(
                "soft_weighting_constraint='probability_simplex' is valid only "
                "for LayerWiseSoftProbability"
            )

        muon_spectral_methods = {
            "GlobalMuonSpectral",
            "LayerWiseMuonSpectral",
            "GlobalMuonMatrixSpectral",
            "LayerWiseMuonMatrixSpectral",
            "LayerWiseMuonMatrixSpectralP",
            "LayerWiseMuonMatrixSpectralSat",
            "LayerWiseMuonMatrixSpectralSatP",
        }
        if self.method in muon_spectral_methods:
            if self.optimizer_type not in ("muon", "hybrid"):
                raise ValueError(
                    f"{self.method} requires optimizer_type='muon' or 'hybrid' "
                    "so Muon parameters are present"
                )
            if self.optimizer_type == "muon":
                # ``muon`` is a first-class experiment label for the
                # Muon-managed-matrix scorer.  ``hybrid`` retains the legacy
                # mixed Muon+AdamW scorer semantics.
                self.muon_surrogate_include_adamw_scores = False
            if self.optimizer_aware_spectral_lambda != 0.0:
                raise ValueError(
                    f"{self.method} does not use legacy optimizer_aware_spectral_lambda"
                )
        if self.method in {
            "GlobalMuonMatrixSpectral",
            "LayerWiseMuonMatrixSpectral",
            "LayerWiseMuonMatrixSpectralP",
            "LayerWiseMuonMatrixSpectralSat",
            "LayerWiseMuonMatrixSpectralSatP",
        } and self.muon_surrogate_include_adamw_scores:
            raise ValueError(
                f"{self.method} requires muon_surrogate_include_adamw_scores=False"
            )
        self.muon_surrogate_mode_weighting = (
            self.muon_surrogate_mode_weighting.lower().replace("-", "_")
        )
        if self.muon_surrogate_mode_weighting not in ("uniform", "singular_value"):
            raise ValueError(
                "muon_surrogate_mode_weighting must be 'uniform' or 'singular_value'"
            )
        expected_muon_variant = {
            "LayerWiseMuonMatrixSpectralP": ("singular_value", False),
            "LayerWiseMuonMatrixSpectralSat": ("uniform", True),
            "LayerWiseMuonMatrixSpectralSatP": ("singular_value", True),
        }.get(self.method)
        if expected_muon_variant is not None:
            actual = (
                self.muon_surrogate_mode_weighting,
                self.muon_surrogate_saturation,
            )
            if actual != expected_muon_variant:
                raise ValueError(
                    f"{self.method} requires mode_weighting="
                    f"{expected_muon_variant[0]!r} and saturation="
                    f"{expected_muon_variant[1]}"
                )
        if self.muon_surrogate_saturation and self.method in {
            "GlobalMuonSpectral", "GlobalMuonMatrixSpectral"
        }:
            raise ValueError(
                "Muon surrogate saturation is currently implemented only for "
                "layerwise selection"
            )

        if self.optimizer_aware_token_normalized_selection:
            if self.method not in {
                "OptimizerAwareGlobalSubset", "LayerWiseOptimizerAwareSubset"
            }:
                raise ValueError(
                    "optimizer_aware_token_normalized_selection is supported only "
                    "for hard global/layerwise optimizer-aware subset methods"
                )
            if self.optimizer_aware_target_mode != "opta":
                raise ValueError(
                    "token-normalized hard selection currently supports OptA only"
                )
            if self.selection_mode != "topk" or self.use_second_order:
                raise ValueError(
                    "token-normalized hard OptA requires topk without second order"
                )

        if self.soft_weighting_steps <= 0:
            raise ValueError("soft_weighting_steps must be positive")
        if self.soft_weighting_lr <= 0:
            raise ValueError("soft_weighting_lr must be positive")
        if self.soft_weighting_tol < 0:
            raise ValueError("soft_weighting_tol must be non-negative")
        if self.soft_weighting_patience <= 0:
            raise ValueError("soft_weighting_patience must be positive")
        if self.soft_weighting_gamma < 0:
            raise ValueError("soft_weighting_gamma must be non-negative")
        if self.muon_surrogate_alpha < 0:
            raise ValueError("muon_surrogate_alpha must be non-negative")
        if self.muon_surrogate_rank <= 0:
            raise ValueError("muon_surrogate_rank must be positive")
        if self.muon_surrogate_full_svd_max_dim <= 0:
            raise ValueError("muon_surrogate_full_svd_max_dim must be positive")
        if self.muon_surrogate_rtol < 0:
            raise ValueError("muon_surrogate_rtol must be non-negative")
        if self.muon_surrogate_oversample < 0:
            raise ValueError("muon_surrogate_oversample must be non-negative")
        if self.muon_surrogate_power_iters < 0:
            raise ValueError("muon_surrogate_power_iters must be non-negative")
        super().__post_init__()
