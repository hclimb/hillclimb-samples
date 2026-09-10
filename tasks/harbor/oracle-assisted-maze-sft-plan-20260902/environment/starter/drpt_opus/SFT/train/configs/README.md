# Method Configs

## Canonical Dolci-Instruct 32K profiles

The model-independent benchmark settings are under `configs/dolci32k/`:

| Setting config | General pool | Shared target artifacts | Final evaluation |
|---|---|---|---|
| `inst_if` | Instruction-32K | Precise-IF 64 grad / 128 val | IFEval + IFBench |
| `reason_math` | Reasoning-32K | MATH 64 grad / 128 val | MATH500 |
| `reason_code` | Reasoning-32K | MBPP 64 grad / 128 val | MBPP+ |
| `mixed_if` | nested Mixed-32K | same Precise-IF files as `inst_if` | IFEval + IFBench |
| `mixed_math` | nested Mixed-32K | same MATH files as `reason_math` | MATH500 |

Choose an exact pinned base model with `--model-profile olmo3_7b`,
`--model-profile qwen3_4b`, or `--model-profile qwen3_8b`. These overlays alter
only model-specific loading, tokenization, and chat formatting; raw example IDs
and their stored 32K candidate order remain identical. OLMo uses the separate
`allenai/olmo-3-tokenizer-instruct-release` tokenizer overlay.

A formal run is single-rank, one epoch, 2,000 logical steps over one fixed
without-replacement pass (32,000 rows at `n=16`). FullTraining retains all 16,
hard selection retains `k=8`, and the Soft/SoftP methods retain continuous
weights. A smaller `--max-steps` is a prefix smoke only. Build and profile the
immutable data before training. The canonical context limit is 4,096 tokens;
changing it does not change raw pool IDs, but it does require the matching
derived `max_seq_len_4096` tokenization report before runtime preflight:

```bash
python SFT/data/prepare_dolci32k.py --build --profile-tokenizers all

bash SFT/train/train.sh -c configs/dolci32k/inst_if \
  -m FullTraining --model-profile qwen3_4b --max-steps 3
```

See `SFT/README.md` for the Dolci32K metadata inventory, exact quotas,
artifact audit commands, tokenizer/truncation reports, and campaign interface.

Each YAML file defines a training method as `{CurationMethod}-{FinetuningMethod}`.
The SFT 10-baseline launcher uses shorter run labels while keeping these YAML
method names internally:

| Run label | Internal config/method |
|---|---|
| `FullTraining` | `FullTraining-Full` |
| `GlobalRaw` | `GlobalSubset-Full` |
| `LayerwiseRaw` | `LayerWiseSubset-Full` |
| `OptGroupRaw` | `OptimizerGroupWise-Full` |
| `GlobalOptA` | `OptimizerAwareGlobalSubset-OptA-Full` |
| `LayerwiseOptA` | `LayerWiseOptimizerAwareSubset-OptA-Full` |
| `GlobalOptANorm` | `OptimizerAwareGlobalSubset-OptANorm-Full` (reuses the Global OptA config with token-normalized hard selection) |
| `LayerwiseOptANorm` | `LayerWiseOptimizerAwareSubset-OptANorm-Full` (reuses the Layerwise OptA config with token-normalized hard selection) |
| `OptGroupOptA` | `OptimizerAwareGroupWise-OptA-Full` |
| `GlobalOptB` | `OptimizerAwareGlobalSubset-OptB-Full` |
| `LayerwiseOptB` | `LayerWiseOptimizerAwareSubset-OptB-Full` |
| `OptGroupOptB` | `OptimizerAwareGroupWise-OptB-Full` |
| `LayerwiseSoft` | `LayerWiseSoftWeighting-Full` (capped simplex) |
| `LayerwiseSoftP` | `LayerWiseSoftProbability-Full` (pure probability simplex) |

`LayerwiseSoft` and `LayerwiseSoftP` both use the optimized continuous weights
directly when assembling the token-normalized raw gradient; neither method
rounds the weights to a top-k subset. Their feasible sets are
`0 <= w_i <= 1, sum_i w_i = k` and `p_i >= 0, sum_i p_i = 1`, respectively.
The probability-simplex variant deliberately removes the `p_i <= 1/k` cap and
therefore is a concentration ablation, not a cardinality-k budget match.
The optional `soft_weighting.replay_precision` field is `fp32` by default.
`bf16_fp32` keeps scoring, solver weights, normalization, bias reduction, and
the assembled output in FP32 while using BF16 operands only for the large CUDA
Linear replay contraction. Unsupported devices and dtypes fall back to FP32.

## Fields

| Field | Values | Description |
|---|---|---|
| `method` | Standard, LayerWiseSubset, OptimizerAwareGroupWise, GlobalSubset | Data curation method |
| `finetuning` | Full, LoRA, MeSO, MeSO-LoRA | Training approach |
| `lora_r`, `lora_alpha`, `lora_dropout` | int, int, float | LoRA hyperparameters |

## Optimizer-Aware Group-Wise Scoring

`OptimizerAwareGroupWise-*` configs use layer/group-wise curation but score each
candidate in an optimizer-induced geometry before assembling the selected raw
gradient. With `optimizer.type: muon`, eligible two-dimensional hidden weights
are updated by official `torch.optim.Muon`; embeddings, norms, biases, output
heads, and other unsupported parameters are updated by official AdamW. The
runtime facade coordinates those two official optimizers rather than replacing
Muon's update implementation. It uses the bundled/local Muon only if the
installed PyTorch Muon is unavailable or incompatible. LoRA adapters default to
AdamW and can be assigned to Muon with
`optimizer_aware.lora_optimizer: muon`. With `optimizer.type: adamw`, all
parameter groups use AdamW geometry and AdamW updates.

`optimizer.type: hybrid` retains the same official-first runtime assignment but
is reserved for legacy, explicitly named `Hybrid` selection ablations that mix
Muon-managed and AdamW-managed parameter scores. Canonical Muon baselines use
`optimizer.type: muon` and score only the Muon-managed matrices when selecting a
Muon surrogate.

Optimizer-aware scoring supports two target modes:

- `opta`: Optimizer-Induced Feasible Set / target-loss majorization.
  Scores `score_i = <P_t g_i, g_target>`, implemented as
  `<g_i, P*_t g_target>` for reduced-ghost scoring.
- `optb`: Optimizer-Step Matching / target-update trajectory matching.
  Scores `score_i = <P_t g_i, P_t g_target>`, implemented as
  `<g_i, P*_t(P_t g_target)>`.

Both modes use optimizer-aware scoring only for ranking. The final selected
update is assembled from raw gradients, so the selected batch is not
preconditioned twice before the optimizer step.

The explicit `GlobalOptANorm` and `LayerwiseOptANorm` launcher aliases retain
the OptA geometry but solve the hard top-k objective with token normalization.
The legacy `GlobalOptA` and `LayerwiseOptA` aliases keep
`optimizer_aware.token_normalized_selection: false` for reproducibility.

The scoring preconditioners include the optimizer step scale used for ranking
across layers/groups: AdamW uses `alpha_t * Diag((sqrt(vhat_t) + eps)^-1)`;
Muon uses the frozen Newton-Schulz linearized map with
`eta_t * shape_scale * (1 - mu^2)` for Nesterov momentum (or
`eta_t * shape_scale * (1 - mu)` without Nesterov).

```yaml
method: OptimizerAwareGroupWise
scoring:
  method: reduced_ghost
optimizer:
  type: muon               # muon; adamw; hybrid only for legacy mixed-score ablations
optimizer_aware:
  target_mode: opta         # opta or optb
  matrix_geometry: muon      # muon, adamw, identity, auto
  vector_geometry: adamw     # adamw, identity
  muon_reference: momentum_proxy
  muon_momentum: 0.95
  muon_steps: 5
  muon_lr_shape_scale: true
  lora_optimizer: adamw
  spectral_lambda: 0.0       # optional Muon spectral penalty
```

## Gradient Compression

Both `score_grad_compression` and `opt_grad_compression` use the same two-stage pipeline:

```yaml
score_grad_compression:   # compresses gradients for influence score computation
  sparsifier: normal-64*64
  projector: none

opt_grad_compression:     # compresses gradients for MeSO optimizer updates
  sparsifier: normal-512*512
  projector: none
```

**Stage 1 — Sparsifier** (factorized random projection):
Reduces each layer's gradient from full dimension to a low-rank sketch.
Format: `METHOD-DIM*DIM` or `none`.

**Stage 2 — Projector** (non-factorized final projection):
Further compresses the sparsified intermediate representation.
Format: `METHOD-DIM` or `none`.

### Named Compression Schemes

| Name | Sparsifier | Projector | Description |
|---|---|---|---|
| **LoGra** | `normal-D*D` | `none` | Gaussian random projection only (default for MeSO) |
| **GraSS** | `random_mask-D*D` | `sjlt-K` | Sparse mask + sparse JL transform |

Examples:
- LoGra with 512×512: `sparsifier: normal-512*512`, `projector: none`
- GraSS with 1024×1024 + 262144: `sparsifier: random_mask-1024*1024`, `projector: sjlt-262144`

### Design Rules

- **score_grad_compression**: Used for influence score computation in LayerWiseSubset/GlobalSubset curation.
  Set `sparsifier: none` for exact scoring (higher accuracy, more memory).
- **opt_grad_compression**: Used by MeSO optimizer for memory-efficient updates.
  When both sections use the same sparsifier value, compressor objects are shared (zero overhead).
- **MeSO + curation**: If `opt_grad_compression` is set and `score_grad_compression` is not,
  scoring uses full (uncompressed) gradients. To share MeSO compression for scoring,
  set `score_grad_compression.sparsifier` to the same value as `opt_grad_compression.sparsifier`.
- **Identity fallback**: If the compression dimension exceeds the layer's actual feature dimension,
  the compressor automatically falls back to identity (no-op).
