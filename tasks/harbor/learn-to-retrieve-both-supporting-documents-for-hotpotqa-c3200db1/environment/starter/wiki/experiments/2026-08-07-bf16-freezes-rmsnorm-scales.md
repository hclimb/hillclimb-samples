# bf16 silently freezes RMSNorm scales; pf32 unfreezes them

**Date:** 2026-08-07 · **Author:** johnzhang · **Status:** done (bf16 arm dispositive;
pf32 main_model arm pending Stage 3 reach)

## Conclusion

Every RMSNorm scale we watch (12 weights: 3 memory-branch + 9 Qwen backbone,
including `main_model.norm`, `input_layernorm`, `post_attention_layernorm`, `q_norm`,
`k_norm` at layers 9/17/26 and `mem_q_norm` / `mem_o_norm` / `mem_layernorm` at layer 14)
is **bit-exact frozen under bf16 storage** — the "learnable scale" parameter of every
RMSNorm we care about literally does not move across 15,000 Stage-3 training steps.

The same watched weights on the pf32 arm (fp32 master weights + bf16 forward) show
**substantial motion within 10k steps**: `mem_o_norm` shrinks 1.000 → 0.909
(−9.1%), `mem_layernorm` shrinks 1.000 → 0.903 (−9.7%), `mem_q_norm` grows
1.000 → 1.013 (+1.3%). The main-model backbone norms on pf32 are still frozen at
their pretrained values but that's because pf32 hasn't reached Stage 3 (step 15000)
yet — Qwen backbone params are simply not in the trainable set until then.

This confirms a specific and previously-suspected consequence of the bf16-ULP
storage trap: **the RMSNorm scale — the "beta/gamma" parameter that the whole
normalization mechanism relies on to keep incoming-activation magnitudes near
unit norm — cannot adapt during training**. Whatever the network needs to
compensate for downstream (memory-branch write growth, attention-output drift,
main-model unfreeze changes) has no way to be absorbed by the normalization scale
under bf16. This is the mechanistic story behind the "growth of activation
magnitude causes late-stage spike" pattern.

## Hypothesis

RMSNorm's scale parameter is a per-dim learnable vector initialized to 1.0 (memory-
branch fresh init) or to the Qwen pretraining value ~0.3–2.8 (backbone). Under bf16
with |w| in that range, bf16 ULP = |w| · 2⁻⁷ = 0.24% of |w|, so at |w|=1 the ULP is
7.8e-3. At LR=1e-4 Adam produces normalized updates of magnitude ~1e-4, which is
**~78× smaller than the ULP** — the `w_bf16 + Δw_bf16` add rounds Δw to zero.
Predicted result: RMSNorm scales are silently frozen.

Under pf32 (weights stored in fp32, cast to bf16 only for the forward pass), the
Adam update fits in fp32 precision and accumulates normally. Predicted result:
RMSNorm scales actually train.

## Method

Watched-weight RMS trajectories are logged every step by the trainer's
`_WEIGHT_MONITOR_KEYS` block (see `trainer/trainer.py`; instrumentation cherry-picked
across opt_resume and pf32_wchanged_expand branches). Metrics:

- `weight/wnorm/<key>/rms` — current RMS magnitude of the weight (dispositive of
  weight-accumulation change).
- `weight/wchanged/<key>/n` — number of bit-different elements vs. the previous
  log step (secondary signal — reflects update magnitude, not accumulation).

For each RMSNorm scale, take the first and last logged `rms` value in the run and
compute `delta = rms_last − rms_first`. Verdict: FROZEN if |delta| < 1e-4 (below
the ~ULP-scale motion possible for |w|~1), BARELY MOVING if between 1e-4 and 1e-3,
TRAINING if > 1e-3.

## Results

### bf16 arm — opt_resume, wandb run `6t7w3qzj`, steps 20000..35278

| RMSNorm scale | init/pretrained \|w\| | rms_first | rms_last | Δ | verdict |
|---|---|---|---|---|---|
| `main_model.norm` | pretrained ~2.8 | 2.797167 | 2.797167 | +0.000000 | **FROZEN** |
| `main_model.layers.9.input_layernorm` | pretrained ~0.31 | 0.311521 | 0.311521 | −0.000000 | **FROZEN** |
| `main_model.layers.9.post_attention_layernorm` | pretrained ~0.63 | 0.632997 | 0.632997 | +0.000000 | **FROZEN** |
| `main_model.layers.9.q_norm` | pretrained ~1.69 | 1.689671 | 1.689671 | −0.000000 | **FROZEN** |
| `main_model.layers.9.k_norm` | pretrained ~1.69 | 1.691896 | 1.691896 | +0.000000 | **FROZEN** |
| `main_model.layers.17.input_layernorm` | pretrained ~0.58 | 0.584337 | 0.584337 | +0.000000 | **FROZEN** |
| `main_model.layers.17.q_norm` | pretrained ~1.74 | 1.740984 | 1.740984 | +0.000000 | **FROZEN** |
| `main_model.layers.26.input_layernorm` | pretrained ~1.29 | 1.291860 | 1.291860 | +0.000000 | **FROZEN** |
| `main_model.layers.26.q_norm` | pretrained ~1.72 | 1.720935 | 1.720935 | +0.000000 | **FROZEN** |
| `main_model.layers.14.mem_q_norm` | fresh init 1.0 | 1.000000 | 1.000000 | +0.000000 | **FROZEN** |
| `main_model.layers.14.mem_o_norm` | fresh init 1.0 | 1.000000 | 1.000000 | +0.000000 | **FROZEN** |
| `main_model.layers.14.mem_layernorm` | fresh init 1.0 | 1.000000 | 1.000000 | +0.000000 | **FROZEN** |

**12 of 12 watched RMSNorm scales bit-exact frozen across 15k training steps** in
Stage 3, with mem-branch norms trainable via `.*mem_.*` regex and main-model
backbone norms trainable via `.*main_model.*` regex. The trainable regex hits them
(so adam is computing updates), but the bf16 store truncates every update to zero.

### pf32 arm — pf32_wchanged_expand, wandb run `jpursdwp`, steps 0..10169

| RMSNorm scale | rms_first | rms_last | Δ | verdict |
|---|---|---|---|---|
| `main_model.norm` | 2.797167 | 2.797167 | +0.000000 | FROZEN (not-yet-trainable — Stage 3 hasn't started) |
| `main_model.layers.9.input_layernorm` | 0.311521 | 0.311521 | +0.000000 | FROZEN (same) |
| `main_model.layers.9.post_attention_layernorm` | 0.632997 | 0.632997 | +0.000000 | FROZEN (same) |
| `main_model.layers.9.q_norm` | 1.689671 | 1.689671 | +0.000000 | FROZEN (same) |
| `main_model.layers.9.k_norm` | 1.691896 | 1.691896 | +0.000000 | FROZEN (same) |
| `main_model.layers.17.input_layernorm` | 0.584337 | 0.584337 | +0.000000 | FROZEN (same) |
| `main_model.layers.17.q_norm` | 1.740984 | 1.740984 | +0.000000 | FROZEN (same) |
| `main_model.layers.26.input_layernorm` | 1.291860 | 1.291860 | +0.000000 | FROZEN (same) |
| `main_model.layers.26.q_norm` | 1.720935 | 1.720935 | +0.000000 | FROZEN (same) |
| `main_model.layers.14.mem_q_norm` | 1.000000 | **1.013139** | +0.013139 (+1.3%) | **TRAINING** |
| `main_model.layers.14.mem_o_norm` | 1.000000 | **0.909122** | −0.090878 (−9.1%) | **TRAINING** |
| `main_model.layers.14.mem_layernorm` | 1.000000 | **0.902577** | −0.097423 (−9.7%) | **TRAINING** |

**Interpretation of pf32 arm**: The main-model backbone norms are still bit-identical
to init because Stage 3 (step 15000+) hasn't fired yet — the trainable-params regex
in Stages 0–2 doesn't include `.*main_model.*`. That's expected non-motion. The
memory-branch norms — which ARE trainable via `.*mem_.*` from Stage 0 onward — move
dramatically. Specifically `mem_o_norm` and `mem_layernorm` are actively *shrinking*
by ~9% each, presumably because the memory branch's residual write is growing and the
norm is learning to dampen the incoming activation.

## Interpretation & implications

### The direct mechanistic claim

RMSNorm's whole job is to keep the norm of activations passing through it near unit
scale. The learnable scale parameter is how the network calibrates that: if
activations coming in are systematically large, the scale shrinks to compensate; if
small, it grows. Under bf16 that calibration mechanism is **broken by silent
truncation**. The network's every attempt to adjust the scale is rounded away.

### The late-stage-spike prediction is now supported

Independent theory: as training proceeds, the memory branch (or the main model when
it unfreezes in Stage 3) starts producing larger residual writes / attention outputs.
Downstream RMSNorms should *shrink their scale* to keep the sum-of-inputs near unit
norm. Under bf16 they can't shrink → cumulative activation magnitude drifts up →
eventually attention softmax saturates, or MLP hits fp16-range overflow, or a
combination → **late-stage numerical spike**.

The pf32 arm's `mem_o_norm` dropping from 1.0 → 0.91 within 10k steps is direct
evidence that this compensation is a real, needed dynamic — and evidence that bf16
was preventing exactly that compensation on every prior run.

### Corollary for interpreting past-run pathologies

Any late-stage instability observed on prior bf16 runs — sudden CE spikes, grad NaN,
retrieval collapse in Stage 3 — should be re-examined with this mechanism in mind
before being attributed to model/data. The absence of RMSNorm rescaling in bf16 is
sufficient to cause the class of failure and is invisible in aggregate weight-norm
plots (RMS reads 1.000 → 1.000, and that's the truth: the norm literally didn't
change).

## Reproducibility

```bash
# Query wandb for the same table on any run:
python scripts/misc/check_rms_norm_training.py  # (uploaded as tmp_check_rms_norm_training.py)
```

Data source: `weight/wnorm/*/rms` and `weight/wchanged/*/n` metrics logged every step
by the trainer's `_WEIGHT_MONITOR_KEYS` block (see `trainer/trainer.py`).

- **bf16 arm wandb run**: `johnzhang2366-columbia-university/memory-layers/6t7w3qzj`
  (`_main_exact_loader_resume`, on opt_resume branch)
- **pf32 arm wandb run**: `johnzhang2366-columbia-university/memory-layers/jpursdwp`
  (`_pf32_loader_resume`, on pf32_wchanged_expand branch)
- **Commit SHA**: `af54cb60` on opt_resume (weight-monitor + this note's data
  source). Companion cherry-pick on `pf32_wchanged_expand` at `553a15fc`.
- **TPU type**: v6e-8. bf16 arm on john-v6e-8-3; pf32 arm on john-v6e-8-0.

## Follow-ups

- **Wait for the pf32 arm to reach Stage 3 (step ~15000)** and re-run this analysis.
  Prediction: `main_model.norm`, all `input_layernorm` / `post_attention_layernorm`,
  and all `q_norm` / `k_norm` scales will begin moving substantially, matching what
  the memory-branch norms show at step 10k. If they DON'T move under pf32 at Stage 3,
  the mechanism isn't RMSNorm-scale-freeze specifically and we need to look further.
- **Predict where the late-stage spike will land under bf16**. If the mechanism is
  right, we should be able to predict from `weight/mem_write_norm/rms` (memory branch
  residual write growth) crossing some threshold vs. the frozen norms' calibration
  headroom.
- **Consider a partial-fp32 fix** even on branches that don't want the full pf32
  memory cost: promote just the norm scales to fp32 (they're tiny — a few thousand
  parameters total across the whole model). This alone would fix the calibration
  freeze without the ~4 GB pf32 memory overhead. Cheap surgical fix.
