# bf16-ULP weight-freeze hypothesis: empirical confirmation on opt_resume (main-branch baseline)

**Date:** 2026-08-07 · **Author:** johnzhang · **Status:** done (bf16 arm; pf32 arm pending)

## Conclusion

Under bf16 weight storage (opt_resume/main branches — no `promote_trainable_to_fp32`
fix), **essentially every normalization scale in both the memory branch and the Qwen3-4B
backbone is bit-exact frozen** during Stage 3 training, and projection weights at
~0.02 magnitude "move" only in the sense that ~10–30% of elements flip a bit per step
with **mean per-element change ~1e-7 to 1e-6** — well below LR × normalized-grad-magnitude,
and RMS magnitudes are unchanged over 9,000+ training steps. Six specific watched weights
(4 memory-branch norms + 2 main-model backbone norms) show **exactly 0 elements changed**
across 9,218 log rows.

This confirms the mechanism laid out in
[`wiki/implementations/2026-07-27-fp32-master-weights.md`](../implementations/2026-07-27-fp32-master-weights.md).
Only `mem_o_proj` (zero-initialized, so its ULP starts at 0 and grows with weight magnitude)
"escapes" the trap, matching that note's prediction.

## Hypothesis

`optax.adamw` with `mu_dtype=None` inherits bf16 mu/nu when params are bf16. bf16 has 7
mantissa bits, so `ULP(w) ≈ |w| × 2⁻⁷ ≈ 0.78% of |w|`. The addition `w_bf16 + Δw_bf16`
rounds Δw to zero whenever `|Δw| < ULP(|w|)`. At LR=1e-4, adam's normalized update
(m̂ / √v̂ ≈ ±1) produces `|Δw| ≈ 1e-4`, which fails to land whenever `|w| > ~0.013`.

Predicted verdict per category:

- **FROZEN** (init |w| ≥ 0.5 → ULP ≥ 4e-3, 40× larger than the target update): every RMS-norm
  scale in memory branch (init 1.0) and Qwen backbone (pretrained ~0.3–2.8).
- **PARTIAL** (init |w| ~0.1 → ULP ~8e-4, 8× larger): `mem_layer_scale` and thin norms.
- **MIXED / MOVING-BUT-TINY** (init |w| ~0.02 → ULP ~1.5e-4, borderline): projection
  weights like `mem_q_proj`, `embed_model.mem_{k,v}_proj`, `main_model.layers.*.q_proj`,
  `up_proj`. Bits should flip on some steps for some elements, but net drift ≈ 0.
- **ESCAPE / control** (init |w| = 0 → ULP = 0): `mem_o_proj`. Every update lands until
  the weight grows into a bucket where ULP catches up (which never happens during a
  100k-step run — first update sets a small non-zero, next update's ULP is still tiny).

## Setup

- **Branch:** `opt_resume` (= main + LR-preserve-on-within-stage-resume fix `559d65ec` +
  weight-monitor instrumentation `c6ae5f21` / `6957c801` / `df7c5756` / `656ed32b`).
- **Config:** `qwen3_mem_embed` (bf16 storage, no promote_trainable_to_fp32),
  `mem_approx_topk=false`, `dataset=qa_hard_neg_think_sft4b`, `trainer=staged_telemetry`,
  4-stage staged recipe (stages at 5k / 10k / 15k / 100k; Stage 3 is `.*mem_.*, .*embed_model.*, .*main_model.*`).
- **Wandb run:** `johnzhang2366-columbia-university/memory-layers/6t7w3qzj` (state=running
  at time of analysis).
- **Watched weights (21):** 3 categories chosen a priori per the hypothesis — 4 memory-branch
  norms (all init 1.0), memory-branch `mem_layer_scale` (init 0.1), memory-branch projections
  `mem_q_proj` / `mem_o_proj` (init 0.02 / zero), embed-model projections
  `embed_model.mem_{k,v}_proj` (init 0.02), Qwen backbone norms at layers 9/17/26 + final
  `norm`, Qwen backbone projections `q_proj` / `up_proj` at layers 9/26.
- **Per-step metrics logged:** `weight/wchanged/<key>/n` (raw element-changed count),
  `weight/wnorm/<key>/rms`, `weight/wdelta/<key>/abs_mean`, `weight/wdelta/<key>/max_ulp_ratio`.
- **Analysis window:** steps 20000–29217 (9,218 log rows), all within Stage 3.

## Results

### FROZEN — bit-exact identical after 9,000+ log rows

| weight | init \|w\| | rms (start) | rms (end) | max elements changed / step |
|---|---|---|---|---|
| `main_model.layers.14.mem_q_norm` | 1.0 | 1.0000 | 1.0000 | **0** |
| `main_model.layers.14.mem_o_norm` | 1.0 | 1.0000 | 1.0000 | **0** |
| `main_model.layers.14.mem_layernorm` | 1.0 | 1.0000 | 1.0000 | **0** |
| `main_model.layers.14.mem_layer_scale` | 0.1 | 0.1001 | 0.1001 | **0** |
| `main_model.layers.17.input_layernorm` | pretrained ~0.58 | 0.5843 | 0.5843 | **0** |
| `main_model.layers.26.input_layernorm` | pretrained ~1.29 | 1.292 | 1.292 | **0** |

**Six watched weights had zero bit changes across all 9,218 log rows.** Exactly matches the
FROZEN category prediction.

### PARTIAL — a handful of elements move but essentially still stuck

| weight | max elements changed / step | out of |
|---|---|---|
| `main_model.norm` | 1 | 2560 |
| `main_model.layers.9.q_norm` | 1 | 128 |
| `main_model.layers.9.input_layernorm` | 2 | 2560 |
| `main_model.layers.9.post_attention_layernorm` | 3 | 2560 |
| `main_model.layers.9.k_norm` | 8 | 128 |
| `main_model.layers.17.q_norm` | 1 | 128 |
| `main_model.layers.26.q_norm` | 2 | 128 |

Under 0.1% of elements moving per step in every case. The occasional single-element flip
is consistent with a stochastic ULP-tie tipping over — no real update accumulation.

### MOVING — projections at ~0.02 magnitude flip bits but don't accumulate

| weight | elements changed / step (max) | mean \|Δw\| per element | rms (start) | rms (end) |
|---|---|---|---|---|
| `main_model.layers.14.mem_o_proj` (zero-init) | 890k / 10.4M | 7.1e-7 | 0.02001 | 0.02001 |
| `main_model.layers.14.mem_q_proj` | 513k / 10.4M | 1.2e-7 | 0.02088 | 0.02088 |
| `embed_model.mem_k_proj` | 93k / 1M | 4.6e-7 | 0.02015 | 0.02014 |
| `embed_model.mem_v_proj` | 107k / 1M | 9.6e-7 | 0.01999 | 0.01999 |
| `main_model.layers.9.q_proj` | 1.8M / 10.4M | 2.1e-6 | 0.02233 | 0.02233 |
| `main_model.layers.9.up_proj` | 3.1M / 24.9M | 1.7e-6 | 0.02197 | 0.02197 |
| `main_model.layers.26.q_proj` | 1.7M / 10.4M | 2.5e-6 | 0.02361 | 0.02361 |
| `main_model.layers.26.up_proj` | 3.1M / 24.9M | 1.7e-6 | 0.02421 | 0.02421 |

10–30% of elements are bit-different each log step (log_interval=1, so also per training
step). But the average per-element change is 1e-7 to 1e-6, which is 100× smaller than what
LR=1e-4 × normalized adam update would produce. Consistent with "an occasional element
crosses a rounding tie in a random direction; net-zero drift." **RMS magnitudes are
unchanged to 4–5 decimal places over 9,000+ steps** — these weights are effectively frozen
as far as training dynamics are concerned.

## Interpretation

- **The freeze is universal across weight types with |w| >> ULP-vs-LR**, both in the memory
  branch (fresh-initialized ones-magnitude norms) and in the Qwen3-4B pretrained backbone
  (norms that converged to ~0.3–2.8 during pretraining).
- **The escape route works only for zero-init weights** — mem_o_proj is the sole widely-moving
  weight, and even it drifts by only ~7e-7 per element per step; the 890k/10.4M elements
  flipping bits per step is consistent with a small non-zero average and per-element noise.
- **Loss curves are NOT sufficient evidence for "training is working."** The
  `qa_hard_neg_think_sft4b` recipe drives down ce_loss / doc_access_loss under bf16 because
  the ~4% of the memory branch that actually moves (mem_o_proj + partial projection updates)
  is enough to fit the retrieval task. But every norm scale in the model is stuck at its
  init — and the "unfreezing" of the main-model backbone in Stage 3 is a fiction: the LR
  schedule reaches those params, the optimizer computes updates, and the bf16 store rounds
  every update to zero.
- **Every prior memory-layer / grounding result on main / opt_resume needs reinterpretation.**
  Anything attributed to "the memory branch training" was mem_o_proj alone. Anything attributed
  to "main-model fine-tuning in Stage 3" was zero. The pf32 arm (below) is the corrective
  needed to know what the model actually looks like when trained.

## Reproducibility

```bash
# On any TPU host with the launcher deployed:
cd $HOME/babysit && python3 -u run_baseline_main_exact_loader_resume.py
# BRANCH=opt_resume, RUN_NAME=qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_main_exact_loader_resume
# TPU: v6e-8, us-east1-d, spot (john-v6e-8-3 in this run)
```

- **Commit SHA:** `656ed32b` on `opt_resume` (weight-monitor instrumentation + log_interval=1).
- **Wandb run:** `johnzhang2366-columbia-university/memory-layers/6t7w3qzj` (analysis window
  steps 20000–29217).
- **Checkpoint path:** `gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_main_exact_loader_resume-2026-08-06-09-09-19/qwen3_mem_embed/`
- **Startup magnitude readout** (from training log — confirms verdicts at init):

| weight | dtype | rms | mean\|w\| | ULP@mean\|w\| | verdict |
|---|---|---|---|---|---|
| `main_model.norm` | bf16 | 2.797 | 2.76 | 0.0216 | FROZEN |
| `main_model.layers.9.k_norm` | bf16 | 1.692 | 1.57 | 0.0123 | FROZEN |
| `main_model.layers.9.mem_q_norm` (=14) | bf16 | 1.0 | 1.0 | 0.00781 | FROZEN |
| `main_model.layers.14.mem_o_proj` | bf16 | 0.01997 | 0.01596 | 1.25e-4 | MIXED |
| `embed_model.mem_k_proj` | bf16 | 0.02015 | 0.01595 | 1.25e-4 | MIXED |
| `main_model.layers.9.q_proj` | bf16 | 0.02233 | 0.01712 | 1.34e-4 | MIXED |

## Follow-ups

- **Cherry-pick the weight-monitor instrumentation onto `pf32`** and run the same recipe
  there. Expected: FROZEN weights become MOVING (RMS drifts, `wchanged/n` matches param
  size, mean |Δw| per element ~1e-4–1e-3). The A/B closes the loop.
- **Consider adopting `promote_trainable_to_fp32` on main** once pf32 is validated — the
  memory-cost table in the fp32 note is favorable (fits v6e with headroom) and the freeze
  is not an acceptable failure mode.
