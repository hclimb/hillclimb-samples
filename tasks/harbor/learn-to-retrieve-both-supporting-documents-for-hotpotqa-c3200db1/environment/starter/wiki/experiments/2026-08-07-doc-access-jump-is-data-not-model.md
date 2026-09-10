# The step-28000 doc_access_acc jump: data-triggered, model-responsive, weight-trapped

**Date:** 2026-08-07 · **Author:** johnzhang · **Status:** done (revised — first pass
missed the wdelta signal; second pass reconciles data + gradient response + bf16 trap)

## Conclusion

The step-function drop in `train/doc_access_acc` from ~0.94 to ~0.86 at step
28000 in run `6t7w3qzj` (`qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_main_exact_loader_resume`)
is **triggered by a data-region difficulty boundary** in the deterministic
training stream. The model responds correctly — gradients on the retrieval-scoring
projections (`mem_q_proj`, `embed_model.mem_k_proj`) **jump 2.5–2.7× and stay
elevated for thousands of steps** — but the bf16-ULP freeze traps those updates,
so weight RMS doesn't accumulate, retrieval accuracy plateaus at the new lower
level (~0.86, doesn't recover), and CE loss keeps drifting on `mem_o_proj` alone.

Two independent lines of evidence:

1. **Frozen-weights probe (dispositive on the CAUSE)**: at IDENTICAL ckpt=28000
   weights, batches from stream position 26001–26200 score
   `doc_access_acc = 0.9726 ± 0.017`; batches from stream position 28001–28200
   score `0.8558 ± 0.052`. **Δ = −0.117 at bit-identical weights.**
2. **Weight-monitor wdelta signal (real model response)**: per-step |Δw|
   on `main_model.layers.14.mem_q_proj` jumps `7.4e-8 → 2.0e-7` (2.7×) at step
   28000 and stays there. `embed_model.mem_k_proj` jumps `3.4e-7 → 8.8e-7`
   (2.6×). Elements-changed counts (`weight/wchanged/*/n`) show the same
   1.66–1.68× step-function at 28k. Sustained shift, not a transient.

The data trigger is upstream; the gradient response is real; the accumulation
is silently killed by the bf16 storage.

## Hypotheses eliminated first

1. **Resume boundary** — no. GCS ckpt creation timestamps for steps 22k/24k/26k/28k/30k/32k
   are separated by 32:40–32:47 min each (continuous training at ~1 it/s × 2000 steps).
   No preemption gap.
2. **Retrieval-path param re-init / optimizer state group mismatch on restore** — no,
   since (1) rules out any restore having happened.
3. **`mem_o_proj` bf16 exponent-bucket crossing** — no. RMS at step 27900 = 0.020013,
   step 28000 = 0.020013 (bit-identical). Nowhere near the next bucket at 2⁻⁵ = 0.03125.
4. **Systematic ckpt-save Grain worker respawn** — no. Sweep across boundaries
   22k/24k/26k/28k/30k/32k/34k showed step 28000 is the ONLY one with |Δacc| > 0.02.
5. **Weight-RMS accumulation discontinuity at 28k** — no. All 21 watched weight
   `weight/wnorm/*/rms` values change by less than 0.002% across the boundary
   (ratio ∈ [0.999996, 1.000018]). BUT: the per-step gradient / update magnitude
   (`weight/wdelta/*/abs_mean` and `weight/wchanged/*/n`) DID jump — see the
   dedicated section below. RMS just doesn't accumulate the jump because of
   bf16-ULP truncation.

## Probe

`scripts/debug/probe_h_b_data_heterogeneity.py`. Frozen-weights forward-only sweep:
loads ckpt=28000 weights, restores Grain loader state from
`ckpt=26000/dataloader_state.json` (pre-jump position), runs 200 forward-only steps;
then restores from `ckpt=28000/dataloader_state.json` (post-jump position), runs
another 200 steps; compares mean `doc_access_acc`.

Ran on john-v6e-8-2 via `run_probe_h_b.py`. Total wallclock: ~1h (dominated by
Grain fast-forward for the two loader-state restores).

## Results — probe

| condition | data stream position | mean doc_access_acc | std | n |
|---|---|---|---|---|
| PRE | steps 26001..26200 | **0.9726** | 0.0165 | 200 |
| POST | steps 28001..28200 | **0.8558** | 0.0523 | 200 |

**Δ (PRE − POST) = +0.117 at IDENTICAL weights.**

Two features:
- Mean drops by ~12 percentage points.
- Std triples (0.017 → 0.052), so post-jump batches are BOTH lower-mean AND
  higher-variance. Consistent with "systematically harder / more heterogeneous
  batches at that stream position."

## Results — weight-monitor gradient / update response

Retrieval-scoring projections show a real, sustained shift in per-step update
magnitude at step 28000, corroborating the H_B "harder batches" finding:

| metric (500-step window) | steps 27500–28000 (pre) | steps 28000–28500 (post) | ratio |
|---|---|---|---|
| `weight/wchanged/embed_model.mem_k_proj/n` (elements/step) | 41,100 | 68,090 | **1.66×** |
| `weight/wchanged/main_model.layers.14.mem_q_proj/n` | 125,200 | 210,200 | **1.68×** |
| `weight/wdelta/embed_model.mem_k_proj/abs_mean` | 3.42e-7 | 8.83e-7 | **2.58×** |
| `weight/wdelta/main_model.layers.14.mem_q_proj/abs_mean` | 7.37e-8 | 2.01e-7 | **2.72×** |
| `weight/wdelta/main_model.layers.14.mem_o_proj/abs_mean` | 1.00e-6 | 9.10e-7 | **0.91×** (dropped) |

The shift is **step-function** at step 28000 (per-10-step zoom around 27950–28050
shows a clean jump right at 28000, not a ramp) and **sustained** — the elevated
mem_q_proj wdelta persists through step 30500 (last window checked). Not a
transient.

The asymmetry between the retrieval-scoring projections (bigger gradients: 2.6–2.7×)
and the output projection `mem_o_proj` (slightly smaller: 0.91×) is exactly what
you'd predict: harder retrieval → the "which memory to pick" path receives more
gradient signal, and the "how to use retrieved memory" path receives slightly less
because the retrieval is failing upstream.

## Interpretation — reconciled

The `qa_hard_neg_think_sft4b` interleave has 4 sources with fixed
`shuffle_seed=42` and `interleave_datasets(seed=42, stopping_strategy="all_exhausted")`.
The mix (which source contributes each batch, and the per-source `.shuffle(buffer_size=100k)`
draw order) makes the effective batch difficulty vary by global step position. Around step
28000 the mix crosses into a region with harder retrieval targets (denser hard-negs,
harder pos-doc discrimination, or both). This is the **root cause**.

The model responds correctly — bigger gradients hit the retrieval-scoring projections
(mem_q_proj, mem_k_proj) exactly as they should. Adam's per-step update magnitude jumps
2.6–2.7× and stays elevated for thousands of steps as the model tries to catch up on
the harder distribution.

But **the bf16-ULP freeze** (see companion note
[`2026-08-07-bf16-ulp-freeze-empirical-confirmation.md`](2026-08-07-bf16-ulp-freeze-empirical-confirmation.md))
prevents accumulation: at |w| ≈ 0.02, bf16 ULP ≈ 1.5e-4, so per-element updates of
~2e-7 round to zero on most elements. RMS doesn't move; the extra bit-flips
(`wchanged/n` at 1.66–1.68×) are the marginal elements that JUST barely cross the
ULP threshold, but even those don't accumulate meaningfully. Result: retrieval
accuracy plateaus at the new lower level (~0.86) rather than recovering back to
the pre-jump 0.94.

**CE loss is decoupled** from this because under bf16 essentially only `mem_o_proj`
(zero-init, escapes the ULP trap) actually trains, and it's downstream of the
retrieval mechanism — it learns to route hidden states via whatever memory the
retrieval happens to pick, good or bad. That's why CE keeps drifting down through
the plateau while doc_access_acc stays stuck.

## Implications

- **`train/doc_access_acc` and `train/doc_access_loss` are step-noisy in ways that
  reflect data-region difficulty, not training dynamics.** Any interpretation of a
  jump/drop in these metrics needs a moving average of at least 2000 steps to
  average out data-region effects. Per-step or per-100-step comparisons across a
  boundary can be entirely spurious.

- **Weight-monitor `wdelta`/`wchanged/n` jumps are NOT independent evidence of a
  model event.** They correctly reflect gradient magnitude responses to data-region
  shifts. Only `weight/wnorm/*/rms` changes are dispositive for "did the model
  actually accumulate a change." In this run, the wdelta 2.7× jump is real gradient
  response but the RMS didn't move, because bf16-ULP truncated the accumulation.

- **`train/ce_loss` was NOT similarly affected** at step 28000 (~0.71 both sides).
  Under bf16 with the fp32-freeze on norms, the LM path is essentially decoupled
  from retrieval quality — CE moves with what `mem_o_proj` alone can compensate for,
  not with what the retrieval mechanism actually retrieves. That decoupling is itself
  documented in `2026-08-07-bf16-ulp-freeze-empirical-confirmation.md` and is why
  retrieval-side metric jumps don't propagate to CE.

- **Under pf32 storage** (currently running as `run_baseline_pf32_loader_resume.py`
  on TPU 0), we expect the retrieval-scoring weights to actually accumulate their
  gradient response to the harder region: RMS should shift, and retrieval accuracy
  should partially recover after the initial drop instead of plateauing. That's an
  open A/B; the pf32 run needs to reach step 28000+ to observe.

- **The Feistel / pre-tokenize preprocessing overhaul discussed elsewhere would fix
  the data-heterogeneity root cause** by globally reshuffling the corpus once,
  spreading hard batches uniformly across the run. It does not fix the bf16-ULP
  weight trap — that's the pf32 fix's job.

## Reproducibility

```bash
# On a TPU box with the opt_resume branch checked out:
HF_HUB_OFFLINE=1 GROUND_HF_PARQUET=$HOME/hf_parquet \
    uv run python scripts/debug/probe_h_b_data_heterogeneity.py \
    --ckpt_dir gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_main_exact_loader_resume-2026-08-06-09-09-19/qwen3_mem_embed \
    --weights_step 28000 \
    --pre_step 26000 \
    --post_step 28000 \
    --n_steps 200
```

- **Commit SHA:** `f12deae7` on `opt_resume` (probe script + HydraConfig-bypass fix).
- **Wandb run under analysis:** `johnzhang2366-columbia-university/memory-layers/6t7w3qzj`.
- **Checkpoint dir:** `gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_main_exact_loader_resume-2026-08-06-09-09-19/qwen3_mem_embed/`.
- **TPU type:** v6e-8 (john-v6e-8-2), us-east1-d, memorylayers project.
- **Probe log:** on-box `/home/underfrog/probe.log` (contains PRE and POST full step
  traces plus the SUMMARY block cited above).

## Follow-ups

- Sweep more data-region boundaries to see if step 28000 is unique or one of many
  similar-magnitude jumps (predicts many with wide MA).
- Wait for the pf32 arm (TPU 0, `run_baseline_pf32_loader_resume`) to reach
  step 28k+ and compare the retrieval-recovery shape. Prediction: under pf32
  the retrieval accuracy dips less deeply and partially recovers as the accumulated
  gradient response actually moves the weights.
- Implement pre-tokenized indexed shards + Feistel index permutation — see
  discussion in the session log; eliminates the data-heterogeneity class of
  artifact entirely and fixes loader-resume simultaneously (orthogonal to the pf32
  fix — this addresses the data trigger, pf32 addresses the accumulation trap).
