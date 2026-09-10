# approx top-k on the grounding 4-layer config (v6e-8 flex slice)

**Date:** 2026-07-20 · **Author:** rohunagrawal · **Status:** done (Tier 1 — step time only)

## Conclusion

**`approx_max_k` @ recall 0.99 cuts the `ground_s1_zeroinit_4layer` train step 2.14×:
2260 ms → 1056 ms** on a v6e-8 slice. That is a **much larger win than the 1.63×** measured for
`qa_hard_neg_think_sft4b` ([2026-07-15](2026-07-15-approx-topk-training.md)), and the reason is
that this config does ~8× the retrieval work — **4 memory layers at `mem_top_k=128`** vs 1 layer
at 64 — so the top-k op dominates a correspondingly larger share of the step. Approx top-k should
be considered **on by default** for any multi-layer grounding run. `qwen3_mem_embed` already sets
`mem_approx_topk: true`, so runs off that model config get this for free.

**Not answered here:** whether a **v5p-4** is faster than v6e for this config — the question that
prompted the run. Two flex v5p-4 requests (us-central1-a, us-east5-a) were still `PENDING` when
this was written, so the v5p arm is outstanding. **Also not measured: quality.** This is a
step-time (Tier-1) result only; no loss/recall A/B was run for *this* config.

## Hypothesis & motivation

The memory-bank top-k is the step's dominant cost at large M, and `2026-07-15` showed
`approx_max_k` recovers 98.7% of exact top-64 at 0.99 recall while cutting the step 1.63×. That
measurement was taken on a **single** memory layer at `top_k=64`. The grounding line runs **four**
layers at `top_k=128`, so the expectation was a strictly larger speedup — the open question was
how much, since the rest of the step (4B forward/backward, optimizer) is unchanged and bounds the
achievable ratio.

## Setup

Real jitted `Trainer._train_step` on a synthetic batch of the true training shapes — the same
harness as 2026-07-15, extended to take extra Hydra overrides (`--overrides`). Stage-0 semantics
(mem+conv trainable, `ce_weight=0`, `doc_access_loss` 0.1). No checkpoint is loaded and no HF data
is touched: step time depends on tensor shapes, not token values.

| Held fixed | Value |
|---|---|
| model | `qwen3_mem_embed`, main `Qwen/Qwen3-4B` |
| memory | `mem_layers=[9,14,20,27]`, `mem_top_k=128`, `mem_o_proj_zero_init=true` |
| trainer | `staged_ground` (stage 0), `tp_devices=1` |
| data shapes | `qa_hard_neg_think_sft4b`: B=16, `seq_len` 512, 16 chunks × 256 → **M = 65,536 bank slots per layer** |
| swept | `MEM_APPROX_TOPK` ∈ {0, 1}; `MEM_APPROX_RECALL=0.99` on the approx arm |

Each arm is its own process (the env toggle is read at import). 5 warmup + 20 timed steps, each
step device-synced (`float(ce_loss)`) so the timing is full step wall-clock.

## Results

Median over n=20, per worker (both processes of the slice report independently):

| arm | worker 1wjb | worker 1z9d | p10 / p90 (1wjb) |
|---|---|---|---|
| exact `jax.lax.top_k` | **2260.12 ms** | 2259.89 ms | 2257.93 / 2262.27 |
| `approx_max_k` @ 0.99 | **1056.64 ms** | 1056.48 ms | 1054.67 / 1057.59 |
| **speedup** | **2.14×** | 2.14× | — |

The two workers agree to within 0.25 ms and the p10–p90 spread is ±2 ms, i.e. the measurement is
essentially noise-free — unsurprising for a synchronized slice on a fixed-shape synthetic batch.

Compile cost (warmup step 0, then step 1) was ~319 s / ~38 s exact and ~301 s / ~35 s approx;
steady state is reached by warmup step 2. Compile is excluded from the medians.

At the 1500-step midtrain length used by `train_musique_ground4layer_midtrain.sh`, the difference
is **~56 min → ~26 min** of step time.

### Comparison to the single-layer result

| config | layers × top_k | exact | approx@0.99 | speedup |
|---|---|---|---|---|
| `qa_hard_neg_think_sft4b` ([07-15](2026-07-15-approx-topk-training.md), v6e-8 single host) | 1 × 64 | 838.7 ms | 515.1 ms | 1.63× |
| `ground_s1_zeroinit_4layer` (this run, v6e-8 slice) | 4 × 128 | 2260 ms | 1056 ms | **2.14×** |

Consistent with top-k being the dominant term: 8× the retrieval work raises the exact step 2.7×
but the approx step only 2.05×.

## Reproducibility

- **Commit:** `600a773` + working-tree changes (`scripts/embed/bench_ground4layer_approx.sh` new,
  `--overrides` added to `scripts/embed/bench_approx_topk.py`).
- **TPU:** **v6e-8 flex slice** — 2 × `ct6e-standard-4t` in `europe-west4-a` under a `2x4`
  workload policy (`TPU_ACCELERATOR_TYPE=v6e-8`, `process_count=2`, 4 local chips each). See the
  [launch runbook §2.3](../infrastructure/experiment-launch-instructions.md) for how that box is
  provisioned — it is **not** the single-host shape, and a single-host launch **hangs**.
- **Command** (must run on **both** workers):

```bash
TRANSPORT=gce ZONE=europe-west4-a PROJECT_ID=memory-layers \
bash scripts/infrastructure/multi-tpu-box-run.sh \
  tpu-v6e-slice-mig-1wjb=scripts/embed/bench_ground4layer_approx.sh \
  tpu-v6e-slice-mig-1z9d=scripts/embed/bench_ground4layer_approx.sh
```

- No checkpoint, no wandb run, no GCS artifacts (synthetic batch, timing only). Box-side log:
  `~/bench_ground4layer_approx.log` on each worker.

## Interpretation & caveats

- **The ratio is the transferable result, not the absolute ms.** Per-device batch here is 2
  (B=16 over 8 chips); on a 4-chip box it is 4, so absolute step times will differ. The
  exact-vs-approx ratio is measured within one hardware config and is the decision-relevant
  number.
- **No quality measurement for this config.** 2026-07-15 measured 98.7% recall@64 at 0.99 for the
  *1-layer, top_k=64* case; recall at 4 layers × 128 was **not** re-measured. Approx is also the
  prime suspect for eval non-reproducibility (44/128 greedy generations diverged between runs, per
  `train_musique_ground4layer_midtrain.sh`), so keep it **off** for comparisons that turn on a
  small gap.
- **Bank size is the batch-derived 65,536**, not a large standing corpus. The
  [long-document result](2026-07-19-long-document-serving-cost.md) found that at a 6.32M-slot bank
  exact top-k *loses* outright; the gap here should widen further with bank size.
- **v5p arm outstanding** — the original question (is v5p-4 faster for this config?) is unanswered.
