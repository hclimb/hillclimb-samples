# Approx top-k during training — does `qa_hard_neg_think_sft4b` benefit?

**Date:** 2026-07-15 · **Author:** rohunagrawal · **Status:** done — adopted (approx@0.99 is the
`qwen3_mem_embed` config default); Tier 2 A/B skipped by decision.

## Conclusion (first)

**Tier 1 (microbench): a large, clean speed win → adopted.** Flipping the train-time memory top-k
to `approx_max_k` cuts the `qa_hard_neg_think_sft4b` step **1.68× at recall 0.95 (838.7 → 499.8 ms)
/ 1.63× at 0.99 (515.1 ms)** — the exact top-k over M=65,536 was ~40% of the whole 4B step.
Selection fidelity is high: approx recovers **98.2% (target 0.95) / 98.7% (target 0.99)** of the
exact top-64 and ~98% of the score mass, and this is a *conservative* estimate (measured on a
random-init layer; a trained layer's peakier scores are easier to recall).

**Decision:** the speedup is large and fidelity high enough that the Tier-2 A/B was judged
unnecessary; **approx@0.99 is now the `qwen3_mem_embed` config default** (`mem_approx_topk: true`,
`mem_approx_recall: 0.99`; env still overrides). See the implementation note
[2026-07-15-approx-topk-config-default.md](../implementations/2026-07-15-approx-topk-config-default.md).
If a quality regression is ever suspected, the deferred Tier-2 A/B (exact vs approx, real data)
is the check — it needs the 4 HF datasets pre-cached first (16 grain workers hit HF's 1000-req/5-min
limit on the uncached repos, which is why Tier 1 used a synthetic batch).

## Hypothesis & motivation

The train step scores every memory key per query token and takes the top-`k`. For this run the
bank is built from the batch, so per step:

```
M = batch(16) × num_chunks_per_doc(16) × doc_chunk_seq_len(256) = 65,536 keys
score tensor: [B=16, mem_num_heads=4, T=512, M=65,536]  →  top-64 over the M axis
```

Exact `top_k` over a 65k axis is sorting-network-bound on TPU; `models/memory_utils.bank_top_k`
already exposes `jax.lax.approx_max_k` behind `MEM_APPROX_TOPK=1` (claimed ~6× faster at ~98%
recall@128 in the code comment). **Hypothesis:** at M≈65k the top-k is a large enough slice of the
step that approx gives a measurable step-time win, and recall-0.95 selection is faithful enough
not to hurt the retrieval gradient. **Risk:** approx misses ~5% of the true top-k each step, which
could weaken the very signal this run trains (which doc to retrieve).

This run takes the replicated `mem_lookup` path (`tp_devices=1` → model axis 1 → full-score-matrix
branch → `bank_top_k`), so `MEM_APPROX_TOPK` is live in exactly this config.

## Setup

- **Model:** `qwen3_mem_embed` (main `Qwen/Qwen3-4B`, embed `Qwen3-Embedding-0.6B`), 1 memory
  layer @ L14, `after_attention`, `mem_num_heads=4`, `mem_k_dim=mem_v_dim=1024`, **`mem_top_k=64`**,
  softmax scores, no product keys / two-pass / GQA / chunking.
- **Data:** `qa_hard_neg_think_sft4b`, `seq_len=512`, `num_chunks_per_doc=16`,
  `doc_chunk_seq_len=256`, `batch_size=16`.
- **Independent variable:** the top-k op — exact (`MEM_APPROX_TOPK=0`) vs approx
  (`MEM_APPROX_TOPK=1`) at `MEM_APPROX_RECALL ∈ {0.95, 0.99}`. Everything else held fixed.
- **Metrics:**
  - *Step time* — median wall-clock of the real jitted train step (`Trainer._train_step`, stage-0
    config: mem+conv trainable, `ce_weight=0`, `doc_access_loss` weight 0.1), warmup excluded.
  - *Selection fidelity* — `mean_recall@64` = fraction of the exact top-64 recovered by approx, and
    the exact-score mass recovered, measured on the **real** per-token score matrix.
- **Hardware:** TRC `v6e-8` (`rohun-v6e-8-0`), `europe-west4-a`.

## Reproducibility

- **Commit:** `62928dc` (branch `train_approx_top_k`) + the bench harness
  `scripts/embed/bench_approx_topk.{py,sh}` (this change).
- **Launch:** from the worktree,
  ```bash
  TPU_NAME=rohun-v6e-8-0 RUN_SCRIPT_PATH=scripts/embed/bench_approx_topk.sh \
    bash scripts/infrastructure/multi-vm-tpu-run.sh
  ```
  which runs, on the box:
  ```bash
  MEM_APPROX_TOPK=0                       uv run python scripts/embed/bench_approx_topk.py --mode time
  MEM_APPROX_TOPK=1 MEM_APPROX_RECALL=0.95 uv run python scripts/embed/bench_approx_topk.py --mode time
  MEM_APPROX_TOPK=1 MEM_APPROX_RECALL=0.99 uv run python scripts/embed/bench_approx_topk.py --mode time
                                           uv run python scripts/embed/bench_approx_topk.py --mode recall
  ```
- **The bench reuses the real training path** (`get_model` → `setup_optimizer` → `get_dataset` →
  `Trainer._train_step`); the only difference from the real run is the top-k op. No checkpoint /
  wandb (microbench). Box-side log: `~/bench_approx_topk.log`.

## Results — Tier 1 (microbenchmark)

### Step time (median, warmup excluded, n=20; synthetic batch of the real shapes)

| top-k op | recall_target | median step (ms) | mean | p10 / p90 | vs exact |
|----------|---------------|------------------|------|-----------|----------|
| exact `top_k` | — | **838.71** | 839.00 | 837.75 / 839.59 | 1.00× |
| approx `approx_max_k` | 0.95 | **499.81** | 500.08 | 499.68 / 500.03 | **1.68× (−40.4%)** |
| approx `approx_max_k` | 0.99 | **515.08** | 515.25 | 514.79 / 515.41 | **1.63× (−38.6%)** |

The exact top-k over M=65,536 accounts for **~340 ms of the 838 ms step (~40%)** — `approx_max_k`
nearly eliminates it, for a **1.6–1.7× whole-step speedup**. Variance is tiny (p90−p10 < 2 ms). The
first (compile) step was ~230 s and is excluded by warmup.

### Selection fidelity (approx vs exact top-64, real model logits on the synthetic batch)

| recall_target | mean_recall@64 | min_recall@64 | score-mass recovered |
|---------------|----------------|---------------|----------------------|
| 0.90 | 0.9731 | 0.7812 | 0.9747 |
| 0.95 | 0.9822 | 0.8125 | 0.9832 |
| 0.99 | 0.9872 | 0.8906 | 0.9880 |

Empirical recall **exceeds** each `recall_target` (0.95 → 98.2%). Measured over 4000
(position, head) rows of the real `[16, 511, 4, 65536]` score tensor. **Caveat:** no checkpoint
is loaded, so these are on a *randomly-initialized* memory layer whose scores are near-uniform;
a trained model's scores are peakier (the positive doc's keys dominate), which makes the top-k
*more* separable — so these numbers are a **conservative** estimate of the trained-model recall.

## Interpretation

Both Tier-1 criteria are met decisively: the approx step is **much** faster (1.6–1.7×, not a
marginal few percent) and selection fidelity is high (98%+). The speedup is inherent to the op —
the score matmul is identical in both arms, so the ~340 ms delta is purely exact-sort vs
approx-sort over 65,536, exactly the sorting-network cost the `bank_top_k` comment predicted.

Caveats / confounds:
- **Random-init recall.** Fidelity was measured without a checkpoint; a trained model should
  recall *at least* this well (peakier scores → larger top-k gaps). The definitive quality signal
  is Tier 2's loss curve, not this number.
- **Synthetic batch.** Faithful for step *time* (shape-driven) and for the recall op, but token
  values are random. Tier 2 uses the real dataset.
- **What approx changes for training:** ~1–2% of the top-64 differ each step, so the retrieval
  gradient is trained against a slightly noised neighbor set. Tier 2 tests whether that matters.

**Next:** Tier 2 A/B — two short `qa_hard_neg_think_sft4b` runs (exact vs `MEM_APPROX_TOPK=1`,
`MEM_APPROX_RECALL=0.99`), identical otherwise, compare train loss + `doc_access` curves on wandb.
Requires the HF dataset pre-cache noted above.
