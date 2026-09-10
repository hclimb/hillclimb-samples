# Train-loop throughput: `log_interval` pipelining + offline-parquet in the recipe

**Date:** 2026-07-16 · **Author:** rohunagrawal · **Status:** done (code landed; measured, not yet run
end-to-end at scale)

Two quality-neutral throughput fixes for the `qa_hard_neg_think_sft4b` recipe, motivated and measured
in [../experiments/2026-07-16-train-speed-axes.md](../experiments/2026-07-16-train-speed-axes.md).

## Motivation

The train loop was gated by two host-side stalls, neither touching the training math:
1. **Per-step device sync.** `Trainer.train()` pulled `float(ce_loss)`/`int(nan)`/`float(grad_norm)`
   and called `wandb.log` **every step**, forcing a device→host sync that serialized the
   drain+redispatch bubble pipelining would hide. Measured cost: **+10% of real step time**
   (544.7→490.3 ms; pipelined ≈ the 487 ms device-bound ceiling).
2. **Live-HF data streaming.** `train_hard_neg_think.sh` had no `HF_HUB_OFFLINE`, so 16 grain workers
   × 4 datasets blew HF's 1000-req/5-min quota during pipeline build → `sleep 68–172 s → retry`
   loop, **TPU idle for minutes**. Offline-parquet reads local shards: 49.7 batch/s, 0 stalls.

## Options & tradeoffs

- **(loop) Rewrite the loop to fully double-buffer / async-log** — biggest reshape, most risk. Not
  needed: a sampling cadence recovers essentially all of the bubble.
- **(loop, chosen) `trainer.log_interval`** — pull losses + `wandb.log` only every N steps; default
  `1` is byte-identical to the old per-step behavior (`step % 1 == 0` always true). Tradeoff: logging
  granularity coarsens at N>1, and the host-side `nan_count` becomes sampled — but the **on-device
  NaN-skip guard in `_train_step` still runs every step**, so training is unaffected.
- **(data, chosen) offline-parquet in the recipe** — `HF_HUB_OFFLINE=1` + `GROUND_HF_PARQUET`, exactly
  what every `train_ground_*.sh` already does. Requires `scripts/misc/precache_hf.sh` on the box once.
  Same *plumbing* (no math change), but `precache_hf.sh` caches only ~50% of shards by default (~18G;
  boot disk is 94% full) = 2.06M rows — more than one 1.6M-example 100k-step epoch, so enough for a run,
  but it's the FIRST half of shards, not the literal full dataset (mild ordering caveat).

## Approach & integration

- `trainer/trainer.py::Trainer.train()` — the per-step host pulls (`int(loss_nan)`, `int(grad_nan)`,
  `float(ce_loss)`, `float(aux["total"])`, `wandb.log`) are wrapped in `if step % log_interval == 0
  or is_last:`; `pbar.update(1)` stays every step.
- `configs/trainer/standard.yaml` — `log_interval: 1` (documented default).
- `scripts/embed/train_hard_neg_think.sh` — prepends `HF_HUB_OFFLINE=1 GROUND_HF_PARQUET=...` and
  sets `trainer.log_interval=10`.
- No change to `_train_step`, the optimizer, or any loss — purely when the host reads results.

## Test record

- **`scripts/embed/bench_train_throughput.py`** (real offline dataloader + real `_train_step`,
  v6e-8, stage 0, 200 steps/arm): baseline (sync-each) **544.67 ms/step** vs pipelined (sync-20)
  **490.32 ms/step** → **+9.98%**; pipelined ≈ the 487 ms compute ceiling; offline data-fetch = 21 ms,
  0 stalls. This exercises the exact pipelined-loop pattern `log_interval>1` produces.
- **`scripts/embed/bench_data_throughput.py`**: live-HF never reached batch 1 (429 loop);
  offline-parquet 49.7 batch/s, 0 stalls, 24× the compute rate.
- **Default (`log_interval=1`) is provably a no-op** — the guard runs every step. Not separately run.
- **Not yet done:** a full multi-thousand-step `train.py` run at `log_interval=10` (deferred — the
  final orbax save is ~46 GB; the pipelined pattern itself is bench-verified above).

## Axis A — `trainer.stop_grad_frozen` (implemented, default OFF, pending A/B)

Also landed a third, gated lever: `trainer.stop_grad_frozen` (default `false`). When on, `_train_step`
`stop_gradient`s the weights not matching the stage's `trainable_params` (frozen set) before the
forward, so XLA prunes their backward — **2.12× the step in the frozen-main stages 0–2** (`bench_axisa.py`:
487→230 ms). Wired via a static `trainable_patterns` arg to `_train_step`, passed by the trainer from
`current_trainable_patterns` (tracked per stage; skipped when "all"-trainable).

**Quality status — exact-neutral, but not adopted-on yet.** `stop_gradient` on a *frozen* weight zeros
only that weight's own grad and cannot change a *trainable* weight's grad in exact arithmetic (proof),
and `bench_axisa_grad.py` confirms the wiring (frozen grads = exactly 0, baseline = 2.5e-3). **However**,
at bf16 the trainable grads differ ~3% median / 7.5% max (reduction-order rounding on the M=65k score
backward — `mem_q_proj` worst). That's purely numerical (same class as remat/sharding non-determinism),
but larger than a trivial no-op, so it is left **OFF** in the recipe pending a short on-vs-off loss A/B.
Enable with `trainer.stop_grad_frozen=true` once that A/B confirms curves track.

## Reference pages touched

- [../training/training-loop.md](../training/training-loop.md) — logging/sync cadence (`log_interval`).
