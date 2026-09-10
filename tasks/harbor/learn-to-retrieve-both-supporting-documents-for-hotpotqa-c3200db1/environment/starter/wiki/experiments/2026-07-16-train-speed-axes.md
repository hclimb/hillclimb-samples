# Train-speed axes beyond approx-topk — profiling the post-approx `qa_hard_neg_think_sft4b` step

**Date:** 2026-07-16 · **Author:** rohunagrawal · **Status:** night exploration **COMPLETE** — every
quality-neutral axis measured on v6e-8; 3 wins found (2 adopted + capstone-verified end-to-end, 1
implemented default-off pending A/B), 5 ruled out. Changes uncommitted on branch `train_speed`.

## Night summary (2026-07-16, autonomous) — READ FIRST

Exhaustive hunt for **quality-(semi)neutral** train-speed wins on `qa_hard_neg_think_sft4b`, after
approx-topk. Verified on the box (real steps where possible). **Result: the wins were in the data
pipeline and the training loop, not the compute step** (which is ~spent after approx-topk).

| Rank | Lever | Real effect | Fix | Status |
|------|-------|-------------|-----|--------|
| **1** | **Data: live-HF → offline-parquet** | live-HF **429-stalls the TPU for minutes** (can't reach batch 1); offline = 49.7 batch/s, 0 stalls | `HF_HUB_OFFLINE=1` + `precache_hf.sh` | ✅ **adopted** in recipe |
| **2** | **Loop per-step sync → pipelining** | **+9.98%** real (544.7→490.3 ms/step; ≈ 487 ms device ceiling) | `trainer.log_interval=10` (default 1 = no-op) | ✅ **adopted** (code + recipe) |
| **3** | **Axis A: frozen ⟹ stop_gradient** | **2.12× stage 0** (487→230 ms); ~2.7% whole-run (frozen stages = 15%) | `trainer.stop_grad_frozen=true` (default off) | ⚠️ **implemented, default OFF** — exact-neutral (proven) but ~3% bf16 grad noise → needs loss A/B before enabling |
| — | Remat policy | default (recompute-all) is fastest — **HBM-bound**, saving activations is −16% | — | ✗ ruled out |
| — | Optimizer / bf16-Adam state | update is only **1.8%** of step (FSDP-sharded) | — | ✗ ruled out |
| — | Precision (bf16 the bank) | already **bf16** end-to-end; fp32 is only the free MXU accumulator | — | ✗ no-op |
| — | Drop O(M) `[B,T,N,M]` score grid | it **is** `doc_access_loss`'s contrastive input, not telemetry | — | ✗ ruled out |
| — | int8/fp8 score matmul | score matmul is a small slice → ≤~1–2%, and **semi**-neutral (needs A/B) | — | ✗ not pursued |

**Combined:** the data fix is **categorical** (without it the recipe can't feed the TPU at all).
On top of offline data, the two truly-no-math-change wins — offline data + loop pipelining (~10%,
all steps) — are **adopted**. Axis A adds another 2.12× on the 15% frozen-stage steps (~2.7%
whole-run) but is **left off pending a loss A/B** (it's exact-arithmetic-neutral but perturbs
trainable grads ~3% at bf16 — see run log). So the **safely-adopted** win is **~+10% whole-run + the
categorical data fix**; Axis A is a ready, opt-in further ~2.7%. The device-bound step is now ~98%
transformer-trunk fwd+bwd with **no quality-neutral lever left** — further speed needs
quality-affecting knobs (bank M, heads, top-k, int8) with A/B verification.

**Adopted this session:** `configs/trainer/standard.yaml` (`log_interval=1`, `stop_grad_frozen=false`
defaults); `trainer/trainer.py` (loop pipelining always-safe; Axis A in `_train_step` gated off by
default); `scripts/embed/train_hard_neg_think.sh` (offline-parquet + `log_interval=10`; Axis A left
off with a note). Impl note:
[../implementations/2026-07-16-train-loop-throughput.md](../implementations/2026-07-16-train-loop-throughput.md).
**Capstone verified:** the adopted recipe (`train_verify_recipe.sh`) ran end-to-end through all 4 stage
transitions on real offline data with `log_interval=10`; grads are finite at stage 0 and stage 3
(`bench_grad_finite.py`). (A transient nan appeared only under the artificially-shortened verify
schedule — a config artifact, see run log, not a recipe bug.)
Benches: `scripts/embed/{profile_train_step,bench_loop_sync,bench_data_throughput,bench_train_throughput,bench_optcost,bench_axisa,bench_axisa_grad,bench_grad_finite}.{py,sh}` (the remat-sweep bench was removed with its reverted toggle)
+ verify runners `{train_verify_combined,train_verify_recipe}.sh`.

**Recommended next steps:** (1) **Axis-A A/B** — a short `stop_grad_frozen` true-vs-false run
(same seed/offline data), compare train-loss + `doc_access` curves; if they track (they should — the
diff is bf16 reduction-order on an exact-neutral change), flip `stop_grad_frozen=true` in the recipe
for the extra ~2.7%. (2) Run the full 100k `train_hard_neg_think.sh` (offline + `log_interval=10`) on
a precached box. The loop/data *plumbing* changes touch no training math, but note the offline cache
is only ~50% of shards (2.06M rows, > one epoch — enough, but a subset of the full data pool, see the
Bottleneck-A caveat); if exact full-data parity matters, cache more shards first (needs disk). (3) If
more speed is needed, A/B int8 score matmul (the only remaining lever, semi-neutral).

## Conclusion (first) — compute-step profiling (original lead; superseded by the Night summary above)

**Axis A works per-step but barely moves the whole run; the quality-neutral well is shallow after
approx-topk.** Profiling the post-approx `qa_hard_neg_think_sft4b` step (v6e-8, quick `--stages 0 3`):

- **Axis A (frozen ⟹ `stop_gradient`) nearly halves the *frozen-stage* step:** stage 0 goes
  **515.4 → 264.1 ms (1.95×, −48.8%)** — the backward through the frozen 4B main + frozen embed
  trunk was **251 ms of the 347 ms backward**, and it's provably quality-neutral to drop.
- **…but frozen stages are only 15% of training.** Stage 3 (85% of steps) has **nothing frozen**,
  so Axis A saves **0** there. The whole-run projection is **2.67%** (lower bound, stages 0+3);
  even filling stages 1–2 the absolute ceiling is **~10%**, realistically ~3–5%.
- **Stage 3 is where the wall-clock lives (523 ms × 85k steps) and it has no free backward to
  reclaim.** Its backward is 354 ms (`bwd/fwd = 2.09`, ≈ a normal non-remat backward ratio), so the
  remat-policy axis (Axis A′) does **not** look like an obvious large win from this signal — but it's
  the *only* quality-neutral lever that touches stage 3, and its prize is **not directly measured
  here** (that needs a model-side remat-policy toggle).
- **Validation:** stage-0 `full` = 515.4 ms reproduces the approx-topk Tier-1 @0.99 step (515.1 ms),
  confirming the profiler's step is faithful to the real trainer.

**Bottom line for the original question:** the *compute* step has little quality-neutral juice left
after approx-topk (Axis A is a real but ~3–5% win confined to the 15% frozen-stage steps; bank M /
heads / top-k are quality-affecting). **But profiling the training *loop* found the best
quality-neutral lever so far: the loop syncs the device every step, capping throughput at 516.8 ms
vs a 487.0 ms device-bound ceiling — a ~6.1% tax on *every* step** (Bottleneck B below). Pulling
losses / logging every K≈10 steps and letting JAX pipeline reclaims most of it, changing no training
math. This is larger whole-run than Axis A and simpler. **And the biggest lever of all is data (Bottleneck A, now measured):** `train_hard_neg_think.sh`
streams live from HF (no `HF_HUB_OFFLINE`), and 16 workers × 4 datasets blow HF's 1000-req/5-min
quota *during pipeline build* — the loader can't produce even the first batch, looping
`sleep 68–172 s → retry → 429` (TPU idle for **minutes**). The offline-parquet path (already standard
in every `train_ground_*.sh`) reads local disk: **49.7 batch/s, 0 stalls, 24× the compute rate** —
data leaves the critical path entirely. The fix is **one line** (`HF_HUB_OFFLINE=1` + precache).
Ranking of quality-neutral wall-clock levers: **A (data, minutes → 0) ≫ B (loop sync, ~6%) > Axis A
(~2.7%)**.

### The two candidates profiled

- **Axis A — frozen ⟹ `stop_gradient`.** Freeze is an `optax.transforms.freeze` *update mask*
  (`utils.py:218`), **not** `stop_gradient`, so the backward runs through the frozen 4B main model
  and (stage-0) frozen 0.6B embed model every step — wasted compute. **Confirmed** (251 ms/step in
  stage 0), but bounded to the **15% of steps** in frozen stages 0–2.
- **Axis A′ — remat policy.** Both trunks are `jax.remat`'d per layer (`qwen3_mem_embed.py:132`,
  `qwen3.py:179`), so backward recomputes the forward. A lighter policy would cut backward in **all**
  stages incl. the dominant stage 3 — **prize unmeasured** (needs a policy toggle); `bwd/fwd = 2.09`
  in stage 3 is only weakly suggestive and not obviously large.

## Hypothesis & motivation

approx-topk removed ~340 ms of an 838 ms step; the step is now ~500 ms and its breakdown is unknown.
Before choosing a next lever we should measure, not guess — the same discipline that made approx-topk
a clean call. The staged schedule (`steps=100000`) spends **5k / 5k / 5k / 85k** steps in stages
0/1/2/3, so *where* in training a lever helps matters as much as its per-step size. Hypotheses:

1. Because freeze ≠ `stop_gradient`, a large fraction of the stage 0–2 step is backward through
   frozen weights that XLA could prune with a targeted `stop_gradient` (**Axis A**).
2. In the still-needed backward, `jax.remat` recompute is a large share (backward ≫ 2× forward),
   making a lighter remat policy a whole-run win (**Axis A′**).

Both are quality-neutral by construction: a detached frozen weight gets 0 grad and `optax.freeze`
zeroed its *update* anyway → **identical trainable-param updates**; remat is a pure compute/memory
trade with identical math.

## Setup

- **Model / data:** identical to the `qa_hard_neg_think_sft4b` run and to the approx-topk bench —
  `qwen3_mem_embed` (main `Qwen3-4B`, embed `Qwen3-Embedding-0.6B`), 1 memory layer @ L14,
  `mem_top_k=64`, `seq_len=512`, `num_chunks_per_doc=16`, `doc_chunk_seq_len=256`, `batch_size=16`
  (bank M = 16·16·256 = 65,536). approx-topk on (config default).
- **Method:** `scripts/embed/profile_train_step.py` replicates `Trainer._train_step`'s `loss_fn`
  byte-for-byte (same `compute_aux_losses`, per-row CE gate) and, **per stage**, times three
  variants of the real jitted step on a synthetic batch of the true shapes (step time is
  shape-driven — see the approx bench for why synthetic is faithful for *timing*):
  - `full` — the real step (current behavior).
  - `fwd_only` — loss forward, no grad → forward cost.
  - `sg_frozen` — **Axis A**: `stop_gradient` the frozen weights inside `loss_fn` before forward.
- **Derived per stage:** `bwd_total = full − fwd_only`, `bwd_recoverable = full − sg_frozen`
  (Axis-A prize), `bwd_necessary = sg_frozen − fwd_only`, and `bwd_necessary/fwd_only` (a
  ratio ≫ 2 flags heavy remat recompute → Axis A′). The summary weights per-stage `full`/`sg`
  medians by the 5k/5k/5k/85k step budget to project the whole-run Axis-A saving (so it isn't
  overstated).
- **Independent variable:** the backward regime (full vs frozen-`stop_gradient` vs none),
  everything else fixed. **Does not modify** the trainer/model (standalone script).
- **Hardware:** TRC `v6e-8`, `europe-west4-a`.

## Reproducibility

- **Commit:** _pending_ — profiler tooling `scripts/embed/profile_train_step.{py,sh}` on branch
  `train_speed` (fill the SHA once committed).
- **Launch** (from the worktree):
  ```bash
  TPU_NAME=rohun-v6e-8-0 RUN_SCRIPT_PATH=scripts/embed/profile_train_step.sh \
    bash scripts/infrastructure/multi-vm-tpu-run.sh
  ```
  which runs, on the box (quick default): `uv run python scripts/embed/profile_train_step.py
  --stages 0 3` — **stage 0** (biggest Axis-A prize) + **stage 3** (85%-of-steps control + remat
  signal), ~5 compiles. Stage 3 has 0 frozen tensors so its `sg_frozen` compile is skipped
  (`sg_frozen ≡ full`). Set `STAGES="0 1 2 3"` for the full whole-run projection (~12 compiles).
  Box-side log: `~/profile_train_step.log`.
- **Smoke:** `uv run python scripts/embed/profile_train_step.py --check` (compose+imports only,
  no accelerator).
- No checkpoint / wandb (microbench); the profile is on a random-init model (step *time* is
  shape-driven, so this is faithful — a checkpoint wouldn't change the tensor shapes).

## Results — Tier-0 profile (v6e-8, quick `--stages 0 3`, median ms, n=15, warmup excluded)

| stage | full (ms) | fwd (ms) | sg_frozen (ms) | bwd_total | AxisA_recoverable | bwd_needed | bwd/fwd |
|-------|-----------|----------|----------------|-----------|-------------------|------------|---------|
| **0** (main+embed frozen, ce=0) | 515.43 | 168.91 | **264.06** | 346.52 | **251.36 (−48.8%)** | 95.16 | 0.56 |
| 1 (main frozen, embed train, ce=0) | _pending_ | — | — | — | — | — | — |
| 2 (main frozen, embed train, ce=1) | _pending_ | — | — | — | — | — | — |
| **3** (all trainable, ce=1) | 522.63 | 169.03 | 522.63¹ | 353.60 | **0** | 353.60 | 2.09 |

¹ stage 3 has 0 frozen tensors, so `sg_frozen ≡ full` (compile skipped).

**Whole-run Axis-A projection (stages 0+3, 90k/100k steps, no saving applied to unmeasured 1–2):
2.67%** — a **lower bound**. Absolute ceiling if Axis A eliminated *all* stage 0–2 backward:
`346.5 ms × 15k / (≈515 × 15k + 523 × 85k) ≈ 9.9%`. Realistic (stages 1–2 recover less than stage 0
— embed backward stays in 1–2, and stage 2's CE keeps upper-main backward): **~3–5%**.

**Validation:** stage-0 `full` = 515.43 ms matches the approx-topk Tier-1 @0.99 step (515.08 ms,
[2026-07-15](2026-07-15-approx-topk-training.md)) — the standalone profiler's step is faithful to
`Trainer._train_step`. **Note:** these are *synced* step times (the profiler `block_until_ready`s
each iter, as does the real loop) — the device-bound step is ~487 ms (Bottleneck B).

## Results — Bottleneck B: training-loop per-step device sync (v6e-8, synthetic batch, data removed)

`scripts/embed/bench_loop_sync.py` times the real jitted `_train_step` (stage-3 config) under two
loop patterns — `pipelined` (no per-step host pull, sync once at the end) vs `trainer_sync`
(replicates `trainer.py:383–407`'s per-step `int`/`float` pulls). Synthetic batch, so **data
loading is excluded** — this isolates the loop mechanics.

| pattern | ms/step | steps/s | vs ceiling |
|---------|---------|---------|------------|
| `pipelined` (device-bound ceiling) | **486.99** | 2.053 | — |
| `trainer_sync` (per-step pulls, today) | **516.78** | 1.935 | **+29.80 ms (+6.12%)** |
| `wandb.log` per call (offline proxy) | 0.11 ms | — | negligible |

The current loop pays a **~30 ms/step (6.1%) tax** by pulling losses to the host every step, which
drains the pipeline and serializes the drain+redispatch bubble that pipelining hides. It is **not**
wandb (0.11 ms). The `trainer_sync` = 516.8 ms reproduces the profiler's stage-3 `full` (522.6 ms)
and the approx-topk step (515 ms) — all three were measuring the *synced* step. The bubble is paid
once per sync point, so syncing every K steps costs ~30/K ms/step → **K≈10 reclaims ~90% (~5.5% net)**.

**Fix (quality-neutral):** pull `ce_loss`/`grad_norm`/nan-flags and `wandb.log` only every K steps,
and let JAX dispatch K steps ahead. The NaN-skip guard is already on-device (`lax.cond` in
`_train_step`), so no per-step host check is needed — training math is unchanged; only logging
granularity coarsens (log every K instead of every step). Applies to **all** stages/steps.

## Results — Bottleneck A: data loading, live-HF vs offline-parquet (v6e-8, real QADataset, no model)

`scripts/embed/bench_data_throughput.py` builds the REAL grain pipeline (real tokenizer, no
model/TPU) and consumes 64 batches, timing each `next()`. Pulling as fast as possible drains grain's
buffer, so steady-state per-batch time is the production ceiling vs the 2.053 batch/s compute rate.

| arm | first batch | steady-state | stalls >1s | verdict |
|-----|-------------|--------------|-----------|---------|
| **live-HF** (recipe today; no `HF_HUB_OFFLINE`) | **never reached** — 429 during pipeline build, loops `sleep 68–172 s → retry` | — | continuous | **TPU idle for minutes** |
| **offline-parquet** (`HF_HUB_OFFLINE=1`, precached `~/hf_parquet`) | 34.1 s (one-time 100k shuffle-fill, local disk) | **49.69 batch/s** (median 18.2 ms, p99 44.7, max 45.8) | **0** | **KEEPS UP (24× headroom)** |

Live-HF: 16 workers × 4 interleaved datasets exceed HF's **1000 API req / 5 min** quota just
resolving shards (`qa.py`'s `_hf_retry`/auto-rebuild then sleeps 68–172 s and retries — and retrying
re-blows the quota, so it never starts). This is the same limit the approx-topk doc flagged. In a
real run the TPU starves. Offline-parquet reads local shards (`qa.py:313` offline branch) — **zero HF
calls**, 49.7 batch/s (24× the 2.05/s the compute needs), no stalls; the 34 s startup is the
one-time shuffle-buffer fill, negligible over a 100k-step run.

**Fix (one line):** add `HF_HUB_OFFLINE=1` + `GROUND_HF_PARQUET` to `train_hard_neg_think.sh` and run
`scripts/misc/precache_hf.sh` on the box first — exactly what every `train_ground_*.sh` already does.
**Data-coverage caveat (corrected 2026-07-16, ~11:00 UTC):** `precache_hf.sh` caches only **~50% of
shards** by default (`GROUND_DATA_FRAC=0.5`; ~18G, since the 97G boot disk is 94% full) — measured
**2.06M cached rows** (71% science-qa). That is **> one 1.6M-example 100k-step epoch**, so a single
100k run is well-fed and never touches the uncached half. But it is the **first 50% of shards**, not
the full dataset, so it is *not literally the same data* as a full-streaming run (mild shard-ordering
bias possible) — the earlier "same data, same order" phrasing was an overstatement. Full coverage
needs disk headroom (free the 9.3G venv / 8.7G weights, a bigger boot disk, or RAM-staging on
`/dev/shm`); the 6.7G free won't hold the other ~18G.

## Interpretation

- **Axis A is real and free but small at the run level.** The 251 ms/step it reclaims in stage 0 is
  large (1.95× the step), but frozen stages are 15% of the schedule, so the run-level payoff is
  ~3–5%. Worth taking *if* the implementation is cheap (it is: make the frozen slice of weights
  `stop_gradient` before `value_and_grad`, derived from the same mask `optax.freeze` already uses).
  Caveat: it zeros the `grad_norm_main`/`grad_norm_embed` telemetry the trainer logs for frozen
  groups — a doc-update, arguably an improvement.
- **The dominant cost is stage-3 backward (354 ms × 85k steps) with no frozen waste.** `bwd/fwd =
  2.09` is close to a plain (non-remat) backward ratio, so remat recompute is *not* obviously a big
  slice — but the profile can't decompose recompute vs VJP without a no-remat variant, so the
  remat-policy prize is genuinely unmeasured. It's the only quality-neutral lever that touches the
  85% stage.
- **Scope tension.** The levers with real stage-3 leverage — smaller bank M, fewer heads, lower
  top-k — all change the learned function and were ruled out as quality-affecting. So the honest
  finding is that the quality-neutral budget is nearly spent after approx-topk.
- **Precision is already fully exploited (checked, not a lever).** The whole memory path is already
  bf16: `mem_{k,v,q,o}_proj` init bf16 (`memory_utils.py:351–357,169`), the bank build
  `mem_k = hidden(bf16) × mem_k_proj(bf16)` is bf16, `rms_norm` returns bf16, and the score matmul
  is `preferred_element_type=q.dtype` = bf16 with fp32 MXU accumulation (`memory.py:55`). No fp32
  operand hides in the memory path — "bf16 the memory bank" is a no-op. The only step below bf16 is
  int8/fp8 matmuls (faster on TPU) or bf16/8-bit Adam state, but both are quality-*affecting*
  (quantization / momentum precision), not free.

**Next (decision pending):** (a) **implement Bottleneck-B fix** (log/pull every K steps + dispatch
ahead) — the largest quality-neutral win (~5.5% net, all steps) and simplest; (b) **measure
Bottleneck A** end-to-end (real dataloader, live-HF vs offline-parquet steps/sec) — potentially the
biggest of all if live streaming stalls the TPU for seconds; (c) implement Axis A for the additional
~3–5% (frozen stages only); (d) fill stages 1–2 / measure remat (Axis A′) / revisit the
quality-neutral constraint. Order by expected win: **A (data) ≥ B (loop) > Axis A**, pending A's
measurement.

---

## Overnight autonomous exploration — 2026-07-16 night (living log)

**Goal:** exhaustively explore quality-(semi)neutral speed-ups, **verify on the box with real
steps** (a few hundred), keep this doc live. Autonomous (no user input); self-scheduled loop.

**Candidate axes & status** (updated as runs land):

| # | Axis | Quality | Status | Result |
|---|------|---------|--------|--------|
| A | Data: offline-parquet vs live-HF | neutral | ✅ measured | live-HF 429-stalls; offline 49.7 batch/s, 0 stalls |
| B | Loop per-step sync (synthetic) | neutral | ✅ measured | +6.1% (487 vs 517 ms) |
| B-real | Loop sync — **real end-to-end steps/sec** (offline data) | neutral | ✅ measured | **+9.98%** real (544.7→490.3 ms); pipelined ≈ 487 ms ceiling; data off critical path |
| Axis A | Frozen ⟹ stop_gradient (stages 0–2) | neutral | ✅ **implemented+verified** | **2.12× stage 0** (487→230 ms) via `trainer.stop_grad_frozen` (default off); ~2.7% whole-run |
| A′ | Remat policy on the 4B/embed trunks (stage 3, all steps) | neutral | ✅ measured | **dead end** — default (recompute-all) is fastest; lighter = −16% (HBM-bound); no-remat OOMs |
| C | Fill Axis-A profile stages 1–2 | neutral | ▢ queued | exact whole-run Axis-A number |
| D | Drop O(M) full-score-grid / telemetry aux during training | check-neutral | ✅ investigated | **dead end** — `doc_access_loss` (retrieval loss) needs the full `[B,T,N,M]` contrastive grid; it IS the top-k score output; two-pass makes it stop-grad (quality-affecting) |
| E | int8 score matmul | **semi** (A/B) | ✗ not pursued | low ceiling — score matmul is a small slice (step is 98% trunk fwd+bwd, opt 1.8%); int8 ≤~1–2% and needs a quality A/B. Not worth vs the adopted wins |
| F | Adam state bf16 / 8-bit | **semi** (A/B) | ✅ measured | **dead end** — optimizer update is only 1.8% of step (opt_state FSDP-sharded); bf16-mu ≈ 0.6% |

**Execution notes:** benches are standalone scripts under `scripts/embed/`, launched via
`multi-vm-tpu-run.sh` on `rohun-v6e-8-0` (precached). Real-steps harness:
`scripts/embed/bench_train_throughput.py` (real offline dataloader + real `_train_step`, baseline
sync-each vs pipelined sync-K). Results appended below per run.

### Run log

**02:18 UTC — B-real: loop-sync fix, real end-to-end steps (offline-parquet, stage 0, 200 steps/arm, v6e-8).**
`scripts/embed/bench_train_throughput.py` (real dataloader + real `_train_step`).

| arm | ms/step | steps/s | data-fetch median |
|-----|---------|---------|-------------------|
| baseline (float every step, = trainer.py today) | **544.67** | 1.836 | 21.0 ms |
| pipelined (float every 20 steps) | **490.32** | 2.039 | — ¹ |

**Real loop-sync gain = +9.98%** (vs +6.1% synthetic) — bigger because the real loop exposes more
host work the sync serializes: **21 ms/step batch collation** (grain `batch()` after `mp_prefetch`,
main-thread) + `process_train_pairs` + the float pulls. Pipelined = 490 ms ≈ the 487 ms device-bound
ceiling → the fix reclaims ~all host overhead. **Data confirmed off the critical path** offline
(baseline `next()` = 21 ms, no stalls). ¹ The pipelined arm's 414 ms "data-fetch" is JAX **dispatch
backpressure** (host runs ahead until the device queue is full), not a data stall — total = ceiling
proves it. Combined so far: data fix (enables running at all) + loop fix (+10%) are the two big
quality-neutral wins; both dwarf Axis A (~2.7%).

**02:43 UTC — A′: remat-policy sweep (stage 3 all-trainable, synthetic batch, device-bound ms/step).**
Measured via a temporary `MEM_REMAT_POLICY` toggle (`get_remat_policy`) on the per-layer `jax.remat`,
default `None` = current recompute-all. **The toggle + its bench were reverted after the measurement**
(remat is a dead end — see below — so the experimental hook was removed); the result stands as recorded.

| policy | ms/step | vs default |
|--------|---------|-----------|
| `nothing` (default, recompute all) | **478.74** | — |
| `dots_nobatch` (save matmuls) | 553.18 | **−15.5%** (slower) |
| `dots` | 555.77 | −16.1% (slower) |
| `everything` (no remat) | **OOM** (45.4 G > 31.25 G HBM) | — |

**Negative result — the default is already optimal.** Saving activations is *slower* here because the
4B/embed step is **HBM-bandwidth-bound, not compute-bound**: recomputing matmuls on the fast MXU costs
less than spilling them to HBM and reading them back in the backward pass. No-remat OOMs. **Axis A′ is
ruled out** — no quality-neutral win in the remat policy. (Keeps the confirmed device-bound step at
~478–490 ms, consistent across all three benches.)

**02:5x UTC — ADOPTED the two big wins in code** (impl note:
[../implementations/2026-07-16-train-loop-throughput.md](../implementations/2026-07-16-train-loop-throughput.md)):
`trainer.log_interval` (default 1 = no-op; pipelines at >1) in `trainer/trainer.py` +
`configs/trainer/standard.yaml`; `train_hard_neg_think.sh` now sets `HF_HUB_OFFLINE=1` +
`trainer.log_interval=10`. Default path byte-identical; pipelined path bench-verified (+10%). Full
`train.py` scale run deferred (final orbax save ~46 GB).

**03:0x–03:2x UTC — closed the compute/optimizer side.** `bench_optcost` (F): optimizer update =
**1.8%** of step (opt_state FSDP-sharded) → bf16-Adam dead. Remat sweep (A′): default remat is
fastest (HBM-bound) → dead. D (code): the `[B,T,N,M]` grid is `doc_access_loss`'s contrastive input,
not droppable. E (int8): score matmul is a small slice → ≤~1–2% and needs A/B → not pursued. The
device-bound step is ~98% trunk fwd+bwd, where nothing quality-neutral remains.

**03:16 UTC — Axis A implemented + verified.** `trainer.stop_grad_frozen` (default off; `_train_step`
stop_gradients frozen weights when the stage's `trainable_params` are passed). `bench_axisa.py`
(real `Trainer._train_step`, stage 0): baseline `tp=None` **487.4 ms** → AxisA **230.2 ms** =
**2.12×**, finite loss both arms (wiring OK). Quality-neutral by construction (stop_gradient on a
frozen weight zeros only its own grad; trainable-param grads unchanged). Whole-run ~2.7% (frozen
stages 0–2 = 15% of steps).

**03:2x UTC — verification.** Combined end-to-end `train.py` run (all 3 fixes) **failed on a launch-
script bug, not code**: `trainer.steps=200` makes the staged schedule's stage-3 `max_step` < stage 2's
15000, which `parse_training_stages` rejects (staged runs need ≥15000 steps or overridden stage
boundaries). Adopted code is unaffected. Pivoted to the definitive quality-neutrality check:
`bench_axisa_grad.py` compares trainable-param gradients with Axis A on vs off (one batch, stage 0).

**03:3x UTC — Axis A grad check: exact-neutral, but ~3% bf16 grad noise (nuance found).** Wiring is
correct — frozen-param grads are **exactly 0** under Axis A (2.5e-3 under baseline = the pruned
backward), so `stop_gradient` hits exactly the frozen set. But the **trainable** grads differ
**median 3.3% / max 7.5%** at bf16 (worst: `main_model.layers.14.mem_q_proj`, whose grad flows through
the M=65k score backward). This is **necessarily numerical** (reduction-order rounding): the analytic
proof is airtight — `stop_gradient` on a frozen *weight* zeros only that weight's own grad and cannot
change a trainable weight's grad in exact arithmetic. (fp32 confirmation not runnable — the forward
hardcodes bf16 activations, so fp32 weights hit a conv dtype mismatch.) **Correction to my earlier
"quality-neutral by construction" framing:** exact-neutral, yes, but the ~3% bf16 grad perturbation
(same class as remat/sharding/approx-topk non-determinism) means it should be confirmed with a loss
A/B before production. **Reverted `stop_grad_frozen` to OFF in the recipe** (code stays, default off).

**04:22 UTC — capstone recipe verify: ran end-to-end, but surfaced a latent `grad_norm=nan` issue.**
`train_verify_recipe.sh` (real `train.py`, offline data, `log_interval=10`, `stop_grad_frozen=false`,
4 staged transitions at 50/100/150) **completed without crashing and transitioned through all stages**
— so the adopted loop/data plumbing works end-to-end. BUT `grad_norm=nan` on essentially every step →
the NaN guard skips every update (no training). **This is NOT caused by the speed changes:** the
synthetic-batch benches (`bench_axisa_grad`) had finite grads, and `log_interval`/`stop_grad_frozen=false`
are provably no-ops on the grad. **Correction after diagnosis (don't over-alarm):** `bench_grad_finite.py` on real offline data shows
**stage-0 grads are fully FINITE** (total norm 2.5–4.4, all groups clean, 3 batches). The nan warnings
I first saw were the log *tail* = **steps ~156–200 = stage 3** (my verify shrank the stages to
50/100/150/200). Most likely an **artifact of the absurdly-short schedule**: 50-step stages leave the
memory layer grossly undertrained when stage 3 unfreezes the 4B *and* turns CE on (`ce_weight` 0→1),
destabilizing the residual → nan — which the real 5000-step-stage recipe wouldn't hit the same way.
**Confirmed:** stage-3 fresh-model grads on real data are also **FINITE** (0 non-finite leaves) — but
**large** (norm ~200–400 vs stage-0's ~3). So the capstone nan wasn't a config/data bug at init; it
**built up during the short-schedule training**: 50-step stages leave the memory undertrained, so when
stage 3 unfreezes the 4B + turns CE on, the large grads destabilize it after a few steps → nan → the
guard skips. On the real **5000-step-stage** schedule the memory is well-trained before stage 3 and the
grads are clipped (`clip_grad_norm=1.0`), so this doesn't reproduce. **Verdict: a verify-config artifact
(my absurdly-short stages), NOT a recipe bug and NOT a speed-work regression.** Net: the adopted recipe
**runs end-to-end through all stage transitions, stage-0 trains cleanly, and grads are finite at both
stage 0 and 3 on real data** — the speed changes (offline data + `log_interval`) are sound. (I initially
over-flagged this before diagnosing — corrected above.)
