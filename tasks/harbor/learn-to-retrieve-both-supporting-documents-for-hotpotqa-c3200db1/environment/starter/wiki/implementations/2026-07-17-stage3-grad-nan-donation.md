# Stage-3 grad NaN: `donate_argnums` triggers it; donation is now off by default

**Date:** 2026-07-17
**Status:** fix landed, verified at the failing checkpoint
**Files:** `trainer/trainer.py`, `configs/trainer/standard.yaml`,
`scripts/embed/debug_donate_ab.sh`, `configs/trainer/staged_debug_{stage2,stage3,all}.yaml`

## Answer first

`jax.jit(..., donate_argnums=(2, 3))` on `Trainer._train_step` makes stage 3 produce **NaN
gradients with a perfectly finite loss**, which deadlocks the run. Donation is now **off by
default** (`MEM_DONATE=none`), which restores the behaviour of the last known-good run
(`707a128`, April) and is verified clean at the exact checkpoint that failed.

Donation is the **trigger, not the root cause**. A latent numerical hazard in the backward
survives this fix; it only fires once `main_model` is trainable, and donation perturbs the
compiled backward enough to tip it. See [Known-remaining risk](#known-remaining-risk).

## Motivation

Run `qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-07-16-17-55-17`
([wandb](https://wandb.ai/memory-layers/memory-layers)) entered stage 3 (index 3 of
`configs/trainer/staged.yaml`, `main_model` unfrozen) at step 15000 and immediately began
reporting `grad_norm_main = NaN` while the loss stayed healthy at ~0.8. It then burned ~3900
steps making **zero** progress before being killed at 18888.

The deadlock mechanism: the NaN guard's `skip_fn` returns weights **and** `opt_state`
unchanged, so Adam's `count` never advances, so the LR stays pinned at the warmup's
`init_value=1e-8`. The observed `train/lr = 9.99e-9` is exactly `float32(1e-8)` — proof that
`count` was 0 and **not one stage-3 step ever succeeded**. Weights were therefore frozen from
step 15000 on, which means **ckpt 16000 and 18000 are bit-identical to W₁₅₀₀₀** — the failing
weight point is saved and reproducible.

A near-identical config had run cleanly in April (`707a128`), so the cause had to be in the
diff.

## Investigation: what was ruled out, and why

| Suspect | Verdict | Evidence |
|---|---|---|
| `mem_approx_topk` (`approx_max_k` vs exact `top_k`) | **cleared** | exact `top_k` gave identical 24/60 NaN |
| `trainer.stop_grad_frozen` / Axis A `stop_gradient` | **cleared** | `false` in the run's saved `.hydra/config.yaml`; gating is correct; and stages 0–2 showed `grad_norm_main = 4.031`, which would be ~0 if the 4B were detached |
| `use_lora` (only stage-dependent field in the rebuilt `forward`) | **cleared** | written in `utils.py`, read **nowhere** in `models/` — dead |
| `optax.transforms.freeze` mask | **cleared** | unchanged since April (context lines only in `git diff 707a128 -- utils.py`) |
| Stage-transition machinery | **cleared** | byte-identical to April |
| `warmup_frac` now parsed (stage 3 LR: peak → warmup) | real change, not the cause | it is *gentler*; explains the pinned 1e-8, not a NaN gradient |
| LR / weight decay / clip | identical to April | 1e-4 / 0.1 / 1.0 in both saved configs |
| The 4B diverging from too-hot updates | **cleared** | loss never exploded (0.70–0.93 throughout); and the real run NaN'd at LR=1e-8 *before any update landed* |

## The measurement that localised it

All arms at **ckpt 16000** (= W₁₅₀₀₀, the exact failing weights), **LR pinned to 1e-12** so the
weights cannot move (~1e-7 total drift over 20 steps), identical batches:

| arm | freeze on main | main updated? | donation | NaN |
|---|---|---|---|---|
| stage 2 | yes | **no** | `(2,3)` | **0/20** |
| stage 3 | all-False mask | yes | `(2,3)` | 7/20 |
| `["all"]` (no freeze wrapper, no `forward` rebuild) | — | yes | `(2,3)` | 7/20 |
| stage 3 | — | yes | **off** | **0/20** |

Then the decisive A/B/C — **one box, back to back**, so the data is identical
(`scripts/embed/debug_donate_ab.sh`):

| `MEM_DONATE` | `donate_argnums` | NaN |
|---|---|---|
| `both` | `(2,3)` | 5/20 |
| `opt` | `(3,)` — opt_state only | **5/20** |
| `none` | `()` | **0/20** |

## Why the obvious explanations are wrong

**It is not weights-buffer corruption.** `MEM_DONATE=opt` donates *only* `opt_state`, which the
backward never reads — yet it NaNs just as much as `both`. The initially attractive story
(donation lets XLA overwrite `w` while the `mem_value_read_kchunk` remat scans still need it for
the recomputed forward) predicts `opt` would be clean. It isn't.

**What donation actually does is perturb the numerics.** Across modes, the same weights and the
same batch give losses differing in the 5th decimal (`0.8455632` vs `0.8455887`). Donation changes
XLA's global buffer assignment and therefore the fusion of the backward. The backward is sitting
close enough to a numerical cliff that this is enough to push ~25% of batches over it.

**`trainable_params` never touched the math.** Every "this cannot affect the gradient" argument
was correct: `grads = f(weights, batch, forward)`, the guard reads *raw* grads via
`optax.global_norm(grads)` before `optimizer.update` runs, and `stop_grad_frozen` is off. What
`trainable_params` changes is the `optimizer` — a `jit` **static arg** — so each stage compiles a
*different* `_train_step`. It reached the **compilation**, not the math. Stage 2 is clean not
because freeze hid a NaN but because freeze zeroes main's updates, so `apply_updates(w, 0) == w`
and XLA never writes that buffer at all.

## Options considered

| Option | Verdict |
|---|---|
| Donate `opt_state` only (`(3,)`) | **rejected** — measured 5/20 NaN, no better than `both` |
| Keep donation, add a NaN-retry/loss-scale | rejected — papers over a backward that produces NaN on ~25% of batches; the other ~75% are then suspect too |
| Find and fix the underlying hazard in `models/memory.py` | right long-term fix, but `memory.py` is +464 lines since April; not an overnight job, and the run is blocked now |
| **Donation off by default** | **chosen** — restores April behaviour, verified clean, costs the HBM saving donation bought |

## Approach

`trainer/trainer.py` gained a `MEM_DONATE` env knob resolving to `donate_argnums`:

```python
_DONATE_MODE = os.environ.get("MEM_DONATE", "none").lower()
_DONATE = {"both": (2, 3), "opt": (3,), "none": ()}.get(_DONATE_MODE, ())
```

Default `none`. `opt`/`both` are retained **only** so the repro stays runnable; both are known bad
and labelled as such in the source comment. An unrecognised value falls back to `()` (safe).

### HBM cost

The docstring claimed donation "halves their HBM footprint" for weights + opt_state. Turning it
off gives that back. It is affordable at this config: the `MEM_DONATE=none` arm ran the real
bs16 / seq512 / chunks16 shape on a v6e-8 without OOM and saved a checkpoint. Throughput cost was
not separately measured — worth an entry in the train-speed experiment log if it matters.

### Secondary fix: the silent-deadlock tripwire

The run looked alive for ~3900 steps (tqdm ticking, loss printing ~0.8) while doing nothing. Two
bugs made that possible:

1. No abort on persistent NaN. Added `trainer.nan_abort_after_samples` (default 50) — abort after
   that many *consecutive* non-finite samples. Checked at the `log_interval` cadence, not per
   step, because reading `loss_nan`/`grad_nan` forces a device sync and the loop is deliberately
   pipelined to avoid one per step. `0` disables.
2. `loss_nan_count` / `grad_nan_count` were incremented **inside** the `log_interval` gate, so at
   `log_interval=10` they under-reported by ~10x. This is why the run looked like "~93% of steps
   skipped" when it was really ~100%. The counters remain sampled (a per-step sync would cost
   throughput) but the tripwire now makes a true deadlock fail loudly instead.

## Test record

Repro scripts are real `train.py` runs, not reimplementations — every attempt to rebuild the
train step standalone diverged from it (mesh/jit context, checkpoint manager, hydra runtime) and
failed on scaffolding rather than the bug.

```
RESUME_FROM=gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-07-16-17-55-17/qwen3_mem_embed/16000 \
  STOP_AT=20 LR=1e-12 bash scripts/embed/debug_donate_ab.sh
```

box `rohun-v6e-8-1` (v6e-8), tmux `dbg-donate`, commit = this change's parent:

```
############ MEM_DONATE=both  (stage=staged_debug_stage3 lr=1e-12 ckpt=16000) ############
WARNING: loss or grad_norm is not finite (loss=0.8455632328987122, grad_norm=nan), skipping update
... 5 total
############ MEM_DONATE=opt   ############
WARNING: loss or grad_norm is not finite (loss=0.845588743686676, grad_norm=nan), skipping update
... 5 total
############ MEM_DONATE=none  ############
Loss: 0.7736 | CE: 0.6701 (w=1.0): 100%|##########| 20/20 [05:03<00:00, 1.42it/s]
[0] Saved step 20 to gs://memory-layers-training/dbgdonate_none-2026-07-17-02-08-04/qwen3_mem_embed
#### 0 NaN ####
```

### End-to-end validation (the one that counts)

A fresh 100k-step run with the fix, launched 2026-07-17 02:32 —
run-dir `qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-07-17-02-32-09`,
box `rohun-v6e-8-0` (v6e-8), `scripts/embed/train_hard_neg_think.sh` unchanged — **walked into
stage 3 on its own and kept going**, where the old run NaN'd on its very first stage-3 step:

```
=== Transitioning to Stage 3 at step 15000 ===
Trainable params: ['.*mem_.*', '.*embed_model.*', '.*main_model.*']
main_model core keys printed trainable: 226
not-finite lines: 0
warmup+cosine decay schedule: peak_lr=0.0001, warmup_steps=8500, decay_steps=76500, min_lr=0.0
```

Confirmations that this is a real pass and not a stalled run:
1. `main_model` is genuinely unfrozen — 226 core keys in the trainable set.
2. Zero non-finite lines through the transition and **7000+ steps beyond it** (step 22001 and
   counting). The old run NaN'd on step 15000 itself and never landed a single stage-3 update.
3. The LR is advancing rather than pinned: zero skipped steps ⇒ Adam's `count` increments ⇒ the
   9.99e-9 (= `float32(1e-8)`) signature of the deadlock is absent.

Stage transitions at 5000 and 10000 also passed clean. Note that stages 0–2 keep `main_model`
frozen and were *always* clean even with the bug, so only step ≥15000 tests the fix.

**Throughput is unchanged across the stage-3 boundary** (~1150 steps/10min in both stage 2 and
stage 3; the one-off ~555 window at 15000 is the jit **recompile**, since `optimizer` and `forward`
are static args). This is expected, not a red flag: `value_and_grad` computes `main_model`'s
gradients in *every* stage — `optax.freeze` zeroes *updates*, not grads — so the frozen-main stages
already pay for the full 4B backward. Eliminating that waste is exactly what Axis A
(`trainer.stop_grad_frozen`) is for, and it is off by default. Do not read equal throughput as
"the 4B isn't training".

Config validation (`python3 -m py_compile trainer/trainer.py` + knob resolution):

```
MEM_DONATE=None     -> donate_argnums=()
MEM_DONATE=none     -> donate_argnums=()
MEM_DONATE=opt      -> donate_argnums=(3,)
MEM_DONATE=both     -> donate_argnums=(2, 3)
MEM_DONATE=garbage  -> donate_argnums=()
nan_abort_after_samples=0   -> fires never (disabled)
nan_abort_after_samples=50  -> fires at 50 samples
```

## Methodology notes (things that cost hours)

- **Cross-box comparisons are invalid for this repro.** Each box has its own
  `$HOME/hf_parquet`, so two boxes see **different data**. The same nominal config gave 7/20 on
  box 0 and 5/20 on box 1. Only same-box, back-to-back arms carry a result. An earlier
  donation-on-vs-off comparison spanned two boxes and was worthless — it happened to agree.
- **The dataloader is deterministic within a box**: the `approx_max_k` A/B hit exactly 24/60 in
  both arms.
- **The first stage2-vs-stage3 control was confounded.** With `trainer.steps=20`,
  `warmup_frac=0.1` ⇒ `warmup_steps=2`, so stage 3 hit the 1e-4 peak in 2 steps and the arms
  drifted apart after step 0. Pinning `LR=1e-12` removed it; the NaN count was unchanged, which
  is itself the proof that the NaN is independent of the updates.
- **tqdm hides step numbers.** Its bar is one `\r`-overwritten line, so NaN warnings appear
  consecutively in the log regardless of which steps produced them. Do not infer timing from
  their position.
- **Use the launcher.** `--internal-ip` is unroutable from the dev box and gcloud silently
  auto-retries, which reads as a hang; `scripts/infrastructure/multi-vm-tpu-run.sh` uses
  `--tunnel-through-iap` and works.

## Known-remaining risk

Donation is a trigger. The backward still has a latent hazard that fires only when `main_model`
is trainable, and any recompile (a shape change, a JAX/XLA upgrade, a new fusion) could re-expose
it. The suspect surface is the code added since April — `models/memory.py` (+464),
`models/qwen3_mem_embed.py` (+157), `models/retrieval_ops.py` (+82) — which contains exactly the
shapes that produce a NaN gradient from a finite forward: `jnp.where(...)` guards (a NaN in the
**untaken** branch still poisons the backward) and divisions guarded only by an additive epsilon.

Next step for whoever picks this up: with `MEM_DONATE=both` (which reproduces in 20 steps at ckpt
16000), read `train/grad_norm_{main_core,main_mem,embed_core,embed_mem}` — these are disjoint
groups, logged even on skipped steps — to name the failing subsystem, then bisect `memory.py`
against `707a128`.
