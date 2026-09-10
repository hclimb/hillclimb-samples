# Adam moments are allocated for FROZEN params too (and what that costs)

**Date:** 2026-07-20 · **Author:** rohunagrawal · **Status:** finding documented; fix **opt-in and
incomplete** (`MEM_MASKED_OPTIMIZER=1`, cannot resume existing checkpoints)

## The finding

**`optax.adamw` allocates `mu` and `nu` for every parameter it is given, and
`optax.transforms.freeze(mask)` does not change that — it only zeroes the resulting updates.**
`utils.py::setup_optimizer_for_stage` builds `chain(clip, adamw, freeze(mask))`, handing `adamw`
the *whole* parameter tree, so **every stage pays 2× all params in optimizer state regardless of
how little is trainable** — about **16 GB/chip** for the 4B in this project.

Measured on CPU (1,000 trainable + 100,000 frozen leaves, fp32, lr 0.1):

| chain | opt-state elements | frozen moved | trainable moved |
|---|---|---|---|
| `chain(clip, adamw, freeze)` — current default | **202,001** (≈2 × *all* params) | 0.0 | 9.9999e-02 |
| `chain(clip, masked(adamw, trainable), freeze)` | **2,001** (≈2 × *trainable*) | 0.0 | 9.9999e-02 |

Same updates, 100× less state. `optax.masked` substitutes `MaskedNode` for unmasked leaves, so
their moments are never created.

## Why it mattered

Chasing why stage C of `staged_ground_mainunfreeze` (unfreeze the main 4B) would not fit a v6e-8
slice. Per-chip budget against v6e's **~31 GB**:

| stage | params | grads | moments | total | result |
|---|---|---|---|---|---|
| A/B — main frozen | 8 GB | small | **16 GB** | ~24 GB | fits |
| C — full unfreeze | 8 GB | 8 GB | 16 GB | ~32 GB | **OOM** (short by ~47 MB) |
| C — LoRA rank 16 | 8 GB + adapters | small | **16 GB (unchanged!)** | ~24 GB + LoRA activations | **OOM** (2.17 MB free) |

The LoRA row is the tell. **LoRA's entire memory advantage was being cancelled**: adapters should
cut moments from 16 GB to megabytes, but `adamw` still allocated them for the frozen 4B, so LoRA
only added params and an extra fp32-accumulating MLP branch — it left *less* free HBM (2.17 MB)
than the full unfreeze (36.65 MB). That inversion is what exposed the bug.

Also note batch size is not the lever: full unfreeze at batch 16 → 36.65 MB free, at batch 8 →
37.26 MB free. Halving the batch bought 0.6 MB, because batch scales activations, not the
replicated parameter/optimizer state.

## The fix, and why it is opt-in

`MEM_MASKED_OPTIMIZER=1` wraps `adamw` in `optax.masked(...)`. `freeze()` **stays in the chain in
both modes** — `masked` passes updates through *unchanged* where its mask is False, so without
`freeze` the frozen leaves would receive their raw gradients and drift. (Dropping it would be a
silent correctness bug, not an error.)

**It is off by default because it cannot resume any existing checkpoint.** `masked` changes the
`opt_state` pytree — `MaskedState(inner_state=(ScaleByAdamState(mu={…: MaskedNode()})))` — and
orbax cannot reconcile that with a checkpoint written under the default chain:

```
ValueError: Item "default" and args "PyTreeRestoreArgs(item={... 'opt_state':
  (EmptyState(), MaskedState(inner_state=(ScaleByAdamState(count=...,
   mu={'embed_model.embed_proj_conv_k_bias': MaskedNode(), ...
```

Every checkpoint in `gs://memory-layers-training` predates this, including
`ground_s1_zeroinit_4layer-2026-07-04-09-42-55/…/38000`. Resuming under the flag dies in
`load_checkpoint`.

## Options weighed

- **Opt-in env flag** (chosen) — keeps the default byte-identical, so no existing run or resume
  changes behaviour, while making the saving available to fresh runs.
- **Switch unconditionally** — rejected: silently breaks resume for every checkpoint in the
  project, and the failure is a restore-time exception rather than anything a reviewer would catch.
- **`optax.masked(set_to_zero())` instead of `freeze`** — equivalent on both state size and
  semantics in the CPU test; `freeze` kept because it is the existing, proven component.

## Tests

CPU-only (`JAX_PLATFORMS=cpu`, so it does not touch the TPU — importing JAX unguarded on one host
of a multi-host slice hangs, see runbook §2.3), script in the session scratchpad, reproduced
inline above. Both allocation counts and update semantics were checked; an earlier version of this
test used bf16 params at lr 1e-4 and reported `trainable_moved=0.0` for **every** chain — the
update rounds away at that dtype/step size, which makes a broken chain look identical to a working
one. **Use fp32 and a visible lr when testing optimizer semantics.**

**Not tested on TPU.** The one attempt died before reaching the optimizer, on the checkpoint
restore above. So the ~16 GB/chip saving is *predicted from the state-size measurement*, not yet
observed as HBM headroom on a real device.

## Follow-ups

1. **`load_checkpoint` fallback** — on `opt_state` structure mismatch, keep weights + step and
   rebuild a fresh optimizer, rather than failing the run. This is what would make the flag usable
   for resumed runs, and it is the blocker for everything else here.
2. **Verify on TPU** that LoRA + the flag actually fits (predicted comfortably: ~8 GB params +
   adapters vs 31 GB/chip). `scripts/embed/bench_lora_hbm.sh` does this with no checkpoint restore.
3. The flag does **not** rescue a full unfreeze — there every param is trainable, so the ~32 GB is
   genuinely required; that needs `tp_devices > 1` or a v5p (~95 GB/chip).

## Addendum (2026-07-22): MEM_MASKED_OPTIMIZER crashes in real training — demoted to BROKEN

First actual training attempt with `MEM_MASKED_OPTIMIZER=1` (staged_ground, doc-code smoke,
v6e slice) dies on step 1 inside `optax.masked`'s inner adamw:
`TypeError: can't multiply sequence by non-int of type 'float'` in
`optax.tree.update_moment` — a `MaskedNode` (an empty NamedTuple, i.e. a tuple) reaches the
`b1 * t` moment update as a leaf, most plausibly an optax-version incompatibility in
`is_leaf` handling. The CPU probe in this note verified allocation counts and one-step
update parity, but not a full jitted train step through `lax.cond`/donation. Until fixed,
the flag should be treated as **broken**, not merely incomplete; the frozen-param moment
cost (~16 GB) stands, and the checkpoint-save headroom problem it was meant to solve
(see the FAILED_PRECONDITION note's second addendum) needs a different mitigation
(sharded/streaming save, or v5p HBM).
