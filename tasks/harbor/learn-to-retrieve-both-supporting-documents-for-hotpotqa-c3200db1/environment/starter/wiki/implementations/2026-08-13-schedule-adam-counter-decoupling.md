# Follow-up: decouple LR schedule counter from Adam bias-correction counter

**Date:** 2026-08-13
**Status:** open (design smell recorded; not implemented)
**Related:** [`2026-08-13-wsd-lr-schedule.md`](2026-08-13-wsd-lr-schedule.md)

## The design smell

`self.lr_schedule(adams[0].count)` at `trainer/trainer.py:307,345` means Adam's `count` field is doing double duty as:

1. **LR schedule position** — needs to reset at every stage transition so warmup restarts from zero (stage-local semantics).
2. **Adam bias-correction counter** — needs to reflect the **maturity of the mu/nu tensors** so the bias-correction terms `1/(1-β₁ᵗ)` and `1/(1-β₂ᵗ)` produce properly-scaled updates.

The moment-graft at `trainer.py:693–722` (fresh `optimizer.init(weights)` → count=0 → positional graft of `mu`/`nu` from the previous stage) is correct for (1) — the LR schedule DOES need to restart. But for (2) it re-applies bias correction meant for cold moments to warm moments that were already accumulated over ~45k steps.

## The magnitude

Under `optax.adamw` defaults (β₁=0.9, β₂=0.999, unoverridden in this repo), the effective update scale for grafted params vs steady state at the first few post-transition steps:

| step | β₂ = 0.95 | β₂ = 0.999 |
|-----:|----------:|-----------:|
| 1    | 2.24×     | 0.32×      |
| 100  | 1.00×     | 0.31×      |
| 1000 | 1.00×     | 0.80×      |
| 3000 | 1.00×     | 0.98×      |

At β₂=0.999 (this repo's value) the residual is real but bounded: **~20% under-scaling at t=1000, washing out to <2% by t=3000**. It looks like a smooth boundary discontinuity in the loss curve — silent, and easy to misattribute to the LR schedule.

## Interim mitigation (already in place)

`configs/trainer/staged_ablation.yaml` uses `warmup_steps: 2500` uniform. That covers:
- main_model's cold-moment floor (`2/(1-β₂) = 2000` steps of warmup before Adam's own bias correction converges).
- The grafted-param artifact for mem_ and embed_model (washed to <10% by t=2500).

Warmup masks the mis-scaling behind an already-small LR. This is sufficient for now.

## The clean fix (deferred)

Carry an explicit stage-local step counter in the train state, and read it in the LR schedule instead of `adams[0].count`. Then the moment graft can copy `count` alongside `mu`/`nu` for groups that were already trainable in the prior stage — Adam's bias correction sees the real momentum maturity, LR schedule sees the stage-local step, no coupling.

Sketch:
1. Add `stage_step: int` to the train state (initialized to 0 on stage transition).
2. Increment `stage_step` in `_train_step` alongside gradients.
3. Change `self.lr_schedule(adams[0].count)` → `self.lr_schedule(stage_step)` at `trainer/trainer.py:307, 345`.
4. In the moment-graft (`trainer.py:706–716`), extend `_replace(mu=..., nu=...)` to `_replace(mu=..., nu=..., count=old_count)` for the Adam leaves whose params were trainable in both stages. Main_model leaves (freshly trainable in stage 3) keep `count=0` since their moments genuinely are cold.
5. Persist `stage_step` in the checkpoint alongside `stage_idx`; restore both on resume.

Scope: touches train state definition, `_train_step` signature, moment-graft code, checkpoint schema. Not a config change. Non-trivial but well-scoped.

## Why not now

- The interim mitigation (warmup 2500) reduces the artifact to <2% by t=3000 — well below noise for the metrics we watch.
- The proper fix wants a test that quantifies "boundary discontinuity vs. no-transition baseline" — a small run comparing loss at t=stage-boundary+{100, 500, 1000, 3000} against a run without the stage transition. Not stood up.
- Priority is the handoff (`staged_ablation.yaml` shipping today), not the transition artifact.

## When to unblock

Do this before the next debugging cycle spent chasing an unexplained loss discontinuity at a stage boundary. Or before any run where β₂ is lowered (β₂ = 0.95 with warmup 500 would put the residual at t=500 around 0.98× — but any experiment that pushes β₂ higher or warmup shorter needs the fix.)
