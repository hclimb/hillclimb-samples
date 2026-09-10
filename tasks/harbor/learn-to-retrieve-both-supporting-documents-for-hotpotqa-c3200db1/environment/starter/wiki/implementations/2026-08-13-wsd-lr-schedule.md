# WSD LR schedule + `staged_ablation` re-timing for teammate CPT handoff

**Date:** 2026-08-13
**Status:** done
**Landed in:** commit on `staging` branch (see `git log`)

## What changed

Added a new `lr_schedule: wsd` option to `utils.py::_build_schedule` (warmup → stable peak → short cooldown), with `decay_steps` / `final_lr_frac` / `decay_shape` per-stage keys. `configs/trainer/staged_ablation.yaml` switches its stage 3 from cosine to WSD (10k linear cooldown, 0.1×peak floor) and bumps uniform warmup from 1000 → 2500 across all four stages. Existing `cosine` and warmup+constant paths untouched.

## Motivation

A teammate wants continual pretraining on Rohun's `multihop-finetuning` mixture from a checkpoint mid-way through our stage-3. **Cosine bakes `total_steps` into the shape from step 0** — any pre-`max_step` checkpoint has already had LR ratchet down and is a bad branch for CPT on a new mixture. WSD's stable phase ends at `max_step - decay_steps` at peak LR, giving an **uncooled 90k branch by construction**: teammate takes that ckpt, cools their own tail on the new mixture; we keep training to 100k for our final artifact.

Secondary: WSD makes future stage-3-boundary sweeps clean. Under cosine, moving the boundary reshapes the whole trajectory. Under WSD with cooldown anchored to end-of-stage, moving the boundary only changes when `main_model` joins.

## Options weighed

| Option | Verdict | Why |
|---|---|---|
| Keep cosine, save extra ckpt with LR-fixup metadata | rejected | Doesn't fix the "shape assumed budget" problem; the checkpoint's opt state still reflects LR decisions from an implicit 100k budget. |
| Global WSD across stages 0–3 (single schedule) | deferred | Requires unwinding the per-stage optimizer rebuild. Not needed for the handoff use case. |
| WSD in stage 3 only, keep stages 0–2 as warmup+constant | **chosen** | Minimal change; per-group ratios (`scale_by_tree`) already apply the shape to mem/embed/main automatically; existing moment-graft (`trainer.py:693–722`) works as-is. |

Shape default: `linear` (simpler). `one_minus_sqrt` (Hägele et al.'s optimal) added as opt-in — the gap is ~1–2% loss improvement.

Warmup floor was originally proposed as uniform 1000. Revised to 2500 after checking β₂: **`optax.adamw` uses default β₂=0.999** (no override anywhere in `utils.py`). This means (a) main_model cold moments in stage 3 need `2/(1-β₂)` = 2000 steps of warmup, and (b) the stage-transition graft resets `count=0` while carrying old `mu`/`nu` forward — an artifact that under-scales grafted-param updates by ~20% at t=1000 and washes out to <2% by t=3000. 2500 covers both.

## How it was built & integrated

**Config schema** — `utils.py::parse_training_stages`:
- New per-stage keys: `decay_steps` (REQUIRED for wsd), `final_lr_frac` (default 0.1), `decay_shape` (`linear` | `one_minus_sqrt`, default `linear`).
- Load-time `raise ValueError` if `lr_schedule == 'wsd'` and `decay_steps` missing.

**Schedule composition** — `utils.py::_build_schedule`:
- Widened outer guard to `lr_schedule_type in ('cosine', 'wsd')`.
- New `wsd` branch: `optax.linear_schedule(warmup) → optax.constant_schedule(peak) → cooldown` composed via `optax.join_schedules` with boundaries `[warmup_steps, warmup_steps + stable_steps]`.
- **`decay_start` is DERIVED internally** as `stage_duration - decay_steps` — never written into config. This keeps the cooldown anchored to `max_step` invariant under stage re-timing; a re-timed boundary can't leave a truncated cooldown ending mid-decay.
- Validation: `warmup_steps > 0` (a zero-length warmup on a fresh optimizer count is exactly the bias-correction hole warmup exists to fill), `warmup_steps + decay_steps < stage_duration` strict (equality = zero-length stable phase = config error), `decay_shape in ('linear', 'one_minus_sqrt')`.

**The phase-local cooldown pitfall** (blocking correctness fix): `optax.join_schedules` calls each sub-schedule with the boundary already subtracted, so the cooldown closure receives `t = 0..decay_steps` (phase-local), NOT the global step. Writing the cooldown with `step - decay_start` inside — the "obvious" global-step formulation — double-subtracts and leaves LR pinned at peak through the entire cooldown. Fix: keep the closure phase-local (`frac = jnp.clip(t / decay_steps, 0, 1)`) and never reference `decay_start` inside it. The unit test `lr(stage_duration) == final_lr_frac * peak` catches the bug; a "mid-cooldown resume matches no-resume" test would not (both paths equally wrong).

**Per-group WSD is free**: `_build_schedule(peak_mem)` produces the single shape; `_scale_by_tree(ratio_tree)` at the tail of the optimizer chain applies per-leaf ratios (`ratio = peak_group / peak_mem`). So mem/embed/main all ride the same WSD shape with different peaks. No plumbing added.

**Moment-graft at stage transitions** is untouched (`trainer/trainer.py:693–722`): fresh `optimizer.init(weights)` (count=0), then positional graft of `mu`/`nu` from the old state. Adam's `count` resets to 0 for the LR schedule — this is what "stage-local warmup" means. The count-doing-double-duty is a design smell recorded separately (see [Follow-ups](#follow-ups--risks)).

## Config: `configs/trainer/staged_ablation.yaml`

Stage 3 layout at `trainer.steps=100000`:
- warmup: stage-local 0→2500 (global 45000→47500)
- stable: stage-local 2500→45000 (global 47500→90000) at peak
- cool: stage-local 45000→55000 (global **90000→100000**) linearly from peak to 0.1×peak

Global 90000 = stage-local 45000 = `stage_duration - decay_steps` = decay start. That's the branch-checkpoint step, and it lands exactly on the standard `checkpoint_interval=5000` cadence (see the [multi-stage-training.md](../training/multi-stage-training.md) handoff section for load flavors).

AUC over stage 3: cosine-to-zero averaged 0.5×peak; WSD-10% with 0.1 floor averages 0.918×peak. Ratio ~1.8× — real, not the 4–5× that "time above 0.9×peak" alarmism would suggest.

## Reference pages updated

- [`wiki/training/multi-stage-training.md`](../training/multi-stage-training.md) — added `wsd`/`decay_steps`/`final_lr_frac`/`decay_shape` to the stage-config table, documented the anchor-to-end invariant, added a `staged_ablation.yaml` recipe subsection with the handoff protocol.

## Tests

`scripts/debug/probe_wsd_schedule.py` — standalone probe, same style as `probe_per_group_lr.py`.

Command (run on a TPU box with the memory-layers venv; local Windows has no `uv`):
```bash
JAX_PLATFORMS=cpu .venv/bin/python scripts/debug/probe_wsd_schedule.py
```

Assertions (all in **stage-local** coordinates — schedule reads `adams[0].count` which is a per-stage counter):
- `lr(0) ≈ 0`, `lr(warmup_steps) == peak`, `lr(stage_duration - decay_steps) == peak`, `lr(stage_duration - 1) ≈ floor`, **`lr(stage_duration) == floor`** (the phase-local bug catcher), `lr(stage_duration + 100) == floor` (upper clip), monotone non-increasing cooldown, no undershoot/overshoot.
- Both `linear` and `one_minus_sqrt` shapes.
- 5 validation errors (missing `decay_steps`, `warmup+decay==duration`, `warmup+decay>duration`, invalid `decay_shape`, `warmup_steps=0`).
- Moment-graft integrity: after a simulated stage 2→3 transition with `mem_`-only gradients, `nu[mem_ leaf]` is non-zero (grafted) and `nu[main_model leaf]` is zero (freshly trainable, was frozen in stage 2 with zero grads). Catches positional graft misalignment silently masquerading as a schedule bug.

Actual output (last lines):
```
  OK: warmup=[0,2500], stable=[2500,45000], cool=[45000,55000], floor=1.00e-05, lr(end)=1.0000e-05
  OK: base schedule bounded and callable at all probe steps; per-group ratios are fixed scalars on top of this shape
  OK: missing decay_steps -> ValueError: Stage 3 has lr_schedule='wsd' but 'decay_steps' is required (cooldown length in stage-local steps)
  OK: warmup + decay == duration -> ValueError: ... must be strictly less than stage_duration (55000); otherwise the stable phase has zero or negative length
  OK: warmup + decay > duration -> ValueError: warmup_steps (2500) + decay_steps (60000) must be strictly less than stage_duration (55000); ...
  OK: invalid decay_shape -> ValueError: decay_shape must be 'linear' or 'one_minus_sqrt', got 'cubic'
  OK: warmup_steps=0 -> ValueError: lr_schedule='wsd' requires warmup_steps > 0 (got 0); a zero-length warmup on a fresh optimizer count is exactly the bias-correction hole WSD's warmup exists to fill
  OK: mem_ nu grafted, non-mem nu is 0 — leaves aligned correctly

=== ALL WSD ASSERTIONS PASSED ===
```

Regression: `scripts/debug/probe_per_group_lr.py` still ends with `=== ALL STAGES BUILT SUCCESSFULLY ===` — WSD additions don't touch the cosine or warmup+constant code paths.

Hydra compose sanity:
```
warmup: [2500, 2500, 2500, 2500]
max_step: [15000, 30000, 45000, 100000]
stage3 lr_schedule: wsd
stage3 decay_steps: 10000
stage3 final_lr_frac: 0.1
stage3 decay_shape: linear
```

Not yet run end-to-end on TPU. Reviewer noted a 5k-step canary doesn't discriminate WSD from cosine (they're within 2% at that point); real divergence starts at ~20k stage-local. Recommend either running to at least stage-local 20k as a canary or committing to the full ablation once training capacity is available.

## Follow-ups & risks

- **Design smell recorded separately**: `adams[0].count` doubles as (a) LR schedule position and (b) Adam bias-correction counter. Resetting to 0 at stage transitions is correct for (a) but silently under-scales grafted-param updates for (b). Warmup 2500 masks the artifact; the proper fix is to decouple the two counters. See [`2026-08-13-schedule-adam-counter-decoupling.md`](2026-08-13-schedule-adam-counter-decoupling.md).
- **Per-stage warmup re-warms already-warm params**: at each stage boundary, mem_'s LR discontinuously drops from peak → warmup start over 2500 steps. Wasted budget (~7.5k of 100k for stages 1–2 transitions, ~7.5%). Fixable via per-group schedule composition; not blocking.
- **Boundary sweep is now cheap** thanks to end-anchored cooldown — a natural follow-up experiment. Any stage-3-start value (any global step 30k–70k, say) trivially becomes a valid arm without reshaping the trajectory.
