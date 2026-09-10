# Multi-Stage Training

Different params, loss weights, and LR schedules across phases of one run — e.g. warm up the
memory before unfreezing the embedding model. Defined under `trainer.training_stages`; parsed
in `utils.py::parse_training_stages`, driven by `Trainer.train`.

## Stage config
Each stage is active from the previous stage's `max_step` until its own. Keys:

| Key | Effect |
|-----|--------|
| `trainable_params` | list of regex; params matching any are trainable, rest frozen (see [optimizer.md](optimizer.md)) |
| `max_step` | end of this stage (strictly increasing; **final must equal `trainer.steps`** — validated) |
| `ce_weight` | CE weight for this stage |
| `aux_losses` | per-loss overrides, merged onto the base `trainer.aux_losses` |
| `lr_schedule` | `cosine`, `wsd`, or (unset) → warmup+constant. |
| `learning_rate` | per-stage peak LR (overrides `trainer.learning_rate`) |
| `warmup_frac` | linear warmup over this fraction of the stage |
| `min_lr` | cosine end value |
| `decay_steps` | **WSD only, REQUIRED.** Cooldown length in stage-local steps. `decay_start` is derived as `stage_duration - decay_steps` so the cooldown lands exactly at `max_step` — never write a `decay_start_step` field or the two can drift under a re-timed boundary and silently ship a truncated cooldown. |
| `final_lr_frac` | **WSD only.** LR floor as fraction of peak. Default `0.1`. |
| `decay_shape` | **WSD only.** `linear` (default) or `one_minus_sqrt` (Hägele et al.'s marginally-better shape). |

**Warmup floor** (both cosine and WSD): under optax `adamw` defaults (β₂=0.999, unoverridden in this repo), a fresh optimizer count needs ~`2/(1-β₂)` = 2000 steps before Adam's bias correction stops distorting updates. The stage-transition graft (see §Transition below) resets `count` to 0 while carrying old `mu`/`nu` forward for previously-trainable groups, which under-scales grafted-param updates by ~20% at step 1000 and washes out to <2% by step 3000. `warmup_steps: 2500` is the derived floor for cross-stage `adamw` at β₂=0.999; shorter warmups leave a measurable transition artifact that looks like a schedule bug from the loss curve. See [`2026-08-13-schedule-adam-counter-decoupling.md`](../implementations/2026-08-13-schedule-adam-counter-decoupling.md) for the design smell and follow-up.

## Transition (`Trainer.train`)
When `get_current_stage_idx(step)` changes:
1. `setup_optimizer_for_stage(cfg, model, stage, all_stages)` → new freeze mask + LR schedule
   (the mask/schedule are static arguments captured by the returned optimizer *object* —
   `optax.transforms.freeze`'s mask isn't stored as array data in `opt_state`, so swapping which
   params are trainable doesn't require touching `opt_state` at all).
2. **In-place reset, not a rebuild:** `opt_state` is reused as-is — `mu`/`nu` (Adam moments) carry
   over unchanged, since there's nothing to "transfer" (they never left). Only the step
   counter(s) that LR schedules index into are reset to 0, via `tree_map` with `is_leaf` matching
   any state node whose namedtuple `_fields` include `'count'` (not `hasattr(x, 'count')` — every
   plain tuple has a built-in `.count()` *method*, unrelated to a state namedtuple's `count`
   *field*). There are typically **two** such nodes for a scheduled `optax.adamw` — Adam's own
   bias-correction counter and the schedule-scaling transform's separate counter — both need
   resetting; resetting only the Adam-moments node silently leaves the LR schedule mid-stream.
   (Until 2026-08-02 this instead called `optimizer.init(weights)` to build a **second full**
   `opt_state` — fresh `mu`/`nu` for every param, full size — while the old one was still
   referenced, then discarded the fresh moments in favor of copying the old ones over. That
   transient ~2x optimizer-state footprint is diagnosed as the cause of a reproducible OOM at a
   stage boundary; see
   [2026-08-02-hard-neg-full-efficient-retrieval.md](../implementations/2026-08-02-hard-neg-full-efficient-retrieval.md).
   The in-place reset is verified byte-identical to that old behavior in
   `tests/test_stage_transition_opt_state.py`.)
3. Refresh `ce_weight` + `aux_loss_config` from the stage; log `stage` / `stage_trainable_params`
   / `stage_ce_weight` to W&B.

**Resume:** a mid-stage checkpoint restores `current==expected`, which would skip the
transition and leak stage-0 settings — so `train()` sets `current_stage_idx=-1` on resume to
force exactly one re-application.

## Canonical recipe — `configs/trainer/staged.yaml`
0 (→5k) mem + conv only, `ce_weight=0`, doc-access only 
· 1 (→10k) + embed model 
· 2 (→15k) add CE 
· 3 (→steps) unfreeze main, cosine LR. Grounding runs instead use
`staged_ground` (main **frozen throughout**, 2 stages) — see
[trainer-configs.md](trainer-configs.md).

## Ablation / branch-handoff recipe — `configs/trainer/staged_ablation.yaml`
Same unfreeze schedule as `staged.yaml` but re-timed for LR sweeps and teammate
handoff: uniform 15k-step windows for stages 0/1/2 (0→15k→30k→45k) and stage 3
runs 45k→`trainer.steps` with **WSD** instead of cosine. The load-bearing
reason is the branch point: cosine bakes `total_steps` into the shape from
step 0, so any pre-`max_step` checkpoint has already had LR ratchet down and
is a bad branch for continued pretraining on a new mixture. WSD's stable
phase ends at `max_step - decay_steps` at peak LR — an uncooled 90k branch by
construction — while the tail of the training run cools 90k→100k for the
final artifact.

**Handoff protocol** (weights-only vs full-opt-state): the full-opt-state
save at 90k already exists on the standard checkpoint cadence
(`checkpoint_interval=5000`, `max_to_keep=16` keeps it through end-of-run).
A teammate picks their flavor via the load path, not a separate stripped
ckpt:
- **Warm restart with my Adam moments** (short data-shock warmup ~300 steps
  since moments carry): call `load_checkpoint` with `resume_step=None` —
  restores full state including `opt_state`.
- **Warm start weights-only** (fresh Adam, needs full ≥2000-step warmup
  again): call `load_checkpoint` with `resume_step=<step>` — the branch at
  `utils.py:764–789` restores only weights and discards `opt_state`.

A filesystem-level "strip opt_state" utility is not possible: the on-disk
layout uses OCDBT, which stores weights and opt_state as interleaved content-
addressed blobs in a single bundle.

See [`2026-08-13-wsd-lr-schedule.md`](../implementations/2026-08-13-wsd-lr-schedule.md).
