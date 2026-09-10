"""Standalone probe: does the WSD LR schedule in utils.py::_build_schedule
produce the right shape at every step, and does the moment-graft in
trainer.py:693-722 land on the right leaves?

The load-bearing correctness check is the phase-local semantics of the
cooldown closure. `optax.join_schedules` SHIFTS the step argument by the
boundary before calling each sub-schedule, so a cooldown written in global
coordinates (with `step - decay_start` inside) double-subtracts and leaves LR
pinned at peak through the entire cooldown, ending well above the intended
floor. `lr(stage_duration) == final_lr_frac * peak` is the assertion that
catches this. A "mid-cooldown resume matches no-resume" test would not.

Assertions are all in STAGE-LOCAL coordinates (schedule reads adams[0].count
which is a per-stage counter reset at every stage transition — see the
count-reset at trainer.py:698 and the LR lookup at trainer.py:307,345).
Evaluating at global 100000 instead of stage-local 55000 silently passes on
the upper-clip and hides the bug.

Also probes moment-graft integrity across a simulated stage 2->3 transition:
grafted mu/nu should land on the right leaves, not shift positionally. This
is separate from the WSD schedule per se, but silent graft misalignment
looks exactly like a schedule bug from the loss curve, so testing it in the
same file avoids a false-attribution debugging cycle.

Run:
    JAX_PLATFORMS=cpu uv run python scripts/debug/probe_wsd_schedule.py
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# script lives at scripts/debug/, so repo root is two levels up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax
import jax.numpy as jnp
import optax
from omegaconf import OmegaConf
from functools import partial

from utils import setup_optimizer_for_stage, parse_training_stages


PEAK_MEM = 1e-4
PEAK_EMBED = 1e-5
PEAK_MAIN = 1e-5


class FakeModel:
    def __init__(self, weights, cfg_dict):
        self.weights = weights
        def _fake_forward(cfg, *a, **k):
            return None
        self.forward = partial(_fake_forward, cfg_dict)
        self.cfg = cfg_dict


def make_synthetic_weights():
    keys = [
        "main_model.norm",
        "main_model.layers.9.q_proj",
        "main_model.layers.14.mem_q_proj",     # mem group
        "main_model.layers.14.mem_o_proj",     # mem group
        "embed_model.layers.5.self_attn.q_proj",
        "embed_model.mem_k_proj",              # mem group
        "embed_model.embed_proj_conv",         # mem group
    ]
    return {k: jnp.ones((4, 4), dtype=jnp.float32) for k in keys}


def make_cfg(stage_3_lr_schedule):
    """4-stage cfg matching staged_ablation.yaml topology. Caller supplies the
    stage-3 lr_schedule block (dict of keys to merge into stage 3)."""
    stage_3 = {
        "trainable_params": [".*mem_.*", ".*embed_model.*", ".*main_model.*"],
        "max_step": 100000,
        "warmup_steps": 2500,
        "ce_weight": 1.0,
    }
    stage_3.update(stage_3_lr_schedule)
    return OmegaConf.create({
        "trainer": {
            "steps": 100000,
            "learning_rate": PEAK_MEM,
            "learning_rates": {"mem": PEAK_MEM, "embed": PEAK_EMBED, "main": PEAK_MAIN},
            "weight_decay": 0.1,
            "clip_grad_norm": 1.0,
            "training_stages": [
                {"trainable_params": [".*mem_.*", ".*embed_proj_conv.*"],
                 "max_step": 15000, "warmup_steps": 2500},
                {"trainable_params": [".*mem_.*", ".*embed_model.*"],
                 "max_step": 30000, "warmup_steps": 2500},
                {"trainable_params": [".*mem_.*", ".*embed_model.*"],
                 "max_step": 45000, "warmup_steps": 2500, "ce_weight": 1.0},
                stage_3,
            ],
        },
        "model": {"trainable_params": ["all"]},
    })


def build_stage_lr(cfg, stage_idx):
    stages = parse_training_stages(cfg)
    weights = make_synthetic_weights()
    cfg_dict = {"main_model": {"use_lora": False}, "embed_model": {"use_lora": False}}
    _, _, lr = setup_optimizer_for_stage(
        cfg, FakeModel(weights, cfg_dict),
        stage_config=stages[stage_idx], all_stages=stages,
    )
    return lr


def approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def test_wsd_shape(shape_name):
    """The load-bearing correctness test. Every assertion in STAGE-LOCAL
    coordinates. Stage-3 duration = 100000 - 45000 = 55000."""
    print(f"\n--- WSD shape test: decay_shape={shape_name} ---")
    warmup_steps = 2500
    decay_steps = 10000
    final_lr_frac = 0.1
    stage_duration = 55000  # 100000 - 45000
    peak = PEAK_MEM  # per-group scaling multiplies this, so mem == peak

    cfg = make_cfg({
        "lr_schedule": "wsd",
        "warmup_steps": warmup_steps,
        "decay_steps": decay_steps,
        "final_lr_frac": final_lr_frac,
        "decay_shape": shape_name,
    })
    lr = build_stage_lr(cfg, stage_idx=3)
    assert callable(lr), "expected a callable schedule for wsd"

    # Warmup boundary
    assert float(lr(0)) < 0.01 * peak, f"lr(0) should be ~init, got {float(lr(0))}"
    assert approx(lr(warmup_steps), peak), \
        f"lr(warmup_steps={warmup_steps}) should be peak={peak}, got {float(lr(warmup_steps))}"
    assert approx(lr(warmup_steps + 1), peak), \
        f"lr just past warmup should still be peak, got {float(lr(warmup_steps + 1))}"

    # Stable phase
    stable_end = stage_duration - decay_steps  # 45000 stage-local
    assert approx(lr(stable_end - 1), peak), \
        f"lr(last stable step={stable_end - 1}) should be peak, got {float(lr(stable_end - 1))}"

    # Cooldown entry: phase-local t=0 -> multiplier=1.0
    # This is where the double-subtract bug would ALSO evaluate to peak, so this
    # assertion alone is not enough; the end-of-cooldown assertion below is what
    # discriminates the two paths.
    assert approx(lr(stable_end), peak), \
        f"lr(cooldown entry={stable_end}) should be peak (phase-local t=0), got {float(lr(stable_end))}"

    # End of cooldown: this catches the phase-local bug. If cooldown does
    # `step - decay_start` internally, at global step stage_duration it computes
    # `stable_end - stable_end = 0` when it should compute `decay_steps`. The
    # LR would sit at peak. With correct phase-local semantics, the arg IS
    # decay_steps here and the multiplier is final_lr_frac.
    floor = final_lr_frac * peak
    assert approx(lr(stage_duration), floor), \
        f"lr(end of stage={stage_duration}) should be floor={floor} (final_lr_frac * peak); " \
        f"got {float(lr(stage_duration))}. Phase-local bug in cooldown closure?"

    # Approach to the floor: last cooldown step just before end
    assert approx(lr(stage_duration - 1), floor, tol=1e-3), \
        f"lr just before end should be ~floor, got {float(lr(stage_duration - 1))}"

    # Upper clip past decay end: LR stays at floor, does NOT go negative.
    # 1 - sqrt(frac) with frac > 1 would go negative without the clip.
    assert approx(lr(stage_duration + 100), floor), \
        f"lr past decay end should stay at floor={floor}, got {float(lr(stage_duration + 100))}"
    assert approx(lr(stage_duration + 10_000), floor), \
        f"lr well past decay end should stay at floor={floor}, got {float(lr(stage_duration + 10_000))}"

    # No undershoot anywhere in the cooldown range
    cooldown_range = jnp.arange(stable_end, stage_duration + 1)
    lrs = jnp.array([float(lr(int(t))) for t in cooldown_range])
    assert bool((lrs >= floor - 1e-8).all()), \
        f"cooldown LR undershoots floor={floor}: min={float(lrs.min())}"
    # And no overshoot past peak
    assert bool((lrs <= peak + 1e-8).all()), \
        f"cooldown LR overshoots peak={peak}: max={float(lrs.max())}"

    # Monotone non-increasing across cooldown
    diffs = jnp.diff(lrs)
    assert bool((diffs <= 1e-8).all()), \
        f"cooldown not monotone non-increasing: max positive diff = {float(diffs.max())}"

    print(f"  OK: warmup=[0,{warmup_steps}], stable=[{warmup_steps},{stable_end}], "
          f"cool=[{stable_end},{stage_duration}], floor={floor:.2e}, "
          f"lr(end)={float(lr(stage_duration)):.4e}")


def test_per_group_ratios_hold_under_wsd():
    """All three groups must ride the same shape via the scale_by_tree ratios.
    Ratio invariant: lr_group(t) / peak_group == lr_mem(t) / peak_mem at any t.

    Since _build_schedule is called with peak_mem (see utils.py per-group path)
    and scale_by_tree multiplies each leaf's update by ratio = peak_group /
    peak_mem, the *effective* per-leaf LR is base_schedule(t) * ratio. We probe
    the base schedule shape here — the per-leaf effective LR follows by
    construction (the ratio is a fixed scalar per leaf, so the shape doesn't
    diverge across groups). This test asserts the shape holds; the ratio-
    application itself is exercised by probe_per_group_lr.py."""
    print("\n--- WSD per-group shape invariance ---")
    cfg = make_cfg({
        "lr_schedule": "wsd",
        "warmup_steps": 2500,
        "decay_steps": 10000,
        "final_lr_frac": 0.1,
        "decay_shape": "linear",
    })
    lr = build_stage_lr(cfg, stage_idx=3)
    # Probe at several representative stage-local steps
    probe_steps = [0, 1000, 2500, 5000, 25000, 44999, 45000, 50000, 55000, 60000]
    peak = PEAK_MEM
    for t in probe_steps:
        val = float(lr(t))
        # Sanity: bounded by floor and peak
        assert 0 <= val <= peak + 1e-8, f"lr({t}) = {val} outside [0, peak]"
    print(f"  OK: base schedule bounded and callable at all probe steps; "
          f"per-group ratios are fixed scalars on top of this shape")


def test_wsd_validation_errors():
    """Missing decay_steps must raise. warmup + decay >= duration must raise.
    Invalid decay_shape must raise. warmup_steps == 0 must raise."""
    print("\n--- WSD validation errors ---")

    # (a) Missing decay_steps: parse_training_stages raises.
    cfg_bad_missing = make_cfg({"lr_schedule": "wsd", "warmup_steps": 2500})
    try:
        parse_training_stages(cfg_bad_missing)
    except ValueError as e:
        assert "decay_steps" in str(e), f"expected decay_steps in error, got {e}"
        print(f"  OK: missing decay_steps -> ValueError: {e}")
    else:
        raise AssertionError("expected ValueError for missing decay_steps")

    # (b) warmup + decay == stage_duration (edge case, zero stable): must raise.
    # Stage 3 duration = 55000, so warmup 2500 + decay 52500 == 55000.
    cfg_bad_eq = make_cfg({
        "lr_schedule": "wsd", "warmup_steps": 2500,
        "decay_steps": 52500, "final_lr_frac": 0.1,
    })
    try:
        build_stage_lr(cfg_bad_eq, stage_idx=3)
    except ValueError as e:
        assert "stable phase has zero" in str(e) or "must be strictly less" in str(e), \
            f"expected stable-phase error, got {e}"
        print(f"  OK: warmup + decay == duration -> ValueError: {e}")
    else:
        raise AssertionError("expected ValueError for zero-length stable phase")

    # (c) warmup + decay > stage_duration: also raises.
    cfg_bad_gt = make_cfg({
        "lr_schedule": "wsd", "warmup_steps": 2500,
        "decay_steps": 60000, "final_lr_frac": 0.1,
    })
    try:
        build_stage_lr(cfg_bad_gt, stage_idx=3)
    except ValueError as e:
        print(f"  OK: warmup + decay > duration -> ValueError: {e}")
    else:
        raise AssertionError("expected ValueError for over-length cooldown")

    # (d) Invalid decay_shape.
    cfg_bad_shape = make_cfg({
        "lr_schedule": "wsd", "warmup_steps": 2500,
        "decay_steps": 10000, "decay_shape": "cubic",
    })
    try:
        build_stage_lr(cfg_bad_shape, stage_idx=3)
    except ValueError as e:
        assert "decay_shape" in str(e), f"expected decay_shape in error, got {e}"
        print(f"  OK: invalid decay_shape -> ValueError: {e}")
    else:
        raise AssertionError("expected ValueError for invalid decay_shape")

    # (e) warmup_steps=0 not allowed for wsd (bias-correction hole).
    cfg_bad_no_warmup = make_cfg({
        "lr_schedule": "wsd", "warmup_steps": 0,
        "decay_steps": 10000,
    })
    try:
        build_stage_lr(cfg_bad_no_warmup, stage_idx=3)
    except ValueError as e:
        assert "warmup_steps" in str(e), f"expected warmup_steps in error, got {e}"
        print(f"  OK: warmup_steps=0 -> ValueError: {e}")
    else:
        raise AssertionError("expected ValueError for warmup_steps=0")


def test_moment_graft_integrity():
    """Reviewer's silent-misalignment concern: the positional graft in
    trainer.py:706-716 assumes `tree_leaves(state, is_leaf=is_adam)` returns
    the SAME leaves in the SAME order across stages. If the trainable mask
    emits MaskedNode placeholders for frozen groups (or the tree structure
    otherwise shifts), the graft misaligns silently — and it looks like a
    schedule bug from the loss curve, not a moment bug.

    Setup: give ONLY mem_ leaves nonzero gradients in stage 2, then graft
    stage 2 -> stage 3. In stage 3 the freshly-trainable main_model leaves'
    nu MUST be 0 (they were never updated in stage 2, because their grads
    were 0). The mem_ leaves' nu MUST be nonzero. If the graft misaligned,
    mem_'s nonzero nu would land on the wrong leaves."""
    print("\n--- moment-graft integrity (stage 2 -> stage 3) ---")
    weights = make_synthetic_weights()
    cfg_dict = {"main_model": {"use_lora": False}, "embed_model": {"use_lora": False}}
    cfg = make_cfg({"lr_schedule": "wsd", "warmup_steps": 2500,
                    "decay_steps": 10000, "final_lr_frac": 0.1})
    stages = parse_training_stages(cfg)

    # Stage 2 optimizer
    model_s2 = FakeModel(weights, cfg_dict)
    opt_s2, _, _ = setup_optimizer_for_stage(cfg, model_s2, stage_config=stages[2], all_stages=stages)
    state_s2 = opt_s2.init(weights)

    # Grads: only mem_ leaves nonzero. mem_ pattern per configs/dataset:
    # anything with "mem_" in the name (mem_q_proj, mem_o_proj, mem_k_proj) or
    # embed_proj_conv. All else = 0.
    def is_mem_leaf(key):
        return "mem_" in key or "embed_proj_conv" in key
    grads = {k: (jnp.ones_like(v) if is_mem_leaf(k) else jnp.zeros_like(v))
             for k, v in weights.items()}
    _, state_s2_after = opt_s2.update(grads, state_s2, weights)

    # Stage 3 optimizer (main_model unlocked)
    model_s3 = FakeModel(weights, cfg_dict)
    opt_s3, _, _ = setup_optimizer_for_stage(cfg, model_s3, stage_config=stages[3], all_stages=stages)
    new_state_s3 = opt_s3.init(weights)

    # Apply the graft (mirrors trainer.py:706-719 EXACTLY).
    def is_adam(x):
        return hasattr(x, 'mu') and hasattr(x, 'nu')
    old_adams = [l for l in jax.tree_util.tree_leaves(state_s2_after, is_leaf=is_adam) if is_adam(l)]
    new_adams = [l for l in jax.tree_util.tree_leaves(new_state_s3, is_leaf=is_adam) if is_adam(l)]
    assert len(old_adams) == len(new_adams) and len(new_adams) > 0, \
        f"Adam state count mismatch (old={len(old_adams)}, new={len(new_adams)}); " \
        f"the graft would fall through to fresh init"

    adam_idx = 0
    def patch_fn(leaf):
        nonlocal adam_idx
        if is_adam(leaf):
            old_a = old_adams[adam_idx]
            adam_idx += 1
            return leaf._replace(mu=old_a.mu, nu=old_a.nu)
        return leaf
    grafted = jax.tree_util.tree_map(patch_fn, new_state_s3, is_leaf=is_adam)

    # Extract the Adam state after graft
    grafted_adams = [l for l in jax.tree_util.tree_leaves(grafted, is_leaf=is_adam) if is_adam(l)]
    assert len(grafted_adams) == 1, f"expected one adamw state leaf, got {len(grafted_adams)}"
    a = grafted_adams[0]

    # Freshly-trainable main_model leaves: since grads for them were 0 in stage 2,
    # their nu is still 0 after graft. If misaligned, mem_'s nonzero nu would
    # land here and the assertion fires.
    for k in ("main_model.norm", "main_model.layers.9.q_proj"):
        nu_k = a.nu[k]
        max_abs = float(jnp.max(jnp.abs(nu_k)))
        assert max_abs == 0.0, \
            f"grafted nu[{k!r}] should be 0 (grads were 0 in stage 2), got max_abs={max_abs}. " \
            f"Positional graft misalignment?"

    # mem_ leaves: got nonzero grads in stage 2, so nu should be nonzero.
    for k in ("main_model.layers.14.mem_q_proj", "embed_model.mem_k_proj",
              "embed_model.embed_proj_conv"):
        nu_k = a.nu[k]
        max_abs = float(jnp.max(jnp.abs(nu_k)))
        assert max_abs > 0.0, \
            f"grafted nu[{k!r}] should be nonzero (mem_ grads were 1 in stage 2), got max_abs={max_abs}. " \
            f"Positional graft misalignment?"

    print(f"  OK: mem_ nu grafted, non-mem nu is 0 — leaves aligned correctly")


def main():
    # Basic shape (both cooldown variants).
    test_wsd_shape("linear")
    test_wsd_shape("one_minus_sqrt")
    # Per-group invariance sanity.
    test_per_group_ratios_hold_under_wsd()
    # Validation errors.
    test_wsd_validation_errors()
    # Moment-graft integrity (separate from WSD, but co-tested to avoid false-
    # attribution debugging when a boundary discontinuity shows up in a run).
    test_moment_graft_integrity()

    print("\n=== ALL WSD ASSERTIONS PASSED ===")


if __name__ == "__main__":
    main()
