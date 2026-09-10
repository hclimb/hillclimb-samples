"""Acceptance test: fp32 master weights actually make trainable params move.

Motivation: even after fixing the trainable-set regex (spec_tokens landing in the mask),
bf16 storage combined with adamw's default `mu_dtype=None` (moments inherit param dtype)
meant `w_bf16 + Δw_bf16` rounded Δw to zero whenever |Δw| < ULP(|w|). At LR=1e-4 against
unit-magnitude norm weights (ULP ~7.8e-3), EVERY step's update rounded away. Confirmed
empirically on perperiod_spec_zeroinit_4layer-2026-07-27-15-52-46 — mem_q_norm,
mem_o_norm, mem_layernorm, mem_layer_scale all showed Δ=0.0 exact across 22 steps.

promote_trainable_to_fp32 stores every trainable leaf in fp32, keeps frozen leaves bf16.
optimizer.init sees fp32 → mu/nu are fp32 automatically.

Test design (informed by feedback from Round 2):
  - Element-count check, not norm. Random-direction updates can partially cancel in the
    aggregate norm even when every element moves.
  - Threshold is `n_changed > 0`, not fraction. Fractional thresholds produce spurious
    reds on norm weights with legit updates near local ULP. Anything > 0 = fix worked
    for this leaf. Log the fraction for informational richness.
  - Log ‖update‖ vs ‖grad‖ per leaf. AdamW's decoupled weight decay moves EVERY
    trainable param even when gradient is zero — a passing element-count test alone
    doesn't prove gradient is reaching the leaf. A leaf with ‖grad‖ ≈ 0 while weights
    move exposes a broken gradient path hiding behind an otherwise-healthy test.

Standalone: no TPU, no HF weights. Runs against synthetic weights with the same tree
layout as the real training pipeline (flat dict, dotted keys).
"""
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))

from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax

from utils import promote_trainable_to_fp32, setup_optimizer_for_stage
from omegaconf import OmegaConf


def _fake_weights(hidden=32):
    """Same layout as merge_weights (models/utils.py:13) produces: flat dict with dotted
    keys. All bf16 at init, matching what qwen3_mem_embed produces."""
    key = jax.random.PRNGKey(0)
    return {
        # Frozen (main 4B base) — should stay bf16 after promotion.
        "main_model.embed_tokens":               jnp.ones((100, hidden), dtype=jnp.bfloat16) * 0.1,
        "main_model.layers.14.self_attn.q_proj": jax.random.normal(key, (hidden, hidden), dtype=jnp.bfloat16),
        # Trainable memory branch — should become fp32.
        # Init magnitudes chosen to match the real add_memory_layer inits so the ULP
        # trap is faithfully reproduced under bf16.
        "main_model.layers.14.mem_q_proj":       jax.random.normal(key, (hidden, hidden), dtype=jnp.bfloat16) * 0.02,
        "main_model.layers.14.mem_q_norm":       jnp.ones((hidden,), dtype=jnp.bfloat16),  # ULP ~7.8e-3, catastrophic under bf16
        "main_model.layers.14.mem_o_norm":       jnp.ones((hidden,), dtype=jnp.bfloat16),
        "main_model.layers.14.mem_layernorm":    jnp.ones((hidden,), dtype=jnp.bfloat16),
        "main_model.layers.14.mem_layer_scale":  jnp.array(0.1, dtype=jnp.bfloat16),  # ULP ~3.9e-4, trapped
        "main_model.layers.14.mem_o_proj":       jnp.zeros((hidden, hidden), dtype=jnp.bfloat16),  # zero-init, escapes even under bf16
        "main_model.mem_k":                      jax.random.normal(key, (16, hidden), dtype=jnp.bfloat16) * 0.02,
        "main_model.mem_v":                      jax.random.normal(key, (16, hidden), dtype=jnp.bfloat16) * 0.02,
        "main_model.spec_tokens":                jax.random.normal(key, (136, hidden), dtype=jnp.bfloat16) * 0.02,
        # embed_model trainable in stage 2
        "embed_model.mem_k_proj":                jax.random.normal(key, (hidden, hidden), dtype=jnp.bfloat16) * 0.02,
    }


def _fake_stages():
    """Same trainable_params as staged_ground.yaml (post-spec_tokens fix)."""
    return [
        {"trainable_params": [".*mem_.*", ".*spec_tokens.*"], "max_step": 1000, "warmup_frac": 0.0},
        {"trainable_params": [".*mem_.*", ".*embed_model.*", ".*value_model.*", ".*spec_tokens.*"],
         "max_step": 2000, "warmup_frac": 0.0},
    ]


def _fake_cfg(stage):
    return OmegaConf.create({
        "trainer": {
            "learning_rate": 1e-4,
            "clip_grad_norm": 1.0,
            "weight_decay": 0.0,  # decouple weight decay from the gradient signal — see docstring
            "steps": 2000,
        },
        "model": {"trainable_params": [".*mem_.*", ".*spec_tokens.*"]},
    }), stage


def test_promotion_preserves_frozen_bf16():
    """Frozen leaves (no trainable pattern match) must stay bf16 to keep memory cost down."""
    w = _fake_weights()
    stages = _fake_stages()
    w2 = promote_trainable_to_fp32(w, stages)
    # main_model.embed_tokens is the frozen 4B base — must NOT be promoted.
    assert w2["main_model.embed_tokens"].dtype == jnp.bfloat16, (
        f"[FAIL] frozen embed_tokens got promoted: dtype={w2['main_model.embed_tokens'].dtype}"
    )
    assert w2["main_model.layers.14.self_attn.q_proj"].dtype == jnp.bfloat16, (
        "[FAIL] frozen self_attn.q_proj got promoted"
    )
    print("[PASS] frozen leaves stay bf16 after promotion")


def test_promotion_lifts_trainable_to_fp32():
    """Every leaf matched by the union of stage regexes must be fp32."""
    w = _fake_weights()
    stages = _fake_stages()
    w2 = promote_trainable_to_fp32(w, stages)
    expected_fp32 = [
        "main_model.layers.14.mem_q_proj",
        "main_model.layers.14.mem_q_norm",
        "main_model.layers.14.mem_o_norm",
        "main_model.layers.14.mem_layernorm",
        "main_model.layers.14.mem_layer_scale",
        "main_model.layers.14.mem_o_proj",
        "main_model.mem_k",
        "main_model.mem_v",
        "main_model.spec_tokens",
        "embed_model.mem_k_proj",
    ]
    for k in expected_fp32:
        assert w2[k].dtype == jnp.float32, f"[FAIL] {k} not promoted: dtype={w2[k].dtype}"
    print(f"[PASS] all {len(expected_fp32)} trainable leaves promoted to fp32")


def _one_step(weights, stage_config, cfg, seed, state=None, optimizer=None):
    """Take one adamw step. Returns (new_weights, new_state, updates_tree, grads_tree).

    Grad = all-ones (well, all-0.01) so adamw's normalized update has consistent sign
    across steps: this test is asking "does gradient reach the leaf," not "does the leaf
    survive a random walk." Scalar params (mem_layer_scale, shape=(1,)) would otherwise
    oscillate under alternating-sign random gradients and could land back on init exactly
    — a spurious red on a working fix.
    """
    if optimizer is None:
        fake_model = SimpleNamespace(
            weights=weights,
            cfg={"main_model": {"num_upfront_spec": 128, "num_per_period_spec": 8},
                 "embed_model": {}},
            forward=partial(lambda c, x: x, {}),
        )
        optimizer, _, _ = setup_optimizer_for_stage(cfg, fake_model, stage_config)
    if state is None:
        state = optimizer.init(weights)
    grads = jax.tree_util.tree_map(
        lambda x: jnp.full(x.shape, 0.01, dtype=x.dtype),
        weights,
    )
    updates, new_state = optimizer.update(grads, state, weights)
    new_weights = jax.tree_util.tree_map(lambda w, u: w + u, weights, updates)
    return new_weights, new_state, updates, grads, optimizer


def test_all_trainable_leaves_move_after_10_steps():
    """The load-bearing test. Under the bug: mem_q_norm et al. show n_changed == 0
    exactly. Under the fix: every trainable leaf shows n_changed > 0.

    Threshold is `n_changed > 0` (from feedback): fraction thresholds spuriously red on
    norm weights near unit magnitude where legit updates land near local fp32 ULP. Log
    the fraction alongside for informational content.
    """
    w = _fake_weights()
    w = promote_trainable_to_fp32(w, _fake_stages())
    stage = _fake_stages()[0]
    cfg, stage = _fake_cfg(stage)

    w_init = jax.tree_util.tree_map(lambda x: x, w)
    all_grads_last = None
    all_updates_last = None
    state = None
    optimizer = None
    for step in range(10):
        w, state, updates, grads, optimizer = _one_step(w, stage, cfg, seed=step, state=state, optimizer=optimizer)
        all_grads_last = grads
        all_updates_last = updates

    trainable_keys = [
        "main_model.layers.14.mem_q_proj",
        "main_model.layers.14.mem_q_norm",
        "main_model.layers.14.mem_o_norm",
        "main_model.layers.14.mem_layernorm",
        "main_model.layers.14.mem_layer_scale",
        "main_model.layers.14.mem_o_proj",
        "main_model.mem_k",
        "main_model.mem_v",
        "main_model.spec_tokens",
    ]
    print(f"\n{'key':<48}  {'n_changed':>10}  {'total':>10}  {'frac':>8}  {'|update|':>10}  {'|grad|':>10}")
    failures = []
    for k in trainable_keys:
        w0 = np.array(w_init[k])
        w1 = np.array(w[k])
        n_changed = int(np.sum(w0 != w1))
        n_total = int(w0.size)
        frac = n_changed / max(n_total, 1)
        # Report |update| vs |grad| — a moving weight with |grad|=0 exposes a gradient
        # path bug hiding behind weight decay (AdamW moves every trainable regardless).
        u_norm = float(jnp.linalg.norm(all_updates_last[k].astype(jnp.float32)))
        g_norm = float(jnp.linalg.norm(all_grads_last[k].astype(jnp.float32)))
        marker = "" if n_changed > 0 else "  <-- FROZEN"
        print(f"{k:<48}  {n_changed:>10}  {n_total:>10}  {frac:>7.1%}  {u_norm:>10.2e}  {g_norm:>10.2e}{marker}")
        if n_changed == 0:
            failures.append(k)
    if failures:
        raise AssertionError(
            f"[FAIL] {len(failures)} trainable leaves show ZERO elements changed after "
            f"10 steps: {failures}. Either promotion missed the leaf or the gradient "
            f"path is broken."
        )
    print(f"[PASS] all {len(trainable_keys)} trainable leaves moved after 10 steps")


def test_frozen_leaves_stay_exact():
    """Under staged_ground stage 1, main_model.embed_tokens must not move (not trainable)."""
    w = _fake_weights()
    w = promote_trainable_to_fp32(w, _fake_stages())
    stage = _fake_stages()[0]
    cfg, stage = _fake_cfg(stage)

    w_init = jax.tree_util.tree_map(lambda x: x, w)
    state = None
    optimizer = None
    for step in range(10):
        w, state, _, _, optimizer = _one_step(w, stage, cfg, seed=step, state=state, optimizer=optimizer)

    frozen_keys = ["main_model.embed_tokens", "main_model.layers.14.self_attn.q_proj"]
    for k in frozen_keys:
        w0 = np.array(w_init[k])
        w1 = np.array(w[k])
        assert np.array_equal(w0, w1), (
            f"[FAIL] frozen leaf {k} moved: max abs diff = {np.max(np.abs(w0 - w1))}"
        )
    print(f"[PASS] frozen leaves unchanged after 10 steps")


if __name__ == "__main__":
    test_promotion_preserves_frozen_bf16()
    test_promotion_lifts_trainable_to_fp32()
    test_all_trainable_leaves_move_after_10_steps()
    test_frozen_leaves_stay_exact()
    print("\nAll fp32-master-weights acceptance tests passed.")
