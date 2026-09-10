"""Tests a proposed fix to trainer.py's stage-transition optimizer-state handling.

CURRENT behavior (trainer.py:441-469): at every stage transition (and on EVERY resume, since
train() forces one via the -1 sentinel at trainer.py:411-412), the code builds a completely FRESH
opt_state via `new_optimizer.init(params)` -- allocating new mu/nu for EVERY param, full size,
regardless of trainability (see utils.py::setup_optimizer_for_stage's own comment on this cost) --
while the OLD opt_state is still referenced, then immediately overwrites the fresh mu/nu with the
old ones to preserve momentum. This is diagnosed as the likely cause of the 2026-08-02
multihop_hard_neg_full OOM at the Stage 0->1 boundary: a transient ~2x optimizer-state footprint
right at the point memory is already tightest. See
wiki/implementations/2026-08-02-hard-neg-full-efficient-retrieval.md.

PROPOSED FIX: since mu/nu are supposed to carry over UNCHANGED anyway (that's the whole point of
the momentum-transfer step being replaced), and optax.transforms.freeze's mask lives in the
optimizer OBJECT (a static argument), not in opt_state's array data, a stage transition should be
able to just reset whatever step counter(s) the LR schedule depends on and reuse the existing
opt_state as-is -- no second allocation, ever.

THE CATCH THIS TEST IS DESIGNED TO CATCH: optax.adamw with a schedule LR is not a single state
node. Adam's own bias-correction count (bundled with mu/nu, what trainer.py's `is_adam` check
would find) is a DIFFERENT node from the schedule-scaling transform's own step counter (which is
what actually indexes into e.g. optax.warmup_cosine_decay_schedule). A fix that only resets the
`is_adam`-matched node's count could silently miss the schedule's own counter, using the WRONG
learning rate on the first post-transition step. This test builds both a naive (is_adam-only) and
a corrected (reset every count-bearing state node) version of the fix and checks which one actually
reproduces the old rebuild-and-patch behavior byte-for-byte.

Run: JAX_PLATFORMS=cpu uv run python tests/test_stage_transition_opt_state.py
(CPU-only, no TPU needed -- this is pure optax/pytree logic.)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import optax


def is_adam(x):
    """Exactly trainer.py's own check (trainer.py:449-450)."""
    return hasattr(x, 'mu') and hasattr(x, 'nu')


def has_count(x):
    # NOT hasattr(x, 'count') -- every plain tuple has a built-in .count() METHOD (e.g.
    # (1,2).count(1)), so that matches ~everything and isn't what we want. Check the namedtuple's
    # declared FIELD names instead.
    return 'count' in getattr(x, '_fields', ())


def make_optimizer(mask, lr):
    """Mirrors utils.py::setup_optimizer_for_stage's optimizer chain (lines 247-253)."""
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=0.01),
        optax.transforms.freeze(mask),
    )


def old_method_transition(old_opt_state, new_optimizer, params):
    """Exactly mirrors trainer.py:441-469: fresh init, then patch mu/nu from the old state."""
    new_opt_state = new_optimizer.init(params)
    old_adams = [l for l in jax.tree_util.tree_leaves(old_opt_state, is_leaf=is_adam) if is_adam(l)]
    new_adams = [l for l in jax.tree_util.tree_leaves(new_opt_state, is_leaf=is_adam) if is_adam(l)]
    assert len(old_adams) == len(new_adams) and len(new_adams) > 0, (
        f"expected matching adam nodes, got {len(old_adams)} old vs {len(new_adams)} new")
    adam_idx = 0

    def patch_fn(leaf):
        nonlocal adam_idx
        if is_adam(leaf):
            old_a = old_adams[adam_idx]
            adam_idx += 1
            return leaf._replace(mu=old_a.mu, nu=old_a.nu)
        return leaf

    return jax.tree_util.tree_map(patch_fn, new_opt_state, is_leaf=is_adam)


def naive_new_method_transition(old_opt_state):
    """First-draft fix: reset only the is_adam-matched node's count. This is what I proposed
    before writing this test -- expected to be WRONG if the schedule keeps its own counter."""
    def reset(leaf):
        return leaf._replace(count=jnp.zeros_like(leaf.count)) if is_adam(leaf) else leaf
    return jax.tree_util.tree_map(reset, old_opt_state, is_leaf=is_adam)


def fixed_new_method_transition(old_opt_state):
    """Corrected fix: reset EVERY count-bearing state node (Adam's bias-correction counter AND
    any separate schedule-scaling counter), touching nothing else. No new arrays allocated for
    mu/nu anywhere -- they're never even visited."""
    def reset(leaf):
        return leaf._replace(count=jnp.zeros_like(leaf.count))
    return jax.tree_util.tree_map(reset, old_opt_state, is_leaf=has_count)


def trees_equal(a, b):
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    if len(leaves_a) != len(leaves_b):
        return False
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(leaves_a, leaves_b))


def get_adam_leaf(opt_state):
    leaves = [l for l in jax.tree_util.tree_leaves(opt_state, is_leaf=is_adam) if is_adam(l)]
    assert len(leaves) == 1
    return leaves[0]


def count_bearing_node_count(opt_state):
    return len([l for l in jax.tree_util.tree_leaves(opt_state, is_leaf=has_count) if has_count(l)])


def main():
    all_ok = True

    params = {"a": jnp.array([1.0, 2.0, 3.0]), "b": jnp.array([4.0, 5.0])}
    # mask convention matches utils.py::map_fn: True = FROZEN, False = trainable.
    mask_a = {"a": False, "b": True}    # Stage A: only "a" trainable (mirrors mem_*-only Stage 0)
    mask_b = {"a": False, "b": False}   # Stage B: both trainable (mirrors embed-model unfreeze)

    lr_a = optax.constant_schedule(0.1)
    # A REAL schedule for stage B (not constant) -- this is what exposes the bug: if stage B's
    # schedule counter isn't reset, the LR used on the first post-transition step is wrong.
    lr_b = optax.warmup_cosine_decay_schedule(
        init_value=1e-8, peak_value=0.1, warmup_steps=2, decay_steps=8, end_value=0.0)

    opt_a = make_optimizer(mask_a, lr_a)
    state_a = opt_a.init(params)

    print(f"count-bearing state nodes in opt_state: {count_bearing_node_count(state_a)}")

    # Run stage-A steps with synthetic gradients so mu/nu accumulate real (nonzero) momentum.
    grads_seq = [
        {"a": jnp.array([0.1, -0.2, 0.05]), "b": jnp.array([0.3, 0.1])},
        {"a": jnp.array([0.05, 0.1, -0.1]), "b": jnp.array([-0.2, 0.05])},
        {"a": jnp.array([-0.1, 0.0, 0.2]), "b": jnp.array([0.1, -0.1])},
    ]
    for g in grads_seq:
        updates, state_a = opt_a.update(g, state_a, params)
        params = optax.apply_updates(params, updates)

    frozen_ok = np.array_equal(np.asarray(params["b"]), np.asarray(jnp.array([4.0, 5.0])))
    print(f"[{'PASS' if frozen_ok else 'FAIL'}] 'b' unchanged after 3 stage-A steps (frozen): {frozen_ok}")
    all_ok &= frozen_ok

    adam_count_after_a = int(get_adam_leaf(state_a).count)
    print(f"stage-A adam count after 3 steps: {adam_count_after_a} (expect 3)")
    all_ok &= (adam_count_after_a == 3)

    opt_b = make_optimizer(mask_b, lr_b)

    state_old = old_method_transition(state_a, opt_b, params)
    state_naive = naive_new_method_transition(state_a)
    state_fixed = fixed_new_method_transition(state_a)

    # mu/nu must be identical across all three -- momentum preservation is table stakes.
    def mu_nu(s):
        leaf = get_adam_leaf(s)
        return (leaf.mu, leaf.nu)
    momentum_ok = trees_equal(mu_nu(state_old), mu_nu(state_naive)) and trees_equal(mu_nu(state_old), mu_nu(state_fixed))
    print(f"[{'PASS' if momentum_ok else 'FAIL'}] mu/nu preserved identically by all three methods: {momentum_ok}")
    all_ok &= momentum_ok

    # Zero-copy proof for the fixed method: the mu/nu arrays are the EXACT SAME objects as
    # stage-A's, never reallocated.
    mu_a_leaves = jax.tree_util.tree_leaves(get_adam_leaf(state_a).mu)
    mu_fixed_leaves = jax.tree_util.tree_leaves(get_adam_leaf(state_fixed).mu)
    same_object = all(x is y for x, y in zip(mu_a_leaves, mu_fixed_leaves))
    print(f"[{'PASS' if same_object else 'FAIL'}] fixed method reuses the exact same mu buffers (zero-copy): {same_object}")
    all_ok &= same_object

    # THE REAL TEST: one stage-B update from each starting state must match the OLD (correct,
    # currently-shipped) method's result byte-for-byte, for the fix to be a safe drop-in.
    grad_b = {"a": jnp.array([0.02, -0.03, 0.01]), "b": jnp.array([-0.15, 0.08])}
    updates_old, next_old = opt_b.update(grad_b, state_old, params)
    updates_naive, next_naive = opt_b.update(grad_b, state_naive, params)
    updates_fixed, next_fixed = opt_b.update(grad_b, state_fixed, params)
    params_old = optax.apply_updates(params, updates_old)
    params_naive = optax.apply_updates(params, updates_naive)
    params_fixed = optax.apply_updates(params, updates_fixed)

    naive_matches_old = trees_equal(params_old, params_naive)
    fixed_matches_old = trees_equal(params_old, params_fixed)
    print(f"[{'FAIL (expected!)' if naive_matches_old else 'confirmed different, as suspected'}] "
          f"naive (is_adam-only reset) matches old method: {naive_matches_old}")
    print(f"[{'PASS' if fixed_matches_old else 'FAIL'}] fixed (reset-every-count) matches old method exactly: {fixed_matches_old}")
    all_ok &= fixed_matches_old
    # naive is EXPECTED to fail -- record it as a finding, don't let it flip the overall verdict.
    if naive_matches_old:
        print("NOTE: naive method unexpectedly matched -- the schedule-counter gap may not apply "
              "to this optax version/chain shape; re-examine before trusting that conclusion.")

    # Check the RAW update tensor, not the post-application param: lr_b's schedule starts at
    # init_value=1e-8 (warmup), so the actual float32 delta to "b" (~magnitude 4-5) can underflow
    # to exactly 0.0 once added to the param even though optax genuinely computed a nonzero
    # update -- checking the rounded param would be a false negative unrelated to the fix itself.
    b_update_nonzero = bool(np.any(np.asarray(updates_fixed["b"]) != 0.0))
    print(f"updates_fixed['b'] = {np.asarray(updates_fixed['b'])} "
          f"(vs. stage-A's opt_a, which would give exactly 0.0 here since 'b' was frozen there)")
    print(f"[{'PASS' if b_update_nonzero else 'FAIL'}] 'b' (frozen in A, trainable in B) receives a "
          f"real nonzero update under the fixed method: {b_update_nonzero}")
    all_ok &= b_update_nonzero

    # And the contrapositive, for a clean A/B: re-applying stage-A's OWN optimizer/mask to the
    # same grad must still zero "b" (proves the mask, not the schedule, is what's gating this).
    updates_a_check, _ = opt_a.update(grad_b, state_a, params)
    b_zero_under_a = bool(np.all(np.asarray(updates_a_check["b"]) == 0.0))
    print(f"[{'PASS' if b_zero_under_a else 'FAIL'}] 'b' stays exactly zero-update under stage-A's own (frozen) mask: {b_zero_under_a}")
    all_ok &= b_zero_under_a

    print(f"\n{'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
