"""Regression test: warm-starting from a checkpoint must not silently undo
promote_trainable_to_fp32.

Motivation: train.py promotes every trainable leaf to fp32 BEFORE checkpoint restore
(utils.py::promote_trainable_to_fp32, so optimizer.init sees fp32 params). But
utils.py::load_checkpoint's warm-start path (`resume_step is not None`) restores weights
via `restore_and_reshard`, which used to only re-apply sharding (`jax.device_put(val,
a.sharding)`) without casting `val` to `a`'s dtype. A checkpoint saved before a leaf was
ever promoted (or by a run on an older commit) stores it as bf16 — restoring that value
into an fp32-promoted target silently re-trapped the leaf in bf16 and its ULP-truncated
adamw updates (see wiki/experiments/2026-08-07-bf16-ulp-freeze-empirical-confirmation.md),
even though `promote_trainable_to_fp32` had just promoted it moments earlier.

Confirmed happening in practice: a live warm-start (multihop_ground4layer_s1warmstart_
no_multihop, 2026-08-13) showed `[promote_fp32] promoted 344 leaves bf16 -> fp32` at init,
then the first weight-monitor snapshot (taken after restore) showed
main_model.layers.14.mem_q_proj / mem_o_proj / embed_model.mem_{k,v}_proj back at
dtype=bfloat16.

This test reproduces that exact sequence with real orbax save/restore against a temp
directory (no TPU, no HF weights): save a "legacy" bf16 checkpoint, then run it through
the real load_checkpoint() warm-start path into an fp32-promoted target and assert the
restored leaves land in fp32 with the source's numeric values preserved.

    JAX_PLATFORMS=cpu uv run python tests/test_warmstart_restore_dtype.py
"""
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from utils import promote_trainable_to_fp32, load_checkpoint, save_checkpoint


TRAINABLE_KEYS = ["main_model.layers.14.mem_q_proj", "embed_model.mem_k_proj"]
FROZEN_KEY = "main_model.embed_tokens"


def _fake_weights(hidden=16):
    key = jax.random.PRNGKey(0)
    return {
        FROZEN_KEY: jnp.ones((8, hidden), dtype=jnp.bfloat16) * 0.1,
        "main_model.layers.14.mem_q_proj": jax.random.normal(key, (hidden, hidden), dtype=jnp.bfloat16) * 0.02,
        "embed_model.mem_k_proj": jax.random.normal(key, (hidden, hidden), dtype=jnp.bfloat16) * 0.02,
    }


def _make_manager(tmp_dir):
    return ocp.CheckpointManager(
        tmp_dir,
        ocp.StandardCheckpointer(),
        options=ocp.CheckpointManagerOptions(max_to_keep=2, save_interval_steps=1),
    )


def test_warmstart_restore_upcasts_legacy_bf16_leaf_to_fp32():
    tmp_dir = tempfile.mkdtemp(prefix="warmstart_restore_test_")
    try:
        # 1. A "legacy" checkpoint: saved with these leaves still bf16 (as if written before
        #    promote_trainable_to_fp32 existed, or by a run on an older commit).
        src_weights = _fake_weights()
        src_manager = _make_manager(tmp_dir)
        save_checkpoint(src_manager, SimpleNamespace(weights=src_weights), {"dummy": jnp.zeros(1)}, step=100)
        src_manager.wait_until_finished()

        # 2. A NEW run's model: same leaves, but promote_trainable_to_fp32 has just run
        #    (mirrors train.py's ordering: promote BEFORE checkpoint restore).
        tgt_weights = _fake_weights()
        stages = [{"trainable_params": [".*mem_.*"], "max_step": 1000}]
        tgt_weights = promote_trainable_to_fp32(tgt_weights, stages)
        for k in TRAINABLE_KEYS:
            assert tgt_weights[k].dtype == jnp.float32, f"setup bug: {k} not promoted before restore"
        model = SimpleNamespace(weights=tgt_weights)

        # 3. Warm-start restore (resume_step set -> load_checkpoint's warm-start branch).
        load_checkpoint(
            checkpoint_manager=None,
            model=model,
            opt_state={"dummy": jnp.zeros(1)},
            resume_step=100,
            resume_from_dir=tmp_dir,
        )

        for k in TRAINABLE_KEYS:
            got = model.weights[k]
            assert got.dtype == jnp.float32, (
                f"[FAIL] {k} restored as {got.dtype}, expected float32 — warm-start restore "
                f"silently undid promote_trainable_to_fp32"
            )
            # Values must match the source (upcast, not corrupted/reset).
            np.testing.assert_allclose(
                np.array(got), np.array(src_weights[k]).astype(np.float32),
                rtol=1e-2, atol=1e-3,  # bf16 source has ~2-3 sig figs
                err_msg=f"[FAIL] {k} restored value diverges from source",
            )
        print(f"[PASS] {len(TRAINABLE_KEYS)} promoted leaves restored as fp32 with source values intact")

        # Frozen leaf (never promoted -> target stays bf16): dtypes already match, restore
        # should be a no-op cast (bf16 -> bf16), values preserved exactly.
        assert model.weights[FROZEN_KEY].dtype == jnp.bfloat16, (
            f"[FAIL] frozen leaf {FROZEN_KEY} unexpectedly promoted to {model.weights[FROZEN_KEY].dtype}"
        )
        np.testing.assert_array_equal(np.array(model.weights[FROZEN_KEY]), np.array(src_weights[FROZEN_KEY]))
        print(f"[PASS] frozen leaf {FROZEN_KEY} stays bf16, values exact")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_warmstart_restore_upcasts_legacy_bf16_leaf_to_fp32()
    print("\nAll warm-start restore dtype tests passed.")
