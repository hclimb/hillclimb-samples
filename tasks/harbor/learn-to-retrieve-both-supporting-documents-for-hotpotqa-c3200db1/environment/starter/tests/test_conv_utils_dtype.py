"""Regression test: apply_conv1d (models/conv_utils.py) must accept a weight/bias dtype that
differs from the input activation's dtype -- e.g. a fp32-promoted `embed_proj_conv_*` weight
(utils.py::promote_trainable_to_fp32, matched via `.*embed_proj_conv.*`) read against bf16
embed-backbone activations. Before the fix, `lax.conv_general_dilated` raised:
    TypeError: lax.conv_general_dilated requires arguments to have the same dtypes,
    got bfloat16, float32.
first hit evaluating gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_
chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45 on the hybrid hotpotqa eval.

Run: JAX_PLATFORMS=cpu uv run python tests/test_conv_utils_dtype.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

from models.conv_utils import apply_conv1d

B, T, C_IN, C_OUT, KERNEL, STRIDE = 3, 17, 8, 5, 4, 4


def make_inputs(seed):
    key = jax.random.PRNGKey(seed)
    kx, kw, kb = jax.random.split(key, 3)
    x = jax.random.normal(kx, (B, T, C_IN), dtype=jnp.float32)
    # dimension_numbers=('NCH','IOH','NCH') in apply_conv1d: rhs (weight) axes are
    # (in_channels, out_channels, kernel) despite the docstring's "(out, in, kernel)" comment.
    weight = jax.random.normal(kw, (C_IN, C_OUT, KERNEL), dtype=jnp.float32)
    bias = jax.random.normal(kb, (C_OUT,), dtype=jnp.float32)
    return x, weight, bias


def run_case(seed):
    x_f32, weight_f32, bias_f32 = make_inputs(seed)
    x_bf16 = x_f32.astype(jnp.bfloat16)

    # Regression case: fp32-promoted weight/bias (as this checkpoint's saved conv leaves are)
    # against bf16 activations -- must not raise.
    out_mixed = apply_conv1d(x_bf16, weight_f32, bias_f32, STRIDE)

    # Reference: everything already cast to x's dtype up front (what the fix effectively does).
    out_ref = apply_conv1d(x_bf16, weight_f32.astype(jnp.bfloat16), bias_f32.astype(jnp.bfloat16), STRIDE)

    dtype_ok = out_mixed.dtype == x_bf16.dtype
    match_ok = np.allclose(np.asarray(out_mixed, dtype=np.float32), np.asarray(out_ref, dtype=np.float32),
                            atol=1e-5, rtol=1e-5)

    # Untouched case: same-dtype weight/bias (the common, pre-existing path) must still work.
    out_same = apply_conv1d(x_f32, weight_f32, bias_f32, STRIDE)
    same_dtype_ok = out_same.dtype == x_f32.dtype

    return dtype_ok, match_ok, same_dtype_ok


def main():
    all_ok = True
    for seed in range(5):
        dtype_ok, match_ok, same_dtype_ok = run_case(seed)
        ok = dtype_ok and match_ok and same_dtype_ok
        all_ok &= ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] seed={seed}: dtype_ok={dtype_ok} match_ok={match_ok} "
              f"same_dtype_ok={same_dtype_ok}")

    print(f"\n{'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
