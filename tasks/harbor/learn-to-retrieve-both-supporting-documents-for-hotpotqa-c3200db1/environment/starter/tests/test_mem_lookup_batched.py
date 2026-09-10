"""Correctness check: mem_lookup_batched (true per-row retrieval, models/memory.py) must be
numerically equivalent to mem_lookup's existing per_query_isolation=True, isolation_group_size=1
masked full-matrix path — batched is a compute/memory optimization of the SAME retrieval, not a
new one. See wiki/implementations/2026-08-02-hard-neg-full-efficient-retrieval.md.

Run: JAX_PLATFORMS=cpu uv run python tests/test_mem_lookup_batched.py
(CPU-only, no mesh, no TPU needed — see wiki/infrastructure/experiment-launch-instructions.md
on probing without touching the TPU backend.)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import AxisType

from models.memory import mem_lookup, mem_lookup_batched, mem_lookup_chunked

# out_sharding=P(...) inside mem_lookup/mem_lookup_batched needs an active Explicit mesh even on
# a single CPU device (see benchmarks/bench_sharded_retrieval.py, tests/benchmark_wall_clock.py).
_mesh = jax.make_mesh((1, 1), ("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit))
jax.set_mesh(_mesh)

B, T, N, H, DV = 4, 6, 3, 16, 20
M_PER_QUERY = 37          # deliberately not a power of 2 / not TC-aligned
TOP_K = 5
RMS_EPS = 1e-6


def make_weights(key, mask_frac_valid=0.85):
    k1, k2, k3 = jax.random.split(key, 3)
    M = B * M_PER_QUERY
    mem_k = jax.random.normal(k1, (M, H), dtype=jnp.float32)
    mem_v = jax.random.normal(k2, (M, DV), dtype=jnp.float32)
    mem_k_norm = jnp.ones((H,), dtype=jnp.float32)
    valid = (jax.random.uniform(k3, (M,)) < mask_frac_valid)
    # guarantee every row has >= TOP_K valid slots so top-k is well-defined on both paths
    valid = valid.reshape(B, M_PER_QUERY)
    valid = valid.at[:, :TOP_K].set(True)
    valid = valid.reshape(M)
    return {"mem_k": mem_k, "mem_v": mem_v, "mem_k_norm": mem_k_norm, "mem_mask": valid.astype(jnp.float32)}


def run_case(seed, mem_score_activation):
    key = jax.random.PRNGKey(seed)
    kq, kw = jax.random.split(key)
    q = jax.random.normal(kq, (B, T, N, H), dtype=jnp.float32)
    w = make_weights(kw)
    cfg = {
        "rms_norm_eps": RMS_EPS,
        "mem_top_k": TOP_K,
        "per_query_isolation": True,
        "isolation_group_size": 1,
        "mem_score_activation": mem_score_activation,
        "mem_softmax_temp": 1.0,
        "mem_phantom_log_n": 0.0,
        "mem_approx_topk": False,  # exact top-k on both paths: makes this a true equivalence check
    }

    ref_scores, ref_values, ref_aux = mem_lookup(q, w, cfg, collect_aux=True)
    batched_scores, batched_values, batched_aux = mem_lookup_batched(q, w, cfg, collect_aux=True)

    # Sort both paths' (index, score, value) triples by global index per (b,n,t) so tie-order
    # differences in top_k don't cause false mismatches.
    ref_idx = np.asarray(ref_aux["mem_top_k_indices"])
    batched_idx = np.asarray(batched_aux["mem_top_k_indices"])
    ref_scores_np = np.asarray(ref_scores)
    batched_scores_np = np.asarray(batched_scores)
    ref_values_np = np.asarray(ref_values)
    batched_values_np = np.asarray(batched_values)

    ref_order = np.argsort(ref_idx, axis=-1)
    batched_order = np.argsort(batched_idx, axis=-1)

    ref_idx_sorted = np.take_along_axis(ref_idx, ref_order, axis=-1)
    batched_idx_sorted = np.take_along_axis(batched_idx, batched_order, axis=-1)
    idx_match = np.array_equal(ref_idx_sorted, batched_idx_sorted)

    ref_scores_sorted = np.take_along_axis(ref_scores_np, ref_order, axis=-1)
    batched_scores_sorted = np.take_along_axis(batched_scores_np, batched_order, axis=-1)
    scores_close = np.allclose(ref_scores_sorted, batched_scores_sorted, atol=1e-5, rtol=1e-5)

    ref_values_sorted = np.take_along_axis(ref_values_np, ref_order[..., None], axis=-2)
    batched_values_sorted = np.take_along_axis(batched_values_np, batched_order[..., None], axis=-2)
    values_close = np.allclose(ref_values_sorted, batched_values_sorted, atol=1e-5, rtol=1e-5)

    # Also check the actual weighted read output (what memory_layer ultimately consumes).
    ref_read = np.einsum("bntk,bntkv->btnv", ref_scores_np, ref_values_np)
    batched_read = np.einsum("bntk,bntkv->btnv", batched_scores_np, batched_values_np)
    read_close = np.allclose(ref_read, batched_read, atol=1e-5, rtol=1e-5)

    return idx_match, scores_close, values_close, read_close


def run_mismatch_case():
    """isolation_group_size != 1 must raise, not silently return wrong results."""
    key = jax.random.PRNGKey(123)
    kq, kw = jax.random.split(key)
    q = jax.random.normal(kq, (B, T, N, H), dtype=jnp.float32)
    w = make_weights(kw)
    cfg = {"rms_norm_eps": RMS_EPS, "mem_top_k": TOP_K, "isolation_group_size": 4}
    try:
        mem_lookup_batched(q, w, cfg, collect_aux=True)
        return False
    except NotImplementedError:
        return True


def run_2d_mask_fallback_case(seed):
    """A genuinely per-example [B, M] mask (hybrid eval's per-query gathered banks: each row
    retrieves its own docs, so validity differs per row, not just by block ownership) must NOT
    raise — mem_lookup_batched should delegate to mem_lookup_chunked (mask-shape-agnostic) and
    match it exactly, since that's literally what happens internally."""
    key = jax.random.PRNGKey(seed)
    kq, kw = jax.random.split(key)
    q = jax.random.normal(kq, (B, T, N, H), dtype=jnp.float32)
    w = make_weights(kw)
    # Build a genuinely per-row-distinct [B, M] mask, matching gather_bank's construction: row b's
    # own block (offset b*M_PER_QUERY) carries its own validity pattern, everything outside that
    # block is 0. Knock out a few extra slots differently per row so rows aren't identical.
    valid_1d = np.asarray(w["mem_mask"]).reshape(B, M_PER_QUERY).copy()
    for b in range(B):
        drop = jax.random.permutation(jax.random.PRNGKey(seed * 100 + b), M_PER_QUERY)[:3]
        valid_1d[b, np.asarray(drop)] = 0.0
        valid_1d[b, :TOP_K] = 1.0  # keep >= TOP_K valid slots so top-k stays well-defined

    M = B * M_PER_QUERY
    full_mask = np.zeros((B, M), dtype=valid_1d.dtype)
    for b in range(B):
        full_mask[b, b * M_PER_QUERY:(b + 1) * M_PER_QUERY] = valid_1d[b]
    w2d = dict(w)
    w2d["mem_mask"] = jnp.asarray(full_mask)

    cfg = {
        "rms_norm_eps": RMS_EPS,
        "mem_top_k": TOP_K,
        "per_query_isolation": True,
        "isolation_group_size": 1,
        "mem_batched_isolation": True,
        "mem_score_activation": "softmax",
        "mem_softmax_temp": 1.0,
        "mem_phantom_log_n": 0.0,
        "mem_approx_topk": False,
    }

    batched_scores, batched_values, batched_aux = mem_lookup_batched(q, w2d, cfg, collect_aux=True)
    chunked_scores, chunked_values, chunked_aux = mem_lookup_chunked(q, w2d, cfg, collect_aux=True)

    idx_match = np.array_equal(np.asarray(batched_aux["mem_top_k_indices"]), np.asarray(chunked_aux["mem_top_k_indices"]))
    scores_close = np.allclose(np.asarray(batched_scores), np.asarray(chunked_scores), atol=1e-5, rtol=1e-5)
    values_close = np.allclose(np.asarray(batched_values), np.asarray(chunked_values), atol=1e-5, rtol=1e-5)
    return idx_match and scores_close and values_close


def main():
    all_ok = True
    for seed in range(5):
        for activation in ("softmax", "relu", "sigmoid"):
            idx_match, scores_close, values_close, read_close = run_case(seed, activation)
            ok = idx_match and scores_close and values_close and read_close
            all_ok &= ok
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] seed={seed} activation={activation}: "
                  f"idx_match={idx_match} scores_close={scores_close} "
                  f"values_close={values_close} read_close={read_close}")

    group_ok = run_mismatch_case()
    all_ok &= group_ok
    print(f"[{'PASS' if group_ok else 'FAIL'}] isolation_group_size=4 raises NotImplementedError: {group_ok}")

    for seed in range(3):
        mask2d_ok = run_2d_mask_fallback_case(seed)
        all_ok &= mask2d_ok
        print(f"[{'PASS' if mask2d_ok else 'FAIL'}] seed={seed} 2D mem_mask falls back to "
              f"mem_lookup_chunked (matches exactly): {mask2d_ok}")

    print(f"\n{'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
