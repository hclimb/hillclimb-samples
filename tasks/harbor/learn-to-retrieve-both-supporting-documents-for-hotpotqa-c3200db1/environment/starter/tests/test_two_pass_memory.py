"""
Compare backward-pass memory cost across three lookup strategies:

  1. Naive chunked    — lax.scan forward + reverse scan in backward  O(M) bwd
  2. Two-pass dense   — prunes backward scan but scatter to grad_mem_k still O(M)
  3. Two-pass sparse  — stop_gradient on the gather; backward only touches O(K) tensors

"Sparse" means grad_mem_k is never materialized as a dense [M, H] array.
Instead, the backward returns (top_k_indices [B,N,T,K], grad_mem_k_k [B,N,T,K,H])
which the optimizer applies with a sparse scatter at update time.

We verify this two ways:
  a) Structural: count lax.scan primitives in the gradient JAXPR.
  b) Quantitative: cost_analysis bytes-accessed grows with M for naive/dense,
     but stays flat for sparse.

Running this test:
    uv run python tests/test_two_pass_memory.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from models.memory import mem_lookup_chunked, mem_lookup_two_pass
from models.qwen3 import rms_norm

os.environ.setdefault("JAX_LOG_COMPILES", "0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_weights(key, B, N, T, M, H, Dv, hidden_size):
    keys = jax.random.split(key, 10)
    return {
        "mem_k":         jax.random.normal(keys[0], (M, H)),
        "mem_k_norm":    jnp.ones((H,)),
        "mem_v":         jax.random.normal(keys[1], (M, Dv)),
        "mem_q_proj":    jax.random.normal(keys[2], (N, H, hidden_size)) * 0.02,
        "mem_q_norm":    jnp.ones((H,)),
        "mem_o_proj":    jax.random.normal(keys[3], (hidden_size, N, Dv)) * 0.02,
        "mem_layernorm": jnp.ones((hidden_size,)),
    }


def make_cfg(top_k, chunk_size=None, two_pass=False):
    return {
        "rms_norm_eps":          1e-6,
        "mem_top_k":             top_k,
        "mem_use_product_keys":  False,
        "mem_lookup_chunk_size": chunk_size,
        "mem_k_prenormed":       False,
        "mem_placement":         "after_attention",
        "two_pass_topk":         two_pass,
    }


def cost_bytes(compiled) -> float:
    c = compiled.cost_analysis()
    if isinstance(c, list):
        return sum(x.get("bytes accessed", 0) for x in c)
    return c.get("bytes accessed", 0)


def count_scan_primitives(jaxpr) -> int:
    count = 0
    def visit(eqn):
        nonlocal count
        if hasattr(eqn.primitive, "name") and "scan" in eqn.primitive.name:
            count += 1
        for param in eqn.params.values():
            if hasattr(param, "jaxpr"):
                for sub_eqn in param.jaxpr.eqns:
                    visit(sub_eqn)
    for eqn in jaxpr.jaxpr.eqns:
        visit(eqn)
    return count


def make_sparse_grad_fn(cfg_d):
    """
    Returns a function that computes gradients with O(K) backward for mem_k/mem_v.

    Standard two-pass: grad_mem_k = zeros([M,H]).at[top_k_indices].add(d_mem_k_k)
      — O(M) because JAX always produces dense gradient tensors.

    Sparse version: stop_gradient on w['mem_k'] before the gather so the backward
    never produces a dense [M, H] gradient. Instead it returns:
      grad_mem_k_k  [B, N, T, K, H]  — only the K touched rows
      top_k_indices [B, N, T, K]     — which rows those are

    The optimizer applies: mem_k = mem_k.at[top_k_indices].add(-lr * grad_mem_k_k)
    """
    sg = jax.lax.stop_gradient

    def sparse_grad(q, w):
        # Pass 1: same stop_gradient chunked scan as two-pass
        _, _, aux = mem_lookup_two_pass(
            sg(q), jax.tree_util.tree_map(sg, w), cfg_d, collect_aux=True
        )
        top_k_indices = sg(aux["mem_top_k_indices"])  # [B, N, T, K]

        # Gather K rows — stop_gradient on source prevents backward scatter into [M, H]
        mem_k_k = rms_norm(sg(w["mem_k"]), sg(w["mem_k_norm"]), cfg_d["rms_norm_eps"])[top_k_indices]
        mem_v_k = sg(w["mem_v"])[top_k_indices]

        # Differentiate only over the K-sized gathered tensors
        def inner(q, mem_k_k, mem_v_k):
            q_t    = jnp.transpose(q, (0, 2, 1, 3))
            logits = jnp.einsum("bntd,bntkd->bntk", q_t, mem_k_k) / jnp.sqrt(
                jnp.array(q.shape[-1], dtype=jnp.float32))
            out    = jnp.einsum("bntk,bntkd->bntd", jax.nn.softmax(logits, axis=-1), mem_v_k)
            return jnp.mean(out ** 2)

        grad_q, grad_mem_k_k, grad_mem_v_k = jax.grad(inner, argnums=(0, 1, 2))(
            q, mem_k_k, mem_v_k
        )
        return grad_q, grad_mem_k_k, grad_mem_v_k, top_k_indices

    return sparse_grad


# ---------------------------------------------------------------------------
# Test 1: backward scan count (structural)
# ---------------------------------------------------------------------------

def test_backward_scan_count():
    """
    Naive chunked gradient JAXPR: 2 scans (forward + backward reverse scan).
    Two-pass dense gradient JAXPR: 1 scan (Pass 1 forward only — no backward scan).
    Two-pass sparse gradient JAXPR: 1 scan (same — sparse changes the scatter, not the scan count).
    """
    B, N, T, M, H, Dv, K = 2, 4, 32, 2048, 32, 32, 16
    chunk_size = 256

    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (B, T, N, H))
    w = make_weights(key, B, N, T, M, H, Dv, hidden_size=64)

    cfg_n = make_cfg(K, chunk_size=chunk_size, two_pass=False)
    cfg_t = make_cfg(K, chunk_size=chunk_size, two_pass=True)

    def loss_naive(q, w):
        s, v, _ = mem_lookup_chunked(q, w, cfg_n)
        return jnp.mean(jnp.einsum("bntk,bntkd->bntd", s, v) ** 2)

    def loss_two_pass(q, w):
        s, v, _ = mem_lookup_two_pass(q, w, cfg_t)
        return jnp.mean(jnp.einsum("bntk,bntkd->bntd", s, v) ** 2)

    scans_naive    = count_scan_primitives(jax.make_jaxpr(jax.grad(loss_naive,    argnums=(0,1)))(q, w))
    scans_dense    = count_scan_primitives(jax.make_jaxpr(jax.grad(loss_two_pass, argnums=(0,1)))(q, w))
    scans_sparse   = count_scan_primitives(jax.make_jaxpr(make_sparse_grad_fn(cfg_t))(q, w))

    print(f"\nlax.scan count in gradient JAXPR  (M={M}, K={K}, chunk_size={chunk_size}):")
    print(f"  Naive chunked  : {scans_naive}  (forward + reverse scan)")
    print(f"  Two-pass dense : {scans_dense}  (Pass 1 forward only; scatter still O(M))")
    print(f"  Two-pass sparse: {scans_sparse}  (Pass 1 forward only; no O(M) scatter)")

    assert scans_naive  == 2, f"Expected 2 scans for naive, got {scans_naive}"
    assert scans_dense  == 1, f"Expected 1 scan for two-pass dense, got {scans_dense}"
    assert scans_sparse == 1, f"Expected 1 scan for two-pass sparse, got {scans_sparse}"

    print("PASS test_backward_scan_count")


# ---------------------------------------------------------------------------
# Test 2: sparse grads are numerically equivalent to dense grads
# ---------------------------------------------------------------------------

def test_sparse_grads_match_dense():
    """
    Sparse grad_mem_k [B,N,T,K,H] scatter-added back to [M,H] must equal
    the dense grad_mem_k from standard two-pass.
    """
    B, N, T, M, H, Dv, K = 2, 4, 32, 1024, 32, 32, 16
    chunk_size = 256

    key = jax.random.PRNGKey(1)
    q = jax.random.normal(key, (B, T, N, H))
    w = make_weights(key, B, N, T, M, H, Dv, hidden_size=64)
    cfg_dense  = {**make_cfg(K, chunk_size=chunk_size, two_pass=True), "sparse_grads": False}
    cfg_sparse = make_cfg(K, chunk_size=chunk_size, two_pass=True)  # sparse_grads=True by default

    def loss_dense(q, w):
        s, v, _ = mem_lookup_two_pass(q, w, cfg_dense)
        return jnp.mean(jnp.einsum("bntk,bntkd->bntd", s, v) ** 2)

    gq_d, gw_d = jax.grad(loss_dense, argnums=(0, 1))(q, w)
    gq_s, gmk_s, gmv_s, idx = make_sparse_grad_fn(cfg_sparse)(q, w)

    # Convert sparse → dense for comparison
    dense_equiv_k = jnp.zeros_like(w["mem_k"]).at[idx].add(gmk_s)
    dense_equiv_v = jnp.zeros_like(w["mem_v"]).at[idx].add(gmv_s)

    diff_q  = float(jnp.max(jnp.abs(gq_d - gq_s)))
    diff_mk = float(jnp.max(jnp.abs(gw_d["mem_k"] - dense_equiv_k)))
    diff_mv = float(jnp.max(jnp.abs(gw_d["mem_v"] - dense_equiv_v)))

    print(f"\nSparse vs dense gradient max diff:")
    print(f"  grad_q  : {diff_q:.2e}")
    print(f"  grad_mk : {diff_mk:.2e}  (sparse shape: {gmk_s.shape}  dense: [{M},{H}])")
    print(f"  grad_mv : {diff_mv:.2e}  (sparse shape: {gmv_s.shape}  dense: [{M},{Dv}])")

    assert diff_q  < 1e-5, f"grad_q mismatch: {diff_q:.2e}"
    assert diff_mk < 1e-3, f"grad_mk mismatch: {diff_mk:.2e}"
    assert diff_mv < 1e-5, f"grad_mv mismatch: {diff_mv:.2e}"

    print("PASS test_sparse_grads_match_dense")


# ---------------------------------------------------------------------------
# Test 3: sparse backward bytes stay flat as M grows
# ---------------------------------------------------------------------------

def test_sparse_backward_bytes_flat():
    """
    Sparse backward cost should be roughly constant as M grows because it
    only differentiates through O(K) tensors.

    Dense two-pass backward grows O(M) due to the scatter into grad_mem_k [M,H].
    """
    B, N, T, K, H, Dv = 2, 4, 32, 16, 32, 32
    chunk_size = 256
    Ms = [512, 1024, 2048, 4096, 8192]

    print(f"\nBackward bytes vs M  (B={B}, N={N}, T={T}, K={K}, chunk_size={chunk_size}):")
    print(f"{'M':>6}  {'dense bwd MB':>14}  {'sparse bwd MB':>15}  {'savings MB':>12}")
    print("-" * 55)

    bwd_sparse_bytes = []
    bwd_dense_bytes  = []

    for M in Ms:
        key = jax.random.PRNGKey(0)
        q = jax.random.normal(key, (B, T, N, H))
        w = make_weights(key, B, N, T, M, H, Dv, hidden_size=64)
        cfg_dense  = {**make_cfg(K, chunk_size=chunk_size, two_pass=True), "sparse_grads": False}
        cfg_sparse = make_cfg(K, chunk_size=chunk_size, two_pass=True)

        def loss_dense(q, w, c=cfg_dense):
            s, v, _ = mem_lookup_two_pass(q, w, c)
            return jnp.mean(jnp.einsum("bntk,bntkd->bntd", s, v) ** 2)

        fwd_b = cost_bytes(jax.jit(loss_dense).lower(q, w).compile())
        gd_b  = cost_bytes(jax.jit(jax.grad(loss_dense, argnums=(0,1))).lower(q, w).compile())
        gs_b  = cost_bytes(jax.jit(make_sparse_grad_fn(cfg_sparse)).lower(q, w).compile())

        bwd_d = gd_b - fwd_b
        bwd_s = gs_b - fwd_b
        bwd_dense_bytes.append(bwd_d)
        bwd_sparse_bytes.append(bwd_s)

        print(f"{M:>6}  {bwd_d/1e6:>14.1f}  {bwd_s/1e6:>15.1f}  {(bwd_d-bwd_s)/1e6:>12.1f}")

    # Dense backward should grow with M; sparse should stay roughly flat
    dense_slope  = np.polyfit(Ms, bwd_dense_bytes,  1)[0]
    sparse_slope = np.polyfit(Ms, bwd_sparse_bytes, 1)[0]

    print(f"\n  Dense backward slope  : {dense_slope:.0f} bytes per unit M  (should be > 0)")
    print(f"  Sparse backward slope : {sparse_slope:.0f} bytes per unit M  (should be ~0)")

    assert dense_slope  > 0,                f"Dense backward should grow with M, slope={dense_slope}"
    assert sparse_slope < dense_slope / 10, f"Sparse backward should be nearly flat, slope={sparse_slope}"

    print("PASS test_sparse_backward_bytes_flat")


# ---------------------------------------------------------------------------
# Test 4: production-scale cost analysis via ShapeDtypeStruct
# ---------------------------------------------------------------------------

def _abstract_weights(N, M, H, Dv, hidden_size, dtype):
    return {
        "mem_k":         jax.ShapeDtypeStruct((M, H), dtype),
        "mem_k_norm":    jax.ShapeDtypeStruct((H,), dtype),
        "mem_v":         jax.ShapeDtypeStruct((M, Dv), dtype),
        "mem_q_proj":    jax.ShapeDtypeStruct((N, H, hidden_size), dtype),
        "mem_q_norm":    jax.ShapeDtypeStruct((H,), dtype),
        "mem_o_proj":    jax.ShapeDtypeStruct((hidden_size, N, Dv), dtype),
        "mem_layernorm": jax.ShapeDtypeStruct((hidden_size,), dtype),
    }


def test_production_scale_cost():
    """
    Per-mini-batch backward cost at sizes from train_two_pass.sh:
        batch_size=128, grad_accum_steps=8 → B_mini = 16
        seq_len T = 512  (pretraining.yaml)
        mem_num_heads N = 4, mem_k_dim H = 1024, mem_v_dim Dv = 1024
        mem_top_k K = 128
        docs_per_query = 4, doc_chunk_seq_len = 256 → M = 128·4·256 = 131072
        (the "512k working" commit suggests M up to 524288)

    Post-fix both paths produce gradient for w['mem_k']/w['mem_v']; the sparse
    path's only remaining win is O(K·H) rms_norm backward via gather-then-norm.
    """
    B, T, N, K, H, Dv = 16, 512, 4, 128, 1024, 1024
    hidden_size = 2560
    chunk_size = 16384
    dtype = jnp.bfloat16

    Ms = [131072, 524288]

    print(f"\nProduction-scale backward bytes-accessed  (B={B}, N={N}, T={T}, K={K}, H={H}, dtype=bf16):")
    print(f"{'M':>8}  {'dense bwd GB':>13}  {'sparse bwd GB':>15}  {'sparse/dense':>13}")
    print("-" * 55)

    for M in Ms:
        q_shape = jax.ShapeDtypeStruct((B, T, N, H), dtype)
        w_shape = _abstract_weights(N, M, H, Dv, hidden_size, dtype)

        cfg_dense  = {**make_cfg(K, chunk_size=chunk_size, two_pass=True), "sparse_grads": False}
        cfg_sparse = {**make_cfg(K, chunk_size=chunk_size, two_pass=True), "sparse_grads": True}

        def loss_dense(q, w, c=cfg_dense):
            s, v, _ = mem_lookup_two_pass(q, w, c)
            return jnp.mean(jnp.einsum("bntk,bntkd->bntd", s, v) ** 2)

        def loss_sparse(q, w, c=cfg_sparse):
            s, v, _ = mem_lookup_two_pass(q, w, c)
            return jnp.mean(jnp.einsum("bntk,bntkd->bntd", s, v) ** 2)

        fwd_d = cost_bytes(jax.jit(loss_dense ).lower(q_shape, w_shape).compile())
        fwd_s = cost_bytes(jax.jit(loss_sparse).lower(q_shape, w_shape).compile())
        gd_b  = cost_bytes(jax.jit(jax.grad(loss_dense,  argnums=(0, 1))).lower(q_shape, w_shape).compile())
        gs_b  = cost_bytes(jax.jit(jax.grad(loss_sparse, argnums=(0, 1))).lower(q_shape, w_shape).compile())

        bwd_d = gd_b - fwd_d
        bwd_s = gs_b - fwd_s
        print(f"{M:>8}  {bwd_d/1e9:>13.3f}  {bwd_s/1e9:>15.3f}  {bwd_s/bwd_d:>13.3f}")

    print("PASS test_production_scale_cost")


# ---------------------------------------------------------------------------
# Test 5: naive chunked vs two-pass dense at production scale
# ---------------------------------------------------------------------------

def test_two_pass_vs_naive_chunked_cost():
    """
    Quantify the forward+backward cost that two-pass saves over mem_lookup_chunked.
    Two-pass's Pass 1 stop_gradient severs backward through the full [B,N,T,M] scoring
    scan (mem_lookup_chunked's backward reruns the scan in reverse). Two-pass also
    uses keys_only=True in Pass 1, skipping the mem_v [M, Dv] read in forward.
    """
    B, T, N, K, H, Dv = 16, 512, 4, 128, 1024, 1024
    hidden_size = 2560
    chunk_size = 16384
    dtype = jnp.bfloat16

    Ms = [131072, 524288]

    print(f"\nNaive chunked vs two-pass dense  (B={B}, N={N}, T={T}, K={K}, H={H}, dtype=bf16):")
    print(f"{'M':>8}  {'naive fwd GB':>13}  {'2pass fwd GB':>13}  {'naive bwd GB':>13}  {'2pass bwd GB':>13}  {'naive/2pass':>12}")
    print("-" * 85)

    for M in Ms:
        q_shape = jax.ShapeDtypeStruct((B, T, N, H), dtype)
        w_shape = _abstract_weights(N, M, H, Dv, hidden_size, dtype)

        cfg_chunked = make_cfg(K, chunk_size=chunk_size, two_pass=False)
        cfg_2pass   = {**make_cfg(K, chunk_size=chunk_size, two_pass=True), "sparse_grads": False}

        def loss_chunked(q, w, c=cfg_chunked):
            s, v, _ = mem_lookup_chunked(q, w, c)
            return jnp.mean(jnp.einsum("bntk,bntkd->bntd", s, v) ** 2)

        def loss_2pass(q, w, c=cfg_2pass):
            s, v, _ = mem_lookup_two_pass(q, w, c)
            return jnp.mean(jnp.einsum("bntk,bntkd->bntd", s, v) ** 2)

        fwd_n = cost_bytes(jax.jit(loss_chunked).lower(q_shape, w_shape).compile())
        fwd_t = cost_bytes(jax.jit(loss_2pass  ).lower(q_shape, w_shape).compile())
        gn_b  = cost_bytes(jax.jit(jax.grad(loss_chunked, argnums=(0, 1))).lower(q_shape, w_shape).compile())
        gt_b  = cost_bytes(jax.jit(jax.grad(loss_2pass,   argnums=(0, 1))).lower(q_shape, w_shape).compile())

        bwd_n = gn_b - fwd_n
        bwd_t = gt_b - fwd_t
        total_ratio = gn_b / gt_b
        print(f"{M:>8}  {fwd_n/1e9:>13.3f}  {fwd_t/1e9:>13.3f}  {bwd_n/1e9:>13.3f}  {bwd_t/1e9:>13.3f}  {total_ratio:>12.3f}")

    print("PASS test_two_pass_vs_naive_chunked_cost")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("test_two_pass_memory.py")
    print("=" * 60)
    test_backward_scan_count()
    test_sparse_grads_match_dense()
    test_sparse_backward_bytes_flat()
    test_production_scale_cost()
    test_two_pass_vs_naive_chunked_cost()
    print("=" * 60)
    print("All tests passed.")
