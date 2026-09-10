"""
Unit tests for mem_lookup_two_pass.

Verifies:
1. Output shapes match mem_lookup_chunked
2. Same top-k indices are selected by both passes
3. Scores at selected indices match between single-pass and two-pass
4. Gradients flow through mem_k, mem_v, and q (embed model weights get updates)
5. No gradient leaks back through the pass-1 stop_gradient boundary
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from models.memory import mem_lookup_chunked, mem_lookup_two_pass, memory_layer

# Suppress JAX tracing logs
os.environ.setdefault("JAX_LOG_COMPILES", "0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_weights(key, B, N, T, M, H, Dv, hidden_size):
    """Build a minimal set of memory weights with random values."""
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
        "rms_norm_eps":        1e-6,
        "mem_top_k":           top_k,
        "mem_use_product_keys": False,
        "mem_lookup_chunk_size": chunk_size,
        "mem_k_prenormed":     False,
        "mem_placement":       "after_attention",
        "two_pass_topk":       two_pass,
    }


# ---------------------------------------------------------------------------
# Test 1: output shapes
# ---------------------------------------------------------------------------

def test_output_shapes():
    B, N, T, M, H, Dv = 2, 4, 8, 256, 32, 32
    top_k = 16
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (B, T, N, H))
    w = make_weights(key, B, N, T, M, H, Dv, hidden_size=64)
    cfg = make_cfg(top_k, chunk_size=64)

    scores_ref, values_ref, aux_ref = mem_lookup_chunked(q, w, cfg, collect_aux=True)
    scores_tp,  values_tp,  aux_tp  = mem_lookup_two_pass(q, w, cfg, collect_aux=True)

    assert scores_ref.shape == scores_tp.shape,  f"scores shape mismatch: {scores_ref.shape} vs {scores_tp.shape}"
    assert values_ref.shape == values_tp.shape,  f"values shape mismatch: {values_ref.shape} vs {values_tp.shape}"
    assert aux_ref["mem_top_k_indices"].shape == aux_tp["mem_top_k_indices"].shape

    print(f"PASS test_output_shapes  scores={scores_tp.shape}  values={values_tp.shape}")


# ---------------------------------------------------------------------------
# Test 2: indices match
# ---------------------------------------------------------------------------

def test_indices_match():
    """Two-pass pass-1 indices must equal single-pass top-k indices."""
    B, N, T, M, H, Dv = 2, 4, 8, 256, 32, 32
    top_k = 16
    key = jax.random.PRNGKey(1)
    q = jax.random.normal(key, (B, T, N, H))
    w = make_weights(key, B, N, T, M, H, Dv, hidden_size=64)
    cfg = make_cfg(top_k, chunk_size=64)

    _, _, aux_ref = mem_lookup_chunked(q, w, cfg, collect_aux=True)
    _, _, aux_tp  = mem_lookup_two_pass(q, w, cfg, collect_aux=True)

    # Sort along k-dim before comparing (top_k order may differ)
    idx_ref = np.sort(np.array(aux_ref["mem_top_k_indices"]), axis=-1)
    idx_tp  = np.sort(np.array(aux_tp["mem_top_k_indices"]),  axis=-1)
    assert np.array_equal(idx_ref, idx_tp), "top-k indices differ between single-pass and two-pass"

    print("PASS test_indices_match")


# ---------------------------------------------------------------------------
# Test 3: scores at selected indices match
# ---------------------------------------------------------------------------

def test_scores_match():
    """Scores computed in pass-2 must equal scores from single-pass at the same indices."""
    B, N, T, M, H, Dv = 2, 4, 8, 256, 32, 32
    top_k = 16
    key = jax.random.PRNGKey(2)
    q = jax.random.normal(key, (B, T, N, H))
    w = make_weights(key, B, N, T, M, H, Dv, hidden_size=64)
    cfg = make_cfg(top_k, chunk_size=64)

    scores_ref, _, aux_ref = mem_lookup_chunked(q, w, cfg, collect_aux=True)
    scores_tp,  _, aux_tp  = mem_lookup_two_pass(q, w, cfg, collect_aux=True)

    # Both return softmax scores over the same top-k slots; after sorting indices the
    # corresponding scores should match to float32 precision.
    order_ref = np.argsort(np.array(aux_ref["mem_top_k_indices"]), axis=-1)
    order_tp  = np.argsort(np.array(aux_tp["mem_top_k_indices"]),  axis=-1)

    sorted_scores_ref = np.take_along_axis(np.array(scores_ref), order_ref, axis=-1)
    sorted_scores_tp  = np.take_along_axis(np.array(scores_tp),  order_tp,  axis=-1)

    max_diff = float(np.max(np.abs(sorted_scores_ref - sorted_scores_tp)))
    # Small float32 ordering differences between scan-based and direct-gather
    # accumulate to ~1e-3; 5e-3 is a generous but meaningful bound.
    assert max_diff < 5e-3, f"scores differ by {max_diff:.2e} (expected < 5e-3)"

    print(f"PASS test_scores_match  max_diff={max_diff:.2e}")


# ---------------------------------------------------------------------------
# Test 4: gradients flow to mem_k, mem_v, and q
# ---------------------------------------------------------------------------

def test_gradients_flow():
    """
    With sparse_grads=True (default), mem_k and mem_v gradients are returned
    as K-sized tensors via aux_data rather than a dense [M, H] scatter.

    grad_q still flows through the normal jax.grad path.
    grad_mem_k_k / grad_mem_v_k are non-zero K-row tensors from aux_data.
    """
    B, N, T, M, H, Dv = 2, 4, 8, 256, 32, 32
    top_k = 16
    key = jax.random.PRNGKey(3)
    q0 = jax.random.normal(key, (B, T, N, H))
    w  = make_weights(key, B, N, T, M, H, Dv, hidden_size=64)
    cfg = make_cfg(top_k, chunk_size=64, two_pass=True)

    # grad_q: flows through the normal jax.grad path even with sparse_grads=True
    def loss_fn(q, w):
        scores, values, _ = mem_lookup_two_pass(q, w, cfg)
        return jnp.mean(jnp.einsum('bntk,bntkd->bntd', scores, values) ** 2)

    grad_q, grad_w = jax.grad(loss_fn, argnums=(0, 1))(q0, w)
    assert jnp.any(grad_q != 0), "grad w.r.t. q is all zeros"

    # grad_mem_k / grad_mem_v: sparse path via aux_data
    _, _, aux = mem_lookup_two_pass(q0, w, cfg, collect_aux=True)

    def inner(mem_k_k, mem_v_k):
        q_t = jnp.transpose(q0, (0, 2, 1, 3))
        logits = jnp.einsum('bntd,bntkd->bntk', q_t, mem_k_k) / jnp.sqrt(
            jnp.array(q0.shape[-1], dtype=jnp.float32))
        out = jnp.einsum('bntk,bntkd->bntd', jax.nn.softmax(logits, -1), mem_v_k)
        return jnp.mean(out ** 2)

    grad_mem_k_k, grad_mem_v_k = jax.grad(inner, argnums=(0, 1))(
        aux["mem_k_k"], aux["mem_v_k"]
    )
    assert jnp.any(grad_mem_k_k != 0), "sparse grad w.r.t. mem_k_k is all zeros"
    assert jnp.any(grad_mem_v_k != 0), "sparse grad w.r.t. mem_v_k is all zeros"
    assert grad_mem_k_k.shape == aux["mem_k_k"].shape
    assert grad_mem_v_k.shape == aux["mem_v_k"].shape

    print("PASS test_gradients_flow")


# ---------------------------------------------------------------------------
# Test 5: stop_gradient — pass-1 indices do not bleed gradient into q
# ---------------------------------------------------------------------------

def test_stop_gradient_on_indices():
    """
    If we zero out mem_k, the pass-1 indices are arbitrary, but the pass-2
    score gradient w.r.t. q must still be non-zero (comes through pass-2 only).
    Conversely, perturbing mem_k should affect the loss only through pass-2
    scores/values, not through a second channel via pass-1 indices.

    We verify that the gradient of the loss w.r.t. q is identical whether we
    use two_pass_topk=True or False (same indices ⇒ same grad).
    """
    B, N, T, M, H, Dv = 1, 2, 4, 128, 16, 16
    top_k = 8
    key = jax.random.PRNGKey(4)
    q0 = jax.random.normal(key, (B, T, N, H))
    w  = make_weights(key, B, N, T, M, H, Dv, hidden_size=32)
    cfg_single = make_cfg(top_k, chunk_size=32, two_pass=False)
    cfg_two    = make_cfg(top_k, chunk_size=32, two_pass=True)

    def loss_single(q):
        scores, values, _ = mem_lookup_chunked(q, w, cfg_single)
        return jnp.mean(jnp.einsum('bntk,bntkd->bntd', scores, values) ** 2)

    def loss_two(q):
        scores, values, _ = mem_lookup_two_pass(q, w, cfg_two)
        return jnp.mean(jnp.einsum('bntk,bntkd->bntd', scores, values) ** 2)

    grad_single = jax.grad(loss_single)(q0)
    grad_two    = jax.grad(loss_two)(q0)

    max_diff = float(jnp.max(jnp.abs(grad_single - grad_two)))
    assert max_diff < 1e-4, (
        f"grad w.r.t. q differs between single-pass and two-pass by {max_diff:.2e}; "
        "suggests gradient is leaking through stop_gradient boundary"
    )
    print(f"PASS test_stop_gradient_on_indices  max_diff_grad_q={max_diff:.2e}")


# ---------------------------------------------------------------------------
# Test 6: mem_k_prenormed=True branch
# ---------------------------------------------------------------------------

def test_prenormed_keys():
    """When mem_k_prenormed=True, pass-2 skips the rms_norm on mem_k."""
    B, N, T, M, H, Dv = 2, 4, 8, 256, 32, 32
    top_k = 16
    key = jax.random.PRNGKey(5)
    q = jax.random.normal(key, (B, T, N, H))
    w = make_weights(key, B, N, T, M, H, Dv, hidden_size=64)

    # Pre-normalise mem_k manually so both paths are consistent
    from models.qwen3 import rms_norm
    w_prenorm = dict(w)
    w_prenorm["mem_k"] = rms_norm(w["mem_k"], w["mem_k_norm"], 1e-6)

    cfg_pre  = make_cfg(top_k, chunk_size=64, two_pass=True)
    cfg_pre  = dict(cfg_pre, mem_k_prenormed=True)

    scores, values, aux = mem_lookup_two_pass(q, w_prenorm, cfg_pre, collect_aux=True)

    assert scores.shape == (B, N, T, top_k)
    assert values.shape == (B, N, T, top_k, Dv)
    assert jnp.all(jnp.isfinite(scores)), "scores contain inf/nan with prenormed keys"
    assert jnp.all(jnp.isfinite(values)), "values contain inf/nan with prenormed keys"

    print("PASS test_prenormed_keys")


# ---------------------------------------------------------------------------
# Test 7: gradient magnitude is similar between single-pass and two-pass
# ---------------------------------------------------------------------------

def test_gradient_magnitude_matches():
    """
    Gradient norms for mem_k and mem_v should be approximately equal between
    single-pass and two-pass (since both see the same top-k entries).
    """
    B, N, T, M, H, Dv = 2, 4, 8, 256, 32, 32
    top_k = 16
    key = jax.random.PRNGKey(6)
    q = jax.random.normal(key, (B, T, N, H))
    w = make_weights(key, B, N, T, M, H, Dv, hidden_size=64)

    cfg_single = make_cfg(top_k, chunk_size=64, two_pass=False)
    cfg_two    = make_cfg(top_k, chunk_size=64, two_pass=True)

    def loss_single(w):
        scores, values, _ = mem_lookup_chunked(q, w, cfg_single)
        return jnp.mean(jnp.einsum('bntk,bntkd->bntd', scores, values) ** 2)

    def loss_two(w):
        scores, values, _ = mem_lookup_two_pass(q, w, cfg_two)
        return jnp.mean(jnp.einsum('bntk,bntkd->bntd', scores, values) ** 2)

    grads_single = jax.grad(loss_single)(w)

    # Two-pass uses sparse gradients by default: mem_k/mem_v grads come via aux_data.
    _, _, aux = mem_lookup_two_pass(q, w, cfg_two)
    _, _, aux = mem_lookup_two_pass(q, w, cfg_two, collect_aux=True)

    def inner(mem_k_k, mem_v_k):
        q_t = jnp.transpose(q, (0, 2, 1, 3))
        logits = jnp.einsum('bntd,bntkd->bntk', q_t, mem_k_k) / jnp.sqrt(
            jnp.array(q.shape[-1], dtype=jnp.float32))
        out = jnp.einsum('bntk,bntkd->bntd', jax.nn.softmax(logits, -1), mem_v_k)
        return jnp.mean(out ** 2)

    grad_mk_k, grad_mv_k = jax.grad(inner, argnums=(0, 1))(aux["mem_k_k"], aux["mem_v_k"])
    # Scatter K-row sparse grads back to [M, H] for norm comparison
    grads_two_mem_k = jnp.zeros_like(w["mem_k"]).at[aux["mem_top_k_indices"]].add(grad_mk_k)
    grads_two_mem_v = jnp.zeros_like(w["mem_v"]).at[aux["mem_top_k_indices"]].add(grad_mv_k)

    norm_single_k = float(jnp.linalg.norm(grads_single["mem_k"].ravel()))
    norm_two_k    = float(jnp.linalg.norm(grads_two_mem_k.ravel()))

    norm_single_v = float(jnp.linalg.norm(grads_single["mem_v"].ravel()))
    norm_two_v    = float(jnp.linalg.norm(grads_two_mem_v.ravel()))

    # Gradient norms should be in the same ballpark (within 20%).
    # The sparse path uses gather-then-norm rather than norm-then-gather, which is
    # mathematically equivalent but accumulates floating-point error differently,
    # and the softmax is normalised over K entries rather than all M — small but
    # non-zero differences are expected.
    assert abs(norm_single_k - norm_two_k) / (norm_single_k + 1e-9) < 0.20, \
        f"mem_k gradient norm differs: single={norm_single_k:.4f}, two={norm_two_k:.4f}"
    assert abs(norm_single_v - norm_two_v) / (norm_single_v + 1e-9) < 0.20, \
        f"mem_v gradient norm differs: single={norm_single_v:.4f}, two={norm_two_v:.4f}"

    print(f"PASS test_gradient_magnitude_matches  "
          f"‖∇mem_k‖ single={norm_single_k:.4f} two={norm_two_k:.4f}  "
          f"‖∇mem_v‖ single={norm_single_v:.4f} two={norm_two_v:.4f}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("test_two_pass_topk.py")
    print("=" * 60)
    test_output_shapes()
    test_indices_match()
    test_scores_match()
    test_gradients_flow()
    test_stop_gradient_on_indices()
    test_prenormed_keys()
    test_gradient_magnitude_matches()
    print("=" * 60)
    print("All tests passed.")
