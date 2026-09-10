"""Correctness + dtype check for the doc_access_per_query_loss / mem_lookup_batched pairing added
to fix the 2026-08-02 "Loss: 0.0000 all night" bug (doc_access_loss needs a full cross-batch score
grid that mem_batched_isolation structurally never builds -- see
wiki/implementations/2026-08-02-hard-neg-full-efficient-retrieval.md).

Checks:
1. mem_lookup_batched(..., collect_aux=True) with cfg['mem_collect_full_scores']=True populates
   aux_data["mem_scores"] with the right shape ([B,T,N,m_per_query]) and dtype (bf16 -- the user
   explicitly asked that everything memory-layer-related, including this loss, stay bf16; this
   also catches the `/ jnp.array(H, dtype=float32)` promotion bug fixed alongside this).
2. compute_doc_access_per_query_loss's value matches a hand-computed numpy reference (log_z -
   log_pos per (b,n,t), restricted to query b's own docs_per_query slots), not cross-checked
   against doc_access_loss itself -- an independent reference, so a shared bug in both wouldn't
   silently cancel out.
3. mem_collect_full_scores=False (the default) does NOT populate mem_scores, and existing
   mem_top_k_indices/mem_top_k_logits behavior (tests/test_mem_lookup_batched.py) is unaffected --
   this feature is additive, opt-in.

Run: JAX_PLATFORMS=cpu uv run python tests/test_doc_access_per_query_loss.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import AxisType

from models.memory import mem_lookup_batched
from losses.doc_access_per_query_loss import compute_doc_access_per_query_loss

_mesh = jax.make_mesh((1, 1), ("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit))
jax.set_mesh(_mesh)

B, T, N, H, DV = 3, 4, 2, 16, 12
DOCS_PER_QUERY = 5
DOC_LEN = 7            # deliberately small/uneven, not chunk-aligned to anything
M_PER_QUERY = DOCS_PER_QUERY * DOC_LEN
TOP_K = 6
RMS_EPS = 1e-6


def make_inputs(key, valid_frac=0.8):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    M = B * M_PER_QUERY
    q = jax.random.normal(k1, (B, T, N, H), dtype=jnp.bfloat16)
    mem_k = jax.random.normal(k2, (M, H), dtype=jnp.bfloat16)
    mem_v = jax.random.normal(k3, (M, DV), dtype=jnp.bfloat16)
    mem_k_norm = jnp.ones((H,), dtype=jnp.bfloat16)

    valid = (jax.random.uniform(k4, (M,)) < valid_frac).reshape(B, M_PER_QUERY)
    valid = valid.at[:, : TOP_K + 1].set(True)  # guarantee enough valid slots for top-k
    valid = valid.reshape(M).astype(jnp.float32)

    pos_doc_mask = (jax.random.uniform(k5, (B, DOCS_PER_QUERY)) < 0.4)
    # guarantee every query has >=1 positive AND >=1 negative doc, else log_pos/log_z degenerate
    pos_doc_mask = pos_doc_mask.at[:, 0].set(True)
    pos_doc_mask = pos_doc_mask.at[:, 1].set(False)
    return q, mem_k, mem_v, mem_k_norm, valid, pos_doc_mask.astype(jnp.int32)


def reference_loss(scores_np, valid_np, pos_doc_mask_np, loss_mask_np):
    """Independent numpy reference: log_z - log_pos per (b,n,t), restricted to query b's own
    docs_per_query slots, averaged like the real loss (sum over valid (b,t) * N, then /count)."""
    Bn, Tn, Nn, Mn = scores_np.shape
    valid_bdl = valid_np.reshape(Bn, DOCS_PER_QUERY, DOC_LEN) > 0
    pos_bdl = (pos_doc_mask_np[:, :, None] > 0) & valid_bdl  # [B, docs_per_query, doc_len]

    total = 0.0
    count = 0.0
    for b in range(Bn):
        for t in range(Tn):
            if loss_mask_np[b, t] == 0:
                continue
            for n in range(Nn):
                s = scores_np[b, t, n].reshape(DOCS_PER_QUERY, DOC_LEN).astype(np.float64)
                v = valid_bdl[b]
                p = pos_bdl[b]
                log_z = np.log(np.sum(np.exp(s[v] - s[v].max())) ) + s[v].max()
                log_pos = np.log(np.sum(np.exp(s[p] - s[p].max()))) + s[p].max()
                total += (log_z - log_pos)
                count += 1
    return total / (np.sum(loss_mask_np) * Nn)


def main():
    all_ok = True
    key = jax.random.PRNGKey(0)
    q, mem_k, mem_v, mem_k_norm, valid, pos_doc_mask = make_inputs(key)
    w = {"mem_k": mem_k, "mem_v": mem_v, "mem_k_norm": mem_k_norm, "mem_mask": valid}

    # --- 1. mem_collect_full_scores=False: no mem_scores, unaffected otherwise ---
    cfg_off = {
        "rms_norm_eps": RMS_EPS, "mem_top_k": TOP_K,
        "isolation_group_size": 1, "mem_approx_topk": False,
        "mem_score_activation": "softmax", "mem_softmax_temp": 1.0, "mem_phantom_log_n": 0.0,
        "mem_collect_full_scores": False,
    }
    _, _, aux_off = mem_lookup_batched(q, w, cfg_off, collect_aux=True)
    ok = "mem_scores" not in aux_off
    print(f"[{'PASS' if ok else 'FAIL'}] mem_collect_full_scores=False -> no mem_scores key: {ok}")
    all_ok &= ok

    # --- 2. mem_collect_full_scores=True: shape + dtype ---
    cfg_on = {**cfg_off, "mem_collect_full_scores": True}
    top_k_scores, top_k_values, aux_on = mem_lookup_batched(q, w, cfg_on, collect_aux=True)
    has_scores = "mem_scores" in aux_on
    print(f"[{'PASS' if has_scores else 'FAIL'}] mem_collect_full_scores=True -> mem_scores present: {has_scores}")
    all_ok &= has_scores

    mem_scores = aux_on["mem_scores"][0]  # (tensor,) 1-tuple convention
    shape_ok = tuple(mem_scores.shape) == (B, T, N, M_PER_QUERY)
    print(f"[{'PASS' if shape_ok else 'FAIL'}] mem_scores shape == (B,T,N,m_per_query): "
          f"{tuple(mem_scores.shape)} vs {(B, T, N, M_PER_QUERY)}")
    all_ok &= shape_ok

    dtype_scores_ok = mem_scores.dtype == jnp.bfloat16
    dtype_logits_ok = aux_on["mem_top_k_logits"].dtype == jnp.bfloat16
    print(f"[{'PASS' if dtype_scores_ok else 'FAIL'}] mem_scores dtype is bf16 (not promoted to fp32): {mem_scores.dtype}")
    print(f"[{'PASS' if dtype_logits_ok else 'FAIL'}] mem_top_k_logits dtype is bf16: {aux_on['mem_top_k_logits'].dtype}")
    all_ok &= dtype_scores_ok
    all_ok &= dtype_logits_ok

    # mem_top_k_logits must be RAW logits (pre-activation), NOT the post-activation top_k_scores
    # returned as the function's own first output -- they should differ (softmax != raw logit).
    raw_vs_activated_differ = not np.allclose(
        np.asarray(aux_on["mem_top_k_logits"], dtype=np.float32),
        np.asarray(top_k_scores, dtype=np.float32),
    )
    print(f"[{'PASS' if raw_vs_activated_differ else 'FAIL'}] mem_top_k_logits (raw) != returned top_k_scores (post-activation): {raw_vs_activated_differ}")
    all_ok &= raw_vs_activated_differ

    # --- 3. Loss value vs. independent numpy reference ---
    loss_mask = jnp.ones((B, T), dtype=jnp.float32)
    input_mask = {"pos_doc_mask": pos_doc_mask, "docs_mask": jnp.ones((B * DOCS_PER_QUERY, DOC_LEN))}
    aux_data = {**aux_on, "effective_mem_mask": valid, "effective_doc_len": DOC_LEN}

    loss = compute_doc_access_per_query_loss(aux_data, loss_mask, input_mask, None, temperature=1.0)
    loss_val = float(loss)

    ref = reference_loss(
        np.asarray(mem_scores, dtype=np.float32),
        np.asarray(valid),
        np.asarray(pos_doc_mask),
        np.asarray(loss_mask),
    )
    close = abs(loss_val - ref) < 1e-2  # bf16-level tolerance
    print(f"[{'PASS' if close else 'FAIL'}] loss matches independent numpy reference: "
          f"{loss_val:.6f} vs {ref:.6f} (diff={abs(loss_val-ref):.6f})")
    all_ok &= close

    # --- 4. Loss is exactly 0 only in the degenerate case (no negatives) sanity check ---
    all_pos_mask = jnp.ones((B, DOCS_PER_QUERY), dtype=jnp.int32)
    input_mask_allpos = {**input_mask, "pos_doc_mask": all_pos_mask}
    loss_allpos = float(compute_doc_access_per_query_loss(aux_data, loss_mask, input_mask_allpos, None))
    near_zero = abs(loss_allpos) < 1e-2
    print(f"[{'PASS' if near_zero else 'FAIL'}] loss ~0 when every doc is 'positive' (no negatives to contrast against): {loss_allpos:.6f}")
    all_ok &= near_zero

    # --- 5. Missing mem_scores -> 0.0, not an error (matches doc_access_loss's own guard) ---
    loss_missing = compute_doc_access_per_query_loss({}, loss_mask, input_mask, None)
    missing_ok = loss_missing == 0.0
    print(f"[{'PASS' if missing_ok else 'FAIL'}] missing mem_scores -> returns 0.0 (not an error): {missing_ok}")
    all_ok &= missing_ok

    print(f"\n{'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
