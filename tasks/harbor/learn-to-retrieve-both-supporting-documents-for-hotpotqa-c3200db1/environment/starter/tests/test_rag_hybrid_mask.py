"""Standalone test for the RAG->memory hybrid eval's mask machinery (CPU, no TPU needed).

    uv run python tests/test_rag_hybrid_mask.py

Covers the pure helpers in evals/gen_large_mem_rag_hybrid.py and the load-bearing assumption
the whole design rests on: retrieval_ops excludes masked slots BEFORE top-k selection, so a
per-query mem_mask is equivalent to physically shrinking the bank.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from evals.gen_large_mem_rag_hybrid import (
    build_doc_slot_mask,
    build_gather_plan,
    gold_coverage,
    rank_unique_doc_ids,
)


def test_rank_unique_doc_ids():
    # chunks 0..5 -> docs (duplicates: two chunks per doc for docs 7 and 9)
    chunk_doc_ids = np.array([7, 9, 7, 3, 9, 5])
    ranking = [2, 0, 4, 3, 1, 5]          # best chunk first
    got = rank_unique_doc_ids(ranking, chunk_doc_ids)
    assert got == [7, 9, 3, 5], got       # dup docs collapse to first hit, order kept
    print("PASS rank_unique_doc_ids")


def test_build_doc_slot_mask():
    # 4 docs x 3 slots each = 12 flat slots; doc 3's last slot is padding (validity 0)
    doc_id_to_flat = {1: {0, 1, 2}, 2: {3, 4, 5}, 3: {6, 7, 8}, 4: {9, 10, 11}}
    validity = np.ones(12, dtype=np.float32)
    validity[8] = 0.0

    mask = build_doc_slot_mask([1, 3], doc_id_to_flat, validity)
    assert mask.dtype == validity.dtype                      # dtype stable across queries
    assert mask[[0, 1, 2, 6, 7]].astype(bool).all()          # selected docs' valid slots on
    assert not mask[8]                                       # padding inside a selected doc stays OFF
    assert not mask[[3, 4, 5, 9, 10, 11]].astype(bool).any() # unselected docs fully OFF

    # unknown doc id is a no-op, not a crash
    mask = build_doc_slot_mask([999], doc_id_to_flat, validity)
    assert not mask.astype(bool).any()
    print("PASS build_doc_slot_mask")


def test_gold_coverage():
    ranked = [10, 20, 30, 40, 50]
    cov = gold_coverage(ranked, gold_ids=[30, 50], ks=[1, 3, 5])
    assert cov[1] == (False, False)
    assert cov[3] == (True, False)        # one of two golds inside top-3
    assert cov[5] == (True, True)
    assert gold_coverage(ranked, gold_ids=[], ks=[3])[3] == (None, None)
    print("PASS gold_coverage")


def test_masked_slots_never_retrieved():
    """The design assumption: retrieval_ops applies mem_mask as float32.min BEFORE top-k."""
    import jax
    import jax.numpy as jnp
    from models.retrieval_ops import _matmul_top_k

    rng = np.random.default_rng(0)
    B, N, T, D, Dv, M, K = 1, 2, 3, 8, 8, 32, 4
    query = jnp.asarray(rng.normal(size=(B, N, T, D)).astype(np.float32))
    keys = jnp.asarray(rng.normal(size=(M, D)).astype(np.float32))
    values = jnp.asarray(rng.normal(size=(M, Dv)).astype(np.float32))

    allowed = np.zeros(M, dtype=bool)
    allowed_idx = rng.choice(M, size=10, replace=False)
    allowed[allowed_idx] = True

    _, _, indices = _matmul_top_k(query, keys, values, jnp.asarray(allowed), K, D, 0)
    got = set(np.array(indices).reshape(-1).tolist())
    assert got <= set(allowed_idx.tolist()), f"retrieved masked slots: {got - set(allowed_idx.tolist())}"

    # A different mask selects from ITS allowed set — the per-query swap is a real restriction.
    allowed2 = ~allowed
    _, _, indices2 = _matmul_top_k(query, keys, values, jnp.asarray(allowed2), K, D, 0)
    got2 = set(np.array(indices2).reshape(-1).tolist())
    assert got2 <= set(np.where(allowed2)[0].tolist())
    assert got.isdisjoint(got2)
    print("PASS masked_slots_never_retrieved")


def test_matmul_top_k_per_example_mask():
    """[B, M] mask: each row retrieves only from ITS OWN allowed set."""
    import jax.numpy as jnp
    from models.retrieval_ops import _matmul_top_k

    rng = np.random.default_rng(2)
    B, N, T, D, Dv, M, K = 2, 2, 3, 8, 8, 32, 4
    query = jnp.asarray(rng.normal(size=(B, N, T, D)).astype(np.float32))
    keys = jnp.asarray(rng.normal(size=(M, D)).astype(np.float32))
    values = jnp.asarray(rng.normal(size=(M, Dv)).astype(np.float32))

    allowed = np.zeros((B, M), dtype=bool)
    allowed[0, :10] = True
    allowed[1, 20:] = True                                   # disjoint per-row sets

    _, _, indices = _matmul_top_k(query, keys, values, jnp.asarray(allowed), K, D, 0)
    idx = np.array(indices)
    assert set(idx[0].reshape(-1).tolist()) <= set(range(10)), "row 0 escaped its mask"
    assert set(idx[1].reshape(-1).tolist()) <= set(range(20, M)), "row 1 escaped its mask"
    print("PASS matmul_top_k_per_example_mask")


def test_per_example_mask_batched_equals_single():
    """The data-parallel mode's load-bearing claim: a [B, M] batched lookup returns, per row,
    exactly what that row returns run ALONE with its [M] mask — through the production
    chunked/replicated path (_scan_chunks), with M chosen to exercise chunk padding."""
    import jax.numpy as jnp
    from models.retrieval_ops import _replicated_top_k

    rng = np.random.default_rng(3)
    B, N, T, D, Dv, M, K, CHUNK = 4, 2, 3, 8, 8, 30, 4, 8   # 30 % 8 != 0 -> pad path
    query = rng.normal(size=(B, N, T, D)).astype(np.float32)
    keys = rng.normal(size=(M, D)).astype(np.float32)
    values = rng.normal(size=(M, Dv)).astype(np.float32)

    allowed = np.zeros((B, M), dtype=bool)
    for b in range(B):
        allowed[b, rng.choice(M, size=10, replace=False)] = True

    scores_b, _, idx_b, _ = _replicated_top_k(
        jnp.asarray(query), jnp.asarray(keys), jnp.asarray(values), K,
        jnp.asarray(allowed), CHUNK, D, Dv, B, N, T, use_mesh=False)
    scores_b, idx_b = np.array(scores_b), np.array(idx_b)

    for b in range(B):
        got = set(idx_b[b].reshape(-1).tolist())
        assert got <= set(np.where(allowed[b])[0].tolist()), f"row {b} escaped its mask"
        s1, _, i1, _ = _replicated_top_k(
            jnp.asarray(query[b:b + 1]), jnp.asarray(keys), jnp.asarray(values), K,
            jnp.asarray(allowed[b]), CHUNK, D, Dv, 1, N, T, use_mesh=False)
        assert np.array_equal(np.array(i1)[0], idx_b[b]), f"row {b}: batched != solo indices"
        assert np.allclose(np.array(s1)[0], scores_b[b], atol=1e-6), f"row {b}: batched != solo scores"
    print("PASS per_example_mask_batched_equals_single")


def test_build_gather_plan():
    """Gathered-bank blocks: ranked order kept, short rows padded with slot 0 (mask-off)."""
    eff = 3
    doc_id_to_flat = {1: {0, 1, 2}, 2: {3, 4, 5}, 3: {6, 7, 8}, 4: {9, 10, 11}}

    flat, n = build_gather_plan([3, 1], doc_id_to_flat, docs_per_row=3, eff_doc_len=eff)
    assert n == 2
    assert flat.tolist() == [6, 7, 8, 0, 1, 2, 0, 0, 0]     # doc 3, doc 1, then pad
    # unknown doc id skipped, not crashed; truncation past docs_per_row
    flat, n = build_gather_plan([2, 999, 4, 1], doc_id_to_flat, docs_per_row=2, eff_doc_len=eff)
    assert n == 2
    assert flat.tolist() == [3, 4, 5, 9, 10, 11]            # 999 skipped, doc 1 truncated

    # gathering keys through the plan == indexing the full bank at those slots
    bank = np.arange(12 * 4).reshape(12, 4).astype(np.float32)
    flat, n = build_gather_plan([4, 2], doc_id_to_flat, docs_per_row=2, eff_doc_len=eff)
    assert np.array_equal(bank[flat], bank[[9, 10, 11, 3, 4, 5]])
    print("PASS build_gather_plan")


def test_non_chunked_lookups_reject_2d_mask():
    """Lookup variants that assume a shared [M] mask must refuse a [B, M] one loudly."""
    import jax.numpy as jnp
    from models.memory import mem_lookup, product_key_lookup

    w = {"mem_mask": jnp.ones((2, 8), dtype=jnp.float32)}
    for fn in (mem_lookup, product_key_lookup):
        try:
            fn(None, w, {})
        except NotImplementedError:
            pass
        else:
            raise AssertionError(f"{fn.__name__} accepted a [B, M] mem_mask")
    print("PASS non_chunked_lookups_reject_2d_mask")


if __name__ == "__main__":
    test_rank_unique_doc_ids()
    test_build_doc_slot_mask()
    test_gold_coverage()
    test_masked_slots_never_retrieved()
    test_matmul_top_k_per_example_mask()
    test_per_example_mask_batched_equals_single()
    test_build_gather_plan()
    test_non_chunked_lookups_reject_2d_mask()
    print("ALL PASS")
