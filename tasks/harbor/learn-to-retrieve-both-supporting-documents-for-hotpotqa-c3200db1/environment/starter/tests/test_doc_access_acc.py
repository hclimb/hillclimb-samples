"""
Tests for _numpy_doc_access_acc and the doc_id_to_flat_indices building logic
used by GenLargeMemEvaluator for all three corpus evals (MuSiQue, HotpotQA, MS MARCO).

These tests are fully self-contained: no JAX, no TPU, no HuggingFace required.
We import _numpy_doc_access_acc directly from evals.gen_large_mem.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from evals.gen_large_mem import (
    _numpy_doc_access_acc,
    _numpy_doc_access_acc_per_example,
    _numpy_doc_hit_rate,
    _numpy_doc_token_hit_rate,
    _numpy_doc_token_hit_rate_per_example,
    _numpy_doc_gen_token_hit_rate,
    _numpy_doc_gen_token_hit_rate_per_example,
    _numpy_doc_hit_rate_gen,
    _numpy_doc_hit_rate_gen_per_example,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_indices(B, H, S, K, values):
    """Fill a (B,H,S,K) array with the given flat values broadcast to all heads/positions."""
    arr = np.full((B, H, S, K), 0, dtype=np.int64)
    for b, row in enumerate(values):
        arr[b] = row
    return arr


def all_valid(n):
    """mem_validity array where every slot is valid."""
    return np.ones(n, dtype=np.float32)


def all_active(B, S):
    """loss_mask where every position is active."""
    return np.ones((B, S), dtype=np.float32)


def build_doc_id_to_flat_indices(corpus_doc_ids, eff_doc_len=1):
    """Replicate the mapping built in GenLargeMemEvaluator.evaluate (lines 155-163)."""
    mapping = {}
    for chunk_idx, doc_id in enumerate(corpus_doc_ids):
        doc_id = int(doc_id)
        if doc_id not in mapping:
            mapping[doc_id] = set()
        base = chunk_idx * eff_doc_len
        for off in range(eff_doc_len):
            mapping[doc_id].add(base + off)
    return mapping


# ---------------------------------------------------------------------------
# _numpy_doc_access_acc — basic correctness
# ---------------------------------------------------------------------------

class TestBasicCorrectness:
    def test_perfect_hit(self):
        """Every retrieved index is in the positive set → acc = 1.0."""
        B, H, S, K = 1, 1, 1, 3
        indices = make_indices(B, H, S, K, [[[5, 6, 7]]])
        pos_sets = [{5, 6, 7}]
        mem_validity = all_valid(10)
        loss_mask = all_active(B, S)
        assert _numpy_doc_access_acc(indices, pos_sets, loss_mask, mem_validity) == pytest.approx(1.0)

    def test_no_hit(self):
        """No retrieved index is in the positive set → acc = 0.0."""
        B, H, S, K = 1, 1, 1, 3
        indices = make_indices(B, H, S, K, [[[0, 1, 2]]])
        pos_sets = [{5, 6, 7}]
        mem_validity = all_valid(10)
        loss_mask = all_active(B, S)
        assert _numpy_doc_access_acc(indices, pos_sets, loss_mask, mem_validity) == pytest.approx(0.0)

    def test_partial_hit(self):
        """2 of 4 retrieved indices are in the positive set → acc = 0.5."""
        B, H, S, K = 1, 1, 1, 4
        indices = make_indices(B, H, S, K, [[[0, 1, 5, 6]]])
        pos_sets = [{5, 6}]
        mem_validity = all_valid(10)
        loss_mask = all_active(B, S)
        assert _numpy_doc_access_acc(indices, pos_sets, loss_mask, mem_validity) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Per-example metrics
# ---------------------------------------------------------------------------

class TestPerExampleMetrics:
    def test_doc_access_acc_per_example(self):
        B, H, S, K = 2, 1, 1, 2
        indices = make_indices(B, H, S, K, [
            [[5, 1]],
            [[0, 2]],
        ])
        pos_sets = [{5}, {8}]
        scores = _numpy_doc_access_acc_per_example(
            indices, pos_sets, all_active(B, S), all_valid(10)
        )
        assert scores == pytest.approx([0.5, 0.0])

    def test_doc_hit_rate_gen_per_example(self):
        B, H, S, K = 3, 1, 2, 2
        indices = make_indices(B, H, S, K, [
            [[5, 1], [0, 0]],
            [[0, 1], [2, 3]],
            [[4, 4], [4, 4]],
        ])
        pos_sets = [{5}, {9}, set()]
        gen_lengths = np.array([2, 2, 2], dtype=np.int32)
        scores = _numpy_doc_hit_rate_gen_per_example(
            indices, pos_sets, gen_lengths, all_valid(10)
        )
        assert scores[:2] == pytest.approx([1.0, 0.0])
        assert scores[2] is None

    def test_doc_gen_token_hit_rate_per_example(self):
        B, H, S, K = 2, 1, 3, 2
        indices = make_indices(B, H, S, K, [
            [[5, 0], [1, 2], [5, 9]],
            [[0, 1], [0, 1], [0, 1]],
        ])
        pos_sets = [{5}, {7}]
        gen_lengths = np.array([3, 2], dtype=np.int32)
        scores = _numpy_doc_gen_token_hit_rate_per_example(
            indices, pos_sets, gen_lengths, all_valid(10)
        )
        assert scores == pytest.approx([2 / 3, 0.0])

    def test_prompt_token_hit_rate_per_example_handles_missing_positive_docs(self):
        B, H, S, K = 2, 1, 2, 2
        indices = make_indices(B, H, S, K, [
            [[5, 0], [0, 0]],
            [[1, 2], [3, 4]],
        ])
        pos_sets = [{5}, set()]
        scores = _numpy_doc_token_hit_rate_per_example(
            indices, pos_sets, all_active(B, S), all_valid(10)
        )
        assert scores[0] == pytest.approx(0.5)
        assert scores[1] is None


# ---------------------------------------------------------------------------
# mem_validity filtering
# ---------------------------------------------------------------------------

class TestMemValidity:
    def test_invalid_slots_excluded(self):
        """Hits on invalid memory slots must not count toward numerator or denominator."""
        B, H, S, K = 1, 1, 1, 4
        # indices 0,1 are in pos_set; index 2 is also in pos_set but invalid;
        # index 3 is not in pos_set and valid.
        indices = make_indices(B, H, S, K, [[[0, 1, 2, 3]]])
        pos_sets = [{0, 1, 2}]
        mem_validity = np.array([1, 1, 0, 1], dtype=np.float32)  # slot 2 invalid
        loss_mask = all_active(B, S)
        # valid positions: 0,1,3  →  hits among valid: 0,1  →  acc = 2/3
        assert _numpy_doc_access_acc(indices, pos_sets, loss_mask, mem_validity) == pytest.approx(2 / 3)

    def test_all_invalid_returns_zero(self):
        """When every memory slot is invalid, acc = 0.0 (no denominator)."""
        B, H, S, K = 1, 1, 1, 3
        indices = make_indices(B, H, S, K, [[[0, 1, 2]]])
        pos_sets = [{0, 1, 2}]
        mem_validity = np.zeros(5, dtype=np.float32)
        loss_mask = all_active(B, S)
        assert _numpy_doc_access_acc(indices, pos_sets, loss_mask, mem_validity) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# loss_mask (prompt_pad_mask) filtering
# ---------------------------------------------------------------------------

class TestLossMask:
    def test_inactive_positions_excluded(self):
        """Positions where loss_mask=0 must not contribute to numerator or denominator."""
        B, H, S, K = 1, 1, 3, 2
        # position 0: active, retrieves [5,6] (in pos_set) → 2 hits / 2 valid
        # position 1: inactive, retrieves [5,6]            → excluded
        # position 2: active, retrieves [0,1] (not in pos_set) → 0 hits / 2 valid
        indices = np.array([[[[5, 6], [5, 6], [0, 1]]]], dtype=np.int64)  # (1,1,3,2)
        pos_sets = [{5, 6}]
        mem_validity = all_valid(10)
        loss_mask = np.array([[1, 0, 1]], dtype=np.float32)
        # valid+active: position 0 (2 slots) + position 2 (2 slots) = 4
        # hits: position 0 → 2
        assert _numpy_doc_access_acc(indices, pos_sets, loss_mask, mem_validity) == pytest.approx(2 / 4)

    def test_all_inactive_returns_zero(self):
        B, H, S, K = 1, 1, 2, 2
        indices = make_indices(B, H, S, K, [[[5, 6], [5, 6]]])
        pos_sets = [{5, 6}]
        mem_validity = all_valid(10)
        loss_mask = np.zeros((B, S), dtype=np.float32)
        assert _numpy_doc_access_acc(indices, pos_sets, loss_mask, mem_validity) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_pos_set(self):
        """Empty positive set → no hits regardless of retrieval."""
        B, H, S, K = 1, 1, 1, 3
        indices = make_indices(B, H, S, K, [[[0, 1, 2]]])
        pos_sets = [set()]
        mem_validity = all_valid(5)
        loss_mask = all_active(B, S)
        assert _numpy_doc_access_acc(indices, pos_sets, loss_mask, mem_validity) == pytest.approx(0.0)

    def test_batch_size_2(self):
        """Two batch items with different pos_sets are scored independently."""
        B, H, S, K = 2, 1, 1, 2
        indices = np.array([[[[0, 1]]], [[[0, 1]]]], dtype=np.int64)
        # item 0: pos_set = {0,1} → 2 hits / 2 valid
        # item 1: pos_set = {}    → 0 hits / 2 valid
        pos_sets = [{0, 1}, set()]
        mem_validity = all_valid(5)
        loss_mask = all_active(B, S)
        # mean over batches is NOT computed here — raw total: 2 hits / 4 valid
        assert _numpy_doc_access_acc(indices, pos_sets, loss_mask, mem_validity) == pytest.approx(0.5)

    def test_multi_head(self):
        """Multi-head (H=2): hits are counted across all heads."""
        B, H, S, K = 1, 2, 1, 2
        # head 0 retrieves [0, 1] — both in pos_set
        # head 1 retrieves [2, 3] — neither in pos_set
        indices = np.array([[[[0, 1]], [[2, 3]]]], dtype=np.int64)  # (1,2,1,2)
        pos_sets = [{0, 1}]
        mem_validity = all_valid(5)
        loss_mask = all_active(B, S)
        # total valid: 4 (2 heads × 1 pos × 2 K), hits: 2
        assert _numpy_doc_access_acc(indices, pos_sets, loss_mask, mem_validity) == pytest.approx(0.5)

    def test_multi_pos_doc(self):
        """pos_set spanning multiple documents (union of their flat indices)."""
        B, H, S, K = 1, 1, 1, 4
        # doc A owns flat indices {10, 11}; doc B owns {20, 21}
        indices = make_indices(B, H, S, K, [[[10, 11, 20, 99]]])
        pos_sets = [{10, 11, 20, 21}]   # union of doc A and doc B
        mem_validity = all_valid(100)
        loss_mask = all_active(B, S)
        # 3 of 4 indices are in pos_set
        assert _numpy_doc_access_acc(indices, pos_sets, loss_mask, mem_validity) == pytest.approx(3 / 4)


# ---------------------------------------------------------------------------
# doc_id_to_flat_indices building — replicates the evaluator's mapping logic
# ---------------------------------------------------------------------------

class TestDocIdToFlatIndices:
    def test_single_doc_single_chunk(self):
        mapping = build_doc_id_to_flat_indices([42])
        assert mapping == {42: {0}}

    def test_two_docs_one_chunk_each(self):
        mapping = build_doc_id_to_flat_indices([0, 1])
        assert mapping == {0: {0}, 1: {1}}

    def test_same_doc_multiple_chunks(self):
        """Multiple chunks with the same doc_id must all land in one set."""
        mapping = build_doc_id_to_flat_indices([7, 7, 7])
        assert mapping == {7: {0, 1, 2}}

    def test_eff_doc_len_stride(self):
        """eff_doc_len=2 means each corpus chunk maps to 2 consecutive flat indices."""
        mapping = build_doc_id_to_flat_indices([0, 1, 0], eff_doc_len=2)
        # chunk 0 (doc 0): flat {0,1}; chunk 1 (doc 1): flat {2,3}; chunk 2 (doc 0): flat {4,5}
        assert mapping == {0: {0, 1, 4, 5}, 1: {2, 3}}

    def test_interleaved_doc_ids(self):
        mapping = build_doc_id_to_flat_indices([3, 5, 3, 5])
        assert mapping == {3: {0, 2}, 5: {1, 3}}


# ---------------------------------------------------------------------------
# End-to-end simulation of corpus eval scenarios (MuSiQue / HotpotQA / MSMARCO)
# ---------------------------------------------------------------------------

class TestCorpusEvalSimulation:
    """
    Simulates the full pipeline that GenLargeMemEvaluator runs:
      1. Build doc_id_to_flat_indices from the corpus chunk stream.
      2. Look up pos_doc_ids from the QA batch to build pos_sets.
      3. Feed retrieved indices into _numpy_doc_access_acc.
    """

    def _run(self, corpus_doc_ids, pos_doc_ids_per_item, retrieved_per_item,
             eff_doc_len=1, mem_size=None, active_slots=None):
        """Helper that mirrors what GenLargeMemEvaluator does at eval time."""
        mapping = build_doc_id_to_flat_indices(corpus_doc_ids, eff_doc_len)

        B = len(pos_doc_ids_per_item)
        H, S, K = 1, 1, len(retrieved_per_item[0])
        indices = np.array([[[[r for r in retrieved_per_item[i]]]]
                            for i in range(B)], dtype=np.int64)

        pos_sets = []
        for ids in pos_doc_ids_per_item:
            pos_flat = set()
            for doc_id in ids:
                if doc_id >= 0:
                    pos_flat.update(mapping.get(doc_id, set()))
            pos_sets.append(pos_flat)

        M = mem_size or (max(corpus_doc_ids) + 1) * eff_doc_len + 10
        mem_validity = all_valid(M)
        if active_slots is not None:
            for slot in range(M):
                if slot not in active_slots:
                    mem_validity[slot] = 0.0

        loss_mask = all_active(B, S)
        return _numpy_doc_access_acc(indices, pos_sets, loss_mask, mem_validity)

    # -- MuSiQue-style: supporting passages indexed by integer corpus IDs --

    def test_musique_perfect_retrieval(self):
        """Model retrieves exactly the flat indices that belong to the positive docs."""
        # Corpus: 5 chunks, each belonging to a different doc
        corpus_doc_ids = [10, 20, 30, 40, 50]
        # Query: docs 10 and 30 are supporting → flat indices {0} and {2}
        acc = self._run(
            corpus_doc_ids,
            pos_doc_ids_per_item=[[10, 30]],
            retrieved_per_item=[[0, 2]],
            mem_size=10,
        )
        assert acc == pytest.approx(1.0)

    def test_musique_no_retrieval(self):
        """Model retrieves distractor flat indices only → acc = 0."""
        corpus_doc_ids = [10, 20, 30, 40, 50]
        acc = self._run(
            corpus_doc_ids,
            pos_doc_ids_per_item=[[10, 30]],
            retrieved_per_item=[[1, 3, 4]],   # docs 20, 40, 50
            mem_size=10,
        )
        assert acc == pytest.approx(0.0)

    def test_musique_missing_pos_doc_id(self):
        """pos_doc_id = -1 means the doc wasn't found in the corpus; treated as no constraint."""
        corpus_doc_ids = [10, 20, 30]
        acc = self._run(
            corpus_doc_ids,
            pos_doc_ids_per_item=[[-1]],   # sentinel: passage not in corpus
            retrieved_per_item=[[0, 1]],
            mem_size=5,
        )
        # -1 is skipped → pos_flat is empty → no hits counted
        assert acc == pytest.approx(0.0)

    # -- HotpotQA-style: distractor docs present; positive docs are a subset --

    def test_hotpotqa_batch_mixed(self):
        """Two queries: one hits perfectly, one misses → aggregate acc = 0.5."""
        # 6 corpus chunks, docs 0–5 each owning one flat index
        corpus_doc_ids = [0, 1, 2, 3, 4, 5]
        acc = self._run(
            corpus_doc_ids,
            pos_doc_ids_per_item=[[0, 1], [2, 3]],
            retrieved_per_item=[
                [0, 1],   # perfect for item 0
                [4, 5],   # misses for item 1 (distractors 4,5)
            ],
            mem_size=8,
        )
        # item 0: 2 hits / 2 valid; item 1: 0 hits / 2 valid → total 2/4 = 0.5
        assert acc == pytest.approx(0.5)

    # -- MS MARCO-style: single positive passage per query --

    def test_msmarco_single_passage(self):
        corpus_doc_ids = [100, 200, 300]
        acc = self._run(
            corpus_doc_ids,
            pos_doc_ids_per_item=[[200]],
            retrieved_per_item=[[0, 1, 2]],   # 1 retrieves flat index 1 = doc 200
            mem_size=5,
        )
        # flat index 1 = doc 200 (positive); indices 0 and 2 are distractors
        assert acc == pytest.approx(1 / 3)

    def test_msmarco_eff_doc_len_2(self):
        """With eff_doc_len=2, a corpus chunk maps to 2 consecutive flat indices."""
        # 3 corpus chunks: doc 0 → flat {0,1}, doc 1 → flat {2,3}, doc 2 → flat {4,5}
        corpus_doc_ids = [0, 1, 2]
        acc = self._run(
            corpus_doc_ids,
            pos_doc_ids_per_item=[[1]],   # doc 1 → flat {2,3}
            retrieved_per_item=[[2, 3, 0]],
            eff_doc_len=2,
            mem_size=8,
        )
        # indices 2 and 3 hit; index 0 misses → acc = 2/3
        assert acc == pytest.approx(2 / 3)

    # -- Validity + ID-based matching interaction --

    def test_pos_slot_marked_invalid(self):
        """If the positive document's memory slot is marked invalid, it shouldn't count as hit."""
        corpus_doc_ids = [0, 1]
        # doc 0 → flat index 0; mark it invalid
        acc = self._run(
            corpus_doc_ids,
            pos_doc_ids_per_item=[[0]],
            retrieved_per_item=[[0, 1]],
            mem_size=5,
            active_slots={1, 2, 3, 4},   # slot 0 is NOT active
        )
        # index 0 is in pos_set but invalid → excluded from numerator and denominator
        # index 1 is valid but not in pos_set
        # total valid: 1 (just index 1), hits: 0
        assert acc == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _numpy_doc_hit_rate — per-example binary recall
# ---------------------------------------------------------------------------

class TestDocHitRate:
    """_numpy_doc_hit_rate returns fraction of examples where any lookup hit."""

    def test_all_examples_hit(self):
        B, H, S, K = 2, 1, 1, 2
        indices = np.array([[[[5, 6]]], [[[5, 6]]]], dtype=np.int64)
        pos_sets = [{5, 6}, {5, 6}]
        assert _numpy_doc_hit_rate(indices, pos_sets, all_active(B, S), all_valid(10)) == pytest.approx(1.0)

    def test_no_examples_hit(self):
        B, H, S, K = 2, 1, 1, 2
        indices = np.array([[[[0, 1]]], [[[0, 1]]]], dtype=np.int64)
        pos_sets = [{5, 6}, {5, 6}]
        assert _numpy_doc_hit_rate(indices, pos_sets, all_active(B, S), all_valid(10)) == pytest.approx(0.0)

    def test_half_examples_hit(self):
        """One example hits, one misses → hit_rate = 0.5."""
        B, H, S, K = 2, 1, 1, 3
        # item 0: retrieves [5,6,7] → all in pos_set → hit
        # item 1: retrieves [0,1,2] → none in pos_set → miss
        indices = np.array([[[[5, 6, 7]]], [[[0, 1, 2]]]], dtype=np.int64)
        pos_sets = [{5, 6, 7}, {5, 6, 7}]
        assert _numpy_doc_hit_rate(indices, pos_sets, all_active(B, S), all_valid(10)) == pytest.approx(0.5)

    def test_single_hit_in_topk_enough(self):
        """A single correct index among K is sufficient for a hit."""
        B, H, S, K = 1, 1, 1, 4
        # Only index 5 is positive; other 3 are distractors
        indices = make_indices(B, H, S, K, [[[0, 1, 2, 5]]])
        pos_sets = [{5}]
        assert _numpy_doc_hit_rate(indices, pos_sets, all_active(B, S), all_valid(10)) == pytest.approx(1.0)

    def test_multi_head_any_head_sufficient(self):
        """Hit in any one head is sufficient; other heads can all miss."""
        B, H, S, K = 1, 3, 1, 2
        # head 0: [0, 1] — miss; head 1: [0, 1] — miss; head 2: [5, 6] — hit
        indices = np.array([[[[0, 1]], [[0, 1]], [[5, 6]]]], dtype=np.int64)
        pos_sets = [{5, 6}]
        assert _numpy_doc_hit_rate(indices, pos_sets, all_active(B, S), all_valid(10)) == pytest.approx(1.0)

    def test_empty_pos_set_excluded_from_denominator(self):
        """Examples with an empty positive set are skipped (don't count in denominator)."""
        B, H, S, K = 3, 1, 1, 2
        indices = np.array([[[[5, 6]]], [[[0, 1]]], [[[5, 6]]]], dtype=np.int64)
        # item 0: hit; item 1: empty pos_set → excluded; item 2: hit
        pos_sets = [{5, 6}, set(), {5, 6}]
        # valid_count = 2 (items 0 and 2); hits = 2 → rate = 1.0
        assert _numpy_doc_hit_rate(indices, pos_sets, all_active(B, S), all_valid(10)) == pytest.approx(1.0)

    def test_invalid_slot_suppresses_hit(self):
        """If the only positive index is in an invalid memory slot, the example misses."""
        B, H, S, K = 1, 1, 1, 1
        indices = make_indices(B, H, S, K, [[[5]]])
        pos_sets = [{5}]
        mem_validity = np.zeros(10, dtype=np.float32)   # all invalid
        assert _numpy_doc_hit_rate(indices, pos_sets, all_active(B, S), mem_validity) == pytest.approx(0.0)

    def test_different_from_access_acc(self):
        """doc_hit_rate is 1.0 when only 1-of-K hits; doc_access_acc would be 1/K."""
        B, H, S, K = 1, 1, 1, 4
        indices = make_indices(B, H, S, K, [[[0, 1, 2, 5]]])
        pos_sets = [{5}]
        mem_validity = all_valid(10)
        loss_mask = all_active(B, S)
        hit_rate = _numpy_doc_hit_rate(indices, pos_sets, loss_mask, mem_validity)
        access_acc = _numpy_doc_access_acc(indices, pos_sets, loss_mask, mem_validity)
        assert hit_rate == pytest.approx(1.0)
        assert access_acc == pytest.approx(1 / 4)


# ---------------------------------------------------------------------------
# _numpy_doc_token_hit_rate — token-level prompt hit rate
# ---------------------------------------------------------------------------

class TestDocTokenHitRate:
    """_numpy_doc_token_hit_rate returns fraction of active tokens with a hit."""

    def test_all_positions_hit(self):
        B, H, S, K = 1, 1, 3, 2
        indices = np.array([[[[5, 6], [5, 6], [5, 6]]]], dtype=np.int64)
        pos_sets = [{5, 6}]
        r = _numpy_doc_token_hit_rate(indices, pos_sets, all_active(B, S), all_valid(10))
        assert r == pytest.approx(1.0)

    def test_no_positions_hit(self):
        B, H, S, K = 1, 1, 3, 2
        indices = np.array([[[[0, 1], [0, 1], [0, 1]]]], dtype=np.int64)
        pos_sets = [{5, 6}]
        r = _numpy_doc_token_hit_rate(indices, pos_sets, all_active(B, S), all_valid(10))
        assert r == pytest.approx(0.0)

    def test_half_positions_hit(self):
        """2 of 4 positions hit → rate = 0.5."""
        B, H, S, K = 1, 1, 4, 1
        # positions 0,1 retrieve pos index; positions 2,3 retrieve distractor
        indices = np.array([[[[5], [5], [0], [0]]]], dtype=np.int64)
        pos_sets = [{5}]
        r = _numpy_doc_token_hit_rate(indices, pos_sets, all_active(B, S), all_valid(10))
        assert r == pytest.approx(0.5)

    def test_inactive_positions_excluded(self):
        """Positions where loss_mask=0 don't count in numerator or denominator."""
        B, H, S, K = 1, 1, 3, 1
        # pos 0: active, hit; pos 1: inactive, hit; pos 2: active, miss
        indices = np.array([[[[5], [5], [0]]]], dtype=np.int64)
        pos_sets = [{5}]
        loss_mask = np.array([[1, 0, 1]], dtype=np.float32)
        r = _numpy_doc_token_hit_rate(indices, pos_sets, loss_mask, all_valid(10))
        # active positions: 0 (hit), 2 (miss) → 1/2
        assert r == pytest.approx(0.5)

    def test_any_head_hit_at_position_counts(self):
        """At a given position, a hit in any head counts for that position."""
        B, H, S, K = 1, 2, 1, 1
        # head 0: [0] miss; head 1: [5] hit
        indices = np.array([[[[0]], [[5]]]], dtype=np.int64)
        pos_sets = [{5}]
        r = _numpy_doc_token_hit_rate(indices, pos_sets, all_active(B, S), all_valid(10))
        assert r == pytest.approx(1.0)

    def test_greater_than_or_equal_to_access_acc(self):
        """hit_rate_by_position >= doc_access_acc always (it's coarser)."""
        B, H, S, K = 1, 2, 4, 3
        rng = np.random.default_rng(42)
        indices = rng.integers(0, 20, size=(B, H, S, K)).astype(np.int64)
        pos_sets = [{5, 10, 15}]
        mem_validity = all_valid(20)
        loss_mask = all_active(B, S)
        hr_pos = _numpy_doc_token_hit_rate(indices, pos_sets, loss_mask, mem_validity)
        acc = _numpy_doc_access_acc(indices, pos_sets, loss_mask, mem_validity)
        assert hr_pos >= acc - 1e-9  # hit_rate_by_position is always >= access_acc

    def test_empty_pos_set_returns_zero(self):
        B, H, S, K = 1, 1, 2, 2
        indices = make_indices(B, H, S, K, [[[5, 6], [5, 6]]])
        pos_sets = [set()]
        r = _numpy_doc_token_hit_rate(indices, pos_sets, all_active(B, S), all_valid(10))
        assert r == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _numpy_doc_gen_token_hit_rate — token-level generated hit rate
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _numpy_doc_hit_rate_gen — decoding example-level recall
# ---------------------------------------------------------------------------

class TestDocHitRateGen:
    """_numpy_doc_hit_rate_gen: fraction of examples where any decode step hit."""

    def test_all_examples_hit(self):
        B, H, S_gen, K = 2, 1, 4, 1
        indices = np.full((B, H, S_gen, K), 5, dtype=np.int64)
        pos_sets = [{5}, {5}]
        gl = np.array([4, 4])
        assert _numpy_doc_hit_rate_gen(indices, pos_sets, gl, all_valid(10)) == pytest.approx(1.0)

    def test_no_examples_hit(self):
        B, H, S_gen, K = 2, 1, 4, 1
        indices = np.zeros((B, H, S_gen, K), dtype=np.int64)
        pos_sets = [{5}, {5}]
        gl = np.array([4, 4])
        assert _numpy_doc_hit_rate_gen(indices, pos_sets, gl, all_valid(10)) == pytest.approx(0.0)

    def test_half_examples_hit(self):
        B, H, S_gen, K = 2, 1, 3, 1
        indices = np.zeros((B, H, S_gen, K), dtype=np.int64)
        indices[0] = 5   # example 0 hits; example 1 misses
        pos_sets = [{5}, {5}]
        gl = np.array([3, 3])
        assert _numpy_doc_hit_rate_gen(indices, pos_sets, gl, all_valid(10)) == pytest.approx(0.5)

    def test_hit_beyond_gen_length_not_counted(self):
        """Tokens beyond gen_length are ignored even if they would be a hit."""
        B, H, S_gen, K = 1, 1, 4, 1
        indices = np.zeros((B, H, S_gen, K), dtype=np.int64)
        indices[0, 0, 3, 0] = 5   # hit only at position 3
        pos_sets = [{5}]
        gl = np.array([2])         # gen stops at position 2 → position 3 excluded
        assert _numpy_doc_hit_rate_gen(indices, pos_sets, gl, all_valid(10)) == pytest.approx(0.0)

    def test_hit_within_gen_length_counted(self):
        B, H, S_gen, K = 1, 1, 4, 1
        indices = np.zeros((B, H, S_gen, K), dtype=np.int64)
        indices[0, 0, 1, 0] = 5   # hit at position 1
        pos_sets = [{5}]
        gl = np.array([4])
        assert _numpy_doc_hit_rate_gen(indices, pos_sets, gl, all_valid(10)) == pytest.approx(1.0)

    def test_single_hit_anywhere_sufficient(self):
        """One hit at any position/head/k is enough for the example to count."""
        B, H, S_gen, K = 1, 2, 5, 3
        indices = np.zeros((B, H, S_gen, K), dtype=np.int64)
        indices[0, 1, 4, 2] = 5   # single hit buried at the last slot
        pos_sets = [{5}]
        gl = np.array([5])
        assert _numpy_doc_hit_rate_gen(indices, pos_sets, gl, all_valid(10)) == pytest.approx(1.0)

    def test_invalid_slot_suppresses_hit(self):
        B, H, S_gen, K = 1, 1, 2, 1
        indices = np.full((B, H, S_gen, K), 5, dtype=np.int64)
        pos_sets = [{5}]
        gl = np.array([2])
        assert _numpy_doc_hit_rate_gen(indices, pos_sets, gl, np.zeros(10)) == pytest.approx(0.0)

    def test_empty_pos_set_excluded(self):
        B, H, S_gen, K = 2, 1, 2, 1
        indices = np.full((B, H, S_gen, K), 5, dtype=np.int64)
        pos_sets = [{5}, set()]    # example 1 has no positive set → excluded
        gl = np.array([2, 2])
        # only example 0 counts → 1 hit / 1 valid = 1.0
        assert _numpy_doc_hit_rate_gen(indices, pos_sets, gl, all_valid(10)) == pytest.approx(1.0)

    def test_differs_from_prompt_hit_rate(self):
        """doc_hit_rate_gen and prompt-based _numpy_doc_hit_rate can differ."""
        B, H, S, K = 1, 1, 3, 1
        # Prompt indices never hit; generation indices always hit
        prompt_indices = np.zeros((B, H, S, K), dtype=np.int64)
        gen_indices    = np.full((B, H, S, K), 5, dtype=np.int64)
        pos_sets = [{5}]
        loss_mask = all_active(B, S)
        gl = np.array([S])
        prompt_rate = _numpy_doc_hit_rate(prompt_indices, pos_sets, loss_mask, all_valid(10))
        gen_rate    = _numpy_doc_hit_rate_gen(gen_indices, pos_sets, gl, all_valid(10))
        assert prompt_rate == pytest.approx(0.0)
        assert gen_rate    == pytest.approx(1.0)


def make_gen_indices(B, H, S_gen, K, val):
    """Fill (B, H, S_gen, K) with a constant value."""
    return np.full((B, H, S_gen, K), val, dtype=np.int64)


class TestDocGenTokenHitRate:
    """_numpy_doc_gen_token_hit_rate returns fraction of generated tokens with a hit."""

    def test_all_positions_hit_full_gen(self):
        B, H, S_gen, K = 2, 1, 4, 1
        indices = make_gen_indices(B, H, S_gen, K, 5)   # always retrieves index 5
        pos_sets = [{5}, {5}]
        gl = np.array([4, 4])
        rate = _numpy_doc_gen_token_hit_rate(
            indices, pos_sets, gl, all_valid(10)
        )
        assert rate == pytest.approx(1.0)

    def test_no_positions_hit(self):
        B, H, S_gen, K = 2, 1, 3, 1
        indices = make_gen_indices(B, H, S_gen, K, 0)   # always retrieves index 0
        pos_sets = [{5}, {5}]
        gl = np.array([3, 3])
        rate = _numpy_doc_gen_token_hit_rate(
            indices, pos_sets, gl, all_valid(10)
        )
        assert rate == pytest.approx(0.0)

    def test_gen_length_truncates_valid_counts(self):
        B, H, S_gen, K = 2, 1, 4, 1
        indices = make_gen_indices(B, H, S_gen, K, 5)
        pos_sets = [{5}, {5}]
        gl = np.array([4, 2])
        rate = _numpy_doc_gen_token_hit_rate(
            indices, pos_sets, gl, all_valid(10)
        )
        assert rate == pytest.approx(1.0)

    def test_partial_hit_across_examples(self):
        B, H, S_gen, K = 2, 1, 3, 1
        indices = np.zeros((B, H, S_gen, K), dtype=np.int64)
        indices[0] = 5
        indices[1] = 0
        pos_sets = [{5}, {5}]
        gl = np.array([3, 3])
        rate = _numpy_doc_gen_token_hit_rate(
            indices, pos_sets, gl, all_valid(10)
        )
        assert rate == pytest.approx(0.5)

    def test_hit_only_at_specific_positions(self):
        B, H, S_gen, K = 1, 1, 4, 1
        indices = np.zeros((B, H, S_gen, K), dtype=np.int64)
        indices[0, 0, 1, 0] = 5   # only position 1 retrieves the positive index
        pos_sets = [{5}]
        gl = np.array([4])
        rate = _numpy_doc_gen_token_hit_rate(
            indices, pos_sets, gl, all_valid(10)
        )
        assert rate == pytest.approx(0.25)

    def test_multi_head_any_head_sufficient(self):
        B, H, S_gen, K = 1, 2, 2, 1
        indices = np.zeros((B, H, S_gen, K), dtype=np.int64)
        indices[0, 1, :, 0] = 5
        pos_sets = [{5}]
        gl = np.array([2])
        rate = _numpy_doc_gen_token_hit_rate(
            indices, pos_sets, gl, all_valid(10)
        )
        assert rate == pytest.approx(1.0)

    def test_invalid_memory_slot_suppresses_hit(self):
        B, H, S_gen, K = 1, 1, 2, 1
        indices = make_gen_indices(B, H, S_gen, K, 5)
        pos_sets = [{5}]
        gl = np.array([2])
        mem_validity = np.zeros(10, dtype=np.float32)   # all invalid
        rate = _numpy_doc_gen_token_hit_rate(
            indices, pos_sets, gl, mem_validity
        )
        assert rate == pytest.approx(0.0)

    def test_empty_pos_set_skipped(self):
        B, H, S_gen, K = 2, 1, 2, 1
        indices = make_gen_indices(B, H, S_gen, K, 5)
        pos_sets = [{5}, set()]
        gl = np.array([2, 2])
        rate = _numpy_doc_gen_token_hit_rate(
            indices, pos_sets, gl, all_valid(10)
        )
        assert rate == pytest.approx(1.0)
