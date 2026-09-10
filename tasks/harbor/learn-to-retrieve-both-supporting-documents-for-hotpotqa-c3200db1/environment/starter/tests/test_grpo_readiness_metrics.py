"""Standalone test for the GRPO-readiness post-hoc metrics (CPU, no TPU needed).

    uv run python tests/test_grpo_readiness_metrics.py

Covers scripts/analysis/grpo_readiness_metrics.py's pure functions: the unbiased pass@k
estimator (Chen et al. 2021) and grouping samples by group_id.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.analysis.grpo_readiness_metrics import pass_at_k, group_samples


def test_pass_at_k_all_correct():
    # Every sample correct -> at least one of any k is always correct.
    for k in range(1, 5):
        assert pass_at_k(4, 4, k) == 1.0
    print("PASS pass_at_k_all_correct")


def test_pass_at_k_all_incorrect():
    # No sample correct -> zero probability regardless of k.
    for k in range(1, 5):
        assert pass_at_k(4, 0, k) == 0.0
    print("PASS pass_at_k_all_incorrect")


def test_pass_at_k_monotonic_in_k():
    # Fixed n, c; pass@k should never decrease as k grows (more tries, so replace a single
    # "sequence" invariant with a monotonic comparison across increasing group_size draws.
    n, c = 8, 3
    vals = [pass_at_k(n, c, k) for k in range(1, n + 1)]
    assert all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1)), vals
    assert vals[-1] == 1.0   # k == n: guaranteed to include at least one correct sample (c > 0)
    print("PASS pass_at_k_monotonic_in_k")


def test_pass_at_1_equals_mean_accuracy():
    # pass@1 over one group == that group's fraction correct (definition check).
    n, c = 5, 2
    assert abs(pass_at_k(n, c, 1) - c / n) < 1e-9
    print("PASS pass_at_1_equals_mean_accuracy")


def test_group_samples():
    samples = [
        {"group_id": 0, "sample_idx": 0, "v": "a"},
        {"group_id": 0, "sample_idx": 1, "v": "b"},
        {"group_id": 1, "sample_idx": 0, "v": "c"},
    ]
    groups = group_samples(samples)
    assert set(groups.keys()) == {0, 1}
    assert len(groups[0]) == 2 and len(groups[1]) == 1
    print("PASS group_samples")


if __name__ == "__main__":
    test_pass_at_k_all_correct()
    test_pass_at_k_all_incorrect()
    test_pass_at_k_monotonic_in_k()
    test_pass_at_1_equals_mean_accuracy()
    test_group_samples()
    print("ALL PASS")
