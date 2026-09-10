from __future__ import annotations

import itertools
import unittest
from unittest.mock import patch

import torch

from drpt.selection.advanced_solvers import exact_token_normalized_topk
from drpt.selection.state import GlobalSubsetState, LayerWiseSubsetState


class ExactTokenNormalizedTopKTests(unittest.TestCase):
    @staticmethod
    def _objective(
        scores: torch.Tensor,
        tokens: torch.Tensor,
        indices: tuple[int, ...] | torch.Tensor,
    ) -> float:
        selected = torch.as_tensor(indices, dtype=torch.long)
        return float(scores[selected].sum() / tokens[selected].sum())

    def test_matches_exhaustive_optimum_and_can_differ_from_raw_topk(self) -> None:
        # Raw top-k chooses {0, 1}, but sample 0's long response makes that set's
        # token-normalized alignment much worse than {1, 2}.
        scores = torch.tensor([10.0, 9.0, 8.0, -2.0])
        tokens = torch.tensor([100.0, 1.0, 1.0, 1.0])
        k = 2

        selected = exact_token_normalized_topk(scores, tokens, k, seed=42)
        selected_value = self._objective(scores, tokens, selected)
        exhaustive_value = max(
            self._objective(scores, tokens, candidate)
            for candidate in itertools.combinations(range(scores.numel()), k)
        )

        self.assertEqual(set(selected.tolist()), {1, 2})
        self.assertEqual(set(torch.topk(scores, k).indices.tolist()), {0, 1})
        self.assertAlmostEqual(selected_value, exhaustive_value, places=7)

    def test_endpoint_budgets_and_negative_scores(self) -> None:
        scores = torch.tensor([-5.0, -2.0, -3.0, -9.0])
        tokens = torch.tensor([10.0, 1.0, 4.0, 2.0])

        singleton = exact_token_normalized_topk(scores, tokens, 1, seed=1)
        self.assertEqual(singleton.tolist(), [0])  # -5/10 beats -2/1, -3/4, -9/2.

        full = exact_token_normalized_topk(scores, tokens, scores.numel(), seed=1)
        self.assertEqual(full.tolist(), list(range(scores.numel())))

    def test_seeded_tie_break_is_reproducible(self) -> None:
        # Every nonempty subset has objective one because score_i == token_i.
        tokens = torch.arange(1, 7, dtype=torch.float32)
        scores = tokens.clone()

        first = exact_token_normalized_topk(scores, tokens, 3, seed=123)
        second = exact_token_normalized_topk(scores, tokens, 3, seed=123)
        self.assertTrue(torch.equal(first, second))

        selections = {
            tuple(exact_token_normalized_topk(scores, tokens, 3, seed=seed).tolist())
            for seed in range(8)
        }
        self.assertGreater(len(selections), 1)

    def test_zero_token_subsets_are_infeasible_but_zero_items_are_allowed(self) -> None:
        scores = torch.tensor([100.0, 90.0, 1.0])
        tokens = torch.tensor([0.0, 0.0, 2.0])

        selected = exact_token_normalized_topk(scores, tokens, 1, seed=7)
        self.assertEqual(selected.tolist(), [2])

        with self.assertRaisesRegex(ValueError, "any valid tokens"):
            exact_token_normalized_topk(scores, torch.zeros_like(tokens), 2, seed=7)

    def test_rejects_invalid_shapes_budgets_and_values(self) -> None:
        scores = torch.tensor([1.0, 2.0, 3.0])
        tokens = torch.tensor([1.0, 2.0, 3.0])

        for k in (0, 4):
            with self.subTest(k=k), self.assertRaises(ValueError):
                exact_token_normalized_topk(scores, tokens, k, seed=0)
        with self.assertRaises(ValueError):
            exact_token_normalized_topk(scores.reshape(1, -1), tokens, 1, seed=0)
        with self.assertRaises(ValueError):
            exact_token_normalized_topk(scores, tokens[:-1], 1, seed=0)
        with self.assertRaises(FloatingPointError):
            exact_token_normalized_topk(
                torch.tensor([1.0, float("nan"), 3.0]), tokens, 1, seed=0
            )
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            exact_token_normalized_topk(
                scores, torch.tensor([1.0, -1.0, 3.0]), 1, seed=0
            )
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            exact_token_normalized_topk(
                scores, torch.tensor([1.0, float("inf"), 3.0]), 1, seed=0
            )


class TokenNormalizedOptAWiringTests(unittest.TestCase):
    @staticmethod
    def _state_kwargs() -> dict:
        return {
            "train_batch_size": 4,
            "num_layers": 1,
            "frac": 0.5,
            "lr": 1e-5,
            "device": "cpu",
            "dtype": torch.float32,
            "use_second_order": False,
            "selection_mode": "topk",
            "record_selections": False,
            "selection_variant": "score",
            "seed": 42,
            "global_step": 3,
            "optimizer_aware": True,
            "optimizer_aware_config": {
                "target_mode": "opta",
                "token_normalized_selection": True,
            },
        }

    @staticmethod
    def _set_tokens(state) -> None:
        tokens = torch.tensor([100.0, 1.0, 1.0, 1.0])
        state.set_token_counts(tokens, tokens.sum(), tokens.sum())

    def test_layerwise_opta_uses_exact_ratio_but_legacy_remains_topk(self) -> None:
        scores = torch.tensor([10.0, 9.0, 8.0, -2.0])
        normalized = LayerWiseSubsetState(
            scoring_method="reduced_ghost", **self._state_kwargs()
        )
        self._set_tokens(normalized)
        selected = normalized._select_indices(scores, layer_idx=0)
        self.assertEqual(set(selected.tolist()), {1, 2})

        legacy_kwargs = self._state_kwargs()
        legacy_kwargs["optimizer_aware_config"] = {
            "target_mode": "opta",
            "token_normalized_selection": False,
        }
        legacy = LayerWiseSubsetState(
            scoring_method="reduced_ghost", **legacy_kwargs
        )
        self._set_tokens(legacy)
        self.assertEqual(set(legacy._select_indices(scores).tolist()), {0, 1})

    def test_global_opta_uses_one_shared_exact_ratio_subset(self) -> None:
        scores = torch.tensor([10.0, 9.0, 8.0, -2.0])
        state = GlobalSubsetState(
            scoring_method="reduced_ghost", one_pass=True,
            **self._state_kwargs(),
        )
        self._set_tokens(state)
        state.grad_dot_scores.copy_(scores)
        selected = state.get_final_selection()
        self.assertEqual(set(selected.tolist()), {1, 2})
        metrics = state.get_diagnostic_metrics()
        self.assertIn("selection/global/token_normalized_objective", metrics)

    def test_soft_diagnostics_skip_exhaustive_normalized_opta(self) -> None:
        scores = torch.tensor([10.0, 9.0, 8.0, -2.0])
        weights = torch.tensor([0.0, 1.0, 1.0, 0.0])
        state = GlobalSubsetState(
            scoring_method="reduced_ghost", one_pass=True,
            **self._state_kwargs(),
        )
        self._set_tokens(state)

        with patch(
            "drpt.selection.advanced_solvers.exact_token_normalized_topk",
            side_effect=AssertionError("exhaustive comparator entered hot path"),
        ):
            state._record_soft_diagnostics("global", weights, {}, scores)
        metrics = state.get_diagnostic_metrics()

        self.assertAlmostEqual(metrics["soft/global/opta_topk_overlap"], 0.5)
        self.assertNotIn("soft/global/opta_norm_topk_overlap", metrics)
        self.assertNotIn("soft/global/opta_norm_reference_gap", metrics)


if __name__ == "__main__":
    unittest.main()
