from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from drpt.selection.advanced_solvers import (
    optimize_soft_weights,
    project_capped_simplex,
    project_probability_simplex,
    weighted_linear_gradients,
    weighted_token_scale,
)
from drpt.selection.backward import LayerWiseSubsetLinearBackward
from drpt.selection.state import LayerWiseSubsetState
from drpt.selection.strategies import (
    _advanced_solver_config,
    create_merged_batch_strategy,
    create_separate_batch_strategy,
)


class ProbabilitySimplexSolverTests(unittest.TestCase):
    def test_probability_projection_is_nonnegative_unit_mass(self) -> None:
        values = torch.tensor([-3.0, 0.2, 1.7, 8.0])
        projected = project_probability_simplex(values)
        self.assertTrue(bool((projected >= 0).all()))
        self.assertAlmostEqual(float(projected.sum()), 1.0, places=6)
        self.assertTrue(torch.allclose(
            projected, torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-7
        ))
        with self.assertRaises(ValueError):
            project_probability_simplex(torch.tensor([0.0, float("nan")]))

    def test_default_constraint_preserves_capped_k_behavior(self) -> None:
        scores = torch.tensor([4.0, 3.0, -2.0, -3.0])
        default, default_diag = optimize_soft_weights(
            lambda w: torch.dot(w, scores),
            4,
            2,
            device="cpu",
            steps=20,
            lr=0.1,
            tolerance=1e-7,
            patience=5,
        )
        explicit, explicit_diag = optimize_soft_weights(
            lambda w: torch.dot(w, scores),
            4,
            2,
            device="cpu",
            steps=20,
            lr=0.1,
            tolerance=1e-7,
            patience=5,
            constraint="capped_simplex",
        )
        self.assertTrue(torch.equal(default, explicit))
        self.assertTrue(torch.allclose(
            default, project_capped_simplex(default, 2), atol=1e-7
        ))
        self.assertEqual(default_diag["feasible_set"], "capped_simplex")
        self.assertEqual(explicit_diag["constraint_mass"], 2.0)

    def test_probability_linear_objective_can_reach_one_hot_vertex(self) -> None:
        scores = torch.tensor([4.0, -1.0, -2.0, -3.0])
        probabilities, diagnostics = optimize_soft_weights(
            lambda p: torch.dot(p, scores),
            4,
            2,  # retained only as the hard-reference budget
            device="cpu",
            steps=20,
            lr=0.1,
            tolerance=1e-7,
            patience=5,
            constraint="probability_simplex",
        )
        self.assertTrue(torch.allclose(
            probabilities, torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6
        ))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
        self.assertEqual(diagnostics["probability_simplex"], 1.0)
        self.assertEqual(diagnostics["explicit_upper_cap"], 0.0)
        self.assertAlmostEqual(diagnostics["ess"], 1.0, places=5)

    def test_token_normalization_prefers_score_per_valid_token(self) -> None:
        # Raw score prefers sample 0 (9 > 8), while the actual fractional
        # objective prefers sample 1 because 8/1 >> 9/100.
        scores = torch.tensor([9.0, 8.0])
        tokens = torch.tensor([100.0, 1.0])

        def objective(probabilities: torch.Tensor) -> torch.Tensor:
            scale = weighted_token_scale(
                probabilities, tokens, base_tokens=tokens.sum()
            )
            return scale * torch.dot(probabilities, scores)

        probabilities, _ = optimize_soft_weights(
            objective,
            2,
            1,
            device="cpu",
            steps=30,
            lr=0.1,
            tolerance=1e-8,
            patience=8,
            constraint="probability_simplex",
        )
        self.assertGreater(float(probabilities[1]), 1.0 - 1e-6)

    def test_zero_token_candidate_never_eliminates_valid_weighted_mass(self) -> None:
        # A zero-token example contributes neither numerator nor denominator.
        # It may retain non-identifiable probability mass, but the returned
        # iterate must still put positive mass on valid tokens before assembly.
        scores = torch.tensor([0.0, 2.0, -1.0])
        tokens = torch.tensor([0.0, 1.0, 2.0])

        def objective(probabilities: torch.Tensor) -> torch.Tensor:
            return weighted_token_scale(
                probabilities, tokens, base_tokens=3.0
            ) * torch.dot(probabilities, scores)

        probabilities, diagnostics = optimize_soft_weights(
            objective,
            3,
            2,
            device="cpu",
            steps=25,
            lr=0.1,
            tolerance=1e-7,
            patience=8,
            constraint="probability_simplex",
        )
        valid_mass = torch.dot(probabilities, tokens)
        self.assertGreater(float(valid_mass), 0.0)
        self.assertTrue(torch.isfinite(objective(probabilities)))
        self.assertTrue(torch.isfinite(torch.tensor(
            diagnostics["final_objective"]
        )))

        with self.assertRaisesRegex(ValueError, "no valid tokens"):
            weighted_token_scale(
                probabilities, torch.zeros_like(tokens), base_tokens=3.0
            )

    def test_weighted_token_update_is_invariant_to_global_weight_scale(self) -> None:
        grad_output = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]])
        inputs = torch.tensor([[2.0, 1.0], [-1.0, 3.0], [4.0, 0.5]])
        tokens = torch.tensor([1.0, 2.0, 5.0])
        probabilities = torch.tensor([0.55, 0.30, 0.15])
        base_tokens = torch.tensor(8.0)

        probability_update, _ = weighted_linear_gradients(
            grad_output,
            inputs,
            probabilities,
            scale=weighted_token_scale(probabilities, tokens, base_tokens),
            has_bias=False,
        )
        scaled_weights = 4.0 * probabilities
        scaled_update, _ = weighted_linear_gradients(
            grad_output,
            inputs,
            scaled_weights,
            scale=weighted_token_scale(scaled_weights, tokens, base_tokens),
            has_bias=False,
        )
        self.assertTrue(torch.allclose(
            probability_update, scaled_update, atol=1e-6
        ))


class ProbabilitySimplexIntegrationTests(unittest.TestCase):
    @staticmethod
    def _state() -> LayerWiseSubsetState:
        state = LayerWiseSubsetState(
            train_batch_size=3,
            num_layers=1,
            frac=2.0 / 3.0,
            lr=1.0,
            device="cpu",
            dtype=torch.float32,
            use_second_order=False,
            selection_mode="topk",
            record_selections=False,
            selection_variant="soft",
            solver_config={
                "soft_weighting_constraint": "probability_simplex",
                "soft_weighting_gamma": 0.0,
                "soft_replay_precision": "bf16_fp32",
            },
            seed=42,
            global_step=0,
            optimizer_aware=True,
            optimizer_aware_config={"optimizer_type": "adamw"},
            scoring_method="reduced_ghost",
        )
        tokens = torch.tensor([1.0, 2.0, 3.0])
        state.set_token_counts(tokens, tokens.sum(), torch.tensor(7.0))
        return state

    def test_backward_uses_interior_probabilities_without_topk_rounding(self) -> None:
        module = nn.Linear(2, 2, bias=False)
        hook = SimpleNamespace(
            optimizer=None,
            optimizer_aware_config={"optimizer_type": "adamw"},
            _get_module_from_idx=lambda _layer_idx: module,
            get_optimizer_group_keys=lambda: ["adamw:matrix"],
        )
        state = self._state()
        train_input = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
        train_grad_output = torch.tensor([[1.0, 2.0], [2.0, -1.0], [0.5, 3.0]])
        val_input = torch.tensor([[1.0, 1.0]])
        val_grad_output = torch.tensor([[1.0, 1.0]])
        merged_input = torch.cat((train_input, val_input), dim=0)
        merged_grad_output = torch.cat((train_grad_output, val_grad_output), dim=0)
        probabilities = torch.tensor([0.55, 0.30, 0.15])

        with patch(
            "drpt.selection.backward._make_soft_linear_objective",
            return_value=lambda p: p.sum(),
        ), patch(
            "drpt.selection.backward._soft_reference_linear_scores",
            return_value=torch.zeros(3),
        ), patch(
            "drpt.selection.backward._optimize_layer_soft_weights",
            return_value=probabilities,
        ), patch(
            "drpt.selection.advanced_solvers.weighted_linear_gradients",
            wraps=weighted_linear_gradients,
        ) as replay:
            actual_weight, actual_bias = LayerWiseSubsetLinearBackward._backward_full(
                hook,
                None,
                state,
                0,
                merged_input,
                None,
                merged_grad_output,
                False,
                False,
            )

        self.assertEqual(
            replay.call_args.kwargs["replay_precision"], "bf16_fp32"
        )
        scale = weighted_token_scale(
            probabilities, state.tokens_per_sample, base_tokens=7.0
        )
        expected_weight, _ = weighted_linear_gradients(
            train_grad_output,
            train_input,
            probabilities,
            scale=scale,
            has_bias=False,
        )
        rounded = torch.tensor([1.0, 1.0, 0.0])
        rounded_weight, _ = weighted_linear_gradients(
            train_grad_output,
            train_input,
            rounded,
            scale=weighted_token_scale(
                rounded, state.tokens_per_sample, base_tokens=7.0
            ),
            has_bias=False,
        )
        self.assertTrue(torch.allclose(actual_weight, expected_weight, atol=1e-6))
        self.assertFalse(torch.allclose(actual_weight, rounded_weight, atol=1e-5))
        self.assertIsNone(actual_bias)

    def test_state_diagnostics_distinguish_probability_constraint_and_mass(self) -> None:
        state = self._state()
        probabilities = torch.tensor([0.55, 0.30, 0.15])
        state._record_soft_diagnostics(
            "layer_0", probabilities, {}, reference_scores=None, layer_idx=0
        )
        metrics = state.get_diagnostic_metrics()
        self.assertEqual(metrics["soft/layer_0/probability_simplex"], 1.0)
        self.assertEqual(metrics["soft/layer_0/explicit_upper_cap"], 0.0)
        self.assertAlmostEqual(
            metrics["soft/layer_0/constraint_mass"], 1.0, places=6
        )
        self.assertAlmostEqual(
            metrics["soft/layer_0/mass_residual"], 0.0, places=6
        )
        self.assertGreater(metrics["soft/layer_0/entropy"], 0.0)
        self.assertGreater(metrics["soft/layer_0/ess"], 1.0)

    def test_factories_force_existing_and_probability_label_semantics(self) -> None:
        merged_hook = SimpleNamespace(wrap_nonlinear_layers=lambda: None)
        separate_hook = SimpleNamespace(check_unhooked_trainable_params=lambda: None)
        for factory, hook in (
            (create_merged_batch_strategy, merged_hook),
            (create_separate_batch_strategy, separate_hook),
        ):
            existing = factory(
                "LayerWiseSoftWeighting",
                hook,
                soft_weighting_constraint="probability_simplex",
            )
            probability = factory(
                "LayerWiseSoftProbability",
                hook,
                soft_weighting_constraint="capped_simplex",
            )
            self.assertEqual(
                _advanced_solver_config(existing)["soft_weighting_constraint"],
                "capped_simplex",
            )
            self.assertEqual(
                _advanced_solver_config(probability)["soft_weighting_constraint"],
                "probability_simplex",
            )


if __name__ == "__main__":
    unittest.main()
