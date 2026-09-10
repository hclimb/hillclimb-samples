from __future__ import annotations

import itertools
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from drpt.selection.advanced_solvers import (
    compute_spectral_modes,
    compute_spectral_modes_with_values,
    muon_live_candidate_transform,
    optimize_batched_linear_soft_weights,
    optimize_soft_weights,
    optimize_soft_weights_streaming,
    optimize_soft_weights_with_grad,
    project_capped_simplex,
    project_capped_simplex_batched,
    project_probability_simplex,
    project_probability_simplex_batched,
    seeded_spectral_saturation_greedy,
    seeded_topk,
    spectral_linear_mode_support,
    spectral_linear_scores,
    spectral_saturated_objective,
    weighted_embedding_gradient,
    weighted_linear_gradients,
    weighted_token_scale,
)
from drpt.optimizer import zeropower_via_newton_schulz
from drpt.validation_cache import ValidationCache
from drpt.selection.utils import compute_embedding_val_gradient
from drpt.hook import GradientHook


class CappedSimplexTests(unittest.TestCase):
    def assert_feasible(self, value: torch.Tensor, mass: float) -> None:
        self.assertTrue(bool((value >= 0).all()))
        self.assertTrue(bool((value <= 1).all()))
        self.assertAlmostEqual(float(value.sum()), mass, places=6)

    def test_projection_handles_negative_values_and_endpoint_budgets(self) -> None:
        values = torch.tensor([-8.0, -2.0, 0.5, 3.0, 9.0])
        for mass in (1, 2, values.numel()):
            projected = project_capped_simplex(values, mass)
            self.assert_feasible(projected, mass)
        self.assertTrue(torch.equal(
            project_capped_simplex(values, values.numel()), torch.ones_like(values)
        ))

    def test_projection_is_euclidean_solution(self) -> None:
        values = torch.tensor([-1.2, 0.1, 0.4, 2.0], dtype=torch.float64)
        result = project_capped_simplex(values, 2)
        # KKT: all unsaturated coordinates share values - projection.
        free = (result > 1e-10) & (result < 1.0 - 1e-10)
        offsets = values[free] - result[free]
        if offsets.numel() > 1:
            self.assertLess(float((offsets - offsets[0]).abs().max()), 1e-10)

    def test_invalid_projection_inputs_fail(self) -> None:
        with self.assertRaises(ValueError):
            project_capped_simplex(torch.tensor([1.0, float("nan")]), 1)
        with self.assertRaises(ValueError):
            project_capped_simplex(torch.ones(2), 3)

    def test_batched_projectors_match_independent_scalar_rows(self) -> None:
        torch.manual_seed(31)
        values = 3.0 * torch.randn(7, 16)
        for mass in (0, 1, 8, 16):
            expected = torch.stack(
                [project_capped_simplex(row, mass) for row in values]
            )
            actual = project_capped_simplex_batched(values, mass)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=0.0))

        expected_probability = torch.stack(
            [project_probability_simplex(row) for row in values]
        )
        actual_probability = project_probability_simplex_batched(values)
        self.assertTrue(
            torch.allclose(
                actual_probability, expected_probability, atol=1e-7, rtol=0.0
            )
        )

    def test_batched_projectors_validate_shape_and_finiteness(self) -> None:
        with self.assertRaises(ValueError):
            project_capped_simplex_batched(torch.ones(3), 1)
        with self.assertRaises(ValueError):
            project_capped_simplex_batched(torch.empty(0, 3), 1)
        with self.assertRaises(ValueError):
            project_probability_simplex_batched(
                torch.tensor([[0.0, float("nan")]])
            )


class SoftWeightingTests(unittest.TestCase):
    @staticmethod
    def _linear_fractional_objective(
        scores: torch.Tensor,
        tokens: torch.Tensor,
        base_tokens: torch.Tensor,
    ):
        def objective(weights: torch.Tensor) -> torch.Tensor:
            scale = weighted_token_scale(weights, tokens, base_tokens)
            return scale * torch.dot(weights, scores)

        return objective

    def test_projected_adam_recovers_mixed_sign_linear_topk(self) -> None:
        scores = torch.tensor([4.0, 3.0, -2.0, -3.0])
        weights, diagnostics = optimize_soft_weights(
            lambda w: torch.dot(w, scores),
            4,
            2,
            device="cpu",
            steps=20,
            lr=0.1,
            tolerance=1e-7,
            patience=5,
        )
        self.assertTrue(torch.allclose(weights, torch.tensor([1.0, 1.0, 0.0, 0.0]), atol=1e-5))
        self.assertGreater(diagnostics["objective_improvement"], 0.0)
        self.assertAlmostEqual(diagnostics["ess"], 2.0, places=4)
        self.assertAlmostEqual(diagnostics["entropy"], math.log(2.0), places=4)

    def test_batched_linear_solver_matches_scalar_solver_per_layer(self) -> None:
        scores = torch.tensor(
            [
                [3.5, 1.0, -2.0, 0.5, 4.0, -1.0],
                [-1.0, 2.0, 0.25, 3.0, -4.0, 1.5],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [5.0, -2.0, 1.0, -3.0, 0.5, 2.5],
            ],
            dtype=torch.float32,
        )
        tokens = torch.tensor([2.0, 7.0, 3.0, 5.0, 11.0, 4.0])
        base_tokens = tokens.sum()
        reference_topk = torch.topk(scores, k=3, dim=1).indices
        common = dict(
            steps=12,
            lr=0.07,
            tolerance=0.0,
            patience=100,
            constraint="capped_simplex",
        )
        batched_weights, batched_diagnostics = (
            optimize_batched_linear_soft_weights(
                scores,
                tokens,
                base_tokens,
                3,
                reference_topk=reference_topk,
                **common,
            )
        )

        for layer_idx, layer_scores in enumerate(scores):
            scalar_weights, scalar_diagnostics = optimize_soft_weights(
                self._linear_fractional_objective(
                    layer_scores, tokens, base_tokens
                ),
                scores.shape[1],
                3,
                device="cpu",
                reference_topk=reference_topk[layer_idx],
                **common,
            )
            self.assertTrue(
                torch.allclose(
                    batched_weights[layer_idx],
                    scalar_weights,
                    atol=2e-6,
                    rtol=1e-6,
                )
            )
            self.assertTrue(
                torch.equal(
                    torch.topk(batched_weights[layer_idx], 3).indices,
                    torch.topk(scalar_weights, 3).indices,
                )
            )
            batched = batched_diagnostics[layer_idx]
            self.assertEqual(batched["iterations"], scalar_diagnostics["iterations"])
            self.assertEqual(batched["converged"], scalar_diagnostics["converged"])
            self.assertEqual(batched["feasible_set"], scalar_diagnostics["feasible_set"])
            self.assertAlmostEqual(
                batched["initial_objective"],
                scalar_diagnostics["initial_objective"],
                places=5,
            )
            self.assertAlmostEqual(
                batched["final_objective"],
                scalar_diagnostics["final_objective"],
                places=5,
            )
            self.assertAlmostEqual(
                batched["topk_overlap"], scalar_diagnostics["topk_overlap"], places=7
            )
            self.assertAlmostEqual(float(batched_weights[layer_idx].sum()), 3.0, places=6)

    def test_batched_linear_solver_keeps_layerwise_early_stopping(self) -> None:
        scores = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [4.0, 2.0, -1.0, -3.0],
            ]
        )
        tokens = torch.ones(4)
        common = dict(
            steps=6,
            lr=0.01,
            tolerance=0.0,
            patience=2,
        )
        weights, diagnostics = optimize_batched_linear_soft_weights(
            scores, tokens, 4.0, 2, **common
        )
        scalar_results = [
            optimize_soft_weights(
                self._linear_fractional_objective(row, tokens, torch.tensor(4.0)),
                4,
                2,
                device="cpu",
                **common,
            )
            for row in scores
        ]
        for layer_idx, (scalar_weights, scalar_diagnostics) in enumerate(
            scalar_results
        ):
            self.assertTrue(
                torch.allclose(weights[layer_idx], scalar_weights, atol=1e-7)
            )
            self.assertEqual(
                diagnostics[layer_idx]["iterations"],
                scalar_diagnostics["iterations"],
            )
            self.assertEqual(
                diagnostics[layer_idx]["converged"],
                scalar_diagnostics["converged"],
            )
        self.assertEqual(diagnostics[0]["iterations"], 2)
        self.assertLess(
            diagnostics[0]["iterations"], diagnostics[1]["iterations"]
        )

    def test_batched_linear_solver_matches_probability_constraint(self) -> None:
        torch.manual_seed(37)
        scores = torch.randn(5, 8)
        tokens = torch.arange(1, 9, dtype=torch.float32)
        common = dict(
            steps=12,
            lr=0.07,
            tolerance=0.0,
            patience=100,
            constraint="probability_simplex",
        )
        actual, diagnostics = optimize_batched_linear_soft_weights(
            scores, tokens, tokens.sum(), 4, **common
        )
        for layer_idx, layer_scores in enumerate(scores):
            expected, expected_diagnostics = optimize_soft_weights(
                self._linear_fractional_objective(
                    layer_scores, tokens, tokens.sum()
                ),
                8,
                4,
                device="cpu",
                **common,
            )
            self.assertTrue(
                torch.allclose(actual[layer_idx], expected, atol=2e-6, rtol=1e-6)
            )
            self.assertEqual(
                diagnostics[layer_idx]["iterations"],
                expected_diagnostics["iterations"],
            )
            self.assertAlmostEqual(float(actual[layer_idx].sum()), 1.0, places=6)

    def test_streaming_objective_matches_joint_objective(self) -> None:
        first_scores = torch.tensor([4.0, 0.0, -2.0, 0.0])
        second_scores = torch.tensor([0.0, 3.0, 0.0, -3.0])
        common = dict(
            n=4, k=2, device="cpu", steps=20, lr=0.1,
            tolerance=1e-7, patience=5,
        )
        joint_weights, joint_diag = optimize_soft_weights(
            lambda w: torch.dot(w, first_scores + second_scores), **common
        )
        stream_weights, stream_diag = optimize_soft_weights_streaming(
            [lambda w: torch.dot(w, first_scores), lambda w: torch.dot(w, second_scores)],
            **common,
        )
        self.assertTrue(torch.allclose(stream_weights, joint_weights, atol=1e-6))
        self.assertAlmostEqual(
            stream_diag["final_objective"], joint_diag["final_objective"], places=6
        )

    def test_streaming_reuses_projected_point_gradient(self) -> None:
        grad_modes = []
        scores = torch.tensor([3.0, 1.0, -1.0, -2.0])

        def objective(weights: torch.Tensor) -> torch.Tensor:
            grad_modes.append(torch.is_grad_enabled())
            return torch.dot(weights, scores)

        optimize_soft_weights_streaming(
            [objective, objective],
            n=4,
            k=2,
            device="cpu",
            steps=4,
            lr=0.1,
            tolerance=0.0,
            patience=100,
        )
        # One evaluation per iterate (initial + four projected points), not the
        # old initial + 2 * steps pattern.  The last point needs no next gradient.
        self.assertEqual(len(grad_modes), 2 * (4 + 1))
        self.assertEqual(sum(grad_modes), 2 * 4)

    def test_cached_gradient_optimizers_match_legacy_projected_adam(self) -> None:
        scores = torch.tensor([3.0, 1.0, -1.0, -2.0])
        n, k, steps, lr = 4, 2, 7, 0.1

        def explicit(current: torch.Tensor):
            value = torch.dot(current, scores) - 0.1 * current.square().sum()
            gradient = scores - 0.2 * current
            return value, gradient

        # Reference the pre-cache loop: evaluate the current point for Adam,
        # then evaluate the projected candidate again for best-point tracking.
        legacy_weights = torch.full((n,), k / n, dtype=torch.float32)
        first, _ = explicit(legacy_weights)
        legacy_best_value = float(first)
        legacy_best_weights = legacy_weights.clone()
        first_moment = torch.zeros_like(legacy_weights)
        second_moment = torch.zeros_like(legacy_weights)
        for step in range(1, steps + 1):
            _, gradient = explicit(legacy_weights)
            first_moment.mul_(0.9).add_(gradient, alpha=0.1)
            second_moment.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
            proposal = legacy_weights + lr * (
                first_moment / (1.0 - 0.9**step)
            ) / ((second_moment / (1.0 - 0.999**step)).sqrt() + 1e-8)
            legacy_weights = project_capped_simplex(proposal, k)
            candidate, _ = explicit(legacy_weights)
            if float(candidate) > legacy_best_value:
                legacy_best_value = float(candidate)
                legacy_best_weights = legacy_weights.clone()

        common = dict(
            n=n, k=k, device="cpu", steps=steps, lr=lr,
            tolerance=0.0, patience=100,
        )
        regular, regular_diag = optimize_soft_weights(
            lambda w: explicit(w)[0], **common
        )
        streaming, streaming_diag = optimize_soft_weights_streaming(
            [lambda w: 0.25 * explicit(w)[0], lambda w: 0.75 * explicit(w)[0]],
            **common,
        )
        explicit_weights, explicit_diag = optimize_soft_weights_with_grad(
            explicit, **common
        )
        for actual, diagnostics in (
            (regular, regular_diag),
            (streaming, streaming_diag),
            (explicit_weights, explicit_diag),
        ):
            self.assertTrue(torch.equal(actual, legacy_best_weights))
            self.assertEqual(diagnostics["final_objective"], legacy_best_value)

    def test_optimization_reenables_grad_inside_custom_backward_context(self) -> None:
        scores = torch.tensor([3.0, -2.0])
        with torch.no_grad():
            regular, _ = optimize_soft_weights(
                lambda w: torch.dot(w, scores),
                2,
                1,
                device="cpu",
                steps=5,
            )
            streaming, _ = optimize_soft_weights_streaming(
                [lambda w: torch.dot(w, scores)],
                2,
                1,
                device="cpu",
                steps=5,
            )
        self.assertTrue(torch.allclose(regular, torch.tensor([1.0, 0.0]), atol=1e-5))
        self.assertTrue(torch.allclose(streaming, regular, atol=1e-6))

    def test_gamma_uses_update_norm_tuple(self) -> None:
        scores = torch.tensor([2.0, -1.0])

        def objective(weights: torch.Tensor):
            aggregate = torch.dot(weights, scores)
            return aggregate, aggregate.square()

        weights, diagnostics = optimize_soft_weights(
            objective, 2, 1, device="cpu", steps=5, gamma=0.25
        )
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertTrue(math.isfinite(diagnostics["final_objective"]))
        with self.assertRaises(ValueError):
            optimize_soft_weights(
                lambda w: w.sum(), 2, 1, device="cpu", steps=1, gamma=1.0
            )

    def test_initial_nonfinite_fails(self) -> None:
        with self.assertRaises(FloatingPointError):
            optimize_soft_weights(
                lambda w: w.sum() * torch.tensor(float("nan")),
                2,
                1,
                device="cpu",
            )

    def test_weighted_token_scale(self) -> None:
        weights = torch.tensor([1.0, 0.5, 0.0])
        tokens = torch.tensor([4.0, 2.0, 10.0])
        scale = weighted_token_scale(weights, tokens, base_tokens=16)
        self.assertAlmostEqual(float(scale), 16.0 / 5.0, places=6)
        with self.assertRaises(ValueError):
            weighted_token_scale(torch.tensor([0.0, 0.0]), torch.tensor([2.0, 3.0]), 5)


class MuonTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_live_momentum_formula_without_ns(self) -> None:
        gradient = torch.randn(3, 2)
        previous = torch.randn(3, 2)
        mu = 0.8
        nesterov = muon_live_candidate_transform(
            gradient, previous, momentum=mu, nesterov=True,
            ns_steps=0, shape_lr_scale=False,
        )
        plain = muon_live_candidate_transform(
            gradient, previous, momentum=mu, nesterov=False,
            ns_steps=0, shape_lr_scale=False,
        )
        self.assertTrue(torch.allclose(
            nesterov, mu**2 * previous + (1 - mu**2) * gradient
        ))
        self.assertTrue(torch.allclose(
            plain, mu * previous + (1 - mu) * gradient
        ))

    def test_tall_wide_and_zero_are_finite(self) -> None:
        for shape in ((7, 3), (3, 7), (4, 4)):
            gradient = torch.zeros(shape) if shape == (4, 4) else torch.randn(shape)
            result = muon_live_candidate_transform(
                gradient, momentum=0.95, ns_steps=5, shape_lr_scale=True
            )
            self.assertEqual(result.shape, gradient.shape)
            self.assertTrue(bool(torch.isfinite(result).all()))
            if not bool(gradient.any()):
                self.assertTrue(torch.equal(result, torch.zeros_like(result)))

    def test_ns_has_finite_gradient_and_matches_finite_difference(self) -> None:
        matrix = torch.randn(3, 2, requires_grad=True)
        probe = torch.randn(3, 2)

        def scalar(candidate: torch.Tensor) -> torch.Tensor:
            transformed = muon_live_candidate_transform(
                candidate, momentum=0.7, nesterov=True,
                ns_steps=3, shape_lr_scale=False,
            )
            return (transformed * probe).sum()

        value = scalar(matrix)
        analytic = torch.autograd.grad(value, matrix)[0]
        self.assertTrue(bool(torch.isfinite(analytic).all()))
        row, column = 1, 0
        step = 1e-3
        with torch.no_grad():
            plus = matrix.detach().clone()
            minus = matrix.detach().clone()
            plus[row, column] += step
            minus[row, column] -= step
            finite_difference = (scalar(plus) - scalar(minus)) / (2 * step)
        self.assertAlmostEqual(
            float(analytic[row, column]), float(finite_difference), delta=3e-3
        )

    def test_bfloat16_optimizer_boundary_matches_live_q(self) -> None:
        gradient = torch.randn(5, 3).to(torch.bfloat16)
        previous = torch.randn(5, 3).to(torch.bfloat16)
        momentum = 0.8
        actual = muon_live_candidate_transform(
            gradient.float(),
            previous,
            optimizer_dtype=torch.bfloat16,
            momentum=momentum,
            nesterov=True,
            ns_steps=3,
            shape_lr_scale=False,
        )
        # Mirror HybridMuonAdamW._step_muon: both the momentum buffer and q are
        # rounded in parameter dtype before NS promotes internally to float32.
        buffer = previous.clone()
        buffer.mul_(momentum).add_(gradient, alpha=1.0 - momentum)
        q = gradient.mul(1.0 - momentum).add(buffer, alpha=momentum)
        expected = zeropower_via_newton_schulz(q, steps=3, eps=1e-7)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(actual, expected))


class SpectralSurrogateTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)

    def test_projected_ghost_matches_explicit_gradient_modes(self) -> None:
        batch, tokens, output_dim, input_dim = 5, 3, 4, 3
        grad_output = torch.randn(batch, tokens, output_dim)
        inputs = torch.randn(batch, tokens, input_dim)
        target = torch.randn(output_dim, input_dim)
        u, v, active_rank = compute_spectral_modes(
            target, full_svd_max_dim=256, rtol=1e-6
        )
        self.assertGreater(active_rank, 0)
        ghost = spectral_linear_mode_support(grad_output, inputs, u, v)
        explicit_gradients = torch.einsum("bso,bsi->boi", grad_output, inputs)
        explicit = torch.einsum("or,boi,ir->br", u, explicit_gradients, v).clamp_min(0)
        self.assertTrue(torch.allclose(ghost, explicit, atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.allclose(
            spectral_linear_scores(grad_output, inputs, u, v),
            explicit.sum(dim=1),
            atol=2e-5,
            rtol=2e-5,
        ))

    def test_singular_values_are_returned_separately_from_unit_modes(self) -> None:
        target = torch.diag(torch.tensor([5.0, 2.0, 1e-8]))
        u, v, beta, active_rank = compute_spectral_modes_with_values(
            target, full_svd_max_dim=256, rtol=1e-6
        )
        self.assertEqual(active_rank, 2)
        self.assertTrue(torch.allclose(beta, torch.tensor([5.0, 2.0])))
        reconstructed = u @ torch.diag(beta) @ v.T
        self.assertTrue(torch.allclose(
            reconstructed, torch.diag(torch.tensor([5.0, 2.0, 0.0])),
            atol=1e-6,
        ))

        legacy_u, legacy_v, legacy_rank = compute_spectral_modes(
            target, full_svd_max_dim=256, rtol=1e-6
        )
        self.assertEqual(legacy_rank, active_rank)
        self.assertTrue(torch.allclose(legacy_u @ legacy_u.T, u @ u.T))
        self.assertTrue(torch.allclose(legacy_v @ legacy_v.T, v @ v.T))

    def test_beta_weighting_matches_positive_target_mode_taylor_sum(self) -> None:
        support = torch.tensor([
            [2.0, 1.0],
            [0.5, 4.0],
            [3.0, 0.0],
        ])
        beta = torch.tensor([5.0, 2.0])
        self.assertTrue(torch.equal(support @ beta, torch.tensor([12.0, 10.5, 15.0])))

    def test_log_saturation_is_monotone_submodular_and_promotes_modes(self) -> None:
        support = torch.tensor([
            [10.0, 0.0],
            [9.0, 0.0],
            [0.0, 4.0],
        ])
        selected = seeded_spectral_saturation_greedy(
            support, 2, alpha=1.0, seed=42
        )
        # Plain modular top-k would choose {0, 1}; diminishing returns make the
        # saturated objective cover the second target mode instead.
        self.assertEqual(set(selected.tolist()), {0, 2})

        a = torch.tensor([0])
        b = torch.tensor([0, 1])
        a_plus = torch.tensor([0, 2])
        b_plus = torch.tensor([0, 1, 2])
        marginal_a = (
            spectral_saturated_objective(support, a_plus)
            - spectral_saturated_objective(support, a)
        )
        marginal_b = (
            spectral_saturated_objective(support, b_plus)
            - spectral_saturated_objective(support, b)
        )
        self.assertGreaterEqual(float(marginal_a), float(marginal_b))
        self.assertGreaterEqual(
            float(spectral_saturated_objective(support, b)),
            float(spectral_saturated_objective(support, a)),
        )

    def test_beta_weighted_saturation_and_seeded_ties(self) -> None:
        support = torch.tensor([
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        beta = torch.tensor([1.0, 3.0])
        subset = torch.tensor([0, 1])
        expected = torch.log1p(torch.tensor(2.0)) + 3.0 * torch.log1p(torch.tensor(1.0))
        self.assertTrue(torch.allclose(
            spectral_saturated_objective(support, subset, alpha=beta), expected
        ))

        tied = torch.zeros(12, 3)
        first = seeded_spectral_saturation_greedy(tied, 5, seed=123)
        second = seeded_spectral_saturation_greedy(tied, 5, seed=123)
        third = seeded_spectral_saturation_greedy(tied, 5, seed=124)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, third))

    def test_saturation_rejects_negative_support_or_mode_weights(self) -> None:
        with self.assertRaises(ValueError):
            seeded_spectral_saturation_greedy(
                torch.tensor([[1.0, -0.1]]), 1, seed=1
            )
        with self.assertRaises(ValueError):
            seeded_spectral_saturation_greedy(
                torch.ones(2, 2), 1, alpha=torch.tensor([1.0, -1.0]), seed=1
            )

    def test_scores_are_modular_and_topk_is_exhaustive_optimum(self) -> None:
        scores = torch.tensor([0.3, 2.0, 1.2, 0.1, 0.8])
        k = 2
        selected = seeded_topk(scores, k, seed=42)
        selected_value = float(scores[selected].sum())
        exhaustive = max(
            float(scores[list(indices)].sum())
            for indices in itertools.combinations(range(scores.numel()), k)
        )
        self.assertAlmostEqual(selected_value, exhaustive, places=6)

        a, b = {0}, {0, 2}
        item = 1
        marginal_a = scores[list(a | {item})].sum() - scores[list(a)].sum()
        marginal_b = scores[list(b | {item})].sum() - scores[list(b)].sum()
        self.assertAlmostEqual(float(marginal_a), float(marginal_b), places=6)

    def test_seeded_tie_break_is_reproducible(self) -> None:
        tied = torch.zeros(12)
        first = seeded_topk(tied, 5, seed=123)
        second = seeded_topk(tied, 5, seed=123)
        third = seeded_topk(tied, 5, seed=124)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, third))

    def test_randomized_modes_are_deterministic_and_zero_target_skips(self) -> None:
        target = torch.randn(40, 36)
        first = compute_spectral_modes(
            target, rank=8, full_svd_max_dim=16,
            oversample=3, power_iters=2, seed=99,
        )
        second = compute_spectral_modes(
            target, rank=8, full_svd_max_dim=16,
            oversample=3, power_iters=2, seed=99,
        )
        self.assertEqual(first[2], second[2])
        # Singular vectors have a sign ambiguity, so compare their projectors.
        self.assertTrue(torch.allclose(first[0] @ first[0].T, second[0] @ second[0].T))
        zero_u, zero_v, zero_rank = compute_spectral_modes(torch.zeros(4, 3))
        self.assertEqual(zero_rank, 0)
        self.assertEqual(zero_u.shape, (4, 0))
        self.assertEqual(zero_v.shape, (3, 0))
        with self.assertRaises(FloatingPointError):
            compute_spectral_modes(torch.tensor([[float("inf")]]))


class WeightedAssemblyTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)

    def test_linear_weight_and_bias_match_weighted_loss_gradient(self) -> None:
        batch, tokens, input_dim, output_dim = 4, 3, 5, 2
        inputs = torch.randn(batch, tokens, input_dim)
        grad_output = torch.randn(batch, tokens, output_dim)
        sample_weights = torch.tensor([1.0, 0.25, 0.5, 0.0])
        scale = torch.tensor(1.7)
        weight = torch.randn(output_dim, input_dim, requires_grad=True)
        bias = torch.randn(output_dim, requires_grad=True)
        output = F.linear(inputs, weight, bias)
        weighted_loss = (
            output
            * grad_output
            * sample_weights[:, None, None]
        ).sum() * scale
        expected_weight, expected_bias = torch.autograd.grad(weighted_loss, (weight, bias))
        actual_weight, actual_bias = weighted_linear_gradients(
            grad_output, inputs, sample_weights, scale=scale, has_bias=True
        )
        self.assertTrue(torch.allclose(actual_weight, expected_weight, atol=1e-6))
        self.assertTrue(torch.allclose(actual_bias, expected_bias, atol=1e-6))

    def test_embedding_matches_weighted_loss_gradient_and_zeros_padding(self) -> None:
        num_embeddings, embedding_dim = 9, 4
        padding_idx = 0
        ids = torch.tensor([[0, 1, 2], [3, 1, 4], [5, 0, 6]])
        grad_output = torch.randn(3, 3, embedding_dim)
        sample_weights = torch.tensor([1.0, 0.5, 0.25])
        scale = torch.tensor(2.0)
        weight = torch.randn(num_embeddings, embedding_dim, requires_grad=True)
        output = F.embedding(ids, weight, padding_idx=padding_idx)
        weighted_loss = (
            output * grad_output * sample_weights[:, None, None]
        ).sum() * scale
        expected = torch.autograd.grad(weighted_loss, weight)[0]
        actual = weighted_embedding_gradient(
            grad_output,
            ids,
            sample_weights,
            num_embeddings=num_embeddings,
            padding_idx=padding_idx,
            scale=scale,
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertTrue(torch.equal(actual[padding_idx], torch.zeros(embedding_dim)))

    def test_bfloat16_factors_keep_continuous_weights_until_final_cast(self) -> None:
        grad_output = torch.tensor(
            [[[1.0, 2.0]], [[-3.0, 4.0]]], dtype=torch.bfloat16
        )
        inputs = torch.tensor(
            [[[2.0, -1.0]], [[1.0, 3.0]]], dtype=torch.bfloat16
        )
        weights = torch.tensor([0.5005, 0.4995], dtype=torch.float32)
        scale = torch.tensor(1.0007, dtype=torch.float32)
        actual_weight, actual_bias = weighted_linear_gradients(
            grad_output, inputs, weights, scale=scale, has_bias=True
        )
        expected_weight = torch.einsum(
            "bso,bsi->oi",
            grad_output.float() * weights[:, None, None],
            inputs.float(),
        ) * scale
        expected_bias = (
            grad_output.float() * weights[:, None, None]
        ).sum(dim=(0, 1)) * scale
        self.assertEqual(actual_weight.dtype, torch.float32)
        self.assertEqual(actual_bias.dtype, torch.float32)
        self.assertTrue(torch.allclose(actual_weight, expected_weight, atol=1e-7))
        self.assertTrue(torch.allclose(actual_bias, expected_bias, atol=1e-7))

    def test_bf16_fp32_request_falls_back_exactly_on_cpu(self) -> None:
        grad_output = torch.randn(3, 5, 7).to(torch.bfloat16)
        inputs = torch.randn(3, 5, 11).to(torch.bfloat16)
        weights = torch.tensor([0.1255, 0.5005, 0.3740], dtype=torch.float32)
        scale = torch.tensor(1.003, dtype=torch.float32)
        expected_weight, expected_bias = weighted_linear_gradients(
            grad_output,
            inputs,
            weights,
            scale=scale,
            has_bias=True,
            replay_precision="fp32",
        )
        actual_weight, actual_bias = weighted_linear_gradients(
            grad_output,
            inputs,
            weights,
            scale=scale,
            has_bias=True,
            replay_precision="bf16_fp32",
        )
        self.assertEqual(actual_weight.dtype, torch.float32)
        self.assertEqual(actual_bias.dtype, torch.float32)
        self.assertTrue(torch.equal(actual_weight, expected_weight))
        self.assertTrue(torch.equal(actual_bias, expected_bias))

    def test_linear_replay_rejects_unknown_precision(self) -> None:
        with self.assertRaisesRegex(ValueError, "replay_precision"):
            weighted_linear_gradients(
                torch.ones(2, 3),
                torch.ones(2, 5),
                torch.ones(2),
                replay_precision="fp16",
            )

    def test_global_soft_replay_forwards_configured_precision(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(5, 4))
        hook = GradientHook(model, ["0"], device="cpu")
        try:
            hook._retained_data[0] = (
                torch.randn(2, 3, 4),
                torch.randn(2, 3, 5),
            )
            state = SimpleNamespace(
                tokens_per_sample=torch.tensor([2.0, 3.0]),
                batch_total_tokens_tensor=torch.tensor(5.0),
                solver_config={"soft_replay_precision": "bf16_fp32"},
            )
            with patch(
                "drpt.selection.advanced_solvers.weighted_linear_gradients",
                return_value=(torch.zeros(4, 5), torch.zeros(4)),
            ) as replay:
                hook.assemble_weighted_gradients_from_retained(
                    torch.tensor([0.4, 0.6]),
                    state,
                )
            self.assertEqual(
                replay.call_args.kwargs["replay_precision"], "bf16_fp32"
            )
        finally:
            hook.clear_retained_data()
            hook.remove_hooks()

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        and "out_dtype" in (torch.mm.__doc__ or ""),
        "CUDA bf16 mm with fp32 output is unavailable",
    )
    def test_cuda_bf16_fp32_rank3_replay_matches_fp32_reference(self) -> None:
        grad_output = torch.randn(4, 32, 37, device="cuda").to(torch.bfloat16)
        inputs = torch.randn(4, 32, 53, device="cuda").to(torch.bfloat16)
        weights = torch.tensor(
            [0.5005, 0.2495, 0.1255, 0.1245],
            device="cuda",
            dtype=torch.float32,
        )
        scale = torch.tensor(1.007, device="cuda", dtype=torch.float32)
        expected_weight, expected_bias = weighted_linear_gradients(
            grad_output,
            inputs,
            weights,
            scale=scale,
            has_bias=True,
            replay_precision="fp32",
        )
        actual_weight, actual_bias = weighted_linear_gradients(
            grad_output,
            inputs,
            weights,
            scale=scale,
            has_bias=True,
            replay_precision="bf16_fp32",
        )
        relative_l2 = (
            (actual_weight - expected_weight).norm()
            / expected_weight.norm().clamp_min(1e-12)
        )
        cosine = F.cosine_similarity(
            actual_weight.flatten(), expected_weight.flatten(), dim=0
        )
        self.assertEqual(actual_weight.dtype, torch.float32)
        self.assertEqual(actual_bias.dtype, torch.float32)
        self.assertLess(float(relative_l2), 3e-3)
        self.assertGreater(float(cosine), 0.99999)
        self.assertTrue(torch.equal(actual_bias, expected_bias))

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        and "out_dtype" in (torch.mm.__doc__ or ""),
        "CUDA bf16 mm with fp32 output is unavailable",
    )
    def test_cuda_bf16_fp32_rank2_replay_without_bias(self) -> None:
        grad_output = torch.randn(64, 29, device="cuda").to(torch.bfloat16)
        inputs = torch.randn(64, 41, device="cuda").to(torch.bfloat16)
        weights = torch.linspace(
            0.01, 1.0, 64, device="cuda", dtype=torch.float32
        )
        expected_weight, _ = weighted_linear_gradients(
            grad_output,
            inputs,
            weights,
            has_bias=False,
            replay_precision="fp32",
        )
        actual_weight, actual_bias = weighted_linear_gradients(
            grad_output,
            inputs,
            weights,
            has_bias=False,
            replay_precision="bf16_fp32",
        )
        relative_l2 = (
            (actual_weight - expected_weight).norm()
            / expected_weight.norm().clamp_min(1e-12)
        )
        cosine = F.cosine_similarity(
            actual_weight.flatten(), expected_weight.flatten(), dim=0
        )
        self.assertEqual(actual_weight.dtype, torch.float32)
        self.assertIsNone(actual_bias)
        self.assertLess(float(relative_l2), 3e-3)
        self.assertGreater(float(cosine), 0.99999)

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is unavailable",
    )
    def test_cuda_without_native_bf16_uses_exact_fp32_fallback(self) -> None:
        grad_output = torch.randn(2, 3, 7, device="cuda").to(torch.bfloat16)
        inputs = torch.randn(2, 3, 11, device="cuda").to(torch.bfloat16)
        weights = torch.tensor([0.5005, 0.4995], device="cuda")
        expected_weight, expected_bias = weighted_linear_gradients(
            grad_output,
            inputs,
            weights,
            replay_precision="fp32",
        )
        with patch(
            "drpt.selection.advanced_solvers.torch.cuda.get_device_capability",
            return_value=(7, 5),
        ), patch.dict(
            "drpt.selection.advanced_solvers._CUDA_NATIVE_BF16_BY_DEVICE",
            {},
            clear=True,
        ):
            actual_weight, actual_bias = weighted_linear_gradients(
                grad_output,
                inputs,
                weights,
                replay_precision="bf16_fp32",
            )
        self.assertTrue(torch.equal(actual_weight, expected_weight))
        self.assertTrue(torch.equal(actual_bias, expected_bias))


class ValidationCachePrecisionTests(unittest.TestCase):
    def test_hook_enables_float32_full_capture(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        hook = GradientHook(model, ["0"], device="cpu")
        try:
            hook.start_val_capture(
                scoring_method="reduced_ghost", full_precision=True
            )
            self.assertEqual(hook.val_cache.accumulation_dtype, torch.float32)
        finally:
            hook.clear_val_buffer()
            hook.remove_hooks()

    def test_full_target_is_contracted_in_requested_float32_dtype(self) -> None:
        torch.manual_seed(31)
        grad_output = torch.randn(3, 4, 5).to(torch.bfloat16)
        inputs = torch.randn(3, 4, 6).to(torch.bfloat16)
        cache = ValidationCache(num_layers=1)
        cache.start_capture(mode="full", accumulation_dtype=torch.float32)
        cache.store_layer(0, grad_output, inputs)
        cache.end_capture(total_tokens=7)
        actual = cache.get_full(0)
        expected = torch.einsum(
            "bso,bsi->oi", grad_output.float(), inputs.float()
        )
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertEqual(cache.total_tokens, 7)

    def test_embedding_target_matches_padding_aware_autograd(self) -> None:
        num_embeddings, dim, padding_idx = 7, 3, 0
        input_ids = torch.tensor([[0, 1, 2], [3, 0, 1]])
        grad_output = torch.arange(18, dtype=torch.float32).reshape(2, 3, dim)
        weight = torch.randn(num_embeddings, dim, requires_grad=True)
        output = F.embedding(input_ids, weight, padding_idx=padding_idx)
        expected = torch.autograd.grad((output * grad_output).sum(), weight)[0]
        actual = compute_embedding_val_gradient(
            grad_output,
            input_ids,
            num_embeddings,
            dim,
            padding_idx=padding_idx,
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual[padding_idx], torch.zeros(dim)))


if __name__ == "__main__":
    unittest.main()
