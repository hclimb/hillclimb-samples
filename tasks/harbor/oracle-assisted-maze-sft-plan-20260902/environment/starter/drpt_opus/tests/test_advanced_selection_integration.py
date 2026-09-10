from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from drpt.selection.advanced_solvers import (
    muon_live_candidate_transform,
    optimize_soft_weights,
    seeded_topk,
    weighted_linear_gradients,
    weighted_token_scale,
)
from drpt.selection.backward import (
    _adamw_candidate_transform,
    _linear_target_gradients,
    _make_soft_linear_objective,
    _muon_candidate_transform,
    _optimize_layer_soft_weights,
    _target_standalone_scale,
)
from drpt.selection.state import GlobalSubsetState, LayerWiseSubsetState
from drpt.selection.strategies import SeparateBatchLayerWiseSoftWeightingStrategy
from drpt.optimizer import HybridMuonAdamW


def _state_kwargs(**overrides):
    values = {
        "train_batch_size": 12,
        "num_layers": 3,
        "frac": 0.25,
        "lr": 1.0,
        "device": "cpu",
        "dtype": torch.float32,
        "use_second_order": False,
        "selection_mode": "topk",
        "record_selections": True,
        "seed": 42,
        "global_step": 7,
    }
    values.update(overrides)
    return values


class RawTargetAndLinearMapperTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(101)

    def test_adamw_candidate_map_applies_group_lr_with_or_without_state(self) -> None:
        parameter = nn.Parameter(torch.zeros(2, 3))
        candidate = torch.randn_like(parameter)
        group_lr = 2.5e-4
        optimizer = SimpleNamespace(state={})
        group = {"lr": group_lr, "betas": (0.9, 0.999), "eps": 1e-8}
        hook = SimpleNamespace(
            optimizer=optimizer,
            optimizer_aware_config={"adam_eps": 1e-8},
            _optimizer_group_by_param_id={id(parameter): group},
        )

        warmup = _adamw_candidate_transform(hook, parameter, candidate)
        self.assertTrue(torch.equal(warmup, candidate * group_lr))

        optimizer.state[parameter] = {
            "step": 1,
            "exp_avg_sq": torch.full_like(parameter, 0.25),
        }
        initialized = _adamw_candidate_transform(hook, parameter, candidate)
        bias_correction2 = 1.0 - group["betas"][1]
        inv_rms = 1.0 / (0.5 / bias_correction2**0.5 + group["eps"])
        self.assertTrue(torch.allclose(
            initialized,
            candidate * (group_lr * inv_rms),
            atol=0.0,
            rtol=1e-6,
        ))

    def test_adamw_candidate_map_uses_raw_target_and_matches_dual_form(self) -> None:
        batch, tokens, output_dim, input_dim = 4, 2, 3, 2
        grad_output = torch.randn(batch, tokens, output_dim)
        inputs = torch.randn(batch, tokens, input_dim)
        target = torch.randn(output_dim, input_dim)
        preconditioner = torch.tensor(
            [[0.25, 2.0], [1.5, 0.5], [3.0, 0.75]], dtype=torch.float32
        )
        weights = torch.tensor([1.0, 0.75, 0.25, 0.0])

        module = nn.Linear(input_dim, output_dim, bias=False)
        hook = SimpleNamespace(
            optimizer=None,
            optimizer_aware_config={"optimizer_type": "adamw"},
            _get_module_from_idx=lambda _: module,
        )
        state = SimpleNamespace(
            tokens_per_sample=torch.ones(batch),
            batch_total_tokens_tensor=torch.tensor(float(batch)),
            solver_config={"soft_weighting_gamma": 0.0},
        )

        def fixed_candidate_map(_hook, _parameter, candidate):
            return candidate * preconditioner

        with patch(
            "drpt.selection.backward._adamw_candidate_transform",
            side_effect=fixed_candidate_map,
        ):
            objective = _make_soft_linear_objective(
                hook,
                state,
                0,
                grad_output,
                inputs,
                target,
                None,
                False,
            )
            actual = objective(weights)

        scale = weighted_token_scale(weights, torch.ones(batch), batch)
        aggregate, _ = weighted_linear_gradients(
            grad_output, inputs, weights, scale=scale, has_bias=False
        )
        primal = torch.sum(target * (preconditioner * aggregate))
        dual = torch.sum((preconditioner * target) * aggregate)
        incorrectly_mapped_target = torch.sum(
            (preconditioner * target) * (preconditioner * aggregate)
        )

        self.assertTrue(torch.allclose(actual, primal, atol=1e-6))
        self.assertTrue(torch.allclose(actual, dual, atol=1e-6))
        self.assertFalse(torch.allclose(actual, incorrectly_mapped_target, atol=1e-4))

    def test_identity_and_frozen_adamw_soft_optima_match_topk(self) -> None:
        raw_scores = torch.tensor([5.0, 2.0, -1.0, -4.0])
        adam_diagonal = torch.tensor([1.5, 0.5, 2.0, 0.25])
        token_counts = torch.ones(4)
        expected_identity = seeded_topk(raw_scores, 2, seed=42).sort().values
        expected_adam = seeded_topk(
            raw_scores * adam_diagonal, 2, seed=42
        ).sort().values

        for scores, expected in (
            (raw_scores, expected_identity),
            (raw_scores * adam_diagonal, expected_adam),
        ):
            def objective(weights):
                scale = weighted_token_scale(weights, token_counts, base_tokens=4)
                return scale * torch.dot(weights, scores)

            weights, _ = optimize_soft_weights(
                objective,
                n=4,
                k=2,
                device="cpu",
                steps=20,
                lr=0.1,
                tolerance=1e-7,
                patience=5,
            )
            actual = seeded_topk(weights, 2, seed=42).sort().values
            self.assertTrue(torch.equal(actual, expected))
            selected_floor = weights[expected].min()
            mask = torch.ones(4, dtype=torch.bool)
            mask[expected] = False
            self.assertGreaterEqual(float(selected_floor), float(weights[mask].max()))

    def test_merged_target_is_restored_without_optimizer_mapping(self) -> None:
        state = SimpleNamespace(
            _use_stored_val=False,
            device="cpu",
            batch_total_tokens_tensor=torch.tensor(15.0),
            train_total_tokens_tensor=torch.tensor(9.0),
        )
        self.assertAlmostEqual(float(_target_standalone_scale(state)), 15.0 / 6.0)
        state._use_stored_val = True
        self.assertEqual(float(_target_standalone_scale(state)), 1.0)

    def test_merged_target_contraction_promotes_bfloat16_factors_first(self) -> None:
        grad_output = torch.tensor(
            [[[1.0078, -0.5039]], [[0.3320, 2.0156]]],
            dtype=torch.bfloat16,
        )
        inputs = torch.tensor(
            [[[0.6641, -1.0078]], [[3.0156, 0.2490]]],
            dtype=torch.bfloat16,
        )
        state = SimpleNamespace(
            _use_stored_val=False,
            device="cpu",
            batch_total_tokens_tensor=torch.tensor(9.0),
            train_total_tokens_tensor=torch.tensor(5.0),
        )
        actual, actual_bias = _linear_target_gradients(
            torch.bfloat16,
            state,
            grad_output,
            inputs,
            None,
            None,
            True,
        )
        scale = 9.0 / 4.0
        expected = torch.einsum(
            "bso,bsi->oi", grad_output.float(), inputs.float()
        ) * scale
        expected_bias = grad_output.float().sum(dim=(0, 1)) * scale
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))
        self.assertTrue(torch.allclose(actual_bias, expected_bias, atol=1e-7))

    def test_live_muon_transform_matches_real_optimizer_group_and_state(self) -> None:
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(3, 4, bias=True)

        class TinyTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([Block()])

        model = TinyTransformer().to(torch.bfloat16)
        optimizer = HybridMuonAdamW(
            model.named_parameters(),
            model,
            lr=2e-3,
            muon_momentum=0.8,
            muon_nesterov=True,
            muon_ns_steps=3,
            muon_eps=2e-7,
            muon_lr_shape_scale=True,
            muon_backend="local",
            _force_local_for_test=True,
        )
        weight = model.model.layers[0].q_proj.weight
        self.assertEqual(optimizer.get_param_optimizer_kind(weight), "muon")
        group = next(
            candidate_group
            for candidate_group in optimizer.param_groups
            if any(parameter is weight for parameter in candidate_group["params"])
        )
        previous = torch.randn_like(weight)
        optimizer.state[weight]["momentum_buffer"] = previous.clone()
        candidate = torch.randn_like(weight)
        hook = SimpleNamespace(
            optimizer=optimizer,
            _optimizer_group_by_param_id={id(weight): group},
            optimizer_aware_config={},
        )
        state = SimpleNamespace(
            solver_config={"soft_weighting_use_optimizer_state": True}
        )
        actual = _muon_candidate_transform(hook, state, weight, candidate)
        expected = muon_live_candidate_transform(
            candidate,
            momentum_buffer=previous,
            optimizer_dtype=weight.dtype,
            momentum=group["momentum"],
            nesterov=group["nesterov"],
            ns_steps=group["muon_ns_steps"],
            eps=group["muon_eps"],
            lr=group["lr"],
            shape_lr_scale=group["muon_lr_shape_scale"],
            adjust_lr_fn=group["adjust_lr_fn"],
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-6))

    def test_separate_soft_rejects_zero_valid_target_tokens(self) -> None:
        cache = SimpleNamespace(get_num_captured=lambda: 1)
        hook = SimpleNamespace(val_cache=cache, val_total_tokens=0)
        strategy = SeparateBatchLayerWiseSoftWeightingStrategy(
            grad_hook=hook,
            frac=0.5,
            use_second_order=False,
            selection_mode="topk",
            record_selections=False,
            scoring_method="reduced_ghost",
        )
        with self.assertRaisesRegex(RuntimeError, "valid target token"):
            strategy._setup_state(batch_size=2, lr=1e-4)


class ScopeAndRandomnessTests(unittest.TestCase):
    def test_layerwise_random_is_reproducible_and_layer_specific(self) -> None:
        first = LayerWiseSubsetState(
            **_state_kwargs(selection_variant="random", scoring_method="reduced_ghost")
        )
        second = LayerWiseSubsetState(
            **_state_kwargs(selection_variant="random", scoring_method="reduced_ghost")
        )
        scores = torch.zeros(first.train_batch_size)

        first_l0 = first._select_indices(scores, layer_idx=0).sort().values
        first_l1 = first._select_indices(scores, layer_idx=1).sort().values
        second_l0 = second._select_indices(scores, layer_idx=0).sort().values

        self.assertTrue(torch.equal(first_l0, second_l0))
        self.assertFalse(torch.equal(first_l0, first_l1))

        next_step = LayerWiseSubsetState(
            **_state_kwargs(
                selection_variant="random",
                scoring_method="reduced_ghost",
                global_step=8,
            )
        )
        self.assertFalse(torch.equal(
            first_l0,
            next_step._select_indices(scores, layer_idx=0).sort().values,
        ))

    def test_global_random_uses_one_shared_subset(self) -> None:
        first = GlobalSubsetState(
            **_state_kwargs(selection_variant="random", scoring_method="reduced_ghost")
        )
        second = GlobalSubsetState(
            **_state_kwargs(selection_variant="random", scoring_method="reduced_ghost")
        )
        selected = first.get_final_selection().sort().values
        repeated = second.get_final_selection().sort().values
        self.assertTrue(torch.equal(selected, repeated))
        self.assertEqual(selected.numel(), first.num_selected)

    def test_global_soft_has_one_weight_vector_layerwise_has_independent_vectors(self) -> None:
        solver_config = {
            "soft_weighting_steps": 20,
            "soft_weighting_lr": 0.1,
            "soft_weighting_tol": 1e-7,
            "soft_weighting_patience": 5,
            "soft_weighting_gamma": 0.0,
        }
        global_state = GlobalSubsetState(
            **_state_kwargs(
                train_batch_size=4,
                frac=0.5,
                selection_variant="soft",
                scoring_method="reduced_ghost",
                solver_config=solver_config,
            )
        )
        score_a = torch.tensor([4.0, 1.0, -1.0, -3.0])
        score_b = torch.tensor([1.0, 3.0, -2.0, -4.0])
        global_state.add_soft_objective(lambda w: torch.dot(w, score_a))
        global_state.add_soft_objective(lambda w: torch.dot(w, score_b))
        shared = global_state.optimize_global_soft_weights()
        self.assertIs(global_state._soft_weights, shared)
        self.assertEqual(shared.shape, (4,))

        layer_state = LayerWiseSubsetState(
            **_state_kwargs(
                train_batch_size=4,
                frac=0.5,
                selection_variant="soft",
                scoring_method="reduced_ghost",
                solver_config=solver_config,
            )
        )
        first = _optimize_layer_soft_weights(
            layer_state, 0, lambda w: torch.dot(w, score_a), score_a
        )
        second = _optimize_layer_soft_weights(
            layer_state, 1, lambda w: torch.dot(w, -score_a), -score_a
        )
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(len(layer_state._selection_records), 2)
        self.assertIsNone(layer_state._soft_weights)


if __name__ == "__main__":
    unittest.main()
