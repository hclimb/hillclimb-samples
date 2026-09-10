from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn as nn
import drpt.selection.advanced_solvers as advanced_solvers

from drpt.selection.backward import (
    GlobalSubsetLinearBackward,
    LayerWiseSubsetLinearBackward,
    _compute_muon_spectral_linear_support,
)
from drpt.selection.state import GlobalSubsetState, LayerWiseSubsetState
from drpt.selection.strategies import (
    _advanced_solver_config,
    _validate_advanced_strategy,
    create_merged_batch_strategy,
    create_separate_batch_strategy,
)


def _state_kwargs(
    state_type,
    *,
    include_adamw_scores: bool,
    mode_weighting: str = "uniform",
    saturation: bool = False,
):
    kwargs = {
        "train_batch_size": 4,
        "num_layers": 1,
        "frac": 0.5,
        "lr": 1.0,
        "device": "cpu",
        "dtype": torch.float32,
        "use_second_order": False,
        "selection_mode": "topk",
        "record_selections": False,
        "selection_variant": "muon_spectral",
        "solver_config": {
            "muon_surrogate_include_adamw_scores": include_adamw_scores,
            "muon_surrogate_mode_weighting": mode_weighting,
            "muon_surrogate_saturation": saturation,
        },
        "seed": 42,
        "global_step": 0,
        "optimizer_aware": True,
        "optimizer_aware_config": {"optimizer_type": "hybrid"},
        "scoring_method": "reduced_ghost",
    }
    if state_type is GlobalSubsetState:
        kwargs["one_pass"] = True
    state = state_type(**kwargs)
    tokens = torch.ones(4)
    state.set_token_counts(tokens, tokens.sum(), torch.tensor(5.0))
    return state


def _hook(module: nn.Module, optimizer_kind: str):
    optimizer = SimpleNamespace(
        get_param_optimizer_kind=lambda _parameter: optimizer_kind,
    )
    return SimpleNamespace(
        optimizer=optimizer,
        optimizer_aware_config={"optimizer_type": "hybrid"},
        _get_module_from_idx=lambda _layer_idx: module,
        get_optimizer_group_keys=lambda: [f"{optimizer_kind}:matrix"],
    )


class MuonMatrixOnlyCriterionTests(unittest.TestCase):
    def test_window_fetches_and_discards_raw_target_only_once(self) -> None:
        torch.manual_seed(2718)
        module = nn.Linear(2, 3, bias=False)
        state = _state_kwargs(
            LayerWiseSubsetState, include_adamw_scores=False
        )
        state._use_stored_val = True
        state.enable_windowed_execution()
        target = torch.randn(3, 2)
        current_target = [target]

        cache = SimpleNamespace()
        cache.get_factorized = Mock(return_value=(None, None))
        cache.get_full = Mock(side_effect=lambda *_args, **_kwargs: current_target[0])
        cache.get_bias_grad = Mock(return_value=None)

        def discard(_layer_idx: int) -> None:
            current_target[0] = None

        cache.discard_weight_gradient = Mock(side_effect=discard)
        hook = _hook(module, "muon")
        hook._val_cache = cache
        grad_output = torch.randn(4, 3)
        inputs = torch.randn(4, 2)

        with patch.object(
            advanced_solvers,
            "compute_spectral_modes_with_values",
            wraps=advanced_solvers.compute_spectral_modes_with_values,
        ) as svd:
            for start in (0, 2):
                end = start + 2
                state.set_window_chunk("score", start, end)
                actual = LayerWiseSubsetLinearBackward._backward_full(
                    hook,
                    None,
                    state,
                    0,
                    inputs[start:end],
                    None,
                    grad_output[start:end],
                    False,
                    True,
                )
                self.assertEqual(actual, (None, None))

        self.assertEqual(cache.get_factorized.call_count, 1)
        self.assertEqual(cache.get_full.call_count, 1)
        self.assertEqual(cache.get_bias_grad.call_count, 1)
        cache.discard_weight_gradient.assert_called_once_with(0)
        self.assertEqual(svd.call_count, 1)
        state.require_complete_window(0)
        self.assertTrue(torch.isfinite(state._window_scores[0]).all())

    def test_window_reuses_target_spectral_modes_across_candidate_chunks(self) -> None:
        torch.manual_seed(314)
        state = _state_kwargs(
            LayerWiseSubsetState, include_adamw_scores=False
        )
        state._use_stored_val = True
        state.enable_windowed_execution()
        target = torch.randn(3, 2)
        grad_output = torch.randn(4, 3)
        inputs = torch.randn(4, 2)

        with patch.object(
            advanced_solvers,
            "compute_spectral_modes_with_values",
            wraps=advanced_solvers.compute_spectral_modes_with_values,
        ) as svd:
            state.set_window_chunk("score", 0, 2)
            first, first_beta, first_rank = _compute_muon_spectral_linear_support(
                SimpleNamespace(), state, 0, grad_output[:2], inputs[:2], target
            )
            state.set_window_chunk("score", 2, 4)
            second, second_beta, second_rank = _compute_muon_spectral_linear_support(
                SimpleNamespace(), state, 0, grad_output[2:], inputs[2:], None
            )

        self.assertEqual(svd.call_count, 1)
        self.assertEqual(first_rank, second_rank)
        self.assertTrue(torch.equal(first_beta, second_beta))
        self.assertEqual(first.shape[0], 2)
        self.assertEqual(second.shape[0], 2)
        self.assertIsNotNone(state.get_window_spectral_modes(0))

    def setUp(self) -> None:
        torch.manual_seed(17)
        self.train_input = torch.randn(4, 2)
        self.train_grad_output = torch.randn(4, 3)
        self.val_input = torch.randn(1, 2)
        self.val_grad_output = torch.randn(1, 3)
        self.merged_input = torch.cat((self.train_input, self.val_input), dim=0)
        self.merged_grad_output = torch.cat(
            (self.train_grad_output, self.val_grad_output), dim=0
        )

    def test_layerwise_adamw_only_linear_uses_full_train_gradient(self) -> None:
        module = nn.Linear(2, 3, bias=False)
        state = _state_kwargs(
            LayerWiseSubsetState, include_adamw_scores=False
        )
        hook = _hook(module, "adamw")

        actual_weight, actual_bias = LayerWiseSubsetLinearBackward._backward_full(
            hook,
            None,
            state,
            0,
            self.merged_input,
            None,
            self.merged_grad_output,
            False,
            False,
        )

        expected = torch.einsum(
            "bo,bi->oi", self.train_grad_output, self.train_input
        ) * (5.0 / 4.0)
        self.assertTrue(torch.allclose(actual_weight, expected, atol=1e-6))
        self.assertIsNone(actual_bias)
        self.assertEqual(state._layer_selections, [(0, 4)])

    def test_muon_matrix_score_omits_adamw_bias_contribution(self) -> None:
        module = nn.Linear(2, 3, bias=True)
        state = _state_kwargs(
            LayerWiseSubsetState, include_adamw_scores=False
        )
        hook = _hook(module, "muon")
        spectral_scores = torch.tensor([4.0, 3.0, 2.0, 1.0])

        with patch(
            "drpt.selection.backward._compute_muon_spectral_linear_scores",
            return_value=(spectral_scores, 1),
        ), patch(
            "drpt.selection.backward._adamw_bias_scores_standalone",
            side_effect=AssertionError("AdamW bias score must be omitted"),
        ):
            actual_weight, actual_bias = LayerWiseSubsetLinearBackward._backward_full(
                hook,
                None,
                state,
                0,
                self.merged_input,
                module.bias,
                self.merged_grad_output,
                False,
                False,
            )

        self.assertIsNotNone(actual_weight)
        self.assertIsNotNone(actual_bias)
        self.assertEqual(set(state._last_selected_indices.tolist()), {0, 1})

    def test_beta_weighted_modular_variant_uses_target_singular_values(self) -> None:
        module = nn.Linear(2, 3, bias=False)
        state = _state_kwargs(
            LayerWiseSubsetState,
            include_adamw_scores=False,
            mode_weighting="singular_value",
        )
        hook = _hook(module, "muon")
        support = torch.tensor([
            [3.0, 0.0],
            [0.0, 3.0],
            [2.0, 2.0],
            [1.0, 1.0],
        ])
        beta = torch.tensor([1.0, 4.0])

        with patch(
            "drpt.selection.backward._compute_muon_spectral_linear_support",
            return_value=(support, beta, 2),
        ):
            LayerWiseSubsetLinearBackward._backward_full(
                hook,
                None,
                state,
                0,
                self.merged_input,
                None,
                self.merged_grad_output,
                False,
                False,
            )

        # beta-weighted scores are [3, 12, 10, 5].
        self.assertEqual(set(state._last_selected_indices.tolist()), {1, 2})

    def test_saturated_variant_runs_greedy_on_mode_support(self) -> None:
        module = nn.Linear(2, 3, bias=False)
        state = _state_kwargs(
            LayerWiseSubsetState,
            include_adamw_scores=False,
            saturation=True,
        )
        hook = _hook(module, "muon")
        support = torch.tensor([
            [10.0, 0.0],
            [9.0, 0.0],
            [0.0, 4.0],
            [0.0, 0.0],
        ])

        with patch(
            "drpt.selection.backward._compute_muon_spectral_linear_support",
            return_value=(support, torch.tensor([1.0, 1.0]), 2),
        ):
            LayerWiseSubsetLinearBackward._backward_full(
                hook,
                None,
                state,
                0,
                self.merged_input,
                None,
                self.merged_grad_output,
                False,
                False,
            )

        self.assertEqual(set(state._last_selected_indices.tolist()), {0, 2})
        metrics = state.get_diagnostic_metrics()
        self.assertIn("spectral/layer_0/saturated_objective", metrics)

    def test_existing_mixed_hybrid_criterion_still_adds_bias_score(self) -> None:
        module = nn.Linear(2, 3, bias=True)
        state = _state_kwargs(
            LayerWiseSubsetState, include_adamw_scores=True
        )
        hook = _hook(module, "muon")
        spectral_scores = torch.zeros(4)
        bias_scores = torch.tensor([1.0, 4.0, 3.0, 2.0])

        with patch(
            "drpt.selection.backward._compute_muon_spectral_linear_scores",
            return_value=(spectral_scores, 1),
        ), patch(
            "drpt.selection.backward._adamw_bias_scores_standalone",
            return_value=bias_scores,
        ) as bias_mock:
            LayerWiseSubsetLinearBackward._backward_full(
                hook,
                None,
                state,
                0,
                self.merged_input,
                module.bias,
                self.merged_grad_output,
                False,
                False,
            )

        bias_mock.assert_called_once()
        self.assertEqual(set(state._last_selected_indices.tolist()), {1, 2})

    def test_global_adamw_linear_contributes_no_score(self) -> None:
        module = nn.Linear(2, 3, bias=False)
        state = _state_kwargs(GlobalSubsetState, include_adamw_scores=False)
        hook = _hook(module, "adamw")

        GlobalSubsetLinearBackward._accumulate_full(
            hook,
            state,
            0,
            self.merged_input,
            self.merged_grad_output,
            False,
            False,
        )

        self.assertTrue(torch.equal(
            state.grad_dot_scores, torch.zeros_like(state.grad_dot_scores)
        ))
        metrics = state.get_diagnostic_metrics()
        self.assertIn("spectral/layer_0/adamw_score_omitted", metrics)

    def test_variant_factories_fix_matrix_only_weighting_and_saturation(self) -> None:
        merged_hook = SimpleNamespace(wrap_nonlinear_layers=lambda: None)
        separate_hook = SimpleNamespace(check_unhooked_trainable_params=lambda: None)
        expectations = {
            "LayerWiseMuonMatrixSpectralP": ("singular_value", False),
            "LayerWiseMuonMatrixSpectralSat": ("uniform", True),
            "LayerWiseMuonMatrixSpectralSatP": ("singular_value", True),
        }
        for method, (weighting, saturation) in expectations.items():
            for factory, hook in (
                (create_merged_batch_strategy, merged_hook),
                (create_separate_batch_strategy, separate_hook),
            ):
                strategy = factory(method, hook)
                config = _advanced_solver_config(strategy)
                self.assertEqual(
                    config["muon_surrogate_mode_weighting"], weighting
                )
                self.assertEqual(
                    config["muon_surrogate_saturation"], saturation
                )
                self.assertFalse(
                    config["muon_surrogate_include_adamw_scores"]
                )

    def test_global_saturation_is_rejected_instead_of_silently_using_modular_scores(self) -> None:
        module = nn.Linear(2, 2, bias=False)
        optimizer = SimpleNamespace(
            get_param_optimizer_kind=lambda _param: "muon"
        )
        grad_hook = SimpleNamespace(
            compression_mode=None,
            optimizer=optimizer,
            layer_name_to_module={"layer": module},
        )
        strategy = SimpleNamespace(
            selection_mode="topk",
            use_second_order=False,
            scoring_method="reduced_ghost",
            grad_hook=grad_hook,
            solver_config={
                "muon_surrogate_saturation": True,
                "muon_surrogate_include_adamw_scores": False,
            },
        )

        with self.assertRaisesRegex(ValueError, "only for layerwise"):
            _validate_advanced_strategy(strategy, "muon_spectral")


if __name__ == "__main__":
    unittest.main()
