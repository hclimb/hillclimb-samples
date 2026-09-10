from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from drpt.hook import GradientHook
from drpt.selection.optimizer_aware import (
    compute_optimizer_aware_linear_scores,
    maybe_precondition_embedding_val,
)
from drpt.selection.state import LayerWiseSubsetState
from drpt.selection.strategies import create_separate_batch_strategy
from drpt.selection.utils import compute_scores_and_similarity
from drpt.validation_cache import ValidationCache


def _window_state(*, diagnostic_interval: int = 0, global_step: int = 1):
    state = LayerWiseSubsetState(
        train_batch_size=4,
        num_layers=1,
        frac=0.5,
        lr=1.0,
        device="cpu",
        dtype=torch.float32,
        use_second_order=False,
        selection_mode="topk",
        optimizer_aware=True,
        optimizer_aware_config={
            "optimizer_type": "adamw",
            "matrix_geometry": "adamw",
            "vector_geometry": "adamw",
            "target_mode": "opta",
        },
        optimizer_aware_diagnostic_interval=diagnostic_interval,
        global_step=global_step,
    )
    state.enable_windowed_execution()
    return state


def _adamw_hook(module: nn.Module, target: torch.Tensor):
    parameter = module.weight
    group = {"lr": 2.0e-3, "betas": (0.9, 0.99), "eps": 1.0e-8}
    optimizer = SimpleNamespace(
        state={
            parameter: {
                "step": torch.tensor(3.0),
                "exp_avg_sq": torch.linspace(
                    0.1, 0.9, parameter.numel(), dtype=torch.float32
                ).reshape_as(parameter),
            }
        }
    )
    cache = ValidationCache(1)
    cache.start_capture(mode="full", accumulation_dtype=torch.float32)
    cache.store_precomputed(0, target)
    cache.end_capture(total_tokens=1)
    return SimpleNamespace(
        optimizer=optimizer,
        optimizer_aware_config={"optimizer_type": "adamw"},
        _optimizer_group_by_param_id={id(parameter): group},
        _get_module_from_idx=lambda _index: module,
        _val_cache=cache,
    ), group


class OptimizerProbeCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(731)

    def test_linear_opta_probe_is_built_once_and_matches_uncached_scores(self):
        module = nn.Linear(3, 2, bias=False)
        target = torch.randn_like(module.weight)
        hook, group = _adamw_hook(module, target)
        state = _window_state()
        grad_output = torch.randn(4, 3, 2)
        inputs = torch.randn(4, 3, 3)

        beta2 = group["betas"][1]
        exp_avg_sq = hook.optimizer.state[module.weight]["exp_avg_sq"]
        inv_rms = 1.0 / (
            exp_avg_sq.sqrt() / (1.0 - beta2**3) ** 0.5 + group["eps"]
        )
        expected_probe = target * inv_rms * group["lr"]
        expected, _ = compute_scores_and_similarity(
            grad_output, inputs, None, None, expected_probe, False
        )

        import drpt.selection.optimizer_aware as optimizer_aware

        observed = []
        with patch.object(
            optimizer_aware,
            "_get_adamw_inv_rms",
            wraps=optimizer_aware._get_adamw_inv_rms,
        ) as build_inv_rms:
            for start in (0, 2):
                state.set_window_chunk("score", start, start + 2)
                raw_target = hook._val_cache.get_full(0)
                scores, _, geometry, raw_scores = (
                    compute_optimizer_aware_linear_scores(
                        hook,
                        state,
                        0,
                        grad_output[start : start + 2],
                        inputs[start : start + 2],
                        None,
                        None,
                        raw_target,
                        False,
                    )
                )
                observed.append(scores)
                self.assertEqual(geometry, "adamw_opta")
                self.assertIsNone(raw_scores)

        self.assertEqual(build_inv_rms.call_count, 1)
        self.assertTrue(torch.allclose(torch.cat(observed), expected, atol=1e-6))
        self.assertIsNone(hook._val_cache.get_full(0))

    def test_diagnostic_step_keeps_raw_target_and_returns_raw_scores(self):
        module = nn.Linear(3, 2, bias=False)
        target = torch.randn_like(module.weight)
        hook, _ = _adamw_hook(module, target)
        state = _window_state(diagnostic_interval=100, global_step=200)
        grad_output = torch.randn(2, 3, 2)
        inputs = torch.randn(2, 3, 3)
        state.set_window_chunk("score", 0, 2)

        scores, _, _, raw_scores = compute_optimizer_aware_linear_scores(
            hook,
            state,
            0,
            grad_output,
            inputs,
            None,
            None,
            hook._val_cache.get_full(0),
            False,
        )
        expected_raw, _ = compute_scores_and_similarity(
            grad_output, inputs, None, None, target, False
        )

        self.assertIsNotNone(raw_scores)
        self.assertTrue(torch.allclose(raw_scores, expected_raw, atol=1e-6))
        self.assertIsNotNone(hook._val_cache.get_full(0))
        self.assertFalse(torch.allclose(scores, raw_scores))

    def test_embedding_probe_is_built_once_and_reuses_the_same_tensor(self):
        module = nn.Embedding(7, 3)
        target = torch.randn_like(module.weight)
        hook, _ = _adamw_hook(module, target)
        state = _window_state()

        import drpt.selection.optimizer_aware as optimizer_aware

        state.set_window_chunk("score", 0, 2)
        with patch.object(
            optimizer_aware,
            "_get_adamw_inv_rms",
            wraps=optimizer_aware._get_adamw_inv_rms,
        ) as build_inv_rms:
            first, first_geometry = maybe_precondition_embedding_val(
                hook, state, 0, hook._val_cache.get_full(0)
            )
            state.set_window_chunk("score", 2, 4)
            second, second_geometry = maybe_precondition_embedding_val(
                hook, state, 0, None
            )

        self.assertEqual(build_inv_rms.call_count, 1)
        self.assertEqual(first_geometry, second_geometry)
        self.assertTrue(torch.equal(first, second))
        self.assertIsNone(hook._val_cache.get_full(0))

    @staticmethod
    def _run_embedding_window(diagnostic_interval: int):
        torch.manual_seed(177)
        model = nn.Sequential(nn.Embedding(11, 3))
        hook = GradientHook(model, ["0"], device="cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3)
        optimizer.state[model[0].weight].update(
            {
                "step": torch.tensor(2.0),
                "exp_avg_sq": torch.linspace(
                    0.2, 0.8, model[0].weight.numel()
                ).reshape_as(model[0].weight),
            }
        )
        hook.set_optimizer(optimizer)
        candidate_ids = torch.tensor(
            [[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.long
        )
        labels = torch.tensor([[-100, 1, 1]] * 4, dtype=torch.long)
        try:
            hook.start_val_capture(
                scoring_method="reduced_ghost", full_precision=True
            )
            model(torch.tensor([[1, 3], [5, 7]])).square().mean().backward()
            hook.end_val_capture(val_total_tokens=2)
            strategy = create_separate_batch_strategy(
                "LayerWiseOptimizerAwareSubset",
                hook,
                frac=0.5,
                selection_mode="topk",
                record_selections=True,
                scoring_method="reduced_ghost",
                seed=42,
                optimizer_aware_diagnostic_interval=diagnostic_interval,
            )
            strategy.execute_windowed_training_step(
                model=model,
                batch_size=4,
                microbatch_size=2,
                labels=labels,
                compute_chunk_loss_fn=lambda start, end: model(
                    candidate_ids[start:end]
                ).square().mean(),
                lr=2.0e-3,
                global_step=100,
            )
            return (
                strategy.last_selection_record[0]["selected_indices"],
                model[0].weight.grad.detach().clone(),
                strategy.last_diagnostic_metrics,
            )
        finally:
            hook.clear_val_buffer()
            hook.remove_hooks()

    def test_two_chunk_embedding_skips_raw_scores_when_diagnostics_are_off(self):
        selected, gradient, metrics = self._run_embedding_window(0)
        self.assertEqual(len(selected), 2)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertFalse(any("raw_opt" in key for key in metrics))

    def test_two_chunk_embedding_samples_raw_scores_without_changing_update(self):
        baseline_selected, baseline_gradient, _ = self._run_embedding_window(0)
        selected, gradient, metrics = self._run_embedding_window(100)
        self.assertEqual(selected, baseline_selected)
        self.assertTrue(torch.equal(gradient, baseline_gradient))
        self.assertTrue(any("raw_opt" in key for key in metrics))


class WindowedSoftFastPathTests(unittest.TestCase):
    def test_adamw_soft_score_phase_never_builds_discarded_objective(self):
        torch.manual_seed(991)
        model = nn.Sequential(nn.Linear(3, 1, bias=False))
        hook = GradientHook(model, ["0"], device="cpu")
        candidates = torch.randn(4, 2, 3)
        targets = torch.randn(4, 2, 1)
        labels = torch.tensor([[-100, 1, 1]] * 4, dtype=torch.long)
        try:
            hook.start_val_capture(
                scoring_method="reduced_ghost", full_precision=True
            )
            F_target = model(torch.randn(2, 2, 3)).square().mean()
            F_target.backward()
            hook.end_val_capture(val_total_tokens=2)
            strategy = create_separate_batch_strategy(
                "LayerWiseSoftWeighting",
                hook,
                frac=0.5,
                selection_mode="topk",
                record_selections=True,
                scoring_method="reduced_ghost",
                seed=42,
                soft_weighting_steps=4,
                soft_weighting_patience=5,
                soft_weighting_gamma=0.0,
            )

            with patch(
                "drpt.selection.backward._make_soft_linear_objective",
                side_effect=AssertionError("discarded objective was built"),
            ):
                _, stats = strategy.execute_windowed_training_step(
                    model=model,
                    batch_size=4,
                    microbatch_size=1,
                    labels=labels,
                    compute_chunk_loss_fn=lambda start, end: (
                        model(candidates[start:end]) - targets[start:end]
                    ).square().mean(),
                    lr=1.0e-3,
                    global_step=1,
                )

            self.assertEqual(stats["selection/logical_candidates"], 4.0)
            record = strategy.last_selection_record[0]
            self.assertEqual(len(record["weights"]), 4)
            self.assertAlmostEqual(sum(record["weights"]), 2.0, places=5)
        finally:
            hook.clear_val_buffer()
            hook.remove_hooks()

    def test_muon_soft_window_skips_diagnostic_reference_probe(self):
        torch.manual_seed(1234)
        model = nn.Sequential(nn.Linear(3, 2, bias=False))
        hook = GradientHook(model, ["0"], device="cpu")

        class _MuonFixtureOptimizer:
            def __init__(self, parameter):
                self.state = {}
                self.param_groups = [{
                    "params": [parameter],
                    "lr": 3.0e-4,
                    "momentum": 0.9,
                    "nesterov": True,
                    "muon_ns_steps": 1,
                    "muon_eps": 1e-7,
                    "muon_lr_shape_scale": True,
                    "adjust_lr_fn": "original",
                }]

            @staticmethod
            def get_param_optimizer_kind(_parameter):
                return "muon"

            def get_param_state(self, parameter):
                return self.state.get(parameter, {})

        hook.set_optimizer(_MuonFixtureOptimizer(model[0].weight))
        candidates = torch.randn(4, 3)
        targets = torch.randn(4, 2)
        labels = torch.tensor([[-100, 1]] * 4, dtype=torch.long)
        try:
            hook.start_val_capture(
                scoring_method="reduced_ghost", full_precision=True
            )
            model(torch.randn(2, 3)).square().mean().backward()
            hook.end_val_capture(val_total_tokens=2)
            strategy = create_separate_batch_strategy(
                "LayerWiseSoftWeighting",
                hook,
                frac=0.5,
                selection_mode="topk",
                record_selections=True,
                scoring_method="reduced_ghost",
                seed=42,
                soft_weighting_steps=2,
                soft_weighting_patience=3,
                soft_weighting_gamma=0.0,
                soft_replay_precision="bf16_fp32",
            )

            with patch(
                "drpt.selection.backward._soft_reference_linear_scores",
                side_effect=AssertionError("diagnostic Muon probe entered hot path"),
            ):
                _, stats = strategy.execute_windowed_training_step(
                    model=model,
                    batch_size=4,
                    microbatch_size=2,
                    labels=labels,
                    compute_chunk_loss_fn=lambda start, end: (
                        model(candidates[start:end]) - targets[start:end]
                    ).square().mean(),
                    lr=3.0e-4,
                    global_step=0,
                )

            self.assertEqual(stats["selection/logical_candidates"], 4.0)
            record = strategy.last_selection_record[0]
            self.assertEqual(len(record["weights"]), 4)
            self.assertNotIn("reference_scores", record)
            self.assertEqual(record["valid_token_counts"], [1.0] * 4)
            self.assertTrue(torch.isfinite(model[0].weight.grad).all())
        finally:
            hook.clear_val_buffer()
            hook.remove_hooks()


if __name__ == "__main__":
    unittest.main()
