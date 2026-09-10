"""CPU invariants for Dolci32k's logical N=16 candidate window.

These tests isolate the global-decision and replay algebra from the model hook:
the same synthetic per-example scores/supports are staged as C=16, C=2, and
C=1, then the finalized decisions and reconstructed parameter gradients are
compared.  Optimizer-map correctness itself is covered by the advanced solver
tests; this file guards against making a decision independently per chunk.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from drpt.hook import GradientHook
from drpt.selection import create_separate_batch_strategy
from drpt.selection.backward import (
    _window_replay_linear_gradient,
    finalize_windowed_layerwise_selection,
)
from drpt.selection.state import LayerWiseSubsetState
from SFT.train.trainer import LayerWiseSubsetTrainer


BASELINES = {
    "adamw/FullTraining": ("full", {}),
    "adamw/LayerwiseRaw": ("score", {}),
    "adamw/LayerwiseSoft": (
        "soft",
        {"soft_weighting_constraint": "capped_simplex"},
    ),
    "adamw/LayerwiseSoftP": (
        "soft",
        {"soft_weighting_constraint": "probability_simplex"},
    ),
    "adamw/LayerwiseOptA": ("opta", {}),
    "muon/FullTraining": ("full", {}),
    "muon/LayerwiseRaw": ("score", {}),
    "muon/LayerwiseSoft": (
        "soft",
        {"soft_weighting_constraint": "capped_simplex"},
    ),
    "muon/LayerwiseSoftP": (
        "soft",
        {"soft_weighting_constraint": "probability_simplex"},
    ),
    "muon/LayerwiseMuonSur": (
        "spectral",
        {"muon_surrogate_mode_weighting": "uniform"},
    ),
    "muon/LayerwiseMuonPSur": (
        "spectral",
        {"muon_surrogate_mode_weighting": "singular_value"},
    ),
    "muon/LayerwiseMuonSatSur": (
        "saturated",
        {
            "muon_surrogate_saturation": True,
            "muon_surrogate_mode_weighting": "uniform",
        },
    ),
    "muon/LayerwiseMuonSatPSur": (
        "saturated",
        {
            "muon_surrogate_saturation": True,
            "muon_surrogate_mode_weighting": "singular_value",
        },
    ),
}


class _FinalizerHook:
    def __init__(self, state):
        self.selection_state = state

    def get_optimizer_group_keys(self):
        return ["fixture_layer"]


class ExactWindowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.scores = torch.tensor(
            [0.2, 2.0, -1.0, 4.0, 3.0, 0.5, 1.5, -0.2],
            dtype=torch.float32,
        )
        self.support = torch.tensor(
            [
                [1.0, 0.0, 0.2],
                [0.0, 1.0, 0.4],
                [0.8, 0.8, 0.1],
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.2],
                [1.5, 1.5, 0.0],
                [0.1, 0.2, 2.0],
                [2.0, 0.2, 0.5],
            ],
            dtype=torch.float32,
        )
        self.beta = torch.tensor([2.0, 1.0, 0.5], dtype=torch.float32)
        self.tokens = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4])
        self.grad_output = torch.randn(8, 3, 4)
        self.inputs = torch.randn(8, 3, 5)

    def _finalize(self, kind: str, config: dict, chunk_size: int):
        variant = {
            "soft": "soft",
            "spectral": "muon_spectral",
            "saturated": "muon_spectral",
        }.get(kind, "score")
        solver_config = {
            "soft_weighting_steps": 12,
            "soft_weighting_lr": 0.1,
            "soft_weighting_tol": 0.0,
            "soft_weighting_patience": 20,
            "soft_weighting_gamma": 0.0,
            **config,
        }
        # This class hand-builds an 8-wide score/support fixture to pin the
        # finalize algebra, which is independent of the campaign's window size.
        # The end-to-end N=16 window is covered by
        # ExactWindowHookIntegrationTests below.
        state = LayerWiseSubsetState(
            train_batch_size=8,
            num_layers=1,
            frac=0.5,
            lr=1.0,
            device="cpu",
            dtype=torch.float32,
            use_second_order=False,
            selection_mode="topk",
            record_selections=True,
            selection_variant=variant,
            solver_config=solver_config,
            seed=42,
            global_step=17,
            optimizer_aware=(kind == "opta"),
            optimizer_aware_config={"target_mode": "opta"},
        )
        total_tokens = self.tokens.sum()
        state.set_token_counts(self.tokens, total_tokens, total_tokens)
        state.enable_windowed_execution()
        if kind == "full":
            state.mark_window_full_batch(0)
        else:
            if kind == "opta":
                all_scores = self.scores * torch.tensor(
                    [1.0, 0.7, 1.2, 0.6, 1.1, 0.9, 1.4, 0.8]
                )
            elif kind == "spectral":
                weighting = config.get("muon_surrogate_mode_weighting")
                weights = self.beta if weighting == "singular_value" else torch.ones_like(self.beta)
                all_scores = self.support @ weights
            else:
                all_scores = self.scores
            for start in range(0, 8, chunk_size):
                end = min(start + chunk_size, 8)
                state.set_window_chunk("score", start, end)
                if kind == "saturated":
                    state.store_window_support(
                        0, self.support[start:end], self.beta
                    )
                else:
                    state.store_window_scores(0, all_scores[start:end])
        finalize_windowed_layerwise_selection(_FinalizerHook(state))
        return state

    @staticmethod
    def _assert_same_decision(testcase, expected, actual, baseline):
        expected_kind, expected_values = expected.get_window_decision(0)
        actual_kind, actual_values = actual.get_window_decision(0)
        testcase.assertEqual(expected_kind, actual_kind, baseline)
        if expected_kind == "indices":
            testcase.assertTrue(
                torch.equal(expected_values, actual_values),
                f"{baseline}: {expected_values} != {actual_values}",
            )
        else:
            testcase.assertTrue(
                torch.allclose(expected_values, actual_values, atol=0.0, rtol=0.0),
                f"{baseline}: soft weights changed with chunk size",
            )

    def _replay(self, state, chunk_size: int):
        weight = torch.zeros(4, 5)
        bias = torch.zeros(4)
        for start in range(0, 8, chunk_size):
            end = min(start + chunk_size, 8)
            state.set_window_chunk("replay", start, end)
            chunk_weight, chunk_bias = _window_replay_linear_gradient(
                _FinalizerHook(state),
                state,
                0,
                self.grad_output[start:end],
                self.inputs[start:end],
                True,
            )
            if chunk_weight is not None:
                weight.add_(chunk_weight)
            if chunk_bias is not None:
                bias.add_(chunk_bias)
        return weight, bias

    def test_window_soft_replay_forwards_configured_precision(self):
        state = self._finalize(
            "soft",
            {
                "soft_weighting_constraint": "capped_simplex",
                "soft_replay_precision": "bf16_fp32",
            },
            8,
        )
        state.set_window_chunk("replay", 0, 8)
        with patch(
            "drpt.selection.advanced_solvers.weighted_linear_gradients",
            return_value=(torch.zeros(4, 5), torch.zeros(4)),
        ) as replay:
            _window_replay_linear_gradient(
                _FinalizerHook(state),
                state,
                0,
                self.grad_output,
                self.inputs,
                True,
            )
        self.assertEqual(
            replay.call_args.kwargs["replay_precision"], "bf16_fp32"
        )

    def test_all_13_baseline_labels_make_one_global_8_way_decision(self):
        self.assertEqual(len(BASELINES), 13)
        for baseline, (kind, config) in BASELINES.items():
            with self.subTest(baseline=baseline):
                unsplit = self._finalize(kind, config, 8)
                for chunk_size in (2, 1):
                    chunked = self._finalize(kind, config, chunk_size)
                    self._assert_same_decision(
                        self, unsplit, chunked, baseline
                    )

    def test_chunked_replay_reconstructs_unsplit_parameter_gradients(self):
        for baseline, (kind, config) in BASELINES.items():
            with self.subTest(baseline=baseline):
                state = self._finalize(kind, config, 8)
                expected_weight, expected_bias = self._replay(state, 8)
                for chunk_size in (2, 1):
                    chunked_state = self._finalize(kind, config, chunk_size)
                    actual_weight, actual_bias = self._replay(
                        chunked_state, chunk_size
                    )
                    self.assertTrue(
                        torch.allclose(
                            expected_weight, actual_weight, atol=2e-6, rtol=2e-6
                        ),
                        baseline,
                    )
                    self.assertTrue(
                        torch.allclose(
                            expected_bias, actual_bias, atol=2e-6, rtol=2e-6
                        ),
                        baseline,
                    )

    def test_cpu_bf16_candidate_factors_reassemble_in_logical_order(self):
        state = LayerWiseSubsetState(
            train_batch_size=8,
            num_layers=1,
            frac=0.5,
            lr=1.0,
            device="cpu",
            dtype=torch.float32,
            selection_variant="soft",
        )
        state.enable_windowed_execution()
        for start in range(0, 8, 2):
            end = start + 2
            state.set_window_chunk("score", start, end)
            state.store_window_factors(
                0, self.grad_output[start:end], self.inputs[start:end]
            )
        grad_output, inputs = state.get_window_factors(0)
        stored_grad_output, stored_inputs, filled = state._window_factors[0]
        self.assertEqual(grad_output.device.type, "cpu")
        self.assertEqual(inputs.device.type, "cpu")
        self.assertEqual(grad_output.dtype, torch.bfloat16)
        self.assertEqual(inputs.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(grad_output, self.grad_output.to(torch.bfloat16)))
        self.assertTrue(torch.equal(inputs, self.inputs.to(torch.bfloat16)))
        self.assertEqual(grad_output.data_ptr(), stored_grad_output.data_ptr())
        self.assertEqual(inputs.data_ptr(), stored_inputs.data_ptr())
        self.assertTrue(bool(filled.all()))

    def test_soft_selection_record_retains_offline_opta_inputs(self):
        state = self._finalize(
            "soft", {"soft_weighting_constraint": "capped_simplex"}, 2
        )
        record = state._selection_records[0]
        self.assertEqual(record["layer_idx"], 0)
        self.assertEqual(record["global_step"], 17)
        self.assertEqual(record["num_selected"], 4)
        self.assertEqual(record["selection_seed"], state.derived_seed(0))
        self.assertEqual(record["valid_token_counts"], self.tokens.float().tolist())
        self.assertEqual(len(record["weights"]), 8)
        self.assertEqual(len(record["reference_scores"]), 8)


class ExactWindowHookIntegrationTests(unittest.TestCase):
    """Exercise the real custom-autograd score/finalize/replay lifecycle."""

    def test_hook_token_counts_match_causal_lm_shift(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 1))
        hook = GradientHook(model, ["0"], device="cpu")
        try:
            labels = torch.tensor(
                [
                    [9, -100, 4],
                    [-100, 7, -100],
                ]
            )
            hook.set_token_counts(labels)
            # The first label is never predicted by a causal LM. Counting the
            # unshifted tensor here would incorrectly report [2, 1].
            self.assertTrue(torch.equal(hook.tokens_per_sample, torch.tensor([1, 1])))
            self.assertEqual(int(hook.total_tokens), 2)
        finally:
            hook.remove_hooks()

    @staticmethod
    def _run(microbatch_size: int, target_microbatch_size: int = 2):
        torch.manual_seed(91)
        model = torch.nn.Sequential(torch.nn.Linear(3, 1, bias=True))
        candidates = torch.randn(16, 4, 3)
        candidate_targets = torch.randn(16, 4, 1)
        target_inputs = torch.tensor(
            [[0.5, -1.0, 0.25], [-0.75, 0.2, 1.5]], dtype=torch.float32
        )
        target_outputs = torch.tensor([[1.0], [-0.5]], dtype=torch.float32)
        # Variable causal-LM response-token counts exercise the complete-window
        # token denominator, not merely equal-size sample averaging.
        labels = torch.full((16, 5), -100, dtype=torch.long)
        for row, count in enumerate((1, 2, 3, 4, 1, 2, 3, 4) * 2):
            labels[row, 1 : count + 1] = 1

        hook = GradientHook(model, ["0"], device="cpu")
        try:
            hook.start_val_capture(
                scoring_method="reduced_ghost", full_precision=True
            )
            model.zero_grad(set_to_none=True)
            for start in range(0, 2, target_microbatch_size):
                end = min(start + target_microbatch_size, 2)
                target_loss = F.mse_loss(
                    model(target_inputs[start:end]), target_outputs[start:end]
                )
                (target_loss * ((end - start) / 2.0)).backward()
            hook.end_val_capture(val_total_tokens=2)

            strategy = create_separate_batch_strategy(
                "LayerWiseSubset",
                hook,
                frac=0.5,
                selection_mode="topk",
                record_selections=True,
                scoring_method="reduced_ghost",
                seed=42,
            )
            before = [parameter.detach().clone() for parameter in model.parameters()]
            loss, stats = strategy.execute_windowed_training_step(
                model=model,
                batch_size=16,
                microbatch_size=microbatch_size,
                labels=labels,
                compute_chunk_loss_fn=lambda start, end: (
                    (
                        model(candidates[start:end])
                        - candidate_targets[start:end]
                    )
                    .square()
                    .squeeze(-1)[labels[start:end, 1:] != -100]
                    .mean()
                ),
                lr=1.0e-5,
                global_step=3,
            )
            self_cache_count = hook.val_cache.get_num_captured()
            if self_cache_count != 0:
                raise AssertionError(
                    "target cache must be released before window replay returns"
                )
            after_backward = [
                parameter.detach().clone() for parameter in model.parameters()
            ]
            gradients = [
                parameter.grad.detach().clone() for parameter in model.parameters()
            ]
            # The strategy owns backward only. The surrounding HF Trainer must
            # perform the single optimizer/scheduler/global-step transition.
            for initial, current in zip(before, after_backward):
                if not torch.equal(initial, current):
                    raise AssertionError("window engine stepped parameters internally")
            optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
            optimizer.step()
            after_step = [parameter.detach().clone() for parameter in model.parameters()]
            selected = strategy.last_selection_record[0]["selected_indices"]
            return loss, stats, selected, gradients, after_step
        finally:
            hook.clear_val_buffer()
            hook.remove_hooks()

    def test_custom_autograd_gradient_and_single_outer_step_match_c16_c2_c1(self):
        reference = self._run(16)
        for chunk_size in (2, 1):
            with self.subTest(chunk_size=chunk_size):
                # Compare both candidate C=1/2 and target C=1 against the
                # unsplit candidate/target reference.
                actual = self._run(chunk_size, target_microbatch_size=1)
                self.assertEqual(actual[2], reference[2])
                self.assertAlmostEqual(float(actual[0]), float(reference[0]), places=6)
                self.assertEqual(actual[1]["selection/logical_candidates"], 16.0)
                for expected, observed in zip(reference[3], actual[3]):
                    self.assertTrue(
                        torch.allclose(expected, observed, atol=2e-6, rtol=2e-6)
                    )
                for expected, observed in zip(reference[4], actual[4]):
                    self.assertTrue(
                        torch.allclose(expected, observed, atol=2e-6, rtol=2e-6)
                    )

    def test_muon_spectral_zero_target_finalizes_as_empty_update(self):
        torch.manual_seed(7)
        model = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False))
        hook = GradientHook(model, ["0"], device="cpu")

        class _MuonFixtureOptimizer:
            def __init__(self, parameter):
                self.param_groups = [{"params": [parameter], "lr": 3.0e-4}]

            @staticmethod
            def get_param_optimizer_kind(_parameter):
                return "muon"

        hook.set_optimizer(_MuonFixtureOptimizer(model[0].weight))
        try:
            hook.start_val_capture(
                scoring_method="reduced_ghost", full_precision=True
            )
            # Exercise the custom backward while producing a genuine cached
            # zero target gradient for this layer.
            (model(torch.randn(2, 3)) * 0.0).sum().backward()
            hook.end_val_capture(val_total_tokens=2)
            strategy = create_separate_batch_strategy(
                "LayerWiseMuonMatrixSpectral",
                hook,
                frac=0.5,
                selection_mode="topk",
                record_selections=True,
                scoring_method="reduced_ghost",
                seed=42,
                muon_surrogate_include_adamw_scores=False,
            )
            candidates = torch.randn(16, 3)
            labels = torch.tensor([[-100, 1]] * 16, dtype=torch.long)
            _loss, stats = strategy.execute_windowed_training_step(
                model=model,
                batch_size=16,
                microbatch_size=1,
                labels=labels,
                compute_chunk_loss_fn=lambda start, end: model(
                    candidates[start:end]
                ).square().mean(),
                lr=3.0e-4,
                global_step=0,
            )
            self.assertEqual(stats["selection/n_selected"], 0)
            self.assertTrue(
                torch.equal(
                    model[0].weight.grad, torch.zeros_like(model[0].weight)
                )
            )
        finally:
            hook.clear_val_buffer()
            hook.remove_hooks()


class TargetSamplerIsolationTests(unittest.TestCase):
    class _TrainerPath:
        get_val_dataloader = LayerWiseSubsetTrainer.get_val_dataloader

        def __init__(self, method: str, data_seed: int):
            self.args = SimpleNamespace(
                method=method,
                dataloader_num_workers=0,
                dataloader_pin_memory=False,
            )
            self.data_collator = lambda rows: torch.tensor(rows)
            self._target_sampler_generator = torch.Generator()
            self._target_sampler_generator.manual_seed(data_seed)

    @staticmethod
    def _two_epoch_stream(path):
        stream = []
        dataset = list(range(16))
        for _ in range(2):
            loader = path.get_val_dataloader(
                dataset, batch_size=2, shuffle=True
            )
            for batch in loader:
                stream.extend(int(value) for value in batch)
        return stream

    def test_target_index_stream_is_method_and_global_rng_independent(self):
        raw_path = self._TrainerPath("LayerWiseSubset", data_seed=43)
        raw_stream = self._two_epoch_stream(raw_path)
        # Simulate a different optimizer/model setup consuming the process-wide
        # RNG before the second method reaches its target sampler.
        torch.manual_seed(999)
        torch.rand(10_000)
        soft_path = self._TrainerPath("LayerWiseSoftWeighting", data_seed=43)
        self.assertEqual(raw_stream, self._two_epoch_stream(soft_path))


if __name__ == "__main__":
    unittest.main()
