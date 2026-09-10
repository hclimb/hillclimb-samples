"""CPU invariants for GlobalRaw/GlobalOptA under the exact candidate window.

Dolci32k drives selection through a logical N=16 candidate window split
into compute chunks (C=1 in the completed campaigns).  Only the layer-wise
strategy implemented that window, so the global methods could not run at all on
those profiles.  The windowed global path reuses the layer-wise machinery and
changes only the decision rule: pool every layer's window scores and pick one
subset shared by all layers, which is what ``GlobalSubsetState`` accumulates in
the unwindowed path.

These tests pin the two properties that makes the reuse legitimate:

1. the decision and the replayed parameter gradients do not depend on C, and
2. they agree with the unwindowed ``GlobalSubset`` one-pass strategy.
"""

from __future__ import annotations

import unittest

import torch

from drpt.hook import GradientHook
from drpt.selection import create_separate_batch_strategy
from drpt.selection.backward import (
    _window_replay_linear_gradient,
    finalize_windowed_global_selection,
    finalize_windowed_layerwise_selection,
)
from drpt.selection.state import LayerWiseSubsetState
from drpt.selection.strategies import (
    SeparateBatchGlobalSubsetOnePassStrategy,
    SeparateBatchOptimizerAwareGlobalSubsetStrategy,
    SeparateBatchWindowedGlobalSubsetStrategy,
    SeparateBatchWindowedOptimizerAwareGlobalSubsetStrategy,
)


NUM_LAYERS = 3
BATCH = 8


class _FinalizerHook:
    def __init__(self, state, num_layers=NUM_LAYERS):
        self.selection_state = state
        self._num_layers = num_layers

    def get_optimizer_group_keys(self):
        return [f"adamw:fixture_{index}" for index in range(self._num_layers)]


class WindowedGlobalFinalizeTests(unittest.TestCase):
    """Pin the pooling algebra independently of the model hook."""

    def setUp(self):
        torch.manual_seed(2026)
        # Per-layer score profiles that disagree, so a global decision is
        # genuinely different from three independent per-layer decisions.
        self.layer_scores = [
            torch.tensor([0.2, 2.0, -1.0, 4.0, 3.0, 0.5, 1.5, -0.2]),
            torch.tensor([3.0, -1.5, 2.5, -2.0, 0.1, 2.2, -0.4, 1.0]),
            torch.tensor([-1.0, 0.4, 1.8, 0.9, -2.5, 3.3, 0.7, 0.2]),
        ]
        self.tokens = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4])
        self.grad_output = torch.randn(BATCH, 3, 4)
        self.inputs = torch.randn(BATCH, 3, 5)

    def _finalize(self, chunk_size: int, *, layerwise: bool = False):
        state = LayerWiseSubsetState(
            train_batch_size=BATCH,
            num_layers=NUM_LAYERS,
            frac=0.5,
            lr=1.0,
            device="cpu",
            dtype=torch.float32,
            use_second_order=False,
            selection_mode="topk",
            record_selections=True,
            selection_variant="score",
            seed=42,
            global_step=17,
        )
        total_tokens = self.tokens.sum()
        state.set_token_counts(self.tokens, total_tokens, total_tokens)
        state.enable_windowed_execution()
        for start in range(0, BATCH, chunk_size):
            end = min(start + chunk_size, BATCH)
            state.set_window_chunk("score", start, end)
            for layer_idx in range(NUM_LAYERS):
                state.store_window_scores(
                    layer_idx, self.layer_scores[layer_idx][start:end]
                )
        hook = _FinalizerHook(state)
        if layerwise:
            finalize_windowed_layerwise_selection(hook)
        else:
            finalize_windowed_global_selection(hook)
        return state

    def _replay(self, state, chunk_size: int, layer_idx: int):
        weight = torch.zeros(4, 5)
        bias = torch.zeros(4)
        for start in range(0, BATCH, chunk_size):
            end = min(start + chunk_size, BATCH)
            state.set_window_chunk("replay", start, end)
            chunk_weight, chunk_bias = _window_replay_linear_gradient(
                _FinalizerHook(state),
                state,
                layer_idx,
                self.grad_output[start:end],
                self.inputs[start:end],
                True,
            )
            if chunk_weight is not None:
                weight.add_(chunk_weight)
            if chunk_bias is not None:
                bias.add_(chunk_bias)
        return weight, bias

    def test_every_layer_receives_the_same_pooled_subset(self):
        state = self._finalize(BATCH)
        expected = torch.topk(sum(self.layer_scores), 4).indices.sort()[0]
        for layer_idx in range(NUM_LAYERS):
            kind, decision = state.get_window_decision(layer_idx)
            self.assertEqual(kind, "indices")
            self.assertTrue(
                torch.equal(decision, expected),
                f"layer {layer_idx}: {decision.tolist()} != {expected.tolist()}",
            )

    def test_fixture_would_be_vacuous_if_layerwise_agreed(self):
        # Guards the tests above: if the per-layer picks happened to coincide,
        # "global == layerwise" would prove nothing about pooling.
        layerwise = self._finalize(BATCH, layerwise=True)
        decisions = {
            tuple(layerwise.get_window_decision(index)[1].tolist())
            for index in range(NUM_LAYERS)
        }
        self.assertGreater(len(decisions), 1)

    def test_decision_is_independent_of_compute_chunk_size(self):
        reference = self._finalize(BATCH)
        for chunk_size in (4, 2, 1):
            with self.subTest(chunk_size=chunk_size):
                chunked = self._finalize(chunk_size)
                for layer_idx in range(NUM_LAYERS):
                    self.assertTrue(
                        torch.equal(
                            reference.get_window_decision(layer_idx)[1],
                            chunked.get_window_decision(layer_idx)[1],
                        )
                    )

    def test_chunked_replay_reconstructs_unsplit_parameter_gradients(self):
        reference = self._finalize(BATCH)
        for layer_idx in range(NUM_LAYERS):
            expected_weight, expected_bias = self._replay(reference, BATCH, layer_idx)
            for chunk_size in (4, 2, 1):
                with self.subTest(chunk_size=chunk_size, layer=layer_idx):
                    chunked = self._finalize(chunk_size)
                    weight, bias = self._replay(chunked, chunk_size, layer_idx)
                    self.assertTrue(
                        torch.allclose(expected_weight, weight, atol=2e-6, rtol=2e-6)
                    )
                    self.assertTrue(
                        torch.allclose(expected_bias, bias, atol=2e-6, rtol=2e-6)
                    )

    def test_soft_variant_is_rejected_rather_than_silently_pooled(self):
        state = LayerWiseSubsetState(
            train_batch_size=BATCH,
            num_layers=1,
            frac=0.5,
            lr=1.0,
            device="cpu",
            selection_variant="soft",
        )
        state.enable_windowed_execution()
        with self.assertRaises(RuntimeError):
            finalize_windowed_global_selection(_FinalizerHook(state, num_layers=1))


class WindowedGlobalMatchesUnwindowedTests(unittest.TestCase):
    """End-to-end parity with the strategy dolci32k could not run."""

    LOGICAL = 16

    @staticmethod
    def _fixture():
        torch.manual_seed(91)
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 4, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(4, 1, bias=True),
        )
        candidates = torch.randn(16, 4, 3)
        candidate_targets = torch.randn(16, 4, 1)
        target_inputs = torch.tensor(
            [[0.5, -1.0, 0.25], [-0.75, 0.2, 1.5]], dtype=torch.float32
        )
        target_outputs = torch.tensor([[1.0], [-0.5]], dtype=torch.float32)
        # Uneven causal-LM response lengths exercise the token denominator
        # rather than plain equal-size averaging.
        labels = torch.full((16, 5), -100, dtype=torch.long)
        for row, count in enumerate((1, 2, 3, 4, 1, 2, 3, 4) * 2):
            labels[row, 1 : count + 1] = 1
        return (
            model,
            candidates,
            candidate_targets,
            target_inputs,
            target_outputs,
            labels,
        )

    @classmethod
    def _capture_target(cls, hook, model, target_inputs, target_outputs):
        hook.start_val_capture(scoring_method="reduced_ghost", full_precision=True)
        model.zero_grad(set_to_none=True)
        loss = (model(target_inputs) - target_outputs).square().mean()
        loss.backward()
        hook.end_val_capture(val_total_tokens=2)

    @classmethod
    def _chunk_loss(cls, model, candidates, candidate_targets, labels, start, end):
        residual = (
            model(candidates[start:end]) - candidate_targets[start:end]
        ).square().squeeze(-1)
        return residual[labels[start:end, 1:] != -100].mean()

    @classmethod
    def _run_windowed(cls, optimizer_aware: bool, microbatch_size: int):
        (
            model,
            candidates,
            candidate_targets,
            target_inputs,
            target_outputs,
            labels,
        ) = cls._fixture()
        hook = GradientHook(model, ["0", "2"], device="cpu")
        try:
            cls._capture_target(hook, model, target_inputs, target_outputs)
            strategy = create_separate_batch_strategy(
                "OptimizerAwareGlobalSubset" if optimizer_aware else "GlobalSubset",
                hook,
                frac=0.5,
                selection_mode="topk",
                record_selections=True,
                scoring_method="reduced_ghost",
                seed=42,
                windowed=True,
            )
            expected_class = (
                SeparateBatchWindowedOptimizerAwareGlobalSubsetStrategy
                if optimizer_aware
                else SeparateBatchWindowedGlobalSubsetStrategy
            )
            assert isinstance(strategy, expected_class), type(strategy)
            _loss, stats = strategy.execute_windowed_training_step(
                model=model,
                batch_size=cls.LOGICAL,
                microbatch_size=microbatch_size,
                labels=labels,
                compute_chunk_loss_fn=lambda start, end: cls._chunk_loss(
                    model, candidates, candidate_targets, labels, start, end
                ),
                lr=1.0e-5,
                global_step=3,
            )
            records = strategy.last_selection_record
            selected = next(
                record["selected_indices"]
                for record in records
                if record.get("layer_idx") == -1
            )
            gradients = [
                parameter.grad.detach().clone() for parameter in model.parameters()
            ]
            return selected, gradients, stats
        finally:
            hook.clear_val_buffer()
            hook.remove_hooks()

    @classmethod
    def _run_unwindowed(cls, optimizer_aware: bool):
        (
            model,
            candidates,
            candidate_targets,
            target_inputs,
            target_outputs,
            labels,
        ) = cls._fixture()
        hook = GradientHook(model, ["0", "2"], device="cpu")
        try:
            cls._capture_target(hook, model, target_inputs, target_outputs)
            strategy = create_separate_batch_strategy(
                "OptimizerAwareGlobalSubset" if optimizer_aware else "GlobalSubset",
                hook,
                frac=0.5,
                selection_mode="topk",
                record_selections=True,
                scoring_method="reduced_ghost",
                subset_mode="one_pass",
                seed=42,
            )
            expected_class = (
                SeparateBatchOptimizerAwareGlobalSubsetStrategy
                if optimizer_aware
                else SeparateBatchGlobalSubsetOnePassStrategy
            )
            assert isinstance(strategy, expected_class), type(strategy)
            _loss, _stats = strategy.execute_training_step(
                model=model,
                batch_size=cls.LOGICAL,
                compute_loss_fn=lambda: (
                    cls._chunk_loss(
                        model, candidates, candidate_targets, labels, 0, cls.LOGICAL
                    ),
                    {},
                ),
                lr=1.0e-5,
                labels=labels,
            )
            selected = strategy.last_selection_record[0]["selected_indices"]
            gradients = [
                parameter.grad.detach().clone() for parameter in model.parameters()
            ]
            return selected, gradients
        finally:
            hook.clear_val_buffer()
            hook.remove_hooks()

    def test_windowed_global_is_chunk_invariant(self):
        for optimizer_aware in (False, True):
            label = "GlobalOptA" if optimizer_aware else "GlobalRaw"
            reference_selected, reference_grads, stats = self._run_windowed(
                optimizer_aware, self.LOGICAL
            )
            self.assertEqual(stats["selection/logical_candidates"], 16.0)
            for chunk_size in (2, 1):
                with self.subTest(method=label, chunk_size=chunk_size):
                    selected, gradients, _ = self._run_windowed(
                        optimizer_aware, chunk_size
                    )
                    self.assertEqual(selected, reference_selected)
                    for expected, observed in zip(reference_grads, gradients):
                        self.assertTrue(
                            torch.allclose(expected, observed, atol=2e-6, rtol=2e-6)
                        )

    def test_windowed_global_matches_unwindowed_global(self):
        for optimizer_aware in (False, True):
            label = "GlobalOptA" if optimizer_aware else "GlobalRaw"
            expected_selected, expected_grads = self._run_unwindowed(optimizer_aware)
            for chunk_size in (16, 2, 1):
                with self.subTest(method=label, chunk_size=chunk_size):
                    selected, gradients, _ = self._run_windowed(
                        optimizer_aware, chunk_size
                    )
                    self.assertEqual(sorted(selected), sorted(expected_selected))
                    for expected, observed in zip(expected_grads, gradients):
                        self.assertTrue(
                            torch.allclose(expected, observed, atol=2e-6, rtol=2e-6)
                        )

    def test_unwindowed_factory_still_returns_the_original_strategies(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 1))
        hook = GradientHook(model, ["0"], device="cpu")
        try:
            self.assertIsInstance(
                create_separate_batch_strategy("GlobalSubset", hook, frac=0.5),
                SeparateBatchGlobalSubsetOnePassStrategy,
            )
            self.assertIsInstance(
                create_separate_batch_strategy(
                    "OptimizerAwareGlobalSubset", hook, frac=0.5
                ),
                SeparateBatchOptimizerAwareGlobalSubsetStrategy,
            )
        finally:
            hook.remove_hooks()


if __name__ == "__main__":
    unittest.main()
