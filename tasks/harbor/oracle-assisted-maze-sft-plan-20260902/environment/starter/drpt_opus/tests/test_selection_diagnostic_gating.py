from __future__ import annotations

import unittest
import tempfile

import torch

from drpt.selection.state import LayerWiseSubsetState
from SFT.train.training_arguments import TrainingArguments


def _make_state(**overrides) -> LayerWiseSubsetState:
    kwargs = {
        "train_batch_size": 4,
        "num_layers": 1,
        "frac": 0.5,
        "lr": 1.0,
        "device": "cpu",
        "dtype": torch.float32,
        "use_second_order": False,
        "selection_mode": "topk",
        "record_selections": True,
    }
    kwargs.update(overrides)
    return LayerWiseSubsetState(**kwargs)


class SelectionDiagnosticGatingTests(unittest.TestCase):
    def test_c16_probe_still_uses_non_reentrant_checkpointing(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=16,
                logical_candidate_batch_size=16,
                candidate_microbatch_size=16,
                gradient_checkpointing=True,
            )
        self.assertEqual(
            args.gradient_checkpointing_kwargs, {"use_reentrant": False}
        )

    def test_optional_raw_diagnostics_do_not_gate_selection_records(self) -> None:
        state = _make_state(optimizer_aware_diagnostic_interval=0)
        state.add_raw_opt_score_diagnostics(
            "layer_0",
            torch.tensor([4.0, 3.0, 2.0, 1.0]),
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )
        self.assertEqual(state.get_diagnostic_metrics(), {})

        tokens = torch.ones(4)
        state.set_token_counts(tokens, tokens.sum(), tokens.sum())
        state.process_layer_gradients(
            torch.eye(4), torch.tensor([4.0, 3.0, 2.0, 1.0]), layer_idx=0
        )
        self.assertEqual(len(state._selection_records), 1)
        self.assertEqual(state._selection_records[0]["selected_indices"], [0, 1])

    def test_interval_and_global_step_control_same_state_diagnostics(self) -> None:
        state = _make_state(
            optimizer_aware_diagnostic_interval=5, global_step=10
        )
        self.assertTrue(state.should_collect_optimizer_diagnostics())
        state.configure_optimizer_diagnostics(interval=5, global_step=11)
        self.assertFalse(state.should_collect_optimizer_diagnostics())
        with self.assertRaises(ValueError):
            state.configure_optimizer_diagnostics(interval=-1)

    def test_diagnostic_harvest_filters_nonfinite_values_in_one_payload(self) -> None:
        state = _make_state()
        state._append_diagnostic("finite", torch.tensor(1.0))
        state._append_diagnostic("finite", torch.tensor(float("nan")))
        state._append_diagnostic("finite", torch.tensor(3.0))
        state._append_diagnostic("nonfinite", torch.tensor(float("inf")))
        self.assertEqual(state.get_diagnostic_metrics(), {"finite": 2.0})

    def test_window_completeness_bitmap_stays_on_cpu(self) -> None:
        state = _make_state()
        state.enable_windowed_execution()
        state.set_window_chunk("score", 0, 2)
        state.store_window_scores(0, torch.tensor([1.0, 2.0]))
        self.assertEqual(state._window_score_filled[0].device.type, "cpu")
        with self.assertRaises(RuntimeError):
            state.require_complete_window(0)
        state.set_window_chunk("score", 2, 4)
        state.store_window_scores(0, torch.tensor([3.0, 4.0]))
        state.require_complete_window(0)


if __name__ == "__main__":
    unittest.main()
