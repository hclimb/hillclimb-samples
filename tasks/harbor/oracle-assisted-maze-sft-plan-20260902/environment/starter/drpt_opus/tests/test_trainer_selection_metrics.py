from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from SFT.train.trainer import LayerWiseSubsetTrainer, _compact_selection_metrics


class CompactSelectionMetricTests(unittest.TestCase):
    def test_compacts_layer_diagnostics_and_ignores_nonfinite_values(self) -> None:
        metrics = {
            "selection/n_selected": torch.tensor(2),
            "soft/layer_0/entropy": 0.5,
            "soft/layer_1/entropy": torch.tensor(1.5),
            "soft/layer_0/ess": 1.0,
            "soft/layer_1/ess": 3.0,
            "soft/layer_0/converged": False,
            "soft/layer_1/converged": True,
            "soft/global/objective_improvement": 4.0,
            "spectral/layer_0/active_rank": 2,
            "spectral/layer_1/active_rank": 6,
            "update/muon_attention/selected_alignment": 7.0,
            "diag/adamw_embedding/raw_opt_topk_overlap": 0.75,
            "soft/layer_2/entropy": float("inf"),
            "unrelated/value": 100.0,
        }

        compact = _compact_selection_metrics(metrics)

        self.assertEqual(compact["selection/n_selected"], 2.0)
        self.assertEqual(compact["soft/entropy"], 1.0)
        self.assertEqual(compact["soft/ess"], 2.0)
        self.assertEqual(compact["soft/converged"], 0.5)
        self.assertEqual(compact["soft/objective_improvement"], 4.0)
        self.assertEqual(compact["spectral/active_rank"], 4.0)
        self.assertEqual(compact["update/muon_attention/selected_alignment"], 7.0)
        self.assertEqual(
            compact["diag/adamw_embedding/raw_opt_topk_overlap"], 0.75
        )
        self.assertNotIn("unrelated/value", compact)
        self.assertTrue(all(math.isfinite(value) for value in compact.values()))

    def test_trainer_logs_compact_payload_and_keeps_detailed_diagnostics(self) -> None:
        detailed = {
            "soft/layer_0/entropy": 0.25,
            "soft/layer_1/entropy": 0.75,
            "soft/layer_0/ess": 1.0,
            "soft/layer_1/ess": 2.0,
        }
        log = Mock()
        owner = SimpleNamespace(
            args=SimpleNamespace(logging_steps=2),
            state=SimpleNamespace(global_step=4),
            selection_strategy=SimpleNamespace(last_diagnostic_metrics=detailed),
            selection_diagnostics_history=[],
            log=log,
        )

        LayerWiseSubsetTrainer._maybe_log_selection_metrics(
            owner, {"selection/n_selected": 2}
        )

        expected = {
            "selection/n_selected": 2.0,
            "soft/entropy": 0.5,
            "soft/ess": 1.5,
        }
        log.assert_called_once_with(expected)
        self.assertEqual(
            owner.selection_diagnostics_history,
            [{"step": 4, **expected}],
        )
        self.assertEqual(owner.selection_strategy.last_diagnostic_metrics, detailed)

    def test_selection_record_stride_is_independent_from_domain_stride(self) -> None:
        owner = SimpleNamespace(
            _record_selections=True,
            _record_selections_freq=100,
            state=SimpleNamespace(global_step=99),
        )
        self.assertFalse(LayerWiseSubsetTrainer._should_record_selections(owner))
        owner.state.global_step = 100
        self.assertTrue(LayerWiseSubsetTrainer._should_record_selections(owner))

    def test_capture_preserves_stable_candidate_pool_indices(self) -> None:
        tokenizer = Mock()
        tokenizer.batch_decode.side_effect = [["train-a", "train-b"], ["target"]]
        owner = SimpleNamespace(
            selection_strategy=SimpleNamespace(
                last_selection_record=[
                    {"layer_idx": 0, "selected_indices": [1], "scores": [0.1, 0.9]}
                ]
            ),
            processing_class=tokenizer,
            state=SimpleNamespace(global_step=100),
            args=SimpleNamespace(method="LayerWiseSubset"),
            _last_meta_idx=[7, 3],
            _selection_records=[],
        )
        train = {"input_ids": torch.tensor([[1], [2]])}
        target = {"input_ids": torch.tensor([[3]])}

        LayerWiseSubsetTrainer._capture_selection_record(owner, train, target)

        self.assertEqual(owner._selection_records[0]["train_meta_idx"], [7, 3])
        self.assertIn("layers", owner._selection_records[0])


if __name__ == "__main__":
    unittest.main()
