from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from drpt.hook import GradientHook
from drpt.optimizer import HybridMuonAdamW, MuonWithAuxAdamW
from drpt.selection.advanced_solvers import muon_live_candidate_transform
from SFT.train.trainer import LayerWiseSubsetTrainer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(13, 4)
        self.hidden = nn.Linear(4, 6)
        self.norm = nn.LayerNorm(6)
        self.lm_head = nn.Linear(6, 13, bias=False)


def make_optimizer(model, **overrides):
    kwargs = {
        "lr": 1e-4,
        "muon_lr": 2e-3,
        "aux_adamw_lr": 3e-4,
        "weight_decay": 0.0,
        "muon_momentum": 0.8,
        "muon_nesterov": True,
        "muon_ns_steps": 3,
        "muon_eps": 1e-7,
        "muon_backend": "auto",
    }
    kwargs.update(overrides)
    return MuonWithAuxAdamW(model.named_parameters(), model, **kwargs)


class MuonRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_compatibility_alias_and_general_hidden_weight_grouping(self):
        self.assertIs(HybridMuonAdamW, MuonWithAuxAdamW)
        model = TinyModel()
        optimizer = make_optimizer(model)

        expected = {
            "embed_tokens.weight": "adamw",
            "hidden.weight": "muon",
            "hidden.bias": "adamw",
            "norm.weight": "adamw",
            "norm.bias": "adamw",
            "lm_head.weight": "adamw",
        }
        for name, parameter in model.named_parameters():
            self.assertEqual(
                optimizer.get_param_optimizer_kind(parameter), expected[name]
            )
            group = optimizer.get_param_group(parameter)
            self.assertIsNotNone(group)
            expected_lr = 2e-3 if expected[name] == "muon" else 3e-4
            self.assertEqual(group["lr"], expected_lr)

        metadata = optimizer.get_runtime_metadata()
        self.assertEqual(metadata["muon_learning_rate_resolved"], 2e-3)
        self.assertEqual(
            metadata["aux_adamw_learning_rate_resolved"], 3e-4
        )

    @unittest.skipUnless(
        getattr(torch.optim, "Muon", None) is not None,
        "official torch.optim.Muon is unavailable",
    )
    def test_official_muon_is_preferred_and_is_the_actual_inner_optimizer(self):
        optimizer = make_optimizer(TinyModel(), muon_backend="auto")
        self.assertEqual(optimizer.requested_muon_backend, "auto")
        self.assertEqual(optimizer.resolved_muon_backend, "torch")
        self.assertIsNone(optimizer.muon_backend_fallback_reason)
        self.assertIsInstance(
            optimizer._inner_optimizers["muon"], torch.optim.Muon
        )
        self.assertIsInstance(
            optimizer._inner_optimizers["adamw"], torch.optim.AdamW
        )

    @unittest.skipUnless(
        getattr(torch.optim, "Muon", None) is not None,
        "official torch.optim.Muon is unavailable",
    )
    def test_legacy_local_request_still_prioritizes_official_muon(self):
        optimizer = make_optimizer(TinyModel(), muon_backend="local")
        self.assertEqual(optimizer.requested_muon_backend, "local")
        self.assertEqual(optimizer.resolved_muon_backend, "torch")
        self.assertIsInstance(
            optimizer._inner_optimizers["muon"], torch.optim.Muon
        )

    def test_unavailable_official_muon_falls_back_even_when_torch_requested(self):
        model = TinyModel()
        with patch.object(torch.optim, "Muon", None):
            optimizer = make_optimizer(model, muon_backend="torch")

        self.assertEqual(optimizer.requested_muon_backend, "torch")
        self.assertEqual(optimizer.resolved_muon_backend, "local")
        self.assertIn(
            "not available", optimizer.muon_backend_fallback_reason
        )
        self.assertNotIn("muon", optimizer._inner_optimizers)
        for parameter in model.parameters():
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()

    def test_official_incompatible_shape_lr_setting_falls_back(self):
        optimizer = make_optimizer(
            TinyModel(),
            muon_backend="torch",
            muon_lr_shape_scale=False,
        )
        self.assertEqual(optimizer.resolved_muon_backend, "local")
        self.assertIn(
            "does not support disabling",
            optimizer.muon_backend_fallback_reason,
        )

    def test_zero_muon_parameter_run_is_rejected(self):
        class NoHiddenMatrix(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(7, 3)
                self.lm_head = nn.Linear(3, 7, bias=False)

        with self.assertRaisesRegex(ValueError, "must not silently become AdamW"):
            make_optimizer(NoHiddenMatrix())

    @unittest.skipUnless(
        getattr(torch.optim, "Muon", None) is not None,
        "official torch.optim.Muon is unavailable",
    )
    def test_local_fallback_update_conforms_to_official_muon(self):
        official_model = TinyModel()
        local_model = TinyModel()
        local_model.load_state_dict(official_model.state_dict())
        official = make_optimizer(official_model, muon_backend="torch")
        local = make_optimizer(
            local_model,
            muon_backend="local",
            _force_local_for_test=True,
        )

        for (_, official_param), (_, local_param) in zip(
            official_model.named_parameters(), local_model.named_parameters()
        ):
            gradient = torch.randn_like(official_param)
            official_param.grad = gradient.clone()
            local_param.grad = gradient.clone()

        official.step()
        local.step()

        for (_, official_param), (_, local_param) in zip(
            official_model.named_parameters(), local_model.named_parameters()
        ):
            self.assertTrue(
                torch.equal(official_param, local_param),
                msg=official.get_param_name(official_param),
            )
        official_state = official.get_param_state(
            official_model.hidden.weight
        )
        local_state = local.get_param_state(local_model.hidden.weight)
        self.assertTrue(
            torch.equal(
                official_state["momentum_buffer"],
                local_state["momentum_buffer"],
            )
        )

    @unittest.skipUnless(
        getattr(torch.optim, "Muon", None) is not None,
        "official torch.optim.Muon is unavailable",
    )
    def test_ranking_hook_uses_exact_runtime_group_and_state(self):
        model = TinyModel()
        model.hidden.weight.data.zero_()
        optimizer = make_optimizer(model, muon_backend="torch")
        hook = GradientHook(model, ["hidden"], device="cpu")
        try:
            hook.set_optimizer(optimizer)
            weight = model.hidden.weight
            wrapper_group = optimizer.get_param_group(weight)
            self.assertIs(
                hook._optimizer_group_by_param_id[id(weight)], wrapper_group
            )
            self.assertEqual(
                hook.get_optimizer_group_keys(), ["muon:linear_other"]
            )

            previous = torch.randn_like(weight)
            gradient = torch.randn_like(weight)
            inner = optimizer._inner_optimizers["muon"]
            inner.state[weight]["momentum_buffer"] = previous.clone()
            expected_update = muon_live_candidate_transform(
                gradient,
                momentum_buffer=previous,
                optimizer_dtype=weight.dtype,
                momentum=wrapper_group["momentum"],
                nesterov=wrapper_group["nesterov"],
                ns_steps=wrapper_group["muon_ns_steps"],
                eps=wrapper_group["muon_eps"],
                lr=wrapper_group["lr"],
                shape_lr_scale=wrapper_group["muon_lr_shape_scale"],
                adjust_lr_fn=wrapper_group["adjust_lr_fn"],
            )
            weight.grad = gradient
            optimizer.step()

            self.assertIs(
                optimizer.get_param_state(weight), inner.state[weight]
            )
            actual_update = -weight.detach()
            cosine = torch.nn.functional.cosine_similarity(
                actual_update.flatten(),
                expected_update.float().flatten(),
                dim=0,
            )
            relative_error = (
                (actual_update - expected_update.float()).norm()
                / actual_update.norm()
            )
            # Ranking intentionally uses a smooth float32 NS surrogate while
            # official Muon rounds NS in bf16; grouping, LR, momentum state,
            # and update direction still come from the exact live optimizer.
            self.assertGreater(float(cosine), 0.999)
            self.assertLess(float(relative_error), 0.02)
        finally:
            hook.remove_hooks()


class MuonMetadataPersistenceTests(unittest.TestCase):
    def test_muon_with_update_compression_is_rejected_before_optimizer_creation(self):
        trainer = object.__new__(LayerWiseSubsetTrainer)
        trainer.model = TinyModel()
        trainer.optimizer = None
        trainer.grad_hook = None
        trainer.has_compression = True
        trainer.args = SimpleNamespace(optimizer_type="muon")

        with self.assertRaisesRegex(
            ValueError, "Refusing to silently replace.*Muon"
        ):
            trainer.create_optimizer()

    def test_trainer_wires_separate_lrs_and_reuses_resolved_optimizer(self):
        class RecordingHook:
            def __init__(self):
                self.optimizer = None
                self.config = {}

            def set_optimizer(self, optimizer):
                self.optimizer = optimizer

            def configure_optimizer_aware(self, **kwargs):
                self.config.update(kwargs)

        model = TinyModel()
        trainer = object.__new__(LayerWiseSubsetTrainer)
        trainer.model = model
        trainer.optimizer = None
        trainer.grad_hook = RecordingHook()
        trainer.has_compression = False
        trainer.args = SimpleNamespace(
            optimizer_type="muon",
            learning_rate=1e-4,
            muon_learning_rate=2e-3,
            aux_adamw_learning_rate=3e-4,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            weight_decay=0.0,
            optimizer_aware_muon_momentum=0.8,
            optimizer_aware_muon_nesterov=True,
            optimizer_aware_muon_steps=3,
            optimizer_aware_muon_eps=1e-7,
            optimizer_aware_muon_lr_shape_scale=True,
            optimizer_aware_muon_adjust_lr_fn="original",
            optimizer_aware_muon_backend="auto",
            optimizer_aware_lora_optimizer="adamw",
        )

        first = trainer.create_optimizer()
        second = trainer.create_optimizer()
        self.assertIs(first, second)
        self.assertIs(trainer.grad_hook.optimizer, first)
        self.assertEqual(
            trainer.grad_hook.config["muon_backend"],
            first.resolved_muon_backend,
        )
        self.assertEqual(
            first.get_runtime_metadata()["muon_learning_rate_resolved"],
            2e-3,
        )
        self.assertEqual(
            first.get_runtime_metadata()[
                "aux_adamw_learning_rate_resolved"
            ],
            3e-4,
        )

    def test_resolved_backend_and_lrs_are_written_after_creation(self):
        from SFT.train.train import _write_run_metadata

        model = TinyModel()
        optimizer = make_optimizer(model)
        with tempfile.TemporaryDirectory() as temp_dir:
            training_args = SimpleNamespace(
                local_rank=-1,
                output_dir=temp_dir,
                method="FullTraining",
                optimizer_type="muon",
                learning_rate=1e-4,
                muon_learning_rate=2e-3,
                aux_adamw_learning_rate=3e-4,
                optimizer_aware_muon_backend="auto",
                selection_frac=1.0,
                selection_mode="topk",
                scoring_method="reduced_ghost",
                subset_mode="one_pass",
                val_strategy="separate_batch",
                seed=42,
                train_dataset_names=["toy"],
                analysis_dataset="toy",
                soft_weighting_steps=20,
                soft_weighting_lr=0.1,
                soft_weighting_tol=1e-5,
                soft_weighting_patience=3,
                soft_weighting_gamma=0.0,
                soft_weighting_use_optimizer_state=True,
                soft_weighting_constraint="capped_simplex",
                soft_replay_precision="fp32",
                muon_surrogate_alpha=1.0,
                muon_surrogate_rank=8,
                muon_surrogate_full_svd_max_dim=64,
                muon_surrogate_rtol=1e-6,
                muon_surrogate_oversample=4,
                muon_surrogate_power_iters=1,
                muon_surrogate_include_adamw_scores=False,
                muon_surrogate_mode_weighting="uniform",
                muon_surrogate_saturation=False,
                optimizer_aware_token_normalized_selection=False,
            )
            model_args = SimpleNamespace(model_name_or_path="toy/model")
            data_args = SimpleNamespace(max_seq_length=128)
            _write_run_metadata(
                training_args, model_args, data_args, optimizer=optimizer
            )
            payload = json.loads(
                (Path(temp_dir) / "run_metadata.json").read_text()
            )

        self.assertEqual(payload["muon_backend_requested"], "auto")
        self.assertIn(payload["muon_backend_resolved"], {"torch", "local"})
        self.assertEqual(payload["muon_learning_rate_resolved"], 2e-3)
        self.assertEqual(
            payload["aux_adamw_learning_rate_resolved"], 3e-4
        )
        self.assertIn("muon_optimizer_implementation", payload)


if __name__ == "__main__":
    unittest.main()
