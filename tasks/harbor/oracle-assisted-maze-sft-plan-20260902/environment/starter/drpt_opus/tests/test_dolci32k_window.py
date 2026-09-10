"""Lightweight invariants for Dolci32K traversal and model overlays."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from torch.utils.data import BatchSampler

from SFT.data.dolci32k.common import ordered_id_sha256
from SFT.data.dolci32k.profile import (
    ADAMW_METHODS,
    MAX_SEQ_LEN,
    MODEL_PROFILES,
    MUON_METHODS,
    SETTING_ORDER,
)
from SFT.train.train import _load_manifest_order
from SFT.train.trainer import LayerWiseSubsetTrainer, ManifestOrderSampler
from SFT.train.training_arguments import TrainingArguments


class CandidateTraversalTests(unittest.TestCase):
    @staticmethod
    def _order():
        values = list(range(32_000))
        random.Random(314159).shuffle(values)
        return values

    def test_one_complete_pass_is_2000_unique_n16_batches(self):
        order = self._order()
        sampler = ManifestOrderSampler(order)
        batches = list(BatchSampler(sampler, batch_size=16, drop_last=True))
        self.assertEqual(len(batches), 2_000)
        flattened = [index for batch in batches for index in batch]
        self.assertEqual(flattened, order)
        self.assertEqual(len(flattened), 32_000)
        self.assertEqual(len(set(flattened)), 32_000)

    def test_stored_order_is_independent_of_global_rng_consumption(self):
        order = self._order()
        first = list(ManifestOrderSampler(order))
        random.seed(999)
        for _ in range(10_000):
            random.random()
        torch.manual_seed(999)
        torch.rand(10_000)
        second = list(ManifestOrderSampler(order))
        self.assertEqual(first, second)

    def test_duplicate_order_indices_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            ManifestOrderSampler([0, 1, 1])

    def test_trainer_sampler_hook_uses_the_manifest_order(self):
        trainer = object.__new__(LayerWiseSubsetTrainer)
        trainer.train_dataset = list(range(8))
        trainer._manifest_train_order = (6, 1, 4, 0, 7, 3, 5, 2)
        sampler = trainer._get_train_sampler()
        self.assertIsInstance(sampler, ManifestOrderSampler)
        self.assertEqual(list(sampler), list(trainer._manifest_train_order))

    def test_stable_ids_map_order_to_raw_rows(self):
        raw_ids = ["a", "b", "c", "d"]
        ordered_ids = ["c", "a", "d", "b"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "order.jsonl"
            path.write_text(
                "".join(
                    json.dumps({"position": position, "id": example_id}) + "\n"
                    for position, example_id in enumerate(ordered_ids)
                ),
                encoding="utf-8",
            )
            indices, digest = _load_manifest_order(
                path,
                [{"id": example_id} for example_id in raw_ids],
                {"ordered_id_sha256": ordered_id_sha256(ordered_ids)},
            )
        self.assertEqual(indices, [2, 0, 3, 1])
        self.assertEqual(digest, ordered_id_sha256(ordered_ids))

    def test_all_model_profiles_resolve_identical_raw_ordered_ids(self):
        raw_ids = ["dolci::source::1", "dolci::source::2", "dolci::source::3"]
        ordered_ids = [raw_ids[2], raw_ids[0], raw_ids[1]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "order.jsonl"
            path.write_text(
                "".join(
                    json.dumps({"position": position, "id": example_id}) + "\n"
                    for position, example_id in enumerate(ordered_ids)
                ),
                encoding="utf-8",
            )
            resolved = {
                alias: _load_manifest_order(
                    path,
                    [{"id": example_id} for example_id in raw_ids],
                    {"ordered_id_sha256": ordered_id_sha256(ordered_ids)},
                )
                for alias in MODEL_PROFILES
            }
        self.assertEqual(set(resolved), {"olmo3_7b", "qwen3_1_7b", "qwen3_4b", "qwen3_8b"})
        self.assertEqual(
            {tuple(indices) for indices, _ in resolved.values()}, {(2, 0, 1)}
        )
        self.assertEqual(
            {digest for _, digest in resolved.values()},
            {ordered_id_sha256(ordered_ids)},
        )


class ModelProfileTests(unittest.TestCase):
    def test_canonical_settings_use_profile_max_seq_length(self):
        self.assertEqual(MAX_SEQ_LEN, 4_096)
        setting_root = (
            Path(__file__).resolve().parents[1]
            / "SFT" / "train" / "configs" / "dolci32k"
        )
        for setting in SETTING_ORDER:
            defaults = yaml.safe_load(
                (setting_root / setting / "defaults.yaml").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(defaults["max_seq_length"], MAX_SEQ_LEN)
            self.assertEqual(defaults["model_profile"], "qwen3_1_7b")
            self.assertEqual(
                defaults["method_config_dir"], "configs/dolci32k_methods"
            )

    def test_registry_has_four_exact_models_and_five_settings(self):
        self.assertEqual(
            tuple(MODEL_PROFILES),
            ("olmo3_7b", "qwen3_1_7b", "qwen3_4b", "qwen3_8b"),
        )
        self.assertEqual(
            SETTING_ORDER,
            ("inst_if", "reason_math", "reason_code", "mixed_if", "mixed_math"),
        )
        # The inherited method registries preserve FullTraining, hard top-k,
        # and continuous Soft/SoftP semantics covered by the Dolci window test.
        self.assertEqual(len(ADAMW_METHODS), 5)
        self.assertEqual(len(MUON_METHODS), 8)
        for methods in (ADAMW_METHODS, MUON_METHODS):
            self.assertEqual(methods[:4], (
                "FullTraining", "LayerwiseRaw", "LayerwiseSoft", "LayerwiseSoftP"
            ))

    def test_yaml_overlays_match_canonical_registry(self):
        model_dir = (
            Path(__file__).resolve().parents[1]
            / "SFT" / "train" / "configs" / "dolci32k" / "models"
        )
        overlays = {
            path.stem: yaml.safe_load(path.read_text(encoding="utf-8"))
            for path in sorted(model_dir.glob("*.yaml"))
        }
        self.assertEqual(set(overlays), set(MODEL_PROFILES))
        field_map = {
            "model": "model_name_or_path",
            "model_revision": "model_revision",
            "tokenizer": "tokenizer_name",
            "tokenizer_revision": "tokenizer_revision",
        }
        for alias, profile in MODEL_PROFILES.items():
            self.assertEqual(overlays[alias]["model_profile"], alias)
            for yaml_field, profile_field in field_map.items():
                self.assertEqual(
                    overlays[alias][yaml_field], profile[profile_field]
                )

    def test_shared_soft_methods_enable_mixed_precision_replay(self):
        method_dir = (
            Path(__file__).resolve().parents[1]
            / "SFT" / "train" / "configs" / "dolci32k_methods"
        )
        for filename in (
            "LayerWiseSoftWeighting-Full.yaml",
            "LayerWiseSoftProbability-Full.yaml",
        ):
            config = yaml.safe_load(
                (method_dir / filename).read_text(encoding="utf-8")
            )
            self.assertEqual(
                config["soft_weighting"]["replay_precision"],
                "bf16_fp32",
                filename,
            )


class WindowTrainingArgumentsTests(unittest.TestCase):
    def test_window_arguments_resolve_non_reentrant_checkpointing(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                per_device_train_batch_size=16,
                logical_candidate_batch_size=16,
                candidate_microbatch_size=1,
                gradient_checkpointing=True,
            )
            self.assertEqual(
                args.gradient_checkpointing_kwargs,
                {"use_reentrant": False},
            )

    def test_window_arguments_reject_hf_gradient_accumulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                ValueError, "gradient_accumulation_steps=1"
            ):
                TrainingArguments(
                    output_dir=tmp,
                    per_device_train_batch_size=16,
                    logical_candidate_batch_size=16,
                    candidate_microbatch_size=1,
                    gradient_accumulation_steps=2,
                )

    def test_soft_replay_precision_defaults_normalizes_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            default_args = TrainingArguments(output_dir=tmp)
            self.assertEqual(default_args.soft_replay_precision, "fp32")
            mixed_args = TrainingArguments(
                output_dir=tmp,
                soft_replay_precision="BF16-FP32",
            )
            self.assertEqual(mixed_args.soft_replay_precision, "bf16_fp32")
            with self.assertRaisesRegex(ValueError, "soft_replay_precision"):
                TrainingArguments(
                    output_dir=tmp,
                    soft_replay_precision="fp16",
                )


if __name__ == "__main__":
    unittest.main()
