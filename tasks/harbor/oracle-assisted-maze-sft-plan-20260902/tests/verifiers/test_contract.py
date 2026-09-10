import tempfile
import unittest
from pathlib import Path

import torch

from utils.contract import (CandidateError, build_catalog, load_examples,
                            save_selections, validate_selections)
from utils.maze import build_groups, partition_ids
from utils.runner import contrastive_losses


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.catalog, self.groups = build_catalog(build_groups(partition_ids("public_train")[:1]))
        self.positive = next(item for item in self.groups[0]["catalog"]
                             if item["kind"] == "canonical")
        self.negative = next(item for item in self.groups[0]["catalog"]
                             if item["kind"] == "failure_error")

    def test_round_trip_uses_catalog_owned_state_and_action(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.npz"
            save_selections(path, [(self.positive["id"], 0.5, False)], self.catalog)
            arrays = load_examples(path)
        self.assertEqual(arrays[0].shape, (1, 5, 17, 17))
        self.assertEqual(arrays[1][0], self.positive["action"])

    def test_unknown_ids_and_changed_treatment_are_rejected(self):
        invalid = [
            [("g999:canonical:0", 1.0, False)],
            [(self.positive["id"], 1.0, True)],
            [(self.negative["id"], 1.0, False)],
            [(self.positive["id"], float("nan"), False)],
            [(self.positive["id"], 0.0, False)],
            [(self.positive["id"], 1.0, 0)],
        ]
        for selections in invalid:
            with self.subTest(selections=selections), self.assertRaises(CandidateError):
                validate_selections(selections, self.catalog)

    def test_negative_loss_penalizes_selected_probability(self):
        targets = torch.tensor([0])
        positive = torch.tensor([False])
        negative = torch.tensor([True])
        low = torch.tensor([[0.0, 4.0]])
        high = torch.tensor([[4.0, 0.0]])
        self.assertLess(contrastive_losses(high, targets, positive).item(),
                        contrastive_losses(low, targets, positive).item())
        self.assertGreater(contrastive_losses(high, targets, negative).item(),
                           contrastive_losses(low, targets, negative).item())

    def test_reference_light_failure_error_is_valid(self):
        catalog, groups = build_catalog(build_groups(partition_ids("public_train")[:2]))
        selections = []
        for group in groups:
            selections.extend((item["id"], 1.0, False) for item in group["catalog"]
                              if item["kind"] == "canonical")
            error = next((item for item in group["catalog"]
                          if item["kind"] == "failure_error"), None)
            if error is not None:
                selections.append((error["id"], 0.1, True))
        arrays = validate_selections(selections, catalog)
        canonical_count = sum(item["kind"] == "canonical"
                              for group in groups for item in group["catalog"])
        error_count = sum(any(item["kind"] == "failure_error" for item in group["catalog"])
                          for group in groups)
        self.assertEqual(len(arrays[1]), canonical_count + error_count)
        self.assertEqual(int(arrays[3].sum()), error_count)
        self.assertTrue((arrays[2][~arrays[3]] == 1.0).all())
        self.assertTrue((arrays[2][arrays[3]] == 0.1).all())


if __name__ == "__main__":
    unittest.main()
