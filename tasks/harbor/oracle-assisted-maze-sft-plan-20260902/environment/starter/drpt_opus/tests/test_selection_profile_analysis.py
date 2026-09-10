from __future__ import annotations

import unittest

from SFT.eval.analysis.selection_profile import (
    _selection_methods_for_optimizer,
    _weighting_contract_for_optimizer,
)


class SelectionProfileContractTests(unittest.TestCase):
    def test_registry_methods_are_all_represented(self) -> None:
        self.assertEqual(
            _selection_methods_for_optimizer("adamw"),
            (
                "GlobalRaw",
                "GlobalOptA",
                "LayerwiseRaw",
                "LayerwiseSoft",
                "LayerwiseSoftP",
                "LayerwiseOptA",
            ),
        )
        self.assertEqual(
            _selection_methods_for_optimizer("muon"),
            (
                "LayerwiseRaw",
                "LayerwiseSoft",
                "LayerwiseSoftP",
                "LayerwiseMuonSur",
                "LayerwiseMuonPSur",
                "LayerwiseMuonSatSur",
                "LayerwiseMuonSatPSur",
            ),
        )

    def test_soft_contract_uses_the_optimizer_family_objective(self) -> None:
        adamw = {
            row["method"]: row for row in _weighting_contract_for_optimizer("adamw")
        }
        muon = {
            row["method"]: row for row in _weighting_contract_for_optimizer("muon")
        }

        self.assertIn("<P g_i, g_target>", adamw["LayerwiseSoft"]["score"])
        self.assertIn("MuonMap", muon["LayerwiseSoft"]["score"])
        self.assertNotIn("GlobalRaw", muon)
        self.assertIn("LayerwiseMuonSatPSur", muon)


if __name__ == "__main__":
    unittest.main()
