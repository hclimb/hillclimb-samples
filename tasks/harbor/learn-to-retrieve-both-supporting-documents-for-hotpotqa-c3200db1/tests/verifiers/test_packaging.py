"""The starter must contain the same required public assets as the verifier."""
from pathlib import Path
import unittest


class PackagingTests(unittest.TestCase):
    def test_starter_public_configuration_matches_protected_package(self):
        root = Path(__file__).resolve().parents[1]
        protected = root / 'public_tests'
        starter = root.parent / 'environment/starter/optifine_public_tests'
        for relative in ('assets/models/bert-tiny/config.json',
                         'assets/models/bert-tiny/asset_manifest.json',
                         'assets/hotpotqa/train/manifest.json',
                         'public_panel/manifest.json', 'utils/oracle.py'):
            with self.subTest(relative=relative):
                self.assertEqual((starter / relative).read_bytes(), (protected / relative).read_bytes())


if __name__ == '__main__':
    unittest.main()
