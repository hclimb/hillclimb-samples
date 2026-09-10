"""Regression tests for Dolci32k generated-artifact path separation."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from SFT.data.prepare_dolci32k import main as prepare_main
from SFT.data.dolci32k.artifacts import ARTIFACT_ROOT_NAME, dolci32k_root


REPO_ROOT = Path(__file__).resolve().parents[1]


class Dolci32KArtifactPathTests(unittest.TestCase):
    def test_default_data_root_does_not_overlap_source_package(self):
        data_root = REPO_ROOT / "SFT" / "data"
        source_package = data_root / "dolci32k"
        artifact_root = dolci32k_root(data_root)

        self.assertEqual(artifact_root, data_root / ARTIFACT_ROOT_NAME)
        self.assertNotEqual(artifact_root, source_package)
        self.assertNotIn(source_package, artifact_root.parents)

    def test_prepare_cli_prints_the_same_canonical_root(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(
            io.StringIO()
        ) as output:
            exit_code = prepare_main(
                ["--data-dir", tmp, "--print-artifact-root"]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue().strip(), str(dolci32k_root(tmp)))


if __name__ == "__main__":
    unittest.main()
