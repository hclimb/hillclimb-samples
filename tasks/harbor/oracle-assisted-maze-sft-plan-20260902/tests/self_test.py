#!/usr/bin/env python3
"""Fast entrypoint for deterministic evaluator unit tests."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(str(ROOT / "verifiers"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
