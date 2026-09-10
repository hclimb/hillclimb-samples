import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from utils.runner import construct_examples, seed_sets


TASK = Path(__file__).resolve().parents[2]
SOLUTION = TASK / "solution"


@unittest.skipUnless(SOLUTION.is_dir(), "control sources are not in the verifier image")
class FrozenControlTests(unittest.TestCase):
    def test_frozen_sources_and_unchanged_starter_match_recorded_hashes(self):
        manifest = json.loads((SOLUTION / "frozen/manifest.json").read_text())
        for name in ("baseline", "sol", "fable"):
            source = (TASK / "environment/starter/maze_task/candidate.py" if name == "baseline"
                      else SOLUTION / f"frozen/{name}.py")
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                             manifest[name]["sha256"], name)
        for bundle in (TASK / "environment/starter", TASK / "tests/public_tests"):
            self.assertFalse((bundle / "solution").exists())
            self.assertFalse((bundle / "frozen").exists())

    def test_selected_control_is_installed_exactly_and_passes_public_contract(self):
        for name in ("sol", "fable"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                starter = root / "starter"
                (starter / "maze_task").mkdir(parents=True)
                shutil.copytree(SOLUTION, root / "solution")
                script = root / "solve.sh"
                script.write_text((SOLUTION / "solve.sh").read_text().replace(
                    "/environment/starter", str(starter)).replace("/solution", str(root / "solution")))
                result = subprocess.run(["sh", str(script)], capture_output=True, text=True,
                                        env={**os.environ, "WERM_MAZE_FROZEN_CONTROL": name})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((starter / "maze_task/candidate.py").read_bytes(),
                                 (SOLUTION / f"frozen/{name}.py").read_bytes())
                construct_examples(starter, seed_sets("full", 1_000_000)[0][0][:4],
                                   root / "examples.npz")

    def test_unknown_control_fails_before_writing_candidate(self):
        result = subprocess.run(["sh", str(SOLUTION / "solve.sh")], capture_output=True, text=True,
                                env={**os.environ, "WERM_MAZE_FROZEN_CONTROL": "unknown"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown frozen control", result.stderr)
