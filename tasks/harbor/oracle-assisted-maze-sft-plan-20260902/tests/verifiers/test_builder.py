from pathlib import Path
import tempfile
import unittest

import numpy as np

from utils.contract import CandidateError, build_catalog, load_examples, read_selections
from utils.runner import canonical_arrays, construct_examples, seed_sets, training_groups

TESTS = Path(__file__).resolve().parents[1]
CANONICAL = "[(x['id'], 1.0, False) for x in groups[0]['catalog'] if x['kind'] == 'canonical']"


class OutputTests(unittest.TestCase):
    def test_symlinks_and_fifos_are_rejected_without_following_or_waiting(self):
        import os

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "regular").write_text("[]")
            (root / "link").symlink_to(root / "regular")
            os.mkfifo(root / "pipe")
            for name in ("link", "pipe"):
                with self.subTest(name=name), self.assertRaises(CandidateError):
                    read_selections(root / name)


class BuilderTests(unittest.TestCase):
    def construct(self, source):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "maze_task").mkdir()
            (root / "maze_task/candidate.py").write_text(source)
            artifact = root / "examples.npz"
            ids = seed_sets("quick", 1_000_000)[0][0][:1]
            construct_examples(root, ids, artifact)
            return load_examples(artifact)

    def test_compliant_selection_round_trip_without_nested_sandbox(self):
        arrays = self.construct(f"def build_examples(groups):\n return {CANONICAL}\n")
        self.assertGreater(len(arrays[1]), 0)

    def test_full_private_panel_matches_trusted_baseline_arrays(self):
        ids = seed_sets("full", 2_000_000, private=True)[0][0]
        starter = TESTS.parent / "environment/starter"
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "examples.npz"
            construct_examples(starter, ids, artifact)
            actual = load_examples(artifact)
        expected = canonical_arrays(training_groups(ids))
        for left, right in zip(actual, expected):
            np.testing.assert_array_equal(left, right)

    def test_parent_uses_catalog_not_edited_input_records(self):
        source = ("def build_examples(groups):\n"
                  " for x in groups[0]['catalog']:\n"
                  "  x['state'].zero_()\n"
                  "  x['action'] = (x['action'] + 1) % 4\n"
                  f" return {CANONICAL}\n")
        states, actions, _, _ = self.construct(source)
        catalog, groups = build_catalog(training_groups(seed_sets("quick", 1_000_000)[0][0][:1]))
        records = [x for x in groups[0]["catalog"] if x["kind"] == "canonical"]
        np.testing.assert_array_equal(actions, [catalog[x["id"]][1] for x in records])
        np.testing.assert_array_equal(states, np.stack([catalog[x["id"]][0] for x in records]))

    def test_forged_catalog_id_and_negative_treatment_are_rejected(self):
        for expression in ("[('not-a-catalog-id', 1.0, False)]",
                           "[(groups[0]['catalog'][0]['id'], 1.0, True)]"):
            with self.subTest(expression=expression), self.assertRaises(CandidateError):
                self.construct(f"def build_examples(groups):\n return {expression}\n")

    def test_output_symlink_is_rejected(self):
        source = ("def build_examples(groups):\n"
                  " import os, sys\n from pathlib import Path\n"
                  " output = Path(sys.argv[2]) / 'selections.json'\n"
                  " output.symlink_to(Path(sys.argv[1]) / 'candidate.py')\n"
                  " os._exit(0)\n")
        with self.assertRaises(CandidateError):
            self.construct(source)


if __name__ == "__main__":
    unittest.main()
