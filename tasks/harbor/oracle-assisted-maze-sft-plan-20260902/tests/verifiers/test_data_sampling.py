import gzip
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import unittest

from utils.maze import _rows, fixture_manifest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SPEC = spec_from_file_location("maze_packer", FIXTURES / "pack_real_data.py")
packer = module_from_spec(SPEC)
SPEC.loader.exec_module(packer)


class SamplingTests(unittest.TestCase):
    def line(self, prompt_id):
        grid = [[1] * 17 for _ in range(17)]
        for bit in range(16):
            grid[1 + bit // 8][1 + bit % 8] = (prompt_id >> bit) & 1
        return (json.dumps({"prompt_id": prompt_id, "grid": grid, "L_star": 28,
                            "ub": 60, "n_samples": 0, "samples": []}) + "\n").encode()

    def test_reservoir_is_reproducible_and_uses_late_source_rows(self):
        lines = [self.line(i) for i in range(1000)]
        empty = {"prompt_ids": [], "grid_sha256": []}
        first, stats = packer.sample_rows(iter(lines), 32, empty)
        second, again = packer.sample_rows(iter(lines), 32, empty)
        self.assertEqual((first, stats), (second, again))
        self.assertGreater(max(line for line, _ in first), 900)
        self.assertEqual(stats["source_rows_scanned"], 1000)
        self.assertEqual(stats["source_bytes"], sum(map(len, lines)))
        self.assertEqual(stats["source_sha256"], hashlib.sha256(b"".join(lines)).hexdigest())
        different, _ = packer.sample_rows(iter(lines), 32, empty, seed=7)
        self.assertNotEqual(first, different)

    def test_sampler_excludes_old_ids_grids_and_duplicate_grids(self):
        lines = [self.line(i) for i in range(10)]
        duplicate = json.loads(lines[3])
        duplicate["prompt_id"] = 100
        lines.append(json.dumps(duplicate).encode() + b"\n")
        exclusions = {"prompt_ids": [0], "grid_sha256": [packer.grid_digest(json.loads(lines[1]))]}
        selected, stats = packer.sample_rows(lines, 8, exclusions)
        self.assertEqual({row["prompt_id"] for _, row in selected}, set(range(2, 10)))
        self.assertEqual(stats["duplicate_rows"], 1)
        self.assertEqual(stats["excluded_or_invalid_rows"], 2)

    def test_short_source_is_an_error_not_a_smaller_dataset(self):
        with self.assertRaisesRegex(RuntimeError, "need 2"):
            packer.sample_rows([self.line(0)], 2, {"prompt_ids": [], "grid_sha256": []})


class ExpandedFixtureTests(unittest.TestCase):
    def test_actual_new_splits_exclude_legacy_ids_and_grids(self):
        legacy = json.loads((FIXTURES / "legacy_exclusions.json").read_text())
        seen_ids, seen_grids = set(legacy["prompt_ids"]), set(legacy["grid_sha256"])
        manifest = fixture_manifest()
        self.assertEqual(manifest["dataset_version"], "private-six-panels-v3")
        self.assertEqual(manifest["selection"]["source_sha256"], packer.SOURCE_SHA256)
        self.assertEqual(manifest["selection"]["source_bytes"], packer.SOURCE_BYTES)
        self.assertEqual(manifest["selection"]["source_rows_scanned"], 1299992)
        counts = {**packer.COUNTS, "private_train": 1152, "private_eval": 576}
        for name, count in counts.items():
            rows = list(_rows(name).values())
            self.assertEqual(len(rows), count)
            grid_hashes = [packer.grid_digest(row) for row in rows]
            self.assertEqual(grid_hashes, manifest["partitions"][name]["grid_sha256"])
            for row, grid in zip(rows, grid_hashes):
                self.assertNotIn(row["prompt_id"], seen_ids)
                self.assertNotIn(grid, seen_grids)
                seen_ids.add(row["prompt_id"])
                seen_grids.add(grid)

    def test_public_bundle_does_not_disclose_private_rows_or_failure_labels(self):
        manifest = fixture_manifest()
        public = FIXTURES.parent / "public_tests/fixtures"
        public_manifest = json.loads((public / "manifest.json").read_text())
        self.assertNotIn("private", public_manifest["panels"])
        self.assertNotIn("seed", public_manifest["selection"])
        self.assertNotIn(str(manifest["selection"]["seed"]), (public / "manifest.json").read_text())
        self.assertNotIn(str(manifest["selection"]["private_extension"]["seed"]),
                         (public / "manifest.json").read_text())
        self.assertEqual(set(public_manifest["partitions"]), {"public_train", "public_eval"})
        self.assertFalse((public / "legacy_exclusions.json").exists())
        self.assertFalse((public / "private_extension_plan.json").exists())
        for name in ("public_train", "public_eval"):
            self.assertEqual(public_manifest["partitions"][name], manifest["partitions"][name])
        labels = json.loads(gzip.decompress((public / "frozen_failures.json.gz").read_bytes()))
        self.assertEqual(set(map(int, labels["actions_by_prompt_id"])),
                         set(manifest["partitions"]["public_train"]["prompt_ids"]))


if __name__ == "__main__":
    unittest.main()
