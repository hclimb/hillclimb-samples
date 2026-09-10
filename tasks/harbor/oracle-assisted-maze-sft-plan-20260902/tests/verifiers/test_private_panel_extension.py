import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
sys.path.insert(0, str(FIXTURES))
import extend_private_panels as extension


class PrivateExtensionTests(unittest.TestCase):
    def test_original_rows_and_failure_labels_are_preserved(self):
        plan = json.loads((FIXTURES / "private_extension_plan.json").read_text())
        manifest = json.loads((FIXTURES / "manifest.json").read_text())
        self.assertEqual(manifest["panels"]["private"], 6)
        self.assertEqual(plan["seed"], 2026090803)
        self.assertEqual(len(plan["prompt_ids"]), 6688)
        base_ids = set(manifest["partitions"]["public_train"]["prompt_ids"])
        for name in ("private_train", "private_eval"):
            _, rows = extension.base_rows(name, plan)
            self.assertEqual(len(rows), plan["base_counts"][name])
            if name == "private_train":
                base_ids.update(row["prompt_id"] for row in rows)
            metadata = manifest["partitions"][name]
            added_ids = metadata["prompt_ids"][plan["base_counts"][name]:]
            added_grids = metadata["grid_sha256"][plan["base_counts"][name]:]
            self.assertEqual(len(added_ids), plan["base_counts"][name])
            self.assertFalse(set(added_ids) & set(plan["prompt_ids"]))
            self.assertFalse(set(added_grids) & set(plan["grid_sha256"]))
        labels = json.loads(gzip.decompress((FIXTURES / "frozen_failures.json.gz").read_bytes()))
        self.assertEqual(len(labels["actions_by_prompt_id"]), 2304)
        labels["actions_by_prompt_id"] = {k: v for k, v in labels["actions_by_prompt_id"].items() if int(k) in base_ids}
        raw = (json.dumps(labels, sort_keys=True, separators=(",", ":")) + "\n").encode()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), plan["base_failures_raw_sha256"])
        record = manifest["selection"]["private_extension"]
        self.assertEqual(record["plan_sha256"], hashlib.sha256((FIXTURES / "private_extension_plan.json").read_bytes()).hexdigest())
        self.assertEqual(record["source_rows_scanned"], 1299992)

    def test_extension_rejects_wrong_source_overlap_and_short_samples(self):
        selected = [(i, {"prompt_id": i, "grid": str(i)}) for i in range(864)]
        selection = {"source_sha256": "pinned", "source_bytes": 123}
        plan = {"prompt_ids": [9999], "grid_sha256": ["9999"]}
        packer = SimpleNamespace(SOURCE_SHA256="pinned", SOURCE_BYTES=123, grid_digest=lambda r: r["grid"])
        with mock.patch.object(extension, "packer", packer):
            extension.validate_selection(selected, selection, plan)
            for bad in (selected[:-1], selected[:-1] + [selected[0]]):
                with self.assertRaises(RuntimeError):
                    extension.validate_selection(bad, selection, plan)
            for blocked in ({"prompt_ids": [1], "grid_sha256": []}, {"prompt_ids": [], "grid_sha256": ["1"]}):
                with self.assertRaises(RuntimeError):
                    extension.validate_selection(selected, selection, blocked)
            with self.assertRaises(RuntimeError):
                extension.validate_selection(selected, {**selection, "source_sha256": "changed"}, plan)
