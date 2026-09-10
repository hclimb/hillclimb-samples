"""Network-free contracts for the model-independent Dolci32K raw artifacts."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from SFT.data.dolci32k import artifacts as artifact_module
from SFT.data.dolci32k import profile as profile_module
from SFT.data.dolci32k.artifacts import (
    MANIFEST_SCHEMA_VERSION,
    profile_fingerprint,
    validate_build,
)
from SFT.data.dolci32k.builder import (
    Dolci32KBuilder,
    _annotate,
    audit_pools,
    inspect_dolci_metadata,
)
from SFT.data.dolci32k.common import (
    atomic_write_json,
    atomic_write_jsonl,
    capped_equal_waterfill,
    largest_remainder,
    make_record,
    normalize_prompt,
    prompt_hash,
)
from SFT.data.dolci32k.profile import (
    DOLCI_SOURCE_DOMAINS,
    EXPECTED_DOMAIN_COUNTS,
    EXPECTED_DOMAIN_SOURCE_COUNTS,
    EXPECTED_SOURCE_COUNTS,
    EXPECTED_TOTAL_ROWS,
    INSTRUCTION_EXCLUDED_DOMAINS,
    INSTRUCTION_SOURCES,
    PRECISE_IF_SOURCE,
    PROFILE_NAME,
    PROFILE_VERSION,
    REASONING_CATEGORY_QUOTAS,
    REASONING_SOURCE_CATEGORY,
    REASONING_SOURCE_QUOTAS,
    SETTINGS,
    BuildSizes,
)
from SFT.data.dolci32k.sources import dolci_domain, validate_dolci_pair


REVIEWED_INVENTORY = {
    "Chat": {"Wildchat": 302_406, "OpenAssistant": 7_132},
    "Coding": {
        "Dolci Instruct Python Algorithms": 186_345,
        "Evol CodeAlpaca": 107_270,
        "Tulu 3 Persona Python": 34_999,
    },
    "Hardcoded Data": {"Hardcoded Data": 69},
    "Math": {
        "Tulu 3 Persona MATH": 149_958,
        "OpenMathInstruct 2": 50_000,
        "Tulu 3 Persona GSM": 49_980,
        "Tulu 3 Persona Algebra": 19_999,
    },
    "Multilingual": {"Aya": 99_987},
    "Other": {"Logic Puzzles": 159_882, "FLAN": 89_981, "TableGPT": 5_000},
    "Precise IF": {"Dolci Instruct Precise IF": 136_833},
    "Reasoning": {"Verifiable Reasoning": 310_572},
    "Safety": {
        "WildJailbreak": 49_965,
        "WildGuardMix": 49_373,
        "CoCoNot": 10_957,
    },
    "Science": {
        "Dolci Instruct OpenThoughts3+ Science": 99_268,
        "SciRiff": 4_557,
    },
    "Tool Use": {"Dolci Instruct Tool Use": 227_579},
}


def _record(
    record_id: str,
    prompt: str,
    *,
    source: str,
    domain: str,
    source_domain: str,
    reasoning_category: str | None = None,
    task_id: int | None = None,
) -> dict:
    metadata = {"source_id": record_id, "source_domain": source_domain}
    if task_id is not None:
        metadata["task_id"] = task_id
    row = make_record(
        record_id=record_id,
        dataset=source,
        domain=domain,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"answer for {record_id}"},
        ],
        metadata=metadata,
    )
    if reasoning_category is not None:
        row["reasoning_category"] = reasoning_category
        row["reasoning_stratum"] = f"{reasoning_category}::{source}"
        row["metadata"]["reasoning_category"] = reasoning_category
    return row


def _benchmark(record_id: str, prompt: str, *, task_id: int | None = None) -> dict:
    metadata = {} if task_id is None else {"task_id": task_id}
    return make_record(
        record_id=record_id,
        dataset=record_id.split("::", 1)[0],
        domain="evaluation",
        messages=[{"role": "user", "content": prompt}],
        metadata=metadata,
    )


def _audit_fixture():
    """Return a two-row-per-split fixture satisfying every audit contract."""
    sizes = BuildSizes(
        general_train=2,
        general_val=2,
        target_grad=1,
        target_val=1,
        mixed_instruction_train=1,
        mixed_reasoning_train=1,
        mixed_instruction_val=1,
        mixed_reasoning_val=1,
    )
    reasoning_quotas = {
        "Math": {"Tulu 3 Persona MATH": {"train": 2, "val": 2}}
    }

    instruction_train = [
        _annotate(
            _record(
                "inst-train-wildchat",
                "unique inst train chat",
                source="Wildchat",
                domain="chat",
                source_domain="Chat",
            ),
            pool="instruction_32k",
            split="train",
            role="general",
        ),
        _annotate(
            _record(
                "inst-train-precise",
                "unique inst train precise",
                source=PRECISE_IF_SOURCE,
                domain="precise_if",
                source_domain="Precise IF",
            ),
            pool="instruction_32k",
            split="train",
            role="general",
        ),
    ]
    instruction_val = [
        _annotate(
            _record(
                "inst-val-wildchat",
                "unique inst val chat",
                source="Wildchat",
                domain="chat",
                source_domain="Chat",
            ),
            pool="instruction_32k",
            split="val",
            role="general",
        ),
        _annotate(
            _record(
                "inst-val-precise",
                "unique inst val precise",
                source=PRECISE_IF_SOURCE,
                domain="precise_if",
                source_domain="Precise IF",
            ),
            pool="instruction_32k",
            split="val",
            role="general",
        ),
    ]

    def reasoning_row(split: str, index: int) -> dict:
        return _annotate(
            _record(
                f"reason-{split}-{index}",
                f"unique reason {split} {index}",
                source="Tulu 3 Persona MATH",
                domain="math",
                source_domain="Math",
                reasoning_category="Math",
            ),
            pool="reasoning_32k",
            split=split,
            role="general",
        )

    reasoning_train = [reasoning_row("train", index) for index in range(2)]
    reasoning_val = [reasoning_row("val", index) for index in range(2)]
    mixed_train = [
        _annotate(
            instruction_train[1],
            pool="mixed_32k",
            split="train",
            role="general",
            parent_pool="instruction_32k",
        ),
        _annotate(
            reasoning_train[0],
            pool="mixed_32k",
            split="train",
            role="general",
            parent_pool="reasoning_32k",
        ),
    ]
    mixed_val = [
        _annotate(
            instruction_val[1],
            pool="mixed_32k",
            split="val",
            role="general",
            parent_pool="instruction_32k",
        ),
        _annotate(
            reasoning_val[0],
            pool="mixed_32k",
            split="val",
            role="general",
            parent_pool="reasoning_32k",
        ),
    ]
    pools = {
        "instruction_32k": {"train": instruction_train, "val": instruction_val},
        "reasoning_32k": {"train": reasoning_train, "val": reasoning_val},
        "mixed_32k": {"train": mixed_train, "val": mixed_val},
    }

    targets = {}
    for target in ("precise_if", "math", "mbpp"):
        targets[target] = {
            "grad": [
                _record(
                    f"{target}-grad",
                    f"unique target {target} grad",
                    source=target,
                    domain=target,
                    source_domain="target",
                    task_id=1 if target == "mbpp" else None,
                )
            ],
            "val": [
                _record(
                    f"{target}-val",
                    f"unique target {target} val",
                    source=target,
                    domain=target,
                    source_domain="target",
                    task_id=2 if target == "mbpp" else None,
                )
            ],
        }
    benchmarks = {
        "ifeval": [_benchmark("ifeval::0", "unique final ifeval")],
        "ifbench": [_benchmark("ifbench::0", "unique final ifbench")],
        "math500": [_benchmark("math500::0", "unique final math")],
        "mbpp_plus": [
            _benchmark("mbpp_plus::1000", "unique final code", task_id=1000)
        ],
    }
    quotas = {
        "instruction": {
            "train": {"Wildchat": 1, PRECISE_IF_SOURCE: 1},
            "val": {"Wildchat": 1, PRECISE_IF_SOURCE: 1},
        },
        "mixed": {
            "instruction_train": {PRECISE_IF_SOURCE: 1},
            "reasoning_train": {"Math::Tulu 3 Persona MATH": 1},
            "instruction_val": {PRECISE_IF_SOURCE: 1},
            "reasoning_val": {"Math::Tulu 3 Persona MATH": 1},
        },
    }
    return pools, targets, benchmarks, sizes, quotas, reasoning_quotas


class ReviewedMetadataTests(unittest.TestCase):
    def test_full_reviewed_inventory_is_exact(self):
        self.assertEqual(EXPECTED_DOMAIN_SOURCE_COUNTS, REVIEWED_INVENTORY)
        self.assertEqual(EXPECTED_TOTAL_ROWS, 2_152_112)
        self.assertEqual(len(EXPECTED_DOMAIN_COUNTS), 11)
        self.assertEqual(len(EXPECTED_SOURCE_COUNTS), 22)
        self.assertEqual(
            EXPECTED_DOMAIN_COUNTS,
            {domain: sum(sources.values()) for domain, sources in REVIEWED_INVENTORY.items()},
        )
        self.assertEqual(sum(EXPECTED_SOURCE_COUNTS.values()), EXPECTED_TOTAL_ROWS)

    def test_metadata_mapping_fails_closed_on_unreviewed_values_and_pairs(self):
        for domain, sources in REVIEWED_INVENTORY.items():
            for source in sources:
                validate_dolci_pair(domain, source)
                self.assertEqual(DOLCI_SOURCE_DOMAINS[source], domain)

        with self.assertRaisesRegex(ValueError, "Unknown literal Dolci source_dataset"):
            validate_dolci_pair("Other", "Unreviewed Source")
        with self.assertRaisesRegex(ValueError, "Unreviewed Dolci domain/source pairing"):
            validate_dolci_pair("Other", "Wildchat")
        with self.assertRaisesRegex(ValueError, "Unknown literal Dolci domain"):
            dolci_domain("Unreviewed Domain")

    def test_inspection_reports_cross_counts_and_rejects_inventory_drift(self):
        rows = [
            {"id": "one", "domain": "Chat", "source_dataset": "Wildchat", "messages": []},
            {"id": "two", "domain": "Chat", "source_dataset": "Wildchat", "messages": []},
            {"id": "three", "domain": "Other", "source_dataset": "FLAN", "messages": []},
        ]
        report = inspect_dolci_metadata(rows, strict_counts=False, representative_per_pair=1)
        self.assertEqual(report["total_examples"], 3)
        self.assertEqual(report["domain_counts"], {"Chat": 2, "Other": 1})
        self.assertEqual(
            report["domain_x_source_dataset_counts"],
            {"Chat": {"Wildchat": 2}, "Other": {"FLAN": 1}},
        )
        with self.assertRaisesRegex(RuntimeError, "row count drift"):
            inspect_dolci_metadata(rows, strict_counts=True)


class AllocationAndMappingTests(unittest.TestCase):
    def test_capped_equal_waterfill_redistributes_shortages_deterministically(self):
        capacities = {"tiny": 1, "beta": 100, "alpha": 100}
        expected = {"tiny": 1, "beta": 5, "alpha": 5}
        self.assertEqual(capped_equal_waterfill(capacities, 11), expected)
        self.assertEqual(
            capped_equal_waterfill(dict(reversed(list(capacities.items()))), 11),
            expected,
        )
        remainder = capped_equal_waterfill({"charlie": 100, "alpha": 100, "beta": 100}, 5)
        self.assertEqual(remainder, {"charlie": 1, "alpha": 2, "beta": 2})
        self.assertEqual(sum(remainder.values()), 5)
        self.assertTrue(all(remainder[key] <= 100 for key in remainder))
        with self.assertRaisesRegex(RuntimeError, "only 3 available"):
            capped_equal_waterfill({"a": 1, "b": 2}, 4)

    def test_reasoning_source_and_category_quotas_are_exact(self):
        expected = {
            "Math": {
                "Tulu 3 Persona MATH": {"train": 3000, "val": 48},
                "OpenMathInstruct 2": {"train": 3000, "val": 48},
                "Tulu 3 Persona GSM": {"train": 3000, "val": 48},
                "Tulu 3 Persona Algebra": {"train": 3000, "val": 48},
            },
            "Coding": {
                "Dolci Instruct Python Algorithms": {"train": 4000, "val": 64},
                "Evol CodeAlpaca": {"train": 4000, "val": 64},
                "Tulu 3 Persona Python": {"train": 4000, "val": 64},
            },
            "Science/Logic/Verifiable": {
                "Verifiable Reasoning": {"train": 2000, "val": 32},
                "Logic Puzzles": {"train": 2000, "val": 32},
                "Dolci Instruct OpenThoughts3+ Science": {"train": 2000, "val": 32},
                "SciRiff": {"train": 2000, "val": 32},
            },
        }
        self.assertEqual(REASONING_SOURCE_QUOTAS, expected)
        self.assertEqual(
            REASONING_CATEGORY_QUOTAS,
            {
                "Math": {"train": 12_000, "val": 192},
                "Coding": {"train": 12_000, "val": 192},
                "Science/Logic/Verifiable": {"train": 8_000, "val": 128},
            },
        )
        self.assertEqual(sum(value["train"] for value in REASONING_CATEGORY_QUOTAS.values()), 32_000)
        self.assertEqual(sum(value["val"] for value in REASONING_CATEGORY_QUOTAS.values()), 512)
        self.assertEqual(
            set(REASONING_SOURCE_CATEGORY),
            {source for sources in expected.values() for source in sources},
        )

    def test_instruction_mapping_excludes_math_coding_science_and_tool_use(self):
        self.assertEqual(
            set(INSTRUCTION_EXCLUDED_DOMAINS),
            {"Math", "Coding", "Science", "Tool Use"},
        )
        expected = {
            source
            for source, domain in DOLCI_SOURCE_DOMAINS.items()
            if domain not in INSTRUCTION_EXCLUDED_DOMAINS
        }
        self.assertEqual(set(INSTRUCTION_SOURCES), expected)
        self.assertEqual(len(INSTRUCTION_SOURCES), 12)
        self.assertIn(PRECISE_IF_SOURCE, INSTRUCTION_SOURCES)
        self.assertIn("Logic Puzzles", INSTRUCTION_SOURCES)
        self.assertIn("Verifiable Reasoning", INSTRUCTION_SOURCES)
        self.assertNotIn("Dolci Instruct OpenThoughts3+ Science", INSTRUCTION_SOURCES)
        self.assertNotIn("SciRiff", INSTRUCTION_SOURCES)
        self.assertTrue(
            all(DOLCI_SOURCE_DOMAINS[source] not in {"Math", "Coding", "Science", "Tool Use"}
                for source in INSTRUCTION_SOURCES)
        )

    def test_instruction_science_exclusion_refills_the_32512_parent(self):
        capacities = {source: 10_000 for source in INSTRUCTION_SOURCES}
        capacities["Hardcoded Data"] = 64
        expected_combined = {
            "Aya": 2_950,
            "CoCoNot": 2_950,
            "Dolci Instruct Precise IF": 2_950,
            "FLAN": 2_950,
            "Hardcoded Data": 64,
            "Logic Puzzles": 2_950,
            "OpenAssistant": 2_950,
            "TableGPT": 2_950,
            "Verifiable Reasoning": 2_950,
            "WildGuardMix": 2_950,
            "WildJailbreak": 2_949,
            "Wildchat": 2_949,
        }
        self.assertEqual(
            capped_equal_waterfill(capacities, 32_512),
            expected_combined,
        )
        expected_val = {
            "Aya": 47,
            "CoCoNot": 47,
            "Dolci Instruct Precise IF": 47,
            "FLAN": 47,
            "Hardcoded Data": 1,
            "Logic Puzzles": 47,
            "OpenAssistant": 46,
            "TableGPT": 46,
            "Verifiable Reasoning": 46,
            "WildGuardMix": 46,
            "WildJailbreak": 46,
            "Wildchat": 46,
        }
        self.assertEqual(largest_remainder(expected_combined, 512), expected_val)
        self.assertEqual(
            {
                source: expected_combined[source] - expected_val[source]
                for source in expected_combined
            },
            {
                "Aya": 2_903,
                "CoCoNot": 2_903,
                "Dolci Instruct Precise IF": 2_903,
                "FLAN": 2_903,
                "Hardcoded Data": 63,
                "Logic Puzzles": 2_903,
                "OpenAssistant": 2_904,
                "TableGPT": 2_904,
                "Verifiable Reasoning": 2_904,
                "WildGuardMix": 2_904,
                "WildJailbreak": 2_903,
                "Wildchat": 2_903,
            },
        )


class DisjointnessAndNestedMixedTests(unittest.TestCase):
    def test_synthetic_pool_audit_enforces_all_identity_and_lineage_contracts(self):
        fixture = _audit_fixture()
        result = audit_pools(
            fixture[0],
            fixture[1],
            fixture[2],
            sizes=fixture[3],
            quotas=fixture[4],
            reasoning_quotas=fixture[5],
        )
        self.assertEqual(result["status"], "passed")
        self.assertIn("train_validation_disjoint", result["checks"])
        self.assertIn("general_target_id_prompt_and_8gram_overlap_zero", result["checks"])
        self.assertIn("mixed_nested_stratified_lineage", result["checks"])

    def test_audit_rejects_target_id_and_normalized_prompt_overlap(self):
        for overlap_kind in ("id", "normalized_prompt"):
            with self.subTest(overlap_kind=overlap_kind):
                pools, targets, benchmarks, sizes, quotas, reasoning_quotas = _audit_fixture()
                candidate = pools["instruction_32k"]["train"][0]
                target = targets["precise_if"]["grad"][0]
                if overlap_kind == "id":
                    candidate["id"] = target["id"]
                else:
                    equivalent = "  UNIQUE\tTARGET PRECISE_IF   GRAD  "
                    self.assertEqual(normalize_prompt(equivalent), normalize_prompt("unique target precise_if grad"))
                    candidate["messages"][0]["content"] = equivalent
                    candidate["prompt_hash"] = prompt_hash(equivalent)
                    candidate["metadata"]["prompt_hash"] = candidate["prompt_hash"]
                with self.assertRaisesRegex(
                    AssertionError,
                    "overlaps (target IDs|normalized target prompts)",
                ):
                    audit_pools(
                        pools,
                        targets,
                        benchmarks,
                        sizes=sizes,
                        quotas=quotas,
                        reasoning_quotas=reasoning_quotas,
                    )

    def test_mixed_builder_is_nested_stratified_and_order_independent(self):
        sizes = BuildSizes(
            general_train=8,
            general_val=8,
            target_grad=1,
            target_val=1,
            mixed_instruction_train=4,
            mixed_reasoning_train=4,
            mixed_instruction_val=4,
            mixed_reasoning_val=4,
        )

        def parents(split: str):
            instruction_sources = ["Wildchat"] * 4 + [PRECISE_IF_SOURCE] * 2 + ["FLAN"] * 2
            instruction = [
                _annotate(
                    _record(
                        f"{split}-i-{index}",
                        f"mixed instruction {split} {index}",
                        source=source,
                        domain="instruction",
                        source_domain=DOLCI_SOURCE_DOMAINS[source],
                    ),
                    pool="instruction_32k",
                    split=split,
                    role="general",
                )
                for index, source in enumerate(instruction_sources)
            ]
            reasoning_sources = [
                ("Math", "Tulu 3 Persona MATH"),
                ("Math", "Tulu 3 Persona MATH"),
                ("Coding", "Evol CodeAlpaca"),
                ("Coding", "Evol CodeAlpaca"),
                ("Science/Logic/Verifiable", "Logic Puzzles"),
                ("Science/Logic/Verifiable", "Logic Puzzles"),
                ("Science/Logic/Verifiable", "SciRiff"),
                ("Science/Logic/Verifiable", "SciRiff"),
            ]
            reasoning = [
                _annotate(
                    _record(
                        f"{split}-r-{index}",
                        f"mixed reasoning {split} {index}",
                        source=source,
                        domain="reasoning",
                        source_domain=DOLCI_SOURCE_DOMAINS[source],
                        reasoning_category=category,
                    ),
                    pool="reasoning_32k",
                    split=split,
                    role="general",
                )
                for index, (category, source) in enumerate(reasoning_sources)
            ]
            return instruction, reasoning

        instruction_train, reasoning_train = parents("train")
        instruction_val, reasoning_val = parents("val")
        builder = Dolci32KBuilder("/tmp", sizes=sizes, loaders=object(), strict_inventory=False)
        mixed_train, mixed_val = builder._build_mixed(
            instruction_train, instruction_val, reasoning_train, reasoning_val
        )
        second = Dolci32KBuilder("/tmp", sizes=sizes, loaders=object(), strict_inventory=False)
        reverse_train, reverse_val = second._build_mixed(
            list(reversed(instruction_train)),
            list(reversed(instruction_val)),
            list(reversed(reasoning_train)),
            list(reversed(reasoning_val)),
        )

        self.assertEqual([row["id"] for row in mixed_train], [row["id"] for row in reverse_train])
        self.assertEqual([row["id"] for row in mixed_val], [row["id"] for row in reverse_val])
        for split, mixed, instruction, reasoning in (
            ("train", mixed_train, instruction_train, reasoning_train),
            ("val", mixed_val, instruction_val, reasoning_val),
        ):
            instruction_half = [row for row in mixed if row["parent_pool"] == "instruction_32k"]
            reasoning_half = [row for row in mixed if row["parent_pool"] == "reasoning_32k"]
            self.assertEqual(len(instruction_half), 4)
            self.assertEqual(len(reasoning_half), 4)
            self.assertLessEqual({row["id"] for row in instruction_half}, {row["id"] for row in instruction})
            self.assertLessEqual({row["id"] for row in reasoning_half}, {row["id"] for row in reasoning})
            self.assertEqual(
                Counter(row["source_dataset"] for row in instruction_half),
                Counter({"Wildchat": 2, PRECISE_IF_SOURCE: 1, "FLAN": 1}),
            )
            self.assertEqual(
                Counter(row["reasoning_stratum"] for row in reasoning_half),
                Counter({
                    "Math::Tulu 3 Persona MATH": 1,
                    "Coding::Evol CodeAlpaca": 1,
                    "Science/Logic/Verifiable::Logic Puzzles": 1,
                    "Science/Logic/Verifiable::SciRiff": 1,
                }),
            )
            self.assertEqual({row["split"] for row in mixed}, {split})


class SettingsAndArtifactIdentityTests(unittest.TestCase):
    def test_paired_settings_share_the_same_target_artifact_names(self):
        self.assertEqual(SETTINGS["inst_if"]["target"], "precise_if")
        self.assertEqual(SETTINGS["mixed_if"]["target"], "precise_if")
        self.assertEqual(SETTINGS["reason_math"]["target"], "math")
        self.assertEqual(SETTINGS["mixed_math"]["target"], "math")
        self.assertEqual(SETTINGS["reason_code"]["target"], "mbpp")
        self.assertNotIn("reason_mbpp", SETTINGS)

    def test_raw_fingerprint_ignores_tokenizer_knobs_but_tracks_allocations(self):
        base_config = copy.deepcopy(profile_module.CONFIG)
        with mock.patch.object(profile_module, "CONFIG", base_config):
            baseline = profile_fingerprint()

        tokenizer_change = copy.deepcopy(base_config)
        tokenizer_change["max_seq_len"] = 16_384
        tokenizer_change["tokenizer_use_fast"] = not base_config["tokenizer_use_fast"]
        tokenizer_change["truncation_warning_threshold"] = 0.25
        tokenizer_change["tokenizer_profiles"]["olmo3_7b"]["tokenizer_revision"] = "derived-only-change"
        with mock.patch.object(profile_module, "CONFIG", tokenizer_change):
            self.assertEqual(profile_fingerprint(), baseline)

        allocation_change = copy.deepcopy(base_config)
        allocation_change["mixed"]["train"]["instruction"] -= 1
        allocation_change["mixed"]["train"]["reasoning"] += 1
        with mock.patch.object(profile_module, "CONFIG", allocation_change):
            self.assertNotEqual(profile_fingerprint(), baseline)

    def test_manifest_tamper_is_rejected_before_artifact_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            build_id = "a" * 64
            build = Path(temporary) / build_id
            build.mkdir()
            manifest = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "profile_name": PROFILE_NAME,
                "profile_version": PROFILE_VERSION,
                "profile_fingerprint": artifact_module.profile_fingerprint(),
                "build_id": build_id,
                "audit_status": "passed",
                "model_independent_raw_membership": True,
                "artifacts": {},
            }
            atomic_write_json(build / "manifest.json", manifest)
            with self.assertRaisesRegex(RuntimeError, "manifest fingerprint mismatch"):
                validate_build(build)

            manifest["profile_name"] = "tampered-profile"
            atomic_write_json(build / "manifest.json", manifest)
            with self.assertRaisesRegex(RuntimeError, "Stale/bad dolci32k manifest"):
                validate_build(build)

    def test_atomic_jsonl_failure_preserves_previous_file_and_cleans_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "rows.jsonl"
            original = json.dumps({"old": True}) + "\n"
            destination.write_text(original, encoding="utf-8")

            def interrupted_rows():
                yield {"new": 1}
                raise RuntimeError("interrupted build")

            with self.assertRaisesRegex(RuntimeError, "interrupted build"):
                atomic_write_jsonl(destination, interrupted_rows())
            self.assertEqual(destination.read_text(encoding="utf-8"), original)
            self.assertEqual(list(Path(temporary).glob(".rows.jsonl.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
