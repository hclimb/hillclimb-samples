"""Build Dolci32K in the fixed order: benchmarks, targets, reasoning, instruction, mixed."""

from __future__ import annotations

import collections
import copy
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .artifacts import (
    MANIFEST_SCHEMA_VERSION, dolci32k_root, profile_fingerprint, validate_build,
)
from .common import (
    atomic_write_json, atomic_write_jsonl, canonical_json, capped_equal_waterfill,
    file_sha256, first_user_prompt, largest_remainder, ordered_id_sha256,
    prompt_hash, read_jsonl, stable_rank,
)
from .decontam import PromptDecontaminator, audit_against, references_from_records
from .profile import (
    BENCHMARKS, CANDIDATE_BATCH_SIZE, CANDIDATE_SEED, DEFAULT_SIZES,
    DOLCI_KNOWN_SOURCES, DOLCI_SOURCE_DOMAINS, EVALPLUS_DATASET_VERSION,
    EVALPLUS_IMAGE, EVALPLUS_VERSION, EXPECTED_DOMAIN_COUNTS,
    EXPECTED_DOMAIN_SOURCE_COUNTS, EXPECTED_SOURCE_COUNTS, EXPECTED_TOTAL_ROWS,
    GENERAL_POOLS, IFBENCH_EVALUATOR_GIT_COMMIT, IFEVAL_EVALUATOR_GIT_COMMIT,
    INSTRUCTION_EXCLUDED_DOMAINS, INSTRUCTION_SOURCES, MATH_VERIFY_VERSION,
    PINNED_SOURCES, PRECISE_IF_SOURCE, PROFILE_NAME, PROFILE_VERSION,
    REASONING_SOURCE_CATEGORY, REASONING_SOURCE_QUOTAS, REASONING_SOURCES,
    FORMAL_STEPS, SEED, SELECTED_SUBSET_SIZE, SETTINGS, TARGETS,
    BuildSizes, artifact_relative_path,
)
from .sources import (
    PinnedLoaders, convert_dolci, convert_if_benchmark, convert_math500,
    convert_math_train, convert_mbpp_plus, convert_mbpp_train,
    has_structural_tool_use, validate_dolci_pair,
)


def _source_id(record: Mapping[str, Any]) -> str:
    return str((record.get("metadata") or {}).get("source_id", record.get("id", "")))


def mbpp_task_number(task_id: Any) -> Optional[int]:
    if task_id is None:
        return None
    try:
        return int(str(task_id).strip().rsplit("/", 1)[-1])
    except ValueError:
        return None


def mbpp_plus_task_numbers(records: Iterable[Mapping[str, Any]]) -> set[int]:
    numbers: set[int] = set()
    for record in records:
        number = mbpp_task_number((record.get("metadata") or {}).get("task_id"))
        if number is None:
            raise RuntimeError(f"MBPP+ record has an unparsable task_id: {record.get('id')!r}")
        numbers.add(number)
    if not numbers:
        raise RuntimeError("MBPP+ benchmark exposed no task IDs")
    return numbers


def _ranked_unique(
    records: Iterable[Mapping[str, Any]], *, namespace: str,
    blocker: Optional[PromptDecontaminator] = None,
    excluded_ids: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    best: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    excluded_ids = excluded_ids or set()
    for raw in records:
        record = dict(raw)
        digest = str(record.get("prompt_hash") or prompt_hash(first_user_prompt(record)))
        if not digest or record.get("id") in excluded_ids or _source_id(record) in excluded_ids:
            continue
        if blocker is not None and blocker.blocked_record(record):
            continue
        rank = stable_rank(namespace, digest, _source_id(record))
        current = best.get(digest)
        if current is None or rank < current[0]:
            best[digest] = (rank, record)
    return [record for _, record in sorted(best.values(), key=lambda value: (value[0], value[1]["id"]))]


def select_target_pair(
    records: Iterable[Mapping[str, Any]], *, namespace: str,
    blocker: PromptDecontaminator, grad_count: int, val_count: int,
    is_usable: Optional[Callable[[Mapping[str, Any]], bool]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ordered = _ranked_unique(records, namespace=namespace, blocker=blocker)
    required = grad_count + val_count
    selected: List[Dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    for record in ordered:
        source_id = _source_id(record)
        if source_id in seen_source_ids or (is_usable and not is_usable(record)):
            continue
        seen_source_ids.add(source_id)
        selected.append(record)
        if len(selected) == required:
            break
    if len(selected) != required:
        raise RuntimeError(f"{namespace}: only {len(selected)} clean unique rows; need {required}")
    grad, val = selected[:grad_count], selected[grad_count:]
    if {_source_id(row) for row in grad} & {_source_id(row) for row in val}:
        raise AssertionError(f"{namespace}: target source IDs overlap")
    if {row["prompt_hash"] for row in grad} & {row["prompt_hash"] for row in val}:
        raise AssertionError(f"{namespace}: target prompts overlap")
    return grad, val


def _counts(records: Sequence[Mapping[str, Any]], key: str) -> Dict[str, int]:
    return dict(sorted(collections.Counter(str(row.get(key, "unknown")) for row in records).items()))


def _cross_counts(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    cross: Dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for row in records:
        cross[str(row.get("domain", "unknown"))][str(row.get("source_dataset", "unknown"))] += 1
    return {domain: dict(sorted(counts.items())) for domain, counts in sorted(cross.items())}


def _proportional_quota(records: Sequence[Mapping[str, Any]], total: int, key: str) -> Dict[str, int]:
    counts = _counts(records, key)
    quota = largest_remainder(counts, total)
    if any(quota[group] > counts[group] for group in quota):
        raise RuntimeError(f"nested quota exceeds parent availability: {quota} vs {counts}")
    return quota


def _nested_subset(
    records: Sequence[Dict[str, Any]], quotas: Mapping[str, int], *,
    group_key: str, namespace: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        grouped[str(record.get(group_key, "unknown"))].append(record)
    selected: List[Dict[str, Any]] = []
    for group, quota in sorted(quotas.items()):
        ordered = sorted(grouped.get(group, []), key=lambda row: (stable_rank(namespace, group, row["id"]), row["id"]))
        if len(ordered) < quota:
            raise RuntimeError(f"Nested subset {namespace}/{group}: available={len(ordered)}, need={quota}")
        selected.extend(ordered[:quota])
    return sorted(selected, key=lambda row: (stable_rank(namespace, row["id"]), row["id"]))


def _protected_index(benchmarks, targets) -> PromptDecontaminator:
    groups = {f"benchmark:{key}": value for key, value in benchmarks.items()}
    for target, splits in targets.items():
        for split, records in splits.items():
            groups[f"target:{target}:{split}"] = records
    return PromptDecontaminator(references_from_records(groups))


def _annotate(record: Mapping[str, Any], *, pool: str, split: str, role: str, parent_pool: Optional[str] = None) -> Dict[str, Any]:
    value = copy.deepcopy(dict(record))
    value.update({"pool_name": pool, "split": split, "data_role": role})
    if parent_pool:
        value["parent_pool"] = parent_pool
    value["metadata"] = dict(value.get("metadata") or {})
    value["metadata"].update({"pool_name": pool, "split": split, "data_role": role})
    if parent_pool:
        value["metadata"]["parent_pool"] = parent_pool
    return value


def _assert_unique(records: Sequence[Mapping[str, Any]], label: str) -> None:
    ids = [str(row.get("id")) for row in records]
    prompts = [str(row.get("prompt_hash")) for row in records]
    if len(ids) != len(set(ids)):
        raise AssertionError(f"{label} contains duplicate IDs")
    if len(prompts) != len(set(prompts)):
        raise AssertionError(f"{label} contains duplicate normalized prompts")


def inspect_dolci_metadata(
    rows: Iterable[Mapping[str, Any]], *, strict_counts: bool = True,
    representative_per_pair: int = 2,
) -> Dict[str, Any]:
    domains: collections.Counter[str] = collections.Counter()
    sources: collections.Counter[str] = collections.Counter()
    cross: Dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    representatives: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        domain, source = str(row.get("domain")), str(row.get("source_dataset"))
        validate_dolci_pair(domain, source)
        domains[domain] += 1
        sources[source] += 1
        cross[domain][source] += 1
        key = f"{domain} :: {source}"
        if len(representatives[key]) < representative_per_pair:
            representatives[key].append({"index": index, "id": row.get("id"), "messages": list(row.get("messages") or [])[:2]})
    report = {
        "total_examples": sum(domains.values()),
        "domain_counts": dict(sorted(domains.items())),
        "source_dataset_counts": dict(sorted(sources.items())),
        "domain_x_source_dataset_counts": {
            domain: dict(sorted(values.items())) for domain, values in sorted(cross.items())
        },
        "representative_examples": dict(sorted(representatives.items())),
    }
    if strict_counts:
        expected_cross = {domain: dict(sources) for domain, sources in EXPECTED_DOMAIN_SOURCE_COUNTS.items()}
        if report["total_examples"] != EXPECTED_TOTAL_ROWS:
            raise RuntimeError(f"Dolci row count drift: {report['total_examples']} != {EXPECTED_TOTAL_ROWS}")
        if report["domain_counts"] != dict(EXPECTED_DOMAIN_COUNTS):
            raise RuntimeError("Dolci domain counts differ from reviewed pinned inventory")
        if report["source_dataset_counts"] != dict(sorted(EXPECTED_SOURCE_COUNTS.items())):
            raise RuntimeError("Dolci source_dataset counts differ from reviewed pinned inventory")
        if report["domain_x_source_dataset_counts"] != expected_cross:
            raise RuntimeError("Dolci domain x source_dataset counts differ from reviewed inventory")
    return report


def _percentages(counts: Mapping[str, int], total: int) -> Dict[str, float]:
    return {key: value / total * 100.0 if total else 0.0 for key, value in sorted(counts.items())}


def audit_pools(
    pools, targets, benchmarks, *, sizes: BuildSizes,
    quotas: Mapping[str, Any],
    reasoning_quotas: Mapping[str, Mapping[str, Mapping[str, int]]] = REASONING_SOURCE_QUOTAS,
) -> Dict[str, Any]:
    """Assert counts, membership, exact identity, nesting, and 8-gram leakage."""
    for task, records in benchmarks.items():
        if not records:
            raise AssertionError(f"{task} final benchmark is empty")
    for pool, splits in pools.items():
        for split, expected in (("train", sizes.general_train), ("val", sizes.general_val)):
            if len(splits[split]) != expected:
                raise AssertionError(f"{pool}/{split} count={len(splits[split])}, expected={expected}")
            _assert_unique(splits[split], f"{pool}/{split}")
        train_ids = {row["id"] for row in splits["train"]}
        val_ids = {row["id"] for row in splits["val"]}
        train_prompts = {row["prompt_hash"] for row in splits["train"]}
        val_prompts = {row["prompt_hash"] for row in splits["val"]}
        if train_ids & val_ids or train_prompts & val_prompts:
            raise AssertionError(f"{pool} train/validation overlap")
    for target, splits in targets.items():
        if len(splits["grad"]) != sizes.target_grad or len(splits["val"]) != sizes.target_val:
            raise AssertionError(f"{target} target fixed-count contract failed")
        _assert_unique(splits["grad"], f"{target}/grad")
        _assert_unique(splits["val"], f"{target}/val")
        if ({row["id"] for row in splits["grad"]} & {row["id"] for row in splits["val"]}
                or {row["prompt_hash"] for row in splits["grad"]} & {row["prompt_hash"] for row in splits["val"]}):
            raise AssertionError(f"{target} grad/validation overlap")
    all_target_rows = [
        row for splits in targets.values() for rows in splits.values() for row in rows
    ]
    if len({row["id"] for row in all_target_rows}) != len(all_target_rows):
        raise AssertionError("Target artifacts overlap one another by stable ID")
    if len({row["prompt_hash"] for row in all_target_rows}) != len(all_target_rows):
        raise AssertionError("Target artifacts overlap one another by normalized prompt")

    expected_reasoning: Dict[str, Dict[str, int]] = {"train": {}, "val": {}}
    expected_categories: Dict[str, collections.Counter[str]] = {
        "train": collections.Counter(), "val": collections.Counter(),
    }
    for category, sources in reasoning_quotas.items():
        for source, split_counts in sources.items():
            for split in ("train", "val"):
                expected_reasoning[split][source] = int(split_counts[split])
                expected_categories[split][category] += int(split_counts[split])
    for split in ("train", "val"):
        observed = _counts(pools["reasoning_32k"][split], "source_dataset")
        if observed != dict(sorted(expected_reasoning[split].items())):
            raise AssertionError(f"Reasoning {split} exact source quota failed: {observed}")
        if _counts(pools["reasoning_32k"][split], "reasoning_category") != dict(sorted(expected_categories[split].items())):
            raise AssertionError(f"Reasoning {split} category quota failed")
    for split in ("train", "val"):
        instruct = pools["instruction_32k"][split]
        if set(_counts(instruct, "source_dataset")) - set(INSTRUCTION_SOURCES):
            raise AssertionError(f"Instruction {split} includes an excluded source")
        literal_domains = {str((row.get("metadata") or {}).get("source_domain")) for row in instruct}
        if literal_domains & set(INSTRUCTION_EXCLUDED_DOMAINS):
            raise AssertionError(f"Instruction {split} includes excluded domains")
        if any(has_structural_tool_use(row, row) for row in instruct):
            raise AssertionError(f"Instruction {split} includes structural tool-use data")
        expected_quota = (quotas.get("instruction") or {}).get(split, {})
        expected_quota = {str(k): int(v) for k, v in expected_quota.items() if int(v)}
        if _counts(instruct, "source_dataset") != dict(sorted(expected_quota.items())):
            raise AssertionError(f"Instruction {split} water-fill quota failed")

    protected = _protected_index(benchmarks, targets)
    final_only = PromptDecontaminator(references_from_records(benchmarks))
    target_ids = {row["id"] for splits in targets.values() for rows in splits.values() for row in rows}
    target_prompts = {row["prompt_hash"] for splits in targets.values() for rows in splits.values() for row in rows}
    for target, splits in targets.items():
        for split, records in splits.items():
            result = audit_against(records, final_only)
            if result["matches"]:
                raise AssertionError(f"{target}/{split} overlaps final evaluation: {result}")
    for pool, splits in pools.items():
        for split, records in splits.items():
            if {row["id"] for row in records} & target_ids:
                raise AssertionError(f"{pool}/{split} overlaps target IDs")
            if {row["prompt_hash"] for row in records} & target_prompts:
                raise AssertionError(f"{pool}/{split} overlaps normalized target prompts")
            result = audit_against(records, protected)
            if result["matches"]:
                raise AssertionError(f"{pool}/{split} overlaps protected prompts: {result}")

    blocked_mbpp_ids = mbpp_plus_task_numbers(benchmarks["mbpp_plus"])
    for split in ("grad", "val"):
        colliding = sorted(
            number for number in (
                mbpp_task_number((row.get("metadata") or {}).get("task_id"))
                for row in targets["mbpp"][split]
            ) if number in blocked_mbpp_ids
        )
        if colliding:
            raise AssertionError(f"mbpp/{split} reuses MBPP+ task IDs: {colliding}")

    instruct_all = pools["instruction_32k"]["train"] + pools["instruction_32k"]["val"]
    reasoning_all = pools["reasoning_32k"]["train"] + pools["reasoning_32k"]["val"]
    if ({row["id"] for row in instruct_all} & {row["id"] for row in reasoning_all}
            or {row["prompt_hash"] for row in instruct_all} & {row["prompt_hash"] for row in reasoning_all}):
        raise AssertionError("Instruction and Reasoning parents overlap by ID or normalized prompt")

    for split, instruction_count, reasoning_count in (
        ("train", sizes.mixed_instruction_train, sizes.mixed_reasoning_train),
        ("val", sizes.mixed_instruction_val, sizes.mixed_reasoning_val),
    ):
        mixed = pools["mixed_32k"][split]
        instruction_ids = {row["id"] for row in pools["instruction_32k"][split]}
        reasoning_ids = {row["id"] for row in pools["reasoning_32k"][split]}
        mixed_instruction = [row for row in mixed if row.get("parent_pool") == "instruction_32k"]
        mixed_reasoning = [row for row in mixed if row.get("parent_pool") == "reasoning_32k"]
        if len(mixed_instruction) != instruction_count or len(mixed_reasoning) != reasoning_count:
            raise AssertionError(f"Mixed {split} parent composition failed")
        if not {row["id"] for row in mixed_instruction} <= instruction_ids:
            raise AssertionError(f"Mixed {split} instruction half is not nested")
        if not {row["id"] for row in mixed_reasoning} <= reasoning_ids:
            raise AssertionError(f"Mixed {split} reasoning half is not nested")
        expected_i = (quotas.get("mixed") or {}).get(f"instruction_{split}", {})
        expected_r = (quotas.get("mixed") or {}).get(f"reasoning_{split}", {})
        expected_i = {str(k): int(v) for k, v in expected_i.items() if int(v)}
        expected_r = {str(k): int(v) for k, v in expected_r.items() if int(v)}
        if _counts(mixed_instruction, "source_dataset") != dict(sorted(expected_i.items())):
            raise AssertionError(f"Mixed instruction {split} stratification failed")
        if _counts(mixed_reasoning, "reasoning_stratum") != dict(sorted(expected_r.items())):
            raise AssertionError(f"Mixed reasoning {split} stratification failed")

    return {
        "status": "passed",
        "decontamination": protected.settings(),
        "checks": [
            "fixed_counts", "unique_ids_and_normalized_prompts_within_splits",
            "train_validation_disjoint", "target_grad_validation_disjoint",
            "target_artifacts_mutually_id_and_prompt_disjoint",
            "target_final_overlap_zero", "general_target_id_prompt_and_8gram_overlap_zero",
            "mbpp_task_id_disjoint_from_mbpp_plus", "reasoning_exact_per_source_quotas",
            "instruction_domain_source_and_structural_tool_exclusion",
            "instruction_reasoning_parent_disjoint", "mixed_nested_stratified_lineage",
        ],
    }


def _representatives(records: Sequence[Mapping[str, Any]], limit_per_source: int = 2) -> Dict[str, List[Dict[str, Any]]]:
    selected: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    ordered = sorted(
        records,
        key=lambda row: (stable_rank("statistics:representative", row["id"]), row["id"]),
    )
    for record in ordered:
        source = str(record.get("source_dataset"))
        if len(selected[source]) >= limit_per_source:
            continue
        messages = record.get("messages") or []
        assistant = next(
            (message.get("content", "") for message in messages if message.get("role") == "assistant"),
            "",
        )
        selected[source].append({
            "id": record.get("id"),
            "domain": record.get("domain"),
            "reasoning_category": record.get("reasoning_category"),
            "prompt_preview": first_user_prompt(record)[:500],
            "assistant_preview": str(assistant)[:500],
        })
    return dict(sorted(selected.items()))


def build_pool_statistics(pools, targets) -> Dict[str, Any]:
    report: Dict[str, Any] = {"general_pools": {}, "targets": {}}
    for pool, splits in pools.items():
        report["general_pools"][pool] = {}
        for split, records in splits.items():
            domains, sources = _counts(records, "domain"), _counts(records, "source_dataset")
            entry = {
                "total_examples": len(records),
                "domain_counts": domains,
                "domain_percentages": _percentages(domains, len(records)),
                "source_dataset_counts": sources,
                "source_dataset_percentages": _percentages(sources, len(records)),
                "domain_x_source_dataset_counts": _cross_counts(records),
                "representative_examples": _representatives(records),
            }
            if pool == "mixed_32k":
                parents = _counts(records, "parent_pool")
                entry["parent_pool_counts"] = parents
                entry["parent_pool_percentages"] = _percentages(parents, len(records))
            report["general_pools"][pool][split] = entry
    for target, splits in targets.items():
        report["targets"][target] = {}
        for split, records in splits.items():
            domains = _counts(records, "domain")
            sources = _counts(records, "source_dataset")
            report["targets"][target][split] = {
                "total_examples": len(records),
                "domain_counts": domains,
                "domain_percentages": _percentages(domains, len(records)),
                "source_dataset_counts": sources,
                "source_dataset_percentages": _percentages(sources, len(records)),
                "domain_x_source_dataset_counts": _cross_counts(records),
                "representative_examples": _representatives(records),
            }
    return report


def build_overlap_statistics(pools, targets) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "pool_train_vs_validation": {},
        "target_grad_vs_validation": {},
        "target_vs_general": {},
    }
    for pool, splits in pools.items():
        report["pool_train_vs_validation"][pool] = {
            "id_overlap": len(
                {row["id"] for row in splits["train"]}
                & {row["id"] for row in splits["val"]}
            ),
            "normalized_prompt_overlap": len(
                {row["prompt_hash"] for row in splits["train"]}
                & {row["prompt_hash"] for row in splits["val"]}
            ),
        }
    for target, splits in targets.items():
        report["target_grad_vs_validation"][target] = {
            "id_overlap": len(
                {row["id"] for row in splits["grad"]}
                & {row["id"] for row in splits["val"]}
            ),
            "normalized_prompt_overlap": len(
                {row["prompt_hash"] for row in splits["grad"]}
                & {row["prompt_hash"] for row in splits["val"]}
            ),
        }
        target_ids = {row["id"] for rows in splits.values() for row in rows}
        target_prompts = {
            row["prompt_hash"] for rows in splits.values() for row in rows
        }
        report["target_vs_general"][target] = {}
        for pool, pool_splits in pools.items():
            general = [row for rows in pool_splits.values() for row in rows]
            report["target_vs_general"][target][pool] = {
                "id_overlap": len(target_ids & {row["id"] for row in general}),
                "normalized_prompt_overlap": len(
                    target_prompts & {row["prompt_hash"] for row in general}
                ),
            }
    instruction = [
        row for rows in pools["instruction_32k"].values() for row in rows
    ]
    reasoning = [
        row for rows in pools["reasoning_32k"].values() for row in rows
    ]
    report["instruction_vs_reasoning_parents"] = {
        "id_overlap": len(
            {row["id"] for row in instruction} & {row["id"] for row in reasoning}
        ),
        "normalized_prompt_overlap": len(
            {row["prompt_hash"] for row in instruction}
            & {row["prompt_hash"] for row in reasoning}
        ),
    }
    return report


class Dolci32KBuilder:
    def __init__(
        self,
        data_dir: str | os.PathLike[str],
        *,
        seed: int = SEED,
        sizes: BuildSizes = DEFAULT_SIZES,
        loaders: Optional[PinnedLoaders] = None,
        strict_inventory: bool = True,
        reasoning_quotas: Mapping[str, Mapping[str, Mapping[str, int]]] = REASONING_SOURCE_QUOTAS,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.root = dolci32k_root(self.data_dir)
        self.seed = int(seed)
        if self.seed != SEED:
            raise ValueError(f"The {PROFILE_NAME} contract fixes seed={SEED}; got {self.seed}")
        self.sizes = sizes
        self.loaders = loaders or PinnedLoaders()
        self.strict_inventory = bool(strict_inventory)
        self.reasoning_quotas = {
            str(category): {
                str(source): {"train": int(counts["train"]), "val": int(counts["val"])}
                for source, counts in sources.items()
            }
            for category, sources in reasoning_quotas.items()
        }
        self.rejects: collections.Counter[str] = collections.Counter()
        self.quotas: Dict[str, Any] = {}
        self.source_inventory: Dict[str, Any] = {}

    def _convert_all(self, rows, converter, reject_key: str, **kwargs):
        converted = []
        for index, row in enumerate(rows):
            record = converter(row, index, **kwargs)
            if record is None:
                self.rejects[f"{reject_key}:unusable"] += 1
            else:
                converted.append(record)
        return converted

    def _build_benchmarks(self) -> Dict[str, List[Dict[str, Any]]]:
        specs = (
            ("ifeval", self.loaders.ifeval(), convert_if_benchmark, {"task": "ifeval"}),
            ("ifbench", self.loaders.ifbench(), convert_if_benchmark, {"task": "ifbench"}),
            ("math500", self.loaders.math500(), convert_math500, {}),
            ("mbpp_plus", self.loaders.mbpp_plus(), convert_mbpp_plus, {}),
        )
        benchmarks: Dict[str, List[Dict[str, Any]]] = {}
        inventory: Dict[str, Dict[str, int]] = {}
        for task, rows, converter, kwargs in specs:
            before = self.rejects[f"{task}:unusable"]
            converted = self._convert_all(rows, converter, task, **kwargs)
            rejected = self.rejects[f"{task}:unusable"] - before
            if rejected or not converted:
                raise RuntimeError(
                    f"Pinned {task} benchmark did not materialize completely: "
                    f"converted={len(converted)}, rejected={rejected}"
                )
            benchmarks[task] = converted
            inventory[task] = {"materialized": len(converted), "rejected": rejected}
        self.source_inventory["benchmarks"] = inventory
        return benchmarks

    def _dolci_precise_records(self) -> List[Dict[str, Any]]:
        precise: List[Dict[str, Any]] = []
        domains: collections.Counter[str] = collections.Counter()
        sources: collections.Counter[str] = collections.Counter()
        cross: Dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
        representatives: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for index, row in enumerate(self.loaders.dolci()):
            domain, source = str(row.get("domain")), str(row.get("source_dataset"))
            validate_dolci_pair(domain, source)
            domains[domain] += 1
            sources[source] += 1
            cross[domain][source] += 1
            pair = f"{domain} :: {source}"
            if len(representatives[pair]) < 2:
                representatives[pair].append({
                    "index": index, "id": row.get("id"),
                    "messages": list(row.get("messages") or [])[:2],
                })
            if source != PRECISE_IF_SOURCE:
                continue
            record = convert_dolci(row, index)
            if record is None:
                self.rejects["dolci:precise_if_unusable"] += 1
            else:
                precise.append(record)
        inventory = {
            "total_examples": sum(domains.values()),
            "domain_counts": dict(sorted(domains.items())),
            "source_dataset_counts": dict(sorted(sources.items())),
            "domain_x_source_dataset_counts": {
                domain: dict(sorted(values.items())) for domain, values in sorted(cross.items())
            },
            "representative_examples": dict(sorted(representatives.items())),
        }
        if self.strict_inventory:
            expected_cross = {
                domain: dict(values)
                for domain, values in EXPECTED_DOMAIN_SOURCE_COUNTS.items()
            }
            failures = {
                "total_examples": inventory["total_examples"] != EXPECTED_TOTAL_ROWS,
                "domain_counts": inventory["domain_counts"] != dict(EXPECTED_DOMAIN_COUNTS),
                "source_dataset_counts": inventory["source_dataset_counts"] != dict(sorted(EXPECTED_SOURCE_COUNTS.items())),
                "domain_x_source_dataset_counts": inventory["domain_x_source_dataset_counts"] != expected_cross,
            }
            if any(failures.values()):
                raise RuntimeError(
                    "Pinned Dolci metadata inventory drifted from the reviewed contract: "
                    f"{[key for key, failed in failures.items() if failed]}"
                )
        self.source_inventory["dolci"] = inventory
        return precise

    def _build_targets(
        self, benchmarks: Mapping[str, Sequence[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        final_index = PromptDecontaminator(references_from_records(benchmarks))
        precise_grad, precise_val = select_target_pair(
            self._dolci_precise_records(),
            namespace="target:precise_if", blocker=final_index,
            grad_count=self.sizes.target_grad, val_count=self.sizes.target_val,
        )
        math_records = []
        for index, row in enumerate(self.loaders.math_train()):
            config = str(row.get("_dolci32k_config", "default"))
            record = convert_math_train(row, index, config=config)
            if record is None:
                self.rejects["math_target:unusable"] += 1
            else:
                math_records.append(record)
        math_grad, math_val = select_target_pair(
            math_records, namespace="target:math", blocker=final_index,
            grad_count=self.sizes.target_grad, val_count=self.sizes.target_val,
        )
        mbpp_records = self._convert_all(
            self.loaders.mbpp_train(), convert_mbpp_train, "mbpp_target"
        )
        blocked_mbpp = mbpp_plus_task_numbers(benchmarks["mbpp_plus"])
        eligible_mbpp = []
        for record in mbpp_records:
            if mbpp_task_number((record.get("metadata") or {}).get("task_id")) in blocked_mbpp:
                self.rejects["mbpp_target:mbpp_plus_task_id_overlap"] += 1
            else:
                eligible_mbpp.append(record)
        mbpp_grad, mbpp_val = select_target_pair(
            eligible_mbpp, namespace="target:mbpp", blocker=final_index,
            grad_count=self.sizes.target_grad, val_count=self.sizes.target_val,
        )
        raw = {
            "precise_if": {"grad": precise_grad, "val": precise_val},
            "math": {"grad": math_grad, "val": math_val},
            "mbpp": {"grad": mbpp_grad, "val": mbpp_val},
        }
        return {
            target: {
                split: [
                    _annotate(row, pool=target, split=split, role="target")
                    for row in records
                ]
                for split, records in splits.items()
            }
            for target, splits in raw.items()
        }

    @staticmethod
    def _open_candidate_db(path: Path, table: str) -> sqlite3.Connection:
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            f"CREATE TABLE {table} ("
            "dedup_key TEXT PRIMARY KEY, group_name TEXT NOT NULL, "
            "sample_rank TEXT NOT NULL, record_id TEXT NOT NULL, record_json TEXT NOT NULL)"
        )
        connection.execute(
            f"CREATE INDEX {table}_group_rank ON {table}(group_name, sample_rank, record_id)"
        )
        return connection

    @staticmethod
    def _temporary_candidate_db(label: str) -> Path:
        """Allocate an ephemeral SQLite file on node-local temporary storage."""

        descriptor, raw_path = tempfile.mkstemp(
            prefix=f"dolci32k-{label}-", suffix=".sqlite"
        )
        os.close(descriptor)
        return Path(raw_path)

    @staticmethod
    def _insert_candidate(
        connection: sqlite3.Connection, table: str, record: Mapping[str, Any],
        group: str, namespace: str,
    ) -> None:
        rank = stable_rank(namespace, group, record["prompt_hash"], record["id"])
        connection.execute(
            f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?) "
            f"ON CONFLICT(dedup_key) DO UPDATE SET "
            "group_name=excluded.group_name,sample_rank=excluded.sample_rank,"
            "record_id=excluded.record_id,record_json=excluded.record_json "
            f"WHERE excluded.sample_rank < {table}.sample_rank",
            (record["prompt_hash"], group, rank, record["id"], canonical_json(record)),
        )

    @staticmethod
    def _take_group(
        connection: sqlite3.Connection, table: str, group: str, count: int,
        excluded_prompt_hashes: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        cursor = connection.execute(
            f"SELECT record_json FROM {table} WHERE group_name=? ORDER BY sample_rank, record_id",
            (group,),
        )
        records: List[Dict[str, Any]] = []
        ids: set[str] = set()
        excluded_prompt_hashes = excluded_prompt_hashes or set()
        for (record_json,) in cursor:
            record = json.loads(record_json)
            if record["id"] in ids or record["prompt_hash"] in excluded_prompt_hashes:
                continue
            ids.add(record["id"])
            records.append(record)
            if len(records) == count:
                break
        if len(records) != count:
            raise RuntimeError(f"{table}/{group}: available unique={len(records)}, need={count}")
        return records

    @staticmethod
    def _cleanup_db(connection: sqlite3.Connection, db_path: Path) -> None:
        connection.close()
        for suffix in ("", "-wal", "-shm"):
            path = Path(str(db_path) + suffix)
            if path.exists():
                path.unlink()

    def _build_reasoning(
        self, staging: Path, protected: PromptDecontaminator,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        train_required = sum(
            counts["train"] for sources in self.reasoning_quotas.values()
            for counts in sources.values()
        )
        val_required = sum(
            counts["val"] for sources in self.reasoning_quotas.values()
            for counts in sources.values()
        )
        if train_required != self.sizes.general_train or val_required != self.sizes.general_val:
            raise RuntimeError(
                "Reasoning per-source quotas do not sum to configured pool sizes: "
                f"train={train_required}/{self.sizes.general_train}, "
                f"val={val_required}/{self.sizes.general_val}"
            )
        db_path = self._temporary_candidate_db("reasoning")
        connection = self._open_candidate_db(db_path, "candidate")
        try:
            for index, row in enumerate(self.loaders.dolci()):
                source = str(row.get("source_dataset"))
                validate_dolci_pair(row.get("domain"), source)
                if source not in REASONING_SOURCES:
                    continue
                record = convert_dolci(row, index)
                if record is None:
                    self.rejects["reasoning:unusable"] += 1
                    continue
                if protected.blocked_record(record):
                    self.rejects["reasoning:protected_prompt"] += 1
                    continue
                category = REASONING_SOURCE_CATEGORY[source]
                record["reasoning_category"] = category
                record["reasoning_stratum"] = f"{category}::{source}"
                record["metadata"]["reasoning_category"] = category
                self._insert_candidate(
                    connection, "candidate", record, source, "reasoning:source"
                )
            connection.commit()
            available = dict(connection.execute(
                "SELECT group_name, COUNT(*) FROM candidate GROUP BY group_name"
            ).fetchall())
            train: List[Dict[str, Any]] = []
            val: List[Dict[str, Any]] = []
            used_prompts: set[str] = set()
            selected_counts: Dict[str, int] = {}
            for category, source_quotas in sorted(self.reasoning_quotas.items()):
                for source, split_quota in sorted(source_quotas.items()):
                    total = split_quota["train"] + split_quota["val"]
                    records = self._take_group(
                        connection, "candidate", source, total,
                        excluded_prompt_hashes=used_prompts,
                    )
                    used_prompts.update(row["prompt_hash"] for row in records)
                    split_order = sorted(
                        records,
                        key=lambda row: (
                            stable_rank("reasoning:split", source, row["id"]),
                            row["id"],
                        ),
                    )
                    source_val = split_order[:split_quota["val"]]
                    source_train = split_order[split_quota["val"]:]
                    train.extend(
                        _annotate(row, pool="reasoning_32k", split="train", role="general")
                        for row in source_train
                    )
                    val.extend(
                        _annotate(row, pool="reasoning_32k", split="val", role="general")
                        for row in source_val
                    )
                    selected_counts[source] = len(records)
            self.quotas["reasoning"] = {
                "policy": "exact_per_source_without_replacement",
                "available_unique_prompts": dict(sorted(available.items())),
                "selected_combined": dict(sorted(selected_counts.items())),
                "per_source": self.reasoning_quotas,
            }
            train.sort(key=lambda row: (stable_rank("reasoning:output:train", row["id"]), row["id"]))
            val.sort(key=lambda row: (stable_rank("reasoning:output:val", row["id"]), row["id"]))
            return train, val
        finally:
            self._cleanup_db(connection, db_path)

    def _build_instruction(
        self,
        staging: Path,
        protected: PromptDecontaminator,
        *,
        reserved_target_ids: set[str],
        reserved_reasoning_ids: set[str],
        reserved_reasoning_prompts: set[str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        db_path = self._temporary_candidate_db("instruction")
        connection = self._open_candidate_db(db_path, "candidate")
        try:
            for index, row in enumerate(self.loaders.dolci()):
                source = str(row.get("source_dataset"))
                validate_dolci_pair(row.get("domain"), source)
                if source not in INSTRUCTION_SOURCES:
                    self.rejects["instruction:excluded_domain_or_source"] += 1
                    continue
                record = convert_dolci(row, index)
                if record is None:
                    self.rejects["instruction:unusable"] += 1
                    continue
                if has_structural_tool_use(row, record):
                    self.rejects["instruction:structural_tool_use"] += 1
                    continue
                if record["id"] in reserved_target_ids:
                    self.rejects["instruction:reserved_target_id"] += 1
                    continue
                if record["id"] in reserved_reasoning_ids:
                    self.rejects["instruction:reserved_reasoning_id"] += 1
                    continue
                if record["prompt_hash"] in reserved_reasoning_prompts:
                    self.rejects["instruction:reserved_reasoning_prompt"] += 1
                    continue
                if protected.blocked_record(record):
                    self.rejects["instruction:protected_prompt"] += 1
                    continue
                self._insert_candidate(
                    connection, "candidate", record, source, "instruction:source"
                )
            connection.commit()
            available = dict(connection.execute(
                "SELECT group_name, COUNT(*) FROM candidate GROUP BY group_name"
            ).fetchall())
            missing = set(INSTRUCTION_SOURCES) - set(available)
            if missing:
                raise RuntimeError(
                    f"Instruction sources have no eligible records after exclusions: {sorted(missing)}"
                )
            combined_total = self.sizes.general_train + self.sizes.general_val
            combined_quota = capped_equal_waterfill(available, combined_total)
            val_quota = largest_remainder(combined_quota, self.sizes.general_val)
            train_quota = {
                source: combined_quota[source] - val_quota[source]
                for source in combined_quota
            }
            train: List[Dict[str, Any]] = []
            val: List[Dict[str, Any]] = []
            used_prompts: set[str] = set()
            for source in sorted(combined_quota):
                count = combined_quota[source]
                if not count:
                    continue
                records = self._take_group(
                    connection, "candidate", source, count,
                    excluded_prompt_hashes=used_prompts,
                )
                used_prompts.update(row["prompt_hash"] for row in records)
                split_order = sorted(
                    records,
                    key=lambda row: (
                        stable_rank("instruction:split", source, row["id"]),
                        row["id"],
                    ),
                )
                source_val = split_order[:val_quota[source]]
                source_train = split_order[val_quota[source]:]
                train.extend(
                    _annotate(row, pool="instruction_32k", split="train", role="general")
                    for row in source_train
                )
                val.extend(
                    _annotate(row, pool="instruction_32k", split="val", role="general")
                    for row in source_val
                )
            self.quotas["instruction"] = {
                "policy": "capped_equal_waterfill_combined_then_stratified_validation",
                "available_unique_prompts": dict(sorted(available.items())),
                "combined": dict(sorted(combined_quota.items())),
                "train": dict(sorted(train_quota.items())),
                "val": dict(sorted(val_quota.items())),
            }
            train.sort(key=lambda row: (stable_rank("instruction:output:train", row["id"]), row["id"]))
            val.sort(key=lambda row: (stable_rank("instruction:output:val", row["id"]), row["id"]))
            return train, val
        finally:
            self._cleanup_db(connection, db_path)


    def _build_mixed(
        self,
        instruction_train: Sequence[Dict[str, Any]],
        instruction_val: Sequence[Dict[str, Any]],
        reasoning_train: Sequence[Dict[str, Any]],
        reasoning_val: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        results: Dict[str, List[Dict[str, Any]]] = {}
        mixed_quotas: Dict[str, Dict[str, int]] = {}
        for split, instruction_parent, reasoning_parent, instruction_count, reasoning_count in (
            (
                "train", instruction_train, reasoning_train,
                self.sizes.mixed_instruction_train, self.sizes.mixed_reasoning_train,
            ),
            (
                "val", instruction_val, reasoning_val,
                self.sizes.mixed_instruction_val, self.sizes.mixed_reasoning_val,
            ),
        ):
            instruction_quota = _proportional_quota(
                instruction_parent, instruction_count, "source_dataset"
            )
            reasoning_quota = _proportional_quota(
                reasoning_parent, reasoning_count, "reasoning_stratum"
            )
            selected_instruction = _nested_subset(
                instruction_parent, instruction_quota,
                group_key="source_dataset", namespace=f"mixed:instruction:{split}",
            )
            selected_reasoning = _nested_subset(
                reasoning_parent, reasoning_quota,
                group_key="reasoning_stratum", namespace=f"mixed:reasoning:{split}",
            )
            mixed_instruction = [
                _annotate(
                    row, pool="mixed_32k", split=split, role="general",
                    parent_pool="instruction_32k",
                )
                for row in selected_instruction
            ]
            mixed_reasoning = [
                _annotate(
                    row, pool="mixed_32k", split=split, role="general",
                    parent_pool="reasoning_32k",
                )
                for row in selected_reasoning
            ]
            if (
                _counts(instruction_parent, "source_dataset").get(PRECISE_IF_SOURCE, 0)
                and not _counts(mixed_instruction, "source_dataset").get(PRECISE_IF_SOURCE, 0)
            ):
                raise AssertionError(f"Mixed {split} accidentally eliminated Precise-IF")
            combined = mixed_instruction + mixed_reasoning
            combined.sort(
                key=lambda row: (stable_rank(f"mixed:output:{split}", row["id"]), row["id"])
            )
            results[split] = combined
            mixed_quotas[f"instruction_{split}"] = dict(sorted(instruction_quota.items()))
            mixed_quotas[f"reasoning_{split}"] = dict(sorted(reasoning_quota.items()))
        self.quotas["mixed"] = {
            **mixed_quotas,
            "policy": "nested_parent_subset_with_largest_remainder_stratification",
        }
        return results["train"], results["val"]

    def _write_artifact(
        self, staging: Path, role: str, name: str,
        records: Sequence[Mapping[str, Any]], lineage: Optional[Sequence[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        relative = artifact_relative_path(role, name)
        path = staging / relative
        count = atomic_write_jsonl(path, records)
        return str(relative), {
            "count": count,
            "sha256": file_sha256(path),
            "source_dataset_counts": _counts(records, "source_dataset"),
            "domain_counts": _counts(records, "domain"),
            "domain_x_source_dataset_counts": _cross_counts(records),
            "lineage": list(lineage or []),
            "tokenization": {
                "status": "not_applied_to_raw_membership",
                "model_independent": True,
            },
        }

    def _write_candidate_order(
        self, staging: Path, pool: str, records: Sequence[Mapping[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        ordered = sorted(
            records,
            key=lambda row: (
                stable_rank(f"candidate_order:{pool}", row["id"], seed=CANDIDATE_SEED),
                row["id"],
            ),
        )
        rows = [
            {"position": position, "id": str(record["id"])}
            for position, record in enumerate(ordered)
        ]
        relative = artifact_relative_path("candidate_orders", pool)
        path = staging / relative
        count = atomic_write_jsonl(path, rows)
        return str(relative), {
            "count": count,
            "sha256": file_sha256(path),
            "candidate_seed": CANDIDATE_SEED,
            "ordered_id_sha256": ordered_id_sha256(row["id"] for row in rows),
            "without_replacement": True,
            "candidate_batch_size": CANDIDATE_BATCH_SIZE,
            "formal_steps": count // CANDIDATE_BATCH_SIZE,
        }

    def build(self) -> Path:
        self.rejects.clear()
        self.quotas.clear()
        self.source_inventory.clear()
        if self.sizes != DEFAULT_SIZES:
            raise ValueError("Promoted Dolci32K builds must use the versioned default sizes")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "builds").mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".build.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.root))
            try:
                benchmarks = self._build_benchmarks()
                targets = self._build_targets(benchmarks)
                protected = _protected_index(benchmarks, targets)

                # The parent-build order is contractual: reasoning membership
                # is finalized first, then exact IDs/prompts are unavailable to
                # the instruction water-fill even for shared eligible sources.
                reasoning_train, reasoning_val = self._build_reasoning(staging, protected)
                reasoning_all = reasoning_train + reasoning_val
                target_ids = {
                    row["id"] for splits in targets.values()
                    for rows in splits.values() for row in rows
                }
                instruction_train, instruction_val = self._build_instruction(
                    staging, protected,
                    reserved_target_ids=target_ids,
                    reserved_reasoning_ids={row["id"] for row in reasoning_all},
                    reserved_reasoning_prompts={row["prompt_hash"] for row in reasoning_all},
                )
                mixed_train, mixed_val = self._build_mixed(
                    instruction_train, instruction_val, reasoning_train, reasoning_val
                )
                pools = {
                    "instruction_32k": {"train": instruction_train, "val": instruction_val},
                    "reasoning_32k": {"train": reasoning_train, "val": reasoning_val},
                    "mixed_32k": {"train": mixed_train, "val": mixed_val},
                }
                audit = audit_pools(
                    pools, targets, benchmarks, sizes=self.sizes,
                    quotas=self.quotas, reasoning_quotas=self.reasoning_quotas,
                )

                artifacts: Dict[str, Any] = {}
                for pool, splits in pools.items():
                    for split, records in splits.items():
                        lineage: List[str] = []
                        if pool == "mixed_32k":
                            lineage = [
                                f"general/instruction_32k/{split}.jsonl",
                                f"general/reasoning_32k/{split}.jsonl",
                            ]
                        relative, entry = self._write_artifact(
                            staging, "general", f"{pool}/{split}", records, lineage
                        )
                        artifacts[relative] = entry
                    relative, entry = self._write_candidate_order(
                        staging, pool, splits["train"]
                    )
                    artifacts[relative] = entry
                for target, splits in targets.items():
                    for split, records in splits.items():
                        relative, entry = self._write_artifact(
                            staging, "targets", f"{target}/{split}", records
                        )
                        artifacts[relative] = entry
                for benchmark, records in benchmarks.items():
                    relative, entry = self._write_artifact(
                        staging, "benchmarks", benchmark, records
                    )
                    entry["tokenization"] = {
                        "status": "generation_only", "truncation": "not_applied"
                    }
                    artifacts[relative] = entry

                statistics = build_pool_statistics(pools, targets)
                statistics.update({
                    "source_inventory": self.source_inventory.get("dolci", {}),
                    "overlap_checks": {
                        "status": "passed",
                        "checks": audit["checks"],
                        "counts": build_overlap_statistics(pools, targets),
                    },
                    "candidate_traversal": {
                        "candidate_batch_size": CANDIDATE_BATCH_SIZE,
                        "selected_subset_size": SELECTED_SUBSET_SIZE,
                        "formal_steps": FORMAL_STEPS,
                        "candidate_exposures": FORMAL_STEPS * CANDIDATE_BATCH_SIZE,
                        "without_replacement": True,
                    },
                })
                statistics_path = staging / "pool_statistics.json"
                atomic_write_json(statistics_path, statistics)
                metadata_files = {
                    "pool_statistics.json": {
                        "sha256": file_sha256(statistics_path),
                    }
                }

                manifest_core = {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "profile_name": PROFILE_NAME,
                    "profile_version": PROFILE_VERSION,
                    "profile_fingerprint": profile_fingerprint(),
                    "seed": self.seed,
                    "candidate_seed": CANDIDATE_SEED,
                    "model_independent_raw_membership": True,
                    "raw_membership_policy": {
                        "tokenizer_filtering": False,
                        "length_filtering": False,
                        "derived_tokenization_caches_may_not_change_membership": True,
                    },
                    "requested_sizes": {
                        "general_train": self.sizes.general_train,
                        "general_val": self.sizes.general_val,
                        "target_grad": self.sizes.target_grad,
                        "target_val": self.sizes.target_val,
                        "mixed_instruction_train": self.sizes.mixed_instruction_train,
                        "mixed_reasoning_train": self.sizes.mixed_reasoning_train,
                        "mixed_instruction_val": self.sizes.mixed_instruction_val,
                        "mixed_reasoning_val": self.sizes.mixed_reasoning_val,
                    },
                    "sources": {
                        key: {
                            "repo": pin.repo, "revision": pin.revision,
                            "config": pin.config, "split": pin.split,
                        }
                        for key, pin in sorted(PINNED_SOURCES.items())
                    },
                    "evaluator_pins": {
                        "ifeval_git": IFEVAL_EVALUATOR_GIT_COMMIT,
                        "ifbench_git": IFBENCH_EVALUATOR_GIT_COMMIT,
                        "evalplus_version": EVALPLUS_VERSION,
                        "evalplus_dataset_version": EVALPLUS_DATASET_VERSION,
                        "evalplus_image": EVALPLUS_IMAGE,
                        "math_verify_version": MATH_VERIFY_VERSION,
                    },
                    "settings": {key: dict(value) for key, value in SETTINGS.items()},
                    "quotas": self.quotas,
                    "source_inventory": self.source_inventory,
                    "reject_counts": dict(sorted(self.rejects.items())),
                    "candidate_traversal": statistics["candidate_traversal"],
                    "artifacts": artifacts,
                    "metadata_files": metadata_files,
                    "audit": audit,
                    "audit_status": "passed",
                }
                build_id = hashlib.sha256(
                    canonical_json(manifest_core).encode("utf-8")
                ).hexdigest()
                atomic_write_json(
                    staging / "manifest.json",
                    dict(manifest_core, build_id=build_id),
                )
                validate_build(staging, expected_build_id=build_id)

                destination = self.root / "builds" / build_id
                if destination.exists():
                    validate_build(destination)
                    shutil.rmtree(staging)
                else:
                    os.replace(staging, destination)
                    validate_build(destination)
                current_tmp = self.root / f".CURRENT.{os.getpid()}.tmp"
                current_tmp.write_text(build_id + "\n", encoding="utf-8")
                os.replace(current_tmp, self.root / "CURRENT")
                return destination
            except BaseException:
                if staging.exists():
                    shutil.rmtree(staging)
                raise


def reaudit_build(
    build: str | os.PathLike[str], *, sizes: BuildSizes = DEFAULT_SIZES,
) -> Dict[str, Any]:
    """Reload immutable files and recompute all data-leakage assertions."""
    build = Path(build)
    manifest = validate_build(build)

    def load(role: str, name: str) -> List[Dict[str, Any]]:
        return list(read_jsonl(build / artifact_relative_path(role, name)))

    pools = {
        pool: {
            split: load("general", f"{pool}/{split}")
            for split in ("train", "val")
        }
        for pool in GENERAL_POOLS
    }
    targets = {
        target: {
            split: load("targets", f"{target}/{split}")
            for split in ("grad", "val")
        }
        for target in TARGETS
    }
    benchmarks = {task: load("benchmarks", task) for task in BENCHMARKS}
    recorded_quotas = manifest.get("quotas", {})
    reasoning_quotas = (
        (recorded_quotas.get("reasoning") or {}).get("per_source")
        or REASONING_SOURCE_QUOTAS
    )
    audit = audit_pools(
        pools, targets, benchmarks, sizes=sizes,
        quotas=recorded_quotas, reasoning_quotas=reasoning_quotas,
    )
    recorded_checks = (manifest.get("audit") or {}).get("checks", [])
    return {
        "status": "passed",
        "build_id": manifest["build_id"],
        "build_dir": str(build),
        "source": "recomputed_from_artifact_files",
        "checks": audit["checks"],
        "checks_added_since_build": sorted(set(audit["checks"]) - set(recorded_checks)),
        "decontamination": audit["decontamination"],
    }
