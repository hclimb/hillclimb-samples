"""Reviewed metadata, pins, and sizes for the Dolci-Instruct 32K profile.

The raw artifact contract is deliberately model-independent.  Model/tokenizer
profiles live in the versioned YAML for derived tokenization and runtime
validation, but are omitted from :func:`raw_profile_config` and therefore from
raw build membership fingerprints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "dolci32k.yaml"


def load_profile_config(path: str | Path = CONFIG_PATH) -> Dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Dolci32K config is not a mapping: {path}")
    return payload


CONFIG = load_profile_config()
PROFILE_NAME = str(CONFIG["profile_name"])
PROFILE_VERSION = int(CONFIG["profile_version"])
SEED = int(CONFIG["seed"])
CANDIDATE_SEED = int(CONFIG["candidate_seed"])
MAX_SEQ_LEN = int(CONFIG["max_seq_len"])
TOKENIZER_USE_FAST = bool(CONFIG["tokenizer_use_fast"])
TRUNCATION_WARNING_THRESHOLD = float(CONFIG["truncation_warning_threshold"])
CANDIDATE_BATCH_SIZE = int(CONFIG["candidate_traversal"]["candidate_batch_size"])
SELECTED_SUBSET_SIZE = int(CONFIG["candidate_traversal"]["selected_subset_size"])
FORMAL_STEPS = int(CONFIG["candidate_traversal"]["formal_steps"])
WORD_NGRAM_SIZE = int(CONFIG["decontamination"]["word_ngram_size"])
NEAR_DUPLICATE_THRESHOLD = float(
    CONFIG["decontamination"]["near_duplicate_threshold"]
)


@dataclass(frozen=True)
class SourcePin:
    key: str
    repo: str
    revision: str
    split: str
    config: Optional[str] = None


PINNED_SOURCES: Mapping[str, SourcePin] = {
    "dolci": SourcePin(
        "dolci",
        str(CONFIG["dataset"]["repo"]),
        str(CONFIG["dataset"]["revision"]),
        str(CONFIG["dataset"]["split"]),
    ),
    **{
        key: SourcePin(key, str(values["repo"]), str(values["revision"]),
                       str(values["split"]), None if values.get("config") is None else str(values["config"]))
        for key, values in CONFIG["sources"].items()
    },
}

MATH_TRAIN_CONFIGS: Tuple[str, ...] = (
    str(PINNED_SOURCES["math_train"].config or "default"),
)
EVALPLUS_PACKAGE = str(CONFIG["evaluators"]["evalplus_package"])
EVALPLUS_VERSION = str(CONFIG["evaluators"]["evalplus_version"])
EVALPLUS_DATASET_VERSION = str(CONFIG["evaluators"]["evalplus_dataset_version"])
EVALPLUS_GIT_COMMIT = str(CONFIG["evaluators"]["evalplus_git"])
EVALPLUS_IMAGE = str(CONFIG["evaluators"]["evalplus_image"])
MATH_VERIFY_VERSION = str(CONFIG["evaluators"]["math_verify_version"])
IFBENCH_EVALUATOR_REPO = str(CONFIG["evaluators"]["ifbench_repo"])
IFBENCH_EVALUATOR_GIT_COMMIT = str(CONFIG["evaluators"]["ifbench_git"])
IFEVAL_EVALUATOR_GIT_COMMIT = str(CONFIG["evaluators"]["ifeval_git"])

BENCHMARK_DEFAULTS: Mapping[str, Mapping[str, object]] = {
    "ifeval": {"max_new_tokens": 2048, "temperature": 0.0, "split": "full"},
    "ifbench": {"max_new_tokens": 2048, "temperature": 0.0, "split": "full"},
    "math500": {"max_new_tokens": 4096, "temperature": 0.0, "split": "full"},
    "mbpp_plus": {"max_new_tokens": 2048, "temperature": 0.0, "split": "full"},
}


# Full, independently verified inventory at the pinned Dolci revision.
EXPECTED_DOMAIN_SOURCE_COUNTS: Mapping[str, Mapping[str, int]] = {
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
    "Safety": {"WildJailbreak": 49_965, "WildGuardMix": 49_373, "CoCoNot": 10_957},
    "Science": {
        "Dolci Instruct OpenThoughts3+ Science": 99_268,
        "SciRiff": 4_557,
    },
    "Tool Use": {"Dolci Instruct Tool Use": 227_579},
}
EXPECTED_DOMAIN_COUNTS: Mapping[str, int] = {
    domain: sum(sources.values())
    for domain, sources in EXPECTED_DOMAIN_SOURCE_COUNTS.items()
}
EXPECTED_SOURCE_COUNTS: Mapping[str, int] = {
    source: count
    for sources in EXPECTED_DOMAIN_SOURCE_COUNTS.values()
    for source, count in sources.items()
}
EXPECTED_TOTAL_ROWS = sum(EXPECTED_DOMAIN_COUNTS.values())
DOLCI_SOURCE_DOMAINS: Mapping[str, str] = {
    source: domain
    for domain, sources in EXPECTED_DOMAIN_SOURCE_COUNTS.items()
    for source in sources
}
DOLCI_KNOWN_SOURCES: Tuple[str, ...] = tuple(sorted(DOLCI_SOURCE_DOMAINS))
DOLCI_DOMAIN_MAP: Mapping[str, str] = {
    "Chat": "chat",
    "Coding": "coding",
    "Hardcoded Data": "hardcoded_data",
    "Math": "math",
    "Multilingual": "multilingual",
    "Other": "other",
    "Precise IF": "precise_if",
    "Reasoning": "reasoning",
    "Safety": "safety",
    "Science": "science",
    "Tool Use": "tool_use",
}

PRECISE_IF_SOURCE = str(CONFIG["targets"]["precise_if_source"])
INSTRUCTION_EXCLUDED_DOMAINS: Tuple[str, ...] = tuple(CONFIG["instruction"]["exclude_domains"])
INSTRUCTION_SOURCES: Tuple[str, ...] = tuple(
    sorted(source for source, domain in DOLCI_SOURCE_DOMAINS.items()
           if domain not in INSTRUCTION_EXCLUDED_DOMAINS)
)
# Backward-friendly explicit name used by data consumers.
DOLCI_ELIGIBLE_SOURCES = INSTRUCTION_SOURCES

REASONING_SOURCE_QUOTAS: Mapping[str, Mapping[str, Mapping[str, int]]] = {
    str(category): {
        str(source): {"train": int(counts["train"]), "val": int(counts["val"])}
        for source, counts in sources.items()
    }
    for category, sources in CONFIG["reasoning"].items()
}
REASONING_SOURCE_CATEGORY: Mapping[str, str] = {
    source: category
    for category, sources in REASONING_SOURCE_QUOTAS.items()
    for source in sources
}
REASONING_SOURCES: Tuple[str, ...] = tuple(sorted(REASONING_SOURCE_CATEGORY))
REASONING_CATEGORY_QUOTAS: Mapping[str, Mapping[str, int]] = {
    category: {
        split: sum(quota[split] for quota in sources.values())
        for split in ("train", "val")
    }
    for category, sources in REASONING_SOURCE_QUOTAS.items()
}


@dataclass(frozen=True)
class BuildSizes:
    general_train: int = int(CONFIG["sizes"]["train_pool"])
    general_val: int = int(CONFIG["sizes"]["val_pool"])
    target_grad: int = int(CONFIG["sizes"]["target_grad"])
    target_val: int = int(CONFIG["sizes"]["target_val"])
    mixed_instruction_train: int = int(CONFIG["mixed"]["train"]["instruction"])
    mixed_reasoning_train: int = int(CONFIG["mixed"]["train"]["reasoning"])
    mixed_instruction_val: int = int(CONFIG["mixed"]["val"]["instruction"])
    mixed_reasoning_val: int = int(CONFIG["mixed"]["val"]["reasoning"])


DEFAULT_SIZES = BuildSizes()
if CANDIDATE_BATCH_SIZE * FORMAL_STEPS != DEFAULT_SIZES.general_train:
    raise ValueError(
        "candidate_traversal must expose the full training pool exactly once"
    )
if not 0 < SELECTED_SUBSET_SIZE <= CANDIDATE_BATCH_SIZE:
    raise ValueError("selected_subset_size must be in [1, candidate_batch_size]")
for _target_name in ("precise_if", "math", "mbpp"):
    if int(CONFIG["targets"][f"{_target_name}_size"]) != DEFAULT_SIZES.target_grad:
        raise ValueError(
            f"targets.{_target_name}_size must equal sizes.target_grad "
            "under the shared target-loader contract"
        )
GENERAL_POOLS: Tuple[str, ...] = ("instruction_32k", "reasoning_32k", "mixed_32k")
# Accept the old spelling only at API boundaries; manifests always use canonical names.
GENERAL_POOL_ALIASES: Mapping[str, str] = {"instruct_32k": "instruction_32k"}
TARGETS: Tuple[str, ...] = ("precise_if", "math", "mbpp")
BENCHMARKS: Tuple[str, ...] = ("ifeval", "ifbench", "math500", "mbpp_plus")
SETTINGS: Mapping[str, Mapping[str, object]] = {
    "inst_if": {"general_pool": "instruction_32k", "target": "precise_if", "benchmarks": ("ifeval", "ifbench")},
    "reason_math": {"general_pool": "reasoning_32k", "target": "math", "benchmarks": ("math500",)},
    "reason_code": {"general_pool": "reasoning_32k", "target": "mbpp", "benchmarks": ("mbpp_plus",)},
    "mixed_if": {"general_pool": "mixed_32k", "target": "precise_if", "benchmarks": ("ifeval", "ifbench")},
    "mixed_math": {"general_pool": "mixed_32k", "target": "math", "benchmarks": ("math500",)},
}
SETTING_ORDER: Tuple[str, ...] = tuple(SETTINGS)
ADAMW_METHODS: Tuple[str, ...] = (
    "FullTraining", "LayerwiseRaw", "LayerwiseSoft", "LayerwiseSoftP", "LayerwiseOptA",
)
MUON_METHODS: Tuple[str, ...] = (
    "FullTraining", "LayerwiseRaw", "LayerwiseSoft", "LayerwiseSoftP",
    "LayerwiseMuonSur", "LayerwiseMuonPSur", "LayerwiseMuonSatSur", "LayerwiseMuonSatPSur",
)

MODEL_PROFILES: Mapping[str, Mapping[str, str]] = {
    alias: {
        "model_name_or_path": str(values["model_repo"]),
        "model_revision": str(values["model_revision"]),
        "tokenizer_name": str(values["tokenizer_repo"]),
        "tokenizer_revision": str(values["tokenizer_revision"]),
    }
    for alias, values in CONFIG["tokenizer_profiles"].items()
}


def raw_profile_config(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Return only fields allowed to affect raw membership and build identity."""
    value = dict(config or CONFIG)
    value.pop("tokenizer_profiles", None)
    value.pop("max_seq_len", None)
    value.pop("tokenizer_use_fast", None)
    value.pop("truncation_warning_threshold", None)
    return value


def artifact_relative_path(role: str, name: str) -> Path:
    role = str(role).strip("/")
    name = str(name).strip("/")
    if role == "general":
        pool, separator, split = name.partition("/")
        pool = GENERAL_POOL_ALIASES.get(pool, pool)
        if not separator or pool not in GENERAL_POOLS or split not in ("train", "val"):
            raise KeyError(f"unknown general artifact {name!r}")
        return Path("general") / pool / f"{split}.jsonl"
    if role == "targets":
        target, separator, split = name.partition("/")
        if not separator or target not in TARGETS or split not in ("grad", "val"):
            raise KeyError(f"unknown target artifact {name!r}")
        return Path("targets") / target / f"{split}.jsonl"
    if role == "benchmarks" and name in BENCHMARKS:
        return Path("benchmarks") / f"{name}.jsonl"
    if role in ("candidate_order", "candidate_orders"):
        pool = GENERAL_POOL_ALIASES.get(name, name)
        if pool not in GENERAL_POOLS:
            raise KeyError(f"unknown candidate-order pool {name!r}")
        return Path("candidate_orders") / f"{pool}.jsonl"
    raise KeyError(f"unknown dolci32k artifact role/name: {role!r}, {name!r}")


def all_artifact_relative_paths() -> Tuple[Path, ...]:
    paths = []
    for pool in GENERAL_POOLS:
        paths.extend(artifact_relative_path("general", f"{pool}/{split}") for split in ("train", "val"))
        paths.append(artifact_relative_path("candidate_orders", pool))
    for target in TARGETS:
        paths.extend(artifact_relative_path("targets", f"{target}/{split}") for split in ("grad", "val"))
    paths.extend(artifact_relative_path("benchmarks", task) for task in BENCHMARKS)
    return tuple(paths)


def fixed_artifact_counts(sizes: BuildSizes = DEFAULT_SIZES) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for pool in GENERAL_POOLS:
        counts[str(artifact_relative_path("general", f"{pool}/train"))] = sizes.general_train
        counts[str(artifact_relative_path("general", f"{pool}/val"))] = sizes.general_val
        counts[str(artifact_relative_path("candidate_orders", pool))] = sizes.general_train
    for target in TARGETS:
        counts[str(artifact_relative_path("targets", f"{target}/grad"))] = sizes.target_grad
        counts[str(artifact_relative_path("targets", f"{target}/val"))] = sizes.target_val
    return counts
