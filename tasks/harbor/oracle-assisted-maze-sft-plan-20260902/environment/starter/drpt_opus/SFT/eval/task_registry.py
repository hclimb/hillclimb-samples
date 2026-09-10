"""Canonical downstream task metadata for immutable 32K SFT profiles."""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from SFT.eval.evalplus_contract import (
    EVALPLUS_CONTAINER_DISTRIBUTION_VERSION,
    EVALPLUS_DATASET_VERSION,
    EVALPLUS_HOST_DISTRIBUTION_VERSION,
    EVALPLUS_IMAGE,
    EVALPLUS_RELEASE_VERSION,
    MBPP_PLUS_DATASET_SHA256,
)


IMMUTABLE_32K_PROFILES: Tuple[str, ...] = ("dolci32k",)


def _profile_module(profile: str):
    if profile not in IMMUTABLE_32K_PROFILES:
        raise ValueError(
            f"unsupported immutable 32K profile {profile!r}; expected one of "
            f"{IMMUTABLE_32K_PROFILES}"
        )
    return importlib.import_module(f"SFT.data.{profile}.profile")


def _profile_settings(profile: str) -> Tuple[Mapping[str, object], Tuple[str, ...]]:
    module = _profile_module(profile)
    settings = getattr(module, "SETTINGS")
    order = tuple(getattr(module, "SETTING_ORDER"))
    if set(settings) != set(order) or len(order) != len(set(order)):
        raise ValueError(f"{profile} has an inconsistent setting registry")
    return settings, order


def _profile_methods(profile: str) -> Dict[str, Tuple[str, ...]]:
    module = _profile_module(profile)
    return {
        "adamw": tuple(getattr(module, "ADAMW_METHODS")),
        "muon": tuple(getattr(module, "MUON_METHODS")),
    }


@dataclass(frozen=True)
class TaskSpec:
    result_filename: str
    primary_metric: str
    default_max_new_tokens: int

    thinking: bool = False

@dataclass(frozen=True)
class DownstreamCell:
    setting: str
    method: str
    task: str


def profile32k_downstream_cells(profile: str, family: str) -> List[DownstreamCell]:
    """Return setting-major/method-minor/task-minor cells from one profile."""
    settings, order = _profile_settings(profile)
    try:
        methods = _profile_methods(profile)[family]
    except KeyError as exc:
        raise ValueError(f"{profile} has no optimizer family {family!r}") from exc
    return [
        DownstreamCell(setting=setting, method=method, task=task)
        for setting in order
        for method in methods
        for task in tuple(str(value) for value in settings[setting]["benchmarks"])
    ]


def profile32k_run_metadata(profile: str, setting: str) -> Tuple[str, str]:
    """Return the canonical run prefix and general-pool label for a setting."""
    settings, _ = _profile_settings(profile)
    try:
        profile_setting = settings[setting]
    except KeyError as exc:
        raise ValueError(f"{profile} has no setting {setting!r}") from exc
    return setting, str(profile_setting["general_pool"])


def profile32k_model_metadata(profile: str, alias: str) -> Dict[str, str]:
    """Return the exact runtime tuple for a profile model alias."""
    module = _profile_module(profile)
    profiles = getattr(module, "MODEL_PROFILES", None)
    if profiles is None:
        raise ValueError(f"{profile} does not define model profiles")
    try:
        metadata = {str(key): str(value) for key, value in profiles[alias].items()}
    except KeyError as exc:
        raise ValueError(
            f"{profile} has no model profile {alias!r}; expected one of "
            f"{tuple(profiles)}"
        ) from exc
    required = {
        "model_name_or_path", "model_revision", "tokenizer_name",
        "tokenizer_revision",
    }
    if required.difference(metadata):
        raise ValueError(f"{profile}/{alias} has incomplete model metadata")
    return metadata


TASK_SPECS: Dict[str, TaskSpec] = {
    "samsum": TaskSpec("samsum_results.json", "rougeL", 128),
    "tydiqa": TaskSpec("tydiqa_results.json", "f1_score", 128),
    "nq_open": TaskSpec("nq_open_results.json", "f1", 32),
    "squad": TaskSpec("squad_results.json", "f1", 32),
    "triviaqa": TaskSpec("triviaqa_results.json", "f1", 32),
    "ifeval": TaskSpec("ifeval_results.json", "prompt_level_strict_acc", 2048, False),
    "ifbench": TaskSpec("ifbench_results.json", "prompt_level_loose_acc", 2048, False),
    "math500": TaskSpec("math500_results.json", "accuracy", 4096, True),
    "mbpp_plus": TaskSpec(
        "mbpp_plus_results.json", "base_plus_extra_pass_at_1", 2048, True
    ),
}

def profile32k_evaluator_pins(profile: str) -> Dict[str, Dict[str, object]]:
    module = _profile_module(profile)
    return {
        "ifeval": {"revision": getattr(module, "IFEVAL_EVALUATOR_GIT_COMMIT")},
        "ifbench": {"revision": getattr(module, "IFBENCH_EVALUATOR_GIT_COMMIT")},
        "math500": {
            "package": "math-verify",
            "version": getattr(module, "MATH_VERIFY_VERSION"),
        },
        "mbpp_plus": {
            "package": "evalplus",
            "version": EVALPLUS_HOST_DISTRIBUTION_VERSION,
            "release": f"v{EVALPLUS_RELEASE_VERSION}",
            "container_distribution_version": EVALPLUS_CONTAINER_DISTRIBUTION_VERSION,
            "dataset_version": EVALPLUS_DATASET_VERSION,
            "dataset_jsonl_sha256": MBPP_PLUS_DATASET_SHA256,
            "image": EVALPLUS_IMAGE,
            "image_digest": EVALPLUS_IMAGE.rsplit("@", 1)[-1],
        },
    }


def tasks_for_setting(setting: str, profile: str = "dolci32k") -> Tuple[str, ...]:
    settings, _ = _profile_settings(profile)
    try:
        return tuple(str(task) for task in settings[setting]["benchmarks"])
    except KeyError as exc:
        raise ValueError(
            f"Unknown {profile} setting {setting!r}; expected one of "
            f"{sorted(settings)}"
        ) from exc


def default_max_new_tokens(task: str) -> int:
    try:
        return TASK_SPECS[task].default_max_new_tokens
    except KeyError as exc:
        raise ValueError(f"Unknown evaluation task {task!r}") from exc


def _main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=IMMUTABLE_32K_PROFILES, default="dolci32k")
    parser.add_argument("--family", choices=("adamw", "muon"))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--index", type=int)
    action.add_argument("--count", action="store_true")
    action.add_argument("--model-profile-info", metavar="ALIAS")
    args = parser.parse_args(argv)
    if args.model_profile_info is not None:
        metadata = profile32k_model_metadata(args.profile, args.model_profile_info)
        print("\t".join((
            args.model_profile_info,
            metadata["model_name_or_path"],
            metadata["model_revision"],
            metadata["tokenizer_name"],
            metadata["tokenizer_revision"],
            Path(metadata["model_name_or_path"]).name,
        )))
        return 0
    if args.family is None:
        parser.error("--family is required unless --model-profile-info is used")
    cells = profile32k_downstream_cells(args.profile, args.family)
    if args.count:
        print(len(cells))
        return 0
    indexed = enumerate(cells)
    if args.index is not None:
        if args.index < 0 or args.index >= len(cells):
            parser.error(
                f"index must be in [0, {len(cells) - 1}] for family {args.family}"
            )
        indexed = ((args.index, cells[args.index]),)
    for index, cell in indexed:
        run_prefix, general_pool = profile32k_run_metadata(args.profile, cell.setting)
        print(
            "\t".join(
                (
                    str(index),
                    cell.setting,
                    cell.method,
                    cell.task,
                    run_prefix,
                    general_pool,
                    str(default_max_new_tokens(cell.task)),
                )
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
