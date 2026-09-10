#!/usr/bin/env python
"""Inspect, build, audit, or profile immutable Dolci-Instruct 32K artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SFT.data.dolci32k.artifacts import (
    dolci32k_root,
    load_manifest,
    resolve_current_build,
    validate_build,
)
from SFT.data.dolci32k.builder import (
    Dolci32KBuilder,
    inspect_dolci_metadata,
    reaudit_build,
)
from SFT.data.dolci32k.profile import (
    GENERAL_POOLS,
    MAX_SEQ_LEN,
    MODEL_PROFILES,
    SEED,
    TARGETS,
    TOKENIZER_USE_FAST,
    TRUNCATION_WARNING_THRESHOLD,
    artifact_relative_path,
)
from SFT.data.dolci32k.sources import PinnedLoaders


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the pinned, model-independent Dolci32K SFT benchmark"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--inspect-metadata", action="store_true",
        help="Scan pinned Dolci and print domain/source/cross counts",
    )
    action.add_argument(
        "--build", action="store_true",
        help="Build, validate, and atomically promote immutable raw artifacts",
    )
    action.add_argument(
        "--audit-only", action="store_true",
        help="Validate hashes/counts of an existing immutable build",
    )
    action.add_argument(
        "--reaudit", action="store_true",
        help="Reload artifact rows and recompute all disjointness/leakage assertions",
    )
    action.add_argument(
        "--print-artifact-root", action="store_true",
        help="Print the generated-artifact root and exit",
    )
    parser.add_argument("--data-dir", default="SFT/data", help="SFT data root")
    parser.add_argument(
        "--build-id", default=None,
        help="Use this immutable build rather than the CURRENT pointer",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Fixed raw-membership seed (must be {SEED})",
    )
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache")
    parser.add_argument(
        "--profile-tokenizers", nargs="?", const="all", default=None,
        choices=("all", *tuple(MODEL_PROFILES)),
        help=(
            "After build/audit, create derived length caches for all or one "
            f"model profile at max_seq_len={MAX_SEQ_LEN}"
        ),
    )
    parser.add_argument(
        "--force-tokenizer-profile", action="store_true",
        help="Recompute matching derived tokenization caches",
    )
    return parser


def _summary(build: Path, manifest: dict) -> dict:
    statistics_path = build / "pool_statistics.json"
    return {
        "status": "passed",
        "build_id": manifest["build_id"],
        "build_dir": str(build),
        "profile": manifest["profile_name"],
        "profile_version": manifest["profile_version"],
        "seed": manifest["seed"],
        "model_independent_raw_membership": manifest[
            "model_independent_raw_membership"
        ],
        "artifact_count": len(manifest["artifacts"]),
        "statistics_path": str(statistics_path),
        "pool_statistics": json.loads(statistics_path.read_text(encoding="utf-8")),
        "audit": manifest["audit"],
    }


def _profile(
    build: Path, manifest: dict, selection: str, *,
    force: bool, cache_dir: str | None = None,
) -> dict:
    from SFT.data.dolci32k.tokenization import (
        persist_tokenization_report,
        profile_tokenizer_profiles,
    )

    artifact_paths = {}
    for pool in GENERAL_POOLS:
        for split in ("train", "val"):
            logical = f"general/{pool}/{split}"
            artifact_paths[logical] = build / artifact_relative_path(
                "general", f"{pool}/{split}"
            )
    for target in TARGETS:
        for split in ("grad", "val"):
            logical = f"targets/{target}/{split}"
            artifact_paths[logical] = build / artifact_relative_path(
                "targets", f"{target}/{split}"
            )
    aliases = tuple(MODEL_PROFILES) if selection == "all" else (selection,)
    profiles = {}
    for alias in aliases:
        profile = dict(MODEL_PROFILES[alias])
        # The profiler accepts this canonical spelling; retain the public
        # training profile's tokenizer_name field as well.
        profile.setdefault(
            "tokenizer_name_or_path",
            profile.get("tokenizer_name") or profile.get("model_name_or_path"),
        )
        profile["use_fast"] = TOKENIZER_USE_FAST
        if cache_dir:
            kwargs = dict(profile.get("from_pretrained_kwargs") or {})
            kwargs.setdefault("cache_dir", cache_dir)
            profile["from_pretrained_kwargs"] = kwargs
        profiles[alias] = profile
    report = profile_tokenizer_profiles(
        artifact_paths,
        raw_build_id=manifest["build_id"],
        tokenizer_profiles=profiles,
        cache_root=dolci32k_root(build.parents[2]) / "tokenization_cache",
        max_seq_len=MAX_SEQ_LEN,
        force=force,
        warning_threshold=TRUNCATION_WARNING_THRESHOLD,
    )
    report_path = persist_tokenization_report(
        report, data_dir=build.parents[2]
    )
    return {**report, "report_path": str(report_path)}


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.seed != SEED:
        raise SystemExit(f"dolci32k fixes --seed {SEED}; got {args.seed}")
    if args.build_id and not (args.audit_only or args.reaudit):
        raise SystemExit("--build-id is valid only with --audit-only or --reaudit")
    if args.profile_tokenizers and args.inspect_metadata:
        raise SystemExit("--profile-tokenizers requires --build, --audit-only, or --reaudit")
    if args.print_artifact_root:
        print(dolci32k_root(args.data_dir))
        return 0
    if args.inspect_metadata:
        report = inspect_dolci_metadata(
            PinnedLoaders(cache_dir=args.cache_dir).dolci(), strict_counts=True
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.build:
        build = Dolci32KBuilder(
            args.data_dir,
            seed=args.seed,
            loaders=PinnedLoaders(cache_dir=args.cache_dir),
        ).build()
        manifest = load_manifest(build)
        result = _summary(build, manifest)
    else:
        build = resolve_current_build(args.data_dir, args.build_id)
        if args.reaudit:
            result = reaudit_build(build)
            manifest = load_manifest(build)
        else:
            manifest = validate_build(build)
            result = _summary(build, manifest)
    if args.profile_tokenizers:
        result["tokenization_profiles"] = _profile(
            build, manifest, args.profile_tokenizers,
            force=args.force_tokenizer_profile,
            cache_dir=args.cache_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
