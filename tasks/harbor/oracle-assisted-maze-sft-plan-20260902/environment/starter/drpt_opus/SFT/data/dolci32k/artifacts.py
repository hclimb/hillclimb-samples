"""Immutable content-addressed Dolci32K build discovery and validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .common import canonical_json, file_sha256, jsonl_count, ordered_id_sha256, read_jsonl
from .profile import (
    DEFAULT_SIZES,
    GENERAL_POOLS,
    PINNED_SOURCES,
    PROFILE_NAME,
    PROFILE_VERSION,
    all_artifact_relative_paths,
    artifact_relative_path,
    fixed_artifact_counts,
    raw_profile_config,
)

MANIFEST_SCHEMA_VERSION = 1
ARTIFACT_ROOT_NAME = "dolci32k_artifacts"


def profile_fingerprint() -> str:
    payload = {
        "raw_profile": raw_profile_config(),
        "sources": {
            key: {
                "repo": pin.repo, "revision": pin.revision,
                "config": pin.config, "split": pin.split,
            }
            for key, pin in sorted(PINNED_SOURCES.items())
        },
        "prompt_identity": "NFKC+casefold+whitespace-collapse",
        "decontamination": {
            "unit": "word-8-gram", "overlap_coefficient_threshold": 0.80,
            "shorter_than_8_words": "exact-only",
            "views": "each-user-turn-and-full-user-context",
        },
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def dolci32k_root(data_dir: str | os.PathLike[str]) -> Path:
    return Path(data_dir).expanduser().resolve() / ARTIFACT_ROOT_NAME


def resolve_current_build(data_dir: str | os.PathLike[str], build_id: Optional[str] = None) -> Path:
    root = dolci32k_root(data_dir)
    if build_id is None:
        current = root / "CURRENT"
        if not current.is_file():
            raise FileNotFoundError(
                f"dolci32k CURRENT pointer is missing: {current}; run prepare_dolci32k.py --build"
            )
        build_id = current.read_text(encoding="utf-8").strip()
    if len(build_id or "") != 64 or any(ch not in "0123456789abcdef" for ch in build_id):
        raise RuntimeError(f"Invalid dolci32k build id: {build_id!r}")
    build = root / "builds" / str(build_id)
    if not build.is_dir():
        raise FileNotFoundError(f"dolci32k build directory is missing: {build}")
    return build


def load_manifest(build: str | os.PathLike[str]) -> Dict[str, Any]:
    path = Path(build) / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Unreadable dolci32k manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"dolci32k manifest is not an object: {path}")
    return payload


def validate_build(build: str | os.PathLike[str], *, expected_build_id: Optional[str] = None) -> Dict[str, Any]:
    build = Path(build)
    manifest = load_manifest(build)
    expected_id = expected_build_id or build.name
    if len(expected_id) != 64 or any(ch not in "0123456789abcdef" for ch in expected_id):
        raise RuntimeError(f"Invalid dolci32k build id: {expected_id!r}")
    required = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "profile_name": PROFILE_NAME,
        "profile_version": PROFILE_VERSION,
        "profile_fingerprint": profile_fingerprint(),
        "build_id": expected_id,
        "audit_status": "passed",
        "model_independent_raw_membership": True,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"Stale/bad dolci32k manifest {build / 'manifest.json'}: "
                f"{key}={manifest.get(key)!r}, expected {expected!r}"
            )
    core = {key: value for key, value in manifest.items() if key != "build_id"}
    computed = hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()
    if computed != expected_id:
        raise RuntimeError(
            f"dolci32k manifest fingerprint mismatch: computed={computed}, directory={expected_id}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("dolci32k manifest has no artifacts mapping")
    missing = {str(path) for path in all_artifact_relative_paths()} - set(artifacts)
    if missing:
        raise RuntimeError(f"dolci32k manifest is missing required artifacts: {sorted(missing)}")
    for relative, entry in sorted(artifacts.items()):
        path = build / relative
        if not path.is_file() or not isinstance(entry, dict):
            raise RuntimeError(f"Missing dolci32k artifact: {path}")
        count, digest = jsonl_count(path), file_sha256(path)
        if count != entry.get("count") or digest != entry.get("sha256"):
            raise RuntimeError(
                f"dolci32k artifact integrity failure: {relative}; "
                f"count={count}/{entry.get('count')} sha256={digest}/{entry.get('sha256')}"
            )
        if relative.startswith("candidate_orders/"):
            rows = list(read_jsonl(path))
            positions = [row.get("position") for row in rows]
            ids = [str(row.get("id", "")) for row in rows]
            if positions != list(range(len(rows))) or len(ids) != len(set(ids)) or "" in ids:
                raise RuntimeError(f"Invalid candidate-order rows: {relative}")
            if ordered_id_sha256(ids) != entry.get("ordered_id_sha256"):
                raise RuntimeError(f"Candidate-order ID hash mismatch: {relative}")
    for relative, expected_count in fixed_artifact_counts(DEFAULT_SIZES).items():
        if artifacts.get(relative, {}).get("count") != expected_count:
            raise RuntimeError(
                f"dolci32k fixed-count contract failed for {relative}: expected {expected_count}"
            )
    for pool in GENERAL_POOLS:
        train_rows = read_jsonl(
            build / artifact_relative_path("general", f"{pool}/train")
        )
        train_ids = {str(row.get("id", "")) for row in train_rows}
        order_rows = read_jsonl(
            build / artifact_relative_path("candidate_orders", pool)
        )
        order_ids = {str(row.get("id", "")) for row in order_rows}
        if train_ids != order_ids:
            raise RuntimeError(
                f"Candidate order membership differs from {pool} train membership"
            )
    for relative, entry in sorted((manifest.get("metadata_files") or {}).items()):
        path = build / relative
        if not path.is_file() or file_sha256(path) != entry.get("sha256"):
            raise RuntimeError(f"dolci32k metadata-file integrity failure: {relative}")
    return manifest


def validate_current_build(data_dir: str | os.PathLike[str], build_id: Optional[str] = None) -> Dict[str, Any]:
    return validate_build(resolve_current_build(data_dir, build_id))


def artifact_path(
    data_dir: str | os.PathLike[str], role: str, name: str, *,
    build_id: Optional[str] = None, validate: bool = True,
) -> Path:
    build = resolve_current_build(data_dir, build_id)
    if validate:
        validate_build(build)
    path = build / artifact_relative_path(role, name)
    if not path.is_file():
        raise FileNotFoundError(f"dolci32k artifact is missing: {path}")
    return path


def candidate_order_path(
    data_dir: str | os.PathLike[str], pool: str, *,
    build_id: Optional[str] = None, validate: bool = True,
) -> Path:
    return artifact_path(
        data_dir, "candidate_orders", pool, build_id=build_id, validate=validate
    )
