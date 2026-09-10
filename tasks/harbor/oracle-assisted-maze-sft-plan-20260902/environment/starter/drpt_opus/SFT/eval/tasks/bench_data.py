"""Shared loader for pinned downstream benchmark records.

Dolci32k reads its four benchmarks from its own immutable
artifact build and fail closed if it is unavailable; the
``{task}_bench_data.jsonl`` fallback remains for ad-hoc local evaluation on
unpinned profiles only.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


#: Profiles whose benchmarks are pinned inside their own immutable build.
PINNED_PROFILES = ("dolci32k",)


def _pinned_artifact_path(
    data_dir: str,
    task: str,
    *,
    profile: str,
    build_id: Optional[str] = None,
    required: bool = False,
) -> Optional[str]:
    """Return one pinned benchmark artifact, optionally failing closed."""
    if profile not in PINNED_PROFILES:
        return None
    try:
        module = __import__(f"SFT.data.{profile}", fromlist=["artifact_path"])
        artifact_path = module.artifact_path
    except (ImportError, AttributeError) as exc:
        if required:
            raise RuntimeError(f"{profile} artifact API is unavailable") from exc
        return None
    try:
        path = artifact_path(
            data_dir, "benchmarks", task, build_id=build_id, validate=True
        )
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        if required:
            raise RuntimeError(
                f"Pinned {profile} benchmark artifact is unavailable for "
                f"task={task!r}, build_id={build_id!r}"
            ) from exc
        return None
    if path and os.path.isfile(path):
        return str(path)
    if required:
        raise FileNotFoundError(
            f"Pinned {profile} benchmark artifact is missing: {path}"
        )
    return None


def load_bench_records(data_dir: str, task: str, k: int = -1) -> List[Dict[str, Any]]:
    """Load the official downstream benchmark records for one task.

    ``k <= 0`` loads the whole benchmark, which is the campaign default; a
    positive ``k`` truncates for smoke tests.
    """
    profile = os.environ.get("DRPT_DOWNSTREAM_PROFILE") or os.environ.get(
        "DRPT_COMPARISON_PROFILE", ""
    )
    # A pinned profile must read its own build. Falling back to the unpinned
    # `SFT/data/eval/` copy would silently score a pinned campaign against
    # whatever happens to sit on local disk.
    pinned_required = profile in PINNED_PROFILES
    build_id = os.environ.get("DRPT_ARTIFACT_BUILD_ID")
    if pinned_required and not build_id:
        raise RuntimeError(
            f"{profile} evaluation requires DRPT_ARTIFACT_BUILD_ID from the "
            "immutable campaign pin"
        )
    path = _pinned_artifact_path(
        data_dir,
        task,
        profile=profile,
        build_id=build_id,
        required=pinned_required,
    )
    if path is None:
        path = os.path.join(data_dir, "eval", task, f"{task}_bench_data.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{task} benchmark data not found: {path}\n"
            "Pinned profiles read benchmarks from their own artifact build; "
            "for an unpinned local run build it with: "
            "python SFT/data/prepare_dolci32k.py --build"
        )

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if k > 0 and len(records) >= k:
                break
    if not records:
        raise ValueError(f"{task} benchmark data is empty: {path}")
    return records


def first_user_content(record: Dict[str, Any]) -> str:
    for message in record.get("messages", []) or []:
        if message.get("role") == "user":
            return message.get("content", "") or ""
    return ""
