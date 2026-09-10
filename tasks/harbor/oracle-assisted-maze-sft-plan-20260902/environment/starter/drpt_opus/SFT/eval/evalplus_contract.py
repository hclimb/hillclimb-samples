"""Immutable EvalPlus execution contract for Dolci32k MBPP+ evaluation.

The official v0.3.1 release points at an OCI image whose installed Python
distribution identifies itself as ``0.4.0.dev2``. Those are two distinct
pieces of provenance: the host-side package/release used to materialize the
registry, and the distribution actually executing generated code inside the
digest-pinned image. Pin both instead of pretending the image metadata is the
release number.
"""

from __future__ import annotations

from SFT.data.dolci32k.profile import (
    EVALPLUS_DATASET_VERSION,
    EVALPLUS_IMAGE,
    EVALPLUS_VERSION,
)


EVALPLUS_RELEASE_VERSION = EVALPLUS_VERSION
EVALPLUS_HOST_DISTRIBUTION_VERSION = EVALPLUS_VERSION
EVALPLUS_CONTAINER_DISTRIBUTION_VERSION = "0.4.0.dev2"

MBPP_PLUS_CACHE_FILENAME = "MbppPlus-v0.2.0.jsonl"
MBPP_PLUS_DATASET_SHA256 = (
    "b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad"
)
MBPP_PLUS_CONTAINER_PATH = f"/opt/drpt/evalplus/{MBPP_PLUS_CACHE_FILENAME}"
EVALPLUS_RUNTIME_CACHE_CONTAINER_PATH = "/workspace/runtime_cache"


__all__ = [
    "EVALPLUS_CONTAINER_DISTRIBUTION_VERSION",
    "EVALPLUS_DATASET_VERSION",
    "EVALPLUS_HOST_DISTRIBUTION_VERSION",
    "EVALPLUS_IMAGE",
    "EVALPLUS_RELEASE_VERSION",
    "EVALPLUS_RUNTIME_CACHE_CONTAINER_PATH",
    "MBPP_PLUS_CACHE_FILENAME",
    "MBPP_PLUS_CONTAINER_PATH",
    "MBPP_PLUS_DATASET_SHA256",
]
