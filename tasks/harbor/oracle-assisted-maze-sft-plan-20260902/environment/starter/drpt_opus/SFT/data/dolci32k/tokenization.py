"""Model-specific token-length diagnostics for model-independent Dolci pools.

Raw Dolci manifests deliberately contain no tokenizer decisions.  This module
builds a derived, content-addressed cache from an explicit mapping of artifact
names to JSONL files.  The cache key binds the raw build, the exact artifact
bytes, the tokenizer's *behaviour* (not its model alias), and the right-
truncation length.  Consequently Qwen model aliases with identical tokenizer
assets share one physical cache while retaining separate report entries. The
profiling and training paths share the exact same supervision-aware windowing
helper; profiling records both the hypothetical right-truncation result and
the final result after the deterministic fallback.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import uuid
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from SFT.data.chat_format import (
    SUPERVISION_TRUNCATION_POLICY,
    SUPERVISION_TRUNCATION_POLICY_VERSION,
    canonicalize_messages_and_tools,
    prepare_assistant_supervision,
)


CACHE_SCHEMA_VERSION = 2
REPORT_SCHEMA_VERSION = 2
PARQUET_FILENAME = "token_lengths.parquet"
SUMMARY_FILENAME = "summary.json"
DEFAULT_WARNING_THRESHOLD = 0.05


class TokenizationDiagnosticError(RuntimeError):
    """A row could not be profiled faithfully with the requested tokenizer."""


class ZeroSupervisionError(RuntimeError):
    """Final truncation policy leaves rows without assistant labels."""


class TokenizationTruncationWarning(RuntimeWarning):
    """More than the configured fraction of an artifact is over length."""


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted((_jsonable(item) for item in value), key=repr)
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def tokenization_report_path(
    data_dir: str | os.PathLike[str], raw_build_id: str, max_seq_len: int
) -> Path:
    """Return the stable derived-report path for one immutable raw build."""

    from SFT.data.dolci32k.artifacts import dolci32k_root

    return (
        dolci32k_root(data_dir)
        / "tokenization_reports"
        / str(raw_build_id)
        / f"max_seq_len_{int(max_seq_len)}.json"
    )


def persist_tokenization_report(
    report: Mapping[str, Any],
    *,
    data_dir: str | os.PathLike[str],
) -> Path:
    """Atomically persist the alias-to-physical-cache report outside raw data."""

    raw_build_id = str(report.get("raw_build_id", ""))
    max_seq_len = int(report.get("max_seq_len", 0))
    if not raw_build_id or max_seq_len <= 0:
        raise ValueError("tokenization report needs raw_build_id and max_seq_len")
    payload = {"report_schema_version": REPORT_SCHEMA_VERSION, **dict(report)}
    path = tokenization_report_path(data_dir, raw_build_id, max_seq_len)
    _atomic_write_json(path, payload)
    return path


def load_tokenization_report(
    path: str | os.PathLike[str],
    *,
    raw_build_id: Optional[str] = None,
    max_seq_len: Optional[int] = None,
) -> Dict[str, Any]:
    """Load a report and fail closed if any referenced physical cache changed."""

    report_path = Path(path).expanduser().resolve()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Unreadable Dolci tokenization report: {report_path}") from exc
    if not isinstance(report, dict) or report.get("report_schema_version") != REPORT_SCHEMA_VERSION:
        raise RuntimeError(f"Invalid Dolci tokenization report schema: {report_path}")
    if raw_build_id is not None and report.get("raw_build_id") != str(raw_build_id):
        raise RuntimeError(
            "Dolci tokenization report raw-build mismatch: "
            f"{report.get('raw_build_id')!r} != {str(raw_build_id)!r}"
        )
    if max_seq_len is not None and report.get("max_seq_len") != int(max_seq_len):
        raise RuntimeError(
            "Dolci tokenization report max-length mismatch: "
            f"{report.get('max_seq_len')!r} != {int(max_seq_len)!r}"
        )
    if (
        report.get("truncation_policy") != SUPERVISION_TRUNCATION_POLICY
        or report.get("truncation_policy_version")
        != SUPERVISION_TRUNCATION_POLICY_VERSION
    ):
        raise RuntimeError(
            "Dolci tokenization report uses a stale truncation policy: "
            f"{report.get('truncation_policy')!r}/"
            f"{report.get('truncation_policy_version')!r}"
        )
    profiles = report.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise RuntimeError(f"Dolci tokenization report has no model profiles: {report_path}")
    for alias, bundle in profiles.items():
        if not isinstance(bundle, dict):
            raise RuntimeError(f"Malformed tokenization profile {alias!r}")
        cache_dir = Path(str(bundle.get("cache_dir", "")))
        cache_key = str(bundle.get("cache_key", ""))
        cached = _cache_is_valid(cache_dir, cache_key)
        if cached is None:
            raise RuntimeError(
                f"Dolci tokenization cache failed integrity validation for {alias!r}: "
                f"{cache_dir}"
            )
        for field in (
            "raw_build_id", "max_seq_len", "tokenizer_fingerprint",
            "parquet_sha256", "raw_artifacts", "truncation_policy",
            "truncation_policy_version",
        ):
            if cached.get(field) != bundle.get(field):
                raise RuntimeError(
                    f"Dolci tokenization cache/report mismatch for {alias!r}: {field}"
                )
    return report


def _import_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised only in broken envs
        raise RuntimeError(
            "Dolci token-length caching requires pyarrow (normally installed "
            "with the datasets dependency)"
        ) from exc
    return pa, pq


def _parquet_schema(metadata: Optional[Mapping[bytes, bytes]] = None):
    pa, _ = _import_pyarrow()
    schema = pa.schema(
        [
            pa.field("artifact", pa.string(), nullable=False),
            pa.field("id", pa.string(), nullable=False),
            pa.field("source_dataset", pa.string(), nullable=False),
            pa.field("domain", pa.string(), nullable=False),
            pa.field("untruncated_length", pa.int64()),
            pa.field("max_seq_len", pa.int32(), nullable=False),
            pa.field("exceeds_max_seq_len", pa.bool_()),
            pa.field("assistant_tokens_total", pa.int64()),
            pa.field("assistant_tokens_retained_by_right_truncation", pa.int64()),
            pa.field("assistant_truncated_by_right_truncation", pa.bool_()),
            pa.field("zero_supervised_after_right_truncation", pa.bool_()),
            pa.field("assistant_tokens_retained", pa.int64()),
            pa.field("assistant_truncated", pa.bool_()),
            pa.field("supervision_preserving_fallback_applied", pa.bool_()),
            pa.field("zero_supervised_after_truncation", pa.bool_()),
            pa.field("assistant_end_retained", pa.bool_()),
            pa.field("truncation_window_start", pa.int64()),
            pa.field("truncation_window_end", pa.int64()),
            pa.field("truncation_policy", pa.string(), nullable=False),
            pa.field("truncation_policy_version", pa.int32(), nullable=False),
            pa.field("assistant_span_method", pa.string(), nullable=False),
            pa.field("diagnostic_error", pa.string()),
        ]
    )
    return schema.with_metadata(dict(metadata or {}))


def _atomic_write_parquet(path: Path, rows: Sequence[Mapping[str, Any]], cache_key: str) -> None:
    pa, pq = _import_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    schema = _parquet_schema(
        {
            b"dolci32k_cache_schema": str(CACHE_SCHEMA_VERSION).encode("ascii"),
            b"dolci32k_cache_key": cache_key.encode("ascii"),
            b"dolci32k_truncation_policy": SUPERVISION_TRUNCATION_POLICY.encode(
                "ascii"
            ),
            b"dolci32k_truncation_policy_version": str(
                SUPERVISION_TRUNCATION_POLICY_VERSION
            ).encode("ascii"),
        }
    )
    try:
        table = pa.Table.from_pylist(list(rows), schema=schema)
        with temporary.open("wb") as handle:
            pq.write_table(table, handle, compression="zstd")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _as_token_list(value: Any) -> List[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError("Expected one tokenized conversation")
        value = value[0]
    return [int(token) for token in value]


def _apply_chat_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    *,
    tokenize: bool,
    **kwargs: Any,
) -> Any:
    template_kwargs = dict(kwargs)
    if tools:
        template_kwargs["tools"] = list(tools)
    return tokenizer.apply_chat_template(
        list(messages), tokenize=tokenize, **template_kwargs
    )


def _tokenize_rendered(tokenizer: Any, rendered: Any) -> List[int]:
    if isinstance(rendered, str):
        rendered = tokenizer.encode(rendered, add_special_tokens=False)
    return _as_token_list(rendered)


def _qwen_assistant_spans(tokenizer: Any, token_ids: Sequence[int]) -> List[Tuple[int, int]]:
    start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    unknown = getattr(tokenizer, "unk_token_id", None)
    if start_id is None or end_id is None or start_id == unknown or end_id == unknown:
        return []
    role_ids = _as_token_list(tokenizer.encode("assistant\n", add_special_tokens=False))
    spans: List[Tuple[int, int]] = []
    index = 0
    while index < len(token_ids):
        if token_ids[index] != start_id:
            index += 1
            continue
        role_start = index + 1
        content_start = role_start + len(role_ids)
        if list(token_ids[role_start:content_start]) != role_ids:
            index += 1
            continue
        end = content_start
        while end < len(token_ids) and token_ids[end] != end_id:
            end += 1
        if end >= len(token_ids):
            if content_start < len(token_ids):
                spans.append((content_start, len(token_ids)))
            break
        spans.append((content_start, end + 1))
        index = end + 1
    return spans


def _mask_spans(mask: Sequence[Any]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, value in enumerate(mask):
        active = bool(value)
        if active and start is None:
            start = index
        elif not active and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def _native_assistant_mask(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
) -> Optional[Tuple[List[int], List[Tuple[int, int]]]]:
    """Use Hugging Face ``{% generation %}`` masks when a template has them."""

    try:
        encoded = _apply_chat_template(
            tokenizer,
            messages,
            tools,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
    except Exception:
        return None
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        return None
    raw_mask = encoded.get("assistant_masks", encoded.get("assistant_mask"))
    if raw_mask is None:
        return None
    token_ids = _as_token_list(encoded["input_ids"])
    mask = _as_token_list(raw_mask)
    if len(token_ids) != len(mask) or not any(mask):
        return None
    return token_ids, _mask_spans(mask)


def _prefix_assistant_spans(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    full_text: str,
) -> Optional[List[Tuple[int, int]]]:
    spans: List[Tuple[int, int]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        try:
            prefix_to = _apply_chat_template(
                tokenizer,
                messages[:index],
                tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prefix_through = _apply_chat_template(
                tokenizer,
                messages[: index + 1],
                tools,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            return None
        if not isinstance(prefix_to, str) or not isinstance(prefix_through, str):
            return None
        if not full_text.startswith(prefix_through):
            return None
        start = len(tokenizer.encode(prefix_to, add_special_tokens=False))
        end = len(tokenizer.encode(prefix_through, add_special_tokens=False))
        if start < end:
            spans.append((start, end))
    return spans or None


def _token_diff_span(
    full_ids: Sequence[int], comparison_ids: Sequence[int]
) -> Optional[Tuple[int, int]]:
    prefix = 0
    shared = min(len(full_ids), len(comparison_ids))
    while prefix < shared and full_ids[prefix] == comparison_ids[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < len(full_ids) - prefix
        and suffix < len(comparison_ids) - prefix
        and full_ids[len(full_ids) - suffix - 1]
        == comparison_ids[len(comparison_ids) - suffix - 1]
    ):
        suffix += 1
    end = len(full_ids) - suffix
    return (prefix, end) if prefix < end else None


def _merge_spans(spans: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    merged: List[List[int]] = []
    for start, end in sorted(spans):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _difference_assistant_spans(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    full_ids: Sequence[int],
) -> Optional[List[Tuple[int, int]]]:
    """Last-resort span inference for templates that are not prefix-stable."""

    spans: List[Tuple[int, int]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        comparison = copy.deepcopy(list(messages))
        comparison[index]["content"] = ""
        comparison[index].pop("tool_calls", None)
        comparison[index].pop("reasoning_content", None)
        try:
            rendered = _apply_chat_template(
                tokenizer,
                comparison,
                tools,
                tokenize=True,
                add_generation_prompt=False,
            )
            comparison_ids = _tokenize_rendered(tokenizer, rendered)
        except Exception:
            return None
        span = _token_diff_span(full_ids, comparison_ids)
        if span is not None:
            spans.append(span)
    return _merge_spans(spans) or None


def diagnose_record(
    record: Mapping[str, Any], tokenizer: Any, max_seq_len: int
) -> Dict[str, Any]:
    """Return diagnostics from the exact formatter used by training.

    Structural record errors or an unsupported template still raise because
    length, span, or fallback statistics must never be fabricated.
    """

    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    try:
        prepared = prepare_assistant_supervision(
            record, tokenizer, max_seq_len
        )
    except Exception as exc:
        raise TokenizationDiagnosticError(str(exc)) from exc
    return dict(prepared["diagnostics"])


def tokenizer_fingerprint_payload(tokenizer: Any) -> Dict[str, Any]:
    """Describe tokenizer behaviour without including a model/profile alias."""

    from SFT.data.get_val_dataset import ensure_chat_template

    ensure_chat_template(tokenizer)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "to_str"):
        vocabulary_payload = str(backend.to_str())
        vocabulary_kind = "backend_tokenizer"
    elif hasattr(tokenizer, "get_vocab"):
        vocabulary_payload = _canonical_json(tokenizer.get_vocab())
        vocabulary_kind = "vocabulary"
    else:
        # Lightweight test/wrapper tokenizers need a stable fallback too;
        # object reprs commonly contain a process-specific memory address.
        vocabulary_payload = _canonical_json(
            {
                "special": getattr(tokenizer, "_special", None),
                "vocab": getattr(tokenizer, "vocab", None),
            }
        )
        vocabulary_kind = "attribute_fallback"
    return {
        "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}",
        "chat_template": _jsonable(getattr(tokenizer, "chat_template", None)),
        "special_tokens_map": _jsonable(
            getattr(tokenizer, "special_tokens_map", {})
        ),
        "all_special_ids": _jsonable(getattr(tokenizer, "all_special_ids", [])),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "vocabulary_kind": vocabulary_kind,
        "vocabulary_sha256": _sha256_text(vocabulary_payload),
    }


def tokenizer_behavior_fingerprint(tokenizer: Any) -> str:
    """Return the content fingerprint used to share physical caches."""

    return _sha256_text(_canonical_json(tokenizer_fingerprint_payload(tokenizer)))


def _profile_provenance(profile: Mapping[str, Any]) -> Dict[str, Any]:
    sensitive = {"token", "use_auth_token", "auth_token"}
    return {
        str(key): _jsonable(value)
        for key, value in sorted(profile.items())
        if str(key) not in sensitive and str(key) != "from_pretrained_kwargs"
    }


def load_tokenizer_profile(profile: Mapping[str, Any]) -> Any:
    """Load a tokenizer from a small explicit profile dictionary.

    Recognized repository keys are ``tokenizer_repo``, ``tokenizer_name``,
    ``tokenizer_name_or_path``, and (as a fallback) ``model_name_or_path``.
    ``tokenizer_revision`` is forwarded independently from a model revision.
    Additional Hugging Face options may be placed in
    ``from_pretrained_kwargs``.
    """

    from transformers import AutoTokenizer

    repository = (
        profile.get("tokenizer_repo")
        or profile.get("tokenizer_name_or_path")
        or profile.get("tokenizer_name")
        or profile.get("model_name_or_path")
        or profile.get("model_repo")
    )
    if not repository:
        raise ValueError("tokenizer profile has no tokenizer repository")
    kwargs = dict(profile.get("from_pretrained_kwargs", {}))
    revision = profile.get("tokenizer_revision")
    if revision is not None:
        kwargs.setdefault("revision", revision)
    for key in ("trust_remote_code", "use_fast", "local_files_only", "token"):
        if key in profile:
            kwargs.setdefault(key, profile[key])
    tokenizer = AutoTokenizer.from_pretrained(str(repository), **kwargs)
    from SFT.data.get_val_dataset import ensure_chat_template

    return ensure_chat_template(tokenizer)


def _artifact_signatures(
    artifact_paths: Mapping[str, str | os.PathLike[str]],
) -> Tuple[Dict[str, Path], Dict[str, Dict[str, Any]]]:
    if not artifact_paths:
        raise ValueError("artifact_paths must not be empty")
    resolved: Dict[str, Path] = {}
    signatures: Dict[str, Dict[str, Any]] = {}
    for raw_name, raw_path in sorted(artifact_paths.items()):
        name = str(raw_name)
        if not name or name in resolved:
            raise ValueError(f"invalid or duplicate artifact name: {raw_name!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved[name] = path
        signatures[name] = {
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return resolved, signatures


def _record_id(record: Mapping[str, Any]) -> str:
    for key in ("id", "example_id", "record_id"):
        value = record.get(key)
        if value is not None and str(value):
            return str(value)
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("stable_id", "example_id", "original_id", "id"):
            value = metadata.get(key)
            if value is not None and str(value):
                return str(value)
    raise TokenizationDiagnosticError(
        "raw artifact row has no model-independent id/example_id/record_id"
    )


def _record_token_input_digest(record: Mapping[str, Any]) -> str:
    payload = {
        key: record.get(key)
        for key in ("messages", "tools", "functions")
        if key in record
    }
    return _sha256_text(_canonical_json(payload))


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TokenizationDiagnosticError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise TokenizationDiagnosticError(
                    f"{path}:{line_number}: JSONL row is not an object"
                )
            yield line_number, value


def _error_diagnostic(max_seq_len: int, error: Exception) -> Dict[str, Any]:
    return {
        "untruncated_length": None,
        "max_seq_len": int(max_seq_len),
        "exceeds_max_seq_len": None,
        "assistant_tokens_total": None,
        "assistant_tokens_retained_by_right_truncation": None,
        "assistant_truncated_by_right_truncation": None,
        "zero_supervised_after_right_truncation": None,
        "assistant_tokens_retained": None,
        "assistant_truncated": None,
        "supervision_preserving_fallback_applied": None,
        "zero_supervised_after_truncation": None,
        "assistant_end_retained": None,
        "truncation_window_start": None,
        "truncation_window_end": None,
        "truncation_policy": SUPERVISION_TRUNCATION_POLICY,
        "truncation_policy_version": SUPERVISION_TRUNCATION_POLICY_VERSION,
        "assistant_span_method": "unresolved",
        "diagnostic_error": f"{error.__class__.__name__}: {error}",
    }


def _build_rows(
    artifacts: Mapping[str, Path],
    tokenizer: Any,
    max_seq_len: int,
    *,
    fail_on_diagnostic_error: bool,
) -> Tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    by_id: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for artifact, path in sorted(artifacts.items()):
        seen_in_artifact: set[str] = set()
        for line_number, record in _iter_jsonl(path):
            try:
                record_id = _record_id(record)
            except TokenizationDiagnosticError as exc:
                raise TokenizationDiagnosticError(
                    f"{path}:{line_number}: {exc}"
                ) from exc
            if record_id in seen_in_artifact:
                raise TokenizationDiagnosticError(
                    f"{path}:{line_number}: duplicate id {record_id!r} within artifact"
                )
            seen_in_artifact.add(record_id)
            input_digest = _record_token_input_digest(record)
            cached = by_id.get(record_id)
            if cached is not None and cached[0] != input_digest:
                raise TokenizationDiagnosticError(
                    f"model-independent id {record_id!r} maps to different chat content "
                    f"across raw artifacts"
                )
            if cached is None:
                try:
                    diagnostic = diagnose_record(record, tokenizer, max_seq_len)
                except Exception as exc:
                    if fail_on_diagnostic_error:
                        raise TokenizationDiagnosticError(
                            f"{path}:{line_number} id={record_id!r}: {exc}"
                        ) from exc
                    diagnostic = _error_diagnostic(max_seq_len, exc)
                by_id[record_id] = (input_digest, diagnostic)
            else:
                diagnostic = cached[1]
            rows.append(
                {
                    "artifact": artifact,
                    "id": record_id,
                    "source_dataset": str(
                        record.get("source_dataset", record.get("dataset", "<unknown>"))
                    ),
                    "domain": str(record.get("domain", "<unknown>")),
                    **diagnostic,
                }
            )
    return rows, len(by_id)


def _percentile(values: Sequence[int], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _flag_breakdown(
    rows: Sequence[Mapping[str, Any]], flag: str
) -> Dict[str, Dict[str, Any]]:
    total_by_source = Counter(str(row["source_dataset"]) for row in rows)
    flagged = Counter(
        str(row["source_dataset"]) for row in rows if row.get(flag) is True
    )
    flagged_total = sum(flagged.values())
    return {
        source: {
            "count": int(count),
            "fraction_of_flagged": count / flagged_total if flagged_total else 0.0,
            "fraction_of_source": count / total_by_source[source],
        }
        for source, count in sorted(flagged.items())
    }


def _summarize_rows(rows: Sequence[Mapping[str, Any]], max_seq_len: int) -> Dict[str, Any]:
    lengths = [
        int(row["untruncated_length"])
        for row in rows
        if row.get("untruncated_length") is not None
    ]
    total = len(rows)
    over = sum(row.get("exceeds_max_seq_len") is True for row in rows)
    assistant_truncated = sum(row.get("assistant_truncated") is True for row in rows)
    right_assistant_truncated = sum(
        row.get("assistant_truncated_by_right_truncation") is True
        for row in rows
    )
    right_zero = [
        row
        for row in rows
        if row.get("zero_supervised_after_right_truncation") is True
    ]
    fallback = [
        row
        for row in rows
        if row.get("supervision_preserving_fallback_applied") is True
    ]
    final_zero = [
        row
        for row in rows
        if row.get("zero_supervised_after_truncation") is True
    ]
    errors = [row for row in rows if row.get("diagnostic_error")]
    method_counts = Counter(str(row["assistant_span_method"]) for row in rows)

    def affected_summary(affected):
        return {
            "count": len(affected),
            "fraction": len(affected) / total if total else 0.0,
            "ids": sorted(str(row["id"]) for row in affected),
            "by_source_dataset": dict(
                sorted(
                    Counter(str(row["source_dataset"]) for row in affected).items()
                )
            ),
        }

    return {
        "examples": total,
        "max_seq_len": int(max_seq_len),
        "truncation_policy": SUPERVISION_TRUNCATION_POLICY,
        "truncation_policy_version": SUPERVISION_TRUNCATION_POLICY_VERSION,
        "token_length": {
            "p50": _percentile(lengths, 0.50),
            "p90": _percentile(lengths, 0.90),
            "p99": _percentile(lengths, 0.99),
            "max": max(lengths) if lengths else None,
            "percentile_method": "linear",
        },
        "exceeds_max_seq_len": {
            "count": over,
            "fraction": over / total if total else 0.0,
        },
        "assistant_truncation": {
            "count": assistant_truncated,
            "fraction": assistant_truncated / total if total else 0.0,
        },
        "assistant_truncation_under_right_truncation": {
            "count": right_assistant_truncated,
            "fraction": right_assistant_truncated / total if total else 0.0,
        },
        "zero_supervised_after_right_truncation": affected_summary(right_zero),
        "supervision_preserving_fallback": affected_summary(fallback),
        "zero_supervised_after_truncation": affected_summary(final_zero),
        "over_length_by_source_dataset": _flag_breakdown(
            rows, "exceeds_max_seq_len"
        ),
        "assistant_truncated_by_source_dataset": _flag_breakdown(
            rows, "assistant_truncated"
        ),
        "right_truncation_assistant_truncated_by_source_dataset": _flag_breakdown(
            rows, "assistant_truncated_by_right_truncation"
        ),
        "fallback_by_source_dataset": _flag_breakdown(
            rows, "supervision_preserving_fallback_applied"
        ),
        "source_dataset_counts": dict(
            sorted(Counter(str(row["source_dataset"]) for row in rows).items())
        ),
        "assistant_span_method_counts": dict(sorted(method_counts.items())),
        "diagnostic_errors": {
            "count": len(errors),
            "ids": sorted(str(row["id"]) for row in errors),
        },
    }


def _statistics(
    rows: Sequence[Mapping[str, Any]],
    max_seq_len: int,
    warning_threshold: float,
) -> Tuple[Dict[str, Any], List[str]]:
    names = sorted({str(row["artifact"]) for row in rows})
    per_artifact = {
        name: _summarize_rows(
            [row for row in rows if row["artifact"] == name], max_seq_len
        )
        for name in names
    }
    messages: List[str] = []
    for name, stats in per_artifact.items():
        fraction = float(stats["exceeds_max_seq_len"]["fraction"])
        if fraction > warning_threshold:
            messages.append(
                "WARNING: artifact "
                f"{name!r} has {fraction:.2%} of examples over max_seq_len="
                f"{max_seq_len}, above the {warning_threshold:.2%} threshold"
            )
    return {
        "artifacts": per_artifact,
        "all_artifact_memberships": _summarize_rows(rows, max_seq_len),
    }, messages


def _cache_is_valid(cache_dir: Path, cache_key: str) -> Optional[Dict[str, Any]]:
    summary_path = cache_dir / SUMMARY_FILENAME
    parquet_path = cache_dir / PARQUET_FILENAME
    if not summary_path.is_file() or not parquet_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("cache_schema_version") != CACHE_SCHEMA_VERSION
            or summary.get("cache_key") != cache_key
            or summary.get("parquet_sha256") != _file_sha256(parquet_path)
            or summary.get("truncation_policy") != SUPERVISION_TRUNCATION_POLICY
            or summary.get("truncation_policy_version")
            != SUPERVISION_TRUNCATION_POLICY_VERSION
        ):
            return None
        _, pq = _import_pyarrow()
        metadata = pq.read_metadata(parquet_path)
        if metadata.num_rows != summary.get("parquet_rows"):
            return None
        file_metadata = metadata.metadata or {}
        if file_metadata.get(b"dolci32k_cache_key") != cache_key.encode("ascii"):
            return None
        if file_metadata.get(b"dolci32k_truncation_policy") != (
            SUPERVISION_TRUNCATION_POLICY.encode("ascii")
        ):
            return None
        if file_metadata.get(b"dolci32k_truncation_policy_version") != str(
            SUPERVISION_TRUNCATION_POLICY_VERSION
        ).encode("ascii"):
            return None
    except Exception:
        return None
    return summary


def _result_with_paths(
    summary: Mapping[str, Any],
    cache_dir: Path,
    *,
    cache_reused: bool,
    tokenizer_profile: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        **dict(summary),
        "cache_dir": str(cache_dir),
        "parquet_path": str(cache_dir / PARQUET_FILENAME),
        "summary_path": str(cache_dir / SUMMARY_FILENAME),
        "cache_reused": bool(cache_reused),
        "requested_tokenizer_profile": _profile_provenance(tokenizer_profile),
    }


def profile_jsonl_artifacts(
    artifact_paths: Mapping[str, str | os.PathLike[str]],
    *,
    raw_build_id: str,
    tokenizer_profile: Mapping[str, Any],
    cache_root: str | os.PathLike[str],
    max_seq_len: int = 2048,
    tokenizer: Any = None,
    force: bool = False,
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
    emit_warnings: bool = True,
    fail_on_diagnostic_error: bool = True,
) -> Dict[str, Any]:
    """Profile explicit raw artifacts and return a JSON-serializable bundle.

    The physical cache is outside the immutable raw build and is reused when
    every cache-key input matches.  Passing an already loaded ``tokenizer`` is
    useful for tests and multi-profile orchestration; otherwise the explicit
    profile is loaded with :func:`load_tokenizer_profile`.
    """

    if not str(raw_build_id):
        raise ValueError("raw_build_id must not be empty")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if not 0.0 <= warning_threshold <= 1.0:
        raise ValueError("warning_threshold must be between zero and one")
    artifacts, signatures = _artifact_signatures(artifact_paths)
    tokenizer = tokenizer if tokenizer is not None else load_tokenizer_profile(tokenizer_profile)
    tokenizer_payload = tokenizer_fingerprint_payload(tokenizer)
    tokenizer_fingerprint = _sha256_text(_canonical_json(tokenizer_payload))
    key_payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "raw_build_id": str(raw_build_id),
        "raw_artifacts": signatures,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "max_seq_len": int(max_seq_len),
        "truncation_policy": SUPERVISION_TRUNCATION_POLICY,
        "truncation_policy_version": SUPERVISION_TRUNCATION_POLICY_VERSION,
    }
    cache_key = _sha256_text(_canonical_json(key_payload))
    cache_root_path = Path(cache_root).expanduser().resolve()
    cache_root_path.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_root_path / cache_key
    lock_path = cache_root_path / f".{cache_key}.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing = None if force else _cache_is_valid(cache_dir, cache_key)
        if existing is not None:
            result = _result_with_paths(
                existing,
                cache_dir,
                cache_reused=True,
                tokenizer_profile=tokenizer_profile,
            )
        else:
            rows, unique_examples = _build_rows(
                artifacts,
                tokenizer,
                max_seq_len,
                fail_on_diagnostic_error=fail_on_diagnostic_error,
            )
            statistics, warning_messages = _statistics(
                rows, max_seq_len, warning_threshold
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = cache_dir / PARQUET_FILENAME
            _atomic_write_parquet(parquet_path, rows, cache_key)
            summary: Dict[str, Any] = {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "cache_key": cache_key,
                "raw_build_id": str(raw_build_id),
                "raw_artifacts": signatures,
                "tokenizer_fingerprint": tokenizer_fingerprint,
                "tokenizer_behavior": tokenizer_payload,
                "tokenizer_provenance": _profile_provenance(tokenizer_profile),
                "max_seq_len": int(max_seq_len),
                "truncation_policy": SUPERVISION_TRUNCATION_POLICY,
                "truncation_policy_version": SUPERVISION_TRUNCATION_POLICY_VERSION,
                "parquet_rows": len(rows),
                "unique_examples": unique_examples,
                "parquet_sha256": _file_sha256(parquet_path),
                "statistics": statistics,
                "warnings": warning_messages,
            }
            _atomic_write_json(cache_dir / SUMMARY_FILENAME, summary)
            result = _result_with_paths(
                summary,
                cache_dir,
                cache_reused=False,
                tokenizer_profile=tokenizer_profile,
            )
    if emit_warnings:
        for message in result["warnings"]:
            warnings.warn(message, TokenizationTruncationWarning, stacklevel=2)
    return result


def profile_tokenizer_profiles(
    artifact_paths: Mapping[str, str | os.PathLike[str]],
    *,
    raw_build_id: str,
    tokenizer_profiles: Mapping[str, Mapping[str, Any]],
    cache_root: str | os.PathLike[str],
    max_seq_len: int = 2048,
    tokenizer_factory: Optional[Callable[[Mapping[str, Any]], Any]] = None,
    force: bool = False,
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
    emit_warnings: bool = True,
    fail_on_diagnostic_error: bool = True,
) -> Dict[str, Any]:
    """Profile many named model aliases, deduplicating behavioural caches."""

    factory = tokenizer_factory or load_tokenizer_profile
    results: Dict[str, Dict[str, Any]] = {}
    physical: Dict[str, Dict[str, Any]] = {}
    emitted: set[str] = set()
    for alias, profile in sorted(tokenizer_profiles.items()):
        tokenizer = factory(profile)
        behavioural_fingerprint = tokenizer_behavior_fingerprint(tokenizer)
        result = profile_jsonl_artifacts(
            artifact_paths,
            raw_build_id=raw_build_id,
            tokenizer_profile=profile,
            cache_root=cache_root,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer,
            # ``--force`` rebuilds each distinct physical tokenizer once, not
            # once per model alias (Qwen3-4B/8B intentionally share a cache).
            force=force and behavioural_fingerprint not in physical,
            warning_threshold=warning_threshold,
            emit_warnings=False,
            fail_on_diagnostic_error=fail_on_diagnostic_error,
        )
        results[str(alias)] = result
        fingerprint = str(result["tokenizer_fingerprint"])
        physical.setdefault(
            fingerprint,
            {
                "cache_key": result["cache_key"],
                "cache_dir": result["cache_dir"],
                "parquet_path": result["parquet_path"],
                "aliases": [],
            },
        )["aliases"].append(str(alias))
        if emit_warnings:
            for message in result["warnings"]:
                if message not in emitted:
                    warnings.warn(
                        message, TokenizationTruncationWarning, stacklevel=2
                    )
                    emitted.add(message)
    return {
        "raw_build_id": str(raw_build_id),
        "max_seq_len": int(max_seq_len),
        "truncation_policy": SUPERVISION_TRUNCATION_POLICY,
        "truncation_policy_version": SUPERVISION_TRUNCATION_POLICY_VERSION,
        "profiles": results,
        "physical_caches": physical,
        "warnings": sorted(emitted)
        if emit_warnings
        else sorted({message for result in results.values() for message in result["warnings"]}),
    }


def read_tokenization_cache(
    cache: str | os.PathLike[str] | Mapping[str, Any],
    *,
    columns: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Read cached rows from a bundle result, cache directory, or Parquet path."""

    if isinstance(cache, Mapping):
        path = Path(str(cache["parquet_path"]))
    else:
        path = Path(cache)
        if path.is_dir():
            path = path / PARQUET_FILENAME
    _, pq = _import_pyarrow()
    return pq.read_table(path, columns=list(columns) if columns else None).to_pylist()


def zero_supervision_report(
    cache: str | os.PathLike[str] | Mapping[str, Any],
    *,
    artifact_names: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Return affected model-independent IDs and source counts for preflight."""

    selected = None if artifact_names is None else {str(name) for name in artifact_names}
    rows = read_tokenization_cache(
        cache,
        columns=(
            "artifact",
            "id",
            "source_dataset",
            "zero_supervised_after_truncation",
        ),
    )
    affected = [
        row
        for row in rows
        if row["zero_supervised_after_truncation"] is True
        and (selected is None or row["artifact"] in selected)
    ]
    return {
        "count": len(affected),
        "ids": sorted({str(row["id"]) for row in affected}),
        "by_artifact": dict(
            sorted(Counter(str(row["artifact"]) for row in affected).items())
        ),
        "by_source_dataset": dict(
            sorted(Counter(str(row["source_dataset"]) for row in affected).items())
        ),
    }


def assert_no_zero_supervision(
    cache: str | os.PathLike[str] | Mapping[str, Any],
    *,
    artifact_names: Optional[Iterable[str]] = None,
) -> None:
    """Fail preflight if the final policy leaves rows without labels."""

    report = zero_supervision_report(cache, artifact_names=artifact_names)
    if report["count"]:
        raise ZeroSupervisionError(
            "final truncation policy leaves rows without assistant supervision: "
            + _canonical_json(report)
        )


# Short aliases for CLI code; the longer names above remain self-documenting.
profile_artifacts = profile_jsonl_artifacts
profile_all_tokenizers = profile_tokenizer_profiles


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "DEFAULT_WARNING_THRESHOLD",
    "TokenizationDiagnosticError",
    "TokenizationTruncationWarning",
    "ZeroSupervisionError",
    "assert_no_zero_supervision",
    "diagnose_record",
    "load_tokenizer_profile",
    "load_tokenization_report",
    "persist_tokenization_report",
    "profile_all_tokenizers",
    "profile_artifacts",
    "profile_jsonl_artifacts",
    "profile_tokenizer_profiles",
    "read_tokenization_cache",
    "tokenizer_behavior_fingerprint",
    "tokenizer_fingerprint_payload",
    "tokenization_report_path",
    "zero_supervision_report",
]
