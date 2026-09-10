"""Deterministic records, allocation, hashing, and atomic JSON primitives."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from SFT.data.dolci32k.profile import SEED

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_prompt(text: Any) -> str:
    if text is None:
        return ""
    return _WHITESPACE_RE.sub(
        " ", unicodedata.normalize("NFKC", str(text)).casefold()
    ).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_hash(text: Any) -> str:
    return sha256_text(normalize_prompt(text))


def stable_rank(namespace: str, *parts: Any, seed: int = SEED) -> str:
    body = "\x1f".join(str(part) for part in parts)
    return sha256_text(f"{seed}\x1e{namespace}\x1e{body}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def first_user_prompt(messages_or_record: Any) -> str:
    messages = (
        messages_or_record.get("messages", [])
        if isinstance(messages_or_record, Mapping)
        else messages_or_record
    )
    for message in messages or []:
        if isinstance(message, Mapping) and message.get("role") == "user":
            value = message.get("content")
            return value if isinstance(value, str) else ""
    return ""


def user_prompt_views(messages_or_record: Any) -> List[str]:
    messages = (
        messages_or_record.get("messages", [])
        if isinstance(messages_or_record, Mapping)
        else messages_or_record
    )
    turns = [
        message.get("content", "")
        for message in (messages or [])
        if isinstance(message, Mapping)
        and message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and message.get("content").strip()
    ]
    if len(turns) > 1:
        turns.append("\n".join(turns))
    return turns


def make_record(
    *,
    record_id: str,
    dataset: str,
    domain: str,
    messages: Sequence[Mapping[str, Any]],
    metadata: Optional[Mapping[str, Any]] = None,
    tools: Optional[Sequence[Mapping[str, Any]]] = None,
    question_hash: Optional[str] = None,
) -> Dict[str, Any]:
    prompt = first_user_prompt(messages)
    record: Dict[str, Any] = {
        "id": str(record_id),
        "dataset": str(dataset),
        "source_dataset": str(dataset),
        "domain": str(domain),
        "messages": [dict(message) for message in messages],
        "prompt_hash": prompt_hash(prompt),
        "metadata": dict(metadata or {}),
    }
    record["metadata"].setdefault("prompt_hash", record["prompt_hash"])
    if tools:
        record["tools"] = [dict(tool) for tool in tools]
    if question_hash:
        record["question_hash"] = str(question_hash)
        record["metadata"].setdefault("question_hash", str(question_hash))
    return record


def is_supervised_record(record: Optional[Mapping[str, Any]]) -> bool:
    if not record or not isinstance(record.get("messages"), list):
        return False
    if not first_user_prompt(record).strip():
        return False
    has_assistant = False
    for message in record["messages"]:
        if not isinstance(message, Mapping):
            return False
        if message.get("role") not in ("system", "user", "assistant", "tool"):
            return False
        if not isinstance(message.get("content"), str):
            return False
        if not message.get("content", "").strip() and not (
            message.get("role") == "assistant" and message.get("tool_calls")
        ):
            return False
        has_assistant |= message.get("role") == "assistant"
    return has_assistant


def largest_remainder(weights: Mapping[str, int], total: int) -> Dict[str, int]:
    if total < 0:
        raise ValueError("allocation total must be non-negative")
    cleaned = {str(key): int(value) for key, value in weights.items()}
    if any(value < 0 for value in cleaned.values()):
        raise ValueError("allocation weights must be non-negative")
    denominator = sum(cleaned.values())
    if denominator <= 0:
        if total == 0:
            return {key: 0 for key in cleaned}
        raise ValueError("at least one allocation weight must be positive")
    allocated = {key: total * value // denominator for key, value in cleaned.items()}
    remaining = total - sum(allocated.values())
    order = sorted(cleaned, key=lambda key: (-(total * cleaned[key] % denominator), key))
    for key in order[:remaining]:
        allocated[key] += 1
    return allocated


def capped_equal_waterfill(available: Mapping[str, int], total: int) -> Dict[str, int]:
    """Allocate as equally as capacity permits, redistributing every shortage.

    No source is sampled with replacement.  Deterministic lexicographic tie
    breaking makes the allocation independent of mapping insertion order.
    """
    capacities = {str(key): int(value) for key, value in available.items()}
    if total < 0 or any(value < 0 for value in capacities.values()):
        raise ValueError("water-fill sizes must be non-negative")
    if total > sum(capacities.values()):
        raise RuntimeError(
            f"capped-equal allocation needs {total}, only {sum(capacities.values())} available"
        )
    allocated = {key: 0 for key in capacities}
    active = sorted(key for key, value in capacities.items() if value > 0)
    remaining = total
    while remaining:
        if not active:
            raise AssertionError("water-fill exhausted sources before allocation completed")
        floor_share, _ = divmod(remaining, len(active))
        capped = [
            key for key in active
            if capacities[key] - allocated[key] < floor_share
        ]
        if capped:
            for key in capped:
                take = capacities[key] - allocated[key]
                allocated[key] += take
                remaining -= take
            active = [key for key in active if key not in set(capped)]
            continue
        for key in active:
            allocated[key] += floor_share
            remaining -= floor_share
        if remaining:
            for key in active:
                if remaining == 0:
                    break
                if allocated[key] < capacities[key]:
                    allocated[key] += 1
                    remaining -= 1
        if remaining:
            active = [key for key in active if allocated[key] < capacities[key]]
            continue
        break
    if sum(allocated.values()) != total:
        raise AssertionError("water-fill allocation did not reach requested total")
    if any(allocated[key] > capacities[key] for key in capacities):
        raise AssertionError("water-fill allocation oversampled a source")
    return allocated


def ordered_id_sha256(records_or_ids: Iterable[Any]) -> str:
    ids = (
        str(value.get("id")) if isinstance(value, Mapping) else str(value)
        for value in records_or_ids
    )
    return sha256_text("".join(identifier + "\n" for identifier in ids))


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row is not an object")
            yield value


def atomic_write_jsonl(path: str | os.PathLike[str], records: Iterable[Mapping[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count


def atomic_write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    path = Path(path)
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


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_count(path: str | os.PathLike[str]) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())
