"""Pinned upstream loaders and network-free row converters for Dolci32K."""

from __future__ import annotations

import importlib.metadata
import json
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

from .common import canonical_json, is_supervised_record, make_record, prompt_hash
from .profile import (
    DOLCI_DOMAIN_MAP,
    DOLCI_SOURCE_DOMAINS,
    EVALPLUS_DATASET_VERSION,
    EVALPLUS_PACKAGE,
    EVALPLUS_VERSION,
    MATH_TRAIN_CONFIGS,
    PINNED_SOURCES,
    SourcePin,
)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("text", "content", "value", "answer", "output"):
            if isinstance(value.get(key), str):
                return value[key]
        return canonical_json(value)
    if isinstance(value, (list, tuple)):
        return "\n".join(filter(None, (_text(item) for item in value)))
    return "" if value is None else str(value)


def dolci_domain(raw_domain: Any) -> str:
    try:
        return DOLCI_DOMAIN_MAP[str(raw_domain)]
    except KeyError:
        raise ValueError(
            f"Unknown literal Dolci domain {raw_domain!r}; classification must be reviewed"
        ) from None


def validate_dolci_pair(raw_domain: Any, raw_source: Any) -> None:
    source = str(raw_source)
    expected = DOLCI_SOURCE_DOMAINS.get(source)
    if expected is None:
        raise ValueError(
            f"Unknown literal Dolci source_dataset {raw_source!r}; classification must be reviewed"
        )
    if str(raw_domain) != expected:
        raise ValueError(
            f"Unreviewed Dolci domain/source pairing: domain={raw_domain!r}, "
            f"source_dataset={raw_source!r}, expected_domain={expected!r}"
        )


def has_structural_tool_use(row: Mapping[str, Any], record: Optional[Mapping[str, Any]] = None) -> bool:
    """Detect tool/function structure independent of the metadata domain."""
    for key in ("tools", "functions", "tool_calls", "function_calls"):
        value = row.get(key)
        if value not in (None, "", [], {}, "[]", "{}"):
            return True
    messages = row.get("messages") or []
    for message in messages if isinstance(messages, (list, tuple)) else []:
        if not isinstance(message, Mapping):
            continue
        if str(message.get("role", "")).casefold() in ("tool", "environment", "function"):
            return True
        if any(message.get(key) not in (None, "", [], {}, "[]", "{}")
               for key in ("tools", "functions", "tool_calls", "function_calls")):
            return True
    if record:
        if record.get("tools"):
            return True
        for message in record.get("messages", []):
            if message.get("role") == "tool" or message.get("tool_calls"):
                return True
    return False


def convert_dolci(row: Mapping[str, Any], index: int) -> Optional[Dict[str, Any]]:
    from SFT.data.chat_format import canonicalize_messages_and_tools

    source_id = row.get("id")
    source_dataset = row.get("source_dataset")
    if source_id in (None, "") or not isinstance(source_dataset, str):
        return None
    validate_dolci_pair(row.get("domain"), source_dataset)
    try:
        messages, tools = canonicalize_messages_and_tools(row)
    except ValueError:
        return None
    record = make_record(
        record_id=f"dolci::{source_dataset}::{source_id}",
        dataset=source_dataset,
        domain=dolci_domain(row.get("domain")),
        messages=messages,
        metadata={
            "source_id": str(source_id),
            "source_dataset": source_dataset,
            "source_domain": str(row.get("domain")),
            "source_index": int(index),
            "source_repo": PINNED_SOURCES["dolci"].repo,
            "source_revision": PINNED_SOURCES["dolci"].revision,
            "source_split": PINNED_SOURCES["dolci"].split,
        },
        tools=tools,
    )
    return record if is_supervised_record(record) else None


def convert_math_train(row: Mapping[str, Any], index: int, *, config: str) -> Optional[Dict[str, Any]]:
    problem = _text(row.get("problem")).strip()
    solution = _text(row.get("solution")).strip()
    if not problem or not solution:
        return None
    source_id = str(row.get("id") or f"{config}:{index}:{prompt_hash(problem)[:16]}")
    record = make_record(
        record_id=f"math_train::{source_id}",
        dataset=PINNED_SOURCES["math_train"].repo,
        domain="math",
        messages=[{"role": "user", "content": problem}, {"role": "assistant", "content": solution}],
        metadata={
            "source_id": source_id, "config": config, "level": row.get("level"),
            "type": row.get("type", config), "source_repo": PINNED_SOURCES["math_train"].repo,
            "source_revision": PINNED_SOURCES["math_train"].revision, "source_split": "train",
        },
    )
    return record if is_supervised_record(record) else None


def render_mbpp_prompt(text: str, tests: Iterable[str]) -> str:
    return (
        "You are an expert Python programmer, and here is your task:\n"
        f"{text.strip()}\nYour code should pass these tests:\n\n"
        + "\n".join(str(test) for test in tests) + "\n"
    )


def convert_mbpp_train(row: Mapping[str, Any], index: int) -> Optional[Dict[str, Any]]:
    text, code = _text(row.get("text")).strip(), _text(row.get("code")).strip()
    tests = list(row.get("test_list") or [])
    if not text or not code or not tests:
        return None
    source_id = str(row.get("task_id", index))
    record = make_record(
        record_id=f"mbpp_train::{source_id}", dataset="mbpp", domain="code",
        messages=[
            {"role": "user", "content": render_mbpp_prompt(text, tests)},
            {"role": "assistant", "content": f"```python\n{code}\n```"},
        ],
        metadata={
            "source_id": source_id, "task_id": row.get("task_id"), "test_list": tests,
            "test_setup_code": row.get("test_setup_code") or "",
            "source_repo": PINNED_SOURCES["mbpp"].repo,
            "source_revision": PINNED_SOURCES["mbpp"].revision, "source_split": "train",
        },
    )
    return record if is_supervised_record(record) else None


def _clean_kwargs(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, (list, tuple)):
        return []
    return [
        {key: value for key, value in entry.items() if value is not None}
        if isinstance(entry, Mapping) else {}
        for entry in raw
    ]


def convert_if_benchmark(row: Mapping[str, Any], index: int, *, task: str) -> Optional[Dict[str, Any]]:
    prompt = _text(row.get("prompt")).strip()
    if not prompt:
        return None
    pin = PINNED_SOURCES[task]
    key = row.get("key", index)
    return make_record(
        record_id=f"{task}::{key}", dataset=task, domain="instruction",
        messages=[{"role": "user", "content": prompt}],
        metadata={
            "key": key, "prompt": prompt,
            "instruction_id_list": list(row.get("instruction_id_list") or []),
            "kwargs": _clean_kwargs(row.get("kwargs")), "source_repo": pin.repo,
            "source_revision": pin.revision, "source_split": pin.split,
        },
    )


def convert_math500(row: Mapping[str, Any], index: int) -> Optional[Dict[str, Any]]:
    problem = _text(row.get("problem")).strip()
    if not problem:
        return None
    unique_id = str(row.get("unique_id") or index)
    pin = PINNED_SOURCES["math500"]
    return make_record(
        record_id=f"math500::{unique_id}", dataset="math500", domain="math",
        messages=[{"role": "user", "content": problem}],
        metadata={
            "problem": problem, "answer": row.get("answer"), "solution": row.get("solution"),
            "subject": row.get("subject"), "level": row.get("level"), "unique_id": unique_id,
            "source_repo": pin.repo, "source_revision": pin.revision, "source_split": pin.split,
        },
    )


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def convert_mbpp_plus(row: Mapping[str, Any], index: int) -> Optional[Dict[str, Any]]:
    prompt = _text(row.get("prompt")).strip()
    if not prompt:
        return None
    task_id = str(row.get("task_id") or row.get("name") or index)
    metadata = _json_safe(dict(row))
    metadata.update({
        "task_id": task_id, "evalplus_package": EVALPLUS_PACKAGE,
        "evalplus_version": EVALPLUS_VERSION,
        "evalplus_dataset_version": EVALPLUS_DATASET_VERSION,
    })
    return make_record(
        record_id=f"mbpp_plus::{task_id}", dataset="mbpp_plus", domain="code",
        messages=[{"role": "user", "content": prompt}], metadata=metadata,
    )


class PinnedLoaders:
    """The only production network boundary; every input revision is pinned."""

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = cache_dir

    def _dataset(self, pin: SourcePin, *, streaming: bool) -> Iterable[Mapping[str, Any]]:
        from datasets import load_dataset
        kwargs: Dict[str, Any] = {
            "split": pin.split, "revision": pin.revision, "streaming": streaming,
        }
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        return load_dataset(pin.repo, **kwargs) if pin.config is None else load_dataset(pin.repo, pin.config, **kwargs)

    def dolci(self) -> Iterable[Mapping[str, Any]]:
        return self._dataset(PINNED_SOURCES["dolci"], streaming=True)

    def ifeval(self) -> Iterable[Mapping[str, Any]]:
        return self._dataset(PINNED_SOURCES["ifeval"], streaming=False)

    def ifbench(self) -> Iterable[Mapping[str, Any]]:
        return self._dataset(PINNED_SOURCES["ifbench"], streaming=False)

    def math500(self) -> Iterable[Mapping[str, Any]]:
        return self._dataset(PINNED_SOURCES["math500"], streaming=False)

    def math_train(self) -> Iterator[Mapping[str, Any]]:
        from datasets import load_dataset
        pin = PINNED_SOURCES["math_train"]
        for config in MATH_TRAIN_CONFIGS:
            kwargs = {
                "split": pin.split, "revision": pin.revision, "streaming": True,
            }
            if self.cache_dir:
                kwargs["cache_dir"] = self.cache_dir
            for row in load_dataset(pin.repo, config, **kwargs):
                value = dict(row)
                value["_dolci32k_config"] = config
                yield value

    def mbpp_train(self) -> Iterable[Mapping[str, Any]]:
        return self._dataset(PINNED_SOURCES["mbpp"], streaming=False)

    def mbpp_plus(self) -> Iterable[Mapping[str, Any]]:
        try:
            installed = importlib.metadata.version(EVALPLUS_PACKAGE)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"{EVALPLUS_PACKAGE}=={EVALPLUS_VERSION} is required") from error
        if installed != EVALPLUS_VERSION:
            raise RuntimeError(f"expected {EVALPLUS_PACKAGE}=={EVALPLUS_VERSION}, found {installed}")
        from evalplus.data import get_mbpp_plus
        tasks = get_mbpp_plus(version=EVALPLUS_DATASET_VERSION)
        for task_id in sorted(tasks):
            row = dict(tasks[task_id])
            row.setdefault("task_id", task_id)
            yield row
