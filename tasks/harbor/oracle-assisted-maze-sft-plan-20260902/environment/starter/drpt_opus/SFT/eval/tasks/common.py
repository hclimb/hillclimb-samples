"""Shared, dependency-light helpers for generative benchmark evaluators."""

from __future__ import annotations

import importlib.metadata
import re
from typing import Any, Dict, Iterable, Mapping, Optional


RESULT_SCHEMA_VERSION = "drpt.eval.result.v1"

_CHAT_MARKERS = (
    "<|im_end|>",
    "<|im_start|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def clean_model_response(text: str, *, strip_thinking: bool = True) -> str:
    """Remove Qwen reasoning and chat-template residue from a completion.

    Qwen3 prompting disables thinking, but the post-processor remains defensive:
    checkpoints can still emit a complete ``<think>...</think>`` block or just a
    leading ``</think>`` marker. Only model-control markup is removed; the
    remaining punctuation/markdown is preserved for instruction verifiers.
    """
    out = text or ""
    if strip_thinking:
        out = _THINK_BLOCK_RE.sub("", out)
        if "</think>" in out.lower():
            out = re.split(r"</think>", out, flags=re.IGNORECASE)[-1]
        elif re.match(r"^\s*<think>", out, flags=re.IGNORECASE):
            return ""
    for marker in _CHAT_MARKERS:
        index = out.find(marker)
        if index != -1:
            out = out[:index]
    return out.strip()


def render_generation_chat(
    tokenizer: Any,
    user_content: str,
    *,
    enable_thinking: bool,
) -> str:
    """Render one evaluation prompt with an explicit Qwen thinking policy.

    Keeping this in the evaluation package makes the final-benchmark policy
    independent from the legacy validation loader, whose prompts intentionally
    disable thinking.  Non-Qwen templates simply ignore the extra template
    variable.
    """
    from SFT.data.get_val_dataset import ensure_chat_template

    ensure_chat_template(tokenizer)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(enable_thinking),
    )


def require_distribution_version(distribution: str, expected: str) -> str:
    """Return an installed dependency version, failing on missing/drifted pins."""
    try:
        actual = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"{distribution}=={expected} is required for this evaluator; "
            "install the pinned evaluator dependency before running it"
        ) from exc
    if actual != expected:
        raise RuntimeError(
            f"{distribution}=={expected} is required, but version {actual} is installed"
        )
    return actual


def single_source_revision(records: Iterable[Mapping[str, Any]]) -> Optional[str]:
    """Extract one dataset revision from materialized benchmark metadata."""
    revisions = {
        str(record.get("metadata", {}).get("source_revision"))
        for record in records
        if record.get("metadata", {}).get("source_revision")
    }
    if len(revisions) > 1:
        raise ValueError(f"Benchmark records contain multiple source revisions: {sorted(revisions)}")
    return next(iter(revisions), None)


def result_provenance(
    *,
    dataset_repository: str,
    dataset_revision: Optional[str],
    dataset_split: str,
    evaluator: Mapping[str, Any],
    n_tasks: int,
    max_new_tokens: int,
    thinking: bool = False,
) -> Dict[str, Any]:
    """Build the common provenance object stored in new task result JSONs."""
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "dataset": {
            "repository": dataset_repository,
            "revision": dataset_revision,
            "split": dataset_split,
            "n_tasks": int(n_tasks),
        },
        "evaluator": dict(evaluator),
        "generation": {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": int(max_new_tokens),
            "thinking": bool(thinking),
            "thinking_output_stripped": True,
        },
    }
