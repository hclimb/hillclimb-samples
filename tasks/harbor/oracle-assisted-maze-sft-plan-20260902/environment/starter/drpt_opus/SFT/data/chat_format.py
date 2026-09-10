"""Canonical chat/tool conversion and assistant-only label construction.

The Dolci tool-use subset predates the OpenAI-style ``tools`` schema used by
the Qwen3 chat template.  This module keeps the conversion in one place and
provides a Qwen-specific token-boundary masker.  The latter deliberately does
not rely on prefix rendering: Qwen3 changes the rendering of an assistant turn
depending on whether it is the final turn, so prefix-based masking silently
loses intermediate assistant/tool-call supervision.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


# This is part of the derived-tokenization contract. Increment the version
# whenever either the selected window or label-mask semantics change.
SUPERVISION_TRUNCATION_POLICY = "right_then_assistant_tail"
SUPERVISION_TRUNCATION_POLICY_VERSION = 1


def _decode_jsonish(value: Any, *, field_name: str) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return []
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in Dolci {field_name}: {value!r}") from exc


def _canonical_tool(tool: Mapping[str, Any]) -> Dict[str, Any]:
    tool = deepcopy(dict(tool))
    if tool.get("type") == "function" and isinstance(tool.get("function"), Mapping):
        return tool
    return {"type": "function", "function": tool}


def _canonical_tool_call(call: Mapping[str, Any]) -> Dict[str, Any]:
    call = deepcopy(dict(call))
    function = call.get("function")
    if isinstance(function, Mapping):
        result = call
        result.setdefault("type", "function")
        return result

    name = call.get("name") or call.get("function_name")
    if not name:
        raise ValueError(f"Tool call has no function name: {call!r}")
    arguments = call.get("arguments", call.get("parameters", {}))
    return {
        "type": "function",
        "function": {"name": str(name), "arguments": arguments},
    }


def canonicalize_messages_and_tools(
    example: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return OpenAI/Qwen-style messages and tools without mutating ``example``.

    Supported Dolci conversions:

    * system/message ``functions`` -> top-level ``tools``;
    * assistant ``function_calls`` -> ``tool_calls``;
    * ``environment`` role -> ``tool`` role;
    * non-string tool results -> canonical JSON strings.
    """

    raw_messages = example.get("messages")
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes)):
        raise ValueError("messages must be a non-string sequence")
    if not raw_messages:
        raise ValueError("messages field is empty")

    raw_tools = _decode_jsonish(
        example.get("tools", example.get("functions", [])), field_name="tools"
    )
    if raw_tools is None:
        raw_tools = []
    if isinstance(raw_tools, Mapping):
        raw_tools = [raw_tools]
    tools = [_canonical_tool(tool) for tool in raw_tools]

    messages: List[Dict[str, Any]] = []
    for raw in raw_messages:
        if not isinstance(raw, Mapping):
            raise ValueError(f"Message is not an object: {raw!r}")
        message = deepcopy(dict(raw))
        role = str(message.get("role", "")).lower()
        if role == "human":
            role = "user"
        elif role in ("gpt", "model"):
            role = "assistant"
        elif role == "environment":
            role = "tool"
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported message role: {role!r}")

        functions = _decode_jsonish(message.pop("functions", []), field_name="functions")
        if functions:
            if isinstance(functions, Mapping):
                functions = [functions]
            tools.extend(_canonical_tool(function) for function in functions)

        raw_calls = message.pop("function_calls", None)
        if raw_calls is not None:
            raw_calls = _decode_jsonish(raw_calls, field_name="function_calls")
            if isinstance(raw_calls, Mapping):
                raw_calls = [raw_calls]
            message["tool_calls"] = [
                _canonical_tool_call(call) for call in (raw_calls or [])
            ]
        elif message.get("tool_calls") is not None:
            calls = _decode_jsonish(message["tool_calls"], field_name="tool_calls")
            if isinstance(calls, Mapping):
                calls = [calls]
            message["tool_calls"] = [_canonical_tool_call(call) for call in calls]

        content = message.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            if role != "tool":
                raise ValueError(
                    f"Non-string content is supported only for tool results, got {role!r}"
                )
            content = json.dumps(content, ensure_ascii=False, sort_keys=True)

        canonical = {"role": role, "content": content}
        if message.get("tool_calls"):
            canonical["tool_calls"] = message["tool_calls"]
        if message.get("tool_call_id") is not None:
            canonical["tool_call_id"] = message["tool_call_id"]
        if message.get("reasoning_content") is not None:
            canonical["reasoning_content"] = message["reasoning_content"]
        messages.append(canonical)

    return messages, tools


def _apply_chat_template(tokenizer, messages, tools, *, tokenize: bool, **kwargs):
    template_kwargs = dict(kwargs)
    if tools:
        template_kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, tokenize=tokenize, **template_kwargs)


def _as_token_list(value: Any) -> List[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Expected one tokenized conversation")
        value = value[0]
    return [int(token) for token in value]


def _qwen_assistant_spans(tokenizer, token_ids: Sequence[int]) -> List[Tuple[int, int]]:
    """Return content spans for Qwen ``im_start assistant`` blocks.

    End offsets include ``<|im_end|>`` so the model learns to terminate the
    assistant turn.  System, user, and tool-response blocks remain masked.
    """

    start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if start_id is None or end_id is None or start_id == unk_id or end_id == unk_id:
        return []
    role_ids = _as_token_list(tokenizer.encode("assistant\n", add_special_tokens=False))
    if not role_ids:
        return []

    spans: List[Tuple[int, int]] = []
    index = 0
    total = len(token_ids)
    while index < total:
        if token_ids[index] != start_id:
            index += 1
            continue
        role_start = index + 1
        role_end = role_start + len(role_ids)
        if list(token_ids[role_start:role_end]) != role_ids:
            index += 1
            continue
        end = role_end
        while end < total and token_ids[end] != end_id:
            end += 1
        if end >= total:
            # A right-truncated final assistant block: supervise the retained
            # answer prefix even though the terminator is absent.
            if role_end < total:
                spans.append((role_end, total))
            break
        spans.append((role_end, end + 1))
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


def _native_assistant_mask(tokenizer, messages, tools):
    """Use Hugging Face ``{% generation %}`` masks when available."""

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


def _prefix_assistant_spans(tokenizer, messages, tools, full_text):
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


def _token_diff_span(full_ids, comparison_ids):
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


def _difference_assistant_spans(tokenizer, messages, tools, full_ids):
    spans: List[Tuple[int, int]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        comparison = deepcopy(list(messages))
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
            comparison_ids = _as_token_list(
                tokenizer.encode(rendered, add_special_tokens=False)
                if isinstance(rendered, str)
                else rendered
            )
        except Exception:
            return None
        span = _token_diff_span(full_ids, comparison_ids)
        if span is not None:
            spans.append(span)
    return _merge_spans(spans) or None


def resolve_assistant_token_spans(example: Mapping[str, Any], tokenizer):
    """Render a record once and resolve all assistant-token spans.

    The returned spans are offsets in the untruncated token sequence. Both
    training and offline diagnostics call this function, preventing their
    masking or truncation behavior from drifting apart.
    """

    from SFT.data.get_val_dataset import ensure_chat_template

    ensure_chat_template(tokenizer)
    messages, tools = canonicalize_messages_and_tools(example)
    if not any(message.get("role") == "assistant" for message in messages):
        raise ValueError("Conversation contains no assistant turn")

    template = str(getattr(tokenizer, "chat_template", ""))
    full_ids: Optional[List[int]] = None
    spans: Optional[List[Tuple[int, int]]] = None
    method = ""
    if "<|im_start|>" in template:
        rendered = _apply_chat_template(
            tokenizer,
            messages,
            tools,
            tokenize=True,
            add_generation_prompt=False,
        )
        if isinstance(rendered, str):
            rendered = tokenizer.encode(rendered, add_special_tokens=False)
        full_ids = _as_token_list(rendered)
        spans = _qwen_assistant_spans(tokenizer, full_ids)
        method = "qwen_im_delimiters"
    else:
        native = _native_assistant_mask(tokenizer, messages, tools)
        if native is not None:
            full_ids, spans = native
            method = "native_assistant_mask"

    if not spans:
        full_text = _apply_chat_template(
            tokenizer,
            messages,
            tools,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not isinstance(full_text, str):
            raise ValueError("Chat template did not return text for span diagnostics")
        if full_ids is None:
            full_ids = _as_token_list(
                tokenizer.encode(full_text, add_special_tokens=False)
            )
        spans = _prefix_assistant_spans(tokenizer, messages, tools, full_text)
        method = "prefix_boundaries"

    if not spans:
        assert full_ids is not None
        spans = _difference_assistant_spans(tokenizer, messages, tools, full_ids)
        method = "full_conversation_token_diff"
    if not spans:
        raise ValueError("Could not resolve assistant token spans")

    assert full_ids is not None
    spans = _merge_spans(spans)
    if any(start < 0 or end > len(full_ids) for start, end in spans):
        raise ValueError("Assistant span falls outside rendered token sequence")
    return full_ids, spans, method


def _retained_assistant_tokens(
    spans: Sequence[Tuple[int, int]], window_start: int, window_end: int
) -> int:
    return sum(
        max(0, min(end, window_end) - max(start, window_start))
        for start, end in spans
    )


def prepare_assistant_supervision(
    example: Mapping[str, Any], tokenizer, max_seq_length: int
) -> Dict[str, Any]:
    """Return the exact input window, label mask, and truncation diagnostics.

    Version 1 preserves historical right truncation whenever it leaves at
    least one assistant label. Only a zero-label right-truncation case uses a
    fallback: a contiguous window ending at the final assistant span. This
    deterministically removes old prompt/context tokens while retaining the
    assistant tail, without changing raw example membership or ordering.
    """

    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")
    full_ids, spans, span_method = resolve_assistant_token_spans(example, tokenizer)
    assistant_tokens_total = sum(end - start for start, end in spans)
    if assistant_tokens_total <= 0:
        raise ValueError("Conversation contains no assistant tokens")

    right_end = min(len(full_ids), max_seq_length)
    right_retained = _retained_assistant_tokens(spans, 0, right_end)
    right_zero = right_retained == 0
    window_start = 0
    window_end = right_end
    fallback_applied = False
    if right_zero and len(full_ids) > max_seq_length:
        # End at the final assistant span rather than the full conversation
        # tail: unusual trailing user/tool context cannot evict supervision.
        _, final_assistant_end = max(spans, key=lambda span: (span[1], span[0]))
        window_end = min(len(full_ids), final_assistant_end)
        window_start = max(0, window_end - max_seq_length)
        fallback_applied = window_start > 0

    input_ids = full_ids[window_start:window_end]
    label_mask = [False] * len(input_ids)
    for start, end in spans:
        local_start = max(start, window_start) - window_start
        local_end = min(end, window_end) - window_start
        if local_start < local_end:
            label_mask[local_start:local_end] = [True] * (local_end - local_start)
    retained = sum(label_mask)
    final_zero = retained == 0
    final_assistant_end = max(end for _, end in spans)
    diagnostics = {
        "untruncated_length": len(full_ids),
        "max_seq_len": int(max_seq_length),
        "exceeds_max_seq_len": len(full_ids) > max_seq_length,
        "assistant_tokens_total": assistant_tokens_total,
        "assistant_tokens_retained_by_right_truncation": right_retained,
        "assistant_truncated_by_right_truncation": right_retained < assistant_tokens_total,
        "zero_supervised_after_right_truncation": right_zero,
        "supervision_preserving_fallback_applied": fallback_applied,
        "assistant_tokens_retained": retained,
        "assistant_truncated": retained < assistant_tokens_total,
        "zero_supervised_after_truncation": final_zero,
        "assistant_end_retained": window_start < final_assistant_end <= window_end,
        "truncation_window_start": window_start,
        "truncation_window_end": window_end,
        "truncation_policy": SUPERVISION_TRUNCATION_POLICY,
        "truncation_policy_version": SUPERVISION_TRUNCATION_POLICY_VERSION,
        "assistant_span_method": span_method,
        "diagnostic_error": None,
    }
    return {"input_ids": input_ids, "label_mask": label_mask, "diagnostics": diagnostics}


def encode_assistant_only(
    example: Mapping[str, Any], tokenizer, max_seq_length: int
) -> Dict[str, torch.Tensor | int | bool]:
    """Render one conversation and label every assistant span.

    Qwen templates are recognized by their native ``im_start``/``im_end``
    delimiters.  Other templates retain the historical prefix-boundary path.
    Diagnostic fields are prefixed with ``_tokenization_`` so callers can
    aggregate and remove them before passing batches to a model.
    """

    prepared = prepare_assistant_supervision(example, tokenizer, max_seq_length)
    input_ids_list = prepared["input_ids"]
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    label_mask = torch.tensor(prepared["label_mask"], dtype=torch.bool)
    labels[label_mask] = input_ids[label_mask]
    if not bool((labels != -100).any()):
        raise ValueError("Supervision-preserving truncation left no assistant label")

    diagnostics = prepared["diagnostics"]
    result = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones_like(input_ids),
    }
    for name in (
        "untruncated_length",
        "assistant_end_retained",
        "zero_supervised_after_right_truncation",
        "supervision_preserving_fallback_applied",
        "zero_supervised_after_truncation",
        "truncation_window_start",
        "truncation_window_end",
        "truncation_policy",
        "truncation_policy_version",
    ):
        result[f"_tokenization_{name}"] = diagnostics[name]
    result["_tokenization_was_truncated"] = diagnostics["exceeds_max_seq_len"]
    return result


TOKENIZATION_DIAGNOSTIC_COLUMNS = (
    "_tokenization_untruncated_length",
    "_tokenization_was_truncated",
    "_tokenization_assistant_end_retained",
    "_tokenization_zero_supervised_after_right_truncation",
    "_tokenization_supervision_preserving_fallback_applied",
    "_tokenization_zero_supervised_after_truncation",
    "_tokenization_truncation_window_start",
    "_tokenization_truncation_window_end",
    "_tokenization_truncation_policy",
    "_tokenization_truncation_policy_version",
)
