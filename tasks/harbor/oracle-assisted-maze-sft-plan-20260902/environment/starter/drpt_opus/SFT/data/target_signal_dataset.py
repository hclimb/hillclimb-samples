"""Build the target-gradient dataset for each target-signal objective.

Every mode produces the same feature shape -- a list of per-prompt mappings
that :class:`SFT.train.target_signal.GroupedTargetCollator` can flatten -- so
the trainer's target dataloader is identical no matter which signal is active::

    [{"candidates": [{"input_ids": ..., "labels": ..., "role": ..., "reward": ...}, ...]}, ...]

``nll`` and ``answer_only_ce`` emit exactly one candidate per prompt and are
therefore drop-in replacements for the historical target dataset; the alternate
signals nest the pre-generated trajectories written by
``SFT.data.build_target_candidates``.

Tokenization always goes through :func:`SFT.data.chat_format.encode_assistant_only`,
the same renderer the campaign uses for every other split, so a candidate
trajectory is formatted exactly like the reference one it competes with.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import torch

from SFT.data.chat_format import encode_assistant_only
from SFT.data.target_candidates import (
    MALFORMED_NEGATIVE_STATUSES,
    ORIGIN_GENERATED,
    ORIGIN_REFERENCE,
    ROLE_CORRECT,
    ROLE_INCORRECT,
    CandidateGroup,
    answer_span_is_degenerate,
    final_answer_char_span,
)
from SFT.train.target_signal import (
    ANSWER_ONLY_CE,
    CORRECT_INCORRECT_MARGIN,
    NLL,
    REWARD_WEIGHTED_SFT,
    canonicalize_target_signal_mode,
)

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

POSITIVE_SOURCE_REFERENCE = "reference"
POSITIVE_SOURCE_GENERATED = "generated"
POSITIVE_SOURCES = (POSITIVE_SOURCE_REFERENCE, POSITIVE_SOURCE_GENERATED)


def _decoded_prefix_length(tokenizer, token_ids: Sequence[int], count: int) -> int:
    if count <= 0:
        return 0
    return len(
        tokenizer.decode(
            list(token_ids[:count]),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def _char_to_token_bound(
    tokenizer,
    token_ids: Sequence[int],
    char_index: int,
    *,
    lower: bool,
) -> int:
    """Map a character offset to a token boundary by prefix-length search.

    The decoded length of a token prefix is non-decreasing in the token count,
    so a binary search finds the boundary in ``O(log n)`` decodes.  ``lower``
    returns the last boundary at or before ``char_index`` (an answer start);
    otherwise the first boundary at or after it (an answer end).
    """

    low, high = 0, len(token_ids)
    while low < high:
        middle = (low + high + 1) // 2 if lower else (low + high) // 2
        length = _decoded_prefix_length(tokenizer, token_ids, middle)
        if lower:
            if length <= char_index:
                low = middle
            else:
                high = middle - 1
        else:
            if length >= char_index:
                high = middle
            else:
                low = middle + 1
    return low


def _final_supervised_run(labels: torch.Tensor) -> tuple[int, int]:
    supervised = (labels != IGNORE_INDEX).nonzero(as_tuple=False).flatten().tolist()
    if not supervised:
        raise ValueError("encoded row has no supervised tokens")
    end = supervised[-1] + 1
    start = supervised[-1]
    for index in reversed(supervised[:-1]):
        if index != start - 1:
            break
        start = index
    return start, end


def restrict_labels_to_final_answer(
    encoded: Mapping[str, Any], tokenizer, target: str
) -> torch.Tensor:
    """Return labels supervising only the final-answer tokens of the last turn.

    Falls back to the unmodified labels when the domain has no separable final
    answer, and never returns an all-masked row: if the mapped span is empty the
    whole final assistant turn is kept instead.
    """

    labels = encoded["labels"].clone()
    if answer_span_is_degenerate(target):
        return labels

    input_ids = encoded["input_ids"]
    span_start, span_end = _final_supervised_run(labels)
    span_ids = input_ids[span_start:span_end].tolist()
    span_text = tokenizer.decode(
        span_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    char_start, char_end = final_answer_char_span(target, span_text)

    token_start = span_start + _char_to_token_bound(
        tokenizer, span_ids, char_start, lower=True
    )
    token_end = span_start + _char_to_token_bound(
        tokenizer, span_ids, char_end, lower=False
    )
    if token_end <= token_start:
        return labels

    restricted = torch.full_like(labels, IGNORE_INDEX)
    restricted[token_start:token_end] = labels[token_start:token_end]
    if not bool((restricted != IGNORE_INDEX).any()):
        return labels
    return restricted


def _encode(messages, tokenizer, max_seq_length: int) -> dict[str, torch.Tensor]:
    encoded = encode_assistant_only({"messages": messages}, tokenizer, max_seq_length)
    return {
        "input_ids": encoded["input_ids"],
        "labels": encoded["labels"],
        "attention_mask": encoded["attention_mask"],
    }


def _messages_with_assistant(row: Mapping[str, Any], content: str) -> list[dict[str, str]]:
    messages = [dict(message) for message in row["messages"]]
    for message in reversed(messages):
        if message.get("role") == "assistant":
            message["content"] = content
            return messages
    raise ValueError(f"target row {row.get('id')!r} has no assistant message to replace")


def _reference_content(row: Mapping[str, Any]) -> str:
    for message in reversed(row["messages"]):
        if message.get("role") == "assistant":
            return str(message["content"])
    raise ValueError(f"target row {row.get('id')!r} has no assistant message")


def build_target_signal_features(
    mode: str,
    rows: Sequence[Mapping[str, Any]],
    tokenizer,
    max_seq_length: int,
    *,
    target: str,
    candidate_groups: Mapping[str, CandidateGroup] | None = None,
    incorrect_reward: float = 0.0,
    positive_source: str = POSITIVE_SOURCE_REFERENCE,
    max_candidates_per_prompt: int | None = None,
    align_prompts: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return grouped target features plus a stats record for run metadata.

    ``align_prompts`` restricts every mode to the prompts the margin objective
    can actually use. The margin loss silently drops a prompt whose generations
    were all correct, so without this the margin arm optimizes over a smaller
    D* than its controls and a downstream difference cannot be attributed to
    the objective alone. Turning it on costs prompts but buys a clean contrast.
    """

    canonical_mode = canonicalize_target_signal_mode(mode)
    if positive_source not in POSITIVE_SOURCES:
        raise ValueError(
            f"positive_source must be one of {POSITIVE_SOURCES}, got {positive_source!r}"
        )
    if not rows:
        raise ValueError("target-signal dataset needs at least one target row")
    if incorrect_reward < 0:
        raise ValueError("incorrect_reward must be nonnegative")

    aligned_from = None
    if align_prompts:
        if not candidate_groups:
            raise ValueError(
                "align_prompts needs the candidate artifact to know which prompts "
                "the margin objective can use"
            )
        usable = {
            group_id
            for group_id, group in candidate_groups.items()
            if group.verifiable and group.ranked_negatives()
        }
        aligned_from = len(rows)
        rows = [row for row in rows if str(row.get("id")) in usable]
        if not rows:
            raise ValueError(
                "align_prompts left no target prompts: the candidate artifact has no "
                "verified-wrong trajectory for any prompt"
            )

    def _tag(stats: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if aligned_from is not None:
            stats["aligned_prompt_subset"] = True
            stats["aligned_from_prompts"] = aligned_from
        return stats

    if canonical_mode in (NLL, ANSWER_ONLY_CE):
        features, stats = _build_single_trajectory_features(
            canonical_mode, rows, tokenizer, max_seq_length, target=target
        )
        _tag(stats)
        return features, stats

    if not candidate_groups:
        raise ValueError(
            f"target signal {canonical_mode!r} requires pre-generated candidates; run "
            "python -m SFT.data.build_target_candidates first"
        )
    if canonical_mode == CORRECT_INCORRECT_MARGIN:
        features, stats = _build_margin_features(
            rows,
            tokenizer,
            max_seq_length,
            candidate_groups=candidate_groups,
            positive_source=positive_source,
        )
    else:
        features, stats = _build_reward_weighted_features(
            rows,
            tokenizer,
            max_seq_length,
            candidate_groups=candidate_groups,
            incorrect_reward=incorrect_reward,
            max_candidates_per_prompt=max_candidates_per_prompt,
        )
    _tag(stats)
    return features, stats


def _build_single_trajectory_features(
    canonical_mode: str,
    rows: Sequence[Mapping[str, Any]],
    tokenizer,
    max_seq_length: int,
    *,
    target: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    features: list[dict[str, Any]] = []
    supervised_tokens = 0
    reference_tokens = 0
    for row in rows:
        encoded = _encode(row["messages"], tokenizer, max_seq_length)
        reference_tokens += int((encoded["labels"] != IGNORE_INDEX).sum())
        if canonical_mode == ANSWER_ONLY_CE:
            encoded["labels"] = restrict_labels_to_final_answer(encoded, tokenizer, target)
        supervised_tokens += int((encoded["labels"] != IGNORE_INDEX).sum())
        features.append(
            {
                "candidates": [
                    {**encoded, "role": ROLE_CORRECT, "reward": 1.0, "id": row.get("id")}
                ]
            }
        )

    degenerate = canonical_mode == ANSWER_ONLY_CE and answer_span_is_degenerate(target)
    if degenerate:
        logger.warning(
            "target %s has no separable final answer; answer_only_ce supervises the "
            "whole assistant turn and is numerically identical to nll here",
            target,
        )
    stats = {
        "mode": canonical_mode,
        "target": target,
        "n_prompts": len(features),
        "n_trajectories": len(features),
        "supervised_tokens": supervised_tokens,
        "reference_supervised_tokens": reference_tokens,
        "answer_span_degenerate": bool(degenerate),
    }
    if canonical_mode == ANSWER_ONLY_CE and reference_tokens:
        stats["answer_token_fraction"] = supervised_tokens / reference_tokens
    return features, stats


def _select_positive(
    row: Mapping[str, Any], group: CandidateGroup, positive_source: str
) -> str | None:
    if positive_source == POSITIVE_SOURCE_REFERENCE:
        return _reference_content(row)
    for candidate in group.with_roles(ROLE_CORRECT):
        if candidate.origin == ORIGIN_GENERATED:
            return candidate.content
    return None


def _build_margin_features(
    rows: Sequence[Mapping[str, Any]],
    tokenizer,
    max_seq_length: int,
    *,
    candidate_groups: Mapping[str, CandidateGroup],
    positive_source: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    features: list[dict[str, Any]] = []
    skipped: dict[str, int] = {"no_group": 0, "unverifiable": 0, "no_negative": 0, "no_positive": 0}
    malformed_negatives = 0

    for row in rows:
        group = candidate_groups.get(str(row.get("id")))
        if group is None:
            skipped["no_group"] += 1
            continue
        if not group.verifiable:
            skipped["unverifiable"] += 1
            continue
        negatives = group.ranked_negatives()
        if not negatives:
            skipped["no_negative"] += 1
            continue
        positive_content = _select_positive(row, group, positive_source)
        if positive_content is None:
            skipped["no_positive"] += 1
            continue
        negative = negatives[0]
        if str(negative.verification.get("status", "")) in MALFORMED_NEGATIVE_STATUSES:
            malformed_negatives += 1

        chosen = _encode(
            _messages_with_assistant(row, positive_content), tokenizer, max_seq_length
        )
        rejected = _encode(
            _messages_with_assistant(row, negative.content), tokenizer, max_seq_length
        )
        features.append(
            {
                "candidates": [
                    {**chosen, "role": ROLE_CORRECT, "reward": 1.0, "id": row.get("id")},
                    {**rejected, "role": ROLE_INCORRECT, "reward": 0.0, "id": row.get("id")},
                ]
            }
        )

    if not features:
        raise ValueError(
            "no correct/incorrect pairs survived: "
            f"skipped={skipped}. Regenerate candidates with more samples "
            "(--num_samples) or a higher --temperature so wrong trajectories appear."
        )
    stats = {
        "mode": CORRECT_INCORRECT_MARGIN,
        "n_prompts": len(features),
        "n_trajectories": 2 * len(features),
        "positive_source": positive_source,
        "skipped": skipped,
        "prompt_coverage": len(features) / len(rows),
        "malformed_negative_pairs": malformed_negatives,
    }
    return features, stats


def _build_reward_weighted_features(
    rows: Sequence[Mapping[str, Any]],
    tokenizer,
    max_seq_length: int,
    *,
    candidate_groups: Mapping[str, CandidateGroup],
    incorrect_reward: float,
    max_candidates_per_prompt: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_candidates_per_prompt is not None and max_candidates_per_prompt < 1:
        raise ValueError("max_candidates_per_prompt must be at least 1")

    features: list[dict[str, Any]] = []
    n_trajectories = 0
    n_positive = 0
    n_negative = 0
    n_reference_only = 0

    for row in rows:
        group = candidate_groups.get(str(row.get("id")))
        # The reference is always a correct trajectory, so a prompt with no
        # usable generations still contributes exactly the NLL term it would
        # have contributed before -- reward weighting only adds trajectories.
        selected: list[tuple[str, float, str]] = [
            (_reference_content(row), 1.0, ORIGIN_REFERENCE)
        ]
        if group is not None and group.verifiable:
            for candidate in group.candidates:
                if candidate.origin != ORIGIN_GENERATED:
                    continue
                reward = 1.0 if candidate.role == ROLE_CORRECT else incorrect_reward
                if reward <= 0:
                    # A zero-weight row costs a forward pass and contributes
                    # nothing to either the numerator or the denominator.
                    continue
                selected.append((candidate.content, reward, ORIGIN_GENERATED))
        else:
            n_reference_only += 1

        if max_candidates_per_prompt is not None:
            selected = selected[:max_candidates_per_prompt]

        candidates = []
        for content, reward, origin in selected:
            encoded = _encode(
                _messages_with_assistant(row, content), tokenizer, max_seq_length
            )
            candidates.append(
                {
                    **encoded,
                    "role": ROLE_CORRECT if reward >= 1.0 else ROLE_INCORRECT,
                    "reward": float(reward),
                    "id": row.get("id"),
                }
            )
            n_trajectories += 1
            if reward >= 1.0:
                n_positive += 1
            else:
                n_negative += 1
        features.append({"candidates": candidates})

    stats = {
        "mode": REWARD_WEIGHTED_SFT,
        "n_prompts": len(features),
        "n_trajectories": n_trajectories,
        "n_positive_trajectories": n_positive,
        "n_weighted_negative_trajectories": n_negative,
        "n_reference_only_prompts": n_reference_only,
        "incorrect_reward": float(incorrect_reward),
        "mean_trajectories_per_prompt": n_trajectories / len(features),
    }
    return features, stats


__all__ = [
    "POSITIVE_SOURCES",
    "POSITIVE_SOURCE_GENERATED",
    "POSITIVE_SOURCE_REFERENCE",
    "build_target_signal_features",
    "restrict_labels_to_final_answer",
]
