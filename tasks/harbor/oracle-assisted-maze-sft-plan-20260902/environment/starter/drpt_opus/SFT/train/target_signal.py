"""Runtime primitives for target-gradient signal objectives.

The module deliberately has no dependency on ``transformers``.  A target
dataset can therefore keep one item per prompt, with one or more tokenized
candidate trajectories nested under ``candidates``.  :class:`GroupedTargetCollator`
flattens only at collation time and records enough metadata to reconstruct the
prompt groups.

All token objectives use the causal-LM convention used by Hugging Face models:
``logits[:, :-1]`` predicts ``labels[:, 1:]`` and ``-100`` labels are ignored.
Consequently ``nll`` is the ordinary global token-mean causal cross entropy.
``answer_only_ce`` intentionally uses the same computation; its different
semantics come from supplying labels in which only final-answer tokens are
unmasked (use ``label_key`` on the collator when those labels have a separate
field name).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


NLL = "nll"
ANSWER_ONLY_CE = "answer_only_ce"
CORRECT_INCORRECT_MARGIN = "correct_incorrect_margin"
REWARD_WEIGHTED_SFT = "reward_weighted_sft"

TARGET_SIGNAL_MODES = (
    NLL,
    ANSWER_ONLY_CE,
    CORRECT_INCORRECT_MARGIN,
    REWARD_WEIGHTED_SFT,
)

IGNORE_INDEX = -100

GROUP_IDS_KEY = "group_ids"
ROLES_KEY = "roles"
REWARDS_KEY = "rewards"

ROLE_REJECTED = -1
ROLE_NEUTRAL = 0
ROLE_CHOSEN = 1
# Correct/incorrect are semantic aliases used by target-data builders.
ROLE_INCORRECT = ROLE_REJECTED
ROLE_CORRECT = ROLE_CHOSEN


_MODE_ALIASES = {
    "ce": NLL,
    "answer_only": ANSWER_ONLY_CE,
    "margin": CORRECT_INCORRECT_MARGIN,
    "reward_weighted": REWARD_WEIGHTED_SFT,
}

_ROLE_ALIASES = {
    "chosen": ROLE_CHOSEN,
    "correct": ROLE_CHOSEN,
    "positive": ROLE_CHOSEN,
    "preferred": ROLE_CHOSEN,
    "rejected": ROLE_REJECTED,
    "incorrect": ROLE_REJECTED,
    "negative": ROLE_REJECTED,
    "dispreferred": ROLE_REJECTED,
    "neutral": ROLE_NEUTRAL,
    "reference": ROLE_NEUTRAL,
    "trajectory": ROLE_NEUTRAL,
}


def canonicalize_target_signal_mode(mode: str) -> str:
    """Return a canonical target-signal mode or raise a useful error."""

    if not isinstance(mode, str):
        raise TypeError(f"target-signal mode must be a string, got {type(mode)!r}")
    normalized = mode.strip().lower().replace("-", "_")
    normalized = _MODE_ALIASES.get(normalized, normalized)
    if normalized not in TARGET_SIGNAL_MODES:
        choices = ", ".join(TARGET_SIGNAL_MODES)
        raise ValueError(f"unknown target-signal mode {mode!r}; expected one of: {choices}")
    return normalized


def encode_candidate_role(role: str | int | Tensor | None) -> int:
    """Encode common role spellings as ``-1``/``0``/``1``."""

    if role is None:
        return ROLE_NEUTRAL
    if isinstance(role, Tensor):
        if role.numel() != 1:
            raise ValueError("a candidate role tensor must contain exactly one value")
        role = role.item()
    if isinstance(role, str):
        normalized = role.strip().lower().replace("-", "_")
        if normalized not in _ROLE_ALIASES:
            choices = ", ".join(sorted(_ROLE_ALIASES))
            raise ValueError(f"unknown candidate role {role!r}; expected one of: {choices}")
        return _ROLE_ALIASES[normalized]
    if isinstance(role, bool) or not isinstance(role, int):
        raise TypeError(f"candidate role must be a string or integer, got {type(role)!r}")
    if role not in (ROLE_REJECTED, ROLE_NEUTRAL, ROLE_CHOSEN):
        raise ValueError("integer candidate roles must be -1 (rejected), 0, or 1 (chosen)")
    return role


def _as_cpu_1d_long(value: Any, *, name: str) -> Tensor:
    result = torch.as_tensor(value, dtype=torch.long).detach().cpu()
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {tuple(result.shape)}")
    return result


@dataclass
class GroupedTargetCollator:
    """Pad and flatten grouped target trajectories.

    Each outer feature remains one logical prompt in the dataset and may use
    either of these forms::

        {"candidates": [{"input_ids": ..., "labels": ..., "role": ...}, ...]}
        {"input_ids": ..., "labels": ...}  # one trajectory for this prompt

    ``trajectories`` is accepted as an alias for ``candidates``.  Candidate
    metadata overrides feature-level defaults.  The returned tensors have
    ``N = sum(len(candidates))`` rows and contain ``group_ids`` in ``[0, B)``.
    Roles are encoded as chosen/correct ``1``, rejected/incorrect ``-1``, and
    neutral ``0``.  Rewards default to ``1.0``.
    """

    pad_token_id: int
    label_pad_token_id: int = IGNORE_INDEX
    label_key: str = "labels"
    candidates_key: str = "candidates"
    padding_side: str = "right"
    pad_to_multiple_of: int | None = None

    def __post_init__(self) -> None:
        if self.padding_side not in ("left", "right"):
            raise ValueError("padding_side must be 'left' or 'right'")
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive")

    def _candidates(self, feature: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        if self.candidates_key in feature:
            candidates = feature[self.candidates_key]
        elif self.candidates_key == "candidates" and "trajectories" in feature:
            candidates = feature["trajectories"]
        else:
            candidates = (feature,)

        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise TypeError(f"{self.candidates_key!r} must be a sequence of mappings")
        if not candidates:
            raise ValueError("each target prompt must contain at least one candidate trajectory")
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise TypeError("every candidate trajectory must be a mapping")
        return candidates

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Tensor]:
        if not features:
            raise ValueError("cannot collate an empty target batch")

        flat_input_ids: list[Tensor] = []
        flat_labels: list[Tensor] = []
        flat_attention_masks: list[Tensor] = []
        group_ids: list[int] = []
        roles: list[int] = []
        rewards: list[float] = []

        for group_id, feature in enumerate(features):
            if not isinstance(feature, Mapping):
                raise TypeError("every target feature must be a mapping")
            for candidate in self._candidates(feature):
                if "input_ids" not in candidate:
                    raise KeyError("candidate trajectory is missing 'input_ids'")
                if self.label_key not in candidate:
                    raise KeyError(
                        f"candidate trajectory is missing {self.label_key!r}; labels must "
                        "already mask prompt/padding tokens with -100"
                    )

                input_ids = _as_cpu_1d_long(candidate["input_ids"], name="input_ids")
                labels = _as_cpu_1d_long(candidate[self.label_key], name=self.label_key)
                if input_ids.numel() == 0:
                    raise ValueError("candidate trajectories may not be empty")
                if labels.shape != input_ids.shape:
                    raise ValueError(
                        "input_ids and labels must have the same length, got "
                        f"{input_ids.numel()} and {labels.numel()}"
                    )

                if "attention_mask" in candidate:
                    attention_mask = _as_cpu_1d_long(
                        candidate["attention_mask"], name="attention_mask"
                    )
                    if attention_mask.shape != input_ids.shape:
                        raise ValueError(
                            "attention_mask and input_ids must have the same length, got "
                            f"{attention_mask.numel()} and {input_ids.numel()}"
                        )
                else:
                    attention_mask = torch.ones_like(input_ids)

                role = candidate.get("role", feature.get("role", ROLE_NEUTRAL))
                reward = candidate.get("reward", feature.get("reward", 1.0))
                if isinstance(reward, Tensor):
                    if reward.numel() != 1:
                        raise ValueError("a candidate reward tensor must contain exactly one value")
                    reward = reward.item()
                try:
                    reward = float(reward)
                except (TypeError, ValueError) as exc:
                    raise TypeError(f"candidate reward must be numeric, got {reward!r}") from exc
                if not torch.isfinite(torch.tensor(reward)):
                    raise ValueError("candidate rewards must be finite")

                flat_input_ids.append(input_ids)
                flat_labels.append(labels)
                flat_attention_masks.append(attention_mask)
                group_ids.append(group_id)
                roles.append(encode_candidate_role(role))
                rewards.append(reward)

        max_length = max(ids.numel() for ids in flat_input_ids)
        if self.pad_to_multiple_of is not None:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple

        num_candidates = len(flat_input_ids)
        input_ids_batch = torch.full(
            (num_candidates, max_length), int(self.pad_token_id), dtype=torch.long
        )
        labels_batch = torch.full(
            (num_candidates, max_length), int(self.label_pad_token_id), dtype=torch.long
        )
        attention_mask_batch = torch.zeros((num_candidates, max_length), dtype=torch.long)

        for row, (input_ids, labels, attention_mask) in enumerate(
            zip(flat_input_ids, flat_labels, flat_attention_masks)
        ):
            length = input_ids.numel()
            start = 0 if self.padding_side == "right" else max_length - length
            end = start + length
            input_ids_batch[row, start:end] = input_ids
            labels_batch[row, start:end] = labels
            attention_mask_batch[row, start:end] = attention_mask

        return {
            "input_ids": input_ids_batch,
            "labels": labels_batch,
            "attention_mask": attention_mask_batch,
            GROUP_IDS_KEY: torch.tensor(group_ids, dtype=torch.long),
            ROLES_KEY: torch.tensor(roles, dtype=torch.long),
            REWARDS_KEY: torch.tensor(rewards, dtype=torch.float32),
        }


def _validate_logits_and_labels(logits: Tensor, labels: Tensor) -> None:
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [batch, sequence, vocab], got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [batch, sequence], got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "logits and labels must agree in batch and sequence dimensions, got "
            f"{tuple(logits.shape[:2])} and {tuple(labels.shape)}"
        )
    if logits.shape[1] < 2:
        raise ValueError("causal token losses require a sequence length of at least two")
    if logits.shape[-1] <= 0:
        raise ValueError("logits must have a non-empty vocabulary dimension")


def shifted_logits_and_labels(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Return the exact causal prediction/target alignment.

    Returned shapes are ``[B, S - 1, V]`` and ``[B, S - 1]``.  The first label
    is never a prediction target and the final logit is never consumed.
    """

    _validate_logits_and_labels(logits, labels)
    return logits[:, :-1, :].contiguous(), labels[:, 1:].to(logits.device).contiguous()


def shifted_token_mask(labels: Tensor, *, ignore_index: int = IGNORE_INDEX) -> Tensor:
    """Return the supervised causal-token mask with shape ``[B, S - 1]``."""

    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [batch, sequence], got {tuple(labels.shape)}")
    if labels.shape[1] < 2:
        raise ValueError("causal token losses require a sequence length of at least two")
    return labels[:, 1:] != ignore_index


def shifted_token_counts(labels: Tensor, *, ignore_index: int = IGNORE_INDEX) -> Tensor:
    """Count supervised shifted tokens independently for each response."""

    return shifted_token_mask(labels, ignore_index=ignore_index).sum(dim=-1)


def shifted_token_logps(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> Tensor:
    """Return selected next-token log probabilities, zero at ignored positions.

    The returned tensor has shape ``[B, S - 1]``.  Use
    :func:`shifted_token_mask` when the distinction between an ignored zero and
    a genuine log probability is needed.

    The log probability is formed as ``gathered_logit - logsumexp(logits)``
    rather than by gathering from a full ``log_softmax``.  The two are
    mathematically identical, but ``log_softmax`` saves a second ``[B, S, V]``
    tensor for backward, and at Qwen3's 151936-token vocabulary that copy alone
    is gigabytes per target microbatch.
    """

    shifted_logits, shifted_labels = shifted_logits_and_labels(logits, labels)
    mask = shifted_labels != ignore_index
    safe_labels = shifted_labels.masked_fill(~mask, 0)

    valid_labels = safe_labels[mask]
    if valid_labels.numel() and (
        bool((valid_labels < 0).any()) or bool((valid_labels >= logits.shape[-1]).any())
    ):
        raise ValueError("a supervised label lies outside the logits vocabulary")

    gathered = shifted_logits.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    selected = gathered - torch.logsumexp(shifted_logits, dim=-1)
    return selected.masked_fill(~mask, 0.0)


@dataclass(frozen=True)
class ResponseLogpStats:
    """Per-response causal log-probability statistics."""

    token_logps: Tensor
    token_mask: Tensor
    sum_logps: Tensor
    mean_logps: Tensor
    token_counts: Tensor


def response_logp_stats(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> ResponseLogpStats:
    """Compute token, sum, and length-normalized response log probabilities."""

    token_logps = shifted_token_logps(logits, labels, ignore_index=ignore_index)
    token_mask = shifted_token_mask(labels.to(logits.device), ignore_index=ignore_index)
    token_counts = token_mask.sum(dim=-1)
    sum_logps = token_logps.sum(dim=-1)
    # Empty rows remain neutral here; objectives which need a response score
    # (the margin loss) reject empty selected rows explicitly below.
    mean_logps = sum_logps / token_counts.clamp_min(1).to(sum_logps.dtype)
    return ResponseLogpStats(
        token_logps=token_logps,
        token_mask=token_mask,
        sum_logps=sum_logps,
        mean_logps=mean_logps,
        token_counts=token_counts,
    )


def causal_token_mean_nll(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> Tensor:
    """Ordinary causal-LM NLL: one global mean over supervised tokens.

    This is the original target-gradient objective.  It intentionally delegates
    to PyTorch cross entropy with the same shift and ignore semantics used by
    causal language models, rather than averaging per-response losses.
    """

    shifted_logits, shifted_labels = shifted_logits_and_labels(logits, labels)
    if not bool((shifted_labels != ignore_index).any()):
        raise ValueError("NLL is undefined because the batch has no supervised shifted tokens")
    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )


def _coerce_response_vector(
    value: Tensor | Sequence[float] | Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> Tensor:
    result = torch.as_tensor(value, device=device, dtype=dtype)
    if result.ndim != 1 or result.numel() != batch_size:
        raise ValueError(f"{name} must have shape [{batch_size}], got {tuple(result.shape)}")
    return result


def _validated_nonnegative_weights(
    response_weights: Tensor | Sequence[float],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> Tensor:
    weights = _coerce_response_vector(
        response_weights,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        name=name,
    )
    if not bool(torch.isfinite(weights).all()):
        raise ValueError(f"{name} must be finite")
    if bool((weights < 0).any()):
        raise ValueError(f"{name} must be nonnegative")
    return weights


def weighted_token_nll_sum(
    logits: Tensor,
    labels: Tensor,
    response_weights: Tensor | Sequence[float],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> Tensor:
    """Return ``sum_i weight_i * sum_t NLL_it`` without normalization."""

    stats = response_logp_stats(logits, labels, ignore_index=ignore_index)
    weights = _validated_nonnegative_weights(
        response_weights,
        batch_size=labels.shape[0],
        device=stats.sum_logps.device,
        dtype=stats.sum_logps.dtype,
        name="response_weights",
    )
    return torch.sum(weights * -stats.sum_logps)


def weighted_token_normalizer(
    labels: Tensor,
    response_weights: Tensor | Sequence[float],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> Tensor:
    """Return ``sum_i weight_i * supervised_token_count_i``."""

    counts = shifted_token_counts(labels, ignore_index=ignore_index)
    if isinstance(response_weights, Tensor):
        device = response_weights.device
        dtype = response_weights.dtype if response_weights.is_floating_point() else torch.float32
    else:
        device = labels.device
        dtype = torch.float32
    counts = counts.to(device=device)
    weights = _validated_nonnegative_weights(
        response_weights,
        batch_size=labels.shape[0],
        device=device,
        dtype=dtype,
        name="response_weights",
    )
    return torch.sum(weights * counts.to(dtype))


def reward_weighted_sft_loss(
    logits: Tensor,
    labels: Tensor,
    rewards: Tensor | Sequence[float],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> Tensor:
    """Reward-weighted token mean ``sum r*NLL / sum r*token_count``.

    Weighting both numerator and token denominator makes the scale independent
    of candidate length and preserves the original NLL when every reward is 1.
    """

    stats = response_logp_stats(logits, labels, ignore_index=ignore_index)
    weights = _validated_nonnegative_weights(
        rewards,
        batch_size=labels.shape[0],
        device=stats.sum_logps.device,
        dtype=stats.sum_logps.dtype,
        name="rewards",
    )
    normalizer = torch.sum(weights * stats.token_counts.to(weights.dtype))
    if not bool(normalizer > 0):
        raise ValueError("reward-weighted SFT needs positive reward mass on supervised tokens")
    numerator = torch.sum(weights * -stats.sum_logps)
    return numerator / normalizer


def correct_incorrect_margin_loss(
    logits: Tensor,
    labels: Tensor,
    group_ids: Tensor | Sequence[int],
    roles: Tensor | Sequence[int],
    *,
    beta: float = 1.0,
    margin: float = 0.0,
    ignore_index: int = IGNORE_INDEX,
) -> Tensor:
    """Pairwise softplus loss over response-mean log probabilities.

    For every group this expects exactly one chosen/correct response and one
    rejected/incorrect response.  Neutral responses are allowed and ignored.
    With ``delta = mean_logp_correct - mean_logp_incorrect``, each pair
    contributes ``softplus(beta * (margin - delta))``.
    """

    if not isinstance(beta, (int, float)) or not torch.isfinite(torch.tensor(float(beta))):
        raise ValueError("beta must be finite")
    if beta <= 0:
        raise ValueError("beta must be positive")
    if not isinstance(margin, (int, float)) or not torch.isfinite(torch.tensor(float(margin))):
        raise ValueError("margin must be finite")

    stats = response_logp_stats(logits, labels, ignore_index=ignore_index)
    batch_size = labels.shape[0]
    ids = _coerce_response_vector(
        group_ids,
        batch_size=batch_size,
        device=stats.mean_logps.device,
        dtype=torch.long,
        name="group_ids",
    )
    encoded_roles = _coerce_response_vector(
        roles,
        batch_size=batch_size,
        device=stats.mean_logps.device,
        dtype=torch.long,
        name="roles",
    )
    if bool(
        ((encoded_roles != ROLE_CHOSEN)
         & (encoded_roles != ROLE_REJECTED)
         & (encoded_roles != ROLE_NEUTRAL)).any()
    ):
        raise ValueError("roles must contain only -1 (rejected), 0, or 1 (chosen)")

    pair_losses: list[Tensor] = []
    for group_id in torch.unique(ids, sorted=True):
        in_group = ids == group_id
        chosen_rows = torch.nonzero(in_group & (encoded_roles == ROLE_CHOSEN), as_tuple=False).flatten()
        rejected_rows = torch.nonzero(
            in_group & (encoded_roles == ROLE_REJECTED), as_tuple=False
        ).flatten()
        if chosen_rows.numel() != 1 or rejected_rows.numel() != 1:
            raise ValueError(
                f"group {int(group_id.item())} must have exactly one chosen and one rejected "
                f"response; found {chosen_rows.numel()} and {rejected_rows.numel()}"
            )
        chosen_row = chosen_rows[0]
        rejected_row = rejected_rows[0]
        if stats.token_counts[chosen_row] == 0 or stats.token_counts[rejected_row] == 0:
            raise ValueError("chosen and rejected responses must each have a supervised token")
        delta = stats.mean_logps[chosen_row] - stats.mean_logps[rejected_row]
        pair_losses.append(F.softplus(float(beta) * (float(margin) - delta)))

    if not pair_losses:
        raise ValueError("margin loss needs at least one correct/incorrect pair")
    return torch.stack(pair_losses).mean()


def compute_target_signal_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    mode: str = NLL,
    group_ids: Tensor | Sequence[int] | None = None,
    roles: Tensor | Sequence[int] | None = None,
    rewards: Tensor | Sequence[float] | None = None,
    beta: float = 1.0,
    margin: float = 0.0,
    ignore_index: int = IGNORE_INDEX,
) -> Tensor:
    """Dispatch one of the four canonical target-gradient losses."""

    canonical_mode = canonicalize_target_signal_mode(mode)
    if canonical_mode in (NLL, ANSWER_ONLY_CE):
        return causal_token_mean_nll(logits, labels, ignore_index=ignore_index)
    if canonical_mode == CORRECT_INCORRECT_MARGIN:
        if group_ids is None or roles is None:
            raise ValueError("correct_incorrect_margin requires group_ids and roles")
        return correct_incorrect_margin_loss(
            logits,
            labels,
            group_ids,
            roles,
            beta=beta,
            margin=margin,
            ignore_index=ignore_index,
        )
    if rewards is None:
        raise ValueError("reward_weighted_sft requires rewards")
    return reward_weighted_sft_loss(logits, labels, rewards, ignore_index=ignore_index)


def compute_target_signal_loss_from_batch(
    logits: Tensor,
    batch: Mapping[str, Any],
    *,
    mode: str = NLL,
    beta: float = 1.0,
    margin: float = 0.0,
    ignore_index: int = IGNORE_INDEX,
) -> Tensor:
    """Convenience dispatcher for a :class:`GroupedTargetCollator` batch."""

    if "labels" not in batch:
        raise KeyError("target batch is missing 'labels'")
    return compute_target_signal_loss(
        logits,
        batch["labels"],
        mode=mode,
        group_ids=batch.get(GROUP_IDS_KEY),
        roles=batch.get(ROLES_KEY),
        rewards=batch.get(REWARDS_KEY),
        beta=beta,
        margin=margin,
        ignore_index=ignore_index,
    )


def model_inputs_from_target_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Strip target-only metadata before forwarding a grouped batch to a model."""

    result: dict[str, Any] = {}
    for key in ("input_ids", "attention_mask", "position_ids", "token_type_ids"):
        if key in batch:
            result[key] = batch[key]
    if "input_ids" not in result:
        raise KeyError("target batch is missing 'input_ids'")
    return result


def token_mean_normalization_weights(
    labels: Tensor,
    response_weights: Tensor | Sequence[float] | None = None,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> Tensor:
    """Return each response's share of a logical weighted-token denominator.

    The values sum to one.  Their sum over a microbatch is the multiplier that
    converts that microbatch's local token-mean loss into its contribution to
    the unsliced logical-batch loss.  Pass rewards as ``response_weights`` for
    reward-weighted SFT; omit them for ordinary NLL/answer-only CE.
    """

    counts = shifted_token_counts(labels, ignore_index=ignore_index)
    if response_weights is None:
        weights = torch.ones(counts.shape[0], device=counts.device, dtype=torch.float32)
    else:
        if isinstance(response_weights, Tensor):
            device = response_weights.device
            dtype = (
                response_weights.dtype if response_weights.is_floating_point() else torch.float32
            )
        else:
            device = counts.device
            dtype = torch.float32
        counts = counts.to(device)
        weights = _validated_nonnegative_weights(
            response_weights,
            batch_size=counts.shape[0],
            device=device,
            dtype=dtype,
            name="response_weights",
        )
    mass = weights * counts.to(weights.dtype)
    total_mass = mass.sum()
    if not bool(total_mass > 0):
        raise ValueError("normalization needs positive weight on at least one supervised token")
    return mass / total_mass


def iter_logical_slices(num_rows: int, microbatch_rows: int) -> Iterator[tuple[int, int]]:
    """Yield half-open flat row slices without dropping a remainder."""

    if isinstance(num_rows, bool) or not isinstance(num_rows, int) or num_rows < 0:
        raise ValueError("num_rows must be a nonnegative integer")
    if isinstance(microbatch_rows, bool) or not isinstance(microbatch_rows, int):
        raise ValueError("microbatch_rows must be a positive integer")
    if microbatch_rows <= 0:
        raise ValueError("microbatch_rows must be a positive integer")
    for start in range(0, num_rows, microbatch_rows):
        yield start, min(start + microbatch_rows, num_rows)


def _contiguous_group_runs(group_ids: Tensor | Sequence[int]) -> list[tuple[int, int]]:
    ids_tensor = torch.as_tensor(group_ids)
    if ids_tensor.ndim != 1:
        raise ValueError(f"group_ids must be one-dimensional, got {tuple(ids_tensor.shape)}")
    ids = ids_tensor.detach().cpu().tolist()
    if not ids:
        return []

    runs: list[tuple[int, int]] = []
    seen: set[Any] = set()
    start = 0
    current = ids[0]
    for index in range(1, len(ids) + 1):
        at_end = index == len(ids)
        next_id = None if at_end else ids[index]
        if at_end or next_id != current:
            if current in seen:
                raise ValueError("each group_id must occupy one contiguous run")
            seen.add(current)
            runs.append((start, index))
            if not at_end:
                start = index
                current = next_id
    return runs


def iter_grouped_logical_slices(
    group_ids: Tensor | Sequence[int],
    groups_per_microbatch: int,
) -> Iterator[tuple[int, int]]:
    """Yield row slices containing whole prompt groups only.

    GroupedTargetCollator emits candidates contiguously, so this helper prevents
    a correct/incorrect pair from being split across target microbatches.
    """

    if isinstance(groups_per_microbatch, bool) or not isinstance(groups_per_microbatch, int):
        raise ValueError("groups_per_microbatch must be a positive integer")
    if groups_per_microbatch <= 0:
        raise ValueError("groups_per_microbatch must be a positive integer")
    runs = _contiguous_group_runs(group_ids)
    for start_group in range(0, len(runs), groups_per_microbatch):
        end_group = min(start_group + groups_per_microbatch, len(runs))
        yield runs[start_group][0], runs[end_group - 1][1]


def slice_normalization_weight(
    normalization_weights: Tensor,
    start: int,
    end: int,
) -> Tensor:
    """Sum precomputed logical normalization weights for ``[start, end)``."""

    if normalization_weights.ndim != 1:
        raise ValueError("normalization_weights must be one-dimensional")
    if start < 0 or end < start or end > normalization_weights.numel():
        raise IndexError(
            f"invalid logical slice [{start}, {end}) for {normalization_weights.numel()} rows"
        )
    return normalization_weights[start:end].sum()


def logical_slice_normalization_weight(
    labels: Tensor,
    start: int,
    end: int,
    response_weights: Tensor | Sequence[float] | None = None,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> Tensor:
    """Convenience composition of token weights and a logical row slice."""

    weights = token_mean_normalization_weights(
        labels,
        response_weights=response_weights,
        ignore_index=ignore_index,
    )
    return slice_normalization_weight(weights, start, end)


def grouped_slice_normalization_weight(
    group_ids: Tensor | Sequence[int],
    start: int,
    end: int,
) -> float:
    """Return a whole-group slice's fraction of the margin-loss denominator."""

    runs = _contiguous_group_runs(group_ids)
    if not runs:
        raise ValueError("group normalization needs at least one group")
    boundaries = {runs[0][0], *(run_end for _, run_end in runs)}
    if start not in boundaries or end not in boundaries or end < start:
        raise ValueError("a grouped logical slice must start and end on group boundaries")
    groups_in_slice = sum(run_start >= start and run_end <= end for run_start, run_end in runs)
    if groups_in_slice == 0:
        raise ValueError("a grouped logical slice must contain at least one complete group")
    return groups_in_slice / len(runs)


def slice_target_batch(batch: Mapping[str, Any], start: int, end: int) -> dict[str, Any]:
    """Slice every candidate-level field while preserving scalar/group context.

    Candidate-level fields are recognized by having the same leading length as
    ``input_ids``.  This covers tensors and ordinary lists/tuples, and leaves
    configuration values or unrelated mappings untouched.
    """

    if "input_ids" not in batch:
        raise KeyError("target batch is missing 'input_ids'")
    input_ids = batch["input_ids"]
    if not isinstance(input_ids, Tensor) or input_ids.ndim < 1:
        raise TypeError("target batch input_ids must be a tensor with a batch dimension")
    num_rows = input_ids.shape[0]
    if start < 0 or end < start or end > num_rows:
        raise IndexError(f"invalid logical slice [{start}, {end}) for {num_rows} rows")

    sliced: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor) and value.ndim >= 1 and value.shape[0] == num_rows:
            sliced[key] = value[start:end]
        elif isinstance(value, list) and len(value) == num_rows:
            sliced[key] = value[start:end]
        elif isinstance(value, tuple) and len(value) == num_rows:
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


__all__ = [
    "ANSWER_ONLY_CE",
    "CORRECT_INCORRECT_MARGIN",
    "GROUP_IDS_KEY",
    "GroupedTargetCollator",
    "IGNORE_INDEX",
    "NLL",
    "REWARDS_KEY",
    "REWARD_WEIGHTED_SFT",
    "ROLES_KEY",
    "ROLE_CHOSEN",
    "ROLE_CORRECT",
    "ROLE_INCORRECT",
    "ROLE_NEUTRAL",
    "ROLE_REJECTED",
    "ResponseLogpStats",
    "TARGET_SIGNAL_MODES",
    "canonicalize_target_signal_mode",
    "causal_token_mean_nll",
    "compute_target_signal_loss",
    "compute_target_signal_loss_from_batch",
    "correct_incorrect_margin_loss",
    "encode_candidate_role",
    "grouped_slice_normalization_weight",
    "iter_grouped_logical_slices",
    "iter_logical_slices",
    "logical_slice_normalization_weight",
    "model_inputs_from_target_batch",
    "response_logp_stats",
    "reward_weighted_sft_loss",
    "shifted_logits_and_labels",
    "shifted_token_counts",
    "shifted_token_logps",
    "shifted_token_mask",
    "slice_normalization_weight",
    "slice_target_batch",
    "token_mean_normalization_weights",
    "weighted_token_nll_sum",
    "weighted_token_normalizer",
]
