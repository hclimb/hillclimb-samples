"""Artifact schema and answer-span rules for alternative target-gradient signals.

The immutable dolci32k build under ``dolci32k_artifacts/builds/<build_id>`` is
content addressed and must not gain new files, so pre-generated candidate
trajectories live in a sibling tree::

    dolci32k_artifacts/target_signals/<build_id>/<target>/candidates.jsonl
    dolci32k_artifacts/target_signals/<build_id>/<target>/manifest.json

One ``candidates.jsonl`` row corresponds to exactly one ``targets/<target>/
grad.jsonl`` row, keyed by ``id``, and nests every trajectory for that prompt::

    {"id": ..., "prompt_hash": ..., "verifiable": true,
     "candidates": [{"content": ..., "role": "correct", "reward": 1.0,
                     "origin": "reference", "verification": {...}}, ...]}

``role``/``reward`` use the encoding in :mod:`SFT.train.target_signal`:
correct ``1``, incorrect ``-1``, neutral ``0``.

Answer spans
------------
``answer_only_ce`` supervises only the final-answer tokens.  What counts as the
final answer is domain specific and deliberately conservative:

``math``
    From the last ``\\boxed`` to the end of the assistant turn.  All 64 rows in
    every shipped build carry a ``\\boxed`` answer.
``mbpp``
    The last fenced code block, fences included.  The reference answers are a
    bare fenced block, so this keeps the program and drops surrounding prose.
``precise_if``
    The whole final assistant turn.  Instruction following has no separable
    final answer: the response *is* the answer.  On the single-turn dolci32k
    targets this makes ``answer_only_ce`` numerically identical to ``nll``,
    which :func:`answer_span_is_degenerate` reports so runs can say so instead
    of implying a distinction that does not exist.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

TARGET_SIGNAL_DIRNAME = "target_signals"
CANDIDATES_FILENAME = "candidates.jsonl"
CANDIDATES_MANIFEST_FILENAME = "manifest.json"

ORIGIN_REFERENCE = "reference"
ORIGIN_GENERATED = "generated"

ROLE_CORRECT = "correct"
ROLE_INCORRECT = "incorrect"
ROLE_NEUTRAL = "neutral"

# A negative that never produced a well-formed answer ("Your input is correct!",
# code that does not parse) teaches the margin loss to reject gibberish, not to
# reject a wrong solution.  These statuses mark those degenerate failures so
# genuinely-wrong-but-well-formed trajectories are preferred as negatives.
MALFORMED_NEGATIVE_STATUSES = frozenset(
    {
        "prediction_parse_error",
        "gold_parse_error",
        "verification_error",
        "empty_program",
        "syntax_error",
    }
)

# High sampling temperatures, which is what it takes to make a strong generator
# fail on an easy prompt, also produce collapsed output -- a few mojibake
# characters that "violate" the check without being an attempt at all. A
# negative shorter than this is noise, and contrasting the reference against
# noise teaches the margin loss to reject noise rather than wrong reasoning.
MIN_INFORMATIVE_NEGATIVE_CHARS = 40

# Targets whose "final answer" is the entire assistant turn.
_WHOLE_TURN_ANSWER_TARGETS = frozenset({"precise_if"})

_BOXED_RE = re.compile(r"\\boxed\b")
_FENCED_BLOCK_RE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)


def target_signal_root(data_dir: str | Path, build_id: str) -> Path:
    """Return the sibling tree that holds pre-generated target candidates."""

    return Path(data_dir) / "dolci32k_artifacts" / TARGET_SIGNAL_DIRNAME / str(build_id)


def candidates_path(data_dir: str | Path, build_id: str, target: str) -> Path:
    return target_signal_root(data_dir, build_id) / str(target) / CANDIDATES_FILENAME


def candidates_manifest_path(data_dir: str | Path, build_id: str, target: str) -> Path:
    return (
        target_signal_root(data_dir, build_id)
        / str(target)
        / CANDIDATES_MANIFEST_FILENAME
    )


def answer_span_is_degenerate(target: str) -> bool:
    """True when the final-answer span is the whole assistant turn."""

    return str(target) in _WHOLE_TURN_ANSWER_TARGETS


def final_answer_char_span(target: str, assistant_text: str) -> tuple[int, int]:
    """Return ``[start, end)`` character bounds of the final answer.

    The span always ends at the last non-whitespace character so trailing
    newlines never become the only supervised tokens.  A domain rule that finds
    no marker falls back to the whole turn rather than guessing.
    """

    if not isinstance(assistant_text, str):
        raise TypeError(f"assistant text must be a string, got {type(assistant_text)!r}")
    stripped_end = len(assistant_text.rstrip())
    if stripped_end == 0:
        raise ValueError("assistant text is empty; it cannot carry a final answer")

    target = str(target)
    start = 0
    if target == "math":
        matches = list(_BOXED_RE.finditer(assistant_text, 0, stripped_end))
        if matches:
            start = matches[-1].start()
    elif target == "mbpp":
        blocks = list(_FENCED_BLOCK_RE.finditer(assistant_text))
        if blocks:
            start = blocks[-1].start()
            stripped_end = blocks[-1].end()
    elif not answer_span_is_degenerate(target):
        raise KeyError(f"no final-answer rule is defined for target {target!r}")

    if start >= stripped_end:
        # A marker at the very end (or an empty fenced block) would leave an
        # empty span; supervising the whole turn is the safe fallback.
        return 0, stripped_end
    return start, stripped_end


@dataclass(frozen=True)
class Candidate:
    """One trajectory for one target prompt."""

    content: str
    role: str
    reward: float
    origin: str
    verification: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "role": self.role,
            "reward": float(self.reward),
            "origin": self.origin,
            "verification": dict(self.verification),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "Candidate":
        missing = {"content", "role", "reward", "origin"} - set(payload)
        if missing:
            raise KeyError(f"candidate row is missing {sorted(missing)}")
        role = str(payload["role"])
        if role not in (ROLE_CORRECT, ROLE_INCORRECT, ROLE_NEUTRAL):
            raise ValueError(f"unknown candidate role {role!r}")
        return cls(
            content=str(payload["content"]),
            role=role,
            reward=float(payload["reward"]),
            origin=str(payload["origin"]),
            verification=dict(payload.get("verification") or {}),
        )


@dataclass(frozen=True)
class CandidateGroup:
    """Every trajectory generated for one target prompt."""

    id: str
    prompt_hash: str
    verifiable: bool
    candidates: tuple[Candidate, ...]

    def with_roles(self, role: str) -> tuple[Candidate, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.role == role)

    def ranked_negatives(
        self, *, min_chars: int = MIN_INFORMATIVE_NEGATIVE_CHARS
    ) -> tuple[Candidate, ...]:
        """Usable incorrect trajectories, most informative first.

        Collapsed output below ``min_chars`` is dropped outright rather than
        ranked last, so a prompt whose only "negative" is noise reports as
        having none. Among what remains, well-formed wrong answers sort ahead
        of malformed ones and longer attempts ahead of shorter ones, so the
        margin loss contrasts the reference against a real solution attempt
        whenever one exists.
        """

        def key(candidate: Candidate) -> tuple[int, int]:
            status = str(candidate.verification.get("status", ""))
            return (int(status in MALFORMED_NEGATIVE_STATUSES), -len(candidate.content))

        usable = [
            candidate
            for candidate in self.with_roles(ROLE_INCORRECT)
            if len(candidate.content.strip()) >= min_chars
        ]
        return tuple(sorted(usable, key=key))

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt_hash": self.prompt_hash,
            "verifiable": bool(self.verifiable),
            "candidates": [candidate.to_json() for candidate in self.candidates],
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "CandidateGroup":
        if "id" not in payload:
            raise KeyError("candidate group row is missing 'id'")
        raw = payload.get("candidates")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise TypeError("candidate group 'candidates' must be a list")
        return cls(
            id=str(payload["id"]),
            prompt_hash=str(payload.get("prompt_hash", "")),
            verifiable=bool(payload.get("verifiable", True)),
            candidates=tuple(Candidate.from_json(item) for item in raw),
        )


def write_candidate_groups(path: str | Path, groups: Sequence[CandidateGroup]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for group in groups:
            handle.write(json.dumps(group.to_json(), ensure_ascii=False) + "\n")
    return path


def iter_candidate_groups(path: str | Path) -> Iterator[CandidateGroup]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield CandidateGroup.from_json(json.loads(line))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc


def load_candidate_groups(path: str | Path) -> dict[str, CandidateGroup]:
    """Load candidates keyed by target-row id, rejecting duplicate ids."""

    groups: dict[str, CandidateGroup] = {}
    for group in iter_candidate_groups(path):
        if group.id in groups:
            raise ValueError(f"{path}: duplicate candidate group id {group.id!r}")
        groups[group.id] = group
    if not groups:
        raise ValueError(f"{path}: contains no candidate groups")
    return groups


__all__ = [
    "CANDIDATES_FILENAME",
    "CANDIDATES_MANIFEST_FILENAME",
    "MALFORMED_NEGATIVE_STATUSES",
    "MIN_INFORMATIVE_NEGATIVE_CHARS",
    "Candidate",
    "CandidateGroup",
    "ORIGIN_GENERATED",
    "ORIGIN_REFERENCE",
    "ROLE_CORRECT",
    "ROLE_INCORRECT",
    "ROLE_NEUTRAL",
    "TARGET_SIGNAL_DIRNAME",
    "answer_span_is_degenerate",
    "candidates_manifest_path",
    "candidates_path",
    "final_answer_char_span",
    "iter_candidate_groups",
    "load_candidate_groups",
    "target_signal_root",
    "write_candidate_groups",
]
