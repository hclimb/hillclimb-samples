"""Exact normalized-prompt and word-8-gram decontamination."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .common import normalize_prompt, prompt_hash, user_prompt_views
from .profile import NEAR_DUPLICATE_THRESHOLD, WORD_NGRAM_SIZE

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def word_tokens(text: Any) -> Tuple[str, ...]:
    return tuple(_WORD_RE.findall(normalize_prompt(text)))


def word_ngrams(text: Any, n: int = WORD_NGRAM_SIZE) -> frozenset[Tuple[str, ...]]:
    tokens = word_tokens(text)
    if len(tokens) < n:
        return frozenset()
    return frozenset(tuple(tokens[index:index + n]) for index in range(len(tokens) - n + 1))


@dataclass(frozen=True)
class DecontamMatch:
    kind: str
    reference_id: str
    score: float


@dataclass(frozen=True)
class _Reference:
    reference_id: str
    exact_hash: str
    ngrams: frozenset[Tuple[str, ...]]


class PromptDecontaminator:
    def __init__(
        self,
        references: Iterable[Tuple[str, str]] = (),
        *,
        ngram_size: int = WORD_NGRAM_SIZE,
        threshold: float = NEAR_DUPLICATE_THRESHOLD,
    ) -> None:
        if ngram_size < 1 or not 0 < threshold <= 1:
            raise ValueError("invalid decontamination parameters")
        self.ngram_size = int(ngram_size)
        self.threshold = float(threshold)
        self._references: List[_Reference] = []
        self._exact: Dict[str, int] = {}
        self._inverted: DefaultDict[Tuple[str, ...], Set[int]] = defaultdict(set)
        for reference_id, prompt in references:
            self.add(str(reference_id), prompt)

    @property
    def reference_count(self) -> int:
        return len(self._references)

    def settings(self) -> Dict[str, Any]:
        return {
            "normalization": "Unicode NFKC + casefold + collapsed whitespace",
            "exact_hash": "sha256",
            "word_ngram_size": self.ngram_size,
            "near_duplicate_threshold": self.threshold,
            "near_duplicate_coverage": "shared_unique_ngrams/min(candidate,reference)",
            "short_prompt_policy": "exact-only",
            "reference_count": self.reference_count,
        }

    def add(self, reference_id: str, prompt: Any) -> None:
        normalized = normalize_prompt(prompt)
        if not normalized:
            raise ValueError(f"empty decontamination reference {reference_id!r}")
        digest = prompt_hash(normalized)
        if digest in self._exact:
            return
        grams = word_ngrams(normalized, self.ngram_size)
        index = len(self._references)
        self._references.append(_Reference(reference_id, digest, grams))
        self._exact[digest] = index
        for gram in grams:
            self._inverted[gram].add(index)

    def match(self, prompt: Any) -> Optional[DecontamMatch]:
        normalized = normalize_prompt(prompt)
        if not normalized:
            return DecontamMatch("empty", "", 1.0)
        digest = prompt_hash(normalized)
        exact = self._exact.get(digest)
        if exact is not None:
            return DecontamMatch("exact", self._references[exact].reference_id, 1.0)
        grams = word_ngrams(normalized, self.ngram_size)
        if not grams:
            return None
        intersections: Counter[int] = Counter()
        for gram in grams:
            intersections.update(self._inverted.get(gram, ()))
        for index, shared in sorted(intersections.items()):
            reference = self._references[index]
            denominator = min(len(grams), len(reference.ngrams))
            score = shared / denominator if denominator else 0.0
            if score >= self.threshold:
                return DecontamMatch("near_8gram", reference.reference_id, score)
        return None

    def match_record(self, record: Mapping[str, Any]) -> Optional[DecontamMatch]:
        for view in user_prompt_views(record):
            match = self.match(view)
            if match is not None:
                return match
        return None

    def blocked(self, prompt: Any) -> bool:
        return self.match(prompt) is not None

    def blocked_record(self, record: Mapping[str, Any]) -> bool:
        return self.match_record(record) is not None


def references_from_records(named_records: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[Tuple[str, str]]:
    references: List[Tuple[str, str]] = []
    for group in sorted(named_records):
        for record in sorted(named_records[group], key=lambda row: str(row.get("id", ""))):
            for view_index, view in enumerate(user_prompt_views(record)):
                references.append((f"{group}:{record.get('id')}:view{view_index}", view))
    return references


def audit_against(records: Iterable[Mapping[str, Any]], index: PromptDecontaminator, *, sample_limit: int = 20) -> Dict[str, Any]:
    counts: Counter[str] = Counter()
    samples: List[Dict[str, Any]] = []
    for record in records:
        match = index.match_record(record)
        if match is None:
            continue
        counts[match.kind] += 1
        if len(samples) < sample_limit:
            samples.append({
                "record_id": record.get("id"), "kind": match.kind,
                "reference_id": match.reference_id, "score": match.score,
            })
    return {"matches": sum(counts.values()), "by_kind": dict(counts), "samples": samples}
