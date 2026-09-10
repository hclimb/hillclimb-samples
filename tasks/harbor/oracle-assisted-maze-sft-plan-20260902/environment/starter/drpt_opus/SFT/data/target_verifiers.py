"""Verifiable correctness checks for generated target trajectories.

Each target domain gets a verifier with the same two-stage shape::

    verifier = build_verifier("math")
    context = verifier.prepare(row)      # None when the row cannot be verified
    result = verifier.verify(context, generated_text)

``prepare`` extracts ground truth from the target row once; ``verify`` scores
one candidate against it.  Splitting them keeps the expensive setup (parsing a
gold answer, recovering instruction constraints) out of the per-sample loop and
makes "this prompt cannot be verified at all" a first-class, recorded outcome
rather than a silent pass.

Domains
-------
``math``
    The gold ``\\boxed`` answer scored with the pinned Math-Verify used by the
    MATH500 evaluator, so a trajectory counted correct here is correct by the
    same rule the benchmark applies.
``mbpp``
    The row's own ``test_list`` executed in a subprocess with a wall clock and
    address-space cap.  This runs model-written code: it is sandboxed only by
    those limits, so run builds on the same trusted machines used for the rest
    of the pipeline.
``precise_if``
    Dolci Precise IF prompts embed IFEval's own instruction descriptions
    verbatim, so the constraints are recovered from the prompt text and then
    checked with the vendored official verifier.  Recovery is deliberately
    lenient: a family this module does not recognize is simply not checked.
    That can miss a violation, but it cannot invent one, and
    :meth:`PreciseIfVerifier.prepare` additionally drops any prompt whose own
    gold reference fails the recovered constraints -- the case where recovery
    itself went wrong.  Missed violations only cost candidate pairs; false
    violations would corrupt them.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from SFT.eval.tasks.common import clean_model_response
from SFT.eval.tasks.mbpp_plus import extract_code

DEFAULT_EXECUTION_TIMEOUT_SEC = 10.0
DEFAULT_EXECUTION_MEMORY_MB = 4096


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of scoring one candidate trajectory."""

    correct: bool
    status: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"correct": bool(self.correct), "status": self.status, **dict(self.detail)}


class TargetVerifier:
    """Interface shared by every domain verifier."""

    name = "base"

    def prepare(self, row: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def verify(self, context: Any, text: str) -> VerificationResult:
        raise NotImplementedError


def _assistant_reference(row: Mapping[str, Any]) -> str:
    messages = row.get("messages") or []
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    raise ValueError(f"target row {row.get('id')!r} has no assistant message")


def _user_prompt(row: Mapping[str, Any]) -> str:
    messages = row.get("messages") or []
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content", ""))
    raise ValueError(f"target row {row.get('id')!r} has no user message")


# --------------------------------------------------------------------------
# math
# --------------------------------------------------------------------------


def extract_boxed_answer(text: str) -> str | None:
    """Return the contents of the last ``\\boxed{...}``, brace-balanced."""

    marker = "\\boxed"
    index = text.rfind(marker)
    if index == -1:
        return None
    cursor = index + len(marker)
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text) or text[cursor] != "{":
        return None
    depth = 0
    start = cursor + 1
    while cursor < len(text):
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
            if depth == 0:
                return text[start:cursor]
        cursor += 1
    return None


class MathVerifier(TargetVerifier):
    """Score a boxed final answer with the pinned MATH500 scorer."""

    name = "math_boxed"

    def __init__(self) -> None:
        from SFT.eval.tasks.math500 import _load_math_verify

        self._parse, self._verify, self.math_verify_version = _load_math_verify()

    def prepare(self, row: Mapping[str, Any]) -> dict[str, Any] | None:
        gold = extract_boxed_answer(_assistant_reference(row))
        if gold is None or not gold.strip():
            return None
        return {"gold_answer": gold}

    def verify(self, context: Mapping[str, Any], text: str) -> VerificationResult:
        from SFT.eval.tasks.math500 import score_math_answer

        scored = score_math_answer(
            context["gold_answer"],
            text,
            parse_fn=self._parse,
            verify_fn=self._verify,
        )
        return VerificationResult(
            correct=bool(scored["correct"]),
            status=str(scored["status"]),
            detail={"gold_answer": context["gold_answer"]},
        )


# --------------------------------------------------------------------------
# mbpp
# --------------------------------------------------------------------------


_HARNESS = """\
import resource, sys
resource.setrlimit(resource.RLIMIT_AS, ({memory_bytes}, {memory_bytes}))
resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
"""


class MbppVerifier(TargetVerifier):
    """Execute the row's own MBPP asserts against the candidate program."""

    name = "mbpp_tests"

    def __init__(
        self,
        *,
        timeout_sec: float = DEFAULT_EXECUTION_TIMEOUT_SEC,
        memory_mb: int = DEFAULT_EXECUTION_MEMORY_MB,
        python_executable: str | None = None,
    ) -> None:
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        if memory_mb <= 0:
            raise ValueError("memory_mb must be positive")
        self.timeout_sec = float(timeout_sec)
        self.memory_mb = int(memory_mb)
        self.python_executable = python_executable or sys.executable

    def prepare(self, row: Mapping[str, Any]) -> dict[str, Any] | None:
        metadata = row.get("metadata") or {}
        tests = metadata.get("test_list") or []
        if not isinstance(tests, Sequence) or isinstance(tests, (str, bytes)) or not tests:
            return None
        return {
            "test_list": [str(test) for test in tests],
            "test_setup_code": str(metadata.get("test_setup_code") or ""),
        }

    def _program(self, context: Mapping[str, Any], code: str) -> str:
        parts = [
            _HARNESS.format(memory_bytes=self.memory_mb * 1024 * 1024),
            code,
            "",
            context["test_setup_code"],
            "",
            *context["test_list"],
            "",
            "print('__DRPT_TESTS_PASSED__')",
        ]
        return "\n".join(parts)

    def verify(self, context: Mapping[str, Any], text: str) -> VerificationResult:
        code = extract_code(clean_model_response(text))
        if not code.strip():
            return VerificationResult(False, "empty_program")
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return VerificationResult(False, "syntax_error", {"error": str(exc)})

        with tempfile.TemporaryDirectory() as workdir:
            script = os.path.join(workdir, "candidate_check.py")
            with open(script, "w", encoding="utf-8") as handle:
                handle.write(self._program(context, code))
            try:
                completed = subprocess.run(
                    [self.python_executable, "-I", "-S", script],
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                    env={"PATH": "/usr/bin:/bin", "HOME": workdir},
                )
            except subprocess.TimeoutExpired:
                return VerificationResult(False, "timeout")

        if "__DRPT_TESTS_PASSED__" in completed.stdout:
            return VerificationResult(True, "correct")
        return VerificationResult(
            False,
            "tests_failed" if completed.returncode else "no_success_marker",
            {"stderr": completed.stderr[-500:]},
        )


# --------------------------------------------------------------------------
# precise_if
# --------------------------------------------------------------------------

_KEYWORD = "keywords:"
_LANGUAGE = "language:"
_LENGTH = "length_constraints:"
_CONTENT = "detectable_content:"
_FORMAT = "detectable_format:"
_COMBINATION = "combination:"
_STARTEND = "startend:"
_CHANGE_CASES = "change_case:"
_PUNCTUATION = "punctuation:"

_RELATIONS = {"at least": "at least", "less than": "less than", "at most": "less than"}

# Every pattern below is the literal description IFEval's own
# ``build_description`` emits, so a match recovers the exact constraint the
# prompt was generated from rather than a paraphrase of it.
_LITERAL_CONSTRAINTS: tuple[tuple[str, str], ...] = (
    ("Entire output should be wrapped in JSON format", _FORMAT + "json_format"),
    (
        "Your answer must contain a title, wrapped in double angular brackets",
        _FORMAT + "title",
    ),
    ("Wrap your entire response with double quotation marks", _STARTEND + "quotation"),
    ("refrain from the use of any commas", _PUNCTUATION + "no_comma"),
    (
        "Your entire response should be in English, and in all capital letters",
        _CHANGE_CASES + "english_capital",
    ),
    (
        "Your entire response should be in English, and in all lowercase letters",
        _CHANGE_CASES + "english_lowercase",
    ),
    (
        "separated by 6 asterisk symbols",
        _COMBINATION + "two_responses",
    ),
)

_LIST_RE = r"\[(?P<items>[^\]]*)\]"
_NUM_RE = r"(?P<num>\d+)"
_REL_RE = r"(?P<relation>at least|less than|at most)"


def _parse_string_list(raw: str) -> list[str]:
    """Parse the ``['a', 'b']`` rendering IFEval descriptions embed."""

    try:
        parsed = ast.literal_eval("[" + raw + "]")
    except (SyntaxError, ValueError):
        return [item.strip().strip("'\"") for item in raw.split(",") if item.strip()]
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed]
    return [str(parsed)]


def recover_if_constraints(prompt: str) -> list[tuple[str, dict[str, Any]]]:
    """Recover ``(instruction_id, kwargs)`` pairs from an IFEval-style prompt.

    Only families whose description template is matched verbatim are returned.
    Unrecognized constraint text is skipped, which makes the resulting check
    lenient rather than wrong -- see this module's docstring.
    """

    recovered: list[tuple[str, dict[str, Any]]] = []

    def add(instruction_id: str, **kwargs: Any) -> None:
        if not any(existing == instruction_id for existing, _ in recovered):
            recovered.append((instruction_id, kwargs))

    for literal, instruction_id in _LITERAL_CONSTRAINTS:
        if literal in prompt:
            add(instruction_id)

    match = re.search(rf"Include keywords {_LIST_RE} in the response", prompt)
    if match:
        add(_KEYWORD + "existence", keywords=_parse_string_list(match.group("items")))

    match = re.search(rf"Do not include keywords {_LIST_RE} in the response", prompt)
    if match:
        add(
            _KEYWORD + "forbidden_words",
            forbidden_words=_parse_string_list(match.group("items")),
        )

    match = re.search(
        rf"the word (?P<keyword>\S+) should appear {_REL_RE} {_NUM_RE} times", prompt
    )
    if match:
        add(
            _KEYWORD + "frequency",
            keyword=match.group("keyword").strip("'\"."),
            relation=_RELATIONS[match.group("relation")],
            frequency=int(match.group("num")),
        )

    match = re.search(
        rf"letter (?P<letter>[a-zA-Z]) should appear {_REL_RE} {_NUM_RE} times", prompt
    )
    if match:
        add(
            _KEYWORD + "letter_frequency",
            letter=match.group("letter"),
            let_relation=_RELATIONS[match.group("relation")],
            let_frequency=int(match.group("num")),
        )

    match = re.search(
        rf"words with all capital letters should appear {_REL_RE} {_NUM_RE} times", prompt
    )
    if match:
        add(
            _CHANGE_CASES + "capital_word_frequency",
            capital_relation=_RELATIONS[match.group("relation")],
            capital_frequency=int(match.group("num")),
        )

    match = re.search(rf"Your response should contain {_REL_RE} {_NUM_RE} sentences", prompt)
    if match:
        add(
            _LENGTH + "number_sentences",
            relation=_RELATIONS[match.group("relation")],
            num_sentences=int(match.group("num")),
        )

    match = re.search(rf"Answer with {_REL_RE} {_NUM_RE} words", prompt)
    if match:
        add(
            _LENGTH + "number_words",
            relation=_RELATIONS[match.group("relation")],
            num_words=int(match.group("num")),
        )

    # The nth-paragraph family reuses the "There should be N paragraphs" opener
    # with a different separator, so it must win over the plain divider form.
    match = re.search(
        r"There should be (?P<num>\d+) paragraphs\. Paragraphs and only paragraphs are "
        r"separated with each other by two new lines[^.]*\. "
        r"Paragraph (?P<nth>\d+) must start with word (?P<word>[^.\s]+)",
        prompt,
    )
    if match:
        add(
            _LENGTH + "nth_paragraph_first_word",
            num_paragraphs=int(match.group("num")),
            nth_paragraph=int(match.group("nth")),
            first_word=match.group("word"),
        )
    else:
        match = re.search(
            r"There should be (?P<num>\d+) paragraphs\. Paragraphs are separated with "
            r"the markdown divider: \*\*\*",
            prompt,
        )
        if match:
            add(_LENGTH + "number_paragraphs", num_paragraphs=int(match.group("num")))

    match = re.search(
        rf"must contain at least {_NUM_RE} placeholders represented by square brackets",
        prompt,
    )
    if match:
        add(_CONTENT + "number_placeholders", num_placeholders=int(match.group("num")))

    match = re.search(
        r"add a postscript starting with (?P<marker>P\.?P\.?S|P\.?S\.?)", prompt
    )
    if match:
        add(_CONTENT + "postscript", postscript_marker=match.group("marker"))

    match = re.search(rf"must contain exactly {_NUM_RE} bullet points", prompt)
    if match:
        add(_FORMAT + "number_bullet_lists", num_bullets=int(match.group("num")))

    match = re.search(rf"Highlight at least {_NUM_RE} sections in your answer", prompt)
    if match:
        add(
            _FORMAT + "number_highlighted_sections", num_highlights=int(match.group("num"))
        )

    match = re.search(
        r"Your response must have (?P<num>\d+) sections\. Mark the beginning of each "
        r"section with (?P<spliter>\S+) X",
        prompt,
    )
    if match:
        add(
            _FORMAT + "multiple_sections",
            section_spliter=match.group("spliter"),
            num_sections=int(match.group("num")),
        )

    match = re.search(
        r"Finish your response with this exact phrase (?P<phrase>.+?)\. "
        r"No other words should follow this phrase",
        prompt,
        re.DOTALL,
    )
    if match:
        add(_STARTEND + "end_checker", end_phrase=match.group("phrase").strip())

    match = re.search(
        r"Your ENTIRE response should be in (?P<language>[A-Za-z]+) language", prompt
    )
    if match:
        code = _language_code(match.group("language"))
        if code:
            add(_LANGUAGE + "response_language", language=code)

    if "Answer with one of the following options:" in prompt:
        add(_FORMAT + "constrained_response")

    return recovered


def _language_code(language_name: str) -> str | None:
    from SFT.eval.tasks.ifeval_lib.instructions import _LANGUAGES

    wanted = language_name.strip().lower()
    for code, name in _LANGUAGES.items():
        if str(name).strip().lower() == wanted:
            return code
    return None


class PreciseIfVerifier(TargetVerifier):
    """Check recovered IFEval constraints with the vendored official verifier."""

    name = "if_constraints"

    def __init__(self, *, strict: bool = False) -> None:
        self.strict = bool(strict)

    def prepare(self, row: Mapping[str, Any]) -> dict[str, Any] | None:
        prompt = _user_prompt(row)
        recovered = recover_if_constraints(prompt)
        if not recovered:
            return None
        context = {
            "prompt": prompt,
            "instruction_id_list": [instruction_id for instruction_id, _ in recovered],
            "kwargs_list": [kwargs for _, kwargs in recovered],
        }
        # Recovery that the gold reference itself fails is recovery that went
        # wrong; drop the prompt rather than mint bogus negatives from it.
        gold = self.verify(context, _assistant_reference(row))
        if not gold.correct:
            return None
        return context

    def verify(self, context: Mapping[str, Any], text: str) -> VerificationResult:
        from SFT.eval.tasks.ifeval_scoring import evaluate_instruction_following

        scored = evaluate_instruction_following(
            prompt=context["prompt"],
            response=clean_model_response(text),
            instruction_id_list=context["instruction_id_list"],
            kwargs_list=context["kwargs_list"],
            strict=self.strict,
        )
        correct = bool(scored["follow_all_instructions"])
        return VerificationResult(
            correct=correct,
            status="correct" if correct else "constraint_violation",
            detail={
                "instruction_id_list": list(scored["instruction_id_list"]),
                "follow_instruction_list": list(scored["follow_instruction_list"]),
            },
        )


_VERIFIER_FACTORIES = {
    "math": MathVerifier,
    "mbpp": MbppVerifier,
    "precise_if": PreciseIfVerifier,
}


def build_verifier(target: str, **kwargs: Any) -> TargetVerifier:
    """Return the verifier for a dolci32k target name."""

    try:
        factory = _VERIFIER_FACTORIES[str(target)]
    except KeyError:
        choices = ", ".join(sorted(_VERIFIER_FACTORIES))
        raise KeyError(f"no verifier for target {target!r}; expected one of: {choices}") from None
    return factory(**kwargs)


__all__ = [
    "DEFAULT_EXECUTION_MEMORY_MB",
    "DEFAULT_EXECUTION_TIMEOUT_SEC",
    "MathVerifier",
    "MbppVerifier",
    "PreciseIfVerifier",
    "TargetVerifier",
    "VerificationResult",
    "build_verifier",
    "extract_boxed_answer",
    "recover_if_constraints",
]
