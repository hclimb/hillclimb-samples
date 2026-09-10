"""MATH-500 greedy pass@1 evaluation with pinned Math-Verify scoring."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from SFT.data.dolci32k.profile import MATH_VERIFY_VERSION
from SFT.eval.tasks.bench_data import first_user_content, load_bench_records
from SFT.eval.tasks.common import (
    clean_model_response,
    render_generation_chat,
    require_distribution_version,
    result_provenance,
    single_source_revision,
)
from ..utils import generate_completions, get_eos_token_ids


DEFAULT_MAX_NEW_TOKENS = 4096
DATASET_REPOSITORY = "HuggingFaceH4/MATH-500"

_PROMPT_TEMPLATE = """Solve the following mathematics problem. Show your reasoning, then put only the final answer inside \\boxed{{}}.

{problem}"""


def _load_math_verify() -> Tuple[Callable[..., Any], Callable[..., Any], str]:
    version = require_distribution_version("math-verify", MATH_VERIFY_VERSION)
    try:
        from math_verify import parse, verify
    except ImportError as exc:  # pragma: no cover - version check normally catches this
        raise RuntimeError("math-verify is installed but cannot be imported") from exc
    return parse, verify, version


def score_math_answer(
    gold_answer: str,
    prediction: str,
    *,
    parse_fn: Callable[..., Any],
    verify_fn: Callable[..., Any],
) -> Dict[str, Any]:
    """Parse and compare one answer, returning auditable failure categories."""
    cleaned_prediction = clean_model_response(prediction)
    response_fields = {
        "raw_generation": prediction,
        "scored_response": cleaned_prediction,
        # Backward-compatible alias for existing generation inspectors.
        "response": cleaned_prediction,
    }
    try:
        # MATH-500's answer field is bare LaTeX; a math delimiter gives the
        # parser the same explicit environment used for model boxed answers.
        parsed_gold = parse_fn(f"${gold_answer}$")
    except Exception as exc:  # noqa: BLE001 - scorer failures are recorded, not hidden
        return {
            "correct": False,
            "status": "gold_parse_error",
            "error": f"{type(exc).__name__}: {exc}",
            **response_fields,
        }
    if not parsed_gold:
        return {
            "correct": False,
            "status": "gold_parse_error",
            "error": "empty parsed gold",
            **response_fields,
        }
    try:
        parsed_prediction = parse_fn(cleaned_prediction)
    except Exception as exc:  # noqa: BLE001
        return {
            "correct": False,
            "status": "prediction_parse_error",
            "error": f"{type(exc).__name__}: {exc}",
            **response_fields,
        }
    if not parsed_prediction:
        return {
            "correct": False,
            "status": "prediction_parse_error",
            "error": "empty parsed prediction",
            **response_fields,
        }
    try:
        correct = bool(verify_fn(parsed_gold, parsed_prediction))
    except Exception as exc:  # noqa: BLE001
        return {
            "correct": False,
            "status": "verification_error",
            "error": f"{type(exc).__name__}: {exc}",
            **response_fields,
        }
    return {
        "correct": correct,
        "status": "correct" if correct else "incorrect",
        "error": "",
        **response_fields,
    }


def compute_accuracy(
    args,
    model,
    tokenizer,
    batch_size: int = 4,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    requested_n = getattr(args, "n_test", -1)
    parse_fn, verify_fn, evaluator_version = _load_math_verify()
    records = load_bench_records(
        getattr(args, "data_dir", "./data"), "math500", k=requested_n
    )
    prompts = []
    for record in records:
        metadata = record.get("metadata", {})
        problem = metadata.get("problem") or first_user_content(record)
        prompts.append(
            render_generation_chat(
                tokenizer,
                _PROMPT_TEMPLATE.format(problem=problem),
                enable_thinking=True,
            )
        )
    generations = generate_completions(
        model,
        tokenizer,
        prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=get_eos_token_ids(tokenizer),
        do_sample=False,
        disable_tqdm=False,
    )

    per_example: List[Dict[str, Any]] = []
    for record, generation in zip(records, generations):
        metadata = record.get("metadata", {})
        gold_answer = metadata.get("answer")
        if gold_answer is None:
            raise ValueError(f"MATH-500 record has no metadata.answer: {record.get('id')}")
        scored = score_math_answer(
            str(gold_answer), generation, parse_fn=parse_fn, verify_fn=verify_fn
        )
        per_example.append(
            {
                "id": record.get("id"),
                "unique_id": metadata.get("unique_id"),
                "gold_answer": gold_answer,
                **scored,
            }
        )

    n_correct = sum(1 for row in per_example if row["correct"])
    n_parse_errors = sum(1 for row in per_example if "parse_error" in row["status"])
    n_verification_errors = sum(
        1 for row in per_example if row["status"] == "verification_error"
    )
    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        with open(
            os.path.join(output_dir, "math500_generations.jsonl"), "w", encoding="utf-8"
        ) as handle:
            for row in per_example:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "evaluation_scope": "full" if requested_n <= 0 else "limited",
        "accuracy": n_correct / len(records) * 100.0,
        "n_test": len(records),
        "n_correct": n_correct,
        "n_parse_errors": n_parse_errors,
        "n_verification_errors": n_verification_errors,
        "provenance": result_provenance(
            dataset_repository=DATASET_REPOSITORY,
            dataset_revision=single_source_revision(records),
            dataset_split="test",
            evaluator={
                "package": "math-verify",
                "version": evaluator_version,
                "primary_metric": "accuracy",
            },
            n_tasks=len(records),
            max_new_tokens=max_new_tokens,
            thinking=True,
        ),
    }
