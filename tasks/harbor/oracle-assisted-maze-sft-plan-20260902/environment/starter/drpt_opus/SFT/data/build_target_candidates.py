#!/usr/bin/env python3
"""Pre-generate and verify candidate trajectories for the 64 target rows.

The alternate target-gradient signals need more than the single reference
trajectory each ``targets/<target>/grad.jsonl`` row carries:

* ``correct_incorrect_margin`` needs a *wrong* trajectory to push down against
  the reference, and
* ``reward_weighted_sft`` needs several trajectories whose correctness reward
  decides how much each one contributes.

Both are produced once, offline, by sampling from a pinned Qwen3 generator and
scoring every sample with the domain's verifier -- Math-Verify for ``math``, the
row's own asserts for ``mbpp``, recovered IFEval constraints for ``precise_if``.
Nothing here runs during training, so the SFT loop stays exactly as cheap as it
is today: no rollouts, no reward model, no online RL.

Output goes to a sibling of the immutable build, keyed by build id, because the
build directory is content addressed and must not gain files::

    SFT/data/dolci32k_artifacts/target_signals/<build_id>/<target>/candidates.jsonl
    SFT/data/dolci32k_artifacts/target_signals/<build_id>/<target>/manifest.json

Example
-------
  python -m SFT.data.build_target_candidates \\
      --target math --generator_profile qwen3_4b --num_samples 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from SFT.data.dolci32k.profile import MODEL_PROFILES, SETTINGS, TARGETS
from SFT.data.target_candidates import (
    ORIGIN_GENERATED,
    ORIGIN_REFERENCE,
    ROLE_CORRECT,
    ROLE_INCORRECT,
    Candidate,
    CandidateGroup,
    candidates_manifest_path,
    candidates_path,
    load_candidate_groups,
    write_candidate_groups,
)
from SFT.data.target_verifiers import build_verifier
from SFT.eval.tasks.common import clean_model_response

logger = logging.getLogger(__name__)

DEFAULT_GENERATOR_PROFILE = "qwen3_4b"

# Only the generator sees these. The stored trajectory is the assistant text
# alone, replayed later against the target row's own unmodified user message,
# so a format nudge here cannot leak into the training data.
_SYSTEM_PROMPTS: Mapping[str, str] = {
    "math": (
        "Solve the mathematics problem. Show your reasoning, then put only the "
        "final answer inside \\boxed{}."
    ),
    "mbpp": (
        "You are an expert Python programmer. Return one complete, self-contained "
        "solution in a single ```python code block. Do not add prose outside the block."
    ),
    "precise_if": (
        "Follow every instruction in the request exactly, including all formatting "
        "and length constraints."
    ),
}

_MAX_NEW_TOKENS: Mapping[str, int] = {"math": 1024, "mbpp": 512, "precise_if": 1536}


def _target_grad_path(data_dir: Path, build_id: str, target: str) -> Path:
    return (
        data_dir
        / "dolci32k_artifacts"
        / "builds"
        / build_id
        / "targets"
        / target
        / "grad.jsonl"
    )


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"target split not found: {path}")
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if not rows:
        raise ValueError(f"target split is empty: {path}")
    return rows


def _user_content(row: Mapping[str, Any]) -> str:
    for message in row["messages"]:
        if message.get("role") == "user":
            return str(message["content"])
    raise ValueError(f"target row {row.get('id')!r} has no user message")


def _render_prompt(tokenizer, target: str, row: Mapping[str, Any]) -> str:
    messages = []
    system = _SYSTEM_PROMPTS.get(target)
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": _user_content(row)})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _stop_token_ids(tokenizer) -> list[int]:
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    for marker in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(marker)
        if isinstance(token_id, int) and token_id >= 0:
            ids.add(token_id)
    return sorted(ids)


def generate_samples(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    num_samples: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int,
    seed: int,
) -> list[list[str]]:
    """Return ``num_samples`` completions for each prompt, in prompt order."""

    results: list[list[str]] = []
    stop_ids = _stop_token_ids(tokenizer)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for start in range(0, len(prompts), batch_size):
        chunk = list(prompts[start : start + batch_size])
        encoded = tokenizer(
            chunk, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator)))
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_samples,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_ids or None,
            )
        prompt_length = encoded["input_ids"].shape[1]
        completions = tokenizer.batch_decode(
            outputs[:, prompt_length:], skip_special_tokens=True
        )
        for index in range(len(chunk)):
            window = completions[index * num_samples : (index + 1) * num_samples]
            results.append([text.strip() for text in window])
        logger.info(
            "generated %d/%d prompts", min(start + batch_size, len(prompts)), len(prompts)
        )
    return results


def build_groups(
    rows: Sequence[Mapping[str, Any]],
    samples: Sequence[Sequence[str]],
    verifier,
) -> tuple[list[CandidateGroup], dict[str, Any]]:
    """Verify every sample and assemble one candidate group per target row."""

    groups: list[CandidateGroup] = []
    counts = {
        "prompts": len(rows),
        "verifiable_prompts": 0,
        "generated": 0,
        "generated_correct": 0,
        "generated_incorrect": 0,
        "duplicate_dropped": 0,
        "empty_dropped": 0,
        "prompts_with_negative": 0,
        "prompts_with_positive": 0,
    }
    statuses: dict[str, int] = {}

    for row, texts in zip(rows, samples):
        context = verifier.prepare(row)
        reference = Candidate(
            content=_reference_content(row),
            role=ROLE_CORRECT,
            reward=1.0,
            origin=ORIGIN_REFERENCE,
            verification={"status": "reference"},
        )
        candidates = [reference]
        if context is None:
            groups.append(
                CandidateGroup(
                    id=str(row["id"]),
                    prompt_hash=str(row.get("prompt_hash", "")),
                    verifiable=False,
                    candidates=tuple(candidates),
                )
            )
            continue

        counts["verifiable_prompts"] += 1
        seen = {reference.content.strip()}
        has_negative = False
        has_positive = False
        for text in texts:
            cleaned = clean_model_response(text)
            if not cleaned.strip():
                counts["empty_dropped"] += 1
                continue
            if cleaned.strip() in seen:
                counts["duplicate_dropped"] += 1
                continue
            seen.add(cleaned.strip())
            result = verifier.verify(context, cleaned)
            statuses[result.status] = statuses.get(result.status, 0) + 1
            counts["generated"] += 1
            if result.correct:
                counts["generated_correct"] += 1
                has_positive = True
            else:
                counts["generated_incorrect"] += 1
                has_negative = True
            candidates.append(
                Candidate(
                    content=cleaned,
                    role=ROLE_CORRECT if result.correct else ROLE_INCORRECT,
                    reward=1.0 if result.correct else 0.0,
                    origin=ORIGIN_GENERATED,
                    verification=result.to_json(),
                )
            )
        counts["prompts_with_negative"] += int(has_negative)
        counts["prompts_with_positive"] += int(has_positive)
        groups.append(
            CandidateGroup(
                id=str(row["id"]),
                prompt_hash=str(row.get("prompt_hash", "")),
                verifiable=True,
                candidates=tuple(candidates),
            )
        )

    counts["status_histogram"] = statuses
    counts["margin_pair_coverage"] = (
        counts["prompts_with_negative"] / len(rows) if rows else 0.0
    )
    return groups, counts


def _reference_content(row: Mapping[str, Any]) -> str:
    for message in reversed(row["messages"]):
        if message.get("role") == "assistant":
            return str(message["content"])
    raise ValueError(f"target row {row.get('id')!r} has no assistant message")


def ids_missing_negatives(groups: Mapping[str, CandidateGroup]) -> list[str]:
    """Verifiable prompts that produced no wrong trajectory, in file order.

    These are the prompts the margin objective has to drop. Unverifiable
    prompts are excluded: no amount of extra sampling makes them scorable.
    """

    return [
        group_id
        for group_id, group in groups.items()
        if group.verifiable and not group.ranked_negatives()
    ]


def merge_top_up(
    existing: Mapping[str, CandidateGroup],
    fresh: Sequence[CandidateGroup],
) -> tuple[dict[str, CandidateGroup], dict[str, int]]:
    """Fold a second sampling pass into the groups already on disk.

    Only generated trajectories are carried over, and only ones whose text is
    not already present: the reference stays exactly the one the immutable
    build shipped, and re-sampling cannot silently duplicate a candidate.
    """

    merged = dict(existing)
    counts = {"topped_up_prompts": 0, "added": 0, "duplicate_dropped": 0, "new_negatives": 0}
    for group in fresh:
        current = merged.get(group.id)
        if current is None:
            raise KeyError(f"top-up produced an unknown prompt id {group.id!r}")
        seen = {candidate.content.strip() for candidate in current.candidates}
        additions = []
        for candidate in group.candidates:
            if candidate.origin != ORIGIN_GENERATED:
                continue
            if candidate.content.strip() in seen:
                counts["duplicate_dropped"] += 1
                continue
            seen.add(candidate.content.strip())
            additions.append(candidate)
            counts["added"] += 1
            if candidate.role == ROLE_INCORRECT:
                counts["new_negatives"] += 1
        if additions:
            counts["topped_up_prompts"] += 1
            merged[group.id] = CandidateGroup(
                id=current.id,
                prompt_hash=current.prompt_hash,
                verifiable=current.verifiable,
                candidates=current.candidates + tuple(additions),
            )
    return merged, counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=sorted(TARGETS),
        help="target pool to build (repeatable; default: every target)",
    )
    parser.add_argument(
        "--setting",
        action="append",
        choices=sorted(SETTINGS),
        help="dolci32k setting whose target should be built (repeatable)",
    )
    parser.add_argument("--data_dir", default=os.environ.get("DRPT_DATA_DIR", "SFT/data"))
    parser.add_argument(
        "--artifact_build_id", default=os.environ.get("DRPT_ARTIFACT_BUILD_ID")
    )
    parser.add_argument(
        "--generator_profile",
        default=DEFAULT_GENERATOR_PROFILE,
        choices=sorted(MODEL_PROFILES),
        help="pinned model profile used for sampling (default: %(default)s)",
    )
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8, help="prompts per generate call")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="smoke-test row cap")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing candidates.jsonl"
    )
    parser.add_argument(
        "--only_missing_negatives",
        action="store_true",
        help=(
            "re-sample only the verifiable prompts that produced no wrong "
            "trajectory and merge the results into the existing file. Use a "
            "higher --temperature or --num_samples than the original pass; "
            "otherwise the model will simply answer them correctly again."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if not args.artifact_build_id:
        raise SystemExit("--artifact_build_id (or DRPT_ARTIFACT_BUILD_ID) is required")
    if args.num_samples < 1:
        raise SystemExit("--num_samples must be at least 1")

    targets = list(args.target or [])
    for setting in args.setting or []:
        target = str(SETTINGS[setting]["target"])
        if target not in targets:
            targets.append(target)
    if not targets:
        targets = sorted(TARGETS)

    if args.only_missing_negatives and args.overwrite:
        raise SystemExit("--only_missing_negatives merges into the existing file; drop --overwrite")

    data_dir = Path(args.data_dir)
    pending = []
    for target in targets:
        destination = candidates_path(data_dir, args.artifact_build_id, target)
        if args.only_missing_negatives:
            if not destination.exists():
                raise SystemExit(f"--only_missing_negatives needs an existing {destination}")
            pending.append(target)
            continue
        if destination.exists() and not args.overwrite:
            logger.info("skipping %s: %s already exists (--overwrite to rebuild)", target, destination)
            continue
        pending.append(target)
    if not pending:
        logger.info("nothing to build")
        return 0

    from transformers import AutoModelForCausalLM, AutoTokenizer

    profile = MODEL_PROFILES[args.generator_profile]
    model_name = profile["model_name_or_path"]
    revision = profile.get("model_revision")
    logger.info("loading generator %s@%s", model_name, revision)
    tokenizer = AutoTokenizer.from_pretrained(
        profile.get("tokenizer_name", model_name),
        revision=profile.get("tokenizer_revision", revision),
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    for target in pending:
        started = time.time()
        destination_path = candidates_path(data_dir, args.artifact_build_id, target)
        rows = _load_rows(_target_grad_path(data_dir, args.artifact_build_id, target))
        if args.limit is not None:
            rows = rows[: args.limit]

        existing_groups = None
        if args.only_missing_negatives:
            existing_groups = load_candidate_groups(destination_path)
            wanted = set(ids_missing_negatives(existing_groups))
            rows = [row for row in rows if str(row["id"]) in wanted]
            if not rows:
                logger.info(
                    "[%s] every verifiable prompt already has a negative; nothing to top up",
                    target,
                )
                continue
            logger.info("[%s] topping up %d prompts with no wrong trajectory", target, len(rows))

        logger.info("[%s] sampling %d completions for %d rows", target, args.num_samples, len(rows))
        prompts = [_render_prompt(tokenizer, target, row) for row in rows]
        samples = generate_samples(
            model,
            tokenizer,
            prompts,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens or _MAX_NEW_TOKENS.get(target, 1024),
            batch_size=args.batch_size,
            seed=args.seed,
        )
        verifier = build_verifier(target)
        groups, counts = build_groups(rows, samples, verifier)

        pass_record = {
            "generator_profile": args.generator_profile,
            "generator_model": model_name,
            "generator_revision": revision,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens or _MAX_NEW_TOKENS.get(target, 1024),
            "seed": args.seed,
            "elapsed_sec": round(time.time() - started, 1),
        }
        manifest_path = candidates_manifest_path(data_dir, args.artifact_build_id, target)

        if existing_groups is not None:
            merged, merge_counts = merge_top_up(existing_groups, groups)
            groups = list(merged.values())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            with_negative = sum(1 for group in groups if group.ranked_negatives())
            manifest["counts"]["prompts_with_negative"] = with_negative
            manifest["counts"]["margin_pair_coverage"] = with_negative / len(groups)
            manifest.setdefault("top_ups", []).append({**pass_record, "counts": merge_counts})
            counts = manifest["counts"]
        else:
            manifest = {
                "target": target,
                "artifact_build_id": args.artifact_build_id,
                "verifier": verifier.name,
                **pass_record,
                "counts": counts,
            }

        destination = write_candidate_groups(destination_path, groups)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        logger.info(
            "[%s] wrote %s (%d groups); margin-pair coverage %.1f%% (%d/%d prompts)",
            target,
            destination,
            len(groups),
            100.0 * counts["margin_pair_coverage"],
            counts["prompts_with_negative"],
            counts["prompts"],
        )
        if counts["prompts_with_negative"] == 0:
            logger.warning(
                "[%s] no wrong trajectory was produced; correct_incorrect_margin has "
                "nothing to train on. Raise --temperature or --num_samples.",
                target,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
