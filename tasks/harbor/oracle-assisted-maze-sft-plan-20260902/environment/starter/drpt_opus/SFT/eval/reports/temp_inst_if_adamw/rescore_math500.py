#!/usr/bin/env python3
"""Re-score MATH500 ignoring the unclosed-<think> formatting failure.

Why: SFT/eval/tasks/common.py::clean_model_response returns "" whenever a
completion *starts* with <think> and never closes it. Several curated
checkpoints emit "<think>\\n\\n<think>\\n\\n<full solution with \\boxed{...}>",
so the scorer sees an empty prediction and marks the item wrong even though the
boxed answer is present and often correct. mixed_math/LayerwiseSoftP hit this on
all 500 items and scored 0.00.

This rescoring keeps everything else identical -- same math_verify parse/verify
the harness uses -- but feeds it the last \\boxed{...} found anywhere in
raw_generation. The gap between the two numbers separates "cannot do the maths"
from "cannot keep the chat format".
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

from math_verify import parse as mv_parse, verify as mv_verify

CAMPAIGN = Path(
    "/home/nakyungl/drpt-next/Dr.Post-Training-Next/drpt_opus/SFT/runs/campaigns/"
    "dolci32k-qwen3_1_7b-s42"
)
SETTINGS = ("reason_math", "mixed_math")
METHODS = ("FullTraining", "LayerwiseRaw", "LayerwiseSoft", "LayerwiseSoftP", "LayerwiseOptA")


def last_boxed(text: str):
    """Return the content of the final \\boxed{...}, brace-balanced."""
    if not text:
        return None
    start = text.rfind(r"\boxed")
    if start == -1:
        return None
    i = text.find("{", start)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def rescore(path: Path):
    """Repair ONLY the items the official scorer could not parse.

    A naive last-\\boxed{} extraction is weaker than the harness's own
    clean_model_response + math_verify on items that already parsed (it drops
    answers stated outside \\boxed{}), so replacing every item with it would
    understate accuracy. Items whose official status is not a parse error keep
    their official verdict; parse-error items get the boxed-answer fallback.
    """
    official = repaired = total = recovered = parse_err = 0
    for line in path.open():
        rec = json.loads(line)
        total += 1
        was_correct = bool(rec.get("correct"))
        official += was_correct
        if rec.get("status") != "prediction_parse_error":
            repaired += was_correct
            continue
        parse_err += 1
        boxed = last_boxed(rec.get("raw_generation") or "")
        ok = False
        if boxed is not None:
            try:
                gold = mv_parse(f"${rec.get('gold_answer','')}$")
                pred = mv_parse(f"$\\boxed{{{boxed}}}$")
                ok = bool(gold) and bool(pred) and bool(mv_verify(gold, pred))
            except Exception:
                ok = False
        repaired += ok
        recovered += ok
    return total, official, repaired, recovered, parse_err


def main() -> int:
    rows = []
    for setting in SETTINGS:
        for method in METHODS:
            hits = glob.glob(str(CAMPAIGN / f"{setting}-{method}-adamw-*" / "math500_generations.jsonl"))
            if not hits:
                continue
            rows.append((setting, method) + rescore(Path(hits[0])))

    print("| setting | method | official | repaired | delta | parse_err | recovered |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for setting, method, total, official, repaired, recovered, parse_err in rows:
        o = 100.0 * official / total
        r = 100.0 * repaired / total
        print(f"| {setting} | {method} | {o:.2f} | {r:.2f} | {r-o:+.2f} | {parse_err} | {recovered} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
