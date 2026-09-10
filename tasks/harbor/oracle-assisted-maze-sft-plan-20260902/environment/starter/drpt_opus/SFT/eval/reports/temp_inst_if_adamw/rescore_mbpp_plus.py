#!/usr/bin/env python3
"""Re-score MBPP+ for generations lost to the unclosed-<think> format failure.

SFT/eval/tasks/mbpp_plus.py::assemble_solution runs the generation through
clean_model_response() first, and that helper returns "" whenever the text
starts with <think> and never closes it. The solution then degrades to the bare
problem prompt, which fails every test -- even though the model did emit a
complete function further down in raw_generation.

This script rebuilds the EvalPlus samples file taking extract_code() straight
off raw_generation for exactly those items (everything else is assembled the
same way the harness does it), then runs EvalPlus locally. CPU only: EvalPlus
executes candidate programs against unit tests, no GPU involved.

Usage:  python rescore_mbpp_plus.py <method> [<method> ...]
"""

from __future__ import annotations

import glob
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/nakyungl/drpt-next/Dr.Post-Training-Next/drpt_opus")

from evalplus.data import get_mbpp_plus

from SFT.eval.tasks.common import clean_model_response
from SFT.eval.tasks.mbpp_plus import _FUNCTION_RE, extract_code

CAMPAIGN = Path(
    "/home/nakyungl/drpt-next/Dr.Post-Training-Next/drpt_opus/SFT/runs/campaigns/"
    "dolci32k-qwen3_1_7b-s42"
)
PYTHON = "/home/nakyungl/miniconda3/envs/drpt-next/bin/python"


def assemble(problem_prompt: str, code: str) -> str:
    if _FUNCTION_RE.search(code):
        return code
    return problem_prompt.rstrip() + "\n" + code


def build(method: str, out_path: Path):
    hits = glob.glob(str(CAMPAIGN / f"reason_code-{method}-adamw-*" / "mbpp_plus_generations.jsonl"))
    if not hits:
        return None
    problems = get_mbpp_plus()
    repaired = 0
    rows = []
    for line in Path(hits[0]).open():
        rec = json.loads(line)
        task_id = rec.get("task_id") or rec.get("id")
        raw = rec.get("raw_generation") or ""
        official_code = extract_code(clean_model_response(raw))
        if official_code:
            code = official_code
        else:
            # The harness lost this one to the format guard; go straight to the
            # untouched generation so the model's actual program is scored.
            code = extract_code(raw)
            if code:
                repaired += 1
        prompt = problems.get(task_id, {}).get("prompt", "")
        rows.append({"task_id": task_id, "solution": assemble(prompt, code)})
    with out_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows), repaired


def main() -> int:
    methods = sys.argv[1:] or ["LayerwiseSoftP"]
    for method in methods:
        with tempfile.TemporaryDirectory() as tmp:
            samples = Path(tmp) / f"{method}_samples.jsonl"
            built = build(method, samples)
            if not built:
                print(f"{method}: generations not found")
                continue
            total, repaired = built
            print(f"\n=== {method}: {total} items, {repaired} repaired ===", flush=True)
            proc = subprocess.run(
                [PYTHON, "-m", "evalplus.evaluate", "--dataset", "mbpp",
                 "--samples", str(samples)],
                capture_output=True, text=True, timeout=3600,
            )
            tail = (proc.stdout + proc.stderr).strip().splitlines()
            for line in [l for l in tail if "pass@1" in l or "mbpp" in l.lower()][-8:]:
                print("   ", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
