#!/usr/bin/env python3
"""llm_judge_cot_accuracy: does the model's own reasoning trace conclude the ground truth,
independent of whatever it separately reported as its final "Generated Answer"?

Two-stage, bias-resistant design (not a single judge call that reads the CoT AND knows the
ground truth at once):
  Stage 1 (extraction, BLIND to ground truth): given only the question and the raw CoT, extract
    the final answer the reasoning itself concludes with, as a short direct sentence — or
    NO_CONCLUSION_REACHED if none is discernible (including the case where the CoT was cut off
    before finishing). The extractor never sees the ground truth, so it cannot rationalize
    toward a match the way a single combined read-and-judge call could.
  Stage 2 (comparison): the extracted answer is compared to the ground truth using the SAME
    prompt/system text as the original in-eval `llm_judge_accuracy` (imported directly from
    evals/metrics/llm_judge.py, not duplicated) — i.e. structurally identical to the original
    metric, just fed a CoT-derived answer instead of the separately-reported `generated_answer`
    field.

For every sample, the CoT is extracted from the raw `generated` field directly (NOT the
precomputed `thinking` field — that extraction returns empty for any sample whose `<think>`
block never hit a closing `</think>` tag before max_new_tokens, which silently drops real
reasoning content).

Writes a sidecar JSON next to DEST: {metrics: {llm_judge_cot_accuracy, ..., n_flipped_correct},
samples: [...]} — arm JSONs are not modified.

Box-side:  INPUTS="a.json b.json ..." [JUDGE_MODEL=Qwen/Qwen3-8B] [TP=4] \
  uv run python scripts/misc/judge_cot_accuracy.py
Each input is a GCS path (gs://...) or local path; sidecars upload beside GCS inputs.
"""
import asyncio
import json
import os
import re
import subprocess
import sys

import dotenv

dotenv.load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from evals.vllm import VLLMInference
from evals.metrics.llm_judge import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE, _parse_score

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-8B")
TP = int(os.environ.get("TP", 4))
BASE_URL = "http://localhost:8000/v1"

NO_CONCLUSION = "NO_CONCLUSION_REACHED"

EXTRACT_SYSTEM = f"""You extract the final answer a model's reasoning trace arrives at. You are \
NOT evaluating correctness and you have NOT been given any ground truth to compare against — \
just report, as a short, direct, self-contained sentence, what conclusion the reasoning itself \
reaches for the given question.

Rules:
- Report only the conclusion the reasoning actually settles on — not intermediate guesses it \
raised and then rejected or moved past along the way.
- Output a short, direct sentence that answers the question on its own — not a summary of the \
reasoning process, and not phrases like "the reasoning concludes that...".
- The reasoning trace may be cut off before finishing (mid-sentence, mid-thought, hit a token \
limit). If a conclusion is clearly implied by what's there even though it's cut off, extract it. \
If genuinely no conclusion is discernible from what's present, output exactly: {NO_CONCLUSION}
- Output ONLY the answer sentence, or exactly {NO_CONCLUSION}. No commentary, no caveats, no \
mention that you are extracting an answer.
"""

EXTRACT_TEMPLATE = """Question (for context only):
{question}

Model's reasoning trace{truncated_note}:
{cot}

What is the final answer this reasoning trace concludes with? Output ONLY the answer sentence, \
or exactly {no_conclusion} if none is discernible."""

TRUNCATED_NOTE = (" (NOTE: this trace was cut off by a token limit before it finished — it "
                   "never reached a natural stopping point)")


def extract_cot(generated_text):
    """Pull the chain-of-thought straight from the raw completion, robust to a `<think>` block
    that never closed (max_new_tokens hit mid-thought) — the precomputed `thinking` field comes
    back empty in exactly that case, silently discarding real reasoning content."""
    text = generated_text or ""
    if "<think>" not in text:
        return "", False
    after_open = text.split("<think>", 1)[1]
    if "</think>" in after_open:
        return after_open.split("</think>", 1)[0].strip(), False
    return after_open.strip(), True  # truncated: no closing tag


def question_of(prompt):
    m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, re.S)
    return (m.group(1) if m else prompt).strip()


def fetch(path):
    if path.startswith("gs://"):
        local = "/tmp/" + os.path.basename(path)
        subprocess.run(["gsutil", "cp", path, local], check=True, capture_output=True)
        return local
    return path


async def judge_file(client, path):
    local = fetch(path)
    data = json.load(open(local))
    rows = []
    for s in data["samples"]:
        cot, truncated = extract_cot(s.get("generated"))
        rows.append({
            "prompt_head": question_of(s["prompt"])[:120],
            "ground_truth": (s.get("ground_truth") or "").strip(),
            "cot": cot,
            "cot_truncated": truncated,
            "original_judge_accuracy": s.get("llm_judge_accuracy"),
            "generated_answer": s.get("generated_answer"),
        })
    sem = asyncio.Semaphore(32)

    async def one(r):
        if not r["cot"]:
            return {**r, "extracted_answer": NO_CONCLUSION, "extraction_output": "[no CoT extracted]",
                    "cot_judge_accuracy": 0.0, "comparison_output": "[no CoT extracted]"}

        # Stage 1 — blind extraction, no ground truth in scope.
        note = TRUNCATED_NOTE if r["cot_truncated"] else ""
        extract_prompt = EXTRACT_TEMPLATE.format(
            question=r["prompt_head"], truncated_note=note, cot=r["cot"], no_conclusion=NO_CONCLUSION)
        async with sem:
            extracted_raw, _ = await client.async_chat(
                prompt=extract_prompt, system=EXTRACT_SYSTEM, max_completion_tokens=256,
                temperature=0.0, thinking=False, top_p=1.0, top_k=-1)
        extracted = (extracted_raw or "").strip()

        # Stage 2 — same prompt/methodology as the original in-eval llm_judge_accuracy, fed the
        # CoT-derived answer instead of the separately-reported generated_answer field.
        answer_for_compare = "[no answer]" if extracted.upper().startswith(NO_CONCLUSION) else extracted
        compare_prompt = JUDGE_USER_TEMPLATE.format(
            question=r["prompt_head"], ground_truth=r["ground_truth"], answer=answer_for_compare)
        async with sem:
            content, reasoning = await client.async_chat(
                prompt=compare_prompt, system=JUDGE_SYSTEM_PROMPT, max_completion_tokens=1024,
                temperature=0.6, thinking=True, top_p=0.95, top_k=20)
        verdict = _parse_score(content)
        return {**r, "extracted_answer": extracted, "extraction_output": extracted_raw[-300:],
                "cot_judge_accuracy": verdict, "comparison_output": content[-400:]}

    out = await asyncio.gather(*[one(r) for r in rows])

    n = len(out)
    cot_acc = sum(r["cot_judge_accuracy"] for r in out) / n if n else None
    complete = [r for r in out if not r["cot_truncated"]]
    cot_acc_complete_only = (sum(r["cot_judge_accuracy"] for r in complete) / len(complete)
                             if complete else None)
    original_acc = (sum(r["original_judge_accuracy"] or 0.0 for r in out) / n) if n else None
    flipped_correct = [r for r in out
                       if (r["original_judge_accuracy"] or 0.0) == 0.0 and r["cot_judge_accuracy"] == 1.0]
    flipped_wrong = [r for r in out
                     if (r["original_judge_accuracy"] or 0.0) == 1.0 and r["cot_judge_accuracy"] == 0.0]
    n_truncated = sum(1 for r in out if r["cot_truncated"])
    n_no_conclusion = sum(1 for r in out if r["extracted_answer"].upper().startswith(NO_CONCLUSION))

    side = {
        "metrics": {
            "llm_judge_cot_accuracy": cot_acc,
            "llm_judge_cot_accuracy_complete_only": cot_acc_complete_only,
            "original_llm_judge_accuracy_same_sample_set": original_acc,
            "n": n,
            "n_cot_truncated": n_truncated,
            "n_no_conclusion_extracted": n_no_conclusion,
            "n_flipped_correct": len(flipped_correct),
            "n_flipped_wrong": len(flipped_wrong),
            "judge_model": JUDGE_MODEL,
            "source": os.path.basename(path),
        },
        "flipped_correct_samples": [
            {k: r[k] for k in ("prompt_head", "ground_truth", "generated_answer", "cot",
                               "extracted_answer", "comparison_output")}
            for r in flipped_correct
        ],
        "samples": out,
    }
    dest = local.replace(".json", "_cot_accuracy.json")
    json.dump(side, open(dest, "w"), indent=2)
    if path.startswith("gs://"):
        gdest = path.replace(".json", "_cot_accuracy.json")
        subprocess.run(["gsutil", "cp", dest, gdest], check=True, capture_output=True)
        out_path = gdest
    else:
        out_path = dest
    print(f"[cot-accuracy] {os.path.basename(path)}: "
          f"cot_acc={cot_acc:.4f} (complete-only={cot_acc_complete_only:.4f}) "
          f"vs original_acc={original_acc:.4f}  n={n}  truncated={n_truncated}  "
          f"no_conclusion={n_no_conclusion}  "
          f"flipped_correct={len(flipped_correct)}  flipped_wrong={len(flipped_wrong)} -> {out_path}")


def main():
    inputs = [p for p in re.split(r"[,\s]+", os.environ.get("INPUTS", "")) if p]
    if not inputs:
        sys.exit("set INPUTS to a space-separated list of results JSONs (gs:// or local)")
    proc = None
    try:
        if not VLLMInference.is_server_ready(BASE_URL):
            proc = VLLMInference.start_server(model=JUDGE_MODEL, base_url=BASE_URL,
                                              max_model_len=8192, tensor_parallel_size=TP)
            VLLMInference.wait_for_server(BASE_URL)
        client = VLLMInference(model=JUDGE_MODEL, base_url=BASE_URL)
        for p in inputs:
            asyncio.run(judge_file(client, p))
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait()


if __name__ == "__main__":
    main()
