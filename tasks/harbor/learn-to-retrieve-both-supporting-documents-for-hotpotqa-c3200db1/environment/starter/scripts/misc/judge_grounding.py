#!/usr/bin/env python3
"""llm_judge_grounding_accuracy: is the generated answer supported by >=1 retrieved doc?

For every sample in a results JSON that carries `generated_answer` + `rag_docs`, ask the
judge whether the answer is supported by the retrieved evidence. Context budget: arms with
k up to 200 cannot fit all docs in the judge window, so each call judges the TOP 5 rag_docs
by content-word overlap with the answer (support, where it exists, is where the overlap
is — the cap is recorded in the output). One call per sample, temperature 0.

Writes a sidecar JSON next to DEST: {metrics: {llm_judge_grounding_accuracy, n, docs_per_call},
samples: [{prompt_head, grounded, judge_output}]} — arm JSONs are not modified.

Box-side:  INPUTS="a.json b.json ..." [TP=4] uv run python scripts/misc/judge_grounding.py
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

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-4B")
TP = int(os.environ.get("TP", 4))
BASE_URL = "http://localhost:8000/v1"

SYSTEM = """You are a grounding judge. Decide whether the generated answer is SUPPORTED by at
least one of the retrieved documents provided. Supported means the core factual claim(s) of
the answer appear in, or follow directly from, some document — not from outside knowledge.
Reply with exactly one word on the last line: SUPPORTED or UNSUPPORTED."""

TEMPLATE = """Question (context only):
{question}

Generated answer:
{answer}

Retrieved documents (top {n} by overlap, of {total} retrieved):
{docs}

Is the generated answer supported by at least one of these documents? Reply SUPPORTED or UNSUPPORTED."""

WORD = re.compile(r"[A-Za-z0-9]+")


def overlap(answer_words, doc):
    dw = set(w.lower() for w in WORD.findall(doc))
    return len(answer_words & dw)


def question_of(prompt):
    m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, re.S)
    return (m.group(1) if m else prompt).strip()


def fetch(path):
    if path.startswith("gs://"):
        local = "/tmp/" + os.path.basename(path)
        subprocess.run(["gsutil", "cp", path, local], check=True, capture_output=True)
        return local
    return path


async def judge_file(client, path, docs_per_call=5):
    local = fetch(path)
    data = json.load(open(local))
    rows = [s for s in data["samples"] if (s.get("generated_answer") or "").strip() and s.get("rag_docs")]
    sem = asyncio.Semaphore(32)

    async def one(s):
        aw = set(w.lower() for w in WORD.findall(s["generated_answer"]))
        docs = sorted(s["rag_docs"], key=lambda d: -overlap(aw, d))[:docs_per_call]
        prompt = TEMPLATE.format(
            question=question_of(s["prompt"]), answer=s["generated_answer"].strip(),
            n=len(docs), total=len(s["rag_docs"]),
            docs="\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs)))
        async with sem:
            content, reasoning = await client.async_chat(
                prompt=prompt, system=SYSTEM, max_completion_tokens=512,
                temperature=0.0, thinking=True, top_p=1.0, top_k=-1)
        verdict = 1.0 if "SUPPORTED" in content.upper().split("UNSUPPORTED")[-1] or \
            (content.upper().rstrip().endswith("SUPPORTED") and not content.upper().rstrip().endswith("UNSUPPORTED")) else 0.0
        # robust parse: last standalone token wins
        toks = re.findall(r"\b(UN)?SUPPORTED\b", content.upper())
        if toks:
            verdict = 0.0 if toks[-1] == "UN" else 1.0
        return {"prompt_head": question_of(s["prompt"])[:120], "grounded": verdict,
                "judge_output": content[-300:]}

    out = await asyncio.gather(*[one(s) for s in rows])
    acc = sum(r["grounded"] for r in out) / len(out) if out else None
    side = {"metrics": {"llm_judge_grounding_accuracy": acc, "n": len(out),
                        "docs_per_call": docs_per_call, "source": os.path.basename(path)},
            "samples": out}
    dest = local.replace(".json", "_grounding.json")
    json.dump(side, open(dest, "w"), indent=2)
    if path.startswith("gs://"):
        gdest = path.replace(".json", "_grounding.json")
        subprocess.run(["gsutil", "cp", dest, gdest], check=True, capture_output=True)
        print(f"[grounding] {os.path.basename(path)}: acc={acc} n={len(out)} -> {gdest}")
    else:
        print(f"[grounding] {os.path.basename(path)}: acc={acc} n={len(out)} -> {dest}")


def main():
    # Comma or whitespace separated — RUN_ENV forwarding can't carry spaces inside a value.
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
