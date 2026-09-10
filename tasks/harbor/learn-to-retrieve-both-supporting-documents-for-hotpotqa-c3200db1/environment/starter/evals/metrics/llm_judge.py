"""LLM-based judge metric using a vLLM server."""

import asyncio
import re

from evals.vllm import VLLMInference

JUDGE_SYSTEM_PROMPT = '''
You are an answer equivalence judge. Your task is to determine whether a generated answer covers the core factual point expressed in a ground truth answer.

The question is provided only for context to help you understand what is being asked — do not use it to independently evaluate correctness. The ground truth answer is the sole source of truth.

Rules:
- The ground truth answer is the divine truth. Judge solely based on whether the generated answer matches it.
- Focus only on whether the core information/claim in the ground truth is present in the generated answer.
- The generated answer may be more verbose, more concise, differently phrased, or written in a different style — this is fine as long as the core point is covered.
- Do NOT penalize for extra information in the generated answer.
- Do NOT require word-for-word matching.
- If the ground truth contains multiple points, the generated answer must cover all of them to be a match.
- If the generated answer contradicts or omits the core point of the ground truth, it is not a match.
'''

JUDGE_USER_TEMPLATE = '''
Question (for context only):
{question}

Ground Truth:
{ground_truth}

Generated Answer:
{answer}

Does the generated answer cover the core point(s) of the ground truth? Output your final verdict in the tag below.
Use exactly one of:
<judgement>match</judgement>
<judgement>not match</judgement>
'''

JUDGE_USER_TEMPLATE_WITH_DOC = '''
Question (for context only):
{question}

Document:
{document}

Ground Truth:
{ground_truth}

Generated Answer:
{answer}

Does the generated answer cover the core point(s) of the ground truth? Output your final verdict in the tag below.
Use exactly one of:
<judgement>match</judgement>
<judgement>not match</judgement>
'''

JUDGEMENT_PATTERN = re.compile(r"<judgement>(.*?)</judgement>", re.IGNORECASE | re.DOTALL)


def _parse_score(text: str) -> float:
    # LAST tag wins, not first: with thinking=True the judge sometimes writes a tentative
    # <judgement> while reasoning through a hypothetical ("if the answer were X, that would be
    # a match...") before landing on a different final verdict after </think>. `.search()` grabs
    # whichever comes first in the string, which silently picks up the hypothetical instead of
    # the real conclusion.
    matches = JUDGEMENT_PATTERN.findall(text)
    if matches:
        return 1.0 if matches[-1].strip().lower() == "match" else 0.0
    # Fallback: check for "not match" before "match" to avoid false positives
    lower = text.lower()
    if "not match" in lower:
        return 0.0
    if "match" in lower:
        return 1.0
    return 0.0


# Exact 0-5 scoring prompt from the MSA paper (Appendix A) / reference impl
# (github.com/EverMind-AI/MSA src/evaluation/llm_judge.py build_score_prompt).
PAPER_SCORE_PROMPT = '''Based on the accuracy, completeness, and relevance of the predicted answer to the real answer in the context of the **query**, assign an objective score from 0 to 5 (5 being the highest, 0 the lowest).

The scoring must strictly adhere to the following criteria. The final output can only be a single number.

Scoring Criteria:

5: The predicted answer is exactly the same as the real answer and correctly answers the query. Differences in wording do not affect factual accuracy.

4: The predicted answer contains all the core information of the real answer, with no errors, but includes a small amount of non-critical redundant content.

3: The predicted answer captures the core information but differs from the real answer in some aspects. The predicted answer is slightly incomplete or imprecise, but contains no errors.

2: The predicted answer is partially relevant to the real answer but omits a significant amount of information or deviates from the core topic of the query.

1: The predicted answer attempts to address the query (maintains basic relevance to the topic) but provides factually incorrect information. It does not contradict the core claim of the real answer, but shows incomplete or inaccurate understanding of the topic.

0. The predicted answer is completely unrelated to the query, consists of gibberish, or is a pure hallucination that shares no logical connection with the real answer.

Query:

{query}

True Answer:

{gold_answer}

Predicted Answer:

{model_answer}

Output only a single number (0, 1, 2, 3, 4, or 5): '''


def _parse_score_05(text: str) -> float:
    """Take the LAST standalone 0-5 digit in the output (paper outputs a bare number)."""
    digits = re.findall(r"[0-5]", text or "")
    return float(int(digits[-1])) if digits else 0.0


def llm_judge_score(
    results,
    model_id="Qwen/Qwen3-8B",
    base_url="http://localhost:8000/v1",
    max_new_tokens=8,
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    concurrency=32,
    tensor_parallel_size=8,
    **kwargs,
):
    """MSA-paper 0-5 LLM judge. Returns (scores in [0,5], judge outputs)."""
    _server_proc = None
    try:
        if not VLLMInference.is_server_ready(base_url):
            _server_proc = VLLMInference.start_server(model=model_id, base_url=base_url, max_model_len=4096, tensor_parallel_size=tensor_parallel_size)
            VLLMInference.wait_for_server(base_url)
        client = VLLMInference(model=model_id, base_url=base_url)

        async def _run_all():
            semaphore = asyncio.Semaphore(concurrency)

            async def _one(r):
                av = r.get("generated_answer")
                answer = av.strip() if av else "[no answer]"
                user_prompt = PAPER_SCORE_PROMPT.format(
                    query=r.get("prompt", ""),
                    gold_answer=r.get("ground_truth", "").strip(),
                    model_answer=answer,
                )
                async with semaphore:
                    content, _ = await client.async_chat(
                        prompt=user_prompt, system=None,
                        max_completion_tokens=max_new_tokens,
                        temperature=temperature, thinking=False,
                        top_p=top_p, top_k=top_k,
                    )
                return content

            return await asyncio.gather(*[_one(r) for r in results])

        outputs = asyncio.run(_run_all())
        scores = [_parse_score_05(t) for t in outputs]
        return scores, outputs
    finally:
        if _server_proc is not None:
            _server_proc.terminate()
            _server_proc.wait()


def llm_judge_accuracy(
    results,
    model_id="Qwen/Qwen3-8B",
    base_url="http://localhost:8000/v1",
    max_new_tokens=1024,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    concurrency=32,
    use_document=False,
    tensor_parallel_size=8,
    **kwargs,
):
    """
    Run LLM judge over all results using a vLLM server.

    Reuses an already-running server if available; otherwise starts one and
    shuts it down when done.

    Returns (scores, outputs) where scores are 0.0/1.0 and outputs are the
    judge response strings.
    """
    _server_proc = None
    try:
        if not VLLMInference.is_server_ready(base_url):
            _server_proc = VLLMInference.start_server(model=model_id, base_url=base_url, max_model_len=4096, tensor_parallel_size=tensor_parallel_size)
            VLLMInference.wait_for_server(base_url)
        client = VLLMInference(model=model_id, base_url=base_url)

        async def _run_all():
            semaphore = asyncio.Semaphore(concurrency)

            async def _one(r):
                answer_value = r.get("generated_answer")
                answer = answer_value.strip() if answer_value is not None else "[Answer incomplete - max tokens reached]"
                if use_document:
                    user_prompt = JUDGE_USER_TEMPLATE_WITH_DOC.format(
                        question=r["prompt"],
                        document=r.get("document", "").strip(),
                        ground_truth=r["ground_truth"].strip(),
                        answer=answer,
                    )
                else:
                    user_prompt = JUDGE_USER_TEMPLATE.format(
                        question=r["prompt"],
                        ground_truth=r["ground_truth"].strip(),
                        answer=answer,
                    )
                async with semaphore:
                    content, reasoning = await client.async_chat(
                        prompt=user_prompt,
                        system=JUDGE_SYSTEM_PROMPT,
                        max_completion_tokens=max_new_tokens,
                        temperature=temperature,
                        thinking=True,
                        top_p=top_p,
                        top_k=top_k,
                    )
                return content, reasoning

            return await asyncio.gather(*[_one(r) for r in results])

        pairs = asyncio.run(_run_all())
        outputs = [p[0] for p in pairs]
        reasonings = [p[1] for p in pairs]
        scores = [_parse_score(text) for text in outputs]
        explanations = []
        for content, reasoning in zip(outputs, reasonings):
            if reasoning:
                explanations.append("<think>" + reasoning + "</think>\n\n" + content)
            else:
                explanations.append(content)
        return scores, explanations
    finally:
        if _server_proc is not None:
            _server_proc.terminate()
            _server_proc.wait()
