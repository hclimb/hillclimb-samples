import os
import sys
import json
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lm import chat as _chat_shared


def _chat(prompt, max_new_tokens=1024):
    return _chat_shared(
        "You are a question-generation assistant. Output only valid JSON.",
        prompt,
        max_new_tokens=max_new_tokens,
    )


def _camel_to_words(s):
    """doctoralAdvisor -> doctoral advisor, almaMater -> alma mater"""
    return re.sub(r'([A-Z])', r' \1', s).lower().strip()


def _validate(question, start, intermediate_names, answer):
    """Returns (ok, reason) for a generated question."""
    q = question.lower()
    start_norm = start.lower().replace("_", " ")
    if start_norm not in q:
        return False, f"missing start entity '{start}'"
    for name in intermediate_names:
        if name.lower().replace("_", " ") in q:
            return False, f"leaks intermediate '{name}'"
    if answer.lower() in q:
        return False, f"answer leaked into question"
    return True, ""


def generate_question(paragraphs, path, max_retries=3):
    """
    paragraphs : dict of {entity_name: text}
    path       : list of {"subject": ..., "relation": ..., "object": ...} dicts

    Returns {"question": str, "answer": str}
    """
    start         = path[0]["subject"].replace("_", " ")
    answer_entity = path[-1]["object"].replace("_", " ")

    chain_lines = [
        f'{hop["subject"].replace("_", " ")} --[{_camel_to_words(hop["relation"])}]--> {hop["object"].replace("_", " ")}'
        for hop in path
    ]
    chain_str = "\n".join(chain_lines)

    context_parts = [
        f'[{entity.replace("_", " ")}]\n{text}'
        for entity, text in paragraphs.items()
        if text
    ]
    context_str = "\n\n".join(context_parts)

    intermediate_names = [hop["object"].replace("_", " ") for hop in path[:-1]]
    forbidden_str = ", ".join(f'"{n}"' for n in intermediate_names)

    prompt = f"""Write a multi-hop question starting from "{start}".

The chain below leads to "{answer_entity}". Do NOT use "{answer_entity}" as the answer.
Instead: find a specific, interesting fact stated in the [{answer_entity}] passage and make THAT the answer.
The question must require the reader to (1) navigate the chain to identify "{answer_entity}", then (2) look up that fact in its passage.

Rules:
1. "{start}" MUST appear by name in the question — it anchors the whole chain.
2. Every hop must be necessary. A reader who only knows about "{start}" should not be able to answer directly.
3. Do NOT name any intermediate entity ({forbidden_str}). Refer to each only through its role in the chain.
4. Do NOT use relative clauses like ", which ...", ", who ...", or ", that ..." to describe intermediates.
5. The answer must be a specific fact extracted from the [{answer_entity}] passage, not just the entity name itself.
6. Use natural, varied phrasing.

Chain:
{chain_str}

Wikipedia passages:
{context_str}

Output ONLY valid JSON with the question and the specific fact as the answer:
{{"question": "...", "answer": "..."}}"""

    for attempt in range(max_retries):
        raw = _chat(prompt, max_new_tokens=1024)
        result = None
        try:
            cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            result = json.loads(cleaned)
        except Exception:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                try:
                    result = json.loads(m.group())
                except Exception:
                    pass

        if result and result.get("question"):
            ok, reason = _validate(result["question"], start, intermediate_names, result["answer"])
            if ok:
                return result
            print(f"    [gen retry {attempt+1}] {reason}: {result['question']!r}")

    return {"question": "", "answer": ""}
