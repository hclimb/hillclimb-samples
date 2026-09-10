"""
Base model QA evaluator: provides pos_doc in the prompt, uses vLLM for generation.
No JAX required — runs in the eval.py parent process after the JAX worker exits.
"""

import asyncio
import json
import os

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from evals.vllm import VLLMInference
from evals.utils import split_thinking

DOC_SEPARATOR = "<|doc_seperator|>"

PROMPT_TEMPLATE = """\
Document(s):
{documents}

Question:
{question}"""


def _format_docs(pos_doc: str) -> str:
    docs = [d.strip() for d in pos_doc.split(DOC_SEPARATOR) if d.strip()]
    if len(docs) == 1:
        return docs[0]
    return "\n\n---\n\n".join(f"[Document {i+1}]\n{d}" for i, d in enumerate(docs))


def run_generation(eval_cfg: dict, dataset_cfg: dict, output_file: str):
    """
    Load QA dataset, run vLLM generation with doc in context, run LLM judge,
    and write results to output_file.

    Args:
        eval_cfg:    The eval sub-config dict (vllm_model, num_samples, etc.)
        dataset_cfg: The dataset sub-config dict (hf_name, split, field_map, etc.)
        output_file: Absolute path to write results JSON.
    """
    hf_token = os.environ.get("HF_TOKEN")

    hf_name = dataset_cfg["hf_name"]
    split = dataset_cfg.get("split", "validation")
    field_map = dataset_cfg.get("field_map") or {}
    num_samples = eval_cfg.get("num_samples", None)

    vllm_model = eval_cfg["vllm_model"]
    vllm_base_url = eval_cfg.get("vllm_base_url", "http://localhost:8001/v1")
    vllm_tensor_parallel_size = eval_cfg.get("vllm_tensor_parallel_size", 8)
    vllm_max_model_len = eval_cfg.get("vllm_max_model_len", 32768)
    max_new_tokens = eval_cfg.get("max_new_tokens", 512)
    temperature = eval_cfg.get("temperature", 0.0)
    concurrency = eval_cfg.get("concurrency", 32)
    max_prompt_tokens = vllm_max_model_len - max_new_tokens

    # ----------------------------------------------------------------
    # Load tokenizer for length filtering
    # ----------------------------------------------------------------
    print(f"[gen_base_model] Loading tokenizer for '{vllm_model}'...")
    tokenizer = AutoTokenizer.from_pretrained(vllm_model, token=hf_token)

    # ----------------------------------------------------------------
    # Load QA dataset
    # ----------------------------------------------------------------
    print(f"[gen_base_model] Loading '{hf_name}' (split={split})...")
    ds = load_dataset(hf_name, split=split, streaming=True, token=hf_token)

    samples = []
    skipped = 0
    for item in ds:
        # Apply field_map to rename columns to standard names
        for std, src in field_map.items():
            if src in item:
                item[std] = item[src]
        question = item.get("question", "")
        answer = item.get("answer", "")
        pos_doc = item.get("pos_doc", "")
        if not question or not answer or not pos_doc:
            continue
        prompt = PROMPT_TEMPLATE.format(documents=_format_docs(pos_doc), question=question)
        prompt_len = len(tokenizer(prompt, truncation=False)["input_ids"])
        if prompt_len > max_prompt_tokens:
            skipped += 1
            continue
        samples.append({"question": question, "answer": answer, "pos_doc": pos_doc, "prompt": prompt})
        if num_samples is not None and len(samples) >= num_samples:
            break

    print(f"[gen_base_model] Loaded {len(samples)} samples (skipped {skipped} exceeding {max_prompt_tokens} tokens).")

    # ----------------------------------------------------------------
    # vLLM generation
    # ----------------------------------------------------------------
    _server_proc = None
    try:
        if not VLLMInference.is_server_ready(vllm_base_url):
            print(f"[gen_base_model] Starting vLLM server: {vllm_model} ...")
            _server_proc = VLLMInference.start_server(
                model=vllm_model,
                base_url=vllm_base_url,
                tensor_parallel_size=vllm_tensor_parallel_size,
                max_model_len=vllm_max_model_len,
                free_tpu=True,
            )
            VLLMInference.wait_for_server(vllm_base_url)
        else:
            print(f"[gen_base_model] Reusing running vLLM server at {vllm_base_url}")

        client = VLLMInference(model=vllm_model, base_url=vllm_base_url)

        async def _run_all():
            semaphore = asyncio.Semaphore(concurrency)

            async def _one(s):
                async with semaphore:
                    content, reasoning = await client.async_chat(
                        prompt=s["prompt"],
                        max_completion_tokens=max_new_tokens,
                        temperature=temperature,
                        thinking=True,
                    )
                return content, reasoning

            tasks = [_one(s) for s in samples]
            return await asyncio.gather(*tasks)

        generated = asyncio.run(_run_all())

    finally:
        if _server_proc is not None:
            _server_proc.terminate()
            _server_proc.wait()

    results = []
    for s, (content, reasoning) in zip(samples, generated):
        # Combine reasoning and content to get the full response for parsing
        full_response = reasoning + content if reasoning else content

        # Parse thinking tags to detect if model hit max tokens before completing
        think_content, answer_content = split_thinking(full_response)

        # If no </think> tag found (answer_content is ""), model hit max tokens
        generated_answer = answer_content if answer_content else None

        results.append({
            "prompt": s["question"],
            "document": s["pos_doc"],
            "generated_answer": generated_answer,
            "reasoning": think_content,
            "ground_truth": s["answer"],
        })


    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({"metrics": {}, "samples": results}, f, indent=2)
    print(f"[gen_base_model] Saved {len(results)} generations to {output_file}")
