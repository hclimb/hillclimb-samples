#!/usr/bin/env python3
"""
generator.py — RAG generation step
====================================

Reads retrieval results (results_results.json style), builds a RAG prompt
with the top-K retrieved documents, calls a vLLM server, and saves
{query, ground_truth, answer} to an output JSON.

Usage:
  python generator.py \
    --input results_results.json \
    --output gen_results.json \
    --model Qwen/Qwen3-32B \
    --base_url http://localhost:8000/v1 \
    --top_k_docs 5 \
    --thinking          # omit for no-think mode
"""

import argparse
import asyncio
import json
import logging
import re
import time
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# RAG prompt builder
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a helpful research assistant. Answer questions based on the "
    "provided context documents."
)


def extract_question(query: str) -> str:
    """Strip Qwen3-Embedding instruction prefix if present, returning the raw question."""
    # Format: "Instruct: ...\nQuery:<question>"
    m = re.search(r"\nQuery:(.*)", query, re.DOTALL)
    if m:
        return m.group(1).strip()
    return query.strip()


def build_rag_prompt(query: str, retrieved: list[dict], top_k: int = 5) -> str:
    """
    Build a RAG prompt with top_k retrieved documents.

    Each entry in `retrieved` must have a 'document' key (already ordered by rank).
    """
    question = extract_question(query)
    docs = retrieved[:top_k]

    doc_block = ""
    for i, doc in enumerate(docs, 1):
        doc_block += f"[Document:]\n{doc['document'].strip()}\n\n"

    prompt = (
        f"You have been provided with {len(docs)} relevant document(s) retrieved to "
        f"help answer the question below.\n\n"
        f"---\n\n"
        f"{doc_block.strip()}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return prompt


# ═══════════════════════════════════════════════════════════════════════════
# Async generation loop
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_transformers(items: list[dict], args) -> list[dict]:
    """Generate using HuggingFace transformers (no server needed, CPU/GPU)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    log.info(f"Loading {args.model} via transformers...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    results = []
    for item in tqdm(items, desc="Generating"):
        prompt = build_rag_prompt(item["query"], item["retrieved"], top_k=args.top_k_docs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append({
            "query": item["query"],
            "ground_truth": item["ground_truth"],
            "answer": answer,
        })
    return results


async def generate_all(items: list[dict], client, args) -> list[dict]:
    sem = asyncio.Semaphore(args.concurrency)
    pbar = tqdm(total=len(items), desc="Generating")

    async def generate_one(item: dict) -> dict:
        prompt = build_rag_prompt(item["query"], item["retrieved"], top_k=args.top_k_docs)
        async with sem:
            answer, _ = await client.async_chat(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                max_completion_tokens=args.max_tokens,
                temperature=args.temperature,
                thinking=args.thinking,
            )
        pbar.update(1)
        return {
            "query": item["query"],
            "ground_truth": item["ground_truth"],
            "answer": answer,
        }

    tasks = [generate_one(item) for item in items]
    results = await asyncio.gather(*tasks)
    pbar.close()
    return list(results)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description="RAG generation step — calls vLLM with retrieved context"
    )
    p.add_argument("--input", required=True,
                   help="Path to retrieval results JSON (e.g. results_results.json)")
    p.add_argument("--output", required=True,
                   help="Path for generation output JSON")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B",
                   help="vLLM model name")
    p.add_argument("--base_url", default="http://localhost:8000/v1",
                   help="vLLM server base URL")
    p.add_argument("--top_k_docs", type=int, default=5,
                   help="Number of retrieved docs to include in the prompt")
    p.add_argument("--max_tokens", type=int, default=1024,
                   help="Max generation tokens per answer")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--thinking", action="store_true",
                   help="Enable chain-of-thought thinking (Qwen3 think mode)")
    p.add_argument("--concurrency", type=int, default=64,
                   help="Number of concurrent async requests to vLLM")
    p.add_argument("--num_questions", type=int, default=None,
                   help="Only process the first N questions (default: all)")
    # Server launch options (optional)
    p.add_argument("--start_server", action="store_true",
                   help="Launch a vLLM server subprocess before running")
    p.add_argument("--tensor_parallel_size", type=int, default=8)
    p.add_argument("--max_model_len", type=int, default=16384)
    p.add_argument("--download_dir", default="/dev/shm")
    p.add_argument("--use_transformers", action="store_true",
                   help="Use HuggingFace transformers directly (no vLLM server needed)")
    return p


def main():
    args = build_parser().parse_args()

    if args.use_transformers:
        log.info("Using HuggingFace transformers for generation (no vLLM server)")
        with open(args.input, encoding="utf-8") as f:
            items = json.load(f)
        if args.num_questions is not None:
            items = items[:args.num_questions]
        log.info(f"Loaded {len(items)} items")
        t0 = time.time()
        results = generate_all_transformers(items, args)
        elapsed = time.time() - t0
        log.info(f"Generated {len(results)} answers in {elapsed:.1f}s")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log.info(f"Saved → {args.output}")
        return

    from vllm import VLLMInference

    server_proc = None
    if args.start_server:
        log.info(f"Starting vLLM server for {args.model}...")
        server_proc = VLLMInference.start_server(
            model=args.model,
            base_url=args.base_url,
            tensor_parallel_size=args.tensor_parallel_size,
            download_dir=args.download_dir,
            max_model_len=args.max_model_len,
        )

    try:
        VLLMInference.wait_for_server(args.base_url)
        client = VLLMInference(model=args.model, base_url=args.base_url)

        log.info(f"Loading retrieval results from {args.input}...")
        with open(args.input, encoding="utf-8") as f:
            items = json.load(f)
        if args.num_questions is not None:
            items = items[:args.num_questions]
        log.info(f"Loaded {len(items)} items")

        log.info(
            f"Generating answers — think={args.thinking}, "
            f"top_k_docs={args.top_k_docs}, concurrency={args.concurrency}"
        )
        t0 = time.time()
        results = asyncio.run(generate_all(items, client, args))
        elapsed = time.time() - t0
        log.info(f"Generated {len(results)} answers in {elapsed:.1f}s ({len(results)/elapsed:.1f} QPS)")

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log.info(f"Saved → {args.output}")

    finally:
        if server_proc is not None:
            import os
            import signal
            log.info("Shutting down vLLM server...")
            try:
                os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            server_proc.wait()


if __name__ == "__main__":
    main()
