"""
LLM-as-judge evaluation script.

Evaluates whether generated answers match ground truth answers using a vLLM judge model.

Usage:
    # JSON input
    uv run python -m llmeval.run_eval --input results.json

    # HuggingFace dataset
    uv run python -m llmeval.run_eval --input owner/dataset --input_type hf --hf_split test

    # Start server automatically
    uv run python -m llmeval.run_eval --input results.json --start_server

    # Custom column names
    uv run python -m llmeval.run_eval --input results.json --ground_truth_col gt --answer_col pred
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_TP_SIZE = 8
DEFAULT_MAX_TOKENS = 2048
DEFAULT_PROMPT_FILE = Path(__file__).parent / "judge_prompt.txt"
JUDGEMENT_PATTERN = re.compile(r"<judgement>(.*?)</judgement>", re.IGNORECASE | re.DOTALL)


def parse_args():
    p = argparse.ArgumentParser(description="LLM-as-judge evaluator")
    p.add_argument("--input", required=True, help="Path to JSON file OR HuggingFace dataset name/path")
    p.add_argument("--input_type", choices=["json", "hf"], default=None, help="Input type (auto-detected if omitted)")
    p.add_argument("--hf_split", default="train", help="HF dataset split to use (default: train)")
    p.add_argument("--ground_truth_col", default="ground_truth", help="Column name for ground truth (default: ground_truth)")
    p.add_argument("--answer_col", default="answer", help="Column name for generated answer (default: answer)")
    p.add_argument("--question_col", default="query", help="Column name for the question (default: question)")
    p.add_argument("--prompt_file", default=str(DEFAULT_PROMPT_FILE), help="Path to judge prompt template file")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"vLLM model name (default: {DEFAULT_MODEL})")
    p.add_argument("--base_url", default=DEFAULT_BASE_URL, help=f"vLLM server base URL (default: {DEFAULT_BASE_URL})")
    p.add_argument("--n", type=int, default=None, help="Number of data points to evaluate (default: all)")
    p.add_argument("--start_server", action="store_true", help="Start a vLLM server before running eval")
    p.add_argument("--concurrency", type=int, default=32, help="Max concurrent requests (default: 2048)")
    p.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help=f"Max tokens for judge response (default: {DEFAULT_MAX_TOKENS})")
    p.add_argument("--output_file", default="eval_results.json", help="Output JSON file with per-item results (default: eval_results.json)")
    p.add_argument("--summary_file", default="eval_summary.json", help="Output JSON file with summary stats (default: eval_summary.json)")
    p.add_argument("--tensor_parallel_size", type=int, default=DEFAULT_TP_SIZE, help=f"Tensor parallel size when starting server (default: {DEFAULT_TP_SIZE})")
    p.add_argument("--tpu_visible_devices", default=None, help="TPU_VISIBLE_DEVICES env var when starting server")
    p.add_argument("--max_model_len", type=int, default=8192, help="Max model length when starting server (default: 8192)")
    p.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature for judge (default: 0.6)")
    p.add_argument("--top_p", type=float, default=0.95, help="Top-p nucleus sampling (default: 0.95)")
    p.add_argument("--top_k", type=int, default=20, help="Top-k sampling (default: 20)")
    return p.parse_args()


def detect_input_type(input_path: str) -> str:
    """Auto-detect whether input is a local JSON file or an HF dataset identifier."""
    p = Path(input_path)
    if p.suffix in (".json", ".jsonl") or p.exists():
        return "json"
    return "hf"


def load_json(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix == ".jsonl":
        with open(p) as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(p) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"JSON file must contain a list of objects, got {type(data)}")


def load_hf(dataset_name: str, split: str) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: uv add datasets", file=sys.stderr)
        sys.exit(1)
    ds = load_dataset(dataset_name, split=split)
    return [dict(row) for row in ds]


def load_prompt_template(path: str) -> str:
    with open(path) as f:
        return f.read()


def build_prompt(template: str, row: dict, ground_truth_col: str, answer_col: str, question_col: str = "question", **extra_fields) -> str:
    """Format the prompt template. Supports {ground_truth}, {answer}, {question}, and any extra fields."""
    kwargs = {k: v for k, v in row.items()}
    # Ensure the canonical names are available regardless of column names
    kwargs["ground_truth"] = row[ground_truth_col]
    kwargs["answer"] = row[answer_col]
    kwargs["question"] = row[question_col]
    kwargs.update(extra_fields)
    return template.format(**kwargs)


def parse_judgement(response: str) -> str:
    """Extract judgement from <judgement>...</judgement> tag. Returns 'match', 'not match', or 'parse_error'."""
    m = JUDGEMENT_PATTERN.search(response)
    if not m:
        return "parse_error"
    return m.group(1).strip().lower()


async def evaluate_all(
    rows: list[dict],
    prompt_template: str,
    ground_truth_col: str,
    answer_col: str,
    question_col: str,
    vllm,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    concurrency: int,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    results = [None] * len(rows)

    async def evaluate_one(idx: int, row: dict):
        prompt = build_prompt(prompt_template, row, ground_truth_col, answer_col, question_col)
        async with semaphore:
            try:
                response, thinking = await vllm.async_chat(
                    prompt=prompt,
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                    thinking=True,
                    top_p=top_p,
                    top_k=top_k,
                )
            except Exception as e:
                print(f"  [idx={idx}] Request failed: {e}", file=sys.stderr)
                results[idx] = {**row, "__judgement__": "error", "__response__": str(e), "__thinking__": None}
                return

        judgement = parse_judgement(response)
        results[idx] = {**row, "__judgement__": judgement, "__response__": response, "__thinking__": thinking}
        status = "✓" if judgement == "match" else ("?" if judgement == "parse_error" else "✗")
        print(f"  [{idx+1}/{len(rows)}] {status} {judgement}")

    tasks = [evaluate_one(i, row) for i, row in enumerate(rows)]
    await asyncio.gather(*tasks)
    return results


def write_summary(results: list[dict], summary_file: str):
    total = len(results)
    match = sum(1 for r in results if r["__judgement__"] == "match")
    not_match = sum(1 for r in results if r["__judgement__"] == "not match")
    parse_errors = sum(1 for r in results if r["__judgement__"] == "parse_error")
    errors = sum(1 for r in results if r["__judgement__"] == "error")

    summary = {
        "total": total,
        "match": match,
        "not_match": not_match,
        "parse_errors": parse_errors,
        "request_errors": errors,
        "accuracy": round(match / total, 4) if total > 0 else 0.0,
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Results: {match}/{total} match  ({summary['accuracy']*100:.1f}%)")
    print(f"  match:        {match}")
    print(f"  not match:    {not_match}")
    print(f"  parse errors: {parse_errors}")
    print(f"  req errors:   {errors}")
    print(f"Summary saved to: {summary_file}")


def main():
    args = parse_args()

    # Detect input type
    input_type = args.input_type or detect_input_type(args.input)
    print(f"Input type: {input_type}")

    # Load data
    if input_type == "json":
        rows = load_json(args.input)
    else:
        rows = load_hf(args.input, args.hf_split)

    # Validate columns
    if rows and args.ground_truth_col not in rows[0]:
        print(f"ERROR: ground_truth column '{args.ground_truth_col}' not found. Available: {list(rows[0].keys())}", file=sys.stderr)
        sys.exit(1)
    if rows and args.answer_col not in rows[0]:
        print(f"ERROR: answer column '{args.answer_col}' not found. Available: {list(rows[0].keys())}", file=sys.stderr)
        sys.exit(1)
    if rows and args.question_col not in rows[0]:
        print(f"ERROR: question column '{args.question_col}' not found. Available: {list(rows[0].keys())}", file=sys.stderr)
        sys.exit(1)

    # Subset
    if args.n is not None:
        rows = rows[: args.n]
    print(f"Evaluating {len(rows)} data points...")

    # Load prompt template
    prompt_template = load_prompt_template(args.prompt_file)

    # Optionally start server
    server_proc = None
    if args.start_server:
        from vllm import VLLMInference
        print(f"Starting vLLM server for model: {args.model}")
        server_proc = VLLMInference.start_server(
            model=args.model,
            base_url=args.base_url,
            tensor_parallel_size=args.tensor_parallel_size,
            tpu_visible_devices=args.tpu_visible_devices,
            max_model_len=args.max_model_len,
        )
        VLLMInference.wait_for_server(args.base_url)
    else:
        from vllm import VLLMInference

    vllm = VLLMInference(model=args.model, base_url=args.base_url)

    try:
        results = asyncio.run(
            evaluate_all(
                rows=rows,
                prompt_template=prompt_template,
                ground_truth_col=args.ground_truth_col,
                answer_col=args.answer_col,
                question_col=args.question_col,
                vllm=vllm,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                concurrency=args.concurrency,
            )
        )
    finally:
        if server_proc is not None:
            import os
            import signal
            print("Shutting down vLLM server...")
            try:
                os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            server_proc.wait()

    # Save results
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {args.output_file}")

    write_summary(results, args.summary_file)


if __name__ == "__main__":
    main()
