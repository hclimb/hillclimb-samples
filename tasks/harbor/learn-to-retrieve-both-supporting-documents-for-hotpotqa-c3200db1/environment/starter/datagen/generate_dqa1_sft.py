#!/usr/bin/env python3
"""
DQA1 SFT data generation.

Reads parquets from vm2825/nemotron-cc-v2-Parsed-DQA1-answered-1.7B-combined-hard-negatives-scored-modified,
generates CoT answers using Qwen3-4B, filters by token length (think + answer <= 512 tokens)
and answer quality (LLM judge), uploads matching parquets to mihir-1999/nemotron-combined-hard-neg.

Cross-machine safe: lists completed parquets in output repo at startup and
skips them. Last-write-wins if two machines race on the same file.

Usage:
    python datagen/generate_dqa1_sft.py
    python datagen/generate_dqa1_sft.py --dry-run   # skip HF uploads
"""

import asyncio
import argparse
import logging
import os
import re
import sys

import dotenv

dotenv.load_dotenv()

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tqdm.asyncio import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'persistent_tpu'))
from vllm import VLLMInference

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ─── constants ────────────────────────────────────────────────────────────────
MODEL          = "Qwen/Qwen3-4B"
BASE_URL       = "http://localhost:8000/v1"
TP_SIZE        = 4
TPU_DEVICES    = "0,1,2,3"
MAX_MODEL_LEN  = 8192
MAX_NUM_SEQS   = 256

INPUT_REPO     = "vm2825/nemotron-cc-v2-Parsed-DQA1-answered-1.7B-combined-hard-negatives-scored-modified"
OUTPUT_REPO    = "mihir-1999/nemotron-combined-hard-neg"

MAX_OUTPUT_TOKENS    = 1024   # vLLM generation budget (generous; filter catches overruns)
MAX_THINK_ANS_TOKENS = 512    # combined think + answer token budget
MAX_CONCURRENCY      = 256
LOCAL_DIR            = "./temp_scienceqa_upload"

SCHEMA = pa.schema([
    ("query",            pa.string()),
    ("pos_doc",          pa.string()),
    ("answer",           pa.string()),
    ("neg_docs",         pa.string()),
    ("neg_scores",       pa.string()),
    ("think",            pa.string()),
    ("generated_answer", pa.string()),
])

# ─── prompts ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = "You are a helpful assistant."

ANSWER_PROMPT = """\
I am going to give you some additional information that might help you answer a question.

Additional Information:
{pos_doc}

INSTRUCTION: First understand what the question is asking, then, step by step, go and retrieve any information that might be helpful. Finally, put all the information together to answer the question below. Be very concise with your reasoning. IMPORTANT: BE VERY CONCISE WITH YOUR THINKING AND PROVIDE THE ANSWER AS SOON AS SOMETHING MAKES SENSE.

QUESTION: {question}"""

JUDGE_PROMPT = """\
Question: {question}
Ground truth answer: {answer}
Model answer: {generated_answer}

Does the model answer correctly answer the question, matching the ground truth? Answer with <verdict>YES</verdict> or <verdict>NO</verdict> only."""


# ─── row processing ───────────────────────────────────────────────────────────
async def process_row(
    row: dict,
    vllm: VLLMInference,
    tokenizer,
    semaphore: asyncio.Semaphore,
    stats: dict,
) -> dict | None:
    """Generate CoT + answer for one row; apply token and judge filters."""
    async with semaphore:
        question = row["query"]
        ground_truth = row["answer"]
        pos_doc = row["pos_doc"]

        # Generation
        try:
            generated_answer, think = await vllm.async_chat(
                prompt=ANSWER_PROMPT.format(
                    pos_doc=pos_doc,
                    question=question,
                ),
                system=SYSTEM_PROMPT,
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.6,
                thinking=True,
            )
        except Exception as e:
            err_str = str(e)
            if "max_tokens" in err_str and "too large" in err_str:
                stats["input_too_long"] += 1
            else:
                logging.warning(f"Generation error: {e}")
                stats["gen_error"] += 1
            return None

        think = (think or "").strip()
        generated_answer = generated_answer.strip()

        # Token filter
        if len(tokenizer.encode(think + generated_answer)) > MAX_THINK_ANS_TOKENS:
            stats["token_filtered"] += 1
            return None

        # Judge filter
        try:
            verdict_text, _ = await vllm.async_chat(
                prompt=JUDGE_PROMPT.format(
                    question=question,
                    answer=ground_truth,
                    generated_answer=generated_answer,
                ),
                system=SYSTEM_PROMPT,
                max_completion_tokens=16,
                temperature=0.0,
                thinking=False,
            )
        except Exception as e:
            logging.warning(f"Judge error: {e}")
            stats["judge_error"] += 1
            return None

        verdict_match = re.search(r"<verdict>(YES|NO)</verdict>", verdict_text, re.IGNORECASE)
        if not verdict_match or verdict_match.group(1).upper() != "YES":
            stats["judge_rejected"] += 1
            return None

        stats["accepted"] += 1
        return {
            "query":            question,
            "pos_doc":          pos_doc,
            "answer":           ground_truth,
            "neg_docs":         str(row.get("neg_docs", "")),
            "neg_scores":       str(row.get("neg_scores", "")),
            "think":            think,
            "generated_answer": generated_answer,
        }


# ─── parquet processing ───────────────────────────────────────────────────────
async def process_parquet(
    parquet_filename: str,
    vllm: VLLMInference,
    tokenizer,
    api: HfApi,
    hf_token: str,
    dry_run: bool,
) -> None:
    # Download
    logging.info(f"Downloading {parquet_filename}...")
    dl_path = hf_hub_download(
        repo_id=INPUT_REPO,
        filename=parquet_filename,
        repo_type="dataset",
        token=hf_token,
    )

    rows = pq.read_table(dl_path).to_pylist()
    logging.info(f"  {len(rows)} rows loaded.")

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    stats = {k: 0 for k in ("accepted", "token_filtered", "judge_rejected", "gen_error", "judge_error", "input_too_long")}

    tasks = [process_row(row, vllm, tokenizer, semaphore, stats) for row in rows]
    results = await tqdm.gather(*tasks, desc=os.path.basename(parquet_filename))
    records = [r for r in results if r is not None]

    logging.info(
        f"  {parquet_filename}: {stats['accepted']} accepted | "
        f"{stats['token_filtered']} token-filtered | "
        f"{stats['judge_rejected']} judge-rejected | "
        f"{stats['input_too_long']} input-too-long | "
        f"{stats['gen_error']} gen-errors | "
        f"{stats['judge_error']} judge-errors"
    )

    if not records:
        logging.info("  No rows passed filters; skipping upload.")
        return

    # Write output parquet
    output_basename = os.path.basename(parquet_filename)
    local_output = os.path.join(LOCAL_DIR, output_basename)
    os.makedirs(LOCAL_DIR, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records, schema=SCHEMA), local_output)
    logging.info(f"  Wrote {len(records)} rows → {local_output}")

    logging.info(f"  Uploading to {OUTPUT_REPO}/{parquet_filename}...")
    api.upload_file(
        path_or_fileobj=local_output,
        path_in_repo=parquet_filename,
        repo_id=OUTPUT_REPO,
        repo_type="dataset",
        commit_message=f"Add SFT data: {parquet_filename}",
        token=hf_token,
    )
    logging.info(f"  Uploaded {parquet_filename}.")
    os.remove(local_output)


# ─── main ─────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Skip HF uploads")
    parser.add_argument("--start", type=int, default=0, help="Start index into sorted parquet list")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive) into sorted parquet list")
    parser.add_argument("--parquet-numbers", type=str, default=None, help="Comma-separated indices into sorted parquet list (overrides --start/--end)")
    parser.add_argument("--tp-size", type=int, default=TP_SIZE, help="Tensor parallel size (number of TPU chips)")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("Set HF_TOKEN or HUGGINGFACE_TOKEN in your environment.")

    api = HfApi(token=hf_token)

    # Determine pending parquets
    logging.info(f"Listing parquets in {INPUT_REPO}...")
    input_files = {
        f for f in api.list_repo_files(INPUT_REPO, repo_type="dataset")
        if f.endswith(".parquet") and "train" in f
    }
    logging.info(f"  {len(input_files)} parquets in input repo.")

    output_files: set[str] = set()
    api.create_repo(OUTPUT_REPO, repo_type="dataset", exist_ok=True)
    try:
        output_files = {
            f for f in api.list_repo_files(OUTPUT_REPO, repo_type="dataset")
            if f.endswith(".parquet")
        }
        logging.info(f"  {len(output_files)} parquets already in output repo.")
    except Exception:
        logging.info("  Output repo empty or does not exist yet.")

    all_input = sorted(input_files)
    if args.parquet_numbers is not None:
        indices = [int(x.strip()) for x in args.parquet_numbers.split(",")]
        selected = [all_input[i] for i in indices if i < len(all_input)]
    else:
        selected = all_input[args.start:args.end]
    pending = [f for f in selected if f not in output_files]
    logging.info(f"  {len(pending)} parquets to process.")

    if not pending:
        logging.info("Nothing to do.")
        return

    # Start vLLM server
    tp = args.tp_size
    tpu_devices = ",".join(str(i) for i in range(tp))
    logging.info(f"Starting {MODEL} on {tp} TPU chips...")
    server_proc = VLLMInference.start_server(
        model=MODEL,
        base_url=BASE_URL,
        tensor_parallel_size=tp,
        tpu_visible_devices=tpu_devices,
        download_dir="/dev/shm",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
    )
    try:
        VLLMInference.wait_for_server(BASE_URL, timeout=600)
    except TimeoutError:
        server_proc.terminate()
        raise

    vllm = VLLMInference(model=MODEL, base_url=BASE_URL)

    logging.info("Loading tokenizer for token counting...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    try:
        for i, parquet_filename in enumerate(pending):
            logging.info(f"[{i+1}/{len(pending)}] {parquet_filename}")
            await process_parquet(parquet_filename, vllm, tokenizer, api, hf_token, args.dry_run)
    finally:
        logging.info("Terminating vLLM server...")
        server_proc.terminate()
        server_proc.wait()
        logging.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
