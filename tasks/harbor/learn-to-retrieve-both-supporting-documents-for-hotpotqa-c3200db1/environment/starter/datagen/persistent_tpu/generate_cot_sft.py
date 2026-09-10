#!/usr/bin/env python3
import os
os.makedirs("/dev/shm/hf_cache", exist_ok=True)
os.environ["HF_HOME"] = "/dev/shm/hf_cache"
os.environ["VLLM_CACHE_ROOT"] = "/dev/shm/hf_cache"
os.environ["HF_HUB_CACHE"] = "/dev/shm/hf_cache"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
os.environ["LIBTPU_INIT_ARGS"] = (
    "--xla_tpu_enable_data_parallel_all_reduce_opt=true "
    "--xla_tpu_data_parallel_opt_different_sized_ops=true "
    "--xla_tpu_enable_async_collective_fusion=true "
    "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true "
    "--xla_tpu_enable_async_collective_fusion_multiple_steps=true"
)

"""
SFT-with-CoT generation, streaming edition.

Combines:
  - generate_scienceqa_sft.py's core prompting (Qwen thinking-mode -> think +
    answer, optional LLM-judge against ground truth).
  - generate_cot.py's scaffolding (HF streaming + row-level resume state file +
    threshold-triggered async uploads).

Input rows are expected to have at least `pos_doc` and `query` (or `question`).
`answer` (ground truth) is optional; the --use-judge filter is only applied
when --use-judge is passed.
"""

import asyncio
import argparse
import json
import logging
import os
import re
import sys
import uuid

import dotenv
dotenv.load_dotenv()

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(__file__))
from vllm_inference import VLLMInference

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ─── defaults ─────────────────────────────────────────────────────────────────
MODEL                = "Qwen/Qwen3-4B"
BASE_URL             = "http://localhost:8000/v1"
TP_SIZE              = 8
TPU_DEVICES          = "0,1,2,3,4,5,6,7"
MAX_MODEL_LEN        = 8192
MAX_NUM_SEQS         = 256

INPUT_REPO           = "vm2825/nemotron-cc-v21-Parsed-QA4-hard-negatives-scored-2M-modified"
HF_USERNAME          = "vm2825"

N_CANDIDATES         = 30_000_000
MAX_OUTPUT_TOKENS    = 1024
MAX_THINK_ANS_TOKENS = 512
MAX_CONCURRENCY      = 256

PARQUET_ROWS         = 50_000
UPLOAD_THRESHOLD_GB  = 0.05
LOCAL_DIR            = "./temp_sft_upload"

# Output schema is inferred from input rows at flush time; we pass through
# every input column and append `think` + `generated_answer`.

# ─── prompts (from generate_scienceqa_sft.py) ─────────────────────────────────
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


# ─── helpers ──────────────────────────────────────────────────────────────────
def get_dir_size(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    return sum(
        os.path.getsize(os.path.join(path, f))
        for f in os.listdir(path)
        if f.endswith(".parquet")
    )


def _upload_and_clear_sync(api: HfApi, local_dir: str, repo: str, upload_count: int):
    logging.info(f"Uploading batch {upload_count} to {repo}...")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=["*.parquet", "*.json"],
        commit_message=f"Upload batch {upload_count}",
    )
    logging.info("Upload successful. Clearing local files...")
    for f in os.listdir(local_dir):
        if f.endswith(".parquet") or f.endswith(".json"):
            os.remove(os.path.join(local_dir, f))


class ResultBuffer:
    def __init__(self, local_dir, parquet_rows, upload_threshold, api, repo, chunk_identity, skip_upload=False):
        self.local_dir = local_dir
        self.parquet_rows = parquet_rows
        self.upload_threshold = upload_threshold
        self.api = api
        self.repo = repo
        self.chunk_identity = chunk_identity
        self.skip_upload = skip_upload

        self.records: list[dict] = []
        self.highest_raw_index = -1
        self.chunk_index = 0
        self.upload_count = 0
        self.total_rows = 0
        self.is_uploading = False
        self._lock = asyncio.Lock()

    async def add(self, record, raw_index):
        async with self._lock:
            self.records.append(record)
            if raw_index > self.highest_raw_index:
                self.highest_raw_index = raw_index
            if len(self.records) >= self.parquet_rows:
                self._flush_locked()

        if not self.skip_upload and not self.is_uploading and get_dir_size(self.local_dir) >= self.upload_threshold:
            self.is_uploading = True
            asyncio.create_task(self._upload_task())

    def _flush_locked(self):
        table = pa.Table.from_pylist(self.records)
        unique_id = uuid.uuid4().hex[:8]
        path = os.path.join(self.local_dir, f"data_{unique_id}_{self.chunk_index:06d}.parquet")
        pq.write_table(table, path)

        state_path = os.path.join(self.local_dir, f"state_chunk_{self.chunk_identity}.json")
        with open(state_path, "w") as f:
            json.dump({"processed_raw_index": self.highest_raw_index}, f)

        self.total_rows += len(table)
        logging.info(f"Wrote {len(table)} rows → {path}  (total: {self.total_rows}, last index: {self.highest_raw_index})")
        self.records = []
        self.chunk_index += 1

    async def _upload_task(self):
        try:
            self.upload_count += 1
            await asyncio.to_thread(_upload_and_clear_sync, self.api, self.local_dir, self.repo, self.upload_count)
        except Exception as e:
            logging.error(f"Upload failed: {e}. Local files kept for retry.")
        finally:
            self.is_uploading = False
            if get_dir_size(self.local_dir) >= self.upload_threshold:
                self.is_uploading = True
                asyncio.create_task(self._upload_task())

    async def flush_and_upload_final(self):
        async with self._lock:
            if self.records:
                self._flush_locked()

        while self.is_uploading:
            await asyncio.sleep(5)

        if not self.skip_upload and get_dir_size(self.local_dir) > 0:
            self.upload_count += 1
            logging.info(f"Final upload of {get_dir_size(self.local_dir)} bytes to {self.repo}...")
            await asyncio.to_thread(_upload_and_clear_sync, self.api, self.local_dir, self.repo, self.upload_count)
        else:
            logging.info("Skipping final upload (no data).")


def fetch_resume_index(repo: str, chunk_identity: str, hf_token: str) -> int:
    from huggingface_hub import hf_hub_download
    try:
        logging.info(f"Checking HF Repo {repo} for state_chunk_{chunk_identity}.json...")
        file_path = hf_hub_download(
            repo_id=repo,
            filename=f"state_chunk_{chunk_identity}.json",
            repo_type="dataset",
            token=hf_token,
        )
        with open(file_path, "r") as f:
            data = json.load(f)
            idx = data.get("processed_raw_index", -1)
            logging.info(f"==> RESUMING chunk {chunk_identity} from raw index {idx} <==")
            return idx
    except Exception:
        logging.info(f"No previous state found for chunk {chunk_identity}. Starting from 0.")
        return -1


# ─── streaming ────────────────────────────────────────────────────────────────
async def stream_documents(dataset_name, hf_token, n_candidates, resume_raw_index=-1, data_files=None):
    queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
    loop = asyncio.get_running_loop()

    def _fill():
        logging.info(f"Streaming {dataset_name} (scanning up to {n_candidates} rows)...")
        load_kwargs = {"path": dataset_name, "split": "train", "streaming": True, "token": hf_token}
        if data_files:
            load_kwargs["data_files"] = data_files
        ds = load_dataset(**load_kwargs)

        start_idx = 0
        if resume_raw_index >= 0:
            logging.info(f"Fast-forwarding past {resume_raw_index} rows...")
            ds = ds.skip(resume_raw_index + 1)
            start_idx = resume_raw_index + 1

        try:
            for raw_index, row in enumerate(ds, start=start_idx):
                if raw_index >= n_candidates:
                    break
                asyncio.run_coroutine_threadsafe(queue.put((raw_index, row)), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    fill_task = asyncio.create_task(asyncio.to_thread(_fill))
    while True:
        item = await queue.get()
        if item is None:
            break
        yield item
    await fill_task


# ─── generation ───────────────────────────────────────────────────────────────
async def generate_row(client, tokenizer, semaphore, row, use_judge, stats):
    async with semaphore:
        question = row.get("query") or row.get("question", "")
        pos_doc = (row.get("pos_doc", "") or "").replace("<doc_seperator>", " ")
        row = {**row, "pos_doc": pos_doc}
        ground_truth = row.get("answer", "") or ""

        try:
            generated_answer, think = await client.async_chat(
                prompt=ANSWER_PROMPT.format(pos_doc=pos_doc, question=question),
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
        generated_answer = (generated_answer or "").strip()

        if len(tokenizer.encode(think + generated_answer)) > MAX_THINK_ANS_TOKENS:
            stats["token_filtered"] += 1
            return None

        if use_judge:
            if not ground_truth:
                stats["judge_skipped_no_gt"] += 1
                return None
            try:
                verdict_text, _ = await client.async_chat(
                    prompt=JUDGE_PROMPT.format(
                        question=question, answer=ground_truth, generated_answer=generated_answer
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
            m = re.search(r"<verdict>(YES|NO)</verdict>", verdict_text, re.IGNORECASE)
            if not m or m.group(1).upper() != "YES":
                stats["judge_rejected"] += 1
                return None

        stats["accepted"] += 1
        return {
            **row,
            "think":            think,
            "generated_answer": generated_answer,
        }


async def process_row(raw_index, row, client, tokenizer, sem, buffer, use_judge, stats):
    result = await generate_row(client, tokenizer, sem, row, use_judge, stats)
    if result is not None:
        await buffer.add(result, raw_index)


# ─── main ─────────────────────────────────────────────────────────────────────
async def amain(args):
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    os.makedirs(args.local_dir, exist_ok=True)
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=args.output_repo, repo_type="dataset", private=args.private, exist_ok=True)
    logging.info(f"Repository {args.output_repo} ready.")

    buffer = None
    try:
        if not VLLMInference.is_server_ready(args.base_url):
            logging.info(f"Starting {args.model} on devices {args.tpu_devices}...")
            VLLMInference.start_server(
                model=args.model,
                base_url=args.base_url,
                tensor_parallel_size=args.tp_size,
                tpu_visible_devices=args.tpu_devices,
                max_num_seqs=args.max_num_seqs,
                max_model_len=args.max_model_len,
                download_dir="/dev/shm",
            )
            await asyncio.to_thread(VLLMInference.wait_for_server, args.base_url, timeout=1200)

        client = VLLMInference(model=args.model, base_url=args.base_url)
        sem = asyncio.Semaphore(args.concurrency)

        logging.info("Loading tokenizer for token counting...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        chunk_identity = "default"
        if args.parquet_numbers:
            chunk_identity = args.parquet_numbers.replace(",", "-")

        resume_idx = fetch_resume_index(args.output_repo, chunk_identity, hf_token)

        buffer = ResultBuffer(
            local_dir=args.local_dir,
            parquet_rows=args.parquet_rows,
            upload_threshold=int(args.upload_threshold_gb * 1024**3),
            api=api,
            repo=args.output_repo,
            chunk_identity=chunk_identity,
            skip_upload=False,
        )

        stats = {k: 0 for k in (
            "total", "accepted", "token_filtered", "judge_rejected",
            "gen_error", "judge_error", "input_too_long", "judge_skipped_no_gt",
        )}
        pending_tasks = set()

        data_files = None
        if args.parquet_numbers:
            parquet_nums = [int(p.strip()) for p in args.parquet_numbers.split(",")]
            data_files = [args.parquet_template.format(p=p) for p in parquet_nums]

        async for raw_index, row in stream_documents(
            args.input_repo, hf_token,
            n_candidates=args.n_candidates,
            resume_raw_index=resume_idx,
            data_files=data_files,
        ):
            stats["total"] += 1
            task = asyncio.create_task(
                process_row(raw_index, row, client, tokenizer, sem, buffer, args.use_judge, stats)
            )
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)

            if len(pending_tasks) >= args.concurrency:
                _, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)

            if stats["total"] % 500 == 0:
                logging.info(
                    f"Dispatched {stats['total']} | Accepted={stats['accepted']} | "
                    f"TokenFilt={stats['token_filtered']} | JudgeRej={stats['judge_rejected']} | "
                    f"GenErr={stats['gen_error']} | Pending={len(pending_tasks)}"
                )

        if pending_tasks:
            logging.info(f"Waiting for remaining {len(pending_tasks)} tasks...")
            try:
                await asyncio.wait_for(asyncio.gather(*pending_tasks), timeout=300)
            except asyncio.TimeoutError:
                logging.warning("Timed out waiting for tasks.")

        logging.info(
            f"Pipeline complete. Total={stats['total']} | Accepted={stats['accepted']} | "
            f"TokenFilt={stats['token_filtered']} | JudgeRej={stats['judge_rejected']} | "
            f"InputTooLong={stats['input_too_long']} | GenErr={stats['gen_error']}"
        )

    finally:
        if buffer is not None:
            logging.info("Flushing and uploading remaining data...")
            await buffer.flush_and_upload_final()
            logging.info(f"Final total uploaded: {buffer.total_rows}")


def build_output_repo(hf_username, input_repo, model_name, parquet_str=None):
    dataset_name = input_repo.split("/")[-1]
    if "32B" in model_name:
        model_suffix = "32B"
    elif "4B" in model_name:
        model_suffix = "4B"
    else:
        model_suffix = "SFT"
    repo = f"{hf_username}/{dataset_name}-SFT-{model_suffix}"
    if parquet_str:
        x = [p.strip() for p in parquet_str.split(",")]
        repo += f"-parts-{x[0]}-{x[-1]}"
    return repo


def main():
    parser = argparse.ArgumentParser(description="Streaming SFT+CoT generation (Qwen thinking-mode)")
    parser.add_argument("--input-repo",          default=INPUT_REPO)
    parser.add_argument("--output-repo",         default=None)
    parser.add_argument("--hf-username",         default=HF_USERNAME)
    parser.add_argument("--local-dir",           default=LOCAL_DIR)
    parser.add_argument("--private",             action="store_true")
    parser.add_argument("--upload-threshold-gb", type=float, default=UPLOAD_THRESHOLD_GB)
    parser.add_argument("--parquet-rows",        type=int, default=PARQUET_ROWS)
    parser.add_argument("--n-candidates",        type=int, default=N_CANDIDATES)
    parser.add_argument("--model",               default=MODEL)
    parser.add_argument("--base-url",            default=BASE_URL)
    parser.add_argument("--tp-size",             type=int, default=TP_SIZE)
    parser.add_argument("--tpu-devices",         default=TPU_DEVICES)
    parser.add_argument("--concurrency",         type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--max-num-seqs",        type=int, default=MAX_NUM_SEQS)
    parser.add_argument("--max-model-len",       type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--parquet-numbers",     type=str, default=None,
                        help="Comma-separated parquet indices; maps via --parquet-template to data_files")
    parser.add_argument("--parquet-template",    default="data/train-{p:05d}-of-00129.parquet",
                        help="Python format string taking {p} to form HF data_files filenames")
    parser.add_argument("--use-judge",           action="store_true", default=False,
                        help="Apply LLM-judge filter against ground-truth answer (requires answer field)")
    parser.add_argument("--debug",               action="store_true")

    args = parser.parse_args()

    if args.debug:
        args.n_candidates = 2000
        logging.info("Debug mode: processing 2000 rows.")

    if args.output_repo is None:
        args.output_repo = build_output_repo(args.hf_username, args.input_repo, args.model, args.parquet_numbers)
        logging.info(f"Auto-generated output repo: {args.output_repo}")

    try:
        asyncio.run(amain(args))
        logging.info("Exiting pipeline successfully (Code 0).")
        sys.exit(0)
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Exiting...")
        sys.exit(130)
    except Exception as e:
        logging.critical(f"Pipeline failed globally: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
