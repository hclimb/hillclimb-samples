#!/usr/bin/env python3
"""
Answer-regeneration pipeline.

Streams rows from a HuggingFace dataset and regenerates answers
for all questions using a 1.7B model. The vLLM server runs
on 8 TPU chips for maximum throughput.

Usage:
    python datagen/answer_all_questions.py \
        --input-repo vm2825/nemotron-cc-v2-Parsed-DQA1 \
        --hf-username vm2825 \
        [--private]
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
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm.asyncio import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from vllm_inference import VLLMInference

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ─── defaults ─────────────────────────────────────────────────────────────────
ANSWER_MODEL           = "Qwen/Qwen3-1.7B"
ANSWER_BASE_URL        = "http://localhost:8001/v1"
ANSWER_TP_SIZE         = 8
ANSWER_TPU_DEVICES     = "0,1,2,3,4,5,6,7"

INPUT_REPO             = "vm2825/nemotron-cc-v2-Parsed-DQA1"
HF_USERNAME            = "vm2825"

N_CANDIDATES           = 30000000
MIN_WORDS              = 128

MAX_CONCURRENCY_ANSWER = 2048
ANSWER_MAX_NUM_SEQS    = 2048

PARQUET_ROWS           = 500000
UPLOAD_THRESHOLD_BYTES = 1 * 1024 * 1024 * 1024  # 10 GB
LOCAL_DIR              = "./temp_qa_upload"

SCHEMA = pa.schema([
    ("id",               pa.string()),
    ("pos_doc",          pa.string()),
    ("question",         pa.string()),
    ("original_answer",  pa.string()),
    ("think",            pa.string()),
    ("synthetic_answer", pa.string()),
])

# ─── prompts ──────────────────────────────────────────────────────────────────
ANSWER_SYSTEM = "You are a helpful assistant."

ANSWER_PROMPT = """Given the document and question below, provide a very very concise but accurate answer.
Do not reference the document directly in your answer.

Document:
{pos_doc}

Question:
{question}"""


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
        allow_patterns="*.parquet",
        commit_message=f"Upload batch {upload_count}",
    )
    logging.info("Upload successful. Clearing local files...")
    for f in os.listdir(local_dir):
        if f.endswith(".parquet"):
            os.remove(os.path.join(local_dir, f))


# ─── result buffer ────────────────────────────────────────────────────────────
class ResultBuffer:
    """
    Accumulates Q&A records, flushes to parquet at PARQUET_ROWS, and
    triggers a background upload to HF when local dir hits UPLOAD_THRESHOLD_BYTES.
    """

    def __init__(self, local_dir: str, parquet_rows: int, upload_threshold: int,
                 api: HfApi, repo: str):
        self.local_dir = local_dir
        self.parquet_rows = parquet_rows
        self.upload_threshold = upload_threshold
        self.api = api
        self.repo = repo

        self.records: list[dict] = []
        self.chunk_index = 0
        self.upload_count = 0
        self.total_rows = 0
        self.is_uploading = False
        self._lock = asyncio.Lock()

    async def add(self, record: dict):
        async with self._lock:
            self.records.append(record)
            if len(self.records) >= self.parquet_rows:
                self._flush_locked()

        if not self.is_uploading and get_dir_size(self.local_dir) >= self.upload_threshold:
            self.is_uploading = True
            asyncio.create_task(self._upload_task())

    def _flush_locked(self):
        table = pa.Table.from_pylist(self.records, schema=SCHEMA)
        path = os.path.join(self.local_dir, f"data_{self.chunk_index:06d}.parquet")
        pq.write_table(table, path)
        self.total_rows += len(table)
        logging.info(f"Wrote {len(table)} rows → {path}  (total: {self.total_rows})")
        self.records = []
        self.chunk_index += 1

    async def _upload_task(self):
        try:
            self.upload_count += 1
            await asyncio.to_thread(
                _upload_and_clear_sync, self.api, self.local_dir, self.repo, self.upload_count
            )
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

        if get_dir_size(self.local_dir) > 0:
            self.upload_count += 1
            await asyncio.to_thread(
                _upload_and_clear_sync, self.api, self.local_dir, self.repo, self.upload_count
            )


# ─── dataset streaming ────────────────────────────────────────────────────────
async def stream_documents(
    dataset_name: str,
    n_candidates: int,
    min_words: int,
    hf_token: str,
    max_docs: int | None = None,
    data_files: list[str] | None = None,
):
    """Async generator yielding rows from a HF dataset."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
    loop = asyncio.get_running_loop()

    def _fill():
        logging.info(f"Streaming {dataset_name} (scanning up to {n_candidates} rows)...")
        load_kwargs = {"path": dataset_name, "split": "train", "streaming": True, "token": hf_token}
        if data_files:
            load_kwargs["data_files"] = data_files
        ds = load_dataset(**load_kwargs)
        count = 0
        try:
            for i, row in enumerate(ds):
                if i >= n_candidates:
                    break
                pos_doc = row.get("pos_doc", "") or ""
                # pos_doc may be a list — join if so
                if isinstance(pos_doc, list):
                    pos_doc = "\n\n".join(pos_doc)
                if len(pos_doc.split()) > min_words:
                    # Carry the original answer forward
                    row["pos_doc"] = pos_doc
                    asyncio.run_coroutine_threadsafe(queue.put(row), loop).result()
                    count += 1
                    if max_docs is not None and count >= max_docs:
                        break
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()
        logging.info(f"Streamed {count} documents passing word limit constraint.")

    fill_task = asyncio.create_task(asyncio.to_thread(_fill))
    while True:
        item = await queue.get()
        if item is None:
            break
        yield item
    await fill_task


# ─── pipeline coroutines ──────────────────────────────────────────────────────
async def generate_answer(
    client: VLLMInference,
    semaphore: asyncio.Semaphore,
    row: dict,
) -> dict:
    """Generate a synthetic answer for the question."""
    async with semaphore:
        try:
            content, think = await client.async_chat(
                prompt=ANSWER_PROMPT.format(
                    pos_doc=row.get("pos_doc", ""),
                    question=row.get("question", ""),
                ),
                system=ANSWER_SYSTEM,
                max_tokens=1024,
                temperature=0.7,
                thinking=True,
            )
            return {
                "id": str(row.get("id", "")),
                "pos_doc": row.get("pos_doc", ""),
                "question": row.get("question", ""),
                "original_answer": row.get("answer", ""),
                "think": think,
                "synthetic_answer": content or "",
            }
        except Exception as e:
            logging.error(f"[answer] id={row.get('id')}: {e}")
            return {
                "id": str(row.get("id", "")),
                "pos_doc": row.get("pos_doc", ""),
                "question": row.get("question", ""),
                "original_answer": row.get("answer", ""),
                "think": None,
                "synthetic_answer": None,
            }


async def process_row(
    row: dict,
    answer_client: VLLMInference,
    answer_sem: asyncio.Semaphore,
    buffer: ResultBuffer,
    stats: dict,
):
    """Full pipeline for one row: answer → buffer."""
    result = await generate_answer(answer_client, answer_sem, row)
    
    # Check if we failed parsing or hit context limit
    ans = result.get("synthetic_answer")
    if ans is not None:
        if "<think>" in ans or "</think>" in ans:
            logging.warning(f"[answer] id={row.get('id')}: Found unparsed <think> tags in answer, skipping. Length: {len(ans.split())} words")
            stats["dropped"] += 1
        else:
            await buffer.add(result)
            stats["answered"] += 1
    else:
        stats["dropped"] += 1


# ─── main ─────────────────────────────────────────────────────────────────────
async def amain(args):
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    os.makedirs(args.local_dir, exist_ok=True)

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=args.output_repo, repo_type="dataset", private=args.private, exist_ok=True)
    logging.info(f"Repository {args.output_repo} ready.")

    answer_proc = None

    try:
        if not VLLMInference.is_server_ready(args.answer_base_url):
            logging.info(f"Starting answer model {args.answer_model} on devices {args.answer_tpu_devices}...")
            answer_proc = VLLMInference.start_server(
                model=args.answer_model, base_url=args.answer_base_url,
                tensor_parallel_size=args.answer_tp,
                tpu_visible_devices=args.answer_tpu_devices,
                max_num_seqs=args.answer_max_num_seqs,
            )
            logging.info("Waiting for answer server...")
            await asyncio.to_thread(VLLMInference.wait_for_server, args.answer_base_url, timeout=600)
            logging.info("Answer server ready.")
        else:
            logging.info("Answer server already running.")

        # ── Pipeline ──────────────────────────────────────────────────────────
        answer_client = VLLMInference(model=args.answer_model, base_url=args.answer_base_url)
        answer_sem = asyncio.Semaphore(args.answer_concurrency)

        buffer = ResultBuffer(
            local_dir=args.local_dir,
            parquet_rows=args.parquet_rows,
            upload_threshold=int(args.upload_threshold_gb * 1024 ** 3),
            api=api,
            repo=args.output_repo,
        )

        stats = {"total": 0, "answered": 0, "dropped": 0}
        tasks: list[asyncio.Task] = []

        data_files = None
        if getattr(args, "parquet_numbers", None):
            parquet_nums = [int(p.strip()) for p in args.parquet_numbers.split(",")]
            data_files = [f"nemotron_dqa_{p:06d}.parquet" for p in parquet_nums]

        async for row in stream_documents(
            args.input_repo, args.n_candidates, args.min_words,
            hf_token, data_files=data_files
        ):
            stats["total"] += 1
            task = asyncio.create_task(
                process_row(row, answer_client, answer_sem, buffer, stats)
            )
            tasks.append(task)

            # Log progress periodically
            if stats["total"] % 500 == 0:
                logging.info(f"Dispatched {stats['total']} | Answered={stats['answered']} | Dropped={stats['dropped']}")

        # Wait for all in-flight tasks
        logging.info(f"Waiting for {len(tasks)} pipeline tasks to complete...")
        await tqdm.gather(*tasks, total=len(tasks))

        logging.info(
            f"Pipeline complete. "
            f"Total Processed={stats['total']} | Successful={stats['answered']} | Dropped={stats['dropped']}"
        )

        # ── Final flush + upload ──────────────────────────────────────────────
        logging.info("Flushing and uploading remaining data...")
        await buffer.flush_and_upload_final()
        logging.info(f"Done. Total rows uploaded: {buffer.total_rows}")

    finally:
        if answer_proc is not None:
            answer_proc.terminate()
            logging.info("Answer server terminated.")


def build_output_repo(hf_username: str, input_repo: str, parquet_str: str | None = None) -> str:
    dataset_name = input_repo.split("/")[-1]
    repo = f"{hf_username}/{dataset_name}-answered-1.7B"
    if parquet_str:
        # e.g. "5,6,10" -> "parts-5-6-10"
        parts = "-".join([p.strip() for p in parquet_str.split(",")])
        repo += f"-parts-{parts}"
    return repo


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate answers for dataset, streaming upload to HuggingFace"
    )

    # Input / output repos
    parser.add_argument("--input-repo",    default=INPUT_REPO,    help="HF dataset repo to load documents from")
    parser.add_argument("--output-repo",   default=None,          help="HF dataset repo to write to (default: auto)")
    parser.add_argument("--hf-username",   default=HF_USERNAME,   help="HF username for auto-generated output repo")

    # Upload settings
    parser.add_argument("--local-dir",             default=LOCAL_DIR,  help="Temp dir for parquet files")
    parser.add_argument("--private",               action="store_true", help="Create private HF repo")
    parser.add_argument("--upload-threshold-gb",   type=float, default=20.0)
    parser.add_argument("--parquet-rows",          type=int, default=PARQUET_ROWS)
    parser.add_argument("--debug",                  action="store_true", help="Debug mode: process only 1000 rows, skip HF upload")

    # Document filtering
    parser.add_argument("--n-candidates",   type=int, default=N_CANDIDATES)
    parser.add_argument("--min-words",      type=int, default=MIN_WORDS)

    # Answer model
    parser.add_argument("--answer-model",       default=ANSWER_MODEL)
    parser.add_argument("--answer-base-url",    default=ANSWER_BASE_URL)
    parser.add_argument("--answer-tp",          type=int, default=ANSWER_TP_SIZE)
    parser.add_argument("--answer-tpu-devices", default=ANSWER_TPU_DEVICES)
    parser.add_argument("--answer-concurrency", type=int, default=MAX_CONCURRENCY_ANSWER)
    parser.add_argument("--answer-max-num-seqs", type=int, default=ANSWER_MAX_NUM_SEQS)
    parser.add_argument("--parquet-numbers",    type=str, default=None, help="Comma-separated integers, e.g. '5,6,10' to process train/train_{N:06d}.parquet")

    args = parser.parse_args()

    if args.debug:
        args.n_candidates = 2000
        logging.info("Debug mode: processing 2000 rows, skipping HF upload.")

    if args.output_repo is None:
        if not args.hf_username:
            parser.error("--hf-username is required when --output-repo is not set")
        args.output_repo = build_output_repo(args.hf_username, args.input_repo, args.parquet_numbers)
        logging.info(f"Auto-generated output repo: {args.output_repo}")

    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
