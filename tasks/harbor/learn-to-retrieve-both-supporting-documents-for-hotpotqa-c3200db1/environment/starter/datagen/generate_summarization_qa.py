#!/usr/bin/env python3
"""
Integrated Q&A generation pipeline with streaming HuggingFace upload.

Streaming pipeline: both query and answer vLLM servers run simultaneously on
separate TPU chip subsets. Documents are dispatched immediately as async tasks,
with query generation chained directly into answer generation.

Usage:
    python datagen/generate_qa.py \
        --input-repo vm2825/nemotron-cc-v21-Parsed-QA4 \
        --hf-username vm2825 \
        [--answer-type SyntheticQA] \
        [--n-candidates 500000] \
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
QUERY_MODEL            = "Qwen/Qwen3-32B"
QUERY_BASE_URL         = "http://localhost:8000/v1"
QUERY_TP_SIZE          = 4
QUERY_TPU_DEVICES      = "0,1,2,3"
QUERY_MAX_NUM_SEQS     = 2048

ANSWER_MODEL           = "Qwen/Qwen3-1.7B"
ANSWER_BASE_URL        = "http://localhost:8001/v1"
ANSWER_TP_SIZE         = 4
ANSWER_TPU_DEVICES     = "4,5,6,7"
ANSWER_MAX_NUM_SEQS    = 2048

INPUT_REPO             = "vm2825/nemotron-cc-v21-Parsed-QA4"
ANSWER_TYPE            = "Summarization"
DATASET_NAME           = "nemotron-cc-v21-Parsed-QA4"
HF_USERNAME            = "ragrawal36"

INPUT_PARQUET_NUMBERS  = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"

N_CANDIDATES           = 10_000_000
MIN_WORDS              = 384
MAX_WORDS              = 1024
INPUT_INTERVAL         = 5

MAX_CONCURRENCY_QUERY  = 1000000
MAX_CONCURRENCY_ANSWER = 1000000

PARQUET_ROWS           = 1000
UPLOAD_THRESHOLD_BYTES = 20 * 1024  # 20 MB
LOCAL_DIR              = "./temp_qa_upload"

SCHEMA = pa.schema([
    ("id",               pa.string()),
    ("pos_doc",          pa.string()),
    ("question",         pa.string()),
    ("think",            pa.string()),
    ("synthetic_answer", pa.string()),
])

# ─── prompts ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = "You are a helpful assistant."

QUERY_PROMPT = """You are an expert test writer, capable of producing challenging comprehension style question-answer pairs based on a provided document.

Your question will be given to a student that should have memorized many documents, including the one provided.

Your goal is to write a question that tests the student's ability to explain the main topic or content of the document.

Since the student needs to recall this specific document out of many other documents, the question should not reference the document directly, but rather allude to summarizing the provided document while pretending it isn't provided.

For example, the question should not include phrases like "the document states..." or "this work says...".

The question should aim to be as comprehensive as possible, while providing the minimal information possible that allows the student to recall that document.

Output only the question between <question></question> tags.

Document:
{pos_doc}"""

ANSWER_PROMPT = """You are an expert at answering comprehension questions about documents.

Given the document and question below, provide a concise but accurate answer. Do not reference the document directly in your answer.

Document:
{pos_doc}

Question:
{question}"""


# ─── helpers ──────────────────────────────────────────────────────────────────
def extract_tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


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
    Generation is never blocked by an upload in progress.
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

        # Check upload threshold outside the lock (asyncio is single-threaded,
        # so no race between awaits)
        if not self.is_uploading and get_dir_size(self.local_dir) >= self.upload_threshold:
            self.is_uploading = True
            asyncio.create_task(self._upload_task())

    def _flush_locked(self):
        """Write buffered records to a new parquet file. Call under self._lock."""
        table = pa.Table.from_pylist(self.records, schema=SCHEMA)
        path = os.path.join(self.local_dir, f"data_{self.chunk_index:06d}.parquet")
        pq.write_table(table, path)
        self.total_rows += len(table)
        logging.info(f"Wrote {len(table)} rows → {path}  (total: {self.total_rows})")
        self.records = []
        self.chunk_index += 1

    async def _upload_task(self):
        """Run upload in a thread so generation keeps going."""
        try:
            self.upload_count += 1
            await asyncio.to_thread(
                _upload_and_clear_sync, self.api, self.local_dir, self.repo, self.upload_count
            )
        except Exception as e:
            logging.error(f"Upload failed: {e}. Local files kept for retry.")
        finally:
            self.is_uploading = False
            # If more files piled up while uploading, trigger another round
            if get_dir_size(self.local_dir) >= self.upload_threshold:
                self.is_uploading = True
                asyncio.create_task(self._upload_task())

    async def flush_and_upload_final(self):
        """Flush remaining records and do a final upload. Call after all generation is done."""
        async with self._lock:
            if self.records:
                self._flush_locked()

        # Wait for any in-progress background upload to finish
        while self.is_uploading:
            await asyncio.sleep(5)

        # Final upload of whatever remains (may be below threshold — upload it anyway)
        if get_dir_size(self.local_dir) > 0:
            self.upload_count += 1
            await asyncio.to_thread(
                _upload_and_clear_sync, self.api, self.local_dir, self.repo, self.upload_count
            )


# ─── generation coroutines ────────────────────────────────────────────────────
async def generate_query(
    client: VLLMInference,
    semaphore: asyncio.Semaphore,
    row: dict,
) -> dict | None:
    async with semaphore:
        try:
            content, _ = await client.async_chat(
                prompt=QUERY_PROMPT.format(pos_doc=row.get("pos_doc", "")),
                system=SYSTEM_PROMPT,
                max_completion_tokens=2048,
                temperature=0.7,
                thinking=False,
            )
            question = extract_tag(content or "", "question")
            if not question:
                # logging.warning(f"Missing <question> tag for id={row.get('id')}")
                return None
            return {"id": row.get("id"), "pos_doc": row.get("pos_doc", ""), "question": question}
        except Exception as e:
            logging.error(f"[query] id={row.get('id')}: {e}")
            return None


async def generate_answer(
    client: VLLMInference,
    semaphore: asyncio.Semaphore,
    entry: dict,
) -> dict:
    async with semaphore:
        try:
            content, think = await client.async_chat(
                prompt=ANSWER_PROMPT.format(pos_doc=entry["pos_doc"], question=entry["question"]),
                system=SYSTEM_PROMPT,
                max_completion_tokens=2048,
                temperature=0.7,
                thinking=True,
            )
            return {**entry, "think": think, "synthetic_answer": content or ""}
        except Exception as e:
            logging.error(f"[answer] id={entry.get('id')}: {e}")
            return {**entry, "think": None, "synthetic_answer": None}


async def process_row(
    row: dict,
    query_client: VLLMInference,
    query_sem: asyncio.Semaphore,
    answer_client: VLLMInference,
    answer_sem: asyncio.Semaphore,
    buffer: ResultBuffer,
    stats: dict,
):
    """Full pipeline for one row: generate_query → generate_answer → buffer."""
    entry = await generate_query(query_client, query_sem, row)
    if entry is None:
        stats["failed"] += 1
        return

    result = await generate_answer(answer_client, answer_sem, entry)
    if result.get("synthetic_answer") is not None:
        await buffer.add(result)
        stats["generated"] += 1
    else:
        stats["failed"] += 1


# ─── dataset streaming ────────────────────────────────────────────────────────
async def stream_documents(
    dataset_name: str,
    n_candidates: int,
    min_words: int,
    max_words: int,
    input_interval: int,
    hf_token: str,
    max_docs: int | None = None,
    data_files: list[str] | None = None,
):
    """Async generator yielding filtered rows from a HF dataset one at a time.

    Runs HF dataset iteration in a background thread and feeds rows into an
    async queue, so the event loop is never blocked and memory usage stays
    bounded (at most 256 rows buffered between producer and consumer).
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
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
                if i % input_interval != 0:
                    continue
                if i >= n_candidates:
                    break
                pos_doc = row.get("pos_doc", "") or ""
                if isinstance(pos_doc, list):
                    pos_doc = "\n\n".join(pos_doc)
                word_count = len(pos_doc.split())
                if word_count > min_words and word_count < max_words:
                    row["pos_doc"] = pos_doc
                    asyncio.run_coroutine_threadsafe(queue.put(row), loop).result()
                    count += 1
                    if max_docs is not None and count >= max_docs:
                        break
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()
        logging.info(f"Streamed {count} documents passing filter.")

    fill_task = asyncio.create_task(asyncio.to_thread(_fill))
    while True:
        item = await queue.get()
        if item is None:
            break
        yield item
    await fill_task


# ─── main ─────────────────────────────────────────────────────────────────────
async def amain(args):
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    os.makedirs(args.local_dir, exist_ok=True)

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=args.output_repo, repo_type="dataset", private=args.private, exist_ok=True)
    logging.info(f"Repository {args.output_repo} ready.")

    query_proc = None
    answer_proc = None

    try:
        # ── Boot servers sequentially to avoid libtpu lockfile race ───────────
        if not VLLMInference.is_server_ready(args.query_base_url):
            logging.info(f"Starting query model {args.query_model} on devices {args.query_tpu_devices}...")
            query_proc = VLLMInference.start_server(
                model=args.query_model, base_url=args.query_base_url,
                tensor_parallel_size=args.query_tp,
                tpu_visible_devices=args.query_tpu_devices,
                max_num_seqs=args.query_max_num_seqs,
            )
            logging.info("Waiting for query server to be ready before starting answer server...")
            await asyncio.to_thread(VLLMInference.wait_for_server, args.query_base_url)
            logging.info("Query server ready.")
        else:
            logging.info("Query server already running.")

        if not VLLMInference.is_server_ready(args.answer_base_url):
            logging.info(f"Starting answer model {args.answer_model} on devices {args.answer_tpu_devices}...")
            answer_proc = VLLMInference.start_server(
                model=args.answer_model, base_url=args.answer_base_url,
                tensor_parallel_size=args.answer_tp,
                tpu_visible_devices=args.answer_tpu_devices,
                max_num_seqs=args.answer_max_num_seqs,
            )
            logging.info("Waiting for answer server...")
            await asyncio.to_thread(VLLMInference.wait_for_server, args.answer_base_url)
            logging.info("Answer server ready.")
        else:
            logging.info("Answer server already running.")

        logging.info("Both servers ready.")

        # ── Pipeline ──────────────────────────────────────────────────────────
        query_client = VLLMInference(model=args.query_model, base_url=args.query_base_url)
        answer_client = VLLMInference(model=args.answer_model, base_url=args.answer_base_url)
        query_sem = asyncio.Semaphore(args.query_concurrency)
        answer_sem = asyncio.Semaphore(args.answer_concurrency)

        buffer = ResultBuffer(
            local_dir=args.local_dir,
            parquet_rows=args.parquet_rows,
            upload_threshold=int(args.upload_threshold_gb * 1024 ** 3),
            api=api,
            repo=args.output_repo,
        )

        stats = {"generated": 0, "failed": 0, "total": 0}
        tasks: list[asyncio.Task] = []

        data_files = None
        if getattr(args, "parquet_numbers", None):
            parquet_nums = [int(p.strip()) for p in args.parquet_numbers.split(",")]
            data_files = [f"train/train_{p:06d}.parquet" for p in parquet_nums]

        async for row in stream_documents(
            args.input_repo, args.n_candidates, args.min_words, args.max_words,
            args.input_interval, hf_token, data_files=data_files,
        ):
            stats["total"] += 1
            task = asyncio.create_task(
                process_row(row, query_client, query_sem, answer_client, answer_sem, buffer, stats)
            )
            tasks.append(task)

            if stats["total"] % 500 == 0:
                logging.info(
                    f"Dispatched {stats['total']} | "
                    f"generated={stats['generated']} | failed={stats['failed']}"
                )

        logging.info(f"Waiting for {len(tasks)} pipeline tasks to complete...")
        await tqdm.gather(*tasks, total=len(tasks))

        logging.info(
            f"Pipeline complete. "
            f"Total={stats['total']} | Generated={stats['generated']} | "
            f"Failed={stats['failed']} | "
            f"Success rate={stats['generated']/max(stats['total'],1)*100:.1f}%"
        )

        # ── Final flush + upload ──────────────────────────────────────────────
        logging.info("Flushing and uploading remaining data...")
        await buffer.flush_and_upload_final()
        logging.info(f"Done. Total rows uploaded: {buffer.total_rows}")

    finally:
        if query_proc is not None:
            query_proc.terminate()
            logging.info("Query server terminated.")
        if answer_proc is not None:
            answer_proc.terminate()
            logging.info("Answer server terminated.")


def build_output_repo(hf_username: str, input_repo: str, answer_type: str, answer_model: str) -> str:
    dataset_name = input_repo.split("/")[-1]
    model_name = answer_model.split("/")[-1]
    return f"{hf_username}/{dataset_name}-{answer_type}-{model_name}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate Q&A pairs from a HF dataset and stream-upload to HuggingFace"
    )

    # Input / output repos
    parser.add_argument("--input-repo",    default=INPUT_REPO,              help="HF dataset repo to load documents from")
    parser.add_argument("--output-repo",   default=None,                    help="HF dataset repo to write parquet to (default: auto-generated)")
    parser.add_argument("--hf-username",   default=HF_USERNAME,             help="HuggingFace username for auto-generated output repo")
    parser.add_argument("--answer-type",   default=ANSWER_TYPE,             help="Label used in the output repo name, e.g. SyntheticQA")

    # Upload settings
    parser.add_argument("--local-dir",             default=LOCAL_DIR,        help="Temp dir for parquet files")
    parser.add_argument("--private",               action="store_true",      help="Create private HF repo")
    parser.add_argument("--upload-threshold-gb",   type=float, default=(UPLOAD_THRESHOLD_BYTES / (1024 ** 3)), help="Upload when local dir exceeds this many GB")
    parser.add_argument("--parquet-rows",          type=int, default=PARQUET_ROWS, help="Max rows per parquet file")
    parser.add_argument("--debug",                 action="store_true",      help="Debug mode: process only 2000 rows")

    # Document filtering
    parser.add_argument("--n-candidates",   type=int, default=N_CANDIDATES*INPUT_INTERVAL,   help="Max dataset rows to scan")
    parser.add_argument("--min-words",      type=int, default=MIN_WORDS,       help="Min words in pos_doc to include")
    parser.add_argument("--max-words",      type=int, default=MAX_WORDS,       help="Max words in pos_doc to include")
    parser.add_argument("--input-interval", type=int, default=INPUT_INTERVAL,  help="Sample every Nth row from dataset")
    parser.add_argument("--parquet-numbers", type=str, default=INPUT_PARQUET_NUMBERS,           help="Comma-separated integers, e.g. '5,6,10' to process train/train_{N:06d}.parquet")

    # Query model
    parser.add_argument("--query-model",         default=QUERY_MODEL)
    parser.add_argument("--query-base-url",      default=QUERY_BASE_URL)
    parser.add_argument("--query-tp",            type=int, default=QUERY_TP_SIZE)
    parser.add_argument("--query-tpu-devices",   default=QUERY_TPU_DEVICES)
    parser.add_argument("--query-concurrency",   type=int, default=MAX_CONCURRENCY_QUERY)
    parser.add_argument("--query-max-num-seqs",  type=int, default=QUERY_MAX_NUM_SEQS)

    # Answer model
    parser.add_argument("--answer-model",        default=ANSWER_MODEL)
    parser.add_argument("--answer-base-url",     default=ANSWER_BASE_URL)
    parser.add_argument("--answer-tp",           type=int, default=ANSWER_TP_SIZE)
    parser.add_argument("--answer-tpu-devices",  default=ANSWER_TPU_DEVICES)
    parser.add_argument("--answer-concurrency",  type=int, default=MAX_CONCURRENCY_ANSWER)
    parser.add_argument("--answer-max-num-seqs", type=int, default=ANSWER_MAX_NUM_SEQS)

    args = parser.parse_args()

    if args.debug:
        args.n_candidates = 2000
        args.input_interval = 1
        logging.info("Debug mode: processing 2000 rows.")

    if args.output_repo is None:
        if not args.hf_username:
            parser.error("--hf-username is required when --output-repo is not set")
        args.output_repo = build_output_repo(args.hf_username, args.input_repo, args.answer_type, args.answer_model)
        logging.info(f"Auto-generated output repo: {args.output_repo}")

    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
