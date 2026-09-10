#!/usr/bin/env python3
"""
Question-rewriting pipeline.

Streams rows from a HuggingFace dataset and rewrites generic questions
into highly specific search queries using a vLLM server.
"""

import asyncio
import argparse
import logging
import os
import re
import sys
import uuid
import json

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
REWRITE_MODEL          = "Qwen/Qwen3-8B"
REWRITE_BASE_URL       = "http://localhost:8001/v1"
REWRITE_TP_SIZE        = 8
REWRITE_TPU_DEVICES    = "0,1,2,3,4,5,6,7"

INPUT_REPO             = "vm2825/nemotron-cc-v21-Parsed-QA4-filtered-1.7B-evensplit"
HF_USERNAME            = "vm2825"

N_CANDIDATES           = 30000000

MAX_CONCURRENCY_REWRITE = 2048
REWRITE_MAX_NUM_SEQS    = 1024

PARQUET_ROWS           = 100000
UPLOAD_THRESHOLD_GB    = 0.25
LOCAL_DIR              = "./temp_rq_upload"

SCHEMA = pa.schema([
    ("id",               pa.string()),
    ("pos_doc",          pa.string()),
    ("question",         pa.string()),
    ("original_answer",  pa.string()),
    ("think",            pa.string()),
    ("synthetic_answer", pa.string()),
    ("synthetic_question", pa.string()),
])

# ─── prompts ──────────────────────────────────────────────────────────────────
REWRITE_SYSTEM = "You are a helpful assistant."

REWRITE_PROMPT = """You are an expert in Information Retrieval (IR) and search engine optimization. Your task is to rewrite generic question-answering (QA) queries into highly specific, "grounded" search queries. 

In standard QA datasets, questions often rely on conversational context (e.g., "What type of cells are the focus of the research?"). If used as a search query in a large database, this would return millions of irrelevant results. 

Your goal is to inject specific entities, methods, or context from the target document into the query so that it becomes self-contained. The rewritten query should be specific enough to retrieve THIS EXACT document, while remaining a natural-sounding question of roughly the same length.

### INSTRUCTIONS:
1. Read the provided Document and the Original Query.
2. Identify the core intent of the question.
3. Extract 1-3 highly specific keywords, entities, or methods from the document that provide context to the generic terms in the query.
4. Rewrite the question to incorporate these specific terms.
5. DO NOT answer the question. ONLY output the rewritten question.
6. Keep the rewritten query concise and close to the original length.
7. Enclose your final rewritten query strictly within <rewritten_query> and </rewritten_query> tags.

### EXAMPLES:

Document: 
White blood cell segmentation by circle detection using electromagnetism-like optimization. Erik Cuevas, Diego Oliva... Medical imaging is a relevant field... The approach is based on a nature-inspired technique called the electromagnetism-like optimization (EMO) algorithm...

Original Query: What type of cells are the focus of the research?
Rewritten Query: <rewritten_query>What type of cells are segmented using the electromagnetism-like optimization (EMO) algorithm?</rewritten_query>

Document:
{pos_doc}

Original Query: 
{question}

Rewritten Query:"""


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


# ─── result buffer ────────────────────────────────────────────────────────────
class ResultBuffer:
    def __init__(self, local_dir: str, parquet_rows: int, upload_threshold: int,
                 api: HfApi, repo: str, chunk_identity: str, skip_upload: bool = False):
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

    async def add(self, record: dict, raw_index: int):
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
        table = pa.Table.from_pylist(self.records, schema=SCHEMA)
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

        if not self.skip_upload and get_dir_size(self.local_dir) > 0:
            self.upload_count += 1
            logging.info(f"Final upload of {get_dir_size(self.local_dir)} bytes to {self.repo}...")
            await asyncio.to_thread(
                _upload_and_clear_sync, self.api, self.local_dir, self.repo, self.upload_count
            )
        else:
            logging.info("Skipping final upload (debug mode or no data).")


def fetch_resume_index(repo: str, chunk_identity: str, hf_token: str) -> int:
    from huggingface_hub import hf_hub_download
    try:
        logging.info(f"Checking HF Repo {repo} for state_chunk_{chunk_identity}.json...")
        file_path = hf_hub_download(
            repo_id=repo,
            filename=f"state_chunk_{chunk_identity}.json",
            repo_type="dataset",
            token=hf_token
        )
        with open(file_path, "r") as f:
            data = json.load(f)
            idx = data.get("processed_raw_index", -1)
            logging.info(f"==> RESUMING chunk {chunk_identity} from raw index {idx} <==")
            return idx
    except Exception:
        logging.info(f"No previous state found for chunk {chunk_identity}. Starting from 0.")
        return -1


# ─── dataset streaming ────────────────────────────────────────────────────────
async def stream_documents(
    dataset_name: str,
    hf_token: str,
    n_candidates: int,
    resume_raw_index: int = -1,
    data_files: list[str] | None = None
):
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


# ─── pipeline coroutines ──────────────────────────────────────────────────────
async def generate_rewrite(
    client: VLLMInference,
    semaphore: asyncio.Semaphore,
    row: dict,
) -> dict:
    async with semaphore:
        try:
            content, _ = await client.async_chat(
                prompt=REWRITE_PROMPT.format(
                    pos_doc=row.get("pos_doc", ""),
                    question=row.get("question", ""),
                ),
                system=REWRITE_SYSTEM,
                max_completion_tokens=512,
                temperature=0.7,
                thinking=False,
            )
            
            # Parse <rewritten_query> tags
            match = re.search(r"<rewritten_query>(.*?)</rewritten_query>", content, re.DOTALL)
            if not match:
                logging.warning(f"[rewrite] id={row.get('id')}: No <rewritten_query> tags found, skipping.")
                return {
                    "id": str(row.get("id", "")),
                    "pos_doc": row.get("pos_doc", ""),
                    "question": row.get("question", ""),
                    "original_answer": row.get("original_answer", "") or row.get("answer", ""),
                    "think": row.get("think", ""),
                    "synthetic_answer": row.get("synthetic_answer", ""),
                    "synthetic_question": None,
                }
                
            rewritten = match.group(1).strip()
            
            return {
                "id": str(row.get("id", "")),
                "pos_doc": row.get("pos_doc", ""),
                "question": row.get("question", ""),
                "original_answer": row.get("original_answer", "") or row.get("answer", ""),
                "think": row.get("think", ""),
                "synthetic_answer": row.get("synthetic_answer", ""),
                "synthetic_question": rewritten,
            }
        except Exception as e:
            logging.error(f"[rewrite] id={row.get('id')}: {e}")
            return {
                "id": str(row.get("id", "")),
                "pos_doc": row.get("pos_doc", ""),
                "question": row.get("question", ""),
                "original_answer": row.get("original_answer", "") or row.get("answer", ""),
                "think": row.get("think", ""),
                "synthetic_answer": row.get("synthetic_answer", ""),
                "synthetic_question": None,
            }


async def process_row(
    raw_index: int,
    row: dict,
    client: VLLMInference,
    sem: asyncio.Semaphore,
    buffer: ResultBuffer,
    stats: dict,
):
    result = await generate_rewrite(client, sem, row)
    if result.get("synthetic_question"):
        await buffer.add(result, raw_index)
        stats["rewritten"] += 1
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

    try:
        if not VLLMInference.is_server_ready(args.rewrite_base_url):
            logging.info(f"Starting server on devices {args.rewrite_tpu_devices}...")
            # Note: We assume the server is managed externally or started here.
            # For simplicity, we'll try to start it if it's not ready.
            VLLMInference.start_server(
                model=args.rewrite_model, base_url=args.rewrite_base_url,
                tensor_parallel_size=args.rewrite_tp,
                tpu_visible_devices=args.rewrite_tpu_devices,
                max_num_seqs=args.rewrite_max_num_seqs,
            )
            await asyncio.to_thread(VLLMInference.wait_for_server, args.rewrite_base_url, timeout=600)
        
        client = VLLMInference(model=args.rewrite_model, base_url=args.rewrite_base_url)
        sem = asyncio.Semaphore(args.rewrite_concurrency)

        chunk_identity = "default"
        if args.parquet_numbers:
            chunk_identity = args.parquet_numbers.replace(",", "-")
            
        resume_idx = fetch_resume_index(args.output_repo, chunk_identity, hf_token)

        buffer = ResultBuffer(
            local_dir=args.local_dir,
            parquet_rows=args.parquet_rows,
            upload_threshold=int(args.upload_threshold_gb * 1024 ** 3),
            api=api,
            repo=args.output_repo,
            chunk_identity=chunk_identity,
            skip_upload=args.debug
        )

        stats = {"total": 0, "rewritten": 0, "dropped": 0}
        pending_tasks = set()

        data_files = None
        if args.parquet_numbers:
            parquet_nums = [int(p.strip()) for p in args.parquet_numbers.split(",")]
            data_files = [f"data/train-{p:05d}.parquet" for p in parquet_nums]

        async for raw_index, row in stream_documents(
            args.input_repo, hf_token, n_candidates=args.n_candidates,
            resume_raw_index=resume_idx, data_files=data_files
        ):
            stats["total"] += 1
            task = asyncio.create_task(process_row(raw_index, row, client, sem, buffer, stats))
            pending_tasks.add(task)

            if len(pending_tasks) >= args.rewrite_concurrency:
                done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)

            if stats["total"] % 500 == 0:
                logging.info(f"Dispatched {stats['total']} | Rewritten={stats['rewritten']} | Dropped={stats['dropped']} | Pending={len(pending_tasks)}")

        if pending_tasks:
            # Wait with a timeout for remaining tasks
            logging.info(f"Waiting for remaining {len(pending_tasks)} pipeline tasks to complete...")
            try:
                await asyncio.wait_for(asyncio.gather(*pending_tasks), timeout=300)
            except asyncio.TimeoutError:
                logging.warning("Timed out waiting for tasks to complete. Proceeding to shutdown.")

        logging.info(f"Pipeline complete. Total={stats['total']} | Successful={stats['rewritten']}")

    finally:
        # Final flush and upload
        logging.info("Flushing and uploading remaining data...")
        await buffer.flush_and_upload_final()
        logging.info(f"Done. Total rows uploaded: {buffer.total_rows}")


def build_output_repo(hf_username: str, input_repo: str, model_name: str, parquet_str: str | None = None) -> str:
    dataset_name = input_repo.split("/")[-1]
    model_suffix = "8B" if "8B" in model_name else "1.7B"
    repo = f"{hf_username}/{dataset_name}-RQ-{model_suffix}"
    if parquet_str:
        parts = "-".join([p.strip() for p in parquet_str.split(",")])
        repo += f"-parts-{parts}"
    return repo


def main():
    parser = argparse.ArgumentParser(description="Rewrite questions for dataset")
    parser.add_argument("--input-repo",    default=INPUT_REPO)
    parser.add_argument("--output-repo",   default=None)
    parser.add_argument("--hf-username",   default=HF_USERNAME)
    parser.add_argument("--local-dir",     default=LOCAL_DIR)
    parser.add_argument("--private",       action="store_true")
    parser.add_argument("--upload-threshold-gb", type=float, default=UPLOAD_THRESHOLD_GB)
    parser.add_argument("--parquet-rows",  type=int, default=PARQUET_ROWS)
    parser.add_argument("--debug",         action="store_true")
    parser.add_argument("--n-candidates",  type=int, default=N_CANDIDATES)
    parser.add_argument("--rewrite-model", default=REWRITE_MODEL)
    parser.add_argument("--rewrite-base-url", default=REWRITE_BASE_URL)
    parser.add_argument("--rewrite-tp",    type=int, default=REWRITE_TP_SIZE)
    parser.add_argument("--rewrite-tpu-devices", default=REWRITE_TPU_DEVICES)
    parser.add_argument("--rewrite_concurrency", type=int, default=MAX_CONCURRENCY_REWRITE)
    parser.add_argument("--rewrite-max-num-seqs", type=int, default=REWRITE_MAX_NUM_SEQS)
    parser.add_argument("--parquet-numbers", type=str, default=None)

    args = parser.parse_args()

    if args.debug:
        args.n_candidates = 5000
        logging.info("Debug mode: processing 2000 rows.")

    if args.output_repo is None:
        args.output_repo = build_output_repo(args.hf_username, args.input_repo, args.rewrite_model, args.parquet_numbers)
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
