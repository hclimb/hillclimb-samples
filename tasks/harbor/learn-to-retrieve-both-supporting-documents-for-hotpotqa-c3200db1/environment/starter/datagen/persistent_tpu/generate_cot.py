#!/usr/bin/env python3
import os
os.makedirs("/dev/shm/hf_cache", exist_ok=True)
os.environ["HF_HOME"] = "/dev/shm/hf_cache"
os.environ["VLLM_CACHE_ROOT"] = "/dev/shm/hf_cache"
os.environ["HF_HUB_CACHE"] = "/dev/shm/hf_cache"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
os.environ["LIBTPU_INIT_ARGS"] = "--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true"

"""
Chain-of-thought generation pipeline.

Streams rows from a HuggingFace dataset and generates a synthetic chain-of-thought
(synthetic_think) for each (question, document, answer) triple using a vLLM server.
The CoT is written as if the reader remembered the document internally and reasoned
their way to the answer — no references to the document itself.
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
COT_MODEL           = "Qwen/Qwen3-8B"
COT_BASE_URL        = "http://localhost:8001/v1"
COT_TP_SIZE         = 8
COT_TPU_DEVICES     = "0,1,2,3,4,5,6,7"

INPUT_REPO          = "vm2825/nemotron-cc-v21-Parsed-QA4-filtered-1.7B-evensplit-RQ-8B"
HF_USERNAME         = os.getenv("HF_USERNAME", "vm2825").strip()

N_CANDIDATES        = 30000000

MAX_CONCURRENCY_COT = 2048
COT_MAX_NUM_SEQS    = 256

PARQUET_ROWS        = 100000
UPLOAD_THRESHOLD_GB = 0.25
LOCAL_DIR           = "./temp_cot_upload"

SCHEMA = pa.schema([
    ("id",                pa.string()),
    ("pos_doc",           pa.string()),
    ("question",          pa.string()),
    ("original_answer",   pa.string()),
    ("think",             pa.string()),
    ("synthetic_answer",  pa.string()),
    ("synthetic_question", pa.string()),
    ("synthetic_think",   pa.string()),
])

# ─── prompts ──────────────────────────────────────────────────────────────────
COT_SYSTEM = "You are a helpful assistant."

# OLD PROMPT REFERENCE:
# COT_PROMPT = """You have deeply read and understood a document. You no longer have the document in front of you — its knowledge is now part of what you know. Given a question and its correct answer, write a natural chain of thought that shows how someone who knows this information would reason step by step to arrive at that answer.
# 
# ### RULES:
# 1. Write only your internal reasoning — do NOT reference "the document", "the text", "the passage", "the article", or any external source. Reason as if this knowledge is simply what you know.
# 2. The reasoning should flow naturally: observe relevant facts you recall, connect them logically, and converge on the answer.
# 3. Keep the chain of thought concise and grounded — no padding, no repetition.
# 4. Enclose your chain of thought strictly within <cot> and </cot> tags.
# 5. After the closing </cot> tag, output the final answer on a new line, verbatim from the provided answer.
# 
# ### INPUT:
# Question: {question}
# 
# Answer: {answer}
# 
# ### OUTPUT:
# <cot>"""

# SECOND PROMPT REFERENCE:
# THIRD PROMPT REFERENCE:
# COT_PROMPT = """You have read and internalized a document. Given the document, a question about it, and the correct answer, write a natural chain of thought that shows how someone who deeply knows this information would reason step by step to arrive at that answer.
# 
# ### RULES:
# 1. Think first about describing the information from the provided knowledge that is relevant to the question.
# 2. DO NOT reference "the document", "the text", "the passage", "the article", or any external source in your chain of thought at all. Reason as if this knowledge is simply part of what you know.
# 3. The reasoning should flow naturally: describe the facts, connect them logically, and converge on the answer.
# 4. Keep the chain of thought concise and grounded — no padding, no repetition. It MUST be under 200 words.
# 5. Enclose your chain of thought strictly within <cot> and </cot> tags.
# 6. After the closing </cot> tag, output the final answer verbatim.
# 
# ### DOCUMENT:
# {pos_doc}
# 
# ### QUESTION:
# {question}
# 
# ### ANSWER:
# {answer}
# 
# Now write your chain of thought:"""

COT_PROMPT = """You have read and internalized a document. Its knowledge is now simply part of what you know. Given the document, a question, and the correct answer, write a chain of thought showing how someone who already knows this information would reason to that answer.

### RULES:
1. State facts directly as things you know — NEVER reference where the information came from.
2. FORBIDDEN phrases (any variant of these will fail):
   - "the document", "the text", "the passage", "the article", "the study", "the paper"
   - "states that", "mentions that", "indicates that", "discusses", "explicitly stated", "as stated by"
   - "according to", "based on the", "it is noted"
3. The reasoning should flow naturally: state the relevant facts, connect them logically, arrive at the answer.
4. The question may contain phrases like "according to the text" or "according to the study" — these are part of the question phrasing only. Your CoT must NEVER echo them. State the facts directly without any such references as if you inherently know them.
5. No padding, no restating the question, no meta-commentary. Under 150 words.
5. Enclose your chain of thought strictly within <cot> and </cot> tags.
6. After </cot>, output the final answer verbatim.

### EXAMPLES:

DOCUMENT: Formic acid oxidation (FAO) activity on single crystal Pd surfaces increases in the order of Pd(110) < Pd(100) < Pd(111). Twin defects in Pd nanocrystals further enhance FAO activity beyond what facet alone can achieve.
QUESTION: Is the FAO activity on Pd(110) higher than on Pd(111)?
ANSWER: No, the FAO activity on Pd(110) is lower than on Pd(111).

BAD <cot> (DO NOT DO THIS):
The study indicates that FAO activity on single crystal Pd surfaces follows the order Pd(110) < Pd(111). Therefore Pd(110) is lower.
</cot>

GOOD <cot> (DO THIS):
FAO activity on Pd single crystal surfaces follows a strict ranking: Pd(110) < Pd(100) < Pd(111). Pd(110) sits at the bottom of this order, Pd(111) at the top. So Pd(110) activity is definitively lower — not higher — than Pd(111).
</cot>
No, the FAO activity on Pd(110) is lower than on Pd(111).

---

DOCUMENT: Stannic oxide nanoparticles and SnO2@rGO nanohybrids were prepared using a facile hydrothermal method. The hydrothermal approach allows uniform coating of SnO2 on reduced graphene oxide sheets.
QUESTION: What method was used to synthesize stannic oxide nanoparticles and SnO2@rGO nanohybrids?
ANSWER: A facile hydrothermal method was used.

BAD <cot> (DO NOT DO THIS):
The synthesis is explicitly stated to be carried out using a facile hydrothermal method. Therefore, the method used is the facile hydrothermal method.
</cot>

GOOD <cot> (DO THIS):
SnO2 nanoparticles and SnO2@rGO nanohybrids are both synthesized via the same route: a facile hydrothermal method. This approach enables uniform SnO2 deposition onto reduced graphene oxide sheets, making it the method of choice for this nanohybrid system.
</cot>
A facile hydrothermal method was used.

---

### DOCUMENT:
{pos_doc}

### QUESTION:
{question}

### ANSWER:
{answer}

<cot>"""

# COT_PROMPT = """You have read and internalized a document. Given the document, a question about it, and the correct answer, write a natural chain of thought that shows how someone who deeply knows this information would reason step by step to arrive at that answer.

# ### RULES:
# 1. Think first about describing the information from the provided knowledge that is relevant to the question.
# 2. DO NOT use phrases like "the document discusses", "the text mentions", or reference "the passage", "the article", or external sources at all. You MUST use the knowledge from the document as if it is already inherently known to you directly while writing the chain of thought.
# 3. The reasoning should flow naturally: describe the facts, connect them logically, and converge on the answer.
# 4. Keep the chain of thought concise and grounded — no padding, no repetition. It MUST be under 200 words.
# 5. Enclose your chain of thought strictly within <cot> and </cot> tags.
# 6. After the closing </cot> tag, output the final answer verbatim.

# ### DOCUMENT:
# {pos_doc}

# ### QUESTION:
# {question}

# ### ANSWER:
# {answer}

# Now write your chain of thought:"""

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


# Forbidden phrases that indicate the model referenced the source document
FORBIDDEN_PHRASES = [
    "the document", "the text", "the passage", "the article", "the study", "the paper",
    "states that", "mentions that", "indicates that", "it discusses", "explicitly stated",
    "as stated by", "according to", "based on the", "it is noted",
]

MAX_RETRIES    = 3
RETRY_TEMPS    = [0.3, 0.7, 1.0]  # escalating temperature on each retry

def _has_violation(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in FORBIDDEN_PHRASES)

def _extract_cot(content: str, answer: str = "") -> str | None:
    # Case 1: full <cot>...</cot> tags present
    match = re.search(r"<cot>(.*?)</cot>", content, re.DOTALL)
    if match:
        return match.group(1).strip() or None

    # Case 2: prompt ends with <cot> so only closing tag present
    match = re.search(r"^(.*?)</cot>", content.strip(), re.DOTALL)
    if match:
        return match.group(1).strip() or None

    # Case 3: no tags at all — model output is plain text ending with the answer verbatim
    # Strip the answer from the end and treat the rest as the CoT
    if answer:
        stripped = content.strip()
        answer_clean = answer.strip()
        if stripped.endswith(answer_clean):
            cot = stripped[: -len(answer_clean)].strip()
            return cot or None

    return None


# ─── pipeline coroutines ──────────────────────────────────────────────────────
async def generate_cot(
    client: VLLMInference,
    semaphore: asyncio.Semaphore,
    row: dict,
) -> dict:
    async with semaphore:
        # Prefer synthetic_answer if available, fall back to original_answer / answer
        answer = (
            row.get("synthetic_answer")
            or row.get("original_answer")
            or row.get("answer", "")
        )
        question = row.get("synthetic_question") or row.get("question", "")

        base_record = {
            "id":                str(row.get("id", "")),
            "pos_doc":           row.get("pos_doc", ""),
            "question":          row.get("question", ""),
            "original_answer":   row.get("original_answer", "") or row.get("answer", ""),
            "think":             row.get("think", ""),
            "synthetic_answer":  row.get("synthetic_answer", ""),
            "synthetic_question": row.get("synthetic_question", ""),
            "synthetic_think":   None,
        }

        prompt = COT_PROMPT.format(
            pos_doc=row.get("pos_doc", ""),
            question=question,
            answer=answer,
        )

        try:
            for attempt in range(MAX_RETRIES + 1):
                temperature = 0 if attempt == 0 else RETRY_TEMPS[attempt - 1]
                content, _ = await client.async_chat(
                    prompt=prompt,
                    system=COT_SYSTEM,
                    max_completion_tokens=1024,
                    temperature=temperature,
                    thinking=False,
                )

                synthetic_think = _extract_cot(content, answer)
                if not synthetic_think:
                    logging.warning(f"[cot] id={row.get('id')} attempt={attempt}: No <cot> tags found.")
                    continue

                if _has_violation(synthetic_think):
                    logging.warning(f"[cot] id={row.get('id')} attempt={attempt}: Violation found, retrying at temp={RETRY_TEMPS[attempt] if attempt < len(RETRY_TEMPS) else 'max'}.")
                    continue

                return {**base_record, "synthetic_think": synthetic_think}

            logging.warning(f"[cot] id={row.get('id')}: All {MAX_RETRIES + 1} attempts failed, dropping row.")
            return base_record

        except Exception as e:
            logging.error(f"[cot] id={row.get('id')}: {e}")
            return base_record


async def process_row(
    raw_index: int,
    row: dict,
    client: VLLMInference,
    sem: asyncio.Semaphore,
    buffer: ResultBuffer,
    stats: dict,
):
    result = await generate_cot(client, sem, row)
    if result.get("synthetic_think"):
        await buffer.add(result, raw_index)
        stats["generated"] += 1
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
        if not VLLMInference.is_server_ready(args.cot_base_url):
            logging.info(f"Starting server on devices {args.cot_tpu_devices}...")
            VLLMInference.start_server(
                model=args.cot_model, base_url=args.cot_base_url,
                tensor_parallel_size=args.cot_tp,
                tpu_visible_devices=args.cot_tpu_devices,
                max_num_seqs=args.cot_max_num_seqs,
            )
            await asyncio.to_thread(VLLMInference.wait_for_server, args.cot_base_url, timeout=1200)

        client = VLLMInference(model=args.cot_model, base_url=args.cot_base_url)
        sem = asyncio.Semaphore(args.cot_concurrency)

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
            skip_upload=False
        )

        stats = {"total": 0, "generated": 0, "dropped": 0}
        pending_tasks = set()

        async def periodic_upload():
            while True:
                await asyncio.sleep(args.upload_interval)
                async with buffer._lock:
                    if buffer.records:
                        buffer._flush_locked()
                if not buffer.is_uploading and get_dir_size(buffer.local_dir) > 0:
                    buffer.is_uploading = True
                    asyncio.create_task(buffer._upload_task())
                    logging.info(f"Periodic upload triggered (every {args.upload_interval}s).")

        upload_task = asyncio.create_task(periodic_upload())

        data_files = None
        if args.parquet_numbers:
            parquet_nums = [int(p.strip()) for p in args.parquet_numbers.split(",")]
            data_files = [f"data/train-{p:05d}-of-00129.parquet" for p in parquet_nums]

        async for raw_index, row in stream_documents(
            args.input_repo, hf_token, n_candidates=args.n_candidates,
            resume_raw_index=resume_idx, data_files=data_files
        ):
            stats["total"] += 1
            task = asyncio.create_task(process_row(raw_index, row, client, sem, buffer, stats))
            pending_tasks.add(task)

            if len(pending_tasks) >= args.cot_concurrency:
                done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)

            if stats["total"] % 500 == 0:
                logging.info(
                    f"Dispatched {stats['total']} | Generated={stats['generated']} | "
                    f"Dropped={stats['dropped']} | Pending={len(pending_tasks)}"
                )

        if pending_tasks:
            logging.info(f"Waiting for remaining {len(pending_tasks)} pipeline tasks to complete...")
            try:
                await asyncio.wait_for(asyncio.gather(*pending_tasks), timeout=300)
            except asyncio.TimeoutError:
                logging.warning("Timed out waiting for tasks to complete. Proceeding to shutdown.")

        logging.info(f"Pipeline complete. Total={stats['total']} | Generated={stats['generated']}")
        upload_task.cancel()

    finally:
        logging.info("Flushing and uploading remaining data...")
        await buffer.flush_and_upload_final()
        logging.info(f"Done. Total rows uploaded: {buffer.total_rows}")


def build_output_repo(hf_username: str, input_repo: str, model_name: str, parquet_str: str | None = None) -> str:
    dataset_name = input_repo.split("/")[-1]
    model_suffix = "32B1" if "32B" in model_name else "8B1"
    repo = f"{hf_username}/{dataset_name}-CoT-{model_suffix}"
    if parquet_str:
        x = [p.strip() for p in parquet_str.split(",")]
        y = [x[0],x[-1]]
        parts = "-".join(y)
        repo += f"-parts-{parts}"
    return repo


def main():
    parser = argparse.ArgumentParser(description="Generate chain-of-thought for dataset")
    parser.add_argument("--input-repo",    default=INPUT_REPO)
    parser.add_argument("--output-repo",   default=None)
    parser.add_argument("--hf-username",   default=HF_USERNAME)
    parser.add_argument("--local-dir",     default=LOCAL_DIR)
    parser.add_argument("--private",       action="store_true")
    parser.add_argument("--upload-threshold-gb", type=float, default=UPLOAD_THRESHOLD_GB)
    parser.add_argument("--parquet-rows",  type=int, default=PARQUET_ROWS)
    parser.add_argument("--debug",         action="store_true")
    parser.add_argument("--n-candidates",  type=int, default=N_CANDIDATES)
    parser.add_argument("--cot-model",     default=COT_MODEL)
    parser.add_argument("--cot-base-url",  default=COT_BASE_URL)
    parser.add_argument("--cot-tp",        type=int, default=COT_TP_SIZE)
    parser.add_argument("--cot-tpu-devices", default=COT_TPU_DEVICES)
    parser.add_argument("--cot-concurrency", type=int, default=MAX_CONCURRENCY_COT)
    parser.add_argument("--cot-max-num-seqs", type=int, default=COT_MAX_NUM_SEQS)
    parser.add_argument("--parquet-numbers", type=str, default=None)
    parser.add_argument("--upload-interval", type=int, default=1800, help="Upload to HF every N seconds (default 1800 = 30 min)")

    args = parser.parse_args()

    if args.debug:
        args.n_candidates = 2000
        logging.info("Debug mode: processing 2000 rows.")

    if args.output_repo is None:
        args.output_repo = build_output_repo(args.hf_username, args.input_repo, args.cot_model, args.parquet_numbers)
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
