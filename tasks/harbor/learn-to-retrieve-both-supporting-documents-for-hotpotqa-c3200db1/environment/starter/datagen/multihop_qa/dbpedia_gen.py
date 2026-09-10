"""Large-scale multihop QA generation from random DBpedia nodes.

Saves results as parquet files locally and pushes each file to HuggingFace
asynchronously as soon as it is written.
"""
import sys, os, json, asyncio, uuid, argparse, logging, time
sys.stdout.reconfigure(line_buffering=True)

# dbpedia_gen.py is in the same directory as its imports, so no path manipulation needed

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from tqdm import tqdm

from kg.dbpedia import get_abstract, get_random_entity
from kg.sampler import sample_path as _sample_path
from retrieval.wiki import entity_to_title, get_paragraph_with_link
from generation.generator import generate_question

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ── defaults ──────────────────────────────────────────────────────────────────
LOCAL_DIR       = "./temp_multihop_upload"
PARQUET_ROWS    = 1000
MAX_CONCURRENCY = 32

SCHEMA = pa.schema([
    ("question",   pa.string()),
    ("answer",     pa.string()),
    ("path",       pa.string()),       # JSON list of hop dicts
    ("paragraphs", pa.string()),       # JSON dict of {entity: text}
])


# ── async upload buffer ───────────────────────────────────────────────────────
def _upload_and_delete_sync(api: HfApi, path: str, repo: str, upload_count: int, retries: int = 5):
    for attempt in range(retries):
        try:
            logging.info(f"Uploading batch {upload_count} ({os.path.basename(path)}) to {repo}... (attempt {attempt+1})")
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=os.path.basename(path),
                repo_id=repo,
                repo_type="dataset",
                commit_message=f"Upload batch {upload_count}",
            )
            logging.info(f"Upload complete. Removing {os.path.basename(path)}.")
            try:
                os.remove(path)
            except OSError as e:
                logging.warning(f"Could not remove {path}: {e}")
            return
        except Exception as e:
            wait = 2 ** attempt
            logging.error(f"Upload attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    logging.error(f"All {retries} upload attempts failed for {os.path.basename(path)}. File kept locally.")


class ResultBuffer:
    def __init__(self, local_dir: str, parquet_rows: int, api: HfApi, repo: str):
        self.local_dir    = local_dir
        self.parquet_rows = parquet_rows
        self.api          = api
        self.repo         = repo

        self.records: list[dict]          = []
        self.chunk_index  = 0
        self.upload_count = 0
        self.total_rows   = 0
        self._upload_tasks: set[asyncio.Task] = set()
        self._lock        = asyncio.Lock()

    async def add(self, record: dict):
        async with self._lock:
            self.records.append(record)
            if len(self.records) >= self.parquet_rows:
                self._flush_and_schedule_upload()

    def _flush_and_schedule_upload(self):
        """Write one parquet file and immediately kick off an async upload. Called under lock."""
        table = pa.Table.from_pylist(self.records, schema=SCHEMA)
        uid   = uuid.uuid4().hex[:8]
        path  = os.path.join(self.local_dir, f"data_{uid}_{self.chunk_index:06d}.parquet")
        pq.write_table(table, path)
        self.total_rows  += len(table)
        self.chunk_index += 1
        self.records      = []
        logging.info(f"Flushed {len(table)} rows → {os.path.basename(path)}  (total: {self.total_rows})")

        self.upload_count += 1
        task = asyncio.create_task(self._upload_task(path, self.upload_count))
        self._upload_tasks.add(task)
        task.add_done_callback(self._upload_tasks.discard)

    async def _upload_task(self, path: str, upload_count: int):
        try:
            await asyncio.to_thread(_upload_and_delete_sync, self.api, path, self.repo, upload_count)
        except Exception as e:
            logging.error(f"Upload failed for {os.path.basename(path)}: {e}. File kept locally.")

    async def flush_and_upload_final(self):
        async with self._lock:
            if self.records:
                self._flush_and_schedule_upload()

        if self._upload_tasks:
            logging.info(f"Waiting for {len(self._upload_tasks)} in-flight uploads...")
            await asyncio.gather(*self._upload_tasks, return_exceptions=True)


# ── generation helpers (sync, run in thread) ──────────────────────────────────
def _fetch_paragraphs(path):
    outgoing     = {s: (rel, o) for s, rel, o in path}
    terminal_uri = path[-1][2]
    seen = {}
    for s, rel, o in path:
        for uri in (s, o):
            title = entity_to_title(uri)
            if title in seen:
                continue
            if uri in outgoing:
                next_rel, next_o = outgoing[uri]
                text = get_paragraph_with_link(title, entity_to_title(next_o), relation=next_rel)
            elif uri == terminal_uri:
                from retrieval.wiki import get_intro
                text = get_intro(title)
            else:
                text = get_paragraph_with_link(title, entity_to_title(s))
            seen[title] = text or get_abstract(uri) or ""
    return seen


def _generate_one_sync(hops: int, paths_per_node: int) -> dict | None:
    """Pick a random node, sample paths_per_node paths, return first successful QA."""
    try:
        start = get_random_entity()
        if not start:
            return None

        for _ in range(paths_per_node):
            try:
                path = _sample_path(start, hops=hops)
                if not path:
                    continue

                chain = [
                    {"subject": entity_to_title(s), "relation": rel.split("/")[-1], "object": entity_to_title(o)}
                    for s, rel, o in path
                ]
                paragraphs = _fetch_paragraphs(path)
                qa = generate_question(paragraphs, chain)
                if qa.get("question"):
                    return {
                        "question":   qa["question"],
                        "answer":     qa["answer"],
                        "path":       json.dumps(chain),
                        "paragraphs": json.dumps(paragraphs),
                    }
            except Exception:
                continue
    except Exception:
        pass
    return None


# ── async pipeline ────────────────────────────────────────────────────────────
async def _worker(sem: asyncio.Semaphore, hops: int, paths_per_node: int,
                  buffer: ResultBuffer, stats: dict, pbar=None):
    async with sem:
        try:
            result = await asyncio.to_thread(_generate_one_sync, hops, paths_per_node)
            if result:
                await buffer.add(result)
                stats["generated"] += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(dropped=stats["dropped"], refresh=False)
            else:
                stats["dropped"] += 1
                if pbar is not None:
                    pbar.set_postfix(dropped=stats["dropped"], refresh=False)
        except Exception as e:
            stats["dropped"] += 1
            logging.warning(f"Worker error (suppressed): {e}")


async def amain(args):
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    os.makedirs(args.local_dir, exist_ok=True)

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=args.hf_repo, repo_type="dataset", private=False, exist_ok=True)
    logging.info(f"Repository {args.hf_repo} ready (public).")

    buffer = ResultBuffer(
        local_dir=args.local_dir,
        parquet_rows=args.parquet_rows,
        api=api,
        repo=args.hf_repo,
    )

    sem   = asyncio.Semaphore(args.concurrency)
    stats = {"generated": 0, "dropped": 0, "target": args.num}

    logging.info(
        f"Generating {args.num} examples | hops={args.hops} | "
        f"paths_per_node={args.paths_per_node} | parquet_rows={args.parquet_rows} | "
        f"concurrency={args.concurrency}"
    )

    pbar = tqdm(total=args.num, unit="ex", dynamic_ncols=True)

    pending: set[asyncio.Task] = set()
    node_attempts = 0
    max_attempts  = args.num * 100  # generous ceiling; drop rate can be high

    while stats["generated"] < args.num and node_attempts < max_attempts:
        node_attempts += 1
        task = asyncio.create_task(
            _worker(sem, args.hops, args.paths_per_node, buffer, stats, pbar)
        )
        pending.add(task)
        task.add_done_callback(pending.discard)

        if len(pending) >= args.concurrency:
            await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

    if pending:
        logging.info(f"Waiting for {len(pending)} in-flight tasks...")
        await asyncio.gather(*pending, return_exceptions=True)

    pbar.close()
    logging.info(f"Generation done. Generated={stats['generated']} Dropped={stats['dropped']}")

    logging.info("Flushing and uploading remaining data...")
    await buffer.flush_and_upload_final()
    logging.info(f"Done. Total rows uploaded: {buffer.total_rows}")


def main():
    parser = argparse.ArgumentParser(description="Multihop QA generation from random DBpedia nodes")
    parser.add_argument("--hf-repo",       type=str, required=True,           help="HuggingFace dataset repo (e.g. username/multihop-qa)")
    parser.add_argument("--num",           type=int, default=20,              help="Total QA examples to generate")
    parser.add_argument("--hops",          type=int, default=3,               help="Hops per sampled path")
    parser.add_argument("--paths-per-node",type=int, default=1,               help="Paths to attempt per sampled node")
    parser.add_argument("--parquet-rows",  type=int, default=PARQUET_ROWS,    help="Rows per parquet file before flush+upload")
    parser.add_argument("--concurrency",   type=int, default=MAX_CONCURRENCY, help="Max concurrent generation workers")
    parser.add_argument("--local-dir",     type=str, default=LOCAL_DIR,       help="Local staging directory for parquet files")
    args = parser.parse_args()

    try:
        asyncio.run(amain(args))
        sys.exit(0)
    except KeyboardInterrupt:
        logging.info("Interrupted.")
        sys.exit(130)
    except Exception as e:
        logging.critical(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
