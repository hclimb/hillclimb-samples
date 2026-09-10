#!/usr/bin/env python3
"""
Two-phase pipeline for BRIGHT bridge QA generation.

Phase 1: Sample pairs of docs from different BRIGHT examples, generate a
         bridging question that requires facts from BOTH docs (Gemini 2.5 Pro).
Phase 2: Generate answers + thinking traces for each question (Qwen3-1.7B via vLLM).

Usage:
    python datagen/gen_qa_cluster.py \
        --n-candidates 50 \
        --answer-model Qwen/Qwen3-1.7B \
        --output-repo underfrog/BRIGHT-BridgeQA-Gemini-Qwen3 \
        --sampling random   # or --sampling cluster
"""

import asyncio
import argparse
import glob as glob_module
import json
import logging
import os
import re
import random
import sys
import dotenv

dotenv.load_dotenv()
print(os.getenv("HF_TOKEN"))

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm.asyncio import tqdm
import vertexai
from vertexai.generative_models import GenerativeModel

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from vllm_inference import VLLMInference

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ─── defaults ─────────────────────────────────────────────────────────────────
GCP_PROJECT            = "memory-layers-data-synthesis"
GCP_LOCATION           = "us-east4"
GEMINI_MODEL           = "gemini-2.5-pro"

ANSWER_MODEL           = "Qwen/Qwen3-1.7B"
ANSWER_BASE_URL        = "http://localhost:8001/v1"
ANSWER_TP_SIZE         = 4

INPUT_REPO             = "xlangai/BRIGHT"
DOMAINS                = ["biology", "earth_science", "economics"]

ANSWER_TYPE            = "BridgeQA"
HF_USERNAME            = "underfrog"

N_CANDIDATES           = 10
MIN_WORDS              = 50
MAX_CONCURRENCY_ANSWER = 8

PARQUET_ROWS           = 25
LOCAL_DIR              = "./temp_qa_upload"
QUERY_CHUNK_SIZE       = 10

DOCS_PER_QUERY         = 8   # 2 pos + 6 neg
SAMPLING_MODE          = "random"  # choise between "random" or "cluster" sampling 

SCHEMA = pa.schema([
    ("question",         pa.string()),
    ("think",            pa.string()),
    ("synthetic_answer", pa.string()),
    ("doc1",             pa.string()),
    ("doc2",             pa.string()),
    ("id1",              pa.string()),
    ("id2",              pa.string()),
    ("pos_doc_ids",      pa.list_(pa.string())),
    ("neg_doc_ids",      pa.list_(pa.string())),
    ("neg_docs",         pa.list_(pa.string())),
])

# ─── prompts ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = "You are a helpful assistant that answers questions using additional information provided to you. When reasoning, refer to the provided content as 'additional information' rather than 'Document 1', 'Document 2', or any numbered label."

QUERY_PROMPT = """You are generating training data for a memory retrieval model.

Given two documents below, generate ONE question where:
- Answering it requires specific facts from Document 1 AND specific facts from Document 2.
- Do NOT make it two standalone parts, i.e. NOT where one part of the question mostly comes from Document 1 and another part of the question mostly comes from Document 2. It must be a single question that requires information from both documents that logically connects.
- No mention of "document", "text", "passage" in the question
- Do NOT ask two standalone sub-questions joined by "and".
- The question should be like a curious person would ask.
- Do NOT provide much context in the question, do NOT say "considering" or similar things.

Document 1:
{doc1}

Document 2:
{doc2}

If either document is predominantly NOT full sentences, or if these documents share no meaningful connection, or the question would be bad, respond with exactly: {{"skip": true}}
Otherwise respond in JSON only, no markdown:
{{"question": "..."}}"""

ANSWER_PROMPT = """Additional Information:
{doc1}

---

{doc2}

---

{neg_docs_formatted}

Question: {question}

Use the additional information above to answer the question. In your thinking, refer to the content as 'the additional information' — never as 'Document 1', 'Document 2', 'the user said', or any numbered/labeled reference.

Think through the question carefully.
"""

# ─── helpers ──────────────────────────────────────────────────────────────────
def same_subtopic(id1: str, id2: str) -> bool:
    def base(doc_id):
        path = doc_id.rsplit(".", 1)[0]
        return re.sub(r'_\d+$', '', path)
    return base(id1) == base(id2)


def get_dir_size(path: str) -> int:
    return sum(
        os.path.getsize(os.path.join(path, f))
        for f in os.listdir(path)
        if f.endswith(".parquet")
    )


def _load_docs_and_examples(hf_token: str):
    """Shared data loading used by both sampling strategies."""
    doc_lookup = {}
    valid_examples = []

    for domain in DOMAINS:
        logging.info(f"  Loading domain: {domain}...")
        examples = load_dataset("xlangai/BRIGHT", "examples", token=hf_token)[domain]
        docs_ds  = load_dataset("xlangai/BRIGHT", "documents", token=hf_token)[domain]
        for d in docs_ds:
            if len(d["content"].split()) >= MIN_WORDS:
                doc_lookup[d["id"]] = d["content"]
        valid = [e for e in examples if len(e["gold_ids"]) >= 2]
        valid_examples.extend(valid)
        logging.info(f"    {domain}: {len(valid)} valid examples")

    logging.info(f"Total docs: {len(doc_lookup)}, valid examples: {len(valid_examples)}")
    return doc_lookup, valid_examples


def _build_pair(id1, id2, doc_lookup, example1, example2):
    excluded     = set(example1["gold_ids"]) | set(example2["gold_ids"])
    neg_candidates = [i for i in doc_lookup if i not in excluded]
    neg_ids      = random.sample(neg_candidates, min(DOCS_PER_QUERY - 2, len(neg_candidates)))
    return {
        "id1": id1,
        "id2": id2,
        "doc1": doc_lookup[id1][:1500],
        "doc2": doc_lookup[id2][:1500],
        "pos_doc_ids": [id1, id2],
        "neg_doc_ids": neg_ids,
        "neg_docs": [doc_lookup[i][:1500] for i in neg_ids],
    }


# ─── sampling method 1: random gold sampling ───────────────────────────────
def load_pairs_random(n_candidates: int, hf_token: str) -> list[dict]:
    """Sample pairs by picking 2 random examples and choosing one gold doc from each."""
    logging.info("Sampling mode: RANDOM")
    doc_lookup, valid_examples = _load_docs_and_examples(hf_token)

    pairs = []
    attempts = 0
    max_attempts = n_candidates * 20

    while len(pairs) < n_candidates and attempts < max_attempts:
        attempts += 1
        example1, example2 = random.sample(valid_examples, 2)

        gold1 = [g for g in example1["gold_ids"] if g in doc_lookup]
        gold2 = [g for g in example2["gold_ids"] if g in doc_lookup]
        if not gold1 or not gold2:
            continue

        id1 = random.choice(gold1)
        id2 = random.choice(gold2)
        if same_subtopic(id1, id2):
            continue

        pairs.append(_build_pair(id1, id2, doc_lookup, example1, example2))

    logging.info(f"Sampled {len(pairs)} pairs ({attempts} attempts, skip rate: {1 - len(pairs)/max(attempts,1):.1%})")
    return pairs


# ─── sample method 2: cluster with sentence transformer ──────────────────────────
def _cluster_gold_docs(gold_doc_ids: list[str], doc_lookup: dict, n_clusters: int = 50) -> dict[int, list[str]]:
    """Cluster gold docs using sentence-transformers embeddings + KMeans."""
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    valid_ids = [i for i in gold_doc_ids if i in doc_lookup]
    if len(valid_ids) < n_clusters:
        n_clusters = max(2, len(valid_ids) // 5)

    logging.info(f"Embedding {len(valid_ids)} gold docs with sentence-transformers...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [doc_lookup[i][:500] for i in valid_ids]
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True)

    logging.info(f"Clustering into {n_clusters} clusters...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    clusters: dict[int, list[str]] = {}
    for idx, c in enumerate(labels):
        clusters.setdefault(int(c), []).append(valid_ids[idx])

    # Merge singletons into fallback, never delete gold docs
    fallback = []
    good_clusters = {}
    for c, ids in clusters.items():
        if len(ids) >= 2:
            good_clusters[c] = ids
        else:
            fallback.extend(ids)
    if len(fallback) >= 2:
        good_clusters[-1] = fallback

    logging.info(f"Usable clusters: {len(good_clusters)}, fallback pool: {len(fallback)} docs")
    return good_clusters


def load_pairs_cluster(n_candidates: int, hf_token: str) -> list[dict]:
    """Sample pairs by picking 2 docs from the same semantic cluster."""
    logging.info("Sampling mode: CLUSTER (sentence-transformers)")
    doc_lookup, valid_examples = _load_docs_and_examples(hf_token)

    all_gold_ids = set()
    for e in valid_examples:
        all_gold_ids.update(e["gold_ids"])
    logging.info(f"Total gold doc ids: {len(all_gold_ids)}")

    # Build example lookup for neg sampling
    gold_to_example: dict[str, dict] = {}
    for e in valid_examples:
        for g in e["gold_ids"]:
            gold_to_example[g] = e

    clusters = _cluster_gold_docs(list(all_gold_ids), doc_lookup)
    cluster_list = list(clusters.values())

    pairs = []
    attempts = 0
    max_attempts = n_candidates * 20

    while len(pairs) < n_candidates and attempts < max_attempts:
        attempts += 1
        cluster_ids = random.choice(cluster_list)
        if len(cluster_ids) < 2:
            continue

        id1, id2 = random.sample(cluster_ids, 2)
        if same_subtopic(id1, id2):
            continue

        example1 = gold_to_example.get(id1, {"gold_ids": [id1]})
        example2 = gold_to_example.get(id2, {"gold_ids": [id2]})
        pairs.append(_build_pair(id1, id2, doc_lookup, example1, example2))

    logging.info(f"Sampled {len(pairs)} pairs ({attempts} attempts, skip rate: {1 - len(pairs)/max(attempts,1):.1%})")
    return pairs


def load_bright_pairs(n_candidates: int, hf_token: str, sampling: str = "random") -> list[dict]:
    """Entry point — switch between random and cluster sampling."""
    if sampling == "cluster":
        return load_pairs_cluster(n_candidates, hf_token)
    else:
        return load_pairs_random(n_candidates, hf_token)


# ─── result buffer ────────────────────────────────────────────────────────────
def _upload_and_clear_sync(api: HfApi, local_dir: str, repo: str, upload_count: int):
    logging.info(f"Uploading batch {upload_count} to {repo}...")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo,
        repo_type="dataset",
        allow_patterns="data_*.parquet",
        commit_message=f"Upload batch {upload_count}",
    )
    logging.info("Upload successful. Clearing local files...")
    for f in os.listdir(local_dir):
        if f.endswith(".parquet"):
            os.remove(os.path.join(local_dir, f))


class ResultBuffer:
    def __init__(self, local_dir: str, parquet_rows: int, upload_threshold: int,
                 api: HfApi, repo: str):
        self.local_dir        = local_dir
        self.parquet_rows     = parquet_rows
        self.upload_threshold = upload_threshold
        self.api              = api
        self.repo             = repo
        self.records: list[dict] = []
        self.chunk_index  = 0
        self.upload_count = 0
        self.total_rows   = 0
        self.is_uploading = False
        self._lock        = asyncio.Lock()

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
        path  = os.path.join(self.local_dir, f"data_{self.chunk_index:06d}.parquet")
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


# ─── Phase 1: Gemini question generation ─────────────────────────────────────
def generate_query_gemini(gemini_model: GenerativeModel, pair: dict) -> dict | None:
    try:
        prompt   = QUERY_PROMPT.format(doc1=pair["doc1"], doc2=pair["doc2"])
        response = gemini_model.generate_content(prompt)

        usage = response.usage_metadata
        
        thoughts = getattr(usage, 'thoughts_token_count', 0) or 0
        logging.info(
                f"Tokens — input: {usage.prompt_token_count}, "
                f"output: {usage.candidates_token_count}, "
                f"thinking: {thoughts}, "
                f"cost: ${(usage.prompt_token_count * 1.25 + (usage.candidates_token_count + thoughts) * 10) / 1_000_000:.6f}"
                )

        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        result = json.loads(text)
        if result.get("skip"):
            logging.info(f"Skipped pair {pair['id1']} / {pair['id2']} — no bridge")
            return None

        question = result.get("question", "").strip()
        if not question:
            logging.warning(f"Empty question for ids={pair['id1']},{pair['id2']}")
            return None

        return {**pair, "question": question}

    except Exception as e:
        logging.error(f"[gemini query] ids={pair.get('id1')},{pair.get('id2')}: {e}")
        return None


def run_phase1_gemini(pairs: list[dict], queries_dir: str):
    logging.info(f"Initializing Vertex AI: project={GCP_PROJECT}, location={GCP_LOCATION}")
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    gemini_model = GenerativeModel(GEMINI_MODEL)

    chunk_n = 0
    for i in range(0, len(pairs), QUERY_CHUNK_SIZE):
        batch  = pairs[i:i + QUERY_CHUNK_SIZE]
        output = []
        for j, pair in enumerate(batch):
            logging.info(f"  [{i+j+1}/{len(pairs)}] {pair['id1']} × {pair['id2']}")
            result = generate_query_gemini(gemini_model, pair)
            if result is not None:
                output.append(result)
        chunk_path = os.path.join(queries_dir, f"chunk_{chunk_n:06d}.json")
        with open(chunk_path, "w") as f:
            json.dump(output, f)
        logging.info(f"Wrote {len(output)}/{len(batch)} questions → {chunk_path}")
        chunk_n += 1

    logging.info(f"Phase 1 complete. Wrote {chunk_n} chunk(s).")


# ─── Phase 2: vLLM answer generation ─────────────────────────────────────────
async def generate_answer(
    client: VLLMInference,
    semaphore: asyncio.Semaphore,
    entry: dict,
) -> dict:
    async with semaphore:
        try:
            neg_docs_formatted = "\n\n---\n\n".join(entry.get("neg_docs", []))
            content, think = await client.async_chat(
                prompt=ANSWER_PROMPT.format(
                    doc1=entry["doc1"],
                    doc2=entry["doc2"],
                    question=entry["question"],
                    neg_docs_formatted=neg_docs_formatted,
                ),
                system=SYSTEM_PROMPT,
                max_tokens=4096,
                temperature=0.7,
                thinking=True,
            )
            return {**entry, "think": think, "synthetic_answer": content or ""}
        except Exception as e:
            logging.error(f"[answer] ids={entry.get('id1')},{entry.get('id2')}: {e}")
            return {**entry, "think": None, "synthetic_answer": None}


# ─── main ─────────────────────────────────────────────────────────────────────
async def amain(args):
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    os.makedirs(args.local_dir, exist_ok=True)
    queries_dir = os.path.join(args.local_dir, "queries")
    os.makedirs(queries_dir, exist_ok=True)

    api = HfApi(token=hf_token)

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    existing_chunks = sorted(glob_module.glob(os.path.join(queries_dir, "chunk_*.json")))
    if existing_chunks:
        logging.info(f"Found {len(existing_chunks)} existing query chunks — skipping Phase 1.")
    else:
        logging.info("Phase 1: loading BRIGHT and sampling doc pairs...")
        pairs = load_bright_pairs(args.n_candidates, hf_token, sampling=args.sampling)
        if not pairs:
            logging.error("No valid pairs found. Exiting.")
            return
        logging.info("Phase 1: generating bridge questions with Gemini...")
        await asyncio.to_thread(run_phase1_gemini, pairs, queries_dir)

    chunk_files = sorted(glob_module.glob(os.path.join(queries_dir, "chunk_*.json")))
    non_empty   = [f for f in chunk_files if len(json.load(open(f))) > 0]
    total_questions = sum(len(json.load(open(f))) for f in non_empty)
    if total_questions == 0:
        logging.warning("Phase 1 produced 0 questions. Exiting.")
        return
    chunk_files = non_empty

    logging.info(f"Phase 1 produced {total_questions} questions. Creating HF repo...")
    api.create_repo(repo_id=args.output_repo, repo_type="dataset", private=args.private, exist_ok=True)
    logging.info(f"Repository {args.output_repo} ready.")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    logging.info(f"Phase 2: generating answers for {len(chunk_files)} chunk(s)...")
    answer_proc = None
    if not VLLMInference.is_server_ready(args.answer_base_url):
        logging.info(f"Starting answer model {args.answer_model}...")
        answer_proc = VLLMInference.start_server(
            model=args.answer_model, base_url=args.answer_base_url,
            tensor_parallel_size=args.answer_tp,
        )
        VLLMInference.wait_for_server(args.answer_base_url)
    else:
        logging.info("Answer server already running.")

    try:
        answer_client = VLLMInference(model=args.answer_model, base_url=args.answer_base_url)
        answer_sem    = asyncio.Semaphore(args.answer_concurrency)

        buffer = ResultBuffer(
            local_dir=args.local_dir,
            parquet_rows=args.parquet_rows,
            upload_threshold=int(args.upload_threshold_gb * 1024 ** 3),
            api=api,
            repo=args.output_repo,
        )

        for chunk_file in chunk_files:
            with open(chunk_file) as f:
                entries = json.load(f)
            logging.info(f"Processing {len(entries)} entries from {os.path.basename(chunk_file)}...")
            tasks   = [generate_answer(answer_client, answer_sem, e) for e in entries]
            results = await tqdm.gather(*tasks, total=len(tasks))
            for r in results:
                if r.get("synthetic_answer") is not None:
                    await buffer.add(r)
            os.remove(chunk_file)

        logging.info("Phase 2 complete. Flushing and uploading remaining data...")
        await buffer.flush_and_upload_final()
        logging.info(f"Done. Total rows uploaded: {buffer.total_rows}")
    finally:
        if answer_proc is not None:
            answer_proc.terminate()
            logging.info("Answer server terminated.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate bridge QA pairs from BRIGHT using Gemini + vLLM"
    )
    parser.add_argument("--output-repo",          default=None)
    parser.add_argument("--hf-username",          default=HF_USERNAME)
    parser.add_argument("--answer-type",          default=ANSWER_TYPE)
    parser.add_argument("--local-dir",            default=LOCAL_DIR)
    parser.add_argument("--private",              action="store_true")
    parser.add_argument("--upload-threshold-gb",  type=float, default=0.001)
    parser.add_argument("--parquet-rows",         type=int,   default=PARQUET_ROWS)
    parser.add_argument("--n-candidates",         type=int,   default=N_CANDIDATES)
    parser.add_argument("--answer-model",         default=ANSWER_MODEL)
    parser.add_argument("--answer-base-url",      default=ANSWER_BASE_URL)
    parser.add_argument("--answer-tp",            type=int,   default=ANSWER_TP_SIZE)
    parser.add_argument("--answer-concurrency",   type=int,   default=MAX_CONCURRENCY_ANSWER)
    parser.add_argument("--sampling",             default=SAMPLING_MODE,
                        choices=["random", "cluster"],
                        help="Pair sampling strategy: 'random' (gold docs) or 'cluster' (semantic clusters)")

    args = parser.parse_args()

    if args.output_repo is None:
        model_name = args.answer_model.split("/")[-1]
        args.output_repo = f"{args.hf_username}/BRIGHT-{args.answer_type}-Gemini-{model_name}"
        logging.info(f"Auto-generated output repo: {args.output_repo}")

    asyncio.run(amain(args))


if __name__ == "__main__":
    main()