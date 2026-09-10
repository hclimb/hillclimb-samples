#!/usr/bin/env python3
"""
hard_negative_mining.py — Large-Scale Hard Negative Mining on TPU (JAX-native)
===============================================================================

Pipeline
--------
  1. Encode documents in 200K-shard chunks  → periodic npy saves
     (automatically resumes if shards already exist)
  2. Load all shards into one contiguous bfloat16 matrix, transfer to TPU
  3. For every 500K-query batch:
       a. Encode  synthetic_query  with the JAX model
       b. Retrieve top-5 docs via JAX sharded matmul + approx_max_k
       c. Remove pos_doc from the top-5 → neg_docs list (len 0-5)
       d. Save shard locally as JSONL
       e. Push shard to HuggingFace as parquet, then continue

Query columns used
------------------
  --query_column    (default: synthetic_query)
  --pos_doc_column  (default: pos_doc)
  --answer_column   (default: original_answer)

Output record per query
-----------------------
  { "query", "answer", "pos_doc", "neg_docs": list[str], "think", "synthetic_answer" }
"""

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

import ml_dtypes  # registers bfloat16 with numpy
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DOC_CHUNK_SIZE   = 200_000   # docs per shard file
QUERY_BATCH_SIZE = 300_000   # queries per mining batch


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description="Hard-negative mining at scale — JAX-native TPU encoding + retrieval"
    )
    # Corpus
    p.add_argument("--doc_dataset",       required=True)
    p.add_argument("--doc_split",         default="train")
    p.add_argument("--doc_column",        required=True)
    p.add_argument("--max_docs",          type=int, default=5_000_000)
    # Queries
    p.add_argument("--query_dataset",     required=True)
    p.add_argument("--query_split",       default="train")
    p.add_argument("--query_column",      default="synthetic_query")
    p.add_argument("--pos_doc_column",    default="pos_doc")
    p.add_argument("--answer_column",           default="original_answer")
    p.add_argument("--think_column",            default="think")
    p.add_argument("--synthetic_answer_column", default="synthetic_answer")
    p.add_argument("--max_queries",       type=int, default=12_000_000)
    # Encoder
    p.add_argument("--model_name",        required=True,
                   help="HF model ID, e.g. Qwen/Qwen3-Embedding-4B")
    p.add_argument("--hf_ckpt_dir",       default="~/weights/huggingface")
    p.add_argument("--tp_devices",        type=int, default=1)
    p.add_argument("--max_doc_length",    type=int, default=1024)
    p.add_argument("--max_query_length",  type=int, default=128)
    p.add_argument("--doc_prefix",        default="")
    p.add_argument("--query_prefix",      default="")
    p.add_argument("--query_task",        default="",
                   help="If set, format: 'Instruct: {task}\\nQuery:{query}'")
    p.add_argument("--encode_batch_size", type=int, default=2048,
                   help="Texts per JAX forward pass (rounded to multiple of n_devices)")
    # Retrieval
    p.add_argument("--search_batch_size", type=int, default=1024)
    p.add_argument("--top_k",            type=int, default=5)
    # I/O
    p.add_argument("--embeddings_dir",    default="./embeddings_cache")
    p.add_argument("--output_dir",        default="./hard_negatives")
    p.add_argument("--hf_output_dataset", required=True,
                   help="HF dataset repo to push shards to, e.g. 'org/hard-negs'")
    p.add_argument("--hf_token",         default=None,
                   help="HuggingFace token (or set HF_TOKEN env var)")
    return p


# ═══════════════════════════════════════════════════════════════════════════
# ENCODER  (same pattern as single_embedding_retrieval.py)
# ═══════════════════════════════════════════════════════════════════════════

def build_encoder(args):
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from models.qwen3 import load as load_qwen3

    n_dev = jax.device_count()
    log.info(f"JAX: {n_dev}× {jax.devices()[0].platform.upper()}")
    log.info(f"Loading {args.model_name} (tp={args.tp_devices})...")
    t0    = time.time()
    model = load_qwen3(
        model_id=args.model_name,
        tp_devices=args.tp_devices,
        load_weights=True,
        hf_ckpt_dir=args.hf_ckpt_dir,
        mask_type="causal",
    )
    log.info(f"Model loaded in {time.time()-t0:.1f}s")

    model.tokenizer.padding_side = "left"
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id

    weights, fwd = model.weights, model.forward

    @jax.jit
    def embed_fn(input_ids, pad_mask):
        hidden = fwd(input_ids, weights, pad_mask=pad_mask, return_hidden=True)
        emb    = hidden[:, -1, :].astype(jnp.float32)
        norm   = jnp.linalg.norm(emb, axis=-1, keepdims=True)
        return emb / (norm + 1e-12)

    # Warmup with the exact batch size encode_texts will use — prevents a
    # recompilation on the first real batch which could hit a smaller-stack thread.
    actual_batch = max(n_dev, (args.encode_batch_size // n_dev) * n_dev)
    log.info(f"Compiling embed kernels (batch={actual_batch})...")
    for seq_len in sorted({args.max_doc_length, args.max_query_length}):
        dummy_ids  = jax.device_put(jnp.zeros((actual_batch, seq_len), jnp.int32), P("data", None))
        dummy_mask = jax.device_put(jnp.ones( (actual_batch, seq_len), jnp.bool_), P("data", None))
        jax.block_until_ready(embed_fn(dummy_ids, dummy_mask))
        log.info(f"  compiled seq_len={seq_len} ✓")

    return model.tokenizer, embed_fn


def encode_texts(texts, tokenizer, embed_fn, max_length, batch_size, desc=""):
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from tqdm import tqdm

    n_dev      = jax.device_count()
    batch_size = max(n_dev, (batch_size // n_dev) * n_dev)
    results    = []

    for start in tqdm(range(0, len(texts), batch_size), desc=desc or "Encoding", unit="batch"):
        chunk    = list(texts[start : start + batch_size])
        actual_b = len(chunk)
        rem = actual_b % n_dev
        if rem:
            chunk += [chunk[-1]] * (n_dev - rem)

        tok = tokenizer(
            chunk,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        ids_jax  = jax.device_put(jnp.array(tok["input_ids"],      jnp.int32), P("data", None))
        mask_jax = jax.device_put(jnp.array(tok["attention_mask"], jnp.bool_), P("data", None))

        embs = embed_fn(ids_jax, mask_jax)
        jax.block_until_ready(embs)
        results.append(np.array(embs)[:actual_b])

    return np.concatenate(results, axis=0)  # float32 [N, D]


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: DOC ENCODING — periodic 200K-chunk saves
# ═══════════════════════════════════════════════════════════════════════════

def _chunk_emb_path(cache_dir: Path, idx: int) -> Path:
    return cache_dir / f"doc_emb_chunk_{idx:06d}.npy"

def _chunk_txt_path(cache_dir: Path, idx: int) -> Path:
    return cache_dir / f"doc_txt_chunk_{idx:06d}.jsonl"


def _flush_doc_chunk(texts: list[str], chunk_idx: int, cache_dir: Path,
                     tokenizer, embed_fn, args) -> int:
    """Encode and save one chunk. Returns embed_dim (or reads it from disk if skipped)."""
    emb_path = _chunk_emb_path(cache_dir, chunk_idx)
    txt_path = _chunk_txt_path(cache_dir, chunk_idx)

    if emb_path.exists() and txt_path.exists():
        log.info(f"  chunk {chunk_idx:04d} already on disk — skipping encode")
        return np.load(emb_path, mmap_mode="r").shape[1]

    t0   = time.time()
    embs = encode_texts(
        texts, tokenizer, embed_fn,
        max_length=args.max_doc_length,
        batch_size=args.encode_batch_size,
        desc=f"Chunk {chunk_idx:04d}",
    )
    np.save(emb_path, embs.astype(ml_dtypes.bfloat16))
    with open(txt_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    log.info(
        f"  chunk {chunk_idx:04d}: {len(texts):,} docs in {elapsed:.1f}s "
        f"({len(texts)/elapsed:,.0f} docs/s) → {emb_path.name}"
    )
    return embs.shape[1]


def run_encode_docs_chunked(args, tokenizer, embed_fn) -> dict:
    """
    Stream and encode all documents in DOC_CHUNK_SIZE shards.
    Writes doc_manifest.json when done. If manifest already exists, skips.
    Returns the manifest dict.
    """
    from datasets import load_dataset

    cache_dir     = Path(args.embeddings_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "doc_manifest.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        log.info(
            f"Doc manifest found: {manifest['total_docs']:,} docs "
            f"in {manifest['num_chunks']} chunks — skipping doc encode"
        )
        return manifest

    log.info(f"Streaming docs from {args.doc_dataset} [{args.doc_split}]...")
    ds = load_dataset(args.doc_dataset, split=args.doc_split, streaming=True)

    chunk_texts: list[str] = []
    chunks_meta: list[dict] = []
    total_docs = 0
    embed_dim  = 0

    for example in ds:
        if total_docs >= args.max_docs:
            break
        text = example.get(args.doc_column, "")
        if not text:
            continue
        chunk_texts.append(args.doc_prefix + str(text))
        total_docs += 1

        if len(chunk_texts) == DOC_CHUNK_SIZE:
            chunk_idx = len(chunks_meta)
            embed_dim = _flush_doc_chunk(chunk_texts, chunk_idx, cache_dir, tokenizer, embed_fn, args)
            chunks_meta.append({
                "chunk_idx": chunk_idx,
                "offset":    total_docs - DOC_CHUNK_SIZE,
                "size":      DOC_CHUNK_SIZE,
            })
            chunk_texts = []
            log.info(f"Progress: {total_docs:,} / {args.max_docs:,} docs encoded")

    # Flush remainder
    if chunk_texts:
        chunk_idx = len(chunks_meta)
        size      = len(chunk_texts)
        embed_dim = _flush_doc_chunk(chunk_texts, chunk_idx, cache_dir, tokenizer, embed_fn, args)
        chunks_meta.append({
            "chunk_idx": chunk_idx,
            "offset":    total_docs - size,
            "size":      size,
        })
        log.info(f"Final chunk {chunk_idx:04d}: {size:,} docs")

    manifest = {
        "total_docs": total_docs,
        "num_chunks": len(chunks_meta),
        "embed_dim":  embed_dim,
        "model":      args.model_name,
        "chunks":     chunks_meta,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(f"Doc manifest written → {manifest_path}")
    return manifest


# ═══════════════════════════════════════════════════════════════════════════
# LOAD ALL CHUNK SHARDS
# ═══════════════════════════════════════════════════════════════════════════

def load_all_doc_embeddings(cache_dir: Path, manifest: dict) -> np.ndarray:
    """Concatenate all chunk npy files → (N, D) bfloat16."""
    parts = []
    for entry in manifest["chunks"]:
        p   = _chunk_emb_path(cache_dir, entry["chunk_idx"])
        arr = np.load(p)
        # np.load may return bfloat16 or uint16 depending on numpy/ml_dtypes version
        if arr.dtype != ml_dtypes.bfloat16:
            arr = arr.view(ml_dtypes.bfloat16)
        parts.append(arr)
        log.info(f"  chunk {entry['chunk_idx']:04d}: {arr.shape[0]:,} docs")
    combined = np.concatenate(parts, axis=0)
    log.info(f"Combined doc matrix: {combined.shape[0]:,} × {combined.shape[1]} bfloat16")
    return combined


def load_all_doc_texts(cache_dir: Path, manifest: dict) -> list[str]:
    texts = []
    for entry in manifest["chunks"]:
        p = _chunk_txt_path(cache_dir, entry["chunk_idx"])
        with open(p, encoding="utf-8") as f:
            for line in f:
                texts.append(json.loads(line)["text"])
    log.info(f"Loaded {len(texts):,} doc texts into CPU RAM")
    return texts


# ═══════════════════════════════════════════════════════════════════════════
# RETRIEVAL KERNEL
# ═══════════════════════════════════════════════════════════════════════════

def build_retrieval_kernel(doc_embs: np.ndarray, args):
    """
    Transfers the full doc matrix to TPU (replicated on every chip) and
    JIT-compiles the search kernel.

    Uses axis name "data" (same as the encoder) so the mesh remains valid
    for encode_texts calls that happen later during query mining.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import NamedSharding, PartitionSpec as P, AxisType

    devices   = jax.devices()
    n_devices = len(devices)
    embed_dim = doc_embs.shape[1]
    top_k     = args.top_k

    # Must match the mesh set by load_qwen3 exactly — same axes, same AxisType.
    # Using plain Mesh(...) defaults to AxisType.Auto which breaks embed_fn
    # when it's called again for query encoding after this kernel is built.
    fsdp = n_devices // args.tp_devices
    mesh = jax.make_mesh(
        (fsdp, args.tp_devices), ('data', 'model'),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    jax.set_mesh(mesh)
    db_sharding = NamedSharding(mesh, P())              # replicated
    q_sharding  = NamedSharding(mesh, P("data", None))  # batch-sharded

    log.info(
        f"Transferring {doc_embs.shape[0]:,} × {embed_dim} doc matrix "
        f"to TPU ({n_devices} chips, replicated)..."
    )
    t0     = time.time()
    db_jax = jax.device_put(doc_embs, db_sharding)
    jax.block_until_ready(db_jax)
    log.info(f"  on TPU in {time.time()-t0:.1f}s")

    # Pad search batch to multiple of n_devices
    B = args.search_batch_size
    if B % n_devices:
        B = ((B + n_devices - 1) // n_devices) * n_devices

    @jax.jit
    def search_fn(queries, database):
        scores = queries @ database.T
        top_scores, top_indices = lax.approx_max_k(scores, k=top_k)
        return top_scores.astype(jnp.float32), top_indices

    # Warm up JIT
    dummy = jax.device_put(np.zeros((B, embed_dim), dtype=ml_dtypes.bfloat16), q_sharding)
    jax.block_until_ready(search_fn(dummy, db_jax))
    log.info("Search kernel compiled ✓")

    return db_jax, search_fn, n_devices, embed_dim, B, q_sharding


# ═══════════════════════════════════════════════════════════════════════════
# PUSH SHARD TO HF
# ═══════════════════════════════════════════════════════════════════════════

def push_shard_to_hf(records: list[dict], batch_idx: int, args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    api = HfApi(token=args.hf_token)
    api.create_repo(repo_id=args.hf_output_dataset, repo_type="dataset", exist_ok=True, private=False)

    schema = pa.schema([
        pa.field("query",            pa.string()),
        pa.field("answer",           pa.string()),
        pa.field("pos_doc",          pa.string()),
        pa.field("neg_docs",         pa.list_(pa.string())),
        pa.field("think",            pa.string()),
        pa.field("synthetic_answer", pa.string()),
    ])
    table = pa.table(
        {
            "query":            [r["query"]            for r in records],
            "answer":           [r["answer"]           for r in records],
            "pos_doc":          [r["pos_doc"]          for r in records],
            "neg_docs":         [r["neg_docs"]         for r in records],
            "think":            [r["think"]            for r in records],
            "synthetic_answer": [r["synthetic_answer"] for r in records],
        },
        schema=schema,
    )

    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)

    t0 = time.time()
    api.upload_file(
        path_or_fileobj=buf,
        path_in_repo=f"data/train-{batch_idx:05d}.parquet",
        repo_id=args.hf_output_dataset,
        repo_type="dataset",
        commit_message=f"shard {batch_idx:05d} ({len(records):,} rows)",
    )
    log.info(f"  pushed shard {batch_idx:05d} in {time.time()-t0:.1f}s → {args.hf_output_dataset}")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: MINE HARD NEGATIVES — 500K-query batches
# ═══════════════════════════════════════════════════════════════════════════

def _mine_one_batch(
    batch_idx: int,
    q_raw: list[str],
    q_fmt: list[str],
    answers: list[str],
    pos_docs: list[str],
    thinks: list[str],
    synthetic_answers: list[str],
    tokenizer, embed_fn,
    db_jax, search_fn,
    embed_dim: int, search_B: int, q_sharding,
    doc_texts: list[str],
    args,
    output_dir: Path,
):
    import jax
    import jax.numpy as jnp
    from tqdm import tqdm

    n      = len(q_raw)
    prefix = args.doc_prefix
    log.info(f"\n╔══ Query batch {batch_idx}: {n:,} queries ══╗")
    t0 = time.time()

    # 1. Encode queries
    q_embs = encode_texts(
        q_fmt, tokenizer, embed_fn,
        max_length=args.max_query_length,
        batch_size=args.encode_batch_size,
        desc=f"Encode Q-batch {batch_idx}",
    )

    # 2. Retrieve top_k for each query
    all_indices = []
    for start in tqdm(range(0, n, search_B), desc="Retrieve", unit="batch"):
        end    = min(start + search_B, n)
        chunk  = q_embs[start:end].astype(ml_dtypes.bfloat16)
        actual = chunk.shape[0]
        if actual < search_B:
            chunk = np.concatenate(
                [chunk, np.zeros((search_B - actual, embed_dim), dtype=chunk.dtype)],
                axis=0,
            )
        q_jax          = jax.device_put(jnp.array(chunk), q_sharding)
        _, top_indices = search_fn(q_jax, db_jax)
        jax.block_until_ready(top_indices)
        all_indices.append(np.array(top_indices)[:actual])

    all_indices = np.clip(np.concatenate(all_indices, axis=0), 0, len(doc_texts) - 1)

    # 3. Build records — strip pos_doc from the top-5 candidates
    records = []
    for i in range(n):
        pos_text = prefix + pos_docs[i]  # prefixed form, matches doc_texts entries
        neg_docs = []
        for rank in range(args.top_k):
            cand = doc_texts[int(all_indices[i, rank])]
            if cand == pos_text:
                continue  # positive doc — skip
            # Strip prefix so neg_docs are consistent with pos_doc (raw text)
            out = cand[len(prefix):] if (prefix and cand.startswith(prefix)) else cand
            neg_docs.append(out)
        records.append({
            "query":            q_raw[i],
            "answer":           answers[i],
            "pos_doc":          pos_docs[i],
            "neg_docs":         neg_docs,
            "think":            thinks[i],
            "synthetic_answer": synthetic_answers[i],
        })

    elapsed = time.time() - t0
    avg_negs = sum(len(r["neg_docs"]) for r in records) / n
    log.info(
        f"Batch {batch_idx}: {n:,} records in {elapsed:.1f}s "
        f"(avg neg_docs: {avg_negs:.2f})"
    )

    # 4. Save locally (staging copy; deleted after successful HF push)
    shard_path = output_dir / f"shard_{batch_idx:05d}.jsonl"
    with open(shard_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"  saved → {shard_path}")

    # 5. Push to HF, then remove local staging file
    push_shard_to_hf(records, batch_idx, args)
    shard_path.unlink()
    log.info(f"  deleted local shard {shard_path.name}")


def run_mine_negatives(
    args, tokenizer, embed_fn,
    db_jax, search_fn, n_devices, embed_dim, search_B, q_sharding,
    doc_texts: list[str],
) -> int:
    from datasets import load_dataset

    output_dir    = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"

    done_batches: set[int] = set()
    if progress_path.exists():
        done_batches = set(json.loads(progress_path.read_text()).get("done_batches", []))
        log.info(f"Resuming: {len(done_batches)} query batch(es) already completed")

    log.info(f"Streaming queries from {args.query_dataset} [{args.query_split}]...")
    ds = load_dataset(args.query_dataset, split=args.query_split, streaming=True)

    buf_raw:              list[str] = []
    buf_fmt:              list[str] = []
    buf_answers:          list[str] = []
    buf_pos_docs:         list[str] = []
    buf_thinks:           list[str] = []
    buf_synthetic_answers: list[str] = []
    total_seen = 0
    batch_idx  = 0

    def maybe_flush(b_idx, final=False):
        nonlocal done_batches
        label = f"Query batch {b_idx}"
        if b_idx in done_batches:
            log.info(f"{label} already done — skipping")
            return
        _mine_one_batch(
            b_idx, buf_raw, buf_fmt, buf_answers, buf_pos_docs,
            buf_thinks, buf_synthetic_answers,
            tokenizer, embed_fn,
            db_jax, search_fn,
            embed_dim, search_B, q_sharding,
            doc_texts, args, output_dir,
        )
        done_batches.add(b_idx)
        progress_path.write_text(json.dumps({"done_batches": sorted(done_batches)}, indent=2))

    for example in ds:
        if total_seen >= args.max_queries:
            break
        query_text = example.get(args.query_column, "")
        if not query_text:
            continue

        pos_doc          = str(example.get(args.pos_doc_column,          ""))
        answer           = str(example.get(args.answer_column,           ""))
        think            = str(example.get(args.think_column,            ""))
        synthetic_answer = str(example.get(args.synthetic_answer_column, ""))

        if args.query_task:
            formatted = f"Instruct: {args.query_task}\nQuery:{query_text}"
        else:
            formatted = args.query_prefix + str(query_text)

        buf_raw.append(str(query_text))
        buf_fmt.append(formatted)
        buf_answers.append(answer)
        buf_pos_docs.append(pos_doc)
        buf_thinks.append(think)
        buf_synthetic_answers.append(synthetic_answer)
        total_seen += 1

        if len(buf_raw) == QUERY_BATCH_SIZE:
            maybe_flush(batch_idx)
            batch_idx += 1
            buf_raw               = []
            buf_fmt               = []
            buf_answers           = []
            buf_pos_docs          = []
            buf_thinks            = []
            buf_synthetic_answers = []
            log.info(f"Running total: {total_seen:,} / {args.max_queries:,} queries")

    # Flush last partial batch
    if buf_raw:
        maybe_flush(batch_idx)
        batch_idx += 1

    log.info(f"\nMining complete: {total_seen:,} queries across {batch_idx} batches")
    return batch_idx


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args   = parser.parse_args()

    log.info("=" * 65)
    log.info("Hard Negative Mining — JAX-native TPU pipeline")
    log.info("=" * 65)
    log.info(f"Doc corpus:   {args.doc_dataset} [{args.doc_split}] (max {args.max_docs:,})")
    log.info(f"Queries:      {args.query_dataset} [{args.query_split}] (max {args.max_queries:,})")
    log.info(f"Doc chunk:    {DOC_CHUNK_SIZE:,}  |  Query batch: {QUERY_BATCH_SIZE:,}")
    log.info(f"Top-K:        {args.top_k}")
    log.info(f"HF output:    {args.hf_output_dataset}")

    cache_dir     = Path(args.embeddings_dir)
    manifest_path = cache_dir / "doc_manifest.json"

    # Always build encoder — needed for both doc and query phases
    log.info("\n╔══ Building JAX encoder ══╗")
    tokenizer, embed_fn = build_encoder(args)

    # Phase 1: encode documents (skipped if manifest exists)
    if not manifest_path.exists():
        log.info("\n╔══ Phase 1: Encode documents (chunked, 200K/shard) ══╗")
    manifest = run_encode_docs_chunked(args, tokenizer, embed_fn)

    # Load all doc shards into CPU RAM, then move to TPU
    log.info("\n╔══ Loading doc embeddings ══╗")
    doc_embs  = load_all_doc_embeddings(cache_dir, manifest)
    doc_texts = load_all_doc_texts(cache_dir, manifest)

    log.info("\n╔══ Building retrieval kernel ══╗")
    db_jax, search_fn, n_devices, embed_dim, search_B, q_sharding = \
        build_retrieval_kernel(doc_embs, args)
    del doc_embs  # free CPU RAM after TPU transfer

    log.info("\n╔══ Phase 2: Mine hard negatives (500K queries/batch) ══╗")
    t0 = time.time()
    n_batches = run_mine_negatives(
        args, tokenizer, embed_fn,
        db_jax, search_fn, n_devices, embed_dim, search_B, q_sharding,
        doc_texts,
    )
    log.info(f"\nTotal time: {(time.time()-t0)/60:.1f} min across {n_batches} query batches")


if __name__ == "__main__":
    import threading
    # MLIR lowering for large models (36+ layers with jax.remat unrolled) recurses
    # deeply in native C++ stack. The main thread's stack can't be resized after
    # process start, but new threads honour threading.stack_size().
    threading.stack_size(1 * 1024 * 1024 * 1024)  # 1 GB
    t = threading.Thread(target=main)
    t.start()
    t.join()
