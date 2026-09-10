#!/usr/bin/env python3
"""
single_embedding_retrieval.py — Single-Embedding RAG on TPU (JAX-native)
=========================================================================

Encoding  : JAX-native Qwen3-Embedding (bidirectional transformer).
            Loads weights via models/qwen3.py, shards the batch across ALL
            TPU chips via FSDP (P('data', None)).

Retrieval : JAX sharded matmul + approx_max_k across all TPU chips.

Pipeline
--------
  1. Load Qwen3-Embedding model into JAX (bidirectional, left-pad, last-token pool)
  2. Encode documents  → doc_embeddings.npy
  3. Encode queries    → query_embeddings.npy
  4. JAX brute-force retrieval
  5. Compute Recall@K / MRR, save results.json

Requirements
------------
  jax[tpu], transformers>=4.51.0, datasets, safetensors, huggingface_hub, ml_dtypes
"""

import argparse
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


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description="Baseline single-embedding RAG — JAX-native encoding + JAX retrieval"
    )

    # Document corpus
    p.add_argument("--doc_dataset", required=True)
    p.add_argument("--doc_split", default="train")
    p.add_argument("--doc_column", required=True)
    p.add_argument("--max_docs", type=int, default=5_000_000)

    # Query dataset
    p.add_argument("--query_dataset", required=True)
    p.add_argument("--query_split", default="test")
    p.add_argument("--query_column", required=True)
    p.add_argument("--query_gt_column", required=True,
                   help="Column with ground-truth doc(s)")
    p.add_argument("--query_answer_column", default="original_answer",
                   help="Column with the text answer to the query")
    p.add_argument("--num_queries", type=int, default=1000)

    # Encoder model
    p.add_argument("--model_name", required=True,
                   help="HF model ID (e.g. Qwen/Qwen3-Embedding-0.6B)")
    p.add_argument("--hf_ckpt_dir", default="~/weights/huggingface",
                   help="Local dir where HF checkpoints are cached / downloaded")
    p.add_argument("--tp_devices", type=int, default=1,
                   help="Tensor-parallel degree (remaining chips used for FSDP)")
    p.add_argument("--max_doc_length", type=int, default=1024,
                   help="Max tokens per document (fixed padding length for TPU)")
    p.add_argument("--max_query_length", type=int, default=512,
                   help="Max tokens per query (fixed padding length for TPU)")
    p.add_argument("--doc_prefix", default="")
    p.add_argument("--query_prefix", default="",
                   help="Flat prefix prepended to each query (ignored if --query_task is set)")
    p.add_argument("--query_task", default="",
                   help="If set, format each query as "
                        "'Instruct: {query_task}\\nQuery:{query}' "
                        "(recommended for Qwen3-Embedding)")
    p.add_argument("--encode_batch_size", type=int, default=512,
                   help="Texts per JAX forward pass (will be rounded to a multiple of n_devices)")

    # Retrieval
    p.add_argument("--search_batch_size", type=int, default=1024)
    p.add_argument("--top_k", type=int, default=100)

    # I/O
    p.add_argument("--embeddings_dir", default="./embeddings_cache")
    p.add_argument("--output", default="results.json")

    return p


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def normalize_ground_truth(gt) -> list[str]:
    if gt is None:
        return []
    if isinstance(gt, str):
        return [gt]
    if isinstance(gt, (int, float)):
        return [str(int(gt))]
    if isinstance(gt, list):
        out = []
        for item in gt:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.append(item.get("text", item.get("content", str(item))))
            else:
                out.append(str(item))
        return out
    return [str(gt)]


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 0: Build JAX encoder
# ═══════════════════════════════════════════════════════════════════════════

def build_encoder(args):
    """
    Load Qwen3-Embedding into JAX with bidirectional attention.

    Returns (tokenizer, jit_embed_fn) where:
      jit_embed_fn(input_ids [B,T], pad_mask [B,T]) -> float32 [B, D]
    Both inputs must be JAX arrays sharded as P('data', None).
    Batch size B must be a multiple of jax.device_count().
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    # models/ lives one level above Baselines/
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from models.qwen3 import load as load_qwen3

    n_dev = jax.device_count()
    log.info(f"JAX: {n_dev}× {jax.devices()[0].platform.upper()}")
    log.info(f"Loading {args.model_name} (tp={args.tp_devices}, fsdp={n_dev // args.tp_devices})...")
    t0 = time.time()

    model = load_qwen3(
        model_id=args.model_name,
        tp_devices=args.tp_devices,
        load_weights=True,
        hf_ckpt_dir=args.hf_ckpt_dir,
        mask_type="causal",   # Qwen3-Embedding is a causal decoder (same as base Qwen3)
    )
    log.info(f"Model loaded in {time.time()-t0:.1f}s")

    # Left-padding: last real token is always at position -1 (the EOS)
    model.tokenizer.padding_side = "left"
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id

    weights = model.weights
    fwd     = model.forward   # partial(forward, cfg) — takes (input_ids, weights, ...)

    @jax.jit
    def embed_fn(input_ids, pad_mask):
        # hidden: [B, T, D]  bfloat16
        hidden = fwd(input_ids, weights, pad_mask=pad_mask, return_hidden=True)
        # Last-token pool: with left-padding the EOS is always at position -1
        emb = hidden[:, -1, :].astype(jnp.float32)       # [B, D]
        # L2 normalize
        norm = jnp.linalg.norm(emb, axis=-1, keepdims=True)
        emb  = emb / (norm + 1e-12)
        return emb  # [B, D] float32

    # Warm up JIT for both doc and query sequence lengths so first real batch
    # doesn't pay compilation overhead.
    log.info("Compiling embed kernels (doc + query lengths)...")
    for seq_len in sorted({args.max_doc_length, args.max_query_length}):
        dummy_ids  = jax.device_put(
            jnp.zeros((n_dev, seq_len), dtype=jnp.int32), P("data", None))
        dummy_mask = jax.device_put(
            jnp.ones((n_dev, seq_len),  dtype=jnp.bool_), P("data", None))
        jax.block_until_ready(embed_fn(dummy_ids, dummy_mask))
        log.info(f"  compiled seq_len={seq_len} ✓")

    return model.tokenizer, embed_fn


def encode_texts(texts, tokenizer, embed_fn, max_length, batch_size, desc=""):
    """
    Encode a list of strings with the JAX embed_fn.

    Uses padding='max_length' so every batch has identical shape → no JIT
    recompilation after the initial warmup.  Returns float32 numpy (N, D).
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from tqdm import tqdm

    n_dev = jax.device_count()
    # Round batch_size to a multiple of n_dev (batch is sharded along FSDP axis)
    batch_size = max(n_dev, (batch_size // n_dev) * n_dev)

    results = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc or "Encoding", unit="batch"):
        chunk    = list(texts[start : start + batch_size])
        actual_b = len(chunk)

        # Pad to multiple of n_dev with a copy of the last item
        rem = actual_b % n_dev
        if rem:
            chunk += [chunk[-1]] * (n_dev - rem)

        tok = tokenizer(
            chunk,
            padding="max_length",   # fixed shape → TPU-friendly, no recompilation
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )

        ids_jax  = jax.device_put(
            jnp.array(tok["input_ids"],      dtype=jnp.int32), P("data", None))
        mask_jax = jax.device_put(
            jnp.array(tok["attention_mask"], dtype=jnp.bool_), P("data", None))

        embs = embed_fn(ids_jax, mask_jax)
        jax.block_until_ready(embs)
        results.append(np.array(embs)[:actual_b])

    return np.concatenate(results, axis=0)  # float32


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1a: ENCODE DOCUMENTS
# ═══════════════════════════════════════════════════════════════════════════

def run_encode_docs(args, tokenizer, embed_fn):
    from datasets import load_dataset

    cache_dir = Path(args.embeddings_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path  = cache_dir / "doc_embeddings.npy"
    txt_path  = cache_dir / "doc_texts.jsonl"
    meta_path = cache_dir / "encode_meta.json"

    if emb_path.exists() and txt_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        log.info(f"Cache hit: {meta['num_docs']:,} × {meta['embed_dim']}d — skipping doc encode")
        return

    log.info(f"Streaming docs from {args.doc_dataset} [{args.doc_split}]...")
    ds = load_dataset(args.doc_dataset, split=args.doc_split, streaming=True)
    all_texts: list[str] = []
    for example in ds:
        if len(all_texts) >= args.max_docs:
            break
        text = example.get(args.doc_column, "")
        if not text:
            continue
        all_texts.append(args.doc_prefix + str(text))
        if len(all_texts) % 500_000 == 0:
            log.info(f"  streamed {len(all_texts):,} docs...")

    total_docs = len(all_texts)
    log.info(f"Streamed {total_docs:,} documents — encoding on TPU...")

    t0 = time.time()
    all_embs = encode_texts(
        all_texts, tokenizer, embed_fn,
        max_length=args.max_doc_length,
        batch_size=args.encode_batch_size,
        desc="Docs",
    )
    elapsed   = time.time() - t0
    embed_dim = all_embs.shape[1]
    log.info(
        f"Encoded {total_docs:,} docs in {elapsed:.0f}s "
        f"({total_docs/elapsed:,.0f} docs/s) — dim={embed_dim}"
    )

    np.save(emb_path, all_embs.astype(ml_dtypes.bfloat16))

    log.info("Saving document texts...")
    with open(txt_path, "w", encoding="utf-8") as f:
        for t in all_texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    meta_path.write_text(json.dumps({
        "num_docs": total_docs, "embed_dim": embed_dim,
        "model": args.model_name, "max_doc_length": args.max_doc_length,
    }, indent=2))
    log.info(f"Saved {total_docs:,} × {embed_dim} ({all_embs.nbytes/1e9:.1f} GB) → {emb_path}")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1b: ENCODE QUERIES
# ═══════════════════════════════════════════════════════════════════════════

def run_encode_queries(args, tokenizer, embed_fn):
    from datasets import load_dataset

    cache_dir  = Path(args.embeddings_dir)
    qemb_path  = cache_dir / "query_embeddings.npy"
    qmeta_path = cache_dir / "query_meta.jsonl"

    if qemb_path.exists() and qmeta_path.exists():
        log.info("Query embeddings cached — skipping query encode")
        return

    log.info(f"Streaming queries from {args.query_dataset} [{args.query_split}]...")
    ds = load_dataset(args.query_dataset, split=args.query_split, streaming=True)
    all_texts: list[str] = []
    all_raw_texts: list[str] = []
    all_gt:    list[list[str]] = []
    all_answers: list[str] = []
    for example in ds:
        if len(all_texts) >= args.num_queries:
            break
        text = example.get(args.query_column, "")
        if not text:
            continue
        # Qwen3-Embedding instruction format (recommended) or flat prefix
        if args.query_task:
            formatted = f"Instruct: {args.query_task}\nQuery:{text}"
        else:
            formatted = args.query_prefix + str(text)
        all_texts.append(formatted)
        all_raw_texts.append(str(text))
        all_gt.append(normalize_ground_truth(example.get(args.query_gt_column)))
        all_answers.append(str(example.get(args.query_answer_column, "")))

    total_queries = len(all_texts)
    log.info(f"Streamed {total_queries} queries — encoding on TPU...")

    all_embs = encode_texts(
        all_texts, tokenizer, embed_fn,
        max_length=args.max_query_length,
        batch_size=args.encode_batch_size,
        desc="Queries",
    )
    np.save(qemb_path, all_embs.astype(ml_dtypes.bfloat16))

    with open(qmeta_path, "w", encoding="utf-8") as f:
        for raw_text, gt, ans in zip(all_raw_texts, all_gt, all_answers):
            f.write(json.dumps({"query": raw_text, "ground_truth": gt, "answer": ans}, ensure_ascii=False) + "\n")

    log.info(f"Saved {total_queries} query embeddings → {qemb_path}")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: RETRIEVAL  (JAX on TPU, all chips)
# ═══════════════════════════════════════════════════════════════════════════

def run_retrieve(args):
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    devices   = jax.devices()
    n_devices = len(devices)
    log.info(f"JAX: {n_devices}× {devices[0].platform.upper()}")
    assert devices[0].platform == "tpu", f"Expected TPU, got {devices[0].platform}"

    cache_dir = Path(args.embeddings_dir)

    log.info("Loading cached embeddings...")
    t0 = time.time()
    doc_embs   = np.load(cache_dir / "doc_embeddings.npy").view(ml_dtypes.bfloat16)
    query_embs = np.load(cache_dir / "query_embeddings.npy").view(ml_dtypes.bfloat16)
    meta       = json.loads((cache_dir / "encode_meta.json").read_text())
    num_docs, embed_dim = doc_embs.shape
    num_queries = query_embs.shape[0]
    log.info(
        f"Loaded in {time.time()-t0:.1f}s: "
        f"docs={num_docs:,}×{embed_dim}, queries={num_queries}"
    )

    query_metas: list[dict] = []
    with open(cache_dir / "query_meta.jsonl") as f:
        for line in f:
            query_metas.append(json.loads(line))

    # ── Memory budget check ──
    per_chip_db     = (num_docs * embed_dim * 2) / n_devices
    per_chip_scores = args.search_batch_size * (num_docs / n_devices) * 2
    total_per_chip  = per_chip_db + per_chip_scores
    hbm_gb = 32.0
    log.info(
        f"Per-chip HBM: DB={per_chip_db/1e9:.2f}GB + "
        f"Scores={per_chip_scores/1e9:.2f}GB = "
        f"{total_per_chip/1e9:.2f}GB / {hbm_gb}GB"
    )
    if total_per_chip / 1e9 > hbm_gb * 0.85:
        log.warning("Tight on HBM — consider reducing --search_batch_size or --max_docs")

    # ── Replicate database; shard queries across chips ──
    # approx_max_k only works correctly when scores are NOT column-sharded.
    # Strategy: replicate the full DB on every chip, shard the query batch
    # along the batch axis (P("d", None)). Each chip then scores B/d queries
    # against the full database and runs approx_max_k independently.
    # HBM budget: num_docs * embed_dim * 2 bytes per chip — fine up to ~50M docs
    # on 32 GB chips.  For larger corpora switch to per-shard top-k + merge.
    mesh        = Mesh(np.array(devices), axis_names=("d",))
    jax.set_mesh(mesh)                             # override any mesh left by the encoder
    db_sharding = NamedSharding(mesh, P())         # replicated on every chip
    q_sharding  = NamedSharding(mesh, P("d", None))  # queries split across chips

    # Pad query batch size to a multiple of n_devices
    B = args.search_batch_size
    if B % n_devices:
        B = ((B + n_devices - 1) // n_devices) * n_devices

    log.info("Transferring database to TPU (replicated)...")
    t0     = time.time()
    db_jax = jax.device_put(doc_embs, db_sharding)
    jax.block_until_ready(db_jax)
    log.info(f"Database on TPU in {time.time()-t0:.1f}s")
    del doc_embs

    # ── Compile search kernel ──
    top_k = args.top_k

    @jax.jit
    def search_fn(queries, database):
        # queries: [B/d, D] per chip  — database: [N, D] replicated
        scores = queries @ database.T           # [B/d, N] per chip
        top_scores, top_indices = lax.approx_max_k(scores, k=top_k)
        return top_scores.astype(jnp.float32), top_indices

    log.info("Compiling search kernel (JIT warmup)...")
    dummy = jax.device_put(np.zeros((B, embed_dim), dtype=ml_dtypes.bfloat16), q_sharding)
    jax.block_until_ready(search_fn(dummy, db_jax))
    log.info("Compiled ✓")

    # ── Search all queries ──
    log.info(f"Searching {num_queries} queries (batch={B}, top_k={top_k})...")
    all_scores, all_indices = [], []
    t0 = time.time()

    for start in range(0, num_queries, B):
        end    = min(start + B, num_queries)
        batch  = query_embs[start:end]
        actual = batch.shape[0]
        if actual < B:
            batch = np.concatenate(
                [batch, np.zeros((B - actual, embed_dim), dtype=batch.dtype)], axis=0
            )
        q_jax = jax.device_put(jnp.array(batch), q_sharding)
        scores, indices = search_fn(q_jax, db_jax)
        jax.block_until_ready(scores)
        all_scores.append(np.array(scores)[:actual])
        all_indices.append(np.array(indices)[:actual])

    all_scores  = np.concatenate(all_scores,  axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    all_indices = np.clip(all_indices, 0, num_docs - 1)

    elapsed = time.time() - t0
    log.info(
        f"Retrieval: {num_queries} queries in {elapsed:.2f}s "
        f"({num_queries/elapsed:,.0f} QPS)"
    )

    # ── Load retrieved doc texts ──
    unique_ids = set(all_indices.flatten().tolist())
    log.info(f"Loading {len(unique_ids):,} unique doc texts...")
    doc_texts: dict[int, str] = {}
    with open(cache_dir / "doc_texts.jsonl") as f:
        for idx, line in enumerate(f):
            if idx in unique_ids:
                doc_texts[idx] = json.loads(line)["text"]
            if len(doc_texts) == len(unique_ids):
                break

    # ── Assemble results ──
    results = []
    for i in range(num_queries):
        retrieved = [
            {
                "rank":      rank + 1,
                "doc_index": int(all_indices[i, rank]),
                "score":     round(float(all_scores[i, rank]), 6),
                "document":  doc_texts.get(int(all_indices[i, rank]), f"[doc_{all_indices[i,rank]}]"),
            }
            for rank in range(top_k)
        ]
        results.append({
            "query":        query_metas[i]["query"],
            "ground_truth":       query_metas[i].get("answer", ""),
            "ground_truth_doc": query_metas[i]["ground_truth"],
            "retrieved":    retrieved,
        })

    # ── Metrics ──
    k_values = sorted({k for k in [1, 5, 10, 20, 50, top_k] if k <= top_k})
    metrics: dict[str, float] = {}

    for k in k_values:
        hits = total = 0
        for r in results:
            if not r["ground_truth_doc"]:
                continue
            ret = {d["document"] for d in r["retrieved"][:k]}
            if any(g in ret for g in r["ground_truth_doc"]):
                hits += 1
            total += 1
        metrics[f"recall@{k}"] = round(hits / total, 4) if total else 0.0

    rr_sum = rr_n = 0
    for r in results:
        if not r["ground_truth_doc"]:
            continue
        gt_set = set(r["ground_truth_doc"])
        rr_n += 1
        for rank, doc in enumerate(r["retrieved"], 1):
            if doc["document"] in gt_set:
                rr_sum += 1.0 / rank
                break
    metrics["mrr"] = round(rr_sum / rr_n, 4) if rr_n else 0.0

    log.info("─── Metrics ───")
    for k, v in metrics.items():
        log.info(f"  {k:>12s}: {v:.4f}")

    # ── Save ──
    output_meta = {
        "config": {
            "model":             args.model_name,
            "max_doc_length":    args.max_doc_length,
            "max_query_length":  args.max_query_length,
            "num_docs":          num_docs,
            "embed_dim":         embed_dim,
            "num_queries":       num_queries,
            "top_k":             top_k,
            "search_batch_size": B,
            "doc_dataset":       args.doc_dataset,
            "query_dataset":     args.query_dataset,
        },
        "metrics": metrics,
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_meta, f, ensure_ascii=False, indent=2)
        
    results_file = str(Path(args.output).with_suffix("")) + "_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    log.info(f"Metadata saved → {args.output}")
    log.info(f"Results saved → {results_file}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args   = parser.parse_args()

    log.info("=" * 65)
    log.info("Baseline Single-Embedding RAG — JAX native (no vLLM)")
    log.info("=" * 65)
    log.info(f"Model:      {args.model_name}")
    log.info(f"Docs:       {args.doc_dataset} [{args.doc_split}] (max {args.max_docs:,})")
    log.info(f"Queries:    {args.query_dataset} [{args.query_split}] (n={args.num_queries})")
    log.info(f"Top-K:      {args.top_k}")
    log.info(f"Cache:      {args.embeddings_dir}")
    log.info(f"Output:     {args.output}")

    cache_dir  = Path(args.embeddings_dir)
    docs_done  = (cache_dir / "doc_embeddings.npy").exists()
    query_done = (cache_dir / "query_embeddings.npy").exists()

    t_total = time.time()

    # ── Phases 1a + 1b: encode (JAX, TPU) ──
    if not docs_done or not query_done:
        log.info("\n╔══ Building JAX encoder ══╗")
        tokenizer, embed_fn = build_encoder(args)

        if not docs_done:
            log.info("\n╔══ Phase 1a: Encode documents ══╗")
            run_encode_docs(args, tokenizer, embed_fn)

        if not query_done:
            log.info("\n╔══ Phase 1b: Encode queries ══╗")
            run_encode_queries(args, tokenizer, embed_fn)
    else:
        log.info("Both embedding caches found — skipping encoding phases")

    # ── Phase 2: retrieval (JAX, no model needed) ──
    log.info("\n╔══ Phase 2: Retrieval (JAX sharded matmul) ══╗")
    run_retrieve(args)

    elapsed = time.time() - t_total
    log.info(f"\nTotal pipeline time: {elapsed/60:.1f} minutes")
    log.info(f"Results: {args.output}")


if __name__ == "__main__":
    main()
