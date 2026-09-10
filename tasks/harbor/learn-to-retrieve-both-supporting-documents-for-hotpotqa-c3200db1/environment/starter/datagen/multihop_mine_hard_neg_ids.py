#!/usr/bin/env python3
"""
Stage 2 of multihop hard-negative mining: retrieve hard negatives as IDs.

For every source question: embed the question, retrieve the nearest --top-k docs from the
Stage-1 embedding matrix, drop every doc belonging to that question's own row, and keep
the first --n-negatives survivors.

Retrieval is a brute-force dense matmul (queries @ docs.T) followed by
jax.lax.approx_max_k, with the doc matrix REPLICATED across chips and queries sharded.
That combination is the measured-fastest top-k on TPU in this repo
(results/MEMORY_SCAN_ANN_FINDINGS.md): gather-based ANN (IVF/PQ) loses by 1-2 orders of
magnitude because XLA cannot lower a data-dependent gather to contiguous DMA. It also
matches TPU-KNN (arXiv 2206.14286).

Why the exclusion set is the whole row, not just one doc: every paragraph of a source row
is supporting evidence for that row's question, so any of them would be a false negative.

Output is IDs only (row_id, pos_doc_ids, neg_doc_ids). Text is never materialized here --
denormalizing 200 negatives x ~5.4M exploded rows would be ~1.5 TB against a 5.4 GB
source. Shards are written resumably and verified for pos/neg disjointness.

Usage:
    python datagen/multihop_mine_hard_neg_ids.py --max-rows 2000    # smoke test
    python datagen/multihop_mine_hard_neg_ids.py                    # full run
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import ml_dtypes
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

CORPUS_DATASET = "mihir-1999/multihop_qa_sft-doc-corpus"
QUERY_TASK = "Given a web search query, retrieve relevant passages that answer the query"


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-dataset", default=CORPUS_DATASET)
    p.add_argument("--rows-config", default="rows")
    p.add_argument("--split", default="train")
    p.add_argument("--emb-dir", default="data/cache/multihop_doc_embeddings")
    p.add_argument("--model-name", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--hf-ckpt-dir", default="~/weights/huggingface")
    p.add_argument("--tp-devices", type=int, default=1)
    p.add_argument("--max-doc-length", type=int, default=512)
    p.add_argument("--max-query-length", type=int, default=128)
    p.add_argument("--encode-batch-size", type=int, default=512)
    p.add_argument("--search-batch-size", type=int, default=2048)
    p.add_argument("--top-k", type=int, default=220,
                   help="Over-fetch before removing the row's own docs.")
    p.add_argument("--n-negatives", type=int, default=200)
    p.add_argument("--query-task", default=QUERY_TASK)
    p.add_argument("--shard-rows", type=int, default=200_000)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--out-dir", default="data/cache/multihop_hard_neg_ids")
    return p


def load_embeddings(emb_dir):
    """Concatenate Stage-1 shards into one (N, D) bfloat16 matrix."""
    emb_dir = Path(emb_dir)
    manifest = json.loads((emb_dir / "manifest.json").read_text())
    parts = []
    for ci in range(manifest["n_chunks"]):
        arr = np.load(emb_dir / f"doc_emb_{ci:05d}.npy")
        if arr.dtype != ml_dtypes.bfloat16:
            arr = arr.view(ml_dtypes.bfloat16)
        parts.append(arr)
    db = np.concatenate(parts, axis=0)
    if db.shape[0] != manifest["n_docs"]:
        raise ValueError(f"embedding rows {db.shape[0]:,} != manifest n_docs {manifest['n_docs']:,}")
    log.info(f"doc matrix: {db.shape[0]:,} x {db.shape[1]} bf16 ({db.nbytes/1e9:.2f} GB)")
    return db


def load_rows(args):
    from datasets import load_dataset
    log.info(f"loading rows {args.corpus_dataset} [{args.rows_config}]")
    ds = load_dataset(args.corpus_dataset, args.rows_config, split=args.split)
    if args.max_rows is not None:
        ds = ds.select(range(min(args.max_rows, len(ds))))
    log.info(f"rows: {len(ds):,}")
    return ds


def build_search_fn(db, args):
    """Replicate the doc matrix, shard queries, jit the matmul + approx_max_k."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import NamedSharding, PartitionSpec as P, AxisType

    n_dev = jax.device_count()
    fsdp = n_dev // args.tp_devices
    mesh = jax.make_mesh((fsdp, args.tp_devices), ("data", "model"),
                         axis_types=(AxisType.Explicit, AxisType.Explicit))
    jax.set_mesh(mesh)

    db_sharding = NamedSharding(mesh, P())              # replicated on every chip
    q_sharding = NamedSharding(mesh, P("data", None))   # batch-sharded

    log.info(f"transferring doc matrix to {n_dev} chip(s), replicated...")
    t0 = time.time()
    db_jax = jax.device_put(db, db_sharding)
    jax.block_until_ready(db_jax)
    log.info(f"  on device in {time.time()-t0:.1f}s")

    top_k = args.top_k

    @jax.jit
    def search_fn(queries, database):
        scores = queries @ database.T
        # approx_max_k is correct only when scores are not column-sharded, which the
        # replicated database guarantees.
        top_scores, top_idx = lax.approx_max_k(scores, k=top_k)
        return top_scores.astype(jnp.float32), top_idx

    B = args.search_batch_size
    if B % n_dev:
        B = ((B + n_dev - 1) // n_dev) * n_dev
    dummy = jax.device_put(np.zeros((B, db.shape[1]), dtype=ml_dtypes.bfloat16), q_sharding)
    jax.block_until_ready(search_fn(dummy, db_jax))
    log.info(f"search kernel compiled (batch={B}, top_k={top_k})")
    return db_jax, search_fn, B, q_sharding


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import jax
    import pyarrow as pa
    import pyarrow.parquet as pq
    from types import SimpleNamespace
    from evals.rag.single_embedding_retrieval import build_encoder, encode_texts

    db = load_embeddings(args.emb_dir)
    ds = load_rows(args)
    n_rows = len(ds)

    enc_args = SimpleNamespace(
        model_name=args.model_name, tp_devices=args.tp_devices,
        hf_ckpt_dir=args.hf_ckpt_dir, max_doc_length=args.max_doc_length,
        max_query_length=args.max_query_length,
    )
    tokenizer, embed_fn = build_encoder(enc_args)
    db_jax, search_fn, search_B, q_sharding = build_search_fn(db, args)
    embed_dim = db.shape[1]
    n_corpus = db.shape[0]
    del db

    n_shards = (n_rows + args.shard_rows - 1) // args.shard_rows
    log.info(f"mining {n_rows:,} rows in {n_shards} shard(s)")

    stats = {"rows": 0, "short": 0, "overlap": 0, "min_negs": args.n_negatives,
             "pos_out_of_range": 0}
    t_start = time.time()

    for si in range(n_shards):
        shard_path = out_dir / f"hard_neg_ids_{si:05d}.parquet"
        lo, hi = si * args.shard_rows, min((si + 1) * args.shard_rows, n_rows)
        if shard_path.exists():
            log.info(f"  shard {si:05d} present — skipping")
            continue

        t0 = time.time()
        sub = ds.select(range(lo, hi))
        row_ids = list(sub["row_id"])
        questions = list(sub["question"])
        pos_lists = [list(p) for p in sub["pos_doc_ids"]]

        formatted = [f"Instruct: {args.query_task}\nQuery:{q}" for q in questions]
        q_embs = encode_texts(formatted, tokenizer, embed_fn,
                              max_length=args.max_query_length,
                              batch_size=args.encode_batch_size,
                              desc=f"encode q shard {si:05d}")

        # Retrieve in fixed-size batches so the jitted kernel never recompiles.
        all_idx = []
        for start in range(0, len(q_embs), search_B):
            chunk = q_embs[start:start + search_B].astype(ml_dtypes.bfloat16)
            actual = chunk.shape[0]
            if actual < search_B:
                chunk = np.concatenate(
                    [chunk, np.zeros((search_B - actual, embed_dim), dtype=chunk.dtype)], axis=0)
            q_jax = jax.device_put(chunk, q_sharding)
            _, idx = search_fn(q_jax, db_jax)
            jax.block_until_ready(idx)
            all_idx.append(np.asarray(idx)[:actual])
        all_idx = np.concatenate(all_idx, axis=0)

        neg_col = []
        for i in range(len(row_ids)):
            pos = pos_lists[i]
            bad = [p for p in pos if not (0 <= p < n_corpus)]
            if bad:
                stats["pos_out_of_range"] += 1
            pos_set = set(pos)
            negs = []
            for cand in all_idx[i]:
                c = int(cand)
                if c in pos_set or not (0 <= c < n_corpus):
                    continue
                negs.append(c)
                if len(negs) >= args.n_negatives:
                    break
            if len(negs) < args.n_negatives:
                stats["short"] += 1
                stats["min_negs"] = min(stats["min_negs"], len(negs))
            if pos_set & set(negs):
                stats["overlap"] += 1
            neg_col.append(negs)
            stats["rows"] += 1

        tmp = shard_path.with_suffix(".parquet.tmp")
        pq.write_table(
            pa.table({
                "row_id": pa.array(row_ids, pa.int32()),
                "pos_doc_ids": pa.array(pos_lists, pa.list_(pa.int32())),
                "neg_doc_ids": pa.array(neg_col, pa.list_(pa.int32())),
            }), tmp)
        tmp.rename(shard_path)

        dt = time.time() - t0
        done = hi
        rate = (hi - lo) / dt
        log.info(f"  shard {si:05d}: {hi-lo:,} rows in {dt:,.1f}s ({rate:,.0f} rows/s) "
                 f"| eta {(n_rows-done)/rate/60:,.1f} min")

    (out_dir / "mine_stats.json").write_text(json.dumps(stats, indent=2))
    total = time.time() - t_start

    print("\n" + "=" * 62)
    print("STAGE 2 — hard-negative IDs")
    print("=" * 62)
    print(f"  rows mined          : {stats['rows']:,}")
    print(f"  corpus docs         : {n_corpus:,}")
    print(f"  negatives per row   : {args.n_negatives} (over-fetched top-{args.top_k})")
    print(f"  rows w/ pos∩neg ≠ ∅ : {stats['overlap']}   <- must be 0")
    print(f"  rows short of {args.n_negatives:<4}  : {stats['short']:,}"
          + (f" (min {stats['min_negs']})" if stats["short"] else ""))
    print(f"  pos ids out of range: {stats['pos_out_of_range']}")
    print(f"  wall time           : {total/60:,.1f} min")
    print(f"  wrote               : {out_dir}/hard_neg_ids_*.parquet")
    if stats["overlap"]:
        log.error("pos/neg overlap detected — this must be zero; failing")
        return 1
    return 0


if __name__ == "__main__":
    import threading
    threading.stack_size(512 * 1024 * 1024)
    rc = []
    t = threading.Thread(target=lambda: rc.append(main()))
    t.start()
    t.join()
    sys.exit(rc[0] if rc else 1)
