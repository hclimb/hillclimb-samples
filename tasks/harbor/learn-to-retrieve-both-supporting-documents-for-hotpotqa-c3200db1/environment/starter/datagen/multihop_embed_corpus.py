#!/usr/bin/env python3
"""
Stage 1 of multihop hard-negative mining: embed the deduplicated doc corpus.

Reads the Stage-0 corpus (one row per unique doc: doc_id, text) from HF, encodes every
doc with Qwen3-Embedding via the JAX encoder already used by the RAG evals
(evals/rag/single_embedding_retrieval.py::build_encoder/encode_texts), and writes
bfloat16 embedding shards plus a manifest.

Resumable: shards already on disk are skipped, so a preempted or interrupted run picks
up where it left off. Row i of the assembled matrix is doc_id i (asserted below), so
retrieval indices are doc_ids directly with no extra mapping.

Usage:
    python datagen/multihop_embed_corpus.py --max-docs 20000     # benchmark slice
    python datagen/multihop_embed_corpus.py                      # full corpus
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


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-dataset", default=CORPUS_DATASET)
    p.add_argument("--corpus-config", default="corpus")
    p.add_argument("--split", default="train")
    p.add_argument("--model-name", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--hf-ckpt-dir", default="~/weights/huggingface")
    p.add_argument("--tp-devices", type=int, default=1)
    p.add_argument("--max-doc-length", type=int, default=512)
    p.add_argument("--max-query-length", type=int, default=128)
    p.add_argument("--encode-batch-size", type=int, default=512)
    p.add_argument("--chunk-size", type=int, default=100_000,
                   help="Docs per saved shard (resume granularity).")
    p.add_argument("--max-docs", type=int, default=None,
                   help="Cap docs (benchmark). Default: whole corpus.")
    p.add_argument("--out-dir", default="data/cache/multihop_doc_embeddings")
    return p


def load_corpus(args):
    """Open the corpus and verify row order == doc_id, WITHOUT materializing text.

    Returns (dataset, n_docs). Text is pulled per chunk in main() instead of all at
    once: materializing 1.29M multi-KB strings up front costs GBs of RAM and minutes of
    Arrow->Python conversion before the TPU does any work at all.
    """
    from datasets import load_dataset

    log.info(f"loading corpus {args.corpus_dataset} [{args.corpus_config}]")
    ds = load_dataset(args.corpus_dataset, args.corpus_config, split=args.split)

    # Only the id column is materialized here (ints, cheap).
    doc_ids = np.asarray(ds["doc_id"], dtype=np.int64)
    n_total = len(doc_ids)

    # Retrieval returns matrix row indices; they are only usable as doc_ids if row i is
    # doc_id i. Stage 0 writes them contiguous and in order — verify, don't reorder.
    if not np.array_equal(doc_ids, np.arange(n_total)):
        raise ValueError(
            f"corpus rows are not doc_id 0..N-1 in order (min={doc_ids.min()}, "
            f"max={doc_ids.max()}, n={n_total}); retrieval indices would not be doc_ids. "
            f"Re-export the corpus sorted by doc_id."
        )

    n_docs = n_total if args.max_docs is None else min(args.max_docs, n_total)
    log.info(f"corpus: {n_total:,} docs (using {n_docs:,}), row index == doc_id verified")
    return ds, n_docs


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds, n_docs = load_corpus(args)

    from types import SimpleNamespace
    from evals.rag.single_embedding_retrieval import build_encoder, encode_texts

    enc_args = SimpleNamespace(
        model_name=args.model_name,
        tp_devices=args.tp_devices,
        hf_ckpt_dir=args.hf_ckpt_dir,
        max_doc_length=args.max_doc_length,
        max_query_length=args.max_query_length,
    )
    tokenizer, embed_fn = build_encoder(enc_args)

    n_chunks = (n_docs + args.chunk_size - 1) // args.chunk_size
    log.info(f"encoding {n_docs:,} docs in {n_chunks} chunk(s) of {args.chunk_size:,}")

    embed_dim = None
    t_start = time.time()
    encoded_this_run = 0

    for ci in range(n_chunks):
        shard = out_dir / f"doc_emb_{ci:05d}.npy"
        lo, hi = ci * args.chunk_size, min((ci + 1) * args.chunk_size, n_docs)

        if shard.exists():
            arr = np.load(shard, mmap_mode="r")
            if arr.shape[0] == (hi - lo):
                embed_dim = arr.shape[1]
                log.info(f"  chunk {ci:05d} present ({arr.shape[0]:,} docs) — skipping")
                continue
            log.warning(f"  chunk {ci:05d} wrong length ({arr.shape[0]} != {hi-lo}) — re-encoding")

        t0 = time.time()
        # Materialize only this chunk's text, not the whole corpus.
        chunk_texts = ds.select(range(lo, hi))["text"]
        embs = encode_texts(chunk_texts, tokenizer, embed_fn,
                            max_length=args.max_doc_length,
                            batch_size=args.encode_batch_size,
                            desc=f"chunk {ci:05d}")
        del chunk_texts
        embed_dim = embs.shape[1]
        # Write to a temp path then rename: a crash mid-write must not leave a
        # truncated shard that a resume would trust.
        # np.save APPENDS '.npy' when the path does not already end in it, so passing
        # 'foo.npy.tmp' silently writes 'foo.npy.tmp.npy' and the rename below fails.
        # Hand it an open file object instead — then the name is used verbatim.
        tmp = shard.with_name(shard.stem + ".tmp.npy")
        with open(tmp, "wb") as fh:
            np.save(fh, embs.astype(ml_dtypes.bfloat16))
        tmp.rename(shard)

        dt = time.time() - t0
        encoded_this_run += (hi - lo)
        rate = (hi - lo) / dt
        remaining = n_docs - hi
        log.info(f"  chunk {ci:05d}: {hi-lo:,} docs in {dt:,.1f}s ({rate:,.0f} docs/s) "
                 f"| eta {remaining/rate/60:,.1f} min for remaining {remaining:,}")

    manifest = {
        "corpus_dataset": args.corpus_dataset,
        "model_name": args.model_name,
        "n_docs": n_docs,
        "embed_dim": embed_dim,
        "chunk_size": args.chunk_size,
        "n_chunks": n_chunks,
        "max_doc_length": args.max_doc_length,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = time.time() - t_start
    print("\n" + "=" * 62)
    print("STAGE 1 — corpus embeddings")
    print("=" * 62)
    print(f"  docs            : {n_docs:,}")
    print(f"  embed dim       : {embed_dim}")
    print(f"  encoded this run: {encoded_this_run:,}")
    print(f"  wall time       : {total/60:,.1f} min")
    if encoded_this_run:
        print(f"  throughput      : {encoded_this_run/total:,.0f} docs/s")
    print(f"  wrote           : {out_dir}/doc_emb_*.npy + manifest.json")
    return 0


if __name__ == "__main__":
    import threading
    # MLIR lowering recurses deeply for large models; the main thread's stack cannot be
    # resized after start, but new threads honour threading.stack_size().
    threading.stack_size(512 * 1024 * 1024)
    rc = []
    t = threading.Thread(target=lambda: rc.append(main()))
    t.start()
    t.join()
    sys.exit(rc[0] if rc else 1)
