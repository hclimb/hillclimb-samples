#!/usr/bin/env python3
"""
hard_negative_scorer.py — Score hard negatives with Qwen3-Reranker (JAX-native)
=================================================================================

For every row in the input HF dataset (query / answer / pos_doc / neg_docs),
scores each neg_doc against the query with Qwen3-Reranker-0.6B and writes a
'neg_scores' column in the same order as 'neg_docs'.

Rows are uploaded to HuggingFace in 200K-row parquet shards (resumable).
"""

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SHARD_SIZE = 200_000

# Exact prompt wrappers from the Qwen3-Reranker HF model card
PREFIX = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the Query and the '
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    '<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

DEFAULT_TASK = "Given a web search query, retrieve relevant passages that answer the query"


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description="Score hard negatives with Qwen3-Reranker — JAX-native TPU pipeline"
    )
    p.add_argument("--input_dataset",     required=True,
                   help="HF dataset produced by hard_negative_mining.py")
    p.add_argument("--input_split",       default="train")
    p.add_argument("--model_name",        default="Qwen/Qwen3-Reranker-0.6B")
    p.add_argument("--hf_ckpt_dir",       default="~/weights/huggingface")
    p.add_argument("--tp_devices",        type=int, default=1)
    p.add_argument("--max_length",        type=int, default=1024,
                   help="Max tokens per pair (prefix + content + suffix)")
    p.add_argument("--score_batch_size",  type=int, default=2048,
                   help="Pairs per JAX forward pass (rounded to multiple of n_devices)")
    p.add_argument("--n_queries",         type=int, default=None,
                   help="Cap total rows processed (default: all)")
    p.add_argument("--task",              default=DEFAULT_TASK,
                   help="Reranker instruction string")
    p.add_argument("--output_dir",        default="./scored_negatives")
    p.add_argument("--hf_output_dataset", required=True,
                   help="HF dataset repo to push scored shards to")
    p.add_argument("--hf_token",          default=None)
    return p


# ═══════════════════════════════════════════════════════════════════════════
# RERANKER BUILD
# ═══════════════════════════════════════════════════════════════════════════

def build_reranker(args):
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
    t0 = time.time()
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

    token_true_id  = model.tokenizer.convert_tokens_to_ids("yes")
    token_false_id = model.tokenizer.convert_tokens_to_ids("no")
    log.info(f"token_true_id={token_true_id}  token_false_id={token_false_id}")

    prefix_tokens = model.tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = model.tokenizer.encode(SUFFIX, add_special_tokens=False)
    log.info(f"prefix_len={len(prefix_tokens)}  suffix_len={len(suffix_tokens)}")

    cfg     = model.cfg
    weights = model.weights
    fwd     = model.forward

    # Pre-extract yes/no embedding vectors — avoids an index gather inside JIT
    out_embed_key = "embed_tokens" if cfg.get("tie_word_embeddings", True) else "lm_head"
    yes_emb = weights[out_embed_key].at[token_true_id,  :].get(out_sharding=P()).astype(jnp.float32)  # [D]
    no_emb  = weights[out_embed_key].at[token_false_id, :].get(out_sharding=P()).astype(jnp.float32)  # [D]

    @jax.jit
    def score_fn(input_ids, pad_mask):
        # hidden: [B, T, D]  (post final RMSNorm, return_hidden=True skips logit proj)
        hidden    = fwd(input_ids, weights, pad_mask=pad_mask, return_hidden=True)
        last      = hidden[:, -1, :].astype(jnp.float32)        # [B, D]
        yes_logit = jnp.einsum("bd,d->b", last, yes_emb)        # [B]
        no_logit  = jnp.einsum("bd,d->b", last, no_emb)         # [B]
        stacked   = jnp.stack([no_logit, yes_logit], axis=1)    # [B, 2]
        log_probs = jax.nn.log_softmax(stacked, axis=1)
        return jnp.exp(log_probs[:, 1])                          # P(yes) ∈ (0, 1)

    # Warmup — compile at the exact batch+seq shape we'll use in scoring
    actual_batch = max(n_dev, (args.score_batch_size // n_dev) * n_dev)
    log.info(f"Compiling score kernel (batch={actual_batch}, seq={args.max_length})...")
    dummy_ids  = jax.device_put(jnp.zeros((actual_batch, args.max_length), jnp.int32), P("data", None))
    dummy_mask = jax.device_put(jnp.ones( (actual_batch, args.max_length), jnp.bool_), P("data", None))
    jax.block_until_ready(score_fn(dummy_ids, dummy_mask))
    log.info("Score kernel compiled ✓")

    return model.tokenizer, score_fn, prefix_tokens, suffix_tokens, n_dev


# ═══════════════════════════════════════════════════════════════════════════
# TOKENISATION
# ═══════════════════════════════════════════════════════════════════════════

def _build_input_arrays(texts, tokenizer, prefix_tokens, suffix_tokens, max_length):
    """
    Replicates the HF Qwen3-Reranker process_inputs logic in numpy:
      1. Tokenise content only (truncation on middle content, not prefix/suffix)
      2. Manually prepend prefix_tokens and append suffix_tokens
      3. Left-pad to max_length

    Returns (input_ids [N, max_length] int32, attn_mask [N, max_length] bool).
    """
    content_max = max_length - len(prefix_tokens) - len(suffix_tokens)
    enc = tokenizer(
        texts,
        padding=False,
        truncation="longest_first",
        max_length=content_max,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    pad_id    = tokenizer.pad_token_id
    n         = len(texts)
    input_ids = np.full((n, max_length), pad_id, dtype=np.int32)
    attn_mask = np.zeros((n, max_length), dtype=np.bool_)
    for i, ids in enumerate(enc["input_ids"]):
        seq = prefix_tokens + ids + suffix_tokens
        seq = seq[-max_length:]          # safety trim from the left if still too long
        l   = len(seq)
        input_ids[i, max_length - l:] = seq
        attn_mask[i, max_length - l:] = True
    return input_ids, attn_mask


# ═══════════════════════════════════════════════════════════════════════════
# PAIR SCORING
# ═══════════════════════════════════════════════════════════════════════════

def format_pair(task, query, doc):
    """Matches format_instruction from the HF Qwen3-Reranker model card."""
    return f"<Instruct>: {task}\n<Query>: {query}\n<Document>: {doc}"


def score_pairs(texts, tokenizer, score_fn, prefix_tokens, suffix_tokens,
                max_length, batch_size, n_dev):
    """
    Score a flat list of formatted pair strings.
    Returns list[float] of P(yes) scores in the same order.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from tqdm import tqdm

    if not texts:
        return []

    batch_size = max(n_dev, (batch_size // n_dev) * n_dev)
    input_ids, attn_mask = _build_input_arrays(
        texts, tokenizer, prefix_tokens, suffix_tokens, max_length
    )
    pad_id     = tokenizer.pad_token_id
    all_scores = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Scoring", unit="batch"):
        end    = min(start + batch_size, len(texts))
        actual = end - start

        ids_b  = input_ids[start:end]
        mask_b = attn_mask[start:end]

        # Pad last batch to full batch_size so JIT sees a consistent shape
        if actual < batch_size:
            deficit = batch_size - actual
            ids_b  = np.concatenate([ids_b,  np.full((deficit, max_length), pad_id, dtype=np.int32)], axis=0)
            mask_b = np.concatenate([mask_b, np.zeros((deficit, max_length), dtype=np.bool_)],         axis=0)

        ids_jax  = jax.device_put(jnp.array(ids_b),  P("data", None))
        mask_jax = jax.device_put(jnp.array(mask_b), P("data", None))

        scores = score_fn(ids_jax, mask_jax)
        jax.block_until_ready(scores)
        all_scores.extend(np.array(scores)[:actual].tolist())

    return all_scores


# ═══════════════════════════════════════════════════════════════════════════
# HF UPLOAD
# ═══════════════════════════════════════════════════════════════════════════

def push_scored_shard(records, shard_idx, args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    api = HfApi(token=args.hf_token)
    api.create_repo(repo_id=args.hf_output_dataset, repo_type="dataset",
                    exist_ok=True, private=False)

    # Build columns dynamically so all input fields are preserved.
    # neg_scores gets an explicit float32 list type; everything else is inferred.
    all_keys = list(records[0].keys())
    columns  = {}
    fields   = []
    for key in all_keys:
        if key == "neg_scores":
            continue  # handled below
        columns[key] = [r[key] for r in records]
        fields.append(pa.field(key, pa.array(columns[key]).type))
    columns["neg_scores"] = [[float(s) for s in r["neg_scores"]] for r in records]
    fields.append(pa.field("neg_scores", pa.list_(pa.float32())))

    schema = pa.schema(fields)
    table  = pa.table(columns, schema=schema)

    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)

    t0 = time.time()
    api.upload_file(
        path_or_fileobj=buf,
        path_in_repo=f"data/train-{shard_idx:05d}.parquet",
        repo_id=args.hf_output_dataset,
        repo_type="dataset",
        commit_message=f"shard {shard_idx:05d} ({len(records):,} rows)",
    )
    log.info(f"  pushed shard {shard_idx:05d} in {time.time()-t0:.1f}s → {args.hf_output_dataset}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_scoring(args, tokenizer, score_fn, prefix_tokens, suffix_tokens, n_dev):
    output_dir    = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"

    done_shards: set[int] = set()
    if progress_path.exists():
        done_shards = set(json.loads(progress_path.read_text()).get("done_shards", []))
        log.info(f"Resuming: {len(done_shards)} shard(s) already completed")

    log.info(f"Streaming {args.input_dataset} [{args.input_split}]...")

    # Workaround for PyArrow 23 bug:
    # "Nested data conversions not implemented for chunked array outputs"
    # We bypass datasets streaming by downloading one parquet file at a time,
    # reading it, then deleting it before fetching the next.
    import tempfile
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem, hf_hub_download

    hf_fs = HfFileSystem(token=args.hf_token)
    parquet_files = sorted(hf_fs.glob(
        f"datasets/{args.input_dataset}/data/{args.input_split}-*.parquet"
    ))
    if not parquet_files:
        parquet_files = sorted(hf_fs.glob(
            f"datasets/{args.input_dataset}/**/*.parquet"
        ))
    log.info(f"Found {len(parquet_files)} parquet file(s)")

    def iter_rows():
        tmp_dir = Path(tempfile.mkdtemp(prefix="hns_parquet_"))
        try:
            for fpath in parquet_files:
                # fpath looks like "datasets/owner/name/data/train-00000.parquet"
                parts = fpath.split("/", 3)  # ["datasets","owner","name","data/..."]
                repo_id  = parts[1] + "/" + parts[2]
                filename = parts[3]
                local = tmp_dir / Path(filename).name
                log.info(f"  Fetching {filename} ...")
                hf_hub_download(
                    repo_id=repo_id, repo_type="dataset",
                    filename=filename, token=args.hf_token,
                    local_dir=str(tmp_dir),
                )
                actual = tmp_dir / filename  # hf_hub_download preserves subpath
                pf = pq.ParquetFile(actual, pre_buffer=False)
                for batch in pf.iter_batches(batch_size=1000, use_threads=False):
                    yield from batch.to_pylist()
                # Delete immediately to reclaim disk space
                actual.unlink(missing_ok=True)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    ds = iter_rows()

    shard_records: list[dict] = []
    shard_idx   = 0
    total_rows  = 0
    total_pairs = 0

    def flush_shard(s_idx, records):
        if s_idx in done_shards:
            log.info(f"Shard {s_idx:05d} already done — skipping")
            return

        n = len(records)
        log.info(f"\n╔══ Shard {s_idx:05d}: scoring {n:,} rows ══╗")
        t0 = time.time()

        # Flatten all (query, neg_doc) pairs across the shard
        flat_texts: list[str] = []
        pair_row:   list[int] = []
        pair_pos:   list[int] = []

        for row_i, rec in enumerate(records):
            for pos_j, neg_doc in enumerate(rec["neg_docs"]):
                flat_texts.append(format_pair(args.task, rec["query"], neg_doc))
                pair_row.append(row_i)
                pair_pos.append(pos_j)

        log.info(f"  {len(flat_texts):,} pairs to score")

        flat_scores = score_pairs(
            flat_texts, tokenizer, score_fn,
            prefix_tokens, suffix_tokens,
            args.max_length, args.score_batch_size, n_dev,
        )

        # Reassemble per-row scores in original neg_docs order
        for rec in records:
            rec["neg_scores"] = [0.0] * len(rec["neg_docs"])
        for row_i, pos_j, sc in zip(pair_row, pair_pos, flat_scores):
            records[row_i]["neg_scores"][pos_j] = float(sc)

        elapsed = time.time() - t0
        avg_negs = sum(len(r["neg_docs"]) for r in records) / max(n, 1)
        log.info(
            f"  scored {len(flat_texts):,} pairs in {elapsed:.1f}s  "
            f"(avg neg_docs per row: {avg_negs:.2f})"
        )

        push_scored_shard(records, s_idx, args)
        done_shards.add(s_idx)
        progress_path.write_text(json.dumps({"done_shards": sorted(done_shards)}, indent=2))

    for example in ds:
        if args.n_queries is not None and total_rows >= args.n_queries:
            break

        neg_docs = example.get("neg_docs") or []
        rec = dict(example)
        rec["neg_docs"] = [str(d) for d in neg_docs]
        shard_records.append(rec)
        total_rows  += 1
        total_pairs += len(neg_docs)

        if len(shard_records) == SHARD_SIZE:
            flush_shard(shard_idx, shard_records)
            shard_idx    += 1
            shard_records = []
            log.info(f"Progress: {total_rows:,} rows | {total_pairs:,} pairs scored so far")

    # Flush the final partial shard
    if shard_records:
        flush_shard(shard_idx, shard_records)
        shard_idx += 1

    log.info(
        f"\nScoring complete: {total_rows:,} rows | {total_pairs:,} pairs | {shard_idx} shards"
    )
    return shard_idx


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = build_parser().parse_args()

    log.info("=" * 65)
    log.info("Hard Negative Scorer — JAX-native Qwen3-Reranker pipeline")
    log.info("=" * 65)
    log.info(f"Input:       {args.input_dataset} [{args.input_split}]")
    log.info(f"Model:       {args.model_name}")
    log.info(f"Max length:  {args.max_length}  Batch: {args.score_batch_size}")
    log.info(f"N queries:   {args.n_queries or 'all'}")
    log.info(f"HF output:   {args.hf_output_dataset}")

    tokenizer, score_fn, prefix_tokens, suffix_tokens, n_dev = build_reranker(args)

    t0       = time.time()
    n_shards = run_scoring(args, tokenizer, score_fn, prefix_tokens, suffix_tokens, n_dev)
    log.info(f"\nTotal time: {(time.time()-t0)/60:.1f} min across {n_shards} shards")


if __name__ == "__main__":
    import threading
    # MLIR lowering for large models recurses deeply — needs bigger stack
    threading.stack_size(1 * 1024 * 1024 * 1024)  # 1 GB
    t = threading.Thread(target=main)
    t.start()
    t.join()
