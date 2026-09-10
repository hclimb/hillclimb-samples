"""V2 verification (plan §Verification): the tokenized ArrayRecord samples are
byte-equal to what qa_transform_item produces on the same raw row.

Pulls N random manifest entries. For each:
  1. Reads the corresponding record from the ArrayRecord shard (via
     ArrayRecordDataSource → deserialize).
  2. Re-runs iter_source_rows + qa_filter_predicate + qa_transform_item on
     the SAME (source, raw_row_id).
  3. Asserts field-by-field equality on the returned dict.

Any mismatch means the runtime path and the preprocess path disagree — a
silent data-content divergence between old and new pipelines, which would
invalidate the whole rework's "same tokens, different order" claim.

Requires a completed preprocess output at --indexed-uri.

Run:
  cd $HOME/memory-layers && source .venv/bin/activate
  HF_HUB_OFFLINE=1 GROUND_HF_PARQUET=$HOME/hf_parquet \\
    python scripts/debug/verify_v2_byte_equality.py \\
      --dataset qa_hard_neg_think_sft4b \\
      --tokenizer Qwen/Qwen3-4B \\
      --indexed-uri gs://memory-layers-training/indexed/SMOKE-<hash> \\
      --n-samples 200
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.qa import qa_filter_predicate, qa_transform_item
from data.preprocess_arrayrecord import iter_source_rows, resolve_dataset_cfg


def compare_samples(actual: dict, expected: dict, entry: dict) -> list[str]:
    """Return list of mismatch descriptions (empty = byte-equal)."""
    mismatches = []
    for k in sorted(set(actual.keys()) | set(expected.keys())):
        if k not in actual:
            mismatches.append(f"  [{k}] MISSING in actual")
            continue
        if k not in expected:
            mismatches.append(f"  [{k}] MISSING in expected (transform)")
            continue
        a, e = actual[k], expected[k]
        if isinstance(a, np.ndarray) and isinstance(e, np.ndarray):
            if a.shape != e.shape:
                mismatches.append(f"  [{k}] shape {a.shape} vs {e.shape}")
            elif a.dtype != e.dtype:
                mismatches.append(f"  [{k}] dtype {a.dtype} vs {e.dtype}")
            elif not np.array_equal(a, e):
                n_diff = int(np.sum(a != e))
                mismatches.append(f"  [{k}] {n_diff}/{a.size} values differ")
        elif type(a) != type(e):
            mismatches.append(f"  [{k}] type {type(a).__name__} vs {type(e).__name__}")
        elif a != e:
            mismatches.append(f"  [{k}] scalar diff: {a!r} vs {e!r}")
    return mismatches


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--indexed-uri", required=True)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import fsspec
    print(f"[v2] loading manifest + metadata from {args.indexed_uri}...", flush=True)
    with fsspec.open(f"{args.indexed_uri}/manifest.json", "r") as f:
        manifest = json.load(f)
    with fsspec.open(f"{args.indexed_uri}/metadata.json", "r") as f:
        metadata = json.load(f)
    print(f"[v2] N={len(manifest)}, config_hash={metadata['config_hash']}", flush=True)

    # Compose the same cfg preprocess used
    ds_cfg = resolve_dataset_cfg(args.dataset)
    seq_len = int(ds_cfg.get("seq_len", 256))
    doc_chunk_seq_len = ds_cfg.get("doc_chunk_seq_len") or seq_len
    num_chunks_per_doc = int(ds_cfg.get("num_chunks_per_doc", 1))
    min_doc_length = int(ds_cfg.get("min_doc_length", 64))
    chat_template = bool(ds_cfg.get("chat_template", False))
    force_thinking = bool(ds_cfg.get("force_thinking", False))
    mask_prefix = bool(ds_cfg.get("mask_prefix", True))
    provide_docs = bool(ds_cfg.get("provide_docs", True))
    doc_length = doc_chunk_seq_len * num_chunks_per_doc
    sources_cfg = ds_cfg["sources"]
    hf_token = os.environ.get("HF_TOKEN")

    print(f"[v2] loading tokenizer {args.tokenizer}...", flush=True)
    from transformers import AutoTokenizer
    tok_path = args.tokenizer
    if not os.path.isdir(tok_path):
        local = os.path.expanduser(f"~/weights/huggingface/{args.tokenizer}")
        if os.path.isdir(local):
            tok_path = local
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    # Sample N random manifest entries
    rng = random.Random(args.seed)
    picks = rng.sample(range(len(manifest)), min(args.n_samples, len(manifest)))
    print(f"[v2] sampling {len(picks)} entries", flush=True)

    # Group picks by source so we can iter each source once
    per_source_picks: dict[str, dict[int, int]] = {}  # source -> {raw_row_id: manifest_index}
    for mi in picks:
        e = manifest[mi]
        per_source_picks.setdefault(e["source"], {})[int(e["raw_row_id"])] = mi

    # Load ArrayRecord shards for reads
    import grain.python as grain
    shard_uris = [
        f"{args.indexed_uri}/samples-{i:05d}.arrayrecord"
        for i in range(int(metadata["n_shards"]))
    ]
    src = grain.ArrayRecordDataSource(shard_uris)

    def read_actual(mi: int) -> dict:
        # Compute global record key from (shard_idx, offset). ArrayRecordDataSource
        # concatenates shards in order, so key = sum(shard_len[0..si-1]) + offset.
        # But we don't have shard lengths. Simpler: iterate manifest in-order to
        # get key. Since manifest is in insertion order, manifest index == key.
        return pickle.loads(src[mi])

    # Now walk each source and re-transform the picked rows
    total_pass = 0
    total_fail = 0
    fail_details = []
    for source_name, picks_map in per_source_picks.items():
        source_cfg = sources_cfg[source_name]
        print(f"\n[v2] --- source: {source_name} ({len(picks_map)} picks) ---", flush=True)
        found = 0
        for raw_row_id, item in iter_source_rows(source_cfg, "train", hf_token):
            if raw_row_id not in picks_map:
                # If we've exhausted picks_map, stop scanning this source
                max_needed = max(picks_map.keys())
                if raw_row_id > max_needed:
                    break
                continue
            mi = picks_map[raw_row_id]
            # Re-run filter + transform
            passed = qa_filter_predicate(
                item, tokenizer, seq_len, chat_template, doc_length,
                force_thinking=force_thinking, min_doc_length=min_doc_length,
                filter_doc_length=provide_docs,
            )
            if not passed:
                total_fail += 1
                fail_details.append(f"[{source_name}:{raw_row_id}] filter now REJECTS (was accepted at preprocess time)")
                found += 1
                if found == len(picks_map):
                    break
                continue
            expected = qa_transform_item(
                item, tokenizer, seq_len, doc_chunk_seq_len, num_chunks_per_doc,
                mask_prefix, chat_template, force_thinking=force_thinking,
            )
            actual = read_actual(mi)
            mm = compare_samples(actual, expected, manifest[mi])
            if mm:
                total_fail += 1
                fail_details.append(f"[{source_name}:{raw_row_id}] MISMATCH:\n" + "\n".join(mm))
            else:
                total_pass += 1
            found += 1
            if found == len(picks_map):
                break

    print("\n[v2] === VERDICT ===", flush=True)
    print(f"[v2] pass: {total_pass}   fail: {total_fail}", flush=True)
    if fail_details:
        print("[v2] failures:", flush=True)
        for d in fail_details[:20]:
            print(d, flush=True)
        if len(fail_details) > 20:
            print(f"  ... and {len(fail_details)-20} more", flush=True)
        return 1
    print("[v2] PASS — all sampled rows are byte-equal between preprocess-time and runtime-transform", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
