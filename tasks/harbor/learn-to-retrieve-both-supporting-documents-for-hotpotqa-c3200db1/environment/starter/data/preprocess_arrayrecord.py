"""One-time preprocessing: tokenize + filter + write ArrayRecord shards + manifest.

For a given dataset config (e.g. qa_hard_neg_think_sft4b) + tokenizer + seq_len +
per-source normalizer settings, walks every raw parquet row, applies the same
filter + transform used by the streaming pipeline (data/qa.py::qa_filter_predicate,
qa_transform_item), and writes the surviving tokenized samples as sharded
ArrayRecord files.

Design invariants (see wiki/experiments/plan zesty-jumping-badger):
  - Records are written in per-source, raw-file order. NO shuffling on disk.
    Runtime IndexSampler(shuffle=True, seed) is the single source of shuffling.
  - Config hash covers EVERY transform input. Any change to tokenizer_id,
    seq_len, doc_chunk_seq_len, num_chunks_per_doc, min_doc_length, or ANY
    per-source setting (hf_name, hf_config, field_map, think_field,
    doc_separator, neg_score_threshold, min_neg_docs, mask_ce, normalizer_type,
    prompt_path) invalidates the hash.
  - metadata.json is written LAST as the completion marker. Preprocessing is
    gated on metadata.json existence, not shard existence — a preempted preprocess
    box doesn't leave a half-written shard set that gets silently reused.
  - Individual shards are also resumable: write to samples-{i:05d}.arrayrecord.tmp,
    rename to .arrayrecord on completion, skip if the completed file exists.
  - Hard-fail (not warn) on: (a) config-hash mismatch with existing metadata.json,
    (b) any source's post-filter proportion dropping >30% relative to its pre-
    filter share (silent staleness / skew is the failure class we're fixing).

Output layout (default): gs://memory-layers-training/indexed/<config-hash>/
  samples-{i:05d}.arrayrecord           tokenized dicts, ArrayRecord format
  manifest.json                          length-N list of (shard_idx, offset, source_id, raw_row_id)
  metadata.json                          config hash, per-source counts — written LAST

Usage:
    HF_HUB_OFFLINE=1 GROUND_HF_PARQUET=$HOME/hf_parquet \\
        uv run python data/preprocess_arrayrecord.py \\
            --dataset qa_hard_neg_think_sft4b \\
            --tokenizer Qwen/Qwen3-4B \\
            --out gs://memory-layers-training/indexed/
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import fsspec
import numpy as np
from tqdm import tqdm

# Repo root on sys.path (script is under data/, run standalone)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.qa import qa_filter_predicate, qa_transform_item, _load_dataset_with_backoff
from data.utils import make_normalizer, make_flashrag_normalizer, make_hotpotqa_normalizer


SHARD_SAMPLES = 25_000   # ~25k samples per shard × ~20KB = ~500MB per shard
MAX_DROP_FRAC = 0.30     # hard-fail if a source's post-filter proportion drops > this


# ── Config resolution via Hydra compose ─────────────────────────────────────────

def resolve_dataset_cfg(dataset_name: str) -> Any:
    """Compose the real train.py config and return cfg.dataset. Matches what the
    training loop sees — same source list, same per-source normalizer settings."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if GlobalHydra().is_initialized():
        GlobalHydra().clear()
    with initialize_config_dir(config_dir=os.path.join(root, "configs"), version_base="1.2"):
        cfg = compose(
            config_name="train",
            overrides=[
                f"dataset={dataset_name}",
                "eval_set@trainer.evals=none",
            ],
        )
    return cfg.dataset


# ── Config hash ─────────────────────────────────────────────────────────────────

def compute_config_hash(dataset_cfg, tokenizer_id: str, target_rows_per_source: Optional[int] = None) -> str:
    """Hash EVERY input to the qa_filter_predicate + qa_transform_item pipeline.

    Any change invalidates the shard set. Deliberately does NOT include
    shuffle_seed (runtime sampler concern) or batch_size (irrelevant to sample
    content).
    """
    def _norm_source(src):
        # Sort keys so ordering in the yaml doesn't affect hash.
        return {
            "hf_name": src.get("hf_name"),
            "hf_config": src.get("hf_config"),
            "field_map": dict(sorted((src.get("field_map") or {}).items())),
            "think_field": src.get("think_field"),
            "doc_separator": src.get("doc_separator"),
            "normalizer_type": src.get("normalizer_type"),
            "flashrag_context_key": src.get("flashrag_context_key"),
            "neg_score_threshold": src.get("neg_score_threshold"),
            "min_neg_docs": src.get("min_neg_docs", 0),
            "mask_ce": src.get("mask_ce", False),
            "prompt_path": src.get("prompt_path"),
        }
    sources = dataset_cfg.get("sources") or {}
    normalized_sources = sorted(
        [(name, _norm_source(src)) for name, src in sources.items()],
        key=lambda x: x[0],
    )
    hash_input = {
        "tokenizer_id": tokenizer_id,
        "seq_len": int(dataset_cfg.get("seq_len", 256)),
        "doc_chunk_seq_len": dataset_cfg.get("doc_chunk_seq_len"),
        "num_chunks_per_doc": int(dataset_cfg.get("num_chunks_per_doc", 1)),
        "min_doc_length": int(dataset_cfg.get("min_doc_length", 64)),
        "chat_template": bool(dataset_cfg.get("chat_template", False)),
        "force_thinking": bool(dataset_cfg.get("force_thinking", False)),
        "mask_prefix": bool(dataset_cfg.get("mask_prefix", True)),
        "provide_docs": bool(dataset_cfg.get("provide_docs", True)),
        # None (default) vs. an int are different datasets — a cap changes which
        # rows survive, so it must invalidate the hash like any other content change.
        "target_rows_per_source": target_rows_per_source,
        # sorted list of (name, normalized-source-dict) tuples
        "sources": normalized_sources,
    }
    canonical = json.dumps(hash_input, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ── ArrayRecord I/O ─────────────────────────────────────────────────────────────

def _open_array_record_writer(path: str):
    """Open an ArrayRecord writer. array_record only supports local paths, not
    gs://. Callers must write locally then upload."""
    from array_record.python.array_record_module import ArrayRecordWriter
    return ArrayRecordWriter(path, "group_size:1")


def serialize_sample(sample: Dict[str, Any]) -> bytes:
    """Serialize the training-sample dict to bytes for ArrayRecord storage.
    Uses pickle+HIGHEST_PROTOCOL — numpy arrays serialize efficiently, dict
    schema is preserved, deserialization is one pickle.loads()."""
    return pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_sample(raw: bytes) -> Dict[str, Any]:
    """Inverse of serialize_sample. Used by QADatasetIndexed at read time."""
    return pickle.loads(raw)


# ── Per-source raw-row iteration (matches qa.py's setup, without the interleave) ─

def iter_source_rows(source_cfg, split, hf_token):
    """Stream rows from ONE source, in raw file order (no per-source shuffle).

    Applies the normalizer + prompt_template step so the raw_item passed to
    qa_filter_predicate / qa_transform_item has the same schema as at training
    time (question, answer, pos_doc, neg_doc, _min_neg_docs, prompt_template, ...).

    Yields (raw_row_id, normalized_item). raw_row_id is the per-source insertion
    index (0-based) — stable across preprocessing runs so long as the source
    file set doesn't change.
    """
    name = source_cfg["hf_name"]
    hf_config = source_cfg.get("hf_config")
    field_map_s = source_cfg.get("field_map")
    think_field_s = source_cfg.get("think_field")
    doc_separator_s = source_cfg.get("doc_separator")
    normalizer_type = source_cfg.get("normalizer_type")
    neg_score_threshold_s = source_cfg.get("neg_score_threshold")
    min_neg_docs_s = source_cfg.get("min_neg_docs", 0)
    mask_ce_s = source_cfg.get("mask_ce", False)
    prompt_path_s = source_cfg.get("prompt_path")
    flashrag_context_key_s = source_cfg.get("flashrag_context_key")

    ds = _load_dataset_with_backoff(name, hf_config, split, hf_token)

    if normalizer_type == "flashrag":
        normalizer = make_flashrag_normalizer(flashrag_context_key_s)
    elif normalizer_type == "hotpotqa":
        normalizer = make_hotpotqa_normalizer()
    else:
        normalizer = make_normalizer(
            field_map_s, think_field_s, doc_separator_s,
            neg_score_threshold=neg_score_threshold_s,
            min_neg_docs=min_neg_docs_s, mask_ce=mask_ce_s,
        )

    tmpl_path = prompt_path_s or "data/prompts/default.txt"
    if not os.path.isabs(tmpl_path):
        # Resolve relative to repo root
        tmpl_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            tmpl_path,
        )
    tmpl = open(tmpl_path).read()

    row_id = 0
    for raw in ds:
        normalized = normalizer(raw)
        normalized["prompt_template"] = tmpl
        yield row_id, normalized
        row_id += 1


# ── Idempotency / staleness guards ──────────────────────────────────────────────

def check_existing_metadata(out_uri: str, config_hash: str) -> Optional[Dict[str, Any]]:
    """Return existing metadata dict if the run is complete for THIS config hash.
    None otherwise. Hard-fails on a metadata hash mismatch (silent staleness protection)."""
    meta_path = f"{out_uri.rstrip('/')}/metadata.json"
    try:
        with fsspec.open(meta_path, "r") as f:
            meta = json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[warn] failed to read {meta_path}: {e}; will re-preprocess", flush=True)
        return None
    if meta.get("config_hash") != config_hash:
        raise RuntimeError(
            f"CONFIG-HASH MISMATCH at {out_uri}: existing metadata.json has "
            f"hash={meta.get('config_hash')!r}, current config hashes to {config_hash!r}. "
            f"Either delete the directory to re-preprocess or point --out at a fresh path. "
            f"Silent staleness is the failure class this rework exists to eliminate; refusing to reuse."
        )
    return meta


def hard_check_source_proportions(pre_filter_counts: Dict[str, int],
                                  post_filter_counts: Dict[str, int]) -> None:
    """Hard-fail if any source's post-filter proportion drops > MAX_DROP_FRAC
    relative to its pre-filter share. Distribution-shift guardrail (§4 of plan)."""
    pre_total = sum(pre_filter_counts.values())
    post_total = sum(post_filter_counts.values())
    if pre_total == 0 or post_total == 0:
        raise RuntimeError(f"empty sample pool (pre_total={pre_total}, post_total={post_total})")
    for name in pre_filter_counts:
        pre_share = pre_filter_counts[name] / pre_total
        post_share = post_filter_counts.get(name, 0) / post_total
        drop_frac = (pre_share - post_share) / pre_share if pre_share > 0 else 0.0
        if drop_frac > MAX_DROP_FRAC:
            raise RuntimeError(
                f"SOURCE {name!r} POST-FILTER PROPORTION DROPPED {100*drop_frac:.1f}% "
                f"(pre-filter share={100*pre_share:.1f}%, post-filter share={100*post_share:.1f}%). "
                f"This exceeds MAX_DROP_FRAC={MAX_DROP_FRAC}. Investigate the filter (likely "
                f"a seq_len/doc_length setting is systematically rejecting this source), "
                f"or bump MAX_DROP_FRAC if this shift is intentional."
            )


# ── Main preprocessing driver ────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="Hydra dataset config name")
    ap.add_argument("--tokenizer", required=True, help="Tokenizer id or local path")
    ap.add_argument("--out", default="gs://memory-layers-training/indexed/",
                    help="Output root; final path is <out>/<config-hash>/")
    ap.add_argument("--split", default="train")
    ap.add_argument("--shard-samples", type=int, default=SHARD_SAMPLES,
                    help="Samples per ArrayRecord shard (default 25000)")
    ap.add_argument("--staging-dir", default="/tmp/preprocess_arrayrecord",
                    help="Local dir for shard writes before upload")
    ap.add_argument("--max-samples-per-source", type=int, default=None,
                    help="Smoke-test cap: after processing this many RAW rows from a "
                         "source, move on. The resulting shards are NOT valid for real "
                         "training (they lie about N and per-source counts) — output "
                         "goes to a distinct 'SMOKE-<config-hash>' path to prevent "
                         "reuse.")
    ap.add_argument("--target-rows-per-source", type=int, default=None,
                    help="Real (non-smoke) balancing cap: stop each source once it has "
                         "TARGET_ROWS_PER_SOURCE surviving (post-filter) samples. Sources "
                         "with fewer available rows than the target simply contribute "
                         "everything they have. Output is written to the normal <hash>/ "
                         "path (valid for training) and IS covered by the config hash, "
                         "since it changes which rows are included.")
    args = ap.parse_args()

    print(f"=== preprocess_arrayrecord ===", flush=True)
    print(f"  dataset:     {args.dataset}", flush=True)
    print(f"  tokenizer:   {args.tokenizer}", flush=True)
    print(f"  out root:    {args.out}", flush=True)
    print(f"  split:       {args.split}", flush=True)

    # 1. Resolve config, compute hash, set output path
    ds_cfg = resolve_dataset_cfg(args.dataset)
    config_hash = compute_config_hash(ds_cfg, args.tokenizer, args.target_rows_per_source)
    # Smoke-test outputs go to a distinct path so they cannot be accidentally
    # reused for training.
    smoke_prefix = "SMOKE-" if args.max_samples_per_source is not None else ""
    out_uri = f"{args.out.rstrip('/')}/{smoke_prefix}{config_hash}"
    print(f"  config hash: {config_hash}", flush=True)
    print(f"  out path:    {out_uri}", flush=True)
    if args.max_samples_per_source is not None:
        print(f"  SMOKE MODE:  max {args.max_samples_per_source} raw rows/source", flush=True)
    if args.target_rows_per_source is not None:
        print(f"  BALANCED:    target {args.target_rows_per_source} post-filter rows/source", flush=True)

    # 2. Idempotency check
    existing = check_existing_metadata(out_uri, config_hash)
    if existing is not None:
        print(f"[skip] metadata.json exists with matching hash. N={existing.get('N')}. "
              f"Delete {out_uri}/metadata.json to force re-preprocess.", flush=True)
        return 0

    # 3. Load tokenizer
    print(f"  loading tokenizer {args.tokenizer}...", flush=True)
    from transformers import AutoTokenizer
    # Try local snapshot first (matches how QADataset loads under HF_HUB_OFFLINE=1)
    tok_path = args.tokenizer
    if not os.path.isdir(tok_path):
        local_cache = os.path.expanduser(f"~/weights/huggingface/{args.tokenizer}")
        if os.path.isdir(local_cache):
            tok_path = local_cache
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    # 4. Iterate every source, apply filter + transform, buffer surviving samples
    seq_len = int(ds_cfg.get("seq_len", 256))
    doc_chunk_seq_len = ds_cfg.get("doc_chunk_seq_len") or seq_len
    num_chunks_per_doc = int(ds_cfg.get("num_chunks_per_doc", 1))
    min_doc_length = int(ds_cfg.get("min_doc_length", 64))
    chat_template = bool(ds_cfg.get("chat_template", False))
    force_thinking = bool(ds_cfg.get("force_thinking", False))
    mask_prefix = bool(ds_cfg.get("mask_prefix", True))
    provide_docs = bool(ds_cfg.get("provide_docs", True))
    doc_length = doc_chunk_seq_len * num_chunks_per_doc

    sources = ds_cfg.get("sources") or {}
    if not sources:
        raise RuntimeError(f"dataset {args.dataset!r} has no `sources:` — single-source mode not supported by this preprocessor")

    os.makedirs(args.staging_dir, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")

    pre_filter_counts: Dict[str, int] = {}
    post_filter_counts: Dict[str, int] = {}
    manifest_entries: List[Tuple[int, int, str, int]] = []  # (shard_idx, offset, source_name, raw_row_id)
    shard_idx = 0
    shard_offset = 0
    current_writer = None
    current_shard_path = None

    def _open_next_shard():
        nonlocal current_writer, current_shard_path, shard_offset
        current_shard_path = os.path.join(args.staging_dir, f"samples-{shard_idx:05d}.arrayrecord")
        tmp_path = current_shard_path + ".tmp"
        # If a .tmp exists from a prior crashed run, remove it and start fresh
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        current_writer = _open_array_record_writer(tmp_path)
        shard_offset = 0

    _open_next_shard()

    def _close_current_shard():
        """Close, rename .tmp -> .arrayrecord, upload, then delete the local copy.
        Uploading per-shard (rather than staging everything and uploading at the
        very end) keeps local disk bounded to ~1 shard regardless of dataset size,
        and matches the "individual shards are resumable" design intent — a crash
        mid-run doesn't lose already-written shards to a disk that never got them
        off the box."""
        nonlocal current_writer
        if current_writer is None:
            return
        current_writer.close()
        current_writer = None
        tmp_path = current_shard_path + ".tmp"
        os.rename(tmp_path, current_shard_path)
        remote = f"{out_uri}/{os.path.basename(current_shard_path)}"
        _copy_to_remote(current_shard_path, remote)
        os.remove(current_shard_path)
        print(f"  uploaded + freed local copy: {remote}", flush=True)

    t0 = time.time()

    for source_name, source_cfg in sources.items():
        print(f"\n[source] {source_name} (hf_name={source_cfg.get('hf_name')})", flush=True)
        n_pre = 0
        n_post = 0
        pbar = tqdm(desc=f"  {source_name}", unit="rows")
        for raw_row_id, item in iter_source_rows(source_cfg, args.split, hf_token):
            n_pre += 1
            pbar.update(1)
            if args.max_samples_per_source is not None and n_pre > args.max_samples_per_source:
                break
            if args.target_rows_per_source is not None and n_post >= args.target_rows_per_source:
                break
            if not qa_filter_predicate(
                item, tokenizer, seq_len, chat_template, doc_length,
                force_thinking=force_thinking, min_doc_length=min_doc_length,
                filter_doc_length=provide_docs,
            ):
                continue
            sample = qa_transform_item(
                item, tokenizer, seq_len, doc_chunk_seq_len, num_chunks_per_doc,
                mask_prefix, chat_template, force_thinking=force_thinking,
            )
            payload = serialize_sample(sample)
            current_writer.write(payload)
            manifest_entries.append((shard_idx, shard_offset, source_name, raw_row_id))
            shard_offset += 1
            n_post += 1
            # Rotate shard when it reaches SHARD_SAMPLES
            if shard_offset >= args.shard_samples:
                _close_current_shard()
                shard_idx += 1
                _open_next_shard()
        pbar.close()
        pre_filter_counts[source_name] = n_pre
        post_filter_counts[source_name] = n_post
        print(f"  {source_name}: pre={n_pre} → post={n_post} ({100*n_post/max(n_pre,1):.1f}% survived)", flush=True)

    # Close last shard (even if partial)
    if shard_offset > 0:
        _close_current_shard()
        n_shards = shard_idx + 1
    else:
        # Delete the empty last shard file that was opened but never written
        if current_writer is not None:
            current_writer.close()
            if os.path.exists(current_shard_path + ".tmp"):
                os.remove(current_shard_path + ".tmp")
        n_shards = shard_idx  # last shard was empty, don't count it

    N = len(manifest_entries)
    elapsed = time.time() - t0
    print(f"\n[done writing local shards] N={N}, n_shards={n_shards}, {elapsed:.1f}s", flush=True)

    # 5. Guardrail: hard-fail if any source's post-filter proportion dropped too much.
    # Skipped in smoke mode AND when balancing via --target-rows-per-source — both
    # deliberately make post-filter counts diverge from natural pre-filter shares
    # (that's the whole point of capping), so the "skew" the check looks for is
    # expected here, not a bug.
    if args.max_samples_per_source is None and args.target_rows_per_source is None:
        hard_check_source_proportions(pre_filter_counts, post_filter_counts)
    else:
        print("[skip] hard_check_source_proportions (smoke or balanced-target mode)", flush=True)

    # 6. Write manifest + metadata locally
    manifest = [
        {"shard_idx": si, "offset": off, "source": src, "raw_row_id": rid}
        for si, off, src, rid in manifest_entries
    ]
    manifest_local = os.path.join(args.staging_dir, "manifest.json")
    with open(manifest_local, "w") as f:
        json.dump(manifest, f)
    print(f"[wrote] {manifest_local} ({os.path.getsize(manifest_local)} bytes)", flush=True)

    metadata_local = os.path.join(args.staging_dir, "metadata.json")
    metadata = {
        "config_hash": config_hash,
        "dataset": args.dataset,
        "tokenizer": args.tokenizer,
        "seq_len": seq_len,
        "doc_chunk_seq_len": doc_chunk_seq_len,
        "num_chunks_per_doc": num_chunks_per_doc,
        "min_doc_length": min_doc_length,
        "N": N,
        "n_shards": n_shards,
        "shard_samples": args.shard_samples,
        "pre_filter_counts": pre_filter_counts,
        "post_filter_counts": post_filter_counts,
        "elapsed_seconds": elapsed,
    }
    with open(metadata_local, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[wrote] {metadata_local}", flush=True)

    # 7. Upload manifest + metadata. Shards were already uploaded (and their local
    # copies freed) as each one closed, in _close_current_shard(). metadata.json
    # goes LAST — completion marker.
    print(f"\n[upload] → {out_uri}", flush=True)
    fs, _ = fsspec.core.url_to_fs(out_uri)
    # Upload manifest
    _copy_to_remote(manifest_local, f"{out_uri}/manifest.json")
    print(f"  uploaded manifest.json", flush=True)
    # Upload metadata LAST — this is the completion marker
    _copy_to_remote(metadata_local, f"{out_uri}/metadata.json")
    print(f"  uploaded metadata.json (COMPLETION MARKER)", flush=True)

    print(f"\n[DONE] N={N}, config_hash={config_hash}, output at {out_uri}", flush=True)
    return 0


def _copy_to_remote(local_path: str, remote_uri: str) -> None:
    """Upload local file to remote (gs:// or local path)."""
    if remote_uri.startswith("gs://") or remote_uri.startswith("s3://"):
        # Use gsutil for gs://; fsspec.copy would work but gsutil is faster on large files
        if remote_uri.startswith("gs://"):
            import subprocess
            subprocess.run(["gsutil", "-q", "cp", local_path, remote_uri], check=True)
        else:
            fs, _ = fsspec.core.url_to_fs(remote_uri)
            fs.put(local_path, remote_uri)
    else:
        # Local file copy
        os.makedirs(os.path.dirname(remote_uri) or ".", exist_ok=True)
        import shutil
        shutil.copy2(local_path, remote_uri)


if __name__ == "__main__":
    sys.exit(main() or 0)
