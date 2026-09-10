#!/usr/bin/env python3
"""
Count the EXACT number of post-filter (surviving) rows a single named source in a
Hydra dataset config would contribute, using the real qa_filter_predicate — no
sampling, no estimate. Used to find a source's true ceiling before balancing other
sources against it via data/preprocess_arrayrecord.py --target-rows-per-source.

CPU-only, no JAX/TPU device use.

    HF_HUB_OFFLINE=1 GROUND_HF_PARQUET=$HOME/hf_parquet \
      uv run python scripts/debug/count_source_post_filter_rows.py \
      --dataset qa_hard_neg_no_multihop_sft4b --source combined_hard_neg_sft4b
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--source", required=True, help="source key within dataset.sources")
    p.add_argument("--trainer", default="staged_ground")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    args = p.parse_args()

    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from tqdm import tqdm
    from data.qa import qa_filter_predicate
    from data.preprocess_arrayrecord import iter_source_rows

    with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base=None):
        cfg = compose(config_name="train", overrides=[
            "model=qwen3_mem_embed",
            f"dataset={args.dataset}",
            f"trainer={args.trainer}",
        ])
    dcfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    sources = dcfg["sources"]
    if args.source not in sources:
        raise SystemExit(f"source {args.source!r} not in dataset {args.dataset!r} sources: {list(sources)}")
    source_cfg = sources[args.source]

    seq_len = int(dcfg.get("seq_len", 256))
    doc_chunk_seq_len = dcfg.get("doc_chunk_seq_len") or seq_len
    num_chunks_per_doc = int(dcfg.get("num_chunks_per_doc", 1))
    min_doc_length = int(dcfg.get("min_doc_length", 64))
    chat_template = bool(dcfg.get("chat_template", False))
    force_thinking = bool(dcfg.get("force_thinking", False))
    provide_docs = bool(dcfg.get("provide_docs", True))
    doc_length = doc_chunk_seq_len * num_chunks_per_doc

    tok_path = args.tokenizer
    local_cache = os.path.expanduser(f"~/weights/huggingface/{args.tokenizer}")
    if not os.path.isdir(tok_path) and os.path.isdir(local_cache):
        tok_path = local_cache
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    hf_token = os.environ.get("HF_TOKEN")
    n_pre = n_post = 0
    pbar = tqdm(desc=f"{args.source}", unit="rows")
    for _raw_row_id, item in iter_source_rows(source_cfg, dcfg.get("split", "train"), hf_token):
        n_pre += 1
        pbar.update(1)
        if qa_filter_predicate(
            item, tokenizer, seq_len, chat_template, doc_length,
            force_thinking=force_thinking, min_doc_length=min_doc_length,
            filter_doc_length=provide_docs,
        ):
            n_post += 1
    pbar.close()

    print(f"\n{args.source}: pre={n_pre} post={n_post} ({100*n_post/max(n_pre,1):.2f}% survived)")
    print(f"TARGET_ROWS_PER_SOURCE={n_post}")


if __name__ == "__main__":
    main()
