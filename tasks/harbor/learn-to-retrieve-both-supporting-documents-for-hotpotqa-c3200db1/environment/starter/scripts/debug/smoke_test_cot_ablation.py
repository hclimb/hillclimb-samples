#!/usr/bin/env python3
"""
CPU-only smoke test for the CoT ablation wiring (data/utils.py::_docid_normalize `cot_field` ->
data/qa.py::qa_transform_item `cot_doc` append; independently, `think_field` -> answer prefix).
No TPU, no JAX device use.

Composes the given dataset config via Hydra (default: multihop_hard_neg_full_cot_ablation),
pulls a few raw rows through the actual normalizer + qa_transform_item path, and asserts
behavior matches whichever of cot_field/think_field that dataset's source has set:
  - cot_field set: `cot_doc` present, matches the raw think field verbatim, and
    qa_transform_item's `pos_doc_mask` has MORE positive-marked chunks than the same row
    processed without the CoT append (the extra doc really lands in the positive slots).
  - cot_field NOT set: `cot_doc` never appears on any row (memory bank untouched), regardless
    of think_field.
  - think_field set: answer is `<think>...</think>`-prefixed with the raw think text.
  - think_field NOT set: answer has no `<think>` prefix.

    HF_HUB_OFFLINE=1 GROUND_HF_PARQUET=$HOME/hf_parquet JAX_PLATFORMS=cpu \
      uv run python scripts/debug/smoke_test_cot_ablation.py \
      [--dataset multihop_hard_neg_full_cot_ablation]
"""
import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="multihop_hard_neg_full_cot_ablation")
    p.add_argument("--n-rows", type=int, default=5)
    args = p.parse_args()

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from data.qa import qa_transform_item, _load_dataset_with_backoff
    from data.utils import make_docid_normalizer

    root = os.path.abspath("configs")
    with initialize_config_dir(config_dir=root, version_base=None):
        cfg = compose(config_name="train", overrides=[
            "model=qwen3_mem_embed",
            f"dataset={args.dataset}",
            "trainer=staged_ground",
            "eval_set@trainer.evals=none",
        ])
    dcfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    sources = dcfg["sources"]
    assert len(sources) == 1, f"expected exactly 1 source, got {list(sources)}"
    source_name, source_cfg = next(iter(sources.items()))
    cot_field = source_cfg.get("cot_field")
    think_field = source_cfg.get("think_field")
    print(f"dataset={args.dataset} source={source_name} cot_field={cot_field!r} think_field={think_field!r}")

    tok_path = "Qwen/Qwen3-4B"
    local_cache = os.path.expanduser(f"~/weights/huggingface/{tok_path}")
    if os.path.isdir(local_cache):
        tok_path = local_cache
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    normalizer = make_docid_normalizer(
        source_cfg["corpus_path"], think_field=think_field,
        max_neg_docs=source_cfg.get("max_neg_docs"),
        min_neg_docs=source_cfg.get("min_neg_docs", 0),
        mask_ce=source_cfg.get("mask_ce", False),
        field_map=source_cfg.get("field_map"), cot_field=cot_field,
    )

    ds = _load_dataset_with_backoff(source_cfg["hf_name"], source_cfg.get("hf_config"),
                                     dcfg.get("split", "train"), None)

    seq_len = int(dcfg.get("seq_len", 512))
    doc_chunk_seq_len = dcfg.get("doc_chunk_seq_len") or seq_len
    num_chunks_per_doc = int(dcfg.get("num_chunks_per_doc", 256))
    mask_prefix = bool(dcfg.get("mask_prefix", True))
    chat_template = bool(dcfg.get("chat_template", False))

    n_checked = 0
    n_verified = 0
    for raw in ds:
        item = normalizer(raw)
        n_checked += 1
        think_raw = raw.get(think_field or "think", "")
        if not think_raw:
            continue

        if cot_field:
            assert item.get("cot_doc") == think_raw, "[FAIL] cot_doc doesn't match raw think field verbatim"
        else:
            assert "cot_doc" not in item, "[FAIL] cot_field unset but cot_doc is present -- memory bank would be touched"

        if think_field:
            assert item["answer"].startswith("<think>") and think_raw in item["answer"], (
                "[FAIL] think_field is set but answer wasn't <think>-prefixed with the raw text"
            )
        else:
            assert not item["answer"].startswith("<think>"), (
                "[FAIL] think_field unset but answer is <think>-prefixed anyway"
            )

        if cot_field:
            sample = qa_transform_item(
                dict(item), tokenizer, seq_len, doc_chunk_seq_len, num_chunks_per_doc,
                mask_prefix, chat_template,
            )
            n_pos_with = int(sample["pos_doc_mask"].sum())
            item_without = dict(item); item_without.pop("cot_doc")
            sample_without = qa_transform_item(
                dict(item_without), tokenizer, seq_len, doc_chunk_seq_len, num_chunks_per_doc,
                mask_prefix, chat_template,
            )
            n_pos_without = int(sample_without["pos_doc_mask"].sum())
            assert n_pos_with > n_pos_without, (
                f"[FAIL] appending cot_doc did not increase positive-marked chunks "
                f"({n_pos_with} vs {n_pos_without})"
            )
            print(f"row {n_checked}: pos chunks with CoT={n_pos_with}, without={n_pos_without}, "
                  f"cot_doc[:60]={item['cot_doc'][:60]!r}")
        else:
            sample = qa_transform_item(
                dict(item), tokenizer, seq_len, doc_chunk_seq_len, num_chunks_per_doc,
                mask_prefix, chat_template,
            )
            print(f"row {n_checked}: no cot_doc (as expected), answer[:60]={item['answer'][:60]!r}")

        n_verified += 1
        if n_verified >= args.n_rows:
            break

    assert n_verified > 0, f"[FAIL] checked {n_checked} rows, none had a `think` value to verify against"
    print(f"\n[PASS] {n_verified} rows verified against cot_field={cot_field!r}, think_field={think_field!r}.")


if __name__ == "__main__":
    main()
