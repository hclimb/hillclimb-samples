"""RAG baseline ALONE — retrieve -> generate -> judge, with no memory model involved.

`rag_eval.py` always runs the memory-model eval worker first and only then the RAG pipeline, so
it needs a checkpoint, a TPU-resident 4B model, and roughly twice the wall clock. When all you
want is the classic-RAG number, that is wasted work.

This calls `rag_eval._run_rag_pipeline` directly — the SAME three subprocess stages
(evals/rag/single_embedding_retrieval.py -> evals/rag/generator.py -> llm_judge_accuracy), not a
reimplementation — so the number is directly comparable to the `<eval_key>/rag_accuracy` that
rag_eval.py would have produced.

    RAG_CORPUS=… QUERY_DS=… NUM_QUERIES=128 TP=4 python scripts/misc/rag_only.py

⚠️ The corpus must be the matched 512-doc one built by
datagen/musique/build_musique_c512_rag_corpus.py. Handing RAG a raw corpus truncated with
--max_docs would search a haystack that mostly lacks the answers; see that script's docstring.
"""
import json
import os
import sys

import dotenv

dotenv.load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rag_eval import _run_rag_pipeline

USER = os.environ.get("HF_USERNAME", "")
TP = int(os.environ.get("TP") or len([d for d in os.listdir("/dev/vfio") if d.isdigit()]) or 4)

# Mirrors configs/rag_eval.yaml's rag: block, with the MuSiQue c512 values applied. Kept explicit
# rather than composed through Hydra so this runs standalone with no config machinery.
rag_cfg = {
    "doc_dataset":   os.environ.get("RAG_CORPUS", f"{USER}/musique-c512-rag-corpus"),
    "query_dataset": os.environ.get("QUERY_DS", f"{USER}/msa-musique-qa-with-ids"),
    "query_column": "question",
    "query_gt_column": "answer",
    "doc_split": "train",
    "doc_column": "text",
    "query_split": "train",
    "num_queries": int(os.environ.get("NUM_QUERIES", 128)),
    "max_docs": int(os.environ.get("MAX_DOCS", 512)),
    "embedding_model": os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
    "hf_ckpt_dir": os.path.expanduser("~/weights/huggingface"),
    "tp_devices": 1,
    # 256, not rag_eval.yaml's 512: the memory eval indexes ONE 256-token chunk per doc
    # (max_chunks_per_doc: 1, chunk_size: 256), so 512 would let RAG read twice as much per doc.
    "max_doc_length": int(os.environ.get("MAX_DOC_LENGTH", 256)),
    "max_query_length": 128,
    "encode_batch_size": 2048,
    "query_task": "",
    # top_k = how many docs the retriever returns; gen_top_k_docs = how many of those actually go
    # into the generator's prompt. Separate knobs: narrowing the prompt does not change retrieval.
    "top_k": int(os.environ.get("TOP_K", 5)),
    "gen_model": os.environ.get("GEN_MODEL", "Qwen/Qwen3-4B"),
    "gen_tensor_parallel_size": TP,
    "gen_max_model_len": 8192,
    "gen_top_k_docs": int(os.environ.get("GEN_TOP_K_DOCS", os.environ.get("TOP_K", 5))),
    "gen_temperature": 0.0,
    "gen_concurrency": 64,
    "judge_model": os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-4B"),
    "judge_tensor_parallel_size": TP,
    "judge_concurrency": 32,
}

out_dir = os.environ.get("OUT_DIR", f"outputs/rag_only_top{rag_cfg['gen_top_k_docs']}")
os.makedirs(out_dir, exist_ok=True)
print(f"[rag_only] corpus={rag_cfg['doc_dataset']} queries={rag_cfg['query_dataset']}")
print(f"[rag_only] n={rag_cfg['num_queries']} retrieve_top_k={rag_cfg['top_k']} "
      f"docs_in_prompt={rag_cfg['gen_top_k_docs']} tp={TP} max_doc_len={rag_cfg['max_doc_length']}")

acc, retrieval_metrics = _run_rag_pipeline("musique_c512", rag_cfg, out_dir)

summary = {"rag_accuracy": acc, **{f"rag_{k}": v for k, v in (retrieval_metrics or {}).items()}}
path = os.path.join(out_dir, "rag_only_summary.json")
with open(path, "w") as f:
    json.dump({"metrics": summary, "config": rag_cfg}, f, indent=2)

print("\n=== RAG BASELINE (no memory model) ===")
for k, v in summary.items():
    print(f"  {k}: {v}")
print(f"\nwrote {path}")
