"""Run the RAG baseline for ONE MSA dataset, bypassing rag_eval.py's JAX worker.

rag_eval.py forces a (slow, irrelevant) membed worker eval before the RAG pipeline.
This driver calls `_run_rag_pipeline` directly with a hand-built rag_cfg, then uploads
the binary LLM-judge accuracy to
    gs://memory-layers-training/pareto_eval/rag/<dataset>.json
in the collector schema {"metrics": {"llm_judge_accuracy": <acc>}}.

Run on a TPU VM (needs the repo venv + TPU). PYTHONPATH must point at the repo root.
    PYTHONPATH=$HOME/memory-layers python scripts/misc/pareto_rag_one.py <dataset>
"""
import json, os, sys

import gcsfs

from rag_eval import _run_rag_pipeline

DATASET = sys.argv[1]
DST = f"memory-layers-training/pareto_eval/rag/{DATASET}.json"

fs = gcsfs.GCSFileSystem()
if fs.exists(DST):
    print(f"SKIP {DATASET} (result exists)")
    sys.exit(0)

hf = os.environ["HF_USERNAME"]
slug = DATASET.replace("_", "-")
rag_cfg = {
    "doc_dataset": f"{hf}/msa-{slug}-docs-with-ids",
    "query_dataset": f"{hf}/msa-{slug}-qa-with-ids",
    "query_column": "question",
    "query_gt_column": "answer",
    "num_queries": 128,
    "doc_split": "train",
    "doc_column": "text",
    "query_split": "train",
    # retrieval
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
    "top_k": 5,
    "max_doc_length": 512,
    "max_query_length": 128,
    "encode_batch_size": int(os.environ.get("RAG_ENCODE_BS", "512")),
    "query_task": "",
    # generation (Qwen3-4B — comparable to the memory-layer base model)
    "gen_model": "Qwen/Qwen3-4B",
    "gen_tensor_parallel_size": 8,
    "gen_max_model_len": 16384,
    "gen_top_k_docs": 5,
    "gen_concurrency": 64,
    "gen_temperature": 0.0,
    # judge (binary LLM-judge accuracy)
    "judge_model": "Qwen/Qwen3-8B",
    "judge_tensor_parallel_size": 8,
    "judge_concurrency": 32,
}

out_dir = os.path.join(os.path.expanduser("~"), "ragout2", DATASET)
os.makedirs(out_dir, exist_ok=True)

acc, ret_metrics = _run_rag_pipeline(DATASET, rag_cfg, out_dir)
print(f"RAG {DATASET} accuracy={acc} retrieval={ret_metrics}")

payload = json.dumps({"metrics": {"llm_judge_accuracy": acc},
                      "retrieval_metrics": ret_metrics}).encode()
with fs.open(DST, "wb") as f:
    f.write(payload)
print(f"UPLOADED gs://{DST}")
