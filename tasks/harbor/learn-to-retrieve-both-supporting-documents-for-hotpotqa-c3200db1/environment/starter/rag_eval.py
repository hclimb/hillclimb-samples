import os
import json
import sys
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from utils import setup_gcs_credentials
from dotenv import load_dotenv
from evals.shared import (
    resolve_train_cfg, apply_checkpoint_model_cfg, init_wandb, run_eval_worker, run_metrics_pipeline,
)

def _run_rag_pipeline(eval_key, rag_cfg, output_dir):
    import subprocess
    rag_dir = os.path.join(output_dir, "rag", eval_key)
    os.makedirs(rag_dir, exist_ok=True)
    
    embeddings_dir = rag_cfg.get("embeddings_dir") or os.path.join(rag_dir, "embeddings")
    ret_meta = os.path.join(rag_dir, "retrieval.json")
    ret_results = os.path.splitext(ret_meta)[0] + "_results.json"
    gen_file = os.path.join(rag_dir, "generated.json")
    summary_file = os.path.join(rag_dir, "summary.json")
    
    cmd = [sys.executable, "evals/rag/single_embedding_retrieval.py",
           "--doc_dataset", str(rag_cfg["doc_dataset"]),
           "--query_dataset", str(rag_cfg["query_dataset"]),
           "--query_column", str(rag_cfg["query_column"]),
           "--query_gt_column", str(rag_cfg["query_gt_column"]),
           "--num_queries", str(rag_cfg["num_queries"]),
           "--model_name", str(rag_cfg["embedding_model"]),
           "--top_k", str(rag_cfg["top_k"]),
           "--embeddings_dir", embeddings_dir,
           "--output", ret_meta]
    for k in ["doc_split", "doc_column", "query_split", "max_docs", "hf_ckpt_dir", "query_hf_config", "doc_hf_config", "query_answer_column", "query_task", "query_prefix", "doc_prefix", "max_doc_length", "max_query_length", "encode_batch_size", "search_batch_size", "tp_devices"]:
        if rag_cfg.get(k): cmd += [f"--{k}", str(rag_cfg[k])]
        
    print(f"[{eval_key}] RAG Step 1/3: Retrieval...")
    subprocess.run(cmd, check=True)
    retrieval_metrics = {}
    if os.path.exists(ret_meta):
        with open(ret_meta) as f:
            retrieval_metrics = (json.load(f) or {}).get("metrics", {}) or {}
    
    gen_cmd = [sys.executable, "evals/rag/generator.py", "--input", ret_results, "--output", gen_file,
               "--model", str(rag_cfg["gen_model"]), "--top_k_docs", str(rag_cfg.get("gen_top_k_docs", 5)),
               "--start_server",
               "--tensor_parallel_size", str(rag_cfg.get("gen_tensor_parallel_size", 8)),
               "--max_model_len", str(rag_cfg.get("gen_max_model_len", 16384)),
               "--concurrency", str(rag_cfg.get("gen_concurrency", 64)),
               "--temperature", str(rag_cfg.get("gen_temperature", 0.0))]
    print(f"[{eval_key}] RAG Step 2/3: Generation...")
    subprocess.run(gen_cmd, check=True)
    
    from evals.metrics.llm_judge import llm_judge_accuracy
    with open(gen_file) as f: gen_data = json.load(f)
    results = [{
        "prompt": r["query"],
        "generated_answer": r["answer"],
        "ground_truth": r.get("ground_truth", r.get("gt_answer", "")),
    } for r in gen_data]
    scores, _ = llm_judge_accuracy(
        results,
        model_id=str(rag_cfg["judge_model"]),
        tensor_parallel_size=int(rag_cfg.get("judge_tensor_parallel_size", 8)),
        concurrency=int(rag_cfg.get("judge_concurrency", 32)),
        use_document=False,
    )
    acc = sum(scores)/len(scores) if scores else 0.0
    with open(summary_file, "w") as f: json.dump({"accuracy": acc}, f)
    return acc, retrieval_metrics

@hydra.main(config_path="configs", config_name="rag_eval", version_base="1.2")
def main(cfg: DictConfig):
    setup_gcs_credentials()
    load_dotenv()
    from hydra.core.hydra_config import HydraConfig
    train_cfg, ckpt_dir, step = resolve_train_cfg(cfg)
    # Checkpoint's saved model config is authoritative for architecture; explicit model.* overrides win.
    cfg = apply_checkpoint_model_cfg(cfg, train_cfg, HydraConfig.get().overrides.task)
    init_wandb(cfg, train_cfg, prefix="rag_eval")

    out_dir = HydraConfig.get().runtime.output_dir
    
    manifest = run_eval_worker(cfg, train_cfg, ckpt_dir, step, out_dir)
    all_metrics = run_metrics_pipeline(manifest, cfg, ckpt_dir, step, out_dir)
    
    if cfg.get("evals"):
        global_rag = cfg.get("rag", {})
        for eval_key, eval_task in cfg.evals.items():
            task_rag = eval_task.get("rag")
            if not task_rag and not global_rag: continue
            
            combined = OmegaConf.to_container(global_rag, resolve=True)
            if task_rag: combined.update(OmegaConf.to_container(task_rag, resolve=True))
            
            try:
                acc, retrieval_metrics = _run_rag_pipeline(eval_key, combined, out_dir)
                all_metrics[f"{eval_key}/rag_accuracy"] = acc
                for metric_name, metric_value in retrieval_metrics.items():
                    all_metrics[f"{eval_key}/rag_{metric_name}"] = metric_value
            except Exception as e:
                print(f"RAG pipeline failed for {eval_key}: {e}")
            
    print("All Evaluation Results:")
    print(json.dumps(all_metrics, indent=2))
    if wandb.run: wandb.log(all_metrics)

if __name__ == "__main__":
    main()
