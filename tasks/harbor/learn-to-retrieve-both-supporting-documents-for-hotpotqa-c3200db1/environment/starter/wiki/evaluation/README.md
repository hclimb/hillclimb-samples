# Evaluation

How checkpoints are scored. Entry point: **`eval.py`** (`uv run eval.py checkpoint_dir=…`).
A dedicated **RAG harness** (`rag_eval.py`) shares the same configs. To launch on TPUs, see
[../infrastructure/experiment-launch-instructions.md](../infrastructure/experiment-launch-instructions.md).

## Flow (`eval.py`)
`resolve_train_cfg` (reconstruct the model from the checkpoint's saved `.hydra/config.yaml`) →
`init_wandb` → **`run_eval_worker`** (spawns a JAX subprocess that runs the evaluators and
writes raw samples + a manifest, then exits, freeing the TPU) → **`run_metrics_pipeline`**
(parent process scores the samples — LLM judge etc.). This two-process split is the core design:
[two-process-design.md](two-process-design.md).

## Config layering
`eval.yaml` → an **eval_set** (a named group of tasks) → each **task**
(`configs/eval/tasks/*.yaml`, composes an eval type + dataset + doc_dataset) → an eval **type**
base (`configs/eval/*.yaml`). Details: [eval-configs.md](eval-configs.md).

## Pages
- [two-process-design.md](two-process-design.md) — the JAX-worker / metrics-parent split, `eval_worker`, the manifest
- [evaluator-types.md](evaluator-types.md) — `get_evaluator` and each evaluator (nll, generation, embed, large-mem, base)
- [metrics.md](metrics.md) — the LLM judge, lexical grounding, and the vLLM server
- [ruler.md](ruler.md) — the RULER long-context benchmark
- [rag-eval.md](rag-eval.md) — `rag_eval.py` classic-RAG harness
- [eval-configs.md](eval-configs.md) — `eval.yaml` / eval_set / task / type layering + override syntax
- [eval-boxes.md](eval-boxes.md) — scoring a run **while it trains**: why judge metrics can't come from the training loop, and how a box logs `eval/*` into the training wandb run

## Quick start
```bash
uv run eval.py checkpoint_dir=gs://…/qwen3_mem_embed/60000        # default: pretraining eval set
uv run eval.py checkpoint_dir=… '~eval_set@evals=pretraining' '+eval_set@evals=msa_evals'
uv run eval.py checkpoint_dir=… eval.gen_large_mem_msa_hotpotqa.eval.num_samples=64
```
