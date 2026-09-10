# RAG Eval Harness

`rag_eval.py` + `evals/rag/` — a **classic retrieval-augmented-generation** pipeline (embed →
retrieve top-k → generate → judge) that reuses the same eval_set/task configs as `eval.py`. Use
it as the RAG baseline / to sweep retrieval settings, vs. the memory-model path in `eval.py`.

## Pipeline (`evals/rag/single_embedding_retrieval.py`)
Driven by each task's **`rag:` block**:
1. Embed the `doc_dataset` corpus with `embedding_model`.
2. For each of `num_queries` queries (`query_column` / `query_gt_column`), retrieve `top_k` docs.
3. Generate an answer with `gen_model` (`evals/rag/generator.py`, `evals/rag/vllm.py`).
4. Judge with `judge_model` → summary.

Writes `embeddings/`, `retrieval.json`, `generated.json`, `summary.json` under
`<out>/rag/<eval_key>/`.

## `rag:` config block (per task)
`doc_dataset`, `query_dataset`, `query_column`, `query_gt_column`, `num_queries`,
`embedding_model`, `top_k`, `gen_model`, `judge_model`. Defaults (from the MSA scripts):
`embedding_model=Qwen/Qwen3-Embedding-0.6B`, `gen_model=Qwen/Qwen3-4B`,
`judge_model=Qwen/Qwen3-8B`, `top_k=5`. Override on the CLI: `rag.top_k=10`.

## `eval.py` vs `rag_eval.py`
| | `eval.py` | `rag_eval.py` |
|-|-----------|---------------|
| Retrieval | the model's **learned memory** | external embedding + kNN |
| Use for | scoring a memory checkpoint | classic-RAG baseline, retrieval-config sweeps |

Launch scripts (`rag_eval_corpus_evals.sh`, `rag_msa_evals.sh`) and their data-prep prereqs are
in [../infrastructure/experiment-launch-instructions.md](../infrastructure/experiment-launch-instructions.md);
the corpora come from [../data/data-preparation.md](../data/data-preparation.md).
