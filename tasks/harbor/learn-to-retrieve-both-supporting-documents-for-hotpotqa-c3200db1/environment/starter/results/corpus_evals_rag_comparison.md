# Corpus Evals: Memory Layers vs RAG Comparison

Checkpoint: `4B_pretraining_cot_unfreeze_all_topk_128_norm-2026-04-12-00-36-01` @ step 100000

W&B run: [rag_eval_Qwen-Qwen3-4B_gen_large_mem_musique_gen_large_mem_hotpotqa_gen_large_mem_msmarco](https://wandb.ai/memory-layers/memory-layers-eval/runs/w5rbucwh)

Date: 2026-04-16 | n=128 questions per dataset (except RAG MS MARCO judge: n=500)

**RAG baseline**: Qwen3-Embedding-0.6B retrieval (top-5) + Qwen3-4B generation
**Memory Layers**: full corpus embedded into memory bank, generation with chunked retrieval (lookup_chunk_size=8192)
**Judge**: Qwen3-4B LLM judge

---

## Generation Quality (LLM Judge)

| Dataset | Mem Layers | RAG | Delta |
|---------|-----------|-----|-------|
| MuSiQue (supporting) | 0.1953 | 0.2188 | -0.023 |
| HotpotQA (distractor) | 0.1719 | 0.5625 | -0.391 |
| MS MARCO QA | 0.2031 | 0.5900 | -0.387 |

---

## Retrieval Quality

| Dataset | doc_access_acc | RAG recall@1 | RAG recall@5 | RAG MRR |
|---------|---------------|-------------|-------------|---------|
| MuSiQue | N/A | 0.0000 | 0.0000 | 0.0000 |
| HotpotQA | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| MS MARCO | 0.0000 | 0.0060 | 0.0060 | 0.0060 |

**Notes on retrieval metrics:**
- `doc_access_acc` is absent for MuSiQue because `provide_docs=False` in the dataset config — the QA dataset doesn't supply document tokens during generation, so there are no positive doc tokens to match against the corpus memory lookup. The 0.0 values for HotpotQA and MS MARCO indicate the corpus hash lookup found no byte-matching chunks between the QA dataset's positive doc tokens and the embedded corpus chunks.
- RAG recall@1/5 uses `query_gt_column=answer`, which checks whether the answer string appears verbatim in retrieved documents. For multi-hop datasets (MuSiQue, HotpotQA), single-step embedding retrieval cannot locate the answer via one retrieval step — the 0.0 is expected. For MS MARCO, the `golden_answers` column contains reference answers that only rarely appear verbatim in the retrieved passages (0.6% hit rate).

---

## All Metrics (JSON)

```json
{
  "gen_large_mem_musique/generated_count": 128,
  "gen_large_mem_musique/llm_judge_accuracy": 0.1953125,
  "gen_large_mem_musique/rag_accuracy": 0.21875,
  "gen_large_mem_musique/rag_recall@1": 0.0,
  "gen_large_mem_musique/rag_recall@5": 0.0,
  "gen_large_mem_musique/rag_mrr": 0.0,
  "gen_large_mem_musique/memory_vs_rag_delta": -0.0234375,
  "gen_large_mem_hotpotqa/generated_count": 128,
  "gen_large_mem_hotpotqa/doc_access_acc": 0.0,
  "gen_large_mem_hotpotqa/llm_judge_accuracy": 0.171875,
  "gen_large_mem_hotpotqa/rag_accuracy": 0.5625,
  "gen_large_mem_hotpotqa/rag_recall@1": 0.0,
  "gen_large_mem_hotpotqa/rag_recall@5": 0.0,
  "gen_large_mem_hotpotqa/rag_mrr": 0.0,
  "gen_large_mem_hotpotqa/memory_vs_rag_delta": -0.390625,
  "gen_large_mem_msmarco/generated_count": 128,
  "gen_large_mem_msmarco/doc_access_acc": 0.0,
  "gen_large_mem_msmarco/llm_judge_accuracy": 0.203125,
  "gen_large_mem_msmarco/rag_accuracy": 0.59,
  "gen_large_mem_msmarco/rag_recall@1": 0.006,
  "gen_large_mem_msmarco/rag_recall@5": 0.006,
  "gen_large_mem_msmarco/rag_mrr": 0.006,
  "gen_large_mem_msmarco/memory_vs_rag_delta": -0.38687499999999997
}
```

---

## Output Files

- Generations: `outputs/2026-04-16/06-56-53/eval_results/step_100000/`
- RAG outputs: `outputs/2026-04-16/06-56-53/rag/`
