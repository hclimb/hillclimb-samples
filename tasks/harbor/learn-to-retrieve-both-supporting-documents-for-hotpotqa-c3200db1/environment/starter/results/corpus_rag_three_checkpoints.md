# Corpus Evals: Memory Layers vs RAG — Three Checkpoints

| Model | Checkpoint | Step |
|-------|-----------|------|
| **Pretrained** | `4B_pretraining_cot_unfreeze_all_topk_128_norm` | 100 000 |
| **Two-Pass 128K** | `4B_pretraining_two_pass-2026-04-17` | 102 250 |
| **Two-Pass 256K** | `4B_pretraining_two_pass-2026-04-18` | 101 000 |

n = 128 questions per dataset. Memory bank embeds full corpus (lookup_chunk_size=8192). RAG: Qwen3-Embedding-0.6B top-5 retrieval + Qwen3-4B generation.

![Corpus RAG comparison](figures/corpus_rag_three_checkpoints.png)

---

## Generation Quality (LLM Judge)

### Memory Layers

| Dataset | Pretrained | Two-Pass 128K | Two-Pass 256K |
|---------|-----------|---------------|---------------|
| MuSiQue | 0.1797 | 0.2969 | 0.2656 |
| HotpotQA | 0.1719 | 0.2188 | 0.2031 |
| MS MARCO | 0.2031 | 0.1875 | 0.2109 |

### RAG (Qwen3-Embedding-0.6B + Qwen3-4B)

| Dataset | Pretrained | Two-Pass 128K | Two-Pass 256K |
|---------|-----------|---------------|---------------|
| MuSiQue | 0.2188 | 0.2188 | 0.1875 |
| HotpotQA | 0.5469 | 0.5469 | 0.5625 |
| MS MARCO | 0.5980 | 0.6080 | 0.5920 |

### Delta (Memory − RAG)

| Dataset | Pretrained | Two-Pass 128K | Two-Pass 256K |
|---------|-----------|---------------|---------------|
| MuSiQue | -0.0391 | +0.0781 | +0.0781 |
| HotpotQA | -0.3750 | -0.3281 | -0.3594 |
| MS MARCO | -0.3949 | -0.4205 | -0.3811 |

---

## Retrieval Quality

### doc_access_acc — fraction of all (head × pos × top-k) slots hitting correct doc

| Dataset | Pretrained | Two-Pass 128K | Two-Pass 256K |
|---------|-----------|---------------|---------------|
| MuSiQue | 0.0331 | 0.0667 | 0.0726 |
| HotpotQA | 0.0128 | 0.0244 | 0.0265 |
| MS MARCO | 0.0049 | 0.0100 | 0.0111 |

### doc_hit_rate — fraction of examples where any lookup hit the correct doc

| Dataset | Pretrained | Two-Pass 128K | Two-Pass 256K |
|---------|-----------|---------------|---------------|
| MuSiQue | N/A | N/A | N/A |
| HotpotQA | N/A | N/A | N/A |
| MS MARCO | N/A | N/A | N/A |

### doc_hit_rate_by_position — fraction of active positions with any hit

| Dataset | Pretrained | Two-Pass 128K | Two-Pass 256K |
|---------|-----------|---------------|---------------|
| MuSiQue | N/A | N/A | N/A |
| HotpotQA | N/A | N/A | N/A |
| MS MARCO | N/A | N/A | N/A |

### RAG recall@5 / MRR (Qwen3-Embedding-0.6B, checkpoint-independent)

| Dataset | recall@1 | recall@5 | MRR |
|---------|---------|---------|-----|
*(Same retrieval model for all checkpoints — values from base run)*

| MuSiQue | 0.0000 | 0.0000 | 0.0000 |
| HotpotQA | 0.0000 | 0.0000 | 0.0000 |
| MS MARCO | 0.0060 | 0.0060 | 0.0060 |

---

## All Metrics (JSON)

### Pretrained (step 100k)

```json
{
  "gen_large_mem_musique/generated_count": 128,
  "gen_large_mem_musique/doc_access_acc": 0.03308872304435742,
  "gen_large_mem_musique/llm_judge_accuracy": 0.1796875,
  "gen_large_mem_hotpotqa/generated_count": 128,
  "gen_large_mem_hotpotqa/doc_access_acc": 0.012802616233932227,
  "gen_large_mem_hotpotqa/llm_judge_accuracy": 0.171875,
  "gen_large_mem_msmarco/generated_count": 128,
  "gen_large_mem_msmarco/doc_access_acc": 0.004942772365810667,
  "gen_large_mem_msmarco/llm_judge_accuracy": 0.203125,
  "gen_large_mem_musique/rag_accuracy": 0.21875,
  "gen_large_mem_musique/rag_recall@1": 0.0,
  "gen_large_mem_musique/rag_recall@5": 0.0,
  "gen_large_mem_musique/rag_mrr": 0.0,
  "gen_large_mem_hotpotqa/rag_accuracy": 0.546875,
  "gen_large_mem_hotpotqa/rag_recall@1": 0.0,
  "gen_large_mem_hotpotqa/rag_recall@5": 0.0,
  "gen_large_mem_hotpotqa/rag_mrr": 0.0,
  "gen_large_mem_msmarco/rag_accuracy": 0.598,
  "gen_large_mem_msmarco/rag_recall@1": 0.006,
  "gen_large_mem_msmarco/rag_recall@5": 0.006,
  "gen_large_mem_msmarco/rag_mrr": 0.006,
  "gen_large_mem_musique/memory_vs_rag_delta": -0.0390625,
  "gen_large_mem_hotpotqa/memory_vs_rag_delta": -0.375,
  "gen_large_mem_msmarco/memory_vs_rag_delta": -0.394875
}
```

### Two-Pass 128K (step 102k)

```json
{
  "gen_large_mem_musique/generated_count": 128,
  "gen_large_mem_musique/doc_access_acc": 0.06665783768835815,
  "gen_large_mem_musique/llm_judge_accuracy": 0.296875,
  "gen_large_mem_hotpotqa/generated_count": 128,
  "gen_large_mem_hotpotqa/doc_access_acc": 0.02441855135422152,
  "gen_large_mem_hotpotqa/llm_judge_accuracy": 0.21875,
  "gen_large_mem_msmarco/generated_count": 128,
  "gen_large_mem_msmarco/doc_access_acc": 0.009959464407315082,
  "gen_large_mem_msmarco/llm_judge_accuracy": 0.1875,
  "gen_large_mem_musique/rag_accuracy": 0.21875,
  "gen_large_mem_musique/rag_recall@1": 0.0,
  "gen_large_mem_musique/rag_recall@5": 0.0,
  "gen_large_mem_musique/rag_mrr": 0.0,
  "gen_large_mem_hotpotqa/rag_accuracy": 0.546875,
  "gen_large_mem_hotpotqa/rag_recall@1": 0.0,
  "gen_large_mem_hotpotqa/rag_recall@5": 0.0,
  "gen_large_mem_hotpotqa/rag_mrr": 0.0,
  "gen_large_mem_msmarco/rag_accuracy": 0.608,
  "gen_large_mem_msmarco/rag_recall@1": 0.006,
  "gen_large_mem_msmarco/rag_recall@5": 0.006,
  "gen_large_mem_msmarco/rag_mrr": 0.006,
  "gen_large_mem_musique/memory_vs_rag_delta": 0.078125,
  "gen_large_mem_hotpotqa/memory_vs_rag_delta": -0.328125,
  "gen_large_mem_msmarco/memory_vs_rag_delta": -0.4205
}
```

### Two-Pass 256K (step 101k)

```json
{
  "gen_large_mem_musique/generated_count": 128,
  "gen_large_mem_musique/doc_access_acc": 0.07255601512058808,
  "gen_large_mem_musique/llm_judge_accuracy": 0.265625,
  "gen_large_mem_hotpotqa/generated_count": 128,
  "gen_large_mem_hotpotqa/doc_access_acc": 0.026465865523094163,
  "gen_large_mem_hotpotqa/llm_judge_accuracy": 0.203125,
  "gen_large_mem_msmarco/generated_count": 128,
  "gen_large_mem_msmarco/doc_access_acc": 0.011089709654392097,
  "gen_large_mem_msmarco/llm_judge_accuracy": 0.2109375,
  "gen_large_mem_musique/rag_accuracy": 0.1875,
  "gen_large_mem_musique/rag_recall@1": 0.0,
  "gen_large_mem_musique/rag_recall@5": 0.0,
  "gen_large_mem_musique/rag_mrr": 0.0,
  "gen_large_mem_hotpotqa/rag_accuracy": 0.5625,
  "gen_large_mem_hotpotqa/rag_recall@1": 0.0,
  "gen_large_mem_hotpotqa/rag_recall@5": 0.0,
  "gen_large_mem_hotpotqa/rag_mrr": 0.0,
  "gen_large_mem_msmarco/rag_accuracy": 0.592,
  "gen_large_mem_msmarco/rag_recall@1": 0.006,
  "gen_large_mem_msmarco/rag_recall@5": 0.006,
  "gen_large_mem_msmarco/rag_mrr": 0.006,
  "gen_large_mem_musique/memory_vs_rag_delta": 0.078125,
  "gen_large_mem_hotpotqa/memory_vs_rag_delta": -0.359375,
  "gen_large_mem_msmarco/memory_vs_rag_delta": -0.38106249999999997
}
```

