# Corpus Evals: Two-Pass vs Pretrained

Comparing retrieval accuracy (`doc_access_acc`) and generation accuracy (`llm_judge_accuracy`) across three checkpoints on the MuSiQue, HotpotQA, and MS MARCO corpus-level retrieval tasks.

Each task embeds the full document corpus into the memory bank and queries with 128 questions. `doc_access_acc` uses pre-resolved integer corpus IDs (`pos_doc_ids`) for exact ground-truth matching.

| Model | Checkpoint |
|-------|-----------|
| **Pretrained** | `4B_pretraining_cot_unfreeze_all_topk_128_norm` @ step 100000 |
| **Two-Pass 128K** | `4B_pretraining_two_pass-2026-04-17` @ step 102250 |
| **Two-Pass 256K** | `4B_pretraining_two_pass-2026-04-18` @ step 101000 |

W&B runs: [Pretrained](https://wandb.ai/memory-layers/memory-layers-eval/runs/736nazjz) · [Two-Pass 128K](https://wandb.ai/memory-layers/memory-layers-eval/runs/megce4bq) · [Two-Pass 256K](https://wandb.ai/memory-layers/memory-layers-eval/runs/nneiqbnm)

---

## Doc Access Accuracy

Higher is better. Fraction of retrieved top-k memory vectors that overlap with the ground-truth document's memory positions.

| Task | Pretrained | Two-Pass 128K | Two-Pass 256K |
|------|-----------|---------------|---------------|
| MuSiQue | 0.0331 | 0.0667 | **0.0726** |
| HotpotQA | 0.0128 | 0.0244 | **0.0265** |
| MS MARCO | 0.0049 | 0.0100 | **0.0111** |
| **Average** | 0.0169 | 0.0337 | **0.0367** |

---

## Generation Accuracy (LLM Judge)

Higher is better.

| Task | Pretrained | Two-Pass 128K | Two-Pass 256K |
|------|-----------|---------------|---------------|
| MuSiQue | 0.2031 | **0.2813** | 0.2500 |
| HotpotQA | 0.1953 | **0.2109** | **0.2109** |
| MS MARCO | 0.1875 | **0.2031** | **0.2031** |
| **Average** | 0.1953 | **0.2318** | 0.2213 |

---

## Key Observations

- **Doc access accuracy**: Both two-pass checkpoints roughly double the pretrained model's doc_access_acc across all tasks. 256K marginally outperforms 128K on retrieval (~9% relative improvement over 128K).
- **Generation accuracy**: Two-pass 128K leads on generation (+3.6pp average over pretrained). Two-pass 256K is slightly behind 128K on generation (-1pp) despite having marginally better retrieval.
- **MuSiQue** shows the largest generation gap — 128K outperforms 256K by 3.1pp. This is the hardest multi-hop task (provide_docs=false), so generation quality is fully retrieval-dependent.
- **Retrieval vs generation correlation**: doc_access_acc is low across the board (max 7%), suggesting the models are not yet reliably retrieving the exact ground-truth documents, but the two-pass training clearly improves both retrieval and generation.
