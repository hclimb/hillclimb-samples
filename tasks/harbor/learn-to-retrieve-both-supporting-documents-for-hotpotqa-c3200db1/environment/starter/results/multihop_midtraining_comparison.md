# Multihop QA Midtraining Comparison

Comparing three checkpoints on the combined pretraining_cot + multihop_qa eval suite.
All runs use batch_size=8.

| Model | Checkpoint |
|-------|-----------|
| **Base** | `4B_pretraining_cot_unfreeze_all_topk_128_norm` @ step 80000 |
| **topk128** | `multihop_qa_sft_midtraining_4B` @ step 10000 |
| **topk512** | `multihop_qa_sft_midtraining_4B_topk512` @ step 10000 |

W&B runs: [base](https://wandb.ai/memory-layers/memory-layers-eval/runs/TODO) · [topk128](https://wandb.ai/memory-layers/memory-layers-eval/runs/jvbhnoza) · [topk512](https://wandb.ai/memory-layers/memory-layers-eval/runs/jslxuurt)

FanoutQA W&B runs (chunk_size=256, 32 questions): [base](https://wandb.ai/memory-layers/memory-layers-eval/runs/m48nlkbv) · [topk128](https://wandb.ai/memory-layers/memory-layers-eval/runs/qrmm5h9l) · [topk512](https://wandb.ai/memory-layers/memory-layers-eval/runs/ak2lxbng)

FanoutQA W&B runs (chunk_size=1024, 32 questions): [base](https://wandb.ai/memory-layers/memory-layers-eval/runs/7falwkj6) · [topk128](https://wandb.ai/memory-layers/memory-layers-eval/runs/vuqem9lx) · [topk512](https://wandb.ai/memory-layers/memory-layers-eval/runs/jl1ba9ug)

---

## Generation Accuracy (LLM Judge)

Higher is better.

| Task | Base | topk128 | topk512 |
|------|------|---------|---------|
| gen_embed_diverse_qa | 0.7812 | 0.7188 | 0.6562 |
| gen_embed_multihop_qa_sft | 0.6250 | **0.9375** | 0.8750 |
| gen_embed_science_doc_completion | 0.6562 | 0.5312 | 0.5312 |
| gen_embed_science_qa | **0.9375** | 0.9062 | 0.8125 |
| gen_embed_science_summarization | **0.9062** | 0.6562 | 0.8125 |
| gen_embed_squad_qa | 0.8281 | 0.7969 | 0.8281 |
| **Average** | **0.7557** | 0.7578 | 0.7526 |

---

## NLL (Negative Log-Likelihood)

Lower is better.

| Task | Base | topk128 | topk512 |
|------|------|---------|---------|
| nll_diverse_qa | **0.4026** | 0.6697 | 0.6715 |
| nll_multihop_qa_sft | 0.7007 | **0.2776** | 0.2780 |
| nll_science_doc_completion | **1.4467** | 1.5095 | 1.4976 |
| nll_science_qa | **0.3642** | 0.6480 | 0.6509 |
| nll_science_summarization | **0.5134** | 0.6866 | 0.6865 |
| nll_squad_qa | 2.0361 | **0.8825** | 0.8740 |
| **Average** | 0.9106 | **0.7790** | 0.7764 |

---

## FanoutQA (LLM Judge, large_mem, 32 questions, chunk_size=256)

Memory bank: 2,137 Wikipedia docs → 77,339 chunks × 256 tokens = ~19.8M memory vectors (~40 GB CPU).

Higher is better.

| Task | Base | topk128 | topk512 |
|------|------|---------|---------|
| gen_large_mem_fanoutqa | 0.0312 | 0.0 | 0.0 |

---

## FanoutQA (LLM Judge, large_mem, 32 questions, chunk_size=1024)

Memory bank: same total tokens as chunk_size=256 (10 chunks × 1024 = 10,240 tokens/doc), ~19.5K chunks × 1024 tokens = ~19.8M memory vectors.

Higher is better.

| Task | Base | topk128 | topk512 |
|------|------|---------|---------|
| gen_large_mem_fanoutqa | 0.0312 | 0.0 | 0.0 |

---

## Key Observations

- **Multihop QA**: Midtraining dramatically improves multihop performance — NLL drops from 0.70 → 0.28 and gen accuracy jumps from 0.625 → 0.937 (topk128). topk512 is slightly behind on multihop (0.875 gen, 0.278 NLL).
- **Science/Diverse tasks**: Base model retains an edge on in-distribution tasks (science_qa, diverse_qa NLL), suggesting some catastrophic forgetting after midtraining.
- **topk128 vs topk512**: Very similar overall; topk128 edges out topk512 on multihop gen accuracy (0.9375 vs 0.875) while topk512 does better on science_summarization gen (0.8125 vs 0.6562). NLL scores are nearly identical across both.
- **squad_qa NLL**: Base model is much worse (2.04) — likely a format mismatch resolved by midtraining.
