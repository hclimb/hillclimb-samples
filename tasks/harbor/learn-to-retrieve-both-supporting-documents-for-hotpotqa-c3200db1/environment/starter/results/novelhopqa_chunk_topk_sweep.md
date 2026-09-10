# NovelHopQA Chunk Size × TopK Sweep

Evaluating how chunk size and topk interact on NovelHopQA hop_1.
8 samples, LLM judge accuracy (no document context), chunk_size=256/1024/4096 tokens.
Total book tokens held constant at ~1M per sample (num_chunks_per_doc = 1M / chunk_size).

Models:
- **topk128**: `multihop_qa_sft_midtraining_4B` @ step 10000
- **topk512**: `multihop_qa_sft_midtraining_4B_topk512` @ step 10000

---

## Results: LLM Judge Accuracy (hop_1, 8 samples)

| chunk_size | num_chunks | topk128 | topk512 |
|-----------|------------|---------|---------|
| 256       | 4096       | 0.125   | 0.250   |
| 1024      | 1024       | 0.125   | 0.000   |
| 4096      | 256        | OOM     | OOM     |

**Note on chunk_size=4096 OOM:** The embedding model's self-attention over 4096-token sequences requires a `[32, 8, 4096, 4096]` f32 KV matrix (~34 GB), exceeding the 33.5 GB HBM limit. Would require chunked attention or a smaller `num_chunks_per_doc` to fit.

---

## Run Tracker

| Model   | chunk_size | Script                                    | Status | W&B | LLM Judge |
|---------|-----------|-------------------------------------------|--------|-----|-----------|
| topk128 | 256       | `eval_novelhopqa_topk128_chunk256.sh`     | [x]    | [eoli7zj3](https://wandb.ai/memory-layers/memory-layers-eval/runs/eoli7zj3) | 0.125 |
| topk128 | 1024      | `eval_novelhopqa_topk128_chunk1024.sh`    | [x]    | [4n9x1gv7](https://wandb.ai/memory-layers/memory-layers-eval/runs/4n9x1gv7) | 0.125 |
| topk128 | 4096      | `eval_novelhopqa_topk128_chunk4096.sh`    | OOM    | [ls5rhwph](https://wandb.ai/memory-layers/memory-layers-eval/runs/ls5rhwph) | OOM   |
| topk512 | 256       | `eval_novelhopqa_topk512_chunk256.sh`     | [x]    | [owplsevy](https://wandb.ai/memory-layers/memory-layers-eval/runs/owplsevy) | 0.250 |
| topk512 | 1024      | `eval_novelhopqa_topk512_chunk1024.sh`    | [x]    | [a10h8miv](https://wandb.ai/memory-layers/memory-layers-eval/runs/a10h8miv) | 0.000 |
| topk512 | 4096      | `eval_novelhopqa_topk512_chunk4096.sh`    | OOM    | —   | OOM   |

All scripts in `scripts/embed/`. Run from repo root.
