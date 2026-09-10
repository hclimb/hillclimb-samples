# Eval Queue

Evals to run in order. Update status as they complete.

---

## FanoutQA — chunk_size=1024

Same total tokens as chunk_size=256 run (10 chunks × 1024 tokens = 10,240 tokens/doc vs 40 × 256).
Memory bank: ~19,335 chunks × 1024 tokens = ~19.8M memory vectors.

| Status | Model | Script |
|--------|-------|--------|
| [x] | base | `scripts/embed/eval_fanoutqa_midtraining_base_chunk1024.sh` |
| [x] | topk128 | `scripts/embed/eval_fanoutqa_midtraining_topk128_chunk1024.sh` |
| [x] | topk512 | `scripts/embed/eval_fanoutqa_midtraining_topk512_chunk1024.sh` |

Results go into `results/multihop_midtraining_comparison.md` under a new FanoutQA chunk_size=1024 section.

---

## Completed

| Model | Eval | Script | W&B |
|-------|------|--------|-----|
| base | FanoutQA chunk_size=256 | `eval_fanoutqa_midtraining_base.sh` | [m48nlkbv](https://wandb.ai/memory-layers/memory-layers-eval/runs/m48nlkbv) |
| topk128 | FanoutQA chunk_size=256 | `eval_fanoutqa_midtraining_topk128.sh` | [qrmm5h9l](https://wandb.ai/memory-layers/memory-layers-eval/runs/qrmm5h9l) |
| topk512 | FanoutQA chunk_size=256 | `eval_fanoutqa_midtraining_topk512.sh` | [ak2lxbng](https://wandb.ai/memory-layers/memory-layers-eval/runs/ak2lxbng) |
