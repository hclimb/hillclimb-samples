# SGLang FlashMLA Prefill Benchmark

Run from a 4x B200 node after installing the refreshed SGLang/FlashMLA trees.

```bash
python benchmarks/sglang_flashmla_prefill/run_prefill_matrix.py \
  --model-path /path/to/DeepSeek-V4-Flash \
  --tensor-parallel-size 4 \
  --ep-size 4 \
  --row-tiles 1,2,4,8
```

The matrix uses the agreed DeepSeek V4 Flash workload shape:

```text
512 tokens     batch=1  topk=512  swa_window=128  reps=3
4096 tokens    batch=4  topk=512  swa_window=128  reps=3
16384 tokens   batch=8  topk=512  swa_window=128  reps=3
65536 tokens   batch=8  topk=512  swa_window=128  reps=3
```

`M=1` is the upstream SGLang/FlashMLA baseline. `M=2/4/8` enables the
row-packed path through `SGLANG_DSV4_FLASHMLA_PREFILL_ROW_TILE_M`.

Summarize speedups:

```bash
python benchmarks/sglang_flashmla_prefill/compare_results.py results/<run>/summary.jsonl
```
