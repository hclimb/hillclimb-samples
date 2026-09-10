# Training-Scale Memory Sweep

Evaluating `generation_embed` (per-example embedded docs) at batch sizes that keep the memory layer
close to the training distribution, as opposed to `gen_large_mem` which loads millions of tokens.

**Checkpoint:** `4B_pretraining_cot_unfreeze_all_topk_128_norm` — step 100000  
**Eval:** 64 samples, LLM judge accuracy (Qwen3-8B), `force_thinking=true`  
**Memory tokens** = `batch_size × num_chunks_per_doc × doc_chunk_seq_len`  
- MS MARCO: `num_chunks_per_doc=4`, `doc_chunk_seq_len=256`  
- HotpotQA: `num_chunks_per_doc=16`, `doc_chunk_seq_len=256`

---

## Results

| Scale | MS MARCO batch_size | MS MARCO mem tokens | MS MARCO accuracy | HotpotQA batch_size | HotpotQA mem tokens | HotpotQA accuracy |
|-------|--------------------:|--------------------:|------------------:|--------------------:|--------------------:|------------------:|
| 1x    | 128                 | 131k                | 32.81%            | 64                  | 262k                | 43.75%            |
| 4x    | 512                 | 524k                | 26.56%            | 256                 | 1.05M               | 35.94%            |
| 8x    | 1024                | 1.05M               | 26.56%            | 512                 | 2.10M               | 29.69%            |

> Note: 4x uses `lookup_chunk_size=8192` to avoid OOM during memory attention.  
> 2x (256/128) crashed due to TPU instability following the initial 4x OOM — no valid results.

---

## Notes

- Training used `batch_size=32, num_chunks_per_doc=4, doc_chunk_seq_len=256` → ~33k memory tokens.
  The "1x" batch sizes here are 4x that training scale; HotpotQA's 16 chunks/doc pushes it higher.
- 4x without `lookup_chunk_size` OOM'd: used 36.69G of 31.25G HBM (exceeded by 5.44G).
