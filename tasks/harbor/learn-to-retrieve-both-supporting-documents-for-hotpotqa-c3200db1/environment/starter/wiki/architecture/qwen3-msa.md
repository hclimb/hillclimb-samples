# `qwen3_msa` — Memory Sparse Attention

`models/qwen3_msa.py` — a JAX port of **MSA** (Memory Sparse Attention, arXiv:2603.23516;
ref impl github.com/EverMind-AI/MSA) for this repo's static-memory eval/inference harness.
⚠ Not in the top-level repo README. Port covers the **inference path only** (no
generative-retrieval / memory-interleave). Design ref: `claude/msa-implementation-plan.md`
("EXACT ALGORITHM SPEC").

## Backbone (MSA-4B)
Qwen3-4B-Instruct-2507: 36 layers, hidden 2560, 32 q / 8 kv heads, head_dim 128,
rope_theta 5e6. **Router projectors** (`router_q_proj [32,128,2560]`,
`router_k_proj [8,128,2560]`) live on the **latter-half layers 18–35 only**; `_init_routers`
seeds them (e.g. `copy` mode). Scalar `temperature` is unused at inference.

## Two phases
1. **`encode_docs`** — run each doc through the backbone independently with **doc-local RoPE**;
   at each router layer capture mean-pooled `K̄, V̄` (post-RoPE/qk-norm) and router `K̄ᴿ`
   (no RoPE), pooling over `pooling_kernel_size=64`-token chunks (`_masked_pool_chunks`).
   Builds the per-doc banks (`_encode_docs_banks`).
2. **Query** — layers 0–17 attend locally; layers 18–35 **route top-k docs** (`route_topk` /
   `_route_doc_scores`: cosine-ish dot → head-mean → query-max → chunk-amax), gather the
   selected pooled `K̄, V̄` (`_gather_ctx` / `select_doc_context`), concat them before the
   query's local KV, and attend (`_query_router_layer`, `_attn`).

## Entry points
- `load` / `init` — `_msa_cfg_from_hf`, load backbone + routers.
- `prefill` + `decode_step` — the inference KV-cache path used by the eval harness.
- `train_forward` — the training path.

**Config:** `configs/model/` (MSA config).
