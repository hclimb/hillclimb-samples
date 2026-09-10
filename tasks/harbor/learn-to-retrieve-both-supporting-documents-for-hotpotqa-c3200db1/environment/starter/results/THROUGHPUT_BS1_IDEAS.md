# BS=1 (single-stream) throughput — levers + plan (2026-07-01)

**Regime:** batch=1, latency-bound decode. Per-step cost dominated by **loading the 4B
transformer weights** (~8GB bf16) from HBM every token — memory-bandwidth bound, NOT
compute-bound. Batch scaling (the big AGG lever) does NOTHING for BS=1: it amortizes weight
loads across the batch, but at B=1 there's nothing to amortize. So BS=1 needs different levers.

Reference (prior): memory-layer per-stream ≈ 115 tok/s (best config); MSA literal single-chip
B=1 = 124.6; base Qwen3-4B single-chip weight-bound ceiling ≈ 8GB / HBM-BW.

## Levers (ranked by expected BS=1 impact)

1. **Tensor-parallel the transformer (tp_devices=8).** ★ TOP LEVER. At B=1 each chip currently
   loads the FULL 8GB of weights/step (mesh data:8,model:1 → data-parallel, weights replicated).
   With tp_devices=8 (mesh data:1,model:8) the weights SHARD across 8 chips → each loads ~1GB/step
   → up to ~8× less weight-bandwidth per step, at the cost of one all-reduce per layer (tiny B=1
   activation, fast v6e ICI). Also makes a TRUE global B=1 expressible (batch on data:1). This is
   the classic small-batch-latency win. NOT the §2b bug (that sharded the CONTRACTION dim on a
   no-op axis → spurious all-reduce; proper TP shards the right dims). TEST tp∈{1,2,4,8} @ B=1.
2. **Weight quantization (int8 / fp8).** Halves/quarters weight bytes/step → ~2-4× decode
   (bandwidth-bound). int8 W8A16 = 4GB/step. Needs a quant path in models/qwen3.py (load int8,
   dequant-in-matmul or int8 matmul). Bigger code effort; stack on top of TP.
3. **Memory-lookup cost at B=1:** approx_max_k (done, +12%); int8 mem_k (score matmul); lower
   mem_top_k (fewer per-step gathers). Smaller fraction than the transformer, but free-ish.
4. **Fuse decode into one kernel (Pallas):** cut per-step dispatch/launch overhead, which is a
   real fraction at B=1 (many tiny ops). Larger effort.
5. **KV cache in bf16 / smaller:** less KV bandwidth/step (second-order vs weights at B=1).
6. **Persistent / captured decode loop:** reduce host-dispatch overhead per step.

## RESULTS

**★ Tensor-parallelism (tp=8) is a BIG BS=1 win.** Base Qwen3-4B, B=1, ctx 2500, 256 decode
(one v6e-8): full-gen e2e **2.237s @ tp=1 (single chip)** → **0.526s @ tp=8** = **~4.25× faster**.
Confirms the diagnosis: at B=1, decode is weight-bandwidth-bound; sharding the 8GB weights across
8 chips cuts per-step weight traffic ~8× (the per-layer all-reduce on tiny B=1 activations is cheap
on v6e ICI). tp=1 decode ≈ 193 tok/s. (Clean tp=8 decode-rate + memory-layer @ tp=8 measured next.)
NOTE: only tp=1 (single chip) and tp=8 (mesh data:1,model:8) allow a literal global B=1; tp=2/4
leave data-axis>1 so B=1 isn't expressible.

**Clean B=1 request throughput (e2e, 256 tok, ctx 2500):** tp=1 = 256/2.236 = **114 tok/s**,
tp=8 = 256/0.526 = **487 tok/s** = **4.25×**. (decode-only rate: tp=1 ≈ 194; tp=8 prefill metric
is a recompile artifact so use e2e.)

**Memory-layer @ tp=8 — ROOT CAUSE: mem_num_heads=4 < 8 chips.** The load-time reshape at
models/memory_utils.py:142 (mem_q_proj/mem_o_proj, 2D→3D) throws under TP, but the deeper issue
is the memory layer sharding scheme: get_memory_sharding shards mem_q_proj P('model','data') and
the forward (memory.py:264/288) uses out_sharding=P('data',None,'model',None) — i.e. it shards the
memory HEAD dim on the model axis. With mem_num_heads=4 and tp=8, 4 doesn't divide 8 → invalid.
The base transformer works (32 q-heads / 8). FIX (non-trivial, coordinated): re-shard the memory
layer on k_dim/v_dim/hidden (1024/1024/2560, all ÷8) instead of the 4 heads — update BOTH
get_memory_sharding weight specs AND the memory.py einsum out_shardings, + the load reshape. Risk:
interacts with the bank sharding. NOT a blind overnight change. Once done, memory-layer B=1 decode
should be ~4× (transformer-dominated) at tp=8. ESTIMATE (transformer proxy): ~100 → ~400 tok/s.

(superseded) reshape-only note: The BASE Qwen3-4B forward is
TP-clean (ran at tp=8 → 4.25×). The membed model is NOT: at tp=8 it throws
`ShardingTypeError: This reshape is not supported ... operand bfloat16[4096@model,2560@data] ->
(4,1024,2560)` — a reshape in the membed forward (embed submodel and/or memory layer, e.g. head-dim
/ product-key reshapes) doesn't pass `out_sharding`, so it breaks when the tensor is model-sharded.
FIX (bounded, not fundamental): add `out_sharding=` to the offending reshapes in models/qwen3.py
(shared) / models/qwen3_mem_embed.py / models/memory*.py so they work under TP. Then also set
mem_shard_axis='model' at tp>1 so the O(M) bank scan splits across the 8 chips too. Expected result:
memory-layer B=1 decode ~4× (transformer-dominated) — same win as the base. Implementation is the
only thing between here and the number; the LEVER (TP for bandwidth-bound B=1) is proven.

**Memory-layer tp=8 — attempt STOPPED (cascade, correctness risk).** The fix is not one reshape:
it needs coordinated edits across (1) memory_utils.py:138-142 (reshape + re-spec mem_q/o_proj onto
k_dim/v_dim not the 4 heads), (2) memory.py:264,282 (change out_sharding head-dim 'model' → k_dim/
v_dim 'model'), (3) the retrieval path (mem_lookup_chunked / sharded_top_k bank shard-axis at tp=8),
all consistent — and a wrong sharding silently corrupts retrieval (wrong acc). Not safe to blind-edit
overnight unsupervised. Memory-layer B=1 @ tp=8 = **~400 tok/s (estimate, transformer-proxy 4×)**;
exact number needs the scoped re-sharding + acc-validation. This is a clean, well-defined follow-up.

## MEMORY-LAYER-SPECIFIC BS=1 levers (do NOT help MSA/RAG)
Already delivered (per-stream, B=8 1/chip proxy — single-stream): these touch ONLY the bank
lookup, so MSA/RAG get nothing from them:
- **bank sharding across 8 chips**: replicated→sharded, big.
- **on-device mem_v** (drop CPU pure_callback): 557→826 agg (+48%).
- **approx_max_k** (recall 0.95): 826→**921** agg (+12%) — i.e. the top_k op was ~12% of decode.
Net: **55→115 per-stream (2.1×)**, ALL memory-specific. (Baseline tk=128 best-config = 921 agg /
115 per-stream, measured clean.)

Further memory-specific squeezes:
- **mem_top_k reduction** (128→32): MEASURED **+3%** (921/115 → 952/119 per-stream, B=8 proxy).
  Small — the gather/merge is a minor slice. (MEM_TOP_K env knob added to memory.py. tk<16 crashes
  approx_max_k's min-k; use exact for tiny k.) Safe free win: lower the default top_k.
- ★PROFILE (measured, B=8 1/chip proxy, best config): bank SIZE barely matters — 5k-vec (md=20) =
  117 vs 512k-vec (md=2000) = 115 per-stream = only −1.6% for 100× more vectors. So the O(M) bank
  scan is NOT the BS=1 bottleneck (small corpora). The memory cost is the TOP_K/GATHER/MERGE
  machinery: tk=32 (119) beats even the tiny-bank tk=128 (117). → int8 mem_k would NOT help (bank
  load is tiny); the lever is TOP_K REDUCTION (delivered, +3%) and retrieval-reuse (skip the top_k/
  gather/merge on decode steps). NOTE: at LARGE corpora (msmarco 75k, triviaqa 10M) the M-scan DOES
  grow (pareto throughput 1774→488/575) — there approx/pruning/reuse matter much more.
- CEILING: after device-v+approx+sharding (2.1x done), the memory lookup is a SMALL single-digit %
  of BS=1 decode for typical corpora (4B transformer dominates). top_k=32 recovers +3%. Retrieval-
  reuse could recover the small remainder (~5-8%) but needs decode-path restructuring (pos is traced
  in the jitted decode loop → lax.cond compiles BOTH branches → no speedup unless a separate no-lookup
  decode trace) + acc validation. Diminishing returns for small corpora; worthwhile at 10M-scale.
- **retrieval-reuse**: cache prefill top_k, skip per-step O(M) scan+top_k+merge during decode
  (relevant docs fixed by the question). Biggest remaining memory-unique lever; needs a mem-cache
  in the jit decode carry. Acc risk (answer tokens reuse question retrieval).
- **int8 mem_k**: quantize bank keys (score matmul is small at B=1 though).
Ceiling: the memory lookup is only ~12-14% of BS=1 decode after device-v+approx (4B transformer
dominates), so remaining memory-specific headroom is single-digit-to-low-double-digit %.

## ANSWER (what improves BS=1 throughput — generic)
1. ★ **Tensor-parallelism tp=8** — proven **4.25×** on the base transformer (114→487 tok/s, B=1).
   THE lever: B=1 is weight-bandwidth-bound; shard the 8GB weights across 8 chips. Memory-layer
   inherits it once its forward is TP-clean (one reshape blocker, above).
2. **int8/fp8 weight quantization** — stack ~2× more (halves weight bytes/step). Complementary.
3. approx_max_k (done, small at B=1), int8 mem_k, lower mem_top_k — memory-lookup trims.
4. Batch scaling does NOTHING for B=1 (amortization needs batch).

## Plan
- Phase A (now, no new code): tp_devices sweep {1,2,4,8} @ B=1 on the BASE transformer (RAG,
  simplest — no bank) → does TP speed single-stream? Then on the memory-layer.
- Phase B: if TP helps, apply to memory-layer @ B=1 (interacts with bank sharding — bank goes
  replicated when data-axis=1; fine for small corpus).
- Phase C: int8 weight quant (if time) — stack on the TP winner.

Env: SINGLE_DEVICE=1 (1×1 mesh, tp=1 single chip baseline); tp_devices=N (Hydra) for TP.
Bench: scripts/embed/rag_repo_speed_bs1.py (base, B=1), run_speed_bench_bs1.sh (memory-layer).

## BS=1 single-chip (no-TP) per-component profile (measured, micro-bench)
Regime: B=1, ONE chip, KV=2560, Qwen3-4B 36L, mem@L14, full 512k-vec bank (NOT sharded — no data-parallel at 1 chip).
Grouped (µs, un-fused micro-bench → ~2x slower than real fused decode; use PROPORTIONS):
- layers 0-13 (before): 6019 (33%)
- layer14 transformer attn+mlp: 430 (2.4%)
- layer14 MEMORY total: 1674 (9.3%)
- layers 15-35 + lm_head (after): 9740 (54%)
- embed: 136
Memory sub-ops (µs): score_scan 826 (49% of mem — bandwidth-bound loading the full 1GB bank/step at 1 chip) >
  mem_o_proj 216 > approx_topk 207 > mem_q_proj 147 > gather 140 ~ softmax_combine 139.
Per-op: one_full_layer 430 (attn 184 + mlp 246); lm_head 712.
Transformer ~90% of step; memory ~9%. est 56 tok/s un-fused (vs measured ~115 at B=8/8-chip SHARDED bank —
different config: 1-chip has the FULL-bank scan, 8-chip shards it to 64k/chip so scan is ~tiny there).
KEY: at literal 1-chip BS=1 the memory cost is dominated by the full-bank score-scan; shrinking what the
scan loads (int8/int4 mem_k, product-keys √M, pooling) is the memory-specific lever (bank sharding needs >1 chip).
