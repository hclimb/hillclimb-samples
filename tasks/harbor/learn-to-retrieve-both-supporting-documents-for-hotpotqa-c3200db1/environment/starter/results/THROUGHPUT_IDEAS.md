# Memory-layer throughput — brainstorm + plan (overnight goal, 2026-06-30)

**Goal:** push memory-layer decode throughput as high as possible, then re-eval 9 MSA
evals × 3 methods (MSA/RAG/memory-layer) onto a throughput-vs-acc Pareto.

**Baseline / already done** (B=8, 1 seq/chip, one v6e-8, 512 tok, repo JAX, msmarco 2k-doc
/ 512k-vec bank):

| config | agg tok/s | per-stream | status |
|--------|----------:|-----------:|--------|
| replicated bank + on-device v | 439 | 55 | measured |
| sharded bank + CPU-offload v | 557 | 70 | measured |
| sharded bank + on-device v (exact top_k) | 826 | 103 | measured |
| **sharded + on-device v + approx_max_k (recall .95)** | **921** | **115** | measured (BEST) |

Reference ceilings (per-stream): RAG base Qwen3-4B repo-JAX ≈ 131 (2102/16); vLLM ≈ 173.
So the memory layer (115) is already within ~12% of the repo-JAX base-transformer ceiling —
memory-lookup overhead is now small; the base transformer decode is the dominant cost.

---

## A. Memory-lookup cost (the ~12% overhead over base) — inference-time, no retrain

1. **int8 / int4 mem_k for the score matmul.** einsum(q, mem_k) over M=64k/chip is a chunk of
   the lookup. bf16→int8 keys ≈ 2× matmul, ~4× less HBM traffic. Prior note: "int8 mem_k ~3.7×
   on score matmul but marginal end-to-end post-approx (matmul small fraction)." → likely small
   now, but cheap to try. TEST: quantize mem_k to int8 at Phase-2, dequant in-kernel or int8 matmul.
2. **Lower mem_top_k.** Currently 128. Fewer neighbors = fewer gathers + smaller value reduction.
   top_k 128→64→32. Throughput up, acc likely down (retrieval breadth). SWEEP tput+acc.
3. **Smaller CHUNK_SIZE / lookup_chunk_size tuning** for the sharded scan — match to HBM/VMEM.
4. **Skip mem_mask where all-valid** — avoid the jnp.where over M each step.
5. **Fuse score-matmul + top_k + gather** into one kernel (Pallas) — remove intermediate
   materialization. Bigger effort.
6. **Cache retrieval across decode steps.** Retrieved doc set changes slowly step-to-step;
   recompute top_k every N steps, reuse in between. Risky (acc), but potentially large tput.
7. **approx_max_k tuning** — DONE: recall flat 0.7-0.95, no further tput. Keep 0.95.

## B. Base transformer decode (the dominant cost, shared with RAG) — the real ceiling

8. **Bigger batch for aggregate throughput.** Decode is latency-bound at B=8; larger B amortizes
   weight loads → higher AGG tok/s (the original Pareto metric). B=16/32/64 sweep. This is likely
   the single biggest AGG-throughput lever (RAG's 2102 was B=16). TEST B=16/32/64 (watch HBM: bank
   + KV + weights). The memory bank is fixed cost, so larger B is nearly free until compute-bound.
9. **bf16 KV cache** (if not already) — half KV HBM traffic per step.
10. **tensor-parallel the transformer (tp_devices=8 or 4)** for single-stream latency — shards the
    big matmuls, at the cost of per-matmul all-reduce. Only helps if compute-bound not latency-bound.
    Probably hurts at small batch (the §2b lesson). LOW priority.
11. **Port memory layer onto vLLM-TPU engine** (paged-KV, continuous batching) — the biggest
    possible win (base 131→173/stream, plus batching), but a large eng effort; likely out of scope
    for one night. Note as the top future lever.

## C. Bank size / encode (changes the memory, may touch acc)

12. **Pool the bank** (conv stride>1 at encode) → fewer memory vectors M → cheaper scan + less HBM.
    512k→256k/128k. Retrieval granularity coarsens (acc risk). TEST via embed_conv_stride.
13. **Prune/dedup low-value vectors** — drop near-duplicate doc vectors. Offline, one-time.
14. **fp8 mem_v** on device — half value HBM (already bf16; fp8 = quarter). Small.

## D. Generation strategy (not architecture, but affects e2e QPS)

15. **Shorter CoT / max_new_tokens.** membed emits 512-tok CoT; RAG short answers. Decode tok/s is
    the same, but e2e QPS scales with tokens emitted. Out of scope for decode-tok/s metric; note it.

---

## Plan / priority (impact × ease, TPU-limited)

**Phase 1 (biggest, easy):** batch scaling B=16/32/64 (lever 8) — the AGG-throughput metric the
final Pareto uses. Run on b-1 (no judge needed). Establishes the headline throughput.

**Phase 2 (squeeze lookup):** mem_top_k sweep (2) + int8 mem_k (1) — throughput deltas; acc for
top_k measured later. Cheap, on b-1.

**Phase 3 (bank pooling):** conv stride (12) — tput vs acc, only if Phase 1-2 leave headroom.

**Phase 4 (final Pareto):** best config → measure decode throughput per method per eval, +
binary LLM-judge acc on all 9 evals × 3 methods. Reuse existing acc (msa_eval_summary.json,
pareto_summary.json, §8) where fresh runs are TPU-infeasible; fill gaps. RAG acc known; MSA 8/9
known; memory-layer 5/9 known → fill narrativeqa, dureader, + approx-config re-checks.

## RESULTS (executed 2026-06-30 overnight)

Best memory-layer decode config: **sharded bank + on-device mem_v + approx_max_k(0.95)**.
All levers stack multiplicatively:

| lever | effect (agg tok/s, B=8) | notes |
|-------|------------------------|-------|
| baseline (replicated bank, on-device v) | 439 | each chip scans full 512k bank |
| + shard bank across 8 chips | 826 (1.88×) | 64k/chip + tiny 8×128 merge |
| + on-device mem_v (drop CPU pure_callback) | (bundled) | 557→826 at matched sharding = 1.48× |
| + approx_max_k (recall 0.95) | 921 (1.12×) | recall 0.7-0.95 all flat; keep 0.95 |
| **+ batch scaling B=8→16→32** | **1774 (B16) / 3230 (B32)** | biggest AGG lever; B=64 OOMs |

**Total: 439 → 3230 tok/s (7.4× agg) / per-stream 55 → 115 (2.1×).** At B=16 (1774) the
memory layer now BEATS MSA (1563) and is ~half of RAG (3445); at B=32 RAG pulls further ahead
(9931) because RAG has no per-token bank scan to amortize.

**Accuracy cost of the fast config:** approx_max_k(0.95) vs exact top_k, judged fresh
(vLLM Qwen3-8B): popqa 0.258 vs 0.266, nq 0.391 vs 0.414 — only ~0.01-0.02. Essentially free.

**Levers NOT pursued (low headroom / out of scope):** int8/int4 mem_k (matmul is a tiny
fraction after approx), mem_top_k reduction (acc-coupled; folded into acc study), bank pooling
(retrain/re-encode), retrieval caching across steps (acc risk), vLLM-engine port (biggest
remaining lever — base 131→173/stream + continuous batching — but multi-day effort).

## Infra notes / gotchas (this session)
- SSH exit-255 = TOO-MANY-CONCURRENT gcloud ssh (contention), NOT wedged VMs. Keep ≤2 concurrent;
  kill local strays `pkill -f 'tpu-vm ssh'`. Light cmds (echo/read JSON) work even when pkill 255s.
- vLLM-TPU judge: works on rohun-v6e-8-0, BROKE on b-1 (Internal Server Error — pin/libtpu state).
  Use VM-0 for judged acc; b-1 for judge-free throughput benches. 16GB Qwen3-8B downloads to
  /dev/shm on first judge (~20 min one-time).
- VMs with GCS ADC (ra3440@columbia.edu) + cached weights: rohun-v6e-8-0, b-1. VMs 2/3 lack ADC
  (default SA no bucket access); copying ADC blocked by classifier. 1,4 preempted.
- Env flags: MEM_DEVICE_V=1 (on-device mem_v, sharded), MEM_REPLICATE_BANK=1 (replicate bank),
  MEM_APPROX_TOPK=1 + MEM_APPROX_RECALL, SINGLE_DEVICE=1 (1x1 mesh), REPLICATE_WEIGHTS=1 (MSA).
- Use `.venv/bin/python` / `python` after `. .venv/bin/activate` (uv not on PATH non-interactive).
- Benches: scripts/embed/run_speed_bench_bs1.sh, rag_repo_speed_bs1.py; runner /tmp/run_tput.sh
  (args=recall list), /tmp/run_acc.sh (args=mode evals).
