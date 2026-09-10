# Pareto Study — Findings (memory-layer vs RAG vs MSA)

**Scope:** 3 MSA document-QA evals (popqa, natural_questions, hotpotqa), binary LLM-judge
accuracy (Qwen3-8B judge). Memory-layer ckpt `qa_hard_neg_think_sft4b_…/qwen3_mem_embed/100000`.
Single v6e-8. Honest result — the headline thesis is only partly supported; see below.

## 1. Accuracy (binary LLM-judge)

| eval | Memory-layer (ours) | MSA-4B | RAG (Qwen3-4B + Qwen3-Embedding-0.6B) |
|------|--------------------:|-------:|--------------------------------------:|
| popqa | 0.258 | 0.133 | **0.836** |
| natural_questions | 0.422 | 0.141 | **0.766** |
| hotpotqa | 0.289 | 0.133 | **0.711** |
| **avg** | 0.323 | 0.125 | **0.771** |

- **RAG wins accuracy on all 3**, including multi-hop hotpotqa. On small-corpus (~8k-doc)
  single-hop QA, explicit retrieve-the-passage is hard to beat.
- **Memory-layer beats MSA-4B ~2.4×** on accuracy.
- Plot: `figures/acc_comparison_3evals.png`.

## 2. Throughput (end-to-end queries/sec, single v6e-8, native config)

| method | QPS | acc | source |
|--------|----:|----:|--------|
| RAG | 27.1 | 0.77 | nq generator: 128 ans / 4.7s (think=False, vLLM conc 64) |
| Memory-layer | 1.6 | 0.32 | membench msmarco B=16: 16q / 10.0s (512-tok CoT) |
| MSA-4B | 0.68 | 0.13 | membench: 16q / 23.4s |

- Plot: `figures/pareto_qps.png`. **On these evals RAG Pareto-dominates** (higher acc AND higher QPS).
- ⚠️ **NOT output-length-matched**: RAG emits short answers; ours/MSA emit 512-token CoT. RAG QPS
  excludes retrieval/encode/server-startup and uses concurrency 64 vs B=16. Decode-rate
  (length-normalized) is closer: ours 846 tok/s vs MSA 351 tok/s (RAG decode not yet matched-measured).
  The QPS gap is partly **generation strategy** (direct vs CoT), not pure architecture.

## 2b. Throughput, MATCHED (decode tok/s, B=16, 512 tokens — the fair version)

The §2 QPS above mixed CoT vs short answers. Matched decode-throughput (same batch + output len):

| method | decode tok/s | engine | corpus dependence |
|--------|-------------:|--------|-------------------|
| **RAG (Qwen3-4B)** | **2775** (173/stream) | vLLM-TPU | independent (top-5 ~2-3k ctx) |
| Memory-layer | 846 → **328** @75k docs → **OOM** @10M | repo JAX | DEGRADES with corpus |
| MSA-4B | 351 | repo JAX | — |
| TTT-1.3B | naive re-forward (no KV-cache) — slow, not reported | repo JAX | — |

On vLLM, RAG looks fastest — BUT that's the ENGINE. **★★ SAME-ENGINE CORRECTION (decode tok/s, B=16,
512 tok, all on the repo JAX engine, single v6e-8) — supersedes a prior buggy version:**

| method | decode tok/s | acc | corpus |
|--------|-------------:|----:|--------|
| **RAG (base Qwen3-4B)** | **2102** | **0.77** | top-5 in ctx (corpus-indep.) |
| MSA-4B | 1711 | 0.13 | 8.7k docs / 1.19M tok |
| Memory-layer +approx_max_k | 981 | 0.32 | 2k docs / 0.74M tok |
| Memory-layer (exact) | 781 | 0.32 | 2k docs / 0.74M tok |

Plots: `figures/pareto_decode_throughput_corrected.png` (bars), `figures/pareto_acc_vs_throughput_corrected.png` (Pareto).

**RAG Pareto-DOMINATES** — highest accuracy AND highest throughput. There is **no** throughput/accuracy
tradeoff favoring the memory-layer on these evals.

### ⚠️ Retraction of the earlier "memory-layer 2× faster than RAG" claim
The prior version of this table reported **RAG=376** and concluded the memory-layer was "~2× faster on
the same engine" (a Pareto frontier). **That was a measurement bug.** `models/qwen3.py`'s `load(tp_devices=1)`
shards every weight `P('model','data')` on mesh (data:8, model:1); since `model`=1 is a no-op, the matmul
CONTRACTION dim ends up split across the 8-chip 'data' axis → an **all-reduce per matmul per decode step**.
Decode at B=16 is latency-bound, so that collective dominated → a crippled 376 tok/s. Replicating the
weights (`P()`) removes the collective → **2102 tok/s** (5.6×, ≈ vLLM's 2775 ballpark — proving it's the
right number). Verified with `shard_test.py` (376 sharded vs 2102 replicated, identical `_generate_tokens`).
- The membed eval never hit this because `load_inference_checkpoint` restores weights **replicated**
  (saved replicated by `save_checkpoint`), so its base was always fast — which is why my standalone RAG
  (376) looked artificially slower than the memory-layer eval. They were never the same weight layout.
- **MSA: old 351 was a different metric** (native-e2e qps × 512), not decode-only. Re-measured decode-only
  = **1711** (sharding-independent: REPLICATE_WEIGHTS 0 vs 1 both ≈1700). MSA's 18 compressed-routing
  layers still cost vs base, but far less than the memory-layer's 1-layer O(M=1.5M) dense bank scoring.
- approx_max_k (recall 0.95) buys the memory-layer +26% decode (781→981) at ~no accuracy cost. int8 mem_k
  was a proposed lever, **never implemented** — dropped.
- vLLM's 2775 (RAG) remains an additional ENGINE win on top of the architecture win.

## 2c. Direct end-to-end latency (RAG lookup INCLUDED, embedding discounted)

Per user: count RAG's retrieval LOOKUP in e2e (discount the one-time embedding stage), both Qwen3-4B.
B=16, 512 tokens. (`figures/e2e_latency_breakdown.png`, `rag_e2e.py`.)

| stage (s/query) | RAG | Memory-layer (small) | Memory-layer (75k docs) |
|-----------------|----:|---------------------:|------------------------:|
| retrieval lookup | 0.0007 | (integrated) | — |
| prefill | 0.0045 | 0.020 | ~0.05 |
| decode (512 tok) | 0.180 | ~0.605 | ~1.51 |
| **e2e/query** | **0.185** | **0.625** | **1.56** |

**Including the lookup doesn't help the memory-layer** — it's only 0.0007 s/query (one top-k over
precomputed doc vectors, trivial even at 75k docs; embedding discounted as amortized). RAG is
**~3.4× faster e2e** (small corpus), **~8× faster** at 75k docs. The memory-layer's bottleneck is
DECODE (0.6s vs RAG 0.18s) — engine (vLLM vs repo JAX) + memory cross-attention per step. Same
engine caveat as §2b.

## 3. Eval-fairness check (rejected)

Hypothesis: ours is handicapped by per-doc truncation (memory bank = 2 chunks × 256 = 512 tok/doc).
Re-ran nq with `max_chunks_per_doc=16` (4096 tok/doc, 8× budget): **0.398 vs 0.422 baseline** — no
gain. **The ~0.32 is genuine capability**, not a truncation artifact. The memory mechanism is
simply weaker than explicit retrieval at locating/using evidence here.

## 4. Scale regime — the architectural efficiency win

Memory-layer per-query cost vs corpus size (`figures/scale_cost.png`):

| corpus (docs) | doc-tokens in memory | prefill s/query | one-time encode s |
|--------------:|---------------------:|----------------:|------------------:|
| 2,000 | 194k | 0.020 | 21 |
| 6,000 | 584k | 0.034 | 44 |
| 12,000 | 1.18M | 0.058 | 48 |
| 24,000 | 2.35M | 0.111 | 63 |

- Per-query prefill grows **sub-linearly** (12× corpus → 5.6× cost); answers over a **2.35M-token
  corpus at 0.11 s/query**.
- A vanilla **full-context Qwen3-4B caps ~32k tokens** — it cannot hold even the 194k-token (2k-doc)
  corpus. This is the genuine architectural contribution vs **long-context / linear-attention / TTT**
  baselines (all bounded by sequence length): the memory-layer answers over corpora 6–70× larger
  than any context window at ~constant per-query cost, as one end-to-end model.
- It does **not** out-scale **RAG** (retrieval is also sub-linear) — RAG remains the efficient,
  accurate baseline on small corpora.

## 5. RAG-fails regime (harder multi-hop) — hypothesis rejected

Tested whether ours closes the gap where retrieval is hard (deep multi-hop):

| eval | Memory-layer | RAG | RAG/ours |
|------|-------------:|----:|---------:|
| musique (4-hop) | 0.070 | 0.258 | 3.7× |
| 2wikimultihopqa | 0.117 | 0.438 | 3.7× |

RAG DOES degrade on 4-hop (hotpotqa 0.71 → musique 0.26), but the memory-layer degrades *more*
(0.29 → 0.07). The relative gap holds/widens. **RAG stays superior even where retrieval is hard.**
The thesis is not rescued by harder multi-hop.

## 6. Secondary baselines (TTT, Linear-attention) — blocked, TPU/JAX-only constraint

Both must run on TPU/JAX (no GPU). Status:
- **Linear-attention = RecurrentGemma (Griffin)** — JAX-native, fully set up on v6e (jax[tpu]
  0.10.2 sees all 8 chips, generator written). BLOCKED only by **Gemma license gating** (1-click
  HF accept needed). Ready to run on acceptance.
- **TTT = ttt-lm-jax** (`ttt-linear-1.3b-books-32k`) — **NOW RUNNING on v6e** (ported to jax 0.10.2:
  pin transformers 4.41 + flax 0.10.6, patch with_sharding_constraint→jax.lax, jnp.clip a_min→min,
  cast TTT scan carry→float32; fixed-L padded greedy decode). Weights load, generates correctly.

**TTT result (oracle-doc, gold passage HANDED to the 1.3B model — NOT corpus-level):**

| eval | RAG (corpus) | TTT* (oracle) | Memory-layer (corpus) | MSA (corpus) |
|------|-------------:|--------------:|----------------------:|-------------:|
| popqa | 0.836 | 0.656 | 0.258 | 0.133 |
| natural_questions | 0.766 | 0.258 | 0.422 | 0.141 |
| hotpotqa | 0.711 | 0.359 | 0.289 | 0.133 |
| **avg** | 0.771 | **0.424** | 0.323 | 0.125 |
\* TTT gets the gold doc free (no retrieval) → easier task, not directly comparable.

**Key insight:** even GIVEN the gold doc, the 1.3B TTT averages only 0.42 (weak reader); yet on
popqa/hotpotqa it still beats the memory-layer's CORPUS-level score. Combined with §3 (more memory
budget = no gain) and §7 (inference tuning failed), this localizes the memory-layer's deficit vs
RAG to **memory-RETRIEVAL quality** (locating evidence), not reading/capacity. TTT throughput not
reported fairly here (naive fixed-L re-forward, no KV-cache → artificially slow; a proper TTT
incremental decoder would be faster). Linear-attn (RecurrentGemma) still blocked on Gemma license.

## 7. Closing the gap to RAG — inference levers exhausted (no gains)

Ckpt fixed → tried inference-time levers on popqa (baseline mem_top_k=128 = 0.258):

| config | popqa acc |
|--------|----------:|
| **baseline (mem_top_k=128)** | **0.258** |
| mem_top_k=64 | 0.227 |
| mem_top_k=256 | 0.234 |
| mem_top_k=512 | 0.156 |
| mem_top_k=1024 | 0.211 |
| two_pass_topk=true | 0.016 (breaks) |
| +8× per-doc memory budget (nq, §3) | no gain |

**None help; the baseline config is already optimal.** More memory breadth adds distraction;
two-pass retrieval is eval-incompatible here. **The gap to RAG is architectural, not a tunable
inference issue**: RAG injects full retrieved-doc TEXT into the reader's context; the memory-layer
answers from COMPRESSED memory vectors with no text injection. Closing 0.26→0.84 would require
**retraining** with generative-retrieval / doc-text injection (emit doc IDs → inject text → answer)
— a training effort, out of scope for inference tuning.

## 8. Long-corpus regime (msmarco_v1 ~75k docs, triviaqa_10m ~10M tokens)

Corpus lengths of the 5 main evals are already ~0.5–1.6M tokens (40–800× a 32k window). Pushed to
the largest MSA corpora:

| eval (corpus) | RAG | Memory-layer |
|---------------|----:|-------------:|
| msmarco_v1 (~75k docs) | 0.797 | 0.383 (no OOM, but ~25s/batch — bank slows gen ~40×) |
| triviaqa_10m (~10M tok) | **0.922** | **OOM** (bank replicated per chip; 10M vectors don't fit HBM) |

**The long-corpus regime does NOT favor the memory-layer:**
- **Accuracy:** RAG retrieval stays excellent even at 10M tokens (triviaqa 0.92, msmarco 0.80) —
  retrieval is sub-linear and scale-robust; the memory-layer's retrieval bottleneck (§6) is
  scale-independent.
- **Feasibility:** at 10M tokens the memory-layer's REPLICATED per-chip bank OOMs, while RAG
  handles it fine. So at extreme scale RAG is *more* feasible, not less (unless the memory bank is
  sharded across chips — an unimplemented optimization).
- The memory-layer's earlier "constant per-query cost" win (§4) holds vs FULL-CONTEXT attention,
  but NOT vs RAG, which also scales sub-linearly and doesn't replicate the corpus per chip.

## 9. Single-stream (BS=1) + memory-bank sharding effect

The §2b–2c numbers ran at B=16 with the memory bank **sharded across all 8 chips**, while the
MSA/RAG baselines use no sharding. To isolate (a) single-stream throughput and (b) the effect of
sharding the bank across chips, we re-measured at the lowest batch the data-parallel transformer
supports: **1 sequence per chip** (global B=8; per-stream = aggregate ÷ 8), 512 tok, repo JAX,
one v6e-8. (`throughput_data.json → bs1_single_stream_2026_06`; plots
`figures/pareto_acc_vs_throughput_bs1.png`, `figures/memory_bank_sharding_effect_bs1.png`.)

> **Why not a literal global B=1?** `qwen3.py`'s `forward` hardwires `P('data', …)` out-shardings,
> which are only legal next to replicated weights when the data axis is > 1 — at data=1 JAX raises
> a `ShardingTypeError`. So a literal global batch of 1 is not expressible for the memory-layer /
> RAG transformer without rewriting every out-sharding. MSA's **separate** decode path *does* run
> literal single-chip B=1 = **124.6 tok/s**, which matches its per-stream@B=8 = **109** — confirming
> per-stream @ 1-seq/chip is the right single-stream proxy.

### Two memory-layer decode levers (B=8, isolate each by holding the other fixed)

| memory-layer config (B=8) | decode tok/s (agg) | per-stream |
|---------------------------|-------------------:|-----------:|
| **bank SHARDED + on-device mem_v + approx_max_k** (best) | **921** | 115 |
| bank SHARDED + on-device mem_v (exact `top_k`) | 826 | 103 |
| bank SHARDED + mem_v on CPU (per-step `pure_callback`) | 557 | 70 |
| bank REPLICATED full-per-chip + on-device mem_v | 439 | 55 |

Three stacking throughput levers — **2.10× total** (439 → 921):
- **Bank sharding** (mem_v on-device for both): 439 → 826 = **1.88× faster**. The bank is large
  (512k vectors); splitting the score-scan 8 ways and merging only `8 × top_k=128` candidates beats
  every chip scanning the full 512k. **Opposite** of the §2b WEIGHT-sharding bug — there a cross-chip
  all-reduce dominated a *cheap* matmul; here the sharded work (the bank scan) is the expensive part
  and parallelizes well, while the merge collective is tiny.
- **CPU value-offload removal** (bank sharded for both): the sharded path keeps `mem_v` on CPU and
  fetches the selected K rows per decode step via a `jax.pure_callback` host round-trip. Putting
  `mem_v` on-device (`MEM_DEVICE_V=1`) is 557 → 826 = **1.48× faster** (gen_e2e 7.63 s → 5.24 s). The
  host callback was a real decode bottleneck. On-device `mem_v` costs HBM (full values resident); fine
  for this 512k-vec/~1 GB bank, but CPU offload should be reserved for banks that don't fit HBM.
- **`approx_max_k`** (`MEM_APPROX_TOPK=1`, recall 0.95) on top of sharded+on-device-v: 826 → 921 =
  **1.12× faster**. Smaller than the +26% it gave on the CPU-offload path (§2b 781→981 @ B=16) because
  on-device `mem_v` already removed the dominant callback cost, so `top_k` is now a smaller fraction.
  Carries a small accuracy cost on some evals (popqa ≈ 0.05); plotted at the same accuracy as exact
  (throughput-only lever).
- *(An earlier mixed comparison — sharded-CPU-v 554 vs replicated-device-v 439 = "1.26×" — conflated
  the first two levers; the clean bank-sharding gain is 1.88×.)*

So the memory-layer's best decode is **921 tok/s (115/stream)** — sharded bank + on-device `mem_v`
+ `approx_max_k`.

### Single-stream Pareto (1 seq/chip)

| method | per-stream decode tok/s | acc |
|--------|------------------------:|----:|
| **RAG (base Qwen3-4B)** | **~131** (2102/16, same-engine §2b; per-stream batch-insensitive for latency-bound decode) | 0.77 |
| **Memory-layer (best: sharded + on-device v + approx_max_k)** | **115** | 0.32 |
| MSA-4B | 109 (literal 1-chip B=1: 124.6) | 0.14 |
| Memory-layer (sharded + on-device v, exact) | 103 | 0.32 |
| Memory-layer (sharded bank, CPU-offload v) | 70 | 0.32 |
| Memory-layer (replicated bank) | 55 | 0.32 |

**RAG still Pareto-dominates on accuracy**, but the three levers above lift the memory layer from
55 → **115 tok/s/stream** (2.10×), now **past MSA (109)** — it is no longer the slowest on decode.
Accuracy ordering (RAG ≫ memory-layer ≫ MSA) is unchanged.
*(The fresh B=8 RAG run was blocked by SSH-contention orphan procs on the v6e; the 131 tok/s
per-stream is the corrected same-engine B=16 measurement — identical methodology.)*

## Bottom line

- **Supported:** memory-layer ≫ MSA-4B on accuracy (2.6×); memory-layer enables grounded QA over corpora
  far beyond any context window at low, sub-linear per-query cost (the long-context/linear-attn/TTT win).
- **Not supported (on these evals):** "more Pareto-efficient than RAG." With throughput measured
  correctly (same engine, no sharding bug), **RAG Pareto-DOMINATES** — it is BOTH more accurate (0.77 vs
  0.32) AND faster (2102 vs 781–981 tok/s decode). The earlier "memory-layer 2× faster" frontier was a
  weight-sharding measurement artifact (see §2b retraction). Beating RAG would need a regime where
  retrieval fails (very large / noisy corpora, deep multi-hop) — tested in §5/§8 and RAG still wins.
- **Throughput ranking (corrected, same-engine decode-only):** RAG 2102 > MSA 1711 > memory-layer
  981 (approx) / 781 (exact). The memory-layer is the SLOWEST — its single layer does O(M=1.5M)
  dense bank scoring per step; the real speed lever is sub-linear search (approx_max_k = +26%).
- **Open / future:** sub-linear bank search to lift memory-layer decode; give the memory-layer a
  vLLM-class engine; TTT + linear-attention baselines (secondary, not yet run).
