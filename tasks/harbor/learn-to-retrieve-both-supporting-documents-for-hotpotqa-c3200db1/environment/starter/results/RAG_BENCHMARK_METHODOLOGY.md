# Emulating a realistic production RAG environment for benchmarking (vs the memory-layer)

Research-backed (2026) design for a REALISTIC prod RAG baseline + how to measure it. The point:
the current repo "RAG" (exact top-5 over a 2k–75k-doc HF slice, in-JAX) is a TOY — it gives RAG
unrealistically perfect retrieval on a tiny corpus. Real prod RAG runs an **ANN index over
1M–1B vectors** at **recall 0.90–0.95** with a **two-stage retrieve→rerank** pipeline, and that's
where the fair comparison against the memory-layer lives (and where the memory-layer OOMs today).

## What "realistic" means (researched numbers)

**1. Corpus scale.** Prod RAG typically **1–100M vectors/chunks**. `pgvector` is the default under
~10M chunks and viable to ~50–100M; beyond that (or web-scale, billions) purpose-built DBs (Milvus,
Qdrant) win. Research benchmarks sweep scale: EnterpriseRAG-Bench uses 5k / 16k / 50k / 160k / 512k
docs. → Benchmark at **a scale sweep: 100k → 1M → 10M → 100M vectors**, not a fixed 2k.

**2. Vector DB.** `pgvector`(+pgvectorscale) for <50M (hit **471 QPS @ 99% recall on 50M×768-d**);
**Qdrant** (~**6ms p50**) or **Milvus** (billion-scale) above ~100M; **FAISS** in-process for a
reproducible, dependency-light benchmark. For an apples-to-apples research bench, FAISS-HNSW or
Qdrant are the cleanest.

**3. ANN index (HNSW) + recall target.** Real-time: **M=12, efConstruction=100, efSearch=50**.
Balanced default: **M=16, efConstruction=200**. High-recall: **M=24, efC=400, efSearch=200–500**.
efSearch must be ≥ k. Target **recall@10 ≈ 0.90–0.95** (NOT 1.0 — this ANN recall gap is THE
realistic difference from the repo's exact retrieval, and where prod RAG actually loses accuracy).

**4. Chunking + embeddings.** **400–512 tokens, 10–20% overlap**, recursive splitting (≈85–90%
recall @400 tok; semantic chunking 91–92%). Match the embedding model's input window. Common embedders:
OpenAI text-embedding-3, Cohere embed-v3, BGE-large, E5, Qwen3-Embedding. → Keep **Qwen3-Embedding-0.6B**
(your current model) for consistency with the memory-layer's encoder, or add BGE-large for realism.

**5. Two-stage retrieve→rerank (the prod pattern).** ANN-retrieve **top 20–50**, then **rerank down
to top 5–10** with a cross-encoder (BGE-reranker-v2, ms-marco-MiniLM, or Cohere Rerank 4.0).
Reranking: **+20–35% accuracy, +100–500ms latency**. For <200ms SLAs use FlashRank or skip on easy
queries. Your current single-stage top-5 skips this — real RAG is two-stage.

**6. Latency / throughput SLOs.** Retrieval p50 ~**6ms** (Qdrant), reranker **+100–500ms**, LLM gen
dominates e2e. Real-time SLA target **sub-200ms** retrieval+rerank (gen on top). Report **QPS at a
fixed recall** (e.g. 471 QPS @99% recall/50M is a real datapoint). Measure **p50/p95/p99**, not just mean.

**7. Standard eval benchmarks + metrics.**
- Retrieval quality: **BEIR** (18 datasets, report **nDCG@10** + **recall@100**), **MS MARCO**
  (nDCG@10, MRR@100), **MTEB** (embeddings). These give the accepted retrieval numbers.
- Generation quality: **RAGAS** (**faithfulness**, **context precision**, **context recall**,
  **answer relevance/correctness**) via an LLM judge — plus your existing binary LLM-judge accuracy.
- End-to-end: recall@k + LLM-judge answer accuracy + latency percentiles + QPS + $/1M-queries.

## Recommended concrete measurement environment (for the memory-layer comparison)

Run BOTH systems on the **same corpus + same query set + same gold labels**:

| component | choice |
|-----------|--------|
| corpus | MS MARCO v2 passages (~138M) or a Wikipedia/enterprise corpus; **scale-sweep 100k/1M/10M/100M vectors** |
| chunking | 512 tok, 64-tok (12.5%) overlap, recursive |
| embedder | Qwen3-Embedding-0.6B (match memory-layer encoder) [+ BGE-large-en-v1.5 for a realistic 2nd point] |
| index | FAISS-HNSW or Qdrant; HNSW **M=16, efConstruction=200**, sweep **efSearch** to hit recall@10 ∈ {0.90, 0.95, 0.99} |
| retrieval | ANN top-50 → cross-encoder rerank (BGE-reranker-v2-m3) → top-5 into the reader |
| reader | Qwen3-4B (same as the RAG baseline / memory-layer base) |
| judge | Qwen3-8B binary LLM-judge (your current) + RAGAS faithfulness/context-recall |

**Metrics to report (per scale point):**
1. Retrieval: recall@{5,10,100}, nDCG@10, MRR; **ANN recall vs exact** (the ANN quality gap).
2. Answer: binary LLM-judge accuracy + RAGAS faithfulness + context-recall.
3. Throughput/latency: **QPS at recall@10=0.95**; retrieval p50/p95/p99, rerank p50/p99, e2e p50/p99.
4. Cost/footprint: index build time, index RAM (GB) [10M×1024-d ≈ 40GB], $/1M queries.

**Why this makes the comparison fair vs the memory-layer:**
- Real RAG at scale has **recall < 1.0** (ANN) — its accuracy drops from the repo's exact-top-5
  numbers. Measure that drop at 1M/10M/100M.
- Real RAG **latency = ANN(≈6ms) + rerank(100–500ms) + gen**; the memory-layer has no separate
  retrieval stage but pays per-decode-step bank cost. Compare **e2e p50/p99 + QPS**, not just decode tok/s.
- The **10M–100M-vector regime is exactly where the memory-layer OOMs today** (bank in HBM) while
  prod RAG (disk/HNSW) runs fine — so this environment surfaces the real architectural tradeoff:
  RAG scales sub-linearly on a vector DB; the memory-layer must hold the corpus in accelerator memory.

## Sources
- Vector DB scale/QPS/pgvector: production decision guides + Timescale/pgvectorscale benchmark.
- HNSW params/recall: Zilliz/Milvus HNSW parameter references.
- Chunking: Chroma/Firecrawl/Weaviate chunking studies (400–512 tok, 10–20% overlap).
- Rerank: cross-encoder/Cohere rerank latency+accuracy guides.
- Benchmarks/metrics: BEIR, MS MARCO, MTEB, RAGAS docs, EnterpriseRAG-Bench.
