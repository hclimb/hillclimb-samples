# Long-document serving cost: RLM vs RAG on LongHealth and QASPER

**Status: complete for the throughput axis (2026-07-19). No accuracy is reported on either
benchmark — see [Scope](#scope-what-this-does-and-does-not-show).**

## Conclusion

**On QASPER, the memory layer serves a query in 1.90 s against RAG@5's 4.43 s — 2.33× faster —
with the whole 6.3M-token corpus indexed and a question-only prompt.** On LongHealth the same
comparison is a narrow win (1.69 s vs ~1.96 s, 1.16×). The difference between the two benchmarks is
document length: QASPER's papers average 5,404 tokens against LongHealth's 1,752, so RAG@5 must
prefill 27,084 tokens on QASPER versus 8,824 on LongHealth, while the memory layer prefills 64
either way.

This is the mirror image of the MuSiQue result
([2026-07-19-musique-corpus-scaling…](2026-07-19-musique-corpus-scaling-and-throughput-pareto.md)),
where 256-token documents left RAG's prompt at ~1.3k tokens and RAG won both axes. The governing
variable is **tokens RAG must place in context**, not corpus size:

| benchmark | tokens/doc | RAG@5 prompt | RAG@5 | RLM (best measured) | winner |
|-----------|-----------|--------------|-------|---------------------|--------|
| MuSiQue   | 256       | 1,344        | 1.73 s | 2.09 s *(4 layers)* | RAG 1.21× |
| LongHealth| 1,752     | 8,824        | ~1.96 s | 1.69 s *(1 layer)*  | RLM 1.16× |
| QASPER    | 5,404     | 27,084       | 4.43 s | **1.90 s** *(1 layer)* | **RLM 2.33×** |

The MuSiQue row is a 4-layer measurement — no 1-layer point was taken at its 524k bank, and since
layer count matters more than the ladder at large banks, a 1-layer MuSiQue number would likely be
faster than 2.09 s. The RAG-wins conclusion there does not depend on it (RAG's 1.73 s already beats
the 1-layer *LongHealth* figure at a smaller bank), but the row is not apples-to-apples with the
other two and is marked accordingly.

Two conditional findings that decide the outcome and must travel with the headline:

1. **The optimization ladder is load-bearing at scale, not a tuning detail.** At QASPER's 6.32M-slot
   bank, exact top-k takes **7.26 s/query and LOSES to RAG@5**; approximate top-k alone brings it to
   2.198 s and int8 keys to 1.900 s — a **3.8× swing** that decides whether the method wins at all.
2. **Layer count dominates at large banks.** Four memory layers at the same bank give 3.36 s/query
   (1.31× RAG) versus one layer's 1.90 s (2.33×), because every layer rescans the full bank.

## Scope: what this does and does not show

**This is a serving-cost measurement, not a quality claim.** No checkpoint in this repo was trained
on long documents — midtraining used 256-token MuSiQue paragraphs — so accuracy on LongHealth or
QASPER would be out-of-distribution and would say nothing about the architecture. Latency at a
given shape is independent of whether the answer is correct, which is why the throughput axis is
reportable and the accuracy axis is not.

**The latencies are a shape-matched microbenchmark, not end-to-end system runs.** Both sides are
composed from the same measured primitives — attention and MLP at the real shapes through 36
layers, real bank sizes, chunked causal prefill — on the same chip
(`scripts/embed/bench_pareto_throughput.py`). This is deliberate: timing the vLLM RAG pipeline
against the repo's JAX decode loop would measure the two serving stacks rather than the two
architectures, and vLLM would likely win on engineering alone. The cost is that these are not
wall-clock numbers from a deployed system.

**Indexing is excluded from per-query latency** and is reported separately below.

## Setup

Both systems are given the same corpus and use their native retrieval:

- **RAG@5** — Qwen3-Embedding-0.6B embeds each full document; the top-5 documents by cosine
  similarity go verbatim into a Qwen3-4B prompt. Prompt length scales with `k × doc_len`.
- **RLM** — frozen Qwen3-4B plus trainable memory layers. Documents are encoded offline in
  256-token chunks through the full 36-layer backbone; hidden states are projected through
  `mem_k_proj`/`mem_v_proj` into one bank slot per document token. At query time the prompt is the
  question alone (~64 tokens) and each memory layer scores every bank slot, takes top-k, and
  attends over the retrieved slots.

Latency = prefill + 100 generated tokens, B=1, single v5p chip, no tensor parallelism.
**Noise floor ~9% run-to-run** (the same config measured 16,799 and 15,073 µs/tok on two runs), so
differences under ~10% are not resolvable.

## QASPER

1,169 papers / 3,598 questions (AI2 v0.3 train+dev), **6,318,182 tokens** under the Qwen3
tokenizer; mean 5,404 tokens/paper (median 5,075, p90 8,107, max 36,505; 1.40 tokens/word).
RAG@5 → 27,084-token prompt. RLM indexes the whole corpus → 6,318,182-slot bank.

| config | prefill | decode | **s/query** | vs RAG@5 |
|--------|---------|--------|-------------|----------|
| RAG@5 whole papers | 2.157 s | 22,733 µs/tok | **4.430** | — |
| RLM 1L, exact top-k / bf16 | 0.018 s | 72,416 | **7.259** | RAG 1.64× |
| RLM 4L, approx / int8 | — | — | **3.362** | RLM 1.31× |
| RLM 1L, approx / bf16 | 0.013 s | 21,846 | **2.198** | RLM 2.01× |
| **RLM 1L, approx / int8** | 0.015 s | **18,853** | **1.900** | **RLM 2.33×** |

At this scale RAG loses on *both* components: prefill 2.157 s vs 0.015 s, and decode 22,733 vs
18,853 µs/tok — a 27k-token KV cache costs more per generated token than a 6.32M-slot bank scan.

## LongHealth

20 patients / 133 documents / 400 questions, **233,041 tokens** (mean 1,752/doc, median 1,534,
max 8,582; **2.12 tokens/word** — clinical text tokenizes far denser than prose). RAG@5 →
8,824-token prompt. RLM bank = 233,041 slots.

| config | s/query | vs RAG@5 |
|--------|---------|----------|
| RAG@5 (interpolated, 8,824-tok prompt) | ~1.96 | — |
| RLM 4L, exact / bf16 | 2.735 | RAG 1.39× |
| RLM 4L, approx / bf16 | 2.121 | ~tie |
| RLM 4L, approx / int8 | 2.066 | ~tie |
| RLM 1L, exact / bf16 | 1.867 | ~tie |
| RLM 1L, approx / bf16 | 1.793 | ~tie |
| **RLM 1L, approx / int8** | **1.686** | **RLM 1.16×** |

Only the last row clears the ~9% noise floor against RAG; the middle rows are ties.

The RAG@5 point is interpolated between measured 7,000-token (1.86 s) and 10,304-token (2.07 s)
prompts; every other value is measured directly.

## The RAG prefill curve

Measured end-to-end at whole-document context sizes (`scripts/embed/longctx_sweep.sh`):

| prompt tokens | prefill | s/query |
|---------------|---------|---------|
| 7,000 | 0.30 s | 1.86 |
| 35,000 | 3.07 s | 4.85 |
| 70,000 | 10.15 s | 12.25 |
| 140,000 | 34.74 s | 37.17 |
| 280,000 | 134.16 s | 137.64 |

Prefill grows **~450×** across a 40× prompt increase — superlinear, approaching quadratic, as
attention requires. The memory layer's prefill is flat at 0.013–0.018 s at every bank size
measured (7k → 6.32M slots), because its prompt is the question regardless of corpus.

## Indexing cost

Encoding a corpus into the bank is a full 36-layer forward pass over every corpus token, but
documents are encoded as **independent 256-token chunks**, so attention is confined within a chunk
and the cost is *linear* in corpus size — unlike a long-context prefill, which is quadratic.
Measured at **47,529 tokens/s** (`scripts/embed/bench_embed_cost.py`):

| corpus | tokens | index time |
|--------|--------|-----------|
| LongHealth | 233,041 | 4.9 s |
| QASPER | 6,318,182 | **133 s** |

This is paid once per corpus, not per query. On LongHealth it is repaid against full-context RAG
after a single query (4.9 s of indexing vs 34.7 s per full-context prefill). **RAG's own indexing
cost was not measured** — Qwen3-Embedding-0.6B over the same tokens is cheaper than our 4B pass but
is not zero, and quoting our figure against an implied zero would be one-sided.

## Corrections to earlier claims in this line of work

- **"QASPER won't make RAG slow" — wrong.** That judgement came from QASPER's *native* setting,
  where each question concerns one paper (~5.4k tokens, prefill ~0.3 s, decode-dominated, a tie).
  At corpus level with whole-paper retrieval it is the *strongest* case for the memory layer of the
  three benchmarks tested.
- **"The ladder and layer count are substitutes" — wrong.** They are complements: at a 233k bank the
  1-layer ladder is worth +10.6% and the 4-layer ladder +32%, and combining 4L→1L with the full
  ladder is +62%. At QASPER's 6.32M bank the ladder alone is +281%.
- **Token estimates from word counts are unreliable.** A 1.35 tokens/word assumption put LongHealth
  at 148k tokens; the true figure is 233,041 (2.12 tokens/word). The reverse error then put QASPER
  at 10–12k tokens/paper when it is 5,404 (1.40 tokens/word). Tokenize, do not estimate.

## Repro

TPU: `v5p-4` (GCE-attached, `TRANSPORT=gce`), single chip, B=1, no tensor parallelism.

```bash
bash scripts/embed/longctx_sweep.sh     # RAG prefill curve 7k..280k + memory at matched banks
bash scripts/embed/qasper_bench.sh      # QASPER: RAG@5 vs 1L ladder at 6.32M slots
bash scripts/embed/layers_ladder.sh     # 1 vs 4 layers x 3 ladder steps at 233k slots
CORPUS_TOKENS=6318182 uv run --no-sync python scripts/embed/bench_embed_cost.py
uv run --no-sync python datagen/longhealth/longhealth_stats.py   # corpus token counts
uv run --no-sync python datagen/longhealth/qasper_stats.py
```

Data: [`results/longdoc/`](../../results/longdoc/) (`longctx.jsonl`, `qasper_bench.jsonl`,
`layers_ladder.jsonl`).
Plot: `results/figures/longdoc_pareto.png` via `scripts/analysis/plot_longdoc_pareto.py`.

**Reading the plot.** Its y-axis is MuSiQue accuracy and its x-axis is QASPER latency — two
different benchmarks. **Only the y-axis names its benchmark**; the x-axis label reads "Long Context
End-to-End Latency (s/query)" without naming QASPER, so the figure cannot be read standalone and
its caption must supply the attribution. It supports the claim *"comparable accuracy,
2.0× lower latency"*; it does **not** show that the memory layer is more accurate on QASPER, which
is unmeasured. Accuracy points are the `hard-neg` checkpoint (1 memory layer), matching the
`n_mem_layers=1` latency configuration, though latency was measured at top-k 128 while that
checkpoint trained at top-k 64 — a mismatch that is conservative for the memory layer (larger k
costs more) but is a mismatch.

## Open

- **Accuracy on a long-document benchmark** is the missing axis, and it needs a long-document
  midtraining run before it means anything. This is the single unlock for turning "comparable
  accuracy, lower latency" into a Pareto claim on one task.
- **RAG indexing cost** unmeasured (see above).
- **Top-k mismatch** between the latency sweeps (128) and the `hard-neg` checkpoint (64).
- **The LongHealth eval pipeline is built but not yet working** — see
  [`wiki/implementations/2026-07-19-longhealth-eval-pipeline.md`](../implementations/2026-07-19-longhealth-eval-pipeline.md).
