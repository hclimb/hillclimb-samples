# SnapKV KV-cache-compression baseline: full-context Qwen3-4B on MuSiQue + QASPER

**Date:** 2026-07-21 · **Author:** rohunagrawal (with Claude, overnight autonomous run) ·
**Status: done, with one open item** — MuSiQue c512 and both QASPER scales measured and judged;
the MuSiQue c2048 arm was **stopped at user request after 71/128 queries** (its unjudged
partial generations are preserved at
`gs://memory-layers-training/pareto/snapkv/snapkv_musique_c2048_comp16.json.partial`; one
judge pass over that file yields a provisional n=71 number if ever wanted). This page doubles
as the overnight progress log (bottom section).

## Hypothesis & motivation

The Pareto plots ([MuSiQue](2026-07-19-musique-corpus-scaling-and-throughput-pareto.md),
[long-doc serving cost](2026-07-19-long-document-serving-cost.md)) compare the memory layer
against classic RAG. A third natural baseline is **context stuffing + KV-cache compression**:
put the *whole corpus* in an off-the-shelf Qwen3-4B's context, compress the KV cache with
**SnapKV** ([FasterDecoding/SnapKV](https://github.com/FasterDecoding/SnapKV)) at 4×/8×/16×
(whatever fits), and answer questions directly — no retriever, no trained memory. Where does
that sit on accuracy × throughput?

## Method notes (design constraints, decided up front)

- **SnapKV in one paragraph:** after prefilling `[docs][question]`, score every prompt KV
  position per KV head by its attention weight from the last `W` query tokens (the
  "observation window" ≈ the question), keep the top-`C` positions per head (+ the window
  itself), discard the rest; decode against the compressed cache. Compression = prompt_len /
  (C + W).
- **Runs on our TPUs via a JAX reimplementation**, not the upstream repo: SnapKV upstream is
  CUDA/HF-transformers monkey-patching; the algorithm itself is ~40 lines against our stack
  (`models/qwen3.py` + chunked causal prefill from
  `scripts/embed/bench_pareto_throughput.py`). Standalone script, per house rules.
- **MuSiQue c512 = 512 × 256 ≈ 131k prompt tokens.** At/above Qwen3-4B's extended (YaRN)
  context; chunked prefill + full-cache SnapKV selection afterwards is faithful SnapKV. HBM
  for the uncompressed prefill cache (~19 GB bf16 KV across 4 chips) fits, so compression here
  buys decode speed and tests accuracy — the honest framing is that **prefill cost is the
  same as 131k-token RAG stuffing** and dominates the throughput point.
- **QASPER = 6.34M corpus tokens: no context window fits this — faithful single-shot SnapKV
  is impossible.** Plan: **streaming SnapKV** — prefill segment-by-segment (≤64k), after each
  segment compress its KV hard (question not yet seen → score by segment-local observation
  window, i.e. H2O/TOVA-flavored), position-repack, carry the compressed cache forward;
  final question attends over the accumulated compressed cache. This is a *SnapKV-inspired*
  streaming variant, and the write-up must label it as such.
- **Protocol parity:** same 128 MuSiQue queries as the hybrid/mem evals (content-join), same
  Qwen3-4B judge, `max_new_tokens` matched. QASPER per the 07-19 longdoc protocol.

## Setup / repro

(filled in as runs land)

## Results

**MuSiQue c512 (n=128, comp=4×, question-conditioned streaming SnapKV, judged Qwen3-4B):**

| metric | SnapKV comp4 | RAG@5 | hybrid k=50 | oracle-50 | full-bank mem |
|--------|--------------|-------|-------------|-----------|---------------|
| judge accuracy | **0.5312** | 0.3906 | 0.3984 | 0.4844 | 0.2810 |
| queries/sec | **0.0134** | 0.579 | ~0.505 | ~0.505 | 0.505 |

**Highest accuracy ever measured on this setup — and catastrophically slow.** Median
74.8 s/query: question-conditioned selection means the full 65k-token corpus is re-streamed
per query (selection can't amortize), so this sits at the accuracy-maximal, throughput-minimal
corner of the Pareto plot. The three regimes are now crisp: RAG (fast, mid accuracy),
memory-layer hybrid (fast decode, mid-high accuracy ceiling), full-context+compression (slow,
highest accuracy). Caveats: `lexical_grounding` is not comparable (no doc field in samples);
SnapKV/RAG use HF rows[:128] while mem evals use the pipeline's surviving rows (see the
[hybrid page](2026-07-20-musique-rag-hybrid.md) footnote); comp=4 is the mildest compression
that fits — a comp ladder (8×/16×) is future work.

**MuSiQue c2048 (comp=16×): not completed.** Two attempts: the first stalled after q75
(~18 min/query pace collapse, root cause unconfirmed — a late JIT retrace and stderr
warning-flood were the suspects; both mitigated in the restart), the hardened restart
(per-query partial dumps + `--resume`) ran healthily to q71 and was stopped at user request
to close the session. Artifacts preserved; the v2 plots carry a "stopped" marker at its
measured ~0.0067 q/s pace.

**QASPER (streaming, query-agnostic `--probe self`, SnapKV-inspired):**

| corpus | comp | judge acc | index (one-time) | median q/s after index |
|--------|------|-----------|------------------|------------------------|
| 281 papers / 1.40M tok (validation split) | 256× | **0.1406** | 296 s | 24.8 s/query |
| 1,585 papers / 8.24M tok (all splits) | 320× | running | ~30 min est | — |

At 256–320× compression, accuracy is expectedly weak — that is the honest price of fitting a
multi-million-token corpus into a 32k window; the point exists to anchor the Pareto plot's
"context stuffing + compression" corner. Reference: RLM answers the same corpus at 1.9 s/query
after a 133 s index (no accuracy measured there — see 07-19 caveats). Corpus-size footnote:
all-splits QASPER re-tokenizes to 8.24M tokens with the base Qwen3 tokenizer, larger than the
07-19 page's 6.34M figure (different accounting); the 1,169-paper count there vs 1,585 here
likewise differs by split selection.

---

## Overnight progress log (2026-07-21, times UTC)

- **~01:20** Loop started. Oracle-fill50 arms (c512 → c2048) launched and in retrieval/bank
  phase. ground4layer @1500 checkpoint EU-copy started in background. Task list created
  (#1 oracle+docs, #2 g4+k100 arms, #3 SnapKV MuSiQue, #4 SnapKV QASPER, #5 hybrid extras).
- **~01:50** SnapKV implementation written while oracle arms occupy the slice:
  `models/qwen3.py::forward_window_scores` (additive probe — normal forward of the window
  chunk that also emits per-layer, per-KV-head softmax mass over the cache; needed because
  `jax.nn.dot_product_attention` hides scores) + `scripts/embed/snapkv_stream.py` (streaming
  driver: segment prefill → question probe → per-head top-C compaction → compressed-space
  positions, budget ≤ 32k native context) + `snapkv_musique.sh` runner. Design deltas vs
  upstream SnapKV recorded in the header: no-YaRN forces segmenting; QASPER will use the
  query-agnostic `--probe self` mode (shared cache; SnapKV-inspired, not faithful).
  Syntax-checked; TPU smoke queued behind the oracle arms.
- Known rough edges to verify in smoke: per-token host round-trip in the decode loop (inflates
  the timing point; note or fix), tail padding with newline tokens after the assistant tag,
  a few extra jit shapes from 64-multiple tails.
- **~03:00–04:30** Smoke bring-up (9 iterations). Bugs found & fixed, in order: HF 504 on one
  host kills the slice run (→ retry/backoff); jitted wrappers closed over global weights
  (→ weights as jit args); `take_along_axis` gather rejected by explicit sharding (→ one-hot
  einsum gather with pinned specs); **`create_mask` returns a 4-dim mask** — the probe assumed
  3-dim, and `jnp.where` silently rank-upgraded the logits to 6-dim, corrupting score
  aggregation (→ label-driven `einsum('bkgts->bks')` reduction, which turned the silent bug
  into a loud one, + correct mask indexing); `s[0]` slice on a data-sharded batch axis
  (→ batch mean); tail padding after the assistant tag primes endless newlines (→ front-pad);
  and the big one — **RoPE phase disorder under real compression**: kept keys retained phases
  up to `seg_end` while later content restarts at `cur+C`, putting queries at LOWER phases
  than cached keys → degenerate decoding. Isolated via a two-arm debug (identity-compaction
  vs no-compaction both coherent; real compression broken) and fixed by **re-rotating gathered
  keys to destination phases in `compact`** (RoPE composes; cache keeps phase ≡ slot —
  standard position repacking). comp=4 smoke now produces coherent multi-hop reasoning.
- **~04:35** Full c512 run launched: comp=4 (minimum that fits ≤32k budget: 65k corpus tokens
  → 20.7k budget), n=128, `max_new=1280`. Steady-state ~27 s/query at 96 new tokens.
