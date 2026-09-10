# MuSiQue: corpus-scaling accuracy and the end-to-end throughput Pareto vs. RAG

**Status: complete (2026-07-19). All four corpus sizes measured for all three systems.**
Contains two explicit corrections to an earlier version of this page — see the Conclusion and the
ground4layer section.

## Conclusion

**On MuSiQue, RAG wins both axes at every corpus size — with one accuracy tie at c8192, where RAG
still wins throughput. There is no Pareto win for the memory layer here, and the goal this
experiment set out to demonstrate is not supported by the data.**

| corpus | RAG | memory layer (best config at the matched bank) | RAG better on both? |
|--------|-----|------------------------------------------------|---------------------|
| 512    | **0.3906** acc @ **0.579** q/s | 0.3281 @ 0.505 (ground4layer) | ✅ |
| 512    | 〃                             | 0.2810 @ 0.505 (hard-neg)     | ✅ |
| 2048   | **0.3281** @ **0.579**         | 0.3125 @ 0.478 (hard-neg)     | ✅ |
| 2048   | 〃                             | 0.3047 @ 0.478 (ground4layer) | ✅ |
| 8192   | 0.2891 @ **0.579**             | 0.28125 @ 0.381 (ground4layer) | throughput only — **accuracy is a tie** (+0.0078, at the noise floor) |
| 8192   | 〃                             | 0.2109 @ 0.381 (hard-neg)     | ✅ |
| 11656  | **0.2656** @ **0.579**         | 0.2344 @ ~0.35 (hard-neg)     | ✅ |
| 11656  | 〃                             | 0.2109 @ ~0.35 (ground4layer) | ✅ |

(The 11656-doc corpus is a 2.98M-slot bank, just past the 2M point measured; its memory throughput
is interpolated from the 2M row and marked `~`.)

The two axes are directly comparable because the eval chunks each document into one 256-token
chunk, so **bank slots = documents × 256** — the swept banks *are* the accuracy corpora
(512 → 131,072; 2048 → 524,288; 8192 → 2,097,152). RAG's prompt is `k×256 + 64 = 1344` tokens no
matter how large the haystack is, so its throughput is one flat number across all corpus sizes,
while the memory layer's degrades as its bank grows.

Two supporting results:

1. **The accuracy decay is not a retrieval failure.** `doc_hit_rate` holds at 0.94–0.99 across a
   23× corpus increase while `mem_pos_weight_mass` collapses 0.340 → 0.084. The model finds the
   gold document and stops attending to it. RAG's `recall@5` decays *harder* (0.609 → 0.391) and
   RAG still scores better — the bottleneck is downstream of retrieval. This is a **within-model**
   diagnostic; it does not rank architectures (see
   [scope](#mechanism-retrieval-succeeds-attention-dilutes)).
2. **More memory layers change nothing consistently.** ground4layer minus hard-neg is +0.047,
   −0.008, +0.070, −0.024 across the four corpus sizes — no ordering, against a ~0.008 noise floor.
   Width neither fixes the dilution nor worsens it.

> **Correction (2026-07-19).** Point 2 previously read "more memory layers make it worse", drawn
> from a single 0.0078 difference at c2048 — *at* the noise floor this same document states. The
> completed four-point curve retracts it. Likewise, RAG's lead over ground4layer narrows across the
> first three points (+0.0625 → +0.0234 → +0.0078) and then reverts at c11656 (+0.0547); that
> narrowing was scatter, not an approaching crossover.

**What remains true for the memory layer**, and what a follow-up should test: its prefill is flat
in document length (~15 ms) while RAG's grows 276× across a 64× document-length increase. Past
**~1,700–3,100 tokens/document** (depending on bank size) the memory layer overtakes RAG on
end-to-end throughput, reaching **2.2× at 8192 tok/doc and 5.9× at 16384** for a 2M-slot bank.
**MuSiQue's documents are 256 tokens — well below that crossover — so none of this is reachable on
this task.**

The win also requires **few, long** documents, not merely long ones: bank = docs × doc_len, so the
speedup shrinks as the corpus grows (see the
[matched-bank table](#the-win-needs-few-long-documents-not-just-long-ones)). It is a hypothesis
about long-document QA (NovelHopQA, narrativeqa), and it is *not* a Pareto claim until accuracy is
measured on such a benchmark.

> **Reading the plot.** Panel 3 (`throughput vs document length`) is a **projection**, not a
> MuSiQue result. Placing it beside the accuracy panel as a single head-to-head would be true
> numbers arranged into a false claim — the throughput advantage belongs to a workload this task
> does not contain. The panel marks MuSiQue's actual operating point for exactly that reason.

## Accuracy vs. corpus size

Identical protocol per point: same 128 MuSiQue queries, matched corpus (`inject_query_gold` keeps
every gold and fills to `target_docs` with distractors; the RAG corpus builder reproduces exactly
that selection), Qwen3-4B LLM judge.

| corpus | RAG (top-5 in prompt) | mem hard-neg @750 (1 layer, top-k 64) | mem ground4layer @1500 (4 layers, top-k 128) |
|--------|----------------------|----------------------------------------|-----------------------------------------------|
| 512    | **0.3906**           | 0.2810                                 | 0.3281                                        |
| 2048   | **0.3281**           | 0.3125                                 | 0.3047                                        |
| 8192   | 0.2891               | 0.2109                                 | 0.28125  *(tie — see below)*                  |
| 11656  | **0.2656**           | 0.2344                                 | 0.2109                                        |

Noise floor: a repeat of RAG @512 gave 0.3984 vs 0.3906, so ~0.008 on n=128. **Differences below
~0.02 are not resolvable** — the 2048 row (RAG 0.3281 vs hard-neg 0.3125) is within about 2× the
noise floor and should be read as "close", not as a win.

**RAG wins every row, with one tie.** At c8192 RAG's margin over ground4layer is +0.0078, which is
*at* the noise floor — that row is a tie on accuracy (RAG still wins throughput there, 0.579 vs
0.381 q/s). RAG's lead over ground4layer narrows monotonically across the first three points
(+0.0625 → +0.0234 → +0.0078) and then reverts at c11656 (+0.0547), so the narrowing was scatter
rather than a trend toward crossover. Three points in a row moving one direction was not enough to
call it, and it did not hold.

Data: [`results/musique_pareto_acc.json`](../../results/musique_pareto_acc.json),
telemetry [`results/musique_pareto_telemetry.json`](../../results/musique_pareto_telemetry.json).

### Mechanism: retrieval succeeds, attention dilutes

The interesting result is *why* the memory layer decays, and the telemetry separates two
hypotheses cleanly:

| corpus | `doc_hit_rate` (gold slot in top-k at all?) | `mem_pos_weight_mass` (softmax mass on gold) |
|--------|--------------------------------------------|----------------------------------------------|
| 512    | 0.984                                      | 0.340                                        |
| 2048   | 0.984                                      | 0.160                                        |
| 8192   | 0.945                                      | 0.113                                        |
| 11656  | 0.940                                      | 0.084                                        |

Hit-rate falls 4.5% while weight-mass falls **4.0×**. The gold document is in the retrieved set
almost every time; the model simply spreads its attention over a growing pool of distractor slots.
RAG's comparable quantity, `recall@5`, decays much harder (0.609 → 0.391) — RAG retrieves *worse*
and still scores *better*, which is the sharpest form of the result: the memory layer's bottleneck
is downstream of retrieval.

**Scope of this claim — it is within-model, not cross-architecture.** ground4layer at c8192 has
`mem_pos_weight_mass` **0.0275**, far *below* hard-neg's 0.113 at the same corpus, and yet scores
**0.281 vs 0.211**. With 4 layers × top-k 128 and ~54 `mem_effective_slots` it gathers enough total
evidence despite thin per-slot mass. So low weight-mass does not by itself predict low accuracy:
the dilution account explains why a *given* model degrades as its bank grows, but it does not rank
architectures against each other. An earlier version of this write-up used it that way; that was
too strong.

### ground4layer: more memory layers change nothing consistently

> **Correction (2026-07-19).** An earlier version of this section concluded "more memory layers
> make it worse", based on ground4layer scoring 0.3047 vs hard-neg's 0.3125 at c2048. That is a
> **0.0078 difference against the ~0.008 noise floor documented above** — I read a tie as a result.
> The full four-point curve does not support that claim, and it does not support its opposite
> either.

The natural response to attention dilution is more memory layers and a wider top-k. `ground4layer`
(4 memory layers, top-k 128) is exactly that. Against hard-neg (1 layer, top-k 64):

| corpus | hard-neg | ground4layer | difference |
|--------|----------|--------------|------------|
| 512    | 0.2810   | 0.3281       | **+0.047** |
| 2048   | 0.3125   | 0.3047       | −0.008 (tie) |
| 8192   | 0.2109   | 0.28125      | **+0.070** |
| 11656  | 0.2344   | 0.2109       | −0.024 |

**No consistent ordering.** Differences swing from −0.024 to +0.070 with no relationship to corpus
size. On this evidence, width neither fixes the dilution nor worsens it; both architectures are
noisy and both lose to RAG at every corpus size. Claiming a direction here in either direction
would be over-reading n=128.

Note also what ground4layer does *not* buy: at c11656 it posts `doc_hit_rate` 0.977 — near-perfect
retrieval — while scoring 0.2109, its worst result and identical to hard-neg's *8192* score.

## Throughput axis: end-to-end queries/sec

Metric: one query = prefill the prompt + generate 100 tokens. B=1, single v5p chip, no tensor
parallelism. Both sides are composed from the same measured primitives at the same shapes on the
same chip — this is deliberately *not* vLLM-vs-JAX, which would measure the serving stack rather
than the architecture.

**Noise floor: ~9% run-to-run.** The same `rag_doclen256` config measured 16799 and 15073 µs/tok on
two separate runs (median-of-50 each). **Throughput differences under ~10% are not resolvable** and
should not be read as wins.

### RAG: prefill dominates and scales superlinearly

| doc_len | prefill | decode (µs/tok) | query time | queries/sec |
|---------|---------|-----------------|------------|-------------|
| 256     | 47.5 ms | 16799 | 1727 ms | 0.579 |
| 1024    | 209.2 ms | 15645 | 1774 ms | 0.564 |
| 2048    | 496.6 ms | 15755 | 2072 ms | 0.483 |
| 4096    | 1363.3 ms | 17913 | 3155 ms | 0.317 |
| 8192    | 4042.3 ms | 18471 | 5889 ms | 0.170 |
| 16384   | **13117.5 ms** | 22321 | 15350 ms | **0.065** |

Prefill grows **276×** across a 64× document-length increase — superlinear, approaching quadratic
at the long end, as attention requires. This is the whole asymmetry: at doc_len 16384 RAG spends
13.1 s prefilling `5 × 16384 = 81,920` prompt tokens before emitting a single output token.

### Memory layer: flat prefill, and an optimization ladder that only pays at large banks

Prompt is the question alone, so **prefill is ~15 ms at every bank size** (vs RAG's 47.5 ms →
13.1 s). The cost moves to decode, where each of the 4 memory layers scans the bank per token.

| bank (= corpus) | exact/bf16 | + approx top-k | + int8 keys | ladder gain |
|-----------------|-----------|----------------|-------------|-------------|
| 131,072 (512 docs)   | 0.482 | 0.505 | 0.491 | **none — inside noise** |
| 524,288 (2048 docs)  | 0.320 | 0.421 | **0.478** | **+49%** |
| 2,097,152 (8192 docs)| 0.141 | 0.344 | **0.381** | **+170%** |

Decode cost per token tracks it exactly: 20,596 → 31,091 → 70,864 µs/tok for exact top-k at
131k / 524k / 2M, against 19,634 → 23,604 → 26,092 µs/tok for approx+int8. The scan is what blows
up, and approximating it is what saves.

**The ladder's value is entirely bank-dependent, and reporting a single number for it would be
wrong in either direction.** At a 512-doc bank the three configs span 5% against a ~9% noise floor
and are not even monotonic (int8 *below* approx-bf16) — that is scatter, not a ladder, and it
matches the prior calibration that decode is transformer-bound until the bank is large enough for
the scan to matter ([approx-topk note](2026-07-15-approx-topk-training.md)). At 524k every step
clears noise decisively (+32%, then +14%, +49% cumulative). At 2M, exact top-k costs 70,864 µs/tok
and collapses throughput to 0.141 q/s, and approx top-k alone recovers **2.44×**.

Measured only at 131k the optimizations look worthless; measured only at 2M they look like a 2.4×
free lunch. Both readings are artifacts of the bank you picked. **This is the memory layer's
strongest engineering result tonight** — at realistic corpus sizes the approximate scan is the
difference between 0.14 and 0.34 q/s — and it is still not enough to reach RAG's flat 0.579.

So the optimizations are a real and useful result for the memory layer. They do not, however,
close the gap: RAG sits at a flat 0.579 q/s on this task, above every memory configuration
measured.

### Why end-to-end and not tok/s — a negative result worth recording

The first throughput measurement asked whether long documents slow RAG's **decode**. They mostly do
not: per-token decode falls only **16799 → 22321 µs/tok (−25%) across a 64× context increase**,
because attention over the KV cache is a small share of a 36-layer step. **The decode-throughput
hypothesis was wrong, and the memory layer has no advantage on that metric at short-to-moderate
document lengths.** (An earlier sweep put this at −17%; −25% is the value from the corrected v3 run
and supersedes it.)

The cost RAG actually pays is **prefill** — `k × doc_len` tokens pushed through the model once per
query before the first output token. A QA query generates ~100 tokens, so prefill is a first-class
term in queries/sec and a decode-only plot hides the entire asymmetry. That is why the metric here
is end-to-end (prefill + 100 generated tokens), not tok/s.

### Where the memory layer does win: a projection, not a result

RAG's prefill grows 276× across a 64× document-length increase; the memory layer's is flat at
~15 ms. Interpolating the measured RAG curve against the best memory config (131k bank, 0.505 q/s)
puts the crossover at **~1,700 tokens/document**, beyond which the memory layer leads:

| doc_len | RAG q/s | memory (131k) q/s | ratio |
|---------|---------|-------------------|-------|
| 256     | 0.579   | 0.505 | RAG 1.15× |
| ~1,700  | ~0.505  | 0.505 | **crossover** |
| 4096    | 0.317   | 0.505 | mem 1.6× |
| 8192    | 0.170   | 0.505 | mem 3.0× |
| 16384   | 0.065   | 0.505 | mem 7.8× |

**MuSiQue's documents are 256 tokens — about 7× below the crossover — so none of this is reachable
on this task.**

#### The win needs *few, long* documents, not just long ones

The table above is cross-bank: it pins the memory layer at 131k slots while stretching RAG's
documents. But bank = `docs × doc_len`, so every (bank, doc_len) pair implies a **specific corpus
size**, and the honest comparison reads them together. Ratio = memory q/s ÷ RAG q/s (>1 = memory
faster); N = implied corpus size in documents. All values measured, none extrapolated:

| bank (best config) | 256 | 1024 | 2048 | 4096 | 8192 | 16384 |
|--------------------|-----|------|------|------|------|-------|
| 131k — 0.505 q/s | 0.87 (N=512) | 0.90 (N=128) | 1.05 (N=64) | 1.59 (N=32) | 2.97 (N=16) | **7.77 (N=8)** |
| 524k — 0.478 q/s | 0.83 (N=2048) | 0.85 (N=512) | 0.99 (N=256) | 1.51 (N=128) | 2.81 (N=64) | 7.35 (N=32) |
| 2M — 0.381 q/s | 0.66 (N=8192) | 0.68 (N=2048) | 0.79 (N=1024) | 1.20 (N=512) | 2.24 (N=256) | 5.86 (N=128) |

Crossover per bank: **1,697 tok/doc** at 131k (≈77-doc corpus), **2,091** at 524k (≈251 docs),
**3,135** at 2M (≈669 docs). It moves right as the corpus grows, because RAG's throughput is flat
in corpus size while the memory layer's is not.

**The headline "7.8×" is an 8-document corpus** — technically true, practically a toy. The
defensible framing is the 2M row: on a corpus of a few hundred long documents the memory layer is
**2.2–5.9×** faster end to end. And at 256 tok/doc it loses at every bank size (0.66–0.87×), which
is the MuSiQue regime and matches the measured result exactly.

This is a hypothesis worth testing on a long-document benchmark (`gen_embed_novelhopqa_hop{1..4}`,
`gen_large_mem_msa_narrativeqa`, `ruler` are already wired up). It is **not** a Pareto claim: no
accuracy has been measured on any such task, and accuracy is the axis the memory layer has been
losing.

### Methodology bug found and fixed — it was biased in the memory layer's favour

The first end-to-end harness modelled prefill as a single fused forward over all `PF` prompt
tokens, which materialised a `PF × PF` attention score matrix and used **non-causal** attention.
Both errors inflate RAG's prefill cost:

- non-causal attention charges ~2× the real attention FLOPs;
- materialising the score tensor adds HBM traffic no real serving stack pays (≈27 GB at doc_len
  4096) — and at doc_len 16384 the tensor would be ≈430 TB, so that point **OOM'd and was silently
  missing** from the first sweep.

Every affected number flattered the memory layer, so all RAG points were re-measured with chunked
causal prefill (`scripts/embed/bench_pareto_throughput.py`): query chunk *c* attends only to the
keys written so far, nothing is materialised. Per-chunk time is linear in KV length, so the
harness probes 4 lengths, fits, and sums over chunks rather than timing all 40 chunks at doc_len
16384.

The memory-layer points are unaffected (`KVLEN=64` → a single chunk → identical code path), which
gives a free harness check: they should reproduce the pre-fix values. The pre-fix run is kept as
`~/pareto_e2e_prefix_model_v1.jsonl` on the box for that comparison and is **not** used in any
plot.

**Anything quoted from the pre-fix sweep is retracted**, including a crossover near doc_len 2–4k
that was derived from it. The fix makes RAG *cheaper*, so the true crossover moves right.

## Repro

TPU: `v5p-4` (GCE-attached, `TRANSPORT=gce`), single chip, B=1, no tensor parallelism.

Accuracy sweeps — memory layer:
```bash
RUN_DIR=musique_ground4layer_midtrain_topk128_seq1024_chunks20_bs16-2026-07-19-00-52-42 \
STEP=1500 SIZES=2048_8192_11656 bash scripts/embed/sweep_mem_corpus.sh
```
RAG: `scripts/embed/sweep_rag_corpus.sh` (same sizes, same 128 queries).

Throughput:
```bash
bash scripts/embed/sweep_pareto_e2e.sh    # 15 points: 6 doc lengths x RAG, 3 banks x 3-step ladder
```

Checkpoints:
- hard-neg: `gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-07-17-02-32-09/qwen3_mem_embed/750`
  > **Correction (2026-07-20).** This pointer is stale — step 750 does not exist in that run
  > dir (either bucket). The "hard-neg @750" scored throughout this page is the **MuSiQue
  > midtrain** checkpoint
  > `gs://memory-layers-training-usc1/musique_sft_midtrain_topk64_seq1024_chunks20_bs32-2026-07-18-17-35-52/qwen3_mem_embed/750`
  > (its c512 score 0.2810 matches the accuracy table exactly; see the
  > [2026-07-18 write-up](2026-07-18-musique-midtraining-vs-rag.md)). A copy now also lives in
  > the EUROPE-WEST4 bucket under the same run-dir path for slice-local reads.
- ground4layer: `gs://memory-layers-training-usc1/musique_ground4layer_midtrain_topk128_seq1024_chunks20_bs16-2026-07-19-00-52-42/qwen3_mem_embed/1500`

Eval JSONs: `gs://memory-layers-training-usc1/<run-dir>/eval/step_<N>/musique_c<size>.json`
Commit SHA: `0955836` (branch `musique-sft`).
Throughput JSONL: `gs://memory-layers-training-usc1/pareto/e2e.jsonl`, committed at
[`results/pareto_e2e.jsonl`](../../results/pareto_e2e.jsonl) (15 points).
Plot: `results/figures/musique_pareto.png` via `scripts/analysis/plot_musique_pareto.py`
(`results/*.json` needs `git add -f` — `*.json` is gitignored repo-wide).

## What would actually change the verdict

Ranked by how much they'd move the result, not by effort:

1. **Fix attention dilution, not retrieval.** Every intervention tried so far (more memory layers,
   wider top-k) improves `doc_hit_rate`, which was never the problem. The target is
   `mem_pos_weight_mass` — the gold slots are being *found* and then out-voted. A sharper
   temperature on the memory softmax, or a learned gate on retrieved-vs-local attention, attacks
   the measured failure directly. The doc-code / slot-PE sketches in the scratchpad were aimed at
   a different problem.
2. **Measure a long-document benchmark end to end.** The throughput crossover is real but
   unaccompanied by any accuracy number on such a task. `gen_embed_novelhopqa_hop{1..4}`,
   `gen_large_mem_msa_narrativeqa`, and `ruler` are already wired. Until one of those has *both*
   axes, the memory layer has no demonstrated Pareto win anywhere.
   **Blocker:** no checkpoint has been trained on long documents — NovelHopQA uses
   `num_chunks_per_doc: 4096` (novel-length), while every checkpoint here was midtrained on
   256-token MuSiQue paragraphs. Evaluating those models there would be badly out-of-distribution
   and the resulting accuracy would say nothing about the architecture. This needs either a
   long-document midtraining run or a deliberate decision to report OOD numbers as such.
3. ~~Re-run the crossover at a matched bank.~~ **Done** — see the matched-bank table above; it is
   derivable from the committed sweep without new runs. The result: the crossover moves right with
   corpus size (1,697 → 3,135 tok/doc), and the win needs few long documents rather than merely
   long ones.

## Reading note on the plot

Every point on `results/figures/musique_pareto.png` is labelled with the conditions it was measured
under — corpus size on the accuracy panel, bank size / document length / ladder step on the
throughput panel. The two panels are **not** a single head-to-head curve and must not be read as
one: accuracy is measured at matched corpora, throughput at matched shapes, and no point mixes the
two.
