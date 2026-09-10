# MuSiQue midtraining vs. classic RAG @ 512-doc corpus

**Date:** 2026-07-18 · **Author:** rohunagrawal · **Status:** done

## Conclusion

**Midtraining on MuSiQue doubled multi-hop accuracy (0.141 → 0.281) and peaked at ~1.8 epochs,
but classic RAG still beats it (0.406) on the same corpus — and RAG is zero-shot while MuSiQue
is in-domain for the memory model.** Retrieval was never the bottleneck: the base checkpoint
already had `doc_hit_rate = 1.000` and scored 0.141. What midtraining bought was *attending to*
what it retrieved (`mem_pos_weight_mass` 0.169 → 0.31), and that gain is complete within one
epoch.

## Results

All rows: MuSiQue, the same 512-doc corpus, the same first 128 queries, judge Qwen3-4B @ TP=4,
`max_new_tokens=1280`, greedy unless noted.

| System | judge acc | `doc_hit_rate` | `mem_pos_weight_mass` | `lexical_grounding` |
|---|---|---|---|---|
| Base ckpt @ 100000 (pre-midtrain) | 0.141 | **1.000** | 0.169 | 0.236 |
| Midtrain @ 500 (1.22 ep) | 0.211 | 0.977 | 0.313 | 0.257 |
| **Midtrain @ 750 (1.82 ep)** | **0.281** | — | — | — |
| Midtrain @ 750, temperature 0.6 | 0.273 | 0.984 | 0.311 | **0.313** |
| Midtrain @ 1250 (3.04 ep) | 0.211 | 0.984 | 0.340 | 0.288 |
| **RAG, top-5 docs in prompt** | 0.398 | recall@5 0.609 | — | — |
| **RAG, top-3 docs in prompt** | **0.406** | recall@5 0.609 | — | — |

Memory @ 750 restricted to *fully answered* questions: 0.324 (23/71). Memory @ 1250 likewise:
0.273 (21/77). ⚠️ That slice conditions on completion, which correlates with question
difficulty — see "What we got wrong" below.

### Reading it

- **Inverted U over epochs**: 0.141 → 0.211 → **0.281** → 0.211 for base/1.22/1.82/3.04 epochs.
  Under-trained at one epoch, overfit by three. Step 750 leads on every slice measured.
- **Retrieval was never the problem.** The base model retrieved the gold document on *every*
  query (`doc_hit_rate` 1.000) and still answered 86% of them wrong. Midtraining's gain tracks
  `mem_pos_weight_mass`, i.e. routing attention to gold slots once retrieved — and that metric is
  flat from step 500 onward (0.313 / 0.311 / 0.340), so everything after epoch 1 is the LM half
  fitting, which past step 750 is net harmful.
- **Temperature 0.6 changed nothing** (0.273 vs 0.281). The SFT data was *generated* at 0.6, so
  matching it at inference was a reasonable guess; it did not curb the model's rambling. It did
  raise `lexical_grounding` to 0.313, the highest of any run — more verbatim copying, no more
  correctness.
- **RAG top-3 ≈ top-5** (0.406 vs 0.398, identical retrieval — only the prompt differs). An
  8-sample gap at n=128 is noise; read it as "no difference", not as fewer distractors helping.

## Setup / reproducibility

- **Checkpoints:** `gs://memory-layers-training-usc1/musique_sft_midtrain_topk64_seq1024_chunks20_bs32-2026-07-18-17-35-52/qwen3_mem_embed/{500,750,1250}`;
  base `…/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-07-17-02-32-09/qwen3_mem_embed/100000`.
- **wandb:** [musique_sft_midtrain_…-17-35-52](https://wandb.ai/memory-layers/memory-layers/runs/musique_sft_midtrain_topk64_seq1024_chun-2026-07-18-17-35-52) — `eval/*` metrics + per-run `eval_results` artifacts (128 samples each, incl. the judge's verdict text).
- **Result JSONs:** `gs://…-2026-07-18-17-35-52/eval/step_<N>/`.
- **TPU:** `v5p-4` (`ct5p-hightpu-4t`, flex-start, `us-central1-a`).
- **Commands:**
  ```bash
  CKPT=<gs://…/qwen3_mem_embed/750> MAX_NEW_TOKENS=1280 \
    bash scripts/embed/eval_musique_sft_midtrain.sh              # add TEMPERATURE=0.6 RUN_TAG=temp06
  GEN_TOP_K_DOCS=3 bash scripts/embed/rag_only_musique_c512.sh   # RAG baseline
  ```
- **Corpus:** `ragrawal36/musique-c512-rag-corpus`, built by
  `datagen/musique/build_musique_c512_rag_corpus.py` — 512 docs = 293 gold + 219 distractor.
  **This is what makes the RAG number meaningful.** `gen_large_mem`'s `inject_query_gold` scans
  the full corpus and keeps every gold before filling to `target_docs`;
  `single_embedding_retrieval --max_docs N` just stops reading after N. Pointing RAG at
  `--max_docs 512` would have given it 512 mostly-irrelevant documents and a near-zero score —
  the memory model "winning" only because it alone was handed the answers.
- **Fairness:** same corpus, same queries, same generator (Qwen3-4B) and judge (Qwen3-4B); RAG
  capped at `max_doc_length=256` because the memory eval indexes one 256-token chunk per doc, and
  the same Qwen3-Embedding-0.6B tower the memory model uses.

## Threats to validity

1. **No noise floor.** Every number is a single n=128 run and run-to-run spread was never
   measured. Differences ≲0.05 (RAG top-3 vs top-5; step 500 vs step 1250) should be treated as
   indistinguishable. The 0.141 → 0.281 jump is large enough to believe; the 0.281 peak itself
   rests on one run.
2. **Greedy decoding is not reproducible here.** Two runs with identical prompts produced
   divergent generations in 44/128 samples, despite `temperature=0.0` returning `argmax`. Prime
   suspect is `mem_approx_topk: true` (`jax.lax.approx_max_k` is shape-dependent, and
   `max_new_tokens` changes the compiled shapes). `mem_approx_topk: false` would give exact
   retrieval and should be used for any comparison that hinges on a small gap.
3. **The memory model is handicapped by verbosity, not only by reasoning.** ~40% of its
   generations never close `</think>` even at 1280 tokens and so emit no answer; the base model,
   trained at a 400-token budget, barely truncates. Some of the RAG gap is "RAG answers concisely
   and the memory model runs out of tokens".
4. **MuSiQue is in-domain for the midtrained model and zero-shot for RAG**, which makes the RAG
   win stronger, not weaker.

## What we got wrong along the way

- **Selection effect on "fully answered" accuracy.** At a 512-token cap, completed generations
  scored 0.432 and we inferred the true rate was ~0.43. Raising the cap to 1280 dropped it to
  0.273 — conditioning on completion was conditioning on *difficulty*, since only short (easy)
  generations finished. Do not read the fully-answered slice as "what it would score without
  truncation".
- **The eval config was mis-sized for the model.** `gen_large_mem_musique_c512.yaml` hardcodes
  `max_new_tokens: 512`, correct for the hard-neg checkpoint it was written for (≤400-token
  budget) but not for a model midtrained at 950. Raising the training budget did not propagate to
  the eval.

## Follow-ups

- **Re-run step 750 to confirm the peak**, and once with `mem_approx_topk: false`. Everything
  now rests on that checkpoint being the best, from a single run, on a non-reproducible eval.
- **Stop at ~1.8 epochs** for any re-train, or lower the LR — 1e-4 with main layers 13/14/15
  unfrozen over 3 epochs on 13k rows is where the regression appears.
- **Fix the verbosity** before re-measuring: the model generating past its 950-token training
  budget is a training defect, and it caps what any eval can show.
- **Out-of-domain check** (`gen_large_mem_msmarco_c512`, `..._hotpotqa_c512`) on base vs 750 —
  whether MuSiQue gains cost general ability. Not yet run.
