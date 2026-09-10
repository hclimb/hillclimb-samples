# Doc-code bank keys: per-chunk identity concatenated into mem_k (+ slim retrieval heads)

**Date:** 2026-07-22 · **Author:** rohunagrawal (with Claude) · **Status:** done — smoke
green (v11), full 40k run launched 2026-07-23 · **Branch:** `ground-doccode`

**What changed:** bank keys can now carry a **document-identity code**:
`memory.mem_doc_code_dim > 0` (default 0 = off, nothing changes anywhere) makes each chunk's
pad-masked mean-pooled embed-tower state pass through a new learnable `doc_code_proj`
`[d_embed, code_dim]` and get **concatenated onto every token key of that chunk**. The
retrieval dim becomes `mem_k_dim + mem_doc_code_dim`; `mem_q_proj`, `mem_q_norm`, and
`mem_k_norm` are sized to it automatically. The retrieval dot product then decomposes
additively into **token-similarity + doc-affinity** — value-position slots ("located in
___") carry a subject binding their local context lacks.

## Motivation

The 2026-07-22 sweep + grounding metric localized the hybrid's failure to the memory read
detaching from its evidence among semantically-close distractors (grounding 0.39–0.48 at
k=200 vs RAG's 0.70–1.00; oracle-vs-retrieved gap ~0.2 with recall excluded), and the
temperature ladder showed sharpening saturates (+0.07 then plateau) — the residual failure
is the mixture's *contents*: key tokens from the gold doc, value tokens from neighbors.
The doc-code attacks the binding directly, at the representation. First training config:
`qwen3_mem_embed_g4l_doccode` — the `ground_s1_zeroinit_4layer` recipe (4 layers
[9,14,20,27], zero-init `mem_o_proj`, frozen main, `staged_ground`,
`qa_hard_neg_think_sft4b`) with `mem_k_dim` 128, `mem_v_dim` 256 (single-bank 16q4kv
dimensionality — deliberately NOT the GQA variant, so the hybrid/gather eval tooling
applies), `mem_doc_code_dim` 128.

## How it was built & integrated

- `models/memory_utils.py::add_memory_layer` — `retr_dim = k_dim + doc_code_dim` sizes
  `mem_q_proj` and both retrieval norms (joint RMS over the concatenated dim — the simple
  choice; per-part norms would decouple scales, not done).
- `models/memory_utils.py::add_kv_head` — new `doc_code_proj` (0.02-scaled normal), single-
  bank path only (GQA banks unsupported).
- `models/qwen3_mem_embed.py::embed_forward` — pool → project → broadcast → concat, gated on
  **weight presence** (`'doc_code_proj' in weights`), since the embed-side cfg does not carry
  memory keys. Sits after the optional conv block, so it uses post-conv states and the
  pooled pad_mask consistently. Per-CHUNK semantics: multi-chunk docs get per-chunk codes
  (chunk = passage identity), documented deliberately.
- Training script `scripts/embed/train_ground_s1_doccode.sh` — the ground_s1 recipe +
  `tp_devices=2`; `SMOKE=1` = 1200 steps (must exceed staged_ground's stage-0 boundary at
  1000) with saves every 400 to validate the save path early. `MEM_MASKED_OPTIMIZER=1` was
  tried and reverted — it crashes in adamw's update_moment (see the optimizer-moment note's
  2026-07-22 addendum); full moments accepted, betting the slimmer arch clears spec64's
  ~39 MB save-time shortfall. Also required cherry-picking dummy_token's LR-in-JIT fix
  (`1a64430`): the smoke reproduced the multi-host FAILED_PRECONDITION on this branch
  EXACTLY (main-lineage training had never run on a slice) — independent cross-branch
  confirmation of that diagnosis.
- Retrieval kernels, hybrid evaluator, losses: **zero changes** — they infer dims from
  array shapes.

## Tests

Smoke record: **v11 GREEN** — 1200/1200 steps, stage-0→1 flip at 1000, all three saves
(400/800/1200) incl. the post-flip save every earlier config failed; loss 0.93 / CE 0.72
through the flip, zero NaN, ~1.29 it/s (single-host tp=1 bs=8),
run `ground_s1_doccode_4layer_smoke-2026-07-22-21-38-05`, `__RUN_EXIT__=0`. Full run:
`ground_s1_doccode_4layer` 40k steps, ckpt every 2000, same config (launched 2026-07-23,
~8.6 h ETA). Config-search ladder that got here (each step measured):
v1 stage-ordering; v2 → reproduced the multi-host FAILED_PRECONDITION exactly (cherry-picked
dummy_token's `1a64430`); v3 → MEM_MASKED_OPTIMIZER TypeError (demoted to broken); v4/v5 →
multi-host save OOM invariant to batch size; v7 → tp=4 made it worse (unshard queues more
gathers); v8b/v10 → single-host compile OOMs at bs16 (tp2: −1.42G, tp1: −627M); v11 → tp=1
bs=8 = July's per-chip layout, fits with ~3.5G margin. The v4-v7 smoke
ladder produced a load-bearing infra finding along the way:

**The multi-host checkpoint save can NEVER fit on v6e (31 GB), at any tp or batch size.**
`save_checkpoint`'s unshard gathers every leaf to REPLICATED and async-enqueues all of them
(`utils.py::unshard`, "async enqueues un-shard") — transiently materializing the full
weights+moments (~24 GB) per chip on top of the resident sharded state. Measured: v4
(tp=2, bs=16) OOM at 47.50M/47.06M-free; v5 (tp=2, bs=12) 47.50M/47.31M — batch size
irrelevant; v7 (tp=4) 5.00M/1.46M-free — MORE sharding just queues more gathers before
dying. This also retro-explains the spec64 step-5000 save death (dummy_token branch).
Every July ground_s1-family run saved fine because they were SINGLE-HOST — the unshard is
the multi-host path. Mitigation here: `SINGLE_HOST=1` (one worker's 4 chips, standalone-TPU
env, tp=2/data=2, ~2× wall clock). Real fix, still open: sharded/streaming orbax save.
Also hit: `MEM_MASKED_OPTIMIZER=1` crashes in adamw's update_moment (optimizer-moment
note's 2026-07-22 addendum) — demoted to broken.

## Follow-ups & risks

- Joint RMS over `[token_key ; doc_code]` couples the two parts' scales; if the code
  dominates or vanishes, per-part norms are the first knob.
- Per-chunk (not per-document) identity for multi-chunk docs.
- Eval-side: the hybrid evaluator works unchanged, but cross-checkpoint comparisons must
  note the different retrieval dim; `mem_k_prenormed` paths untouched.
