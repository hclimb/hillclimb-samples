#!/usr/bin/env bash
# Warm-starts multihop_hard_neg_full from qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_
# pf32_indexed_lr_masked (step 100000) -- a DIFFERENT lineage from
# train_multihop_ground4layer_s1warmstart_no_multihop.sh's ground_s1_zeroinit_4layer_no_multihop
# source. Uses staged_batched_isolation_docaccess_warmstart (built on `staged`, main_model
# trainable throughout via learning_rates.main) rather than the _ground_ variant (frozen main) --
# this source checkpoint already unfroze main_model through staged.yaml's own stage D, so
# keeping it trainable continues that lineage instead of re-freezing a model that's already
# past that stage.
#
# ARCHITECTURE MUST MATCH THE SOURCE CHECKPOINT (mismatch silently falls back to fresh-init
# leaves, not an error -- see train_multihop_ground4layer_s1warmstart.sh's header): this
# checkpoint is SINGLE-LAYER (model.memory default mem_layers=[14] in configs/model/
# qwen3_mem_embed.yaml already matches -- no override needed) at mem_top_k=64 (override below;
# default is 128) and mem_o_proj_zero_init=false (also already the default). per_query_isolation
# + mem_batched_isolation are added fresh here (safe: they change ONLY the runtime retrieval
# mode, not any weight shape) -- required for multihop_hard_neg_full's 200-hard-neg-per-query
# scale to fit at all (see wiki/implementations/2026-08-02-hard-neg-full-efficient-retrieval.md),
# and doc_access_per_query_loss (not doc_access_loss) is required under that mode.
#
# SEED: dataset.shuffle_seed overridden away from the repo-wide default (42) specifically so
# this run's data order doesn't replay whatever a same-seed run already saw -- avoids the
# "weird epoch artifacts" of two runs walking through the deterministic interleave/pool in
# lockstep.
#
# At tp_devices=1 this OOMs compiling jit__train_step (RESOURCE_EXHAUSTED: needs 23.06G, 20.35G
# free -- short by ~2.7G): unfrozen main_model + multihop_hard_neg_full's large activation
# footprint (256 chunks/doc, bs=8) doesn't fit replicated on one v6e chip.
#
# tp_devices=2 (the usual fix, see wiki/training/optimizer.md) does NOT work here: Qwen3's
# embed_tokens/lm_head vocab dim (151669 = 7 x 21667, no factor of 2) can't be sharded 2 ways --
# `jax.device_put` raises "array axis 0 is partitioned 2 times, but does not evenly divide the
# dimension size 151669" the moment the checkpoint restore tries to reshard it. A structural
# limit, not something a different tp_devices value routes around (151669's only nontrivial
# divisor besides itself is 7).
#
# Fix 1: trainer's trainable_params excludes `main_model.embed_tokens`/`lm_head` and
# `embed_model.embed_tokens` (see staged_batched_isolation_docaccess_warmstart.yaml) -- they
# stay bf16, frozen, no optimizer moments. Fine-tuning the token embedding table was never the
# point of this recipe; the transformer layers + final norm still fully unfreeze. Closed ~0.8G
# of the ~2.7G gap (confirmed 2026-08-13: OOM shrank from "need 23.06G, 20.35G free" to "need
# 22.75G, 20.86G free") -- not enough alone.
#
# Fix 2: dataset.num_chunks_per_doc cut 256->224 (~12%) to shrink the remaining ~1.89G, at the
# cost of hard-neg coverage (~204 docs/query -> ~180). Chose this over reverting main_model to
# frozen (which would have kept full 200-neg coverage) to preserve tonight's "keep main_model
# trainable" decision. `jax.remat` is already applied to every transformer layer
# (models/qwen3.py) -- gradient checkpointing isn't an additional lever here, it's already on.
#
# DATA: same offline-parquet + doc-id corpus as the ground4layer scripts. Stage on the box first
# (idempotent): uv run python datagen/download_multihop_hardneg.py
LRMASKED_RUNDIR="${LRMASKED_RUNDIR:-gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45}"
LRMASKED_CKPT="${LRMASKED_RUNDIR}/qwen3_mem_embed/${LRMASKED_STEP:-100000}"
HF_HUB_OFFLINE=1 GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet}" \
MULTIHOP_CORPUS="${MULTIHOP_CORPUS:-$HOME/hf_parquet/multihop_doc_corpus.arrow}" \
uv run train.py \
    ${RUN_START_TIME:+"+trainer.run_start_time=$RUN_START_TIME"} \
    model=qwen3_mem_embed \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=64 \
    +model.memory.per_query_isolation=true \
    +model.memory.isolation_group_size=1 \
    +model.memory.mem_batched_isolation=true \
    model.memory.mem_collect_full_scores=true \
    dataset=multihop_hard_neg_full \
    dataset.shuffle_seed="${MULTIHOP_SHUFFLE_SEED:-20260813}" \
    dataset.num_chunks_per_doc="${MULTIHOP_NUM_CHUNKS_PER_DOC:-224}" \
    trainer=staged_batched_isolation_docaccess_warmstart \
    trainer.checkpoint_interval=200 \
    trainer.max_to_keep=20 \
    trainer.log_interval=10 \
    trainer.resume_from="${RESUME_FROM:-$LRMASKED_CKPT}" \
    +trainer.wandb_run_id=auto \
    +trainer.run_name="multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8"
# RESUME_FROM (env, run-dir with NO trailing step) takes priority over the warm start above and
# resumes THIS run (model + optimizer + dataloader) from its own latest checkpoint after a
# preemption -- same convention as the ground4layer scripts.
