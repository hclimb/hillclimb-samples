#!/usr/bin/env bash
# Finetunes the hard-neg-think 4B checkpoint (qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16
# _pf32_indexed_lr_masked, step 100000 -- https://wandb.ai/johnzhang2366-columbia-university/
# memory-layers/runs/qa_hard_neg_think_sft4b_topk64_seq512_ch-2026-08-09-05-49-45) on
# vm2825/hotpotqa-hard-neg-reasoning-embedding-modified-SFT-4B-parts-0-1 (78,755 HotpotQA rows,
# 1 pos_doc + 3 neg_docs each). Same warm-start checkpoint as train_multihop_lrmasked_warmstart.sh
# and its CoT-ablation siblings, but this arm is a plain single-source QA hard-neg finetune (no
# per-query/batched isolation -- only 3 negs/row, nowhere near multihop_hard_neg_full's ~200/query
# scale that machinery exists for), so it follows the simpler
# train_musique_sft_midtrain.sh / train_multihop_sft_midtrain.sh recipe instead, but with
# trainer=midtraining_full_telemetry (NOT the layers-13/14/15-only midtraining_telemetry those two
# scripts use): the checkpoint's own lineage (staged.yaml) reaches its stage 3 with mem_* +
# embed_model + the WHOLE main_model trainable, not just a 3-layer neighbourhood, so this
# continues that same trainable set as one stage rather than narrowing it. NOT staged* itself --
# its stage 0 re-freezes main_model and zeroes ce_weight for a fresh-model warmup, which would
# undo a checkpoint that's already past that stage.
#
# neg_score_threshold: 0.0 (configs/dataset/sources/hotpotqa_hard_neg_reasoning_modified.yaml) --
# this dataset's neg_scores column is a constant placeholder ("0.0<doc_seperator>0.0<doc_seperator>
# 0.0" on every one of the 78,755 rows, confirmed via the HF datasets-server rows endpoint), not a
# real mined similarity score, so there is nothing to threshold away; 0.0 keeps all 3 negs/row
# (make_normalizer keeps score <= threshold, and 0.0 <= 0.0) and states that intent explicitly
# rather than inheriting the sibling sources' 0.95 (tuned for a real 0-1 similarity score, where it
# also happens to pass 0.0 -- but for the wrong reason, and would silently stop doing so if this
# dataset ever gets real scores added later).
#
# PREREQUISITE — stage the parquets first, on the box:
#     bash scripts/misc/download_hotpotqa_hard_neg_reasoning_data.sh
# BOTH env vars below are required. GROUND_HF_PARQUET alone does NOTHING (data/qa.py:313 gates
# the offline branch on HF_HUB_OFFLINE == "1"), and live-HF is a livelock, not a fallback
# (wiki/data/hf-rate-limits.md).
#
# ─────────────────────────────────────────────────────────────────────────────────────────
# WARM START, NOT RESUME. RESUME_FROM ends in "/100000" and that trailing step is LOAD-BEARING:
# utils.py::setup_checkpointing parses a trailing digit group into resume_step, and
# load_checkpoint then takes the "warm start: restore weights only, fresh optimizer, step 0"
# branch. Drop the "/100000" and it takes the OTHER branch -- full resume of weights + optimizer +
# dataloader position at step 100000 -- which, with trainer.steps=10000, would exit immediately
# having trained nothing.
RESUME_FROM="${RESUME_FROM:-gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45/qwen3_mem_embed/100000}"
#
# model.memory.mem_top_k=64 MUST match the checkpoint (default is 128); every other memory/embed
# field, and dataset seq_len/doc_chunk_seq_len/num_chunks_per_doc/batch_size, already match the
# run that produced it (configs/dataset/hotpotqa_hard_neg_reasoning_modified_finetune.yaml mirrors
# qa_hard_neg_think_sft4b.yaml's shape) -- so the memory bank is the same 16 x 16 x 256 = 65,536
# slots/batch it was trained at, not a resharded/resized one.
#
# learning_rates {mem:1e-4, embed:1e-5, main:1e-5} now live in configs/trainer/midtraining.yaml
# itself (not overridden here) -- mirrors staged.yaml's own per-group rates
# (utils.py::setup_optimizer_for_stage, opt-in via cfg.trainer.learning_rates), inherited by every
# midtraining/midtraining_telemetry/midtraining_frozen_telemetry/midtraining_full_telemetry
# config. Needed here specifically because midtraining_full_telemetry unfreezes the WHOLE
# main_model (not just layers 13/14/15): a single uniform rate would either apply staged's
# aggressive mem-only rate to the full 4B (staged.yaml/staged_sim never run main_model above
# 1e-5, "to protect the 4B") or under-train the memory layers relative to how this checkpoint's
# own lineage was produced. ⚠️ Base-config change, not a per-script override -- it also changes
# the effective LR for train_musique_sft_midtrain.sh / train_multihop_sft_midtrain.sh (currently
# a uniform trainer.learning_rate=1e-4 override, now inert) and
# train_musique_ground4layer_midtrain.sh on any future re-run; not reconciled with those scripts'
# own documented comparisons.
# warmup_frac=0.1 added because midtraining_full_telemetry ships with none: a warm start restores
# weights but NOT the optimizer, so Adam's moments start at zero and land on an already-converged
# 4B without it -- same reasoning as every other warm-start script here.
#
# steps=10000 x batch_size=16 = 160,000 samples over 78,755 rows = ~2.03 epochs. checkpoint_interval
# =500 x max_to_keep=20 = a 10,000-step window, i.e. the whole run stays evaluable (same "interval x
# keep >= total steps" rule as every other run here -- see wiki/infrastructure/checkpointing.md).
#
# ⚠️ COST (midtraining_full_telemetry's own header warning): unfreezing the whole main_model
# allocates Adam moments for every one of its params, on top of a memory bank that's already
# large. Not expected to OOM here specifically: this is the SAME trainable set at the SAME bank
# size (16 x 16 x 256 = 65,536 slots/batch) that the checkpoint's own stage 3 (staged.yaml) already
# trained at to reach step 100000 -- if that fit, this does too. If it still OOMs, drop
# dataset.batch_size or dataset.num_chunks_per_doc first (same fallback ladder as every other
# script here), not the trainable set.
#
# EVALS: in-loop eval is off, same rationale as every other script in this directory --
# llm_judge_accuracy is computed in eval.py's PARENT process after the JAX worker frees the TPU,
# so a task's `metrics:` block is inert in-loop (data/qa.py comments in train_hard_neg_think.sh
# have the full explanation). Score checkpoints from a separate eval box instead.
HF_HUB_OFFLINE=1 GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet_hotpotqa_hardneg_reasoning}" \
uv run train.py \
    ${RUN_START_TIME:+"+trainer.run_start_time=$RUN_START_TIME"} \
    model=qwen3_mem_embed \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=64 \
    dataset=hotpotqa_hard_neg_reasoning_modified_finetune \
    trainer=midtraining_full_telemetry \
    trainer.resume_from="$RESUME_FROM" \
    trainer.steps=10000 \
    +trainer.training_stages.0.warmup_frac=0.1 \
    trainer.checkpoint_interval=500 \
    trainer.max_to_keep=20 \
    trainer.log_interval=10 \
    trainer.eval_interval=1000000000 \
    'eval_set@trainer.evals=none' \
    +trainer.wandb_run_id=auto \
    +trainer.run_name="hotpotqa_hard_neg_reasoning_finetune_topk64_seq512_chunks16_bs16"
# RESUME_FROM (env, run-dir with NO trailing step) takes priority over the warm start above and
# resumes THIS run (model + optimizer + dataloader) from its own latest checkpoint after a
# preemption -- same convention as the other midtrain/warm-start scripts.
