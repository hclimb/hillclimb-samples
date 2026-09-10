#!/usr/bin/env bash
# Parametric launcher for the grounding@1024 experiments. Same recipe as
# train_hard_neg_think.sh (the qhn64 baseline: qa_hard_neg_think_sft4b, mem_top_k=64,
# seq512/chunks16/bs16) with ONE lever changed per run via $EXTRA. Trains FRESH (random
# mem init on Qwen3-4B), matching how the baseline was trained. Preemption-resume via
# $RESUME_FROM (latest checkpoint of THIS run; empty -> fresh).
#
#   RUN_NAME=ground1024_relu EXTRA="model.memory.mem_score_activation=relu" \
#   RESUME_FROM="gs://.../NNNN" bash scripts/embed/ground1024.sh
set -e
RUN_NAME="${RUN_NAME:?set RUN_NAME}"
EXTRA="${EXTRA:-}"

# Offline HF: read pre-cached local parquet (scripts/misc/precache_hf.sh) => ZERO HF API calls,
# no 429 (5 boxes x 16 grain workers would otherwise blow the 1000-req/5min quota). Requires the
# parquet cache to exist; the babysitter precaches on fresh/recreated boxes before dispatch.
export HF_HUB_OFFLINE=1
export GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet}"

uv run train.py \
    model=qwen3_mem_embed \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=64 \
    dataset=qa_hard_neg_think_sft4b \
    trainer.steps=100000 \
    trainer.eval_interval=100000000 \
    trainer.checkpoint_interval=2000 \
    eval_set@trainer.evals=none \
    +trainer.run_name="$RUN_NAME" \
    ${RESUME_FROM:+trainer.resume_from=$RESUME_FROM} \
    $EXTRA
