#!/usr/bin/env bash
# Stage 1 grounding run: NEW architecture + NEW recipe.
#   arch changes vs control: zero-init mem_o_proj (memory branch starts as identity;
#     any CE drop is provably memory-sourced) + multi-layer memory [9,14,20,27]
#     (more signal reaches the residual stream; later layers re-query an x that already
#     carries earlier reads -> iterative re-retrieval).
#   recipe/data identical to control (staged_ground, qa_hard_neg_think_sft4b), so the
#     Stage-1 delta over control is purely zero-init + multi-layer.
# top_k fixed at 128 across all runs; swept post-hoc via MEM_TOP_K, never folded in.
# See grounding_experiments_plan.md "Stage 1".
# Reads the pre-downloaded local parquet subset (scripts/misc/precache_hf.sh) offline — no HF 429.
HF_HUB_OFFLINE=1 uv run train.py \
    model=qwen3_mem_embed \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=128 \
    model.memory.mem_o_proj_zero_init=true \
    'model.memory.mem_layers=[9,14,20,27]' \
    dataset=qa_hard_neg_think_sft4b \
    trainer=staged_ground \
    trainer.checkpoint_interval=2000 \
    +trainer.run_name="ground_s1_zeroinit_4layer" \
    ${RESUME_FROM:+ trainer.resume_from="$RESUME_FROM"}
# RESUME_FROM (env) = a run-dir like gs://.../ground_s1_..-<ts>/qwen3_mem_embed -> full resume
# (restores step + optimizer + dataloader position). Empty = fresh launch from step 0.
