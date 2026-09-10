#!/usr/bin/env bash
# Control run for the grounding experiments: OLD architecture + NEW recipe.
# Apples-to-apples baseline the grounding stages are measured against — holds recipe
# and data fixed so any Stage-1 delta is purely zero-init + multi-layer.
#   old arch: mem_layers=[14], mem_o_proj old init (*0.02, NOT zero), 4 heads,
#             single shared Qwen3-Embedding-0.6B (no K/V split).
#   new recipe: staged_ground (2-stage, main frozen), qa_hard_neg_think_sft4b data.
# In-loop eval is OFF (staged_ground eval_interval); judged accuracy comes from the
# dedicated eval boxes. See grounding_experiments_plan.md "Control run".
# Reads the pre-downloaded local parquet subset (scripts/misc/precache_hf.sh) offline — no HF 429.
HF_HUB_OFFLINE=1 uv run train.py \
    model=qwen3_mem_embed \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=128 \
    dataset=qa_hard_neg_think_sft4b \
    trainer=staged_ground \
    trainer.checkpoint_interval=2000 \
    +trainer.run_name="ground_control" \
    ${RESUME_FROM:+ trainer.resume_from="$RESUME_FROM"}
# RESUME_FROM (env) = run-dir gs://.../qwen3_mem_embed -> full resume (step+optimizer+dataloader).
