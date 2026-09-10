#!/usr/bin/env bash
# Stage 2 grounding run: separate KEY / VALUE models (stacks on Stage 1 arch).
#   embed_model = key model (contrastive trunk); value_model = base Qwen3-0.6B (trainable),
#   supplying memory VALUES with the per-token surface fidelity the contrastive trunk loses.
#   Carries Stage 1 forward: zero-init mem_o_proj + mem_layers=[9,14,20,27].
# Recipe/data identical to control/s1 (staged_ground, qa_hard_neg_think_sft4b); Stage B
# trainable_params already covers .*value_model.*. See grounding plan Stage 2.
# Reads the pre-downloaded local parquet subset (scripts/misc/precache_hf.sh) offline — no HF 429.
HF_HUB_OFFLINE=1 uv run train.py \
    model=qwen3_mem_embed_ground_s2 \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=128 \
    dataset=qa_hard_neg_think_sft4b \
    trainer=staged_ground \
    trainer.checkpoint_interval=2000 \
    +trainer.run_name="ground_s2_kv_split" \
    ${RESUME_FROM:+ trainer.resume_from="$RESUME_FROM"}
# RESUME_FROM (env) = run-dir gs://.../qwen3_mem_embed -> full resume (step+optimizer+dataloader).
