#!/usr/bin/env bash
# Copy-init QA + small MLP LoRA on the MAIN model (gate/up/down_proj, rank 16). Tests whether letting
# the frozen main model learn to USE the memory-injected residual cuts parametric dependence.
#   arch = qwen3_mem_embed_copyinit_lora (s2 K/V split, zero-init o_proj, 4 mem layers, top_k=4,
#     main_model.lora enabled). Warm-start from the doc-copy pretrain (ground_copy_topk4 step 4000).
#   A/Bs against box1 = ground_s2_copyinit_topk4 (same recipe, NO LoRA).
# Stage B trainable overridden to ALSO train the main LoRA adapters (*_a_proj/*_b_proj); main base 4B
# stays frozen. LoRA B zero-init -> starts as no-op. Replaces ground_s2_kv_split on this box.
# Warm start = resume_from ends in /4000 (weights only, fresh optimizer, step 0). On preemption,
# RESUME_FROM (own latest run-dir) overrides for a full resume.
HF_HUB_OFFLINE=1 uv run train.py \
    model=qwen3_mem_embed_copyinit_lora \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=4 \
    dataset=qa_hard_neg_think_sft4b \
    trainer=staged_ground \
    trainer.checkpoint_interval=2000 \
    'trainer.training_stages.1.trainable_params=[".*mem_.*",".*embed_model.*",".*value_model.*",".*main_model.*_a_proj",".*main_model.*_b_proj"]' \
    trainer.resume_from="${RESUME_FROM:-gs://memory-layers-training/ground_copy_topk4-2026-07-04-21-37-04/qwen3_mem_embed/4000}" \
    +trainer.run_name="ground_copyinit_lora_tk4"
