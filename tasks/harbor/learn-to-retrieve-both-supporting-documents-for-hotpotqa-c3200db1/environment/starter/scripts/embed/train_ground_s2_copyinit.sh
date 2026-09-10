#!/usr/bin/env bash
# ground_s2_kv_split QA run WARM-STARTED from the doc-copy pretrain (ground_copy_topk4 step 4000).
# Same s2 arch/recipe (K/V split, zero-init o_proj, mem_layers=[9,14,20,27], staged_ground, QA data)
# but initialized from the copy-pretrained weights: the memory read/write channel that copy training
# made sharp+gold-locked (eff_slots ~2.5, pos_mass .97) is carried in; QA fine-tune should then learn
# selective extraction on top of a working channel.
#   run_name = ground_s2_copyinit (NOT ground_s2_kv_split -> avoids colliding with box2's run).
#   top_k = 128 (standard s2). NOTE: copy pretrain used top_k=4; at 128 the read has 128 candidates
#     and may re-diffuse -> if the sharp read degrades, rerun with model.memory.mem_top_k=4.
# Warm start = resume_from ends in /4000 -> load_checkpoint restores WEIGHTS ONLY (fresh optimizer,
# step 0). On preemption, RESUME_FROM (own latest run-dir) overrides for a full resume.
HF_HUB_OFFLINE=1 uv run train.py \
    model=qwen3_mem_embed_ground_s2 \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=128 \
    dataset=qa_hard_neg_think_sft4b \
    trainer=staged_ground \
    trainer.checkpoint_interval=2000 \
    trainer.resume_from="${RESUME_FROM:-gs://memory-layers-training/ground_copy_topk4-2026-07-04-21-37-04/qwen3_mem_embed/4000}" \
    +trainer.run_name="ground_s2_copyinit"
