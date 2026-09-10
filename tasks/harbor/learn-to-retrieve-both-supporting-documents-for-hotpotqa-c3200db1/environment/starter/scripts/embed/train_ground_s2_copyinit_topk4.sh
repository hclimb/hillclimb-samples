#!/usr/bin/env bash
# s2 QA run warm-started from the doc-copy pretrain (ground_copy_topk4 step 4000), at mem_top_k=4
# — matching the top_k the copy channel was trained at, so the sharp/gold-locked read (eff_slots
# ~2.5, pos_mass .97) is preserved into QA instead of re-diffusing over 128 candidates. Companion to
# ground_s2_copyinit (top_k=128); this box tests whether keeping top_k=4 carries the copy channel.
#   run_name = ground_s2_copyinit_topk4. Replaces s1 (which plateaued: eval acc uncorrelated w/ steps).
# Warm start = resume_from ends in /4000 -> weights only, fresh optimizer, step 0. On preemption,
# RESUME_FROM (own latest run-dir) overrides for a full resume.
HF_HUB_OFFLINE=1 uv run train.py \
    model=qwen3_mem_embed_ground_s2 \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=4 \
    dataset=qa_hard_neg_think_sft4b \
    trainer=staged_ground \
    trainer.checkpoint_interval=2000 \
    trainer.resume_from="${RESUME_FROM:-gs://memory-layers-training/ground_copy_topk4-2026-07-04-21-37-04/qwen3_mem_embed/4000}" \
    +trainer.run_name="ground_s2_copyinit_topk4"
