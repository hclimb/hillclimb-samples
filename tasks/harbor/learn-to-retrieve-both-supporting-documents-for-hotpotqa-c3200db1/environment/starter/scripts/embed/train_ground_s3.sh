#!/usr/bin/env bash
# Stage 3 grounding run: span/neighbor value readout (stacks on Stage 2 K/V split).
#   span_window=1 => when slot t is retrieved, pool values of t-1,t,t+1 (respecting doc
#   boundaries) so the answer token rides in even when the match keyed on a cue token.
#   Carries Stage 1+2 forward: zero-init, mem_layers=[9,14,20,27], separate value model.
# Recipe/data identical to the other runs. See grounding plan Stage 3.
# Reads the pre-downloaded local parquet subset (scripts/misc/precache_hf.sh) offline — no HF 429.
HF_HUB_OFFLINE=1 uv run train.py \
    model=qwen3_mem_embed_ground_s3 \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=128 \
    dataset=qa_hard_neg_think_sft4b \
    trainer=staged_ground \
    trainer.checkpoint_interval=2000 \
    +trainer.run_name="ground_s3_span" \
    ${RESUME_FROM:+ trainer.resume_from="$RESUME_FROM"}
# RESUME_FROM (env) = run-dir gs://.../qwen3_mem_embed -> full resume (step+optimizer+dataloader).
