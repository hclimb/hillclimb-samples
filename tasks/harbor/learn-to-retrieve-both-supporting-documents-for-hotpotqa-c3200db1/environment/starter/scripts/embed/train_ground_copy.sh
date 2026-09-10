#!/usr/bin/env bash
# Doc-copy grounding experiment (replaces the killed s3 on box3). REMOVABLE — see data/doc_copy.py.
#   arch = Stage-2 (K/V split, zero-init mem_o_proj, mem_layers=[9,14,20,27]) but mem_top_k=4
#     (sharp read; s2 at top_k=128 stayed diffuse ~32 effective slots).
#   objective = reproduce the positive DOCUMENT from the memory bank (dataset=doc_copy_hard_neg);
#     the QA answer/CoT is ignored. Forces the read to carry the document's high-information tokens.
# Same staged_ground recipe (frozen main model; only mem_/embed_/value_ params train).
# Reads the pre-downloaded local parquet subset (scripts/misc/precache_hf.sh) offline — no HF 429.
HF_HUB_OFFLINE=1 uv run train.py \
    model=qwen3_mem_embed_copy_topk4 \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=4 \
    dataset=doc_copy_hard_neg \
    trainer=staged_ground \
    trainer.checkpoint_interval=2000 \
    +trainer.run_name="ground_copy_topk4" \
    ${RESUME_FROM:+ trainer.resume_from="$RESUME_FROM"}
# RESUME_FROM (env) = run-dir gs://.../qwen3_mem_embed -> full resume (step+optimizer+dataloader).
