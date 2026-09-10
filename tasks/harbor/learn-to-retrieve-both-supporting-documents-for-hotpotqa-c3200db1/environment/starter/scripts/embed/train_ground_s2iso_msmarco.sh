#!/usr/bin/env bash
# ground_s2_kv_split arch + per-query isolation + FULL fine-tune, on msmarco only. Tests whether the
# K/V-split arch also grounds strongly under the winning recipe (full-FT + isolation + reliable
# single-doc memory) that made 4B_msmarco_triplets... hit ~0.84 isolated-oracle. Unlike the frozen
# s2 runs, ALL params train (trainable=[".*"]). 1 doc/query (isolated bank), no doc_access (no
# retrieval yet). msmarco loads ONLINE (grounding-exps offline-parquet has no msmarco-triplets).
# OOM -> reduce dataset.batch_size (full-FT 4B + embed/value is heavier than the frozen runs).
uv run train.py \
    model=qwen3_mem_embed_ground_s2 \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=32 \
    +model.memory.per_query_isolation=true \
    dataset=msmarco_triplets_sft4b \
    trainer=standard_ground \
    eval_set@trainer.evals=none \
    trainer.steps=80000 \
    trainer.checkpoint_interval=10000 \
    trainer.aux_losses.doc_access_loss.enabled=false \
    '+trainer.training_stages=[{trainable_params: [".*"], max_step: ${trainer.steps}, ce_weight: 1.0, lr_schedule: cosine, warmup_frac: 0.05}]' \
    +trainer.run_name="ground_s2iso_msmarco_ft" \
    ${RESUME_FROM:+ trainer.resume_from="$RESUME_FROM"}
