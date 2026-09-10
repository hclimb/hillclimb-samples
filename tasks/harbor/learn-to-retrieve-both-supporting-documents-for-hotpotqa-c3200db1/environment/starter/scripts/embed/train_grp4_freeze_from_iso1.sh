#!/usr/bin/env bash
# 4-doc retrieval curriculum from grounded iso1@20k, but with the MAIN MODEL FROZEN — only the
# memory read/write params (mem_*) + embed/retrieval params train. Motivation: full fine-tuning
# DEGRADED the grounding the init came with (best at 2k, monotonic decline after; the 4-doc
# answer-CE + doc_access_loss drifted the reader toward parametric shortcuts). Freezing the main
# transformer preserves the grounded reading while the memory params learn to retrieve among 4.
# Same arch/init as train_grp4_from_iso1.sh; doc_access_loss stays ON (0.1). msmarco ONLINE.
INIT_CKPT="gs://memory-layers-training/msmarco_iso1_valce-2026-07-06-12-15-28/qwen3_mem_embed/26000"
uv run train.py \
    model=qwen3_mem_embed \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=32 \
    +model.memory.per_query_isolation=true \
    +model.memory.isolation_group_size=4 \
    dataset=msmarco_triplets_sft4b \
    trainer=standard_ground \
    eval_set@trainer.evals=msmarco_valce \
    trainer.steps=80000 \
    trainer.eval_interval=2000 \
    trainer.checkpoint_interval=2000 \
    trainer.aux_losses.doc_access_loss.enabled=true \
    trainer.aux_losses.doc_access_loss.weight=0.1 \
    '+trainer.training_stages=[{trainable_params: [".*mem_.*", ".*embed_model.*"], max_step: ${trainer.steps}, ce_weight: 1.0, lr_schedule: cosine, warmup_frac: 0.05}]' \
    +trainer.run_name="ground_grp4_freeze_from_iso1_20k" \
    trainer.resume_from="${RESUME_FROM:-$INIT_CKPT}"
