#!/usr/bin/env bash
# Same as train_grp4_from_iso1.sh (4-doc from grounded iso1@20k, FULL fine-tune) but with
# doc_access_loss OFF — isolates whether doc_access_loss (summed over all timesteps) is what
# degraded the grounding. If grounding still decays without it, the culprit is full-FT on the
# 4-doc answer-CE itself; if it holds, doc_access_loss was the problem. msmarco ONLINE.
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
    trainer.aux_losses.doc_access_loss.enabled=false \
    trainer.aux_losses.doc_access_loss.weight=0.0 \
    '+trainer.training_stages=[{trainable_params: [".*"], max_step: ${trainer.steps}, ce_weight: 1.0, lr_schedule: cosine, warmup_frac: 0.05}]' \
    +trainer.run_name="ground_grp4_noda_from_iso1_20k" \
    trainer.resume_from="${RESUME_FROM:-$INIT_CKPT}"
