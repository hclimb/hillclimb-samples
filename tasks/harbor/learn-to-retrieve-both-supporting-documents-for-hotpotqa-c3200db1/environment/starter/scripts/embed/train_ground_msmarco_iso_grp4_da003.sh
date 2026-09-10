#!/usr/bin/env bash
# Retrieval-curriculum step: warm-start from the 1-doc msmarco per-query-isolation checkpoint
# (step 60000, ~0.84 isolated grounding) and fine-tune to distinguish among 4 docs, where the 3
# negatives are STANDARD in-batch negs (other queries' docs) via isolation_group_size=4 -- NOT
# dataset hard-negs. Bank = the batch's pos docs (1 chunk each, C=1, msmarco_triplets 1-doc); the
# group mask shows each query its own doc + 3 batch-mates. doc_access_loss supervises picking the
# pos among the 4. Goal: keep grounding, ADD retrieval. Same arch as the msmarco run
# (qwen3_mem_embed, mem_layers=[14], top_k=32) so the checkpoint loads. FULL fine-tune. msmarco ONLINE.
# NOTE batch_size must be divisible by the group size (32 % 4 == 0). OOM -> reduce dataset.batch_size
# (keep it a multiple of 4).
INIT_CKPT="gs://memory-layers-training/4B_msmarco_triplets_topk32_per_query_isolation-2026-07-05-01-00-30/qwen3_mem_embed/60000"
uv run train.py \
    model=qwen3_mem_embed \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=32 \
    +model.memory.per_query_isolation=true \
    +model.memory.isolation_group_size=4 \
    dataset=msmarco_triplets_sft4b \
    trainer=standard_ground \
    eval_set@trainer.evals=pretraining \
    trainer.steps=80000 \
    trainer.checkpoint_interval=10000 \
    trainer.aux_losses.doc_access_loss.enabled=true \
    trainer.aux_losses.doc_access_loss.weight=0.03 \
    '+trainer.training_stages=[{trainable_params: [".*"], max_step: ${trainer.steps}, ce_weight: 1.0, lr_schedule: cosine, warmup_frac: 0.05}]' \
    trainer.resume_from="${RESUME_FROM:-$INIT_CKPT}" \
    +trainer.run_name="ground_msmarco_iso_grp4_da003_ft"
