#!/usr/bin/env bash
# 4-doc retrieval curriculum, warm-started from the FRESH well-grounded 1-doc checkpoint
# msmarco_iso1_valce@20000 (clean bs=1 oracle ~0.79, train CE ~0.27 = well short of the
# memorization regime, so NOT overfit — the whole point of the 1-doc rerun). Same arch as the
# 1-doc run (qwen3_mem_embed, mem_top_k=32, per_query_isolation) so weights load; now
# isolation_group_size=4 shows each query its own doc + 3 in-batch negs, and doc_access_loss
# supervises picking the positive among the 4. FULL fine-tune, msmarco ONLINE. Frequent ckpts
# (2000) so we can judge the 4-doc oracle (bs=4) per step and watch grounding-vs-retrieval.
# eval_set=msmarco_valce (cheap held-out CE; oracle LLM-judge acc is graded externally per ckpt).
INIT_CKPT="gs://memory-layers-training/msmarco_iso1_valce-2026-07-06-12-15-28/qwen3_mem_embed/20000"
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
    '+trainer.training_stages=[{trainable_params: [".*"], max_step: ${trainer.steps}, ce_weight: 1.0, lr_schedule: cosine, warmup_frac: 0.05}]' \
    +trainer.run_name="ground_grp4_from_iso1_20k" \
    trainer.resume_from="${RESUME_FROM:-$INIT_CKPT}"
