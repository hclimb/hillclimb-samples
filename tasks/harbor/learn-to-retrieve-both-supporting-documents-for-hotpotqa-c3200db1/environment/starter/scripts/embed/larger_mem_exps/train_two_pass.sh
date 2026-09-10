#!/bin/bash
set -e
set -a; source "$(dirname "$0")/../../../.env"; set +a

CHECKPOINT=gs://memory-layers-training/4B_pretraining_cot_unfreeze_all_topk_128_norm-2026-04-12-00-36-01/qwen3_mem_embed

uv run train.py \
    model=qwen3_mem_embed \
    dataset=pretraining \
    eval_set@trainer.evals=pretraining \
    +trainer.run_name="4B_pretraining_two_pass" \
    'model.memory.two_pass_topk=true' \
    'model.memory.mem_lookup_chunk_size=16384' \
    'dataset.batch_size=512' \
    'model.trainable_params=[".*"]' \
    'trainer.training_stages=null' \
    '+trainer.lr_schedule=cosine' \
    '+trainer.grad_accum_steps=32' \
    '++trainer.ce_weight=0.1' \
    'trainer.aux_losses.doc_access_loss.enabled=false' \
    'trainer.aux_losses.doc_access_acc.enabled=false' \
    '+trainer.aux_losses.doc_access_top_k_loss.enabled=true' \
    '+trainer.aux_losses.doc_access_top_k_loss.weight=1.0' \
    '+trainer.aux_losses.doc_access_top_k_loss.temperature=1.0' \
    'trainer.steps=115000' \
    'trainer.checkpoint_interval=250' \
    "trainer.resume_from=${CHECKPOINT}"
