#!/usr/bin/env bash
# Rerun of the 1-doc MS MARCO per-query-isolation training (the recipe that reached ~0.84
# isolated-oracle grounding), from base Qwen3-4B (fresh memory routers), FULL fine-tune.
# Purpose: track TRAIN CE vs held-out MS MARCO VAL CE at a fine interval to find the
# generalization sweet spot — the first checkpoint whose LLM-judge oracle accuracy peaks ~0.8,
# BEFORE overfitting sets in (we saw the 4-doc run drive train CE->0.05 while val CE rose and
# grounding fell). That checkpoint becomes the new init for the 4-doc retrieval curriculum.
#
# per_query_isolation + isolation_group_size=1 => each query sees only its OWN doc (1-doc bank).
# eval_set=msmarco_valce => ONLY the msmarco val-CE nll eval (frequent). doc_access off (no
# retrieval yet). Frequent checkpoints so the eval box can judge oracle accuracy per step and
# pinpoint the first 0.8. msmarco loads ONLINE. OOM -> reduce dataset.batch_size.
uv run train.py \
    model=qwen3_mem_embed \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=32 \
    +model.memory.per_query_isolation=true \
    +model.memory.isolation_group_size=1 \
    dataset=msmarco_triplets_sft4b \
    trainer=standard_ground \
    eval_set@trainer.evals=msmarco_valce \
    trainer.steps=80000 \
    trainer.eval_interval=1000 \
    trainer.checkpoint_interval=2000 \
    trainer.aux_losses.doc_access_loss.enabled=false \
    '+trainer.training_stages=[{trainable_params: [".*"], max_step: ${trainer.steps}, ce_weight: 1.0, lr_schedule: cosine, warmup_frac: 0.05}]' \
    +trainer.run_name="msmarco_iso1_valce" \
    ${RESUME_FROM:+ trainer.resume_from="$RESUME_FROM"}
