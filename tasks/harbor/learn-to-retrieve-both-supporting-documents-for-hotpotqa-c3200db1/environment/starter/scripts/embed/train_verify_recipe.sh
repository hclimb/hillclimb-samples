#!/bin/bash
# Capstone: run the ADOPTED recipe end-to-end via real train.py — offline-parquet + log_interval=10,
# through all 4 staged transitions (boundaries shrunk to 50/100/150 so a 200-step run exercises them).
# stop_grad_frozen stays OFF (recipe default). Checkpoint -> /dev/shm (RAM, disposable). Confirms the
# adopted stack trains without error across stages. Tees to $REC_LOG.
#   TPU_NAME=rohun-v6e-8-0 RUN_SCRIPT_PATH=scripts/embed/train_verify_recipe.sh \
#     bash scripts/infrastructure/multi-vm-tpu-run.sh
REC_LOG="${REC_LOG:-$HOME/train_verify_recipe.log}"
set -a; . "$HOME/.env" 2>/dev/null || . ".env" 2>/dev/null || true; set +a
rm -rf /dev/shm/verifyrecipe_DELETEME
{
  echo "############ host $(hostname)  $(date -u) ############"
  GCS_BUCKET= HF_HUB_OFFLINE=1 GROUND_HF_PARQUET="$HOME/hf_parquet" \
  uv run train.py \
    model=qwen3_mem_embed \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=64 \
    dataset=qa_hard_neg_think_sft4b \
    trainer.steps=200 \
    trainer.log_interval=10 \
    trainer.training_stages.0.max_step=50 \
    trainer.training_stages.1.max_step=100 \
    trainer.training_stages.2.max_step=150 \
    trainer.eval_interval=0 \
    trainer.checkpoint_interval=100000 \
    trainer.use_wandb=false \
    eval_set@trainer.evals=none \
    hydra.run.dir=/dev/shm/verifyrecipe_DELETEME 2>&1 \
    | grep -vE 'Grain multiprocess worker profiling' | tail -70 \
    || echo "!!!! RECIPE VERIFY FAILED"
  rm -rf /dev/shm/verifyrecipe_DELETEME
  echo "############ DONE  $(date -u) ############"
} 2>&1 | tee "$REC_LOG"
