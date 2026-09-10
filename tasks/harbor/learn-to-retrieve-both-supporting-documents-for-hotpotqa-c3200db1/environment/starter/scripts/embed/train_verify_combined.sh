#!/bin/bash
# End-to-end verification of the FULLY-OPTIMIZED recipe: offline-parquet + log_interval=10 (loop
# pipelining) + stop_grad_frozen=true (Axis A). Runs real train.py for 200 steps (stage 0). Checkpoint
# -> /dev/shm (RAM, fast, disposable) via empty GCS_BUCKET + hydra.run.dir. Confirms the adopted stack
# trains without error and measures real steps/sec. Tees to $VER_LOG.
#   TPU_NAME=rohun-v6e-8-0 RUN_SCRIPT_PATH=scripts/embed/train_verify_combined.sh \
#     bash scripts/infrastructure/multi-vm-tpu-run.sh
VER_LOG="${VER_LOG:-$HOME/train_verify_combined.log}"
set -a; . "$HOME/.env" 2>/dev/null || . ".env" 2>/dev/null || true; set +a
rm -rf /dev/shm/verifyrun_DELETEME
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
    trainer.stop_grad_frozen=true \
    trainer.eval_interval=0 \
    trainer.checkpoint_interval=100000 \
    trainer.use_wandb=false \
    eval_set@trainer.evals=none \
    hydra.run.dir=/dev/shm/verifyrun_DELETEME 2>&1 \
    | grep -vE 'Grain multiprocess worker profiling' | tail -80 \
    || echo "!!!! VERIFY RUN FAILED"
  rm -rf /dev/shm/verifyrun_DELETEME
  echo "############ DONE  $(date -u) ############"
} 2>&1 | tee "$VER_LOG"
