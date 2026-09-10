#!/bin/bash
# Fit + speed probe for the "all ~204 docs in the memory layer" recipe.
#
# NOT a training run. Answers two questions before anything long is launched:
#   1. does num_chunks_per_doc=256 at BS (default 2) fit in HBM, or do we need BS=1?
#   2. what does one step cost? (the 16-chunk recipe is ~490ms; 8-16x slower is expected)
#
# Checkpointing and evals are off so nothing is written to GCS and the TPU is released
# quickly. Override BS / CHUNKS / STEPS from the environment to sweep.
#
# trainer=standard, NOT staged: staged.yaml hardcodes stage boundaries at 5000/10000/15000
# with the last stage at ${trainer.steps}, so a short probe trips
# "Stage 3 max_step (6) must be > stage 2 max_step (15000)". Memory footprint per step is
# what we are measuring and that does not depend on the stage schedule.
set -uo pipefail

BS="${BS:-2}"
CHUNKS="${CHUNKS:-256}"
STEPS="${STEPS:-6}"
PROBE_LOG="${PROBE_LOG:-$HOME/probe_multihop_bs${BS}_ch${CHUNKS}.log}"

{
  echo "######## probe | $(hostname) | $(date -u) ########"
  echo "######## batch_size=$BS num_chunks_per_doc=$CHUNKS steps=$STEPS ########"

  # GCS_BUCKET="" makes _build_gcs_run_dir return None, so checkpointing stays local and the
  # probe never touches GCS. Without it, setup_checkpointing tries to CREATE a bucket at
  # startup and dies on the .env placeholders (GCS_BUCKET_PROJECT currently holds an inline
  # '# required: ...' comment, so the project id arrives empty -> 400 Unknown project id).
  HF_HUB_OFFLINE=1 \
  GCS_BUCKET="" \
  GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet}" \
  MULTIHOP_CORPUS="${MULTIHOP_CORPUS:-$HOME/hf_parquet/multihop_doc_corpus.arrow}" \
  uv run train.py \
      model=qwen3_mem_embed \
      model.main_model.model_id="Qwen/Qwen3-4B" \
      model.memory.mem_top_k=64 \
      dataset=multihop_hard_neg_full \
      dataset.batch_size="$BS" \
      dataset.num_chunks_per_doc="$CHUNKS" \
      dataset.num_workers=4 \
      trainer=standard \
      trainer.steps="$STEPS" \
      trainer.log_interval=1 \
      trainer.checkpoint_interval=1000000000 \
      trainer.eval_interval=1000000000 \
      trainer.use_wandb=false \
      'eval_set@trainer.evals=none' \
      +trainer.run_name="probe_multihop_bs${BS}_ch${CHUNKS}"
  echo "PROBE_EXIT=$?"
  echo "######## DONE $(date -u) ########"
} 2>&1 | tee "$PROBE_LOG"
