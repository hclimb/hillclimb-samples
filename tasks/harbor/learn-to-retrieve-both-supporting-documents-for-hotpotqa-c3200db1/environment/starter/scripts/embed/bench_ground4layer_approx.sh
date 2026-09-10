#!/bin/bash
# A/B step-time bench for the ground_s1_zeroinit_4layer config: exact vs approx top-k on
# whatever box this runs on (written for the flex-start v5p-4; runbook §2.2, TRANSPORT=gce).
# Mirrors bench_approx_topk.sh but at the grounding shapes (train_ground_s1.sh):
# mem_layers=[9,14,20,27], mem_top_k=128, zero-init o_proj, trainer=staged_ground, same
# qa_hard_neg_think_sft4b dataset shapes -> M = 16x16x256 = 65,536 bank slots per layer.
# Synthetic batch (no HF/data staging needed); stage-0 semantics like the original bench.
#
# Launch:
#   TPU_NAME=tpu-v5p-4-uscentral1a ZONE=us-central1-a PROJECT_ID=memory-layers TRANSPORT=gce \
#   RUN_SCRIPT_PATH=scripts/embed/bench_ground4layer_approx.sh \
#     bash scripts/infrastructure/multi-vm-tpu-run.sh
BENCH_LOG="${BENCH_LOG:-$HOME/bench_ground4layer_approx.log}"
GROUND_OVERRIDES=(
  model.memory.mem_top_k=128
  model.memory.mem_o_proj_zero_init=true
  'model.memory.mem_layers=[9,14,20,27]'
  trainer=staged_ground
)
{
  echo "############ host $(hostname)  $(date -u) ############"
  echo "############ EXACT  top_k (MEM_APPROX_TOPK=0) ############"
  MEM_APPROX_TOPK=0 uv run python scripts/embed/bench_approx_topk.py --mode time \
      --overrides "${GROUND_OVERRIDES[@]}"

  echo "############ APPROX top_k  recall_target=0.99 ############"
  MEM_APPROX_TOPK=1 MEM_APPROX_RECALL=0.99 uv run python scripts/embed/bench_approx_topk.py --mode time \
      --overrides "${GROUND_OVERRIDES[@]}"

  echo "############ DONE  $(date -u) ############"
} 2>&1 | tee "$BENCH_LOG"
