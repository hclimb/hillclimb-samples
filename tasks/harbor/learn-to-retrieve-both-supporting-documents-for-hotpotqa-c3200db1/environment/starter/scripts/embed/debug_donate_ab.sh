#!/bin/bash
# A/B/C for the stage-3 grad NaN: vary ONLY donate_argnums, on one box, back to back.
#
# Why: with weights pinned (LR=1e-12 => ~1e-7 drift over 20 steps) and a deterministic dataloader,
# stage 3 NaN'd 7/20 with the default donate_argnums=(2,3) but ran 0/20 clean with donation off.
# Those two ran on different boxes, so this re-runs all arms on ONE box to kill that confound and
# to test whether donating opt_state ALONE is safe (which keeps ~half the HBM saving).
#
#   both -> (2,3)  the current default; expected ~7/20 NaN
#   opt  -> (3,)   opt_state only; weights buffer untouched
#   none -> ()     April's behaviour; expected 0/20
#
# If opt==none==clean and both==NaN, the fix is to stop donating the WEIGHTS buffer.
#
#   RESUME_FROM=gs://.../qwen3_mem_embed/16000 bash scripts/embed/debug_donate_ab.sh
set -uo pipefail
cd "${REPO_DIR:-$HOME/memory-layers}"
set -a; . .env 2>/dev/null || . "$HOME/.env" 2>/dev/null || true; set +a

RESUME_FROM="${RESUME_FROM:?set RESUME_FROM=gs://.../qwen3_mem_embed/<step>}"
N="${STOP_AT:-20}"
LR="${LR:-1e-12}"                 # pin the weights: any NaN is the backward at W_ckpt, not drift
MODES="${MODES:-both opt none}"
STAGE="${STAGE:-staged_debug_stage3}"

export HF_HUB_OFFLINE=1 GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet}"

for mode in $MODES; do
  echo
  echo "############ MEM_DONATE=$mode  (stage=$STAGE lr=$LR ckpt=$(basename "$RESUME_FROM")) ############"
  MEM_DONATE="$mode" uv run train.py \
      model=qwen3_mem_embed model.main_model.model_id="Qwen/Qwen3-4B" model.memory.mem_top_k=64 \
      dataset=qa_hard_neg_think_sft4b \
      trainer=$STAGE trainer.steps="$N" trainer.resume_from="$RESUME_FROM" \
      trainer.learning_rate=$LR \
      trainer.checkpoint_interval=1000000 trainer.log_interval=1 trainer.eval_interval=1000000000 \
      'eval_set@trainer.evals=none' +trainer.run_name="dbgdonate_${mode}" 2>&1 \
    | grep -viE "^WARNING:absl|Grain multiprocess|^(embed_model|main_model)\." \
    | grep -aE "not finite|/${N} \[|Traceback|Error" | tail -6
  echo "#### MEM_DONATE=$mode done ####"
done
