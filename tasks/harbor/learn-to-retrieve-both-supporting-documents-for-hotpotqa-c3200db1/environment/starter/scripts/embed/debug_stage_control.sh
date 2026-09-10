#!/bin/bash
# CONTROL for the stage-3 NaN: same checkpoint, same batches, ONLY trainable_params differs.
#   stage3 = [mem, embed, main]   (the failing config)
#   stage2 = [mem, embed]         (main frozen — the config that ran clean to 15000)
# trainable_params must not affect the gradient (it only builds optax's freeze mask, applied after
# value_and_grad). If stage2 is finite and stage3 NaNs on identical weights+data, that assumption
# is false and it is the bug.
#   RESUME_FROM=gs://.../qwen3_mem_embed/14000 bash scripts/embed/debug_stage_control.sh
set -uo pipefail
cd "${REPO_DIR:-$HOME/memory-layers}"
set -a; . .env 2>/dev/null || . "$HOME/.env" 2>/dev/null || true; set +a
RESUME_FROM="${RESUME_FROM:?}"; N="${STOP_AT:-20}"
# LR: pin the peak LR to remove the one confound in this control. Only step 0 has matched weights:
# after a clean step, stage3 has updated main_model and stage2 has not, so the arms drift apart and
# a late NaN could be drift, not trainable_params. LR=1e-12 (with the schedule's hardcoded 1e-8
# init, so <=1e-8 throughout) holds the weights at W_ckpt to ~1e-7 total drift over 20 steps => all
# 20 batches are evaluated at the SAME weights in BOTH arms.
#   stage3 NaN + stage2 clean under a pinned LR => trainable_params IS reaching the backward.
#   both clean                                  => the earlier 7/20 was drift after all.
LR="${LR:-}"
export HF_HUB_OFFLINE=1 GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet}"

# STAGES: which trainer configs to run, in order. Default = the stage2-vs-stage3 control.
# Override to bisect, e.g. STAGES="staged_debug_all staged_debug_stage3".
STAGES="${STAGES:-staged_debug_stage2 staged_debug_stage3}"

for stage in $STAGES; do
  echo; echo "############ TRAINER=$stage LR=${LR:-default} ############"
  uv run train.py \
      model=qwen3_mem_embed model.main_model.model_id="Qwen/Qwen3-4B" model.memory.mem_top_k=64 \
      dataset=qa_hard_neg_think_sft4b \
      trainer=$stage trainer.steps="$N" trainer.resume_from="$RESUME_FROM" \
      ${LR:+trainer.learning_rate=$LR} \
      trainer.checkpoint_interval=1000000 trainer.log_interval=1 trainer.eval_interval=1000000000 \
      'eval_set@trainer.evals=none' +trainer.run_name="dbgctl_${stage}" 2>&1 \
    | grep -viE "^WARNING:absl|Grain multiprocess|^(embed_model|main_model)\." \
    | grep -E "Trainable params|not finite|/${N} \[|Traceback|Error" | tail -8
done
