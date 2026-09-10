#!/usr/bin/env bash
# Relaunch a grounding run RESUMING from its latest checkpoint (full resume: model + optimizer +
# dataloader position), so a preemption/reboot does NOT restart from step 0. Run ON the box on a
# CLEAN TPU (e.g. right after a reboot). It launches in tmux session 'train'.
#
#   bash scripts/misc/resume_launch.sh <stage_script> <run_name>
#   e.g. bash scripts/misc/resume_launch.sh train_ground_s1.sh ground_s1_zeroinit_4layer
#
# It finds the newest run-dir (by timestamp) that has a checkpoint and passes RESUME_FROM = that
# run's qwen3_mem_embed DIRECTORY (not dir/step) — the trainer then restores the LATEST step +
# optimizer + dataloader state (trainer/trainer.py: full-resume path, resume_step is None). A new
# run-dir is created for the resumed run; the eval box resolves each step to the newest dir that
# has it, so the eval curve stays continuous across the resume.
#
# NOTE (untested in this offline-parquet config): fast-forward replays the deterministic stream to
# the saved per-worker position (~150-200k source items/worker at step 8k, grows with step) — can
# take minutes; and it assumes the stream is byte-for-byte deterministic across restarts (stable
# sorted parquet file order + fixed shuffle/interleave seeds). Verify the "Restored dataloader
# state" log line appears and the step continues (not "stream starts from 0").
set -uo pipefail
SCRIPT="$1"; RUN="$2"
# gcloud account for the gsutil calls below: inherit the GCS identity (.env's GCS_USER_EMAIL).
export CLOUDSDK_CORE_ACCOUNT="${CLOUDSDK_CORE_ACCOUNT:-${GCS_USER_EMAIL:-rohunagrawal@gmail.com}}"
BUCKET="${GCS_BUCKET:-memory-layers-training}"
DIR=$(gsutil ls "gs://$BUCKET/${RUN}-*/qwen3_mem_embed/*/" 2>/dev/null \
      | grep -oE "gs://$BUCKET/${RUN}-[0-9-]+/qwen3_mem_embed" | sort -u | tail -1)
if [ -n "$DIR" ]; then
  export RESUME_FROM="$DIR"
  echo "RESUMING $RUN from $DIR (latest checkpoint step + dataloader state)"
else
  echo "no existing checkpoint for $RUN -> fresh launch from step 0"
fi
cd "$HOME/memory-layers" && source scripts/infrastructure/setup_shell.sh && bash "scripts/embed/$SCRIPT"
