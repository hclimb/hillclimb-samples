#!/bin/bash
# Bottleneck #1 microbench: REAL dataloader throughput, live-HF vs offline-parquet.
# No model/TPU — pure CPU data pipeline. Tees to $DATA_LOG. Launch from a worktree with:
#   TPU_NAME=rohun-v6e-8-0 RUN_SCRIPT_PATH=scripts/embed/bench_data_throughput.sh \
#     bash scripts/infrastructure/multi-vm-tpu-run.sh
DATA_LOG="${DATA_LOG:-$HOME/bench_data_throughput.log}"
BATCHES="${BATCHES:-64}"
MAXSEC="${MAXSEC:-240}"
set -a; . "$HOME/.env" 2>/dev/null || . ".env" 2>/dev/null || true; set +a   # HF_TOKEN for streaming
{
  echo "############ host $(hostname)  $(date -u) ############"

  echo "==== ARM 1: LIVE-HF STREAMING (what train_hard_neg_think.sh does today) ===="
  ( unset HF_HUB_OFFLINE; uv run python scripts/embed/bench_data_throughput.py --batches "$BATCHES" --max-seconds "$MAXSEC" ) \
    || echo "!!!! LIVE-HF ARM FAILED"

  echo
  echo "==== ARM 2: OFFLINE-PARQUET (the fix — only if precached) ===="
  PARQ="${GROUND_HF_PARQUET:-$HOME/hf_parquet}"
  NEED="vm2825__science-qa-hard-neg-think vm2825__diverseqa-hard-neg-think vm2825__triviaqa-hotpotqa-nq-squad-msmarco-hard-neg-sft4b ragrawal36__multihop_qa_sft"
  MISSING=0
  for r in $NEED; do [ -d "$PARQ/$r" ] || { echo "MISSING: $PARQ/$r"; MISSING=1; }; done
  if [ "$MISSING" = "0" ]; then
    HF_HUB_OFFLINE=1 GROUND_HF_PARQUET="$PARQ" uv run python scripts/embed/bench_data_throughput.py --batches "$BATCHES" --max-seconds "$MAXSEC" \
      || echo "!!!! OFFLINE ARM FAILED"
  else
    echo "==== ARM 2 SKIPPED: parquet not cached at $PARQ. To enable: bash scripts/misc/precache_hf.sh ===="
  fi

  echo "############ DONE  $(date -u) ############"
} 2>&1 | tee "$DATA_LOG"
