#!/bin/bash
# Multihop hard-negative mining: Stage 1 (embed corpus) -> Stage 2 (mine negative IDs).
# Designed to run UNATTENDED on a TPU box. Both stages are resumable (Stage 1 skips
# existing embedding shards, Stage 2 skips existing id shards), so a rerun after any
# failure continues rather than restarting.
#
# Results are pushed to HF at the end so they survive the box being lost.
#
# Env overrides:
#   MAX_DOCS / MAX_ROWS   cap work (benchmark / smoke test)
#   PUSH_HF=0             skip the upload
#   HN_LOG                log path (default ~/multihop_hardneg.log)
set -uo pipefail

HN_LOG="${HN_LOG:-$HOME/multihop_hardneg.log}"
EMB_DIR="${EMB_DIR:-data/cache/multihop_doc_embeddings}"
IDS_DIR="${IDS_DIR:-data/cache/multihop_hard_neg_ids}"
PUSH_HF="${PUSH_HF:-1}"

{
  echo "######## multihop hard-neg pipeline | host $(hostname) | $(date -u) ########"

  # Both stages are resumable (completed shards are skipped), so retrying after a
  # transient failure continues rather than restarting. Bounded so a hard bug cannot
  # crash-loop all night.
  ATTEMPTS="${ATTEMPTS:-3}"

  echo "######## STAGE 1: embed corpus ########"
  s1=1
  for a in $(seq 1 "$ATTEMPTS"); do
    echo "--- stage 1 attempt $a/$ATTEMPTS ($(date -u)) ---"
    uv run python datagen/multihop_embed_corpus.py \
        --out-dir "$EMB_DIR" \
        ${MAX_DOCS:+--max-docs $MAX_DOCS}
    s1=$?
    [ $s1 -eq 0 ] && break
    echo "--- stage 1 failed rc=$s1; retrying in 60s ---"
    sleep 60
  done
  echo "STAGE1_EXIT=$s1"
  if [ $s1 -ne 0 ]; then
    echo "######## stage 1 failed (rc=$s1); not starting stage 2 ########"
    echo "######## DONE $(date -u) ########"
    exit $s1
  fi

  echo "######## STAGE 2: mine hard-negative IDs ########"
  s2=1
  for a in $(seq 1 "$ATTEMPTS"); do
    echo "--- stage 2 attempt $a/$ATTEMPTS ($(date -u)) ---"
    uv run python datagen/multihop_mine_hard_neg_ids.py \
        --emb-dir "$EMB_DIR" \
        --out-dir "$IDS_DIR" \
        ${MAX_ROWS:+--max-rows $MAX_ROWS}
    s2=$?
    [ $s2 -eq 0 ] && break
    echo "--- stage 2 failed rc=$s2; retrying in 60s ---"
    sleep 60
  done
  echo "STAGE2_EXIT=$s2"
  if [ $s2 -ne 0 ]; then
    echo "######## stage 2 failed (rc=$s2) ########"
    echo "######## DONE $(date -u) ########"
    exit $s2
  fi

  if [ "$PUSH_HF" = "1" ]; then
    echo "######## PUSH: upload id shards to HF ########"
    uv run python datagen/multihop_push_hard_neg_ids.py --ids-dir "$IDS_DIR"
    echo "PUSH_EXIT=$?"
  else
    echo "######## PUSH skipped (PUSH_HF=0) ########"
  fi

  echo "######## DONE $(date -u) ########"
} 2>&1 | tee "$HN_LOG"
