#!/bin/bash
# Single-stream decode-throughput benchmark on one v6e-8.
#
# A literal GLOBAL batch of 1 is not expressible for the memory-layer / RAG: qwen3.py's
# forward hardwires P('data', ...) out-shardings, which are only legal next to replicated
# weights when the data axis is > 1 (at data=1 JAX rejects P('data') vs P() as mismatched).
# So we use the minimum the data-parallel transformer supports: B = n_chips = 8, i.e.
# 1 SEQUENCE PER CHIP, and report PER-STREAM tok/s = aggregate / 8. Decode is latency-
# bound, so per-stream @ 1-seq/chip is the single-stream number. (MSA's separate decode
# path DOES run literal single-chip B=1 — see the msa_1chip run — as a cross-check.)
#
# Memory-bank sharding isolation (the question: does splitting the bank across the 8 chips
# help or hurt?): membed at B=8 with bank SHARDED (default) vs REPLICATED on every chip
# (MEM_REPLICATE_BANK=1, no cross-chip merge — how MSA/RAG run). Same batch/chips; only the
# bank layout differs.
set -a; source "$(dirname "$0")/../../.env" 2>/dev/null; set +a
. "$HOME/.local/bin/env" 2>/dev/null || true
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL}/adc.json"
export GCLOUD_PROJECT="${GCS_BUCKET_PROJECT}"

BENCH_ROOT="${BENCH_ROOT:-$HOME/membench_bs1}"
MAXD="${MAXD:-2000}"
export BENCH_REPS="${BENCH_REPS:-5}"
CKPT=gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-04-19-20-17-51/qwen3_mem_embed/100000
mkdir -p "$BENCH_ROOT"

run_membed () {   # $1=subdir  $2=BS  $3=num_samples  $4=replicate_bank(0|1)
  local sub="$1" bs="$2" ns="$3" repl="$4"
  local dir="$BENCH_ROOT/$sub"; mkdir -p "$dir"
  export MEMBENCH="$dir" MEM_REPLICATE_BANK="$repl"
  echo "======== MEMBED $sub  BS=$bs replicate_bank=$repl ========"
  uv run --no-sync eval.py \
      checkpoint_dir=${CKPT} \
      '~eval_set@evals=pretraining' \
      "+eval/tasks@evals.msmarco_v1=gen_large_mem_msa_msmarco_v1" \
      "evals.msmarco_v1.eval.type=generation_large_mem" \
      "evals.msmarco_v1.eval.doc_access_acc=false" \
      "evals.msmarco_v1.eval.doc_dataset.max_docs=${MAXD}" \
      "evals.msmarco_v1.eval.num_samples=${ns}" \
      "evals.msmarco_v1.dataset.batch_size=${bs}" \
      "~evals.msmarco_v1.eval.metrics" \
      tp_devices=1 use_wandb=false 2>&1 | tail -16 || echo "!!!! MEMBED $sub FAILED"
  unset MEM_REPLICATE_BANK
}

run_msa () {   # $1=subdir  $2=BS  $3=num_samples  $4=single_device(0|1)
  local sub="$1" bs="$2" ns="$3" single="$4"
  export MEMBENCH="$BENCH_ROOT/$sub"; mkdir -p "$MEMBENCH"
  [ "$single" = "1" ] && export SINGLE_DEVICE=1 || { unset SINGLE_DEVICE; export REPLICATE_WEIGHTS=1; }
  echo "######## MSA $sub  BS=$bs single_device=$single ########"
  uv run --no-sync eval.py \
      model=qwen3_msa \
      '~eval_set@evals=pretraining' \
      "+eval/tasks@evals.msmarco_v1=gen_large_mem_msa_msmarco_v1" \
      "evals.msmarco_v1.eval.doc_dataset.max_docs=${MAXD}" \
      "evals.msmarco_v1.eval.num_samples=${ns}" \
      "evals.msmarco_v1.dataset.batch_size=${bs}" \
      "~evals.msmarco_v1.eval.metrics" \
      tp_devices=1 use_wandb=false 2>&1 | tail -14 || echo "!!!! MSA $sub FAILED"
  unset SINGLE_DEVICE REPLICATE_WEIGHTS
}

# --- per-stream @ B=8 (1 seq/chip), the comparable single-stream regime ---
run_msa    msa            8 16 0
run_membed membed_sharded 8 16 0     # bank SHARDED across 8 chips (+ cross-chip merge)
run_membed membed_replicated 8 16 1  # bank REPLICATED on each chip (no merge) = no sharding

# --- literal single-chip B=1 cross-check (MSA's decode path supports it) ---
run_msa    msa_1chip      1 4  1

echo "===== BENCH FILES ====="
for f in msa/bench_msa.json msa_1chip/bench_msa.json \
         membed_sharded/bench_membed.json membed_replicated/bench_membed.json; do
  echo "--- $f ---"; cat "$BENCH_ROOT/$f" 2>/dev/null
done
echo "SPEED_BENCH_BS1_DONE"
