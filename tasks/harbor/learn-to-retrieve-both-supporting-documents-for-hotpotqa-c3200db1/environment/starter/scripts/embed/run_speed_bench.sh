#!/bin/bash
# Inference-speed benchmark: qwen3_mem_embed vs MSA-4B on MS MARCO v1.
# Same corpus cap (max_docs=2000 — both fit one v6e-8 chip; MSA OOMs at 4000) and
# same query budget. Each evaluator, when MEMBENCH is set, times corpus-encode +
# per-batch prefill (single forward over doc memory) + decode/e2e, and records the
# doc-token budget. Writes bench_msa.json / bench_membed.json into $BENCH_DIR.
set -a; source "$(dirname "$0")/../../.env"; set +a
. "$HOME/.local/bin/env" 2>/dev/null || true
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL}/adc.json"
export GCLOUD_PROJECT="${GCS_BUCKET_PROJECT}"

BENCH_DIR="${BENCH_DIR:-$HOME/membench}"
export MEMBENCH="$BENCH_DIR"
mkdir -p "$BENCH_DIR"

MAXD=2000
NS=64          # 4 batches of 16 -> drop batch 0 (JIT warmup) in analysis
CKPT=gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-04-19-20-17-51/qwen3_mem_embed/100000

echo "================ BENCH MSA msmarco_v1 ================"
uv run --no-sync eval.py \
    model=qwen3_msa \
    '~eval_set@evals=pretraining' \
    "+eval/tasks@evals.msmarco_v1=gen_large_mem_msa_msmarco_v1" \
    "evals.msmarco_v1.eval.doc_dataset.max_docs=${MAXD}" \
    "evals.msmarco_v1.eval.num_samples=${NS}" \
    "~evals.msmarco_v1.eval.metrics" \
    tp_devices=1 use_wandb=false 2>&1 | tail -8 || echo "!!!! MSA FAILED"

echo "================ BENCH MEMBED msmarco_v1 ================"
uv run --no-sync eval.py \
    checkpoint_dir=${CKPT} \
    '~eval_set@evals=pretraining' \
    "+eval/tasks@evals.msmarco_v1=gen_large_mem_msa_msmarco_v1" \
    "evals.msmarco_v1.eval.type=generation_large_mem" \
    "evals.msmarco_v1.eval.doc_access_acc=false" \
    "evals.msmarco_v1.eval.doc_dataset.max_docs=${MAXD}" \
    "evals.msmarco_v1.eval.num_samples=${NS}" \
    "~evals.msmarco_v1.eval.metrics" \
    tp_devices=1 use_wandb=false 2>&1 | tail -8 || echo "!!!! MEMBED FAILED"

echo "===== BENCH FILES ====="
cat "$BENCH_DIR/bench_msa.json" 2>/dev/null
cat "$BENCH_DIR/bench_membed.json" 2>/dev/null
echo "SPEED_BENCH_DONE"
