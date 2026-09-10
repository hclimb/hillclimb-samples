#!/bin/bash
# Runs ON a box, unattended. MEMORY-LAYER accuracy vs CORPUS SIZE on MuSiQue — the matching half
# of scripts/embed/sweep_rag_corpus.sh.
#
#   RUN_DIR=<run-dir> STEP=<n> SIZES=512_2048_8192_11656 \
#     bash scripts/embed/sweep_mem_corpus.sh
#
# THE POINT. RAG retrieves top-k WHOLE DOCUMENTS by one pooled embedding, so its recall decays as
# the haystack grows (already only recall@5 0.609 at 512 docs). The memory layer scores every
# token slot in the bank and takes top-k over all of them, so its retrieval should decay more
# slowly. If a crossover exists this sweep and the RAG one bracket it.
#
# Both sweeps MUST use the same sizes and the same 128 queries or the comparison says nothing.
# The memory eval builds its corpus internally via inject_query_gold (scan the full corpus, keep
# every gold, fill to target_docs with distractors); build_musique_c512_rag_corpus.py reproduces
# exactly that selection for RAG. So `target_docs=N` here and `--target-docs N` there describe the
# same haystack.
set -uo pipefail
set -a; . "$HOME/.env"; set +a
export PATH="$HOME/.local/bin:$PATH"
cd "${REPO_DIR:-$HOME/memory-layers}"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL:?}/adc.json"
export GCLOUD_PROJECT="${GCS_BUCKET_PROJECT}"

BUCKET="${CKPT_BUCKET:-memory-layers-training-usc1}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
STEP="${STEP:?set STEP (the checkpoint to sweep)}"
SIZES="${SIZES:-512_2048_8192_11656}"; SIZES="${SIZES//_/ }"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1280}"
TP=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]*$'); [ "$TP" -gt 0 ] 2>/dev/null || TP=4
CKPT="gs://$BUCKET/$RUN_DIR/qwen3_mem_embed/$STEP"

echo "[mem-sweep] ckpt=$CKPT sizes=[$SIZES] n=$NUM_SAMPLES tp=$TP"
uv pip install --quiet fastapi==0.115.6 starlette==0.41.3 prometheus-fastapi-instrumentator==7.0.0

for n in $SIZES; do
  name="musique_c${n}"
  dst="gs://$BUCKET/$RUN_DIR/eval/step_${STEP}/${name}.json"
  if gsutil -q stat "$dst" 2>/dev/null; then echo "[mem-sweep] c$n done — skipping"; continue; fi
  echo "[mem-sweep] ===== corpus $n docs ====="
  pkill -f "[v]llm serve" 2>/dev/null || true
  sudo fuser -k -9 /dev/vfio/[0-9]* 2>/dev/null || true
  sleep 3

  # max_docs stays null so inject_query_gold scans the FULL corpus for golds; target_docs sets the
  # final size. Setting max_docs=n instead would truncate the scan and lose golds — the same trap
  # the RAG corpus builder exists to avoid.
  uv run --no-sync eval.py \
      checkpoint_dir="$CKPT" \
      '~eval_set@evals=pretraining' \
      "+eval/tasks@evals.musique=gen_large_mem_musique_c512" \
      "evals.musique.eval.num_samples=$NUM_SAMPLES" \
      "evals.musique.eval.max_new_tokens=$MAX_NEW_TOKENS" \
      "evals.musique.eval.doc_dataset.target_docs=$n" \
      "evals.musique.eval.doc_dataset.max_docs=null" \
      "+evals.musique.eval.metrics.llm_judge_accuracy.model_id=Qwen/Qwen3-4B" \
      "+evals.musique.eval.metrics.llm_judge_accuracy.tensor_parallel_size=$TP" \
    && CKPT="$CKPT" EVAL_TASK="$name" uv run --no-sync python scripts/misc/log_eval_to_wandb.py \
    || echo "[mem-sweep] c$n FAILED — continuing"
done
echo "[mem-sweep] DONE ckpt=$CKPT sizes=[$SIZES]"
