#!/bin/bash
# Runs ON a box. Sweeps a fixed set of RAG top_k values (auto-K disabled) through
# eval_msa_hybrid.sh, unmodified, one full invocation per K -- records llm_judge_accuracy /
# llm_judge_score / lexical_grounding AND end-to-end wall-clock latency per K, i.e. the RAG@K
# accuracy/latency tradeoff curve for one checkpoint.
#
# Caveat baked into every latency number: each K re-runs eval.py from scratch, so every
# invocation repeats the (K-independent) corpus-encoding + vLLM cold-start cost alongside the
# (K-dependent) longer-context generation cost. Absolute latencies are inflated by that shared
# setup cost; the relative trend across K still reflects the true K-dependent cost.
#
#   DS=<dataset> RUN_DIR=<run-dir> STEP=<n> [KS="5,10,25,50,100,150,200"] [NUM_SAMPLES=128] \
#     bash scripts/embed/sweep_topk_hybrid.sh
set -uo pipefail
set -a; . "$HOME/.env"; set +a
export PATH="$HOME/.local/bin:$PATH"
cd "${REPO_DIR:-$HOME/memory-layers}"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL:?}/adc.json"
export GCLOUD_PROJECT="${GCS_BUCKET_PROJECT}"

BUCKET="${CKPT_BUCKET:-memory-layers-training}"
DS="${DS:?set DS}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
STEP="${STEP:?set STEP}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
KS="${KS:-5,10,25,50,100,150,200}"

SUMMARY="/tmp/topk_sweep_summary.jsonl"
: > "$SUMMARY"

for K in ${KS//,/ }; do
  echo "[topk-sweep] ===== K=$K ====="
  START=$(date +%s)
  DS="$DS" RUN_DIR="$RUN_DIR" STEP="$STEP" NUM_SAMPLES="$NUM_SAMPLES" \
  NAME_SUFFIX="_k${K}" \
  EXTRA_OVERRIDES="evals.msa_hybrid.eval.rag.top_k=$K,evals.msa_hybrid.eval.rag.auto_k_threshold=null" \
    bash scripts/embed/eval_msa_hybrid.sh
  RC=$?
  END=$(date +%s)
  LATENCY=$((END - START))

  if [ $RC -ne 0 ]; then
    echo "[topk-sweep] K=$K FAILED rc=$RC after ${LATENCY}s"
    echo "{\"k\": $K, \"status\": \"failed\", \"latency_seconds\": $LATENCY}" | tee -a "$SUMMARY"
    continue
  fi

  # eval_msa_hybrid.sh's own naming: msa_${DS}_c10000_hybrid_autok${NAME_SUFFIX}
  name="msa_${DS}_c10000_hybrid_autok_k${K}"
  dst="gs://$BUCKET/$RUN_DIR/eval/step_${STEP}/${name}.json"
  local_copy="/tmp/${name}.json"
  gsutil cp "$dst" "$local_copy" 2>/dev/null

  if [ -f "$local_copy" ]; then
    python3 -c "
import json
d = json.load(open('$local_copy'))
m = d.get('metrics', {})
row = {
    'k': $K,
    'latency_seconds': $LATENCY,
    'llm_judge_accuracy': m.get('llm_judge_accuracy'),
    'llm_judge_score': m.get('llm_judge_score'),
    'lexical_grounding': m.get('lexical_grounding'),
    'rag_top_k_actual': m.get('rag_top_k'),
    'rag_all_golds_at_this_k': m.get(f'rag_all_golds@{$K}'),
    'rag_any_gold_at_this_k': m.get(f'rag_any_gold@{$K}'),
    'mean_active_bank_slots': m.get('mean_active_bank_slots'),
}
print(json.dumps(row))
" | tee -a "$SUMMARY"
  else
    echo "[topk-sweep] K=$K: no result JSON at $dst after ${LATENCY}s"
    echo "{\"k\": $K, \"status\": \"no_result_json\", \"latency_seconds\": $LATENCY}" | tee -a "$SUMMARY"
  fi
done

echo "[topk-sweep] FULL SUMMARY:"
cat "$SUMMARY"
gsutil cp "$SUMMARY" "gs://$BUCKET/$RUN_DIR/eval/step_${STEP}/topk_sweep_summary.jsonl"
echo "[topk-sweep] DONE"
