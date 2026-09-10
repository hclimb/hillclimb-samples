#!/bin/bash
# Runs ON a box, unattended. RAG->memory HYBRID accuracy on MS-MARCO (msa_msmarco_v1):
# retrieve top-k docs per query (same encoder + settings as the RAG baseline), restrict the
# memory bank to those docs' slots, generate, judge. See evals/gen_large_mem_rag_hybrid.py for
# the design and scripts/embed/eval_musique_hybrid.sh for the MuSiQue original this mirrors.
#
#   RUN_DIR=<run-dir> STEP=<n> [SIZES=10000] [RAG_KS=50] [NAME_SUFFIX=_smoke] \
#     bash scripts/embed/eval_msmarco_hybrid.sh
#   ORACLE=1 [ORACLE_FILL_TO=50] RUN_DIR=… STEP=…  # perfect-retrieval ceiling arm
#
# Defaults differ from the MuSiQue runner where MS-MARCO differs:
#   SIZES=10000           — the target corpus (2.56M slots; ~75.6k-doc source corpus)
#   MAX_NEW_TOKENS=512    — no EOS early-stop in decode, so every query pays the full budget;
#                           MS-MARCO answers are short (1280 was for MuSiQue think-chains)
#   NAME_SUFFIX           — appended to the result name; use for smokes so the GCS
#                           idempotency check doesn't make the real arm skip later
#
# MULTI-HOST SLICE (2 x ct6e-standard-4t): launch this SAME script on BOTH workers via
# multi-tpu-box-run.sh (runbook §2.3) — a single-host launch hangs. Only the JAX process-0
# host writes the results JSON, so the vLLM judge and the wandb/GCS upload below self-gate
# to that host via the file-existence checks.
set -uo pipefail
set -a; . "$HOME/.env"; set +a
export PATH="$HOME/.local/bin:$PATH"
cd "${REPO_DIR:-$HOME/memory-layers}"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL:?}/adc.json"
export GCLOUD_PROJECT="${GCS_BUCKET_PROJECT}"

# Approx bank retrieval — ALWAYS preferred per policy (wiki/architecture/retrieval-modes.md):
# much faster, and its ~99% recall shifts sit well under the judge noise floor. Explicit here
# so the choice survives checkpoints whose cfg says otherwise. The evaluator's
# row_divergence_rate metric stays as the within-run determinism monitor.
export MEM_APPROX_TOPK=1

# On a multi-host slice the vLLM judge must treat this host as a standalone 4-chip box
# (evals/vllm.py reads this; harmless on single-host boxes where it just isn't needed).
export VLLM_TPU_LOCAL_ONLY=1

# Default to the EU bucket: the v6e slice is in europe-west4 and the hard-neg base checkpoint
# (qa_hard_neg_think_…-2026-07-17-02-32-09 @100000) has a slice-local copy there (original in
# -usc1, same run-dir path).
BUCKET="${CKPT_BUCKET:-memory-layers-training}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
STEP="${STEP:?set STEP (the checkpoint to eval)}"
SIZES="${SIZES:-10000}"; SIZES="${SIZES//_/ }"
RAG_KS="${RAG_KS:-50}"; RAG_KS="${RAG_KS//_/ }"
ORACLE="${ORACLE:-0}"
ORACLE_FILL_TO="${ORACLE_FILL_TO:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
NAME_SUFFIX="${NAME_SUFFIX:-}"
# Extra Hydra overrides, space-separated (e.g. "evals.msmarco_hybrid.dataset.batch_size=1"
# to force the legacy tiled mode for an equivalence check). Word-split on purpose.
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
TP=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]*$'); [ "$TP" -gt 0 ] 2>/dev/null || TP=4
CKPT="gs://$BUCKET/$RUN_DIR/qwen3_mem_embed/$STEP"

echo "[msmarco-hybrid] ckpt=$CKPT sizes=[$SIZES] ks=[$RAG_KS] oracle=$ORACLE n=$NUM_SAMPLES tp=$TP max_new=$MAX_NEW_TOKENS suffix='$NAME_SUFFIX'"
uv pip install --quiet fastapi==0.115.6 starlette==0.41.3 prometheus-fastapi-instrumentator==7.0.0

run_arm() {
  local n="$1" name="$2"; shift 2
  local dst="gs://$BUCKET/$RUN_DIR/eval/step_${STEP}/${name}.json"
  if gsutil -q stat "$dst" 2>/dev/null; then echo "[msmarco-hybrid] $name done — skipping"; return 0; fi
  echo "[msmarco-hybrid] ===== $name ====="
  pkill -f "[v]llm serve" 2>/dev/null || true
  sudo fuser -k -9 /dev/vfio/[0-9]* 2>/dev/null || true
  sleep 3

  # max_docs stays null so the gold scan sees the FULL corpus; target_docs sets the size —
  # same trap-avoidance as sweep_mem_corpus.sh.
  uv run --no-sync eval.py \
      checkpoint_dir="$CKPT" \
      '~eval_set@evals=pretraining' \
      "+eval/tasks@evals.msmarco_hybrid=gen_large_mem_msmarco_hybrid" \
      "evals.msmarco_hybrid.eval.num_samples=$NUM_SAMPLES" \
      "evals.msmarco_hybrid.eval.max_new_tokens=$MAX_NEW_TOKENS" \
      "evals.msmarco_hybrid.eval.doc_dataset.target_docs=$n" \
      "evals.msmarco_hybrid.eval.output_file=outputs/${name}.json" \
      "+evals.msmarco_hybrid.eval.metrics.llm_judge_accuracy.model_id=Qwen/Qwen3-4B" \
      "+evals.msmarco_hybrid.eval.metrics.llm_judge_accuracy.tensor_parallel_size=$TP" \
      $EXTRA_OVERRIDES \
      "$@" \
    || { echo "[msmarco-hybrid] $name FAILED — continuing"; return 1; }

  # Upload + wandb only where the results JSON exists (JAX process 0's host on a slice).
  if find outputs -path "*eval_results*" -name "${name}.json" 2>/dev/null | grep -q .; then
    CKPT="$CKPT" EVAL_TASK="$name" \
    RESULTS_GLOB="outputs/**/eval_results/**/${name}.json" \
      uv run --no-sync python scripts/misc/log_eval_to_wandb.py \
      || echo "[msmarco-hybrid] $name: log_eval_to_wandb failed"
  else
    echo "[msmarco-hybrid] $name: no local results JSON (secondary slice host) — skipping upload"
  fi
}

for n in $SIZES; do
  if [ "$ORACLE" = "1" ]; then
    name="msmarco_c${n}_hybrid_oracle"
    [ "$ORACLE_FILL_TO" != "0" ] && name="${name}_fill${ORACLE_FILL_TO}"
    run_arm "$n" "${name}${NAME_SUFFIX}" \
      "evals.msmarco_hybrid.eval.rag.oracle=true" \
      "evals.msmarco_hybrid.eval.rag.oracle_fill_to=$ORACLE_FILL_TO"
  else
    for k in $RAG_KS; do
      run_arm "$n" "msmarco_c${n}_hybrid_k${k}${NAME_SUFFIX}" \
        "evals.msmarco_hybrid.eval.rag.top_k=$k"
    done
  fi
done

echo "[msmarco-hybrid] DONE"
