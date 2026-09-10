#!/bin/bash
# Overnight MSA sweep driver — launch this SAME script on BOTH slice workers.
# Per dataset, in order: (A) hybrid autoK arm on both hosts (JAX self-syncs the two
# processes), then (B) RAG@10 + viewer-JSON on RAG_HOST only while the other host polls GCS
# for the artifact so both hosts enter the next dataset's phase A together. Every artifact
# is GCS-idempotent, so the driver can be killed and relaunched at any point and it resumes
# where it left off. msmarco_v1 is excluded (done 2026-07-21/22).
#
#   RUN_DIR=<run-dir> STEP=100000 bash scripts/embed/msa_sweep_overnight.sh
set -uo pipefail
set -a; . "$HOME/.env"; set +a
export PATH="$HOME/.local/bin:$PATH"
cd "${REPO_DIR:-$HOME/memory-layers}"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL:?}/adc.json"
export GCLOUD_PROJECT="${GCS_BUCKET_PROJECT}"

export CKPT_BUCKET="${CKPT_BUCKET:-memory-layers-training}"
export RUN_DIR="${RUN_DIR:?set RUN_DIR}"
export STEP="${STEP:-100000}"
BUCKET="$CKPT_BUCKET"
DATASETS="${DATASETS:-hotpotqa 2wikimultihopqa natural_questions popqa triviaqa_10m dureader narrativeqa musique}"
RAG_HOST="${RAG_HOST:-tpu-v6e-slice-mig-1z9d}"
RAG_WAIT_MIN="${RAG_WAIT_MIN:-90}"
HOST="$(hostname)"

gexists() { gsutil -q stat "$1" 2>/dev/null; }

for ds in $DATASETS; do
  hyb="gs://$BUCKET/$RUN_DIR/eval/step_${STEP}/msa_${ds}_c10000_hybrid_autok.json"
  ragsam="gs://$BUCKET/$RUN_DIR/eval/step_${STEP}/msa_${ds}_c10000_rag_top10_samples.json"
  echo "[sweep][$HOST] ===== $ds ====="

  # Phase A — hybrid (both hosts; must run the same eval.py together on a slice).
  if gexists "$hyb"; then
    echo "[sweep][$HOST] $ds hybrid already done"
  else
    DS="$ds" bash scripts/embed/eval_msa_hybrid.sh \
      || echo "[sweep][$HOST] $ds hybrid FAILED — continuing"
  fi

  # The hybrid artifact is uploaded by the JAX rank-0 host AFTER its judge metrics; the
  # other host's eval.py exits right after generation. Wait for the artifact so the
  # non-rank-0 host doesn't race ahead into the next dataset's JAX init (300s init
  # timeout would cascade-fail every remaining dataset).
  if ! gexists "$hyb"; then
    echo "[sweep][$HOST] waiting for $ds hybrid artifact (max 45m)..."
    for _ in $(seq 45); do gexists "$hyb" && break; sleep 60; done
  fi

  # Phase B — RAG baseline + per-sample JSON (RAG_HOST only; peer polls to stay in step).
  if gexists "$hyb"; then
    if [ "$HOST" = "$RAG_HOST" ]; then
      if gexists "$ragsam"; then
        echo "[sweep][$HOST] $ds rag already done"
      else
        DS="$ds" bash scripts/embed/eval_msa_rag.sh \
          || echo "[sweep][$HOST] $ds rag FAILED — continuing"
      fi
    else
      echo "[sweep][$HOST] waiting for $ds rag artifact (max ${RAG_WAIT_MIN}m)..."
      for _ in $(seq "$RAG_WAIT_MIN"); do gexists "$ragsam" && break; sleep 60; done
    fi
  else
    echo "[sweep][$HOST] $ds hybrid artifact missing — skipping RAG"
  fi
done
echo "[sweep][$HOST] ALL DONE"
