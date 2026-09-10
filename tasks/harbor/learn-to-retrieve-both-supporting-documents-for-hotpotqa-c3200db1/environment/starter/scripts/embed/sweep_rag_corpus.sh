#!/bin/bash
# Runs ON a box, unattended. RAG accuracy vs CORPUS SIZE on MuSiQue.
#
#   SIZES=512_2048_8192_11656 bash scripts/embed/sweep_rag_corpus.sh
#
# This is the accuracy half of the Pareto story. RAG retrieves top-k whole documents by a single
# pooled embedding, so its recall falls as the haystack grows (already only recall@5 0.609 at 512
# docs); the memory model retrieves token-level slots over the whole bank. If there is a crossover
# it should appear here, and BOTH systems must be swept over the SAME sizes for the comparison to
# mean anything — scripts/embed/sweep_mem_corpus.sh is the matching half.
#
# For each size it (1) builds a matched corpus containing every gold document for the eval queries
# plus distractors up to the target, exactly as gen_large_mem's inject_query_gold does, then
# (2) runs retrieve -> generate -> judge against it. Without (1) RAG gets a haystack that mostly
# lacks the answers and scores ~0 — see datagen/musique/build_musique_c512_rag_corpus.py.
set -uo pipefail
set -a; . "$HOME/.env"; set +a
export PATH="$HOME/.local/bin:$PATH"
cd "${REPO_DIR:-$HOME/memory-layers}"

SIZES="${SIZES:-512_2048_8192_11656}"; SIZES="${SIZES//_/ }"
NUM_QUERIES="${NUM_QUERIES:-128}"
GEN_TOP_K_DOCS="${GEN_TOP_K_DOCS:-5}"
OUT_ROOT="${OUT_ROOT:-$HOME/rag_corpus_sweep}"
mkdir -p "$OUT_ROOT"

echo "[rag-sweep] sizes=[$SIZES] n=$NUM_QUERIES docs_in_prompt=$GEN_TOP_K_DOCS"
uv pip install --quiet fastapi==0.115.6 starlette==0.41.3 prometheus-fastapi-instrumentator==7.0.0

for n in $SIZES; do
  repo="${HF_USERNAME}/musique-c${n}-rag-corpus"
  res="$OUT_ROOT/rag_c${n}.json"
  if [ -f "$res" ]; then echo "[rag-sweep] c$n already done — skipping"; continue; fi

  echo "[rag-sweep] ===== corpus $n docs ====="
  # Idempotent: build_… pushes to the Hub, and a re-run just overwrites with identical content.
  uv run --no-sync python datagen/musique/build_musique_c512_rag_corpus.py \
      --num-queries "$NUM_QUERIES" --target-docs "$n" --out-repo "$repo" 2>&1 | tail -3

  pkill -f "[v]llm serve" 2>/dev/null || true
  sudo fuser -k -9 /dev/vfio/[0-9]* 2>/dev/null || true
  sleep 3

  RAG_CORPUS="$repo" NUM_QUERIES="$NUM_QUERIES" MAX_DOCS="$n" \
  GEN_TOP_K_DOCS="$GEN_TOP_K_DOCS" OUT_DIR="$OUT_ROOT/c$n" \
    uv run --no-sync python scripts/misc/rag_only.py 2>&1 | tail -8

  # Keep a flat per-size summary so the plot script does not have to walk hydra dirs.
  cp "$OUT_ROOT/c$n/rag_only_summary.json" "$res" 2>/dev/null \
    && echo "[rag-sweep] c$n -> $res" \
    || echo "[rag-sweep] c$n produced no summary"
done

echo "[rag-sweep] ===== SUMMARY ====="
for n in $SIZES; do
  [ -f "$OUT_ROOT/rag_c${n}.json" ] && \
    python3 -c "import json;d=json.load(open('$OUT_ROOT/rag_c${n}.json'))['metrics'];print(f\"  c$n: acc={d['rag_accuracy']:.4f} recall@5={d.get('rag_recall@5','-')} mrr={d.get('rag_mrr','-')}\")"
done
gsutil -q cp -r "$OUT_ROOT" "gs://${CKPT_BUCKET:-memory-layers-training-usc1}/pareto/rag_corpus_sweep/" 2>/dev/null \
  && echo "[rag-sweep] results -> gs://${CKPT_BUCKET:-memory-layers-training-usc1}/pareto/rag_corpus_sweep/"
echo "[rag-sweep] DONE"
