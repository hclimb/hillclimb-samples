#!/bin/bash
# Runs ON a box. Classic-RAG baseline vs the memory model on the SAME MuSiQue 512-doc corpus.
#
# rag_eval.py runs BOTH in one invocation: the memory-model eval worker first, then the RAG
# pipeline (retrieve -> generate -> judge), so the two numbers come from one process against one
# corpus and one query set. That is the whole point — a baseline measured on a different haystack
# is not a baseline.
#
#   CKPT=<gs://…/qwen3_mem_embed/<step>> bash scripts/embed/rag_baseline_musique_c512.sh
#
# ⚠️ THE CORPUS MUST BE THE MATCHED ONE. `single_embedding_retrieval.py --max_docs N` simply stops
# reading after N docs, whereas the memory eval's `inject_query_gold: true` scans the FULL corpus,
# keeps every gold doc for the eval queries, then fills to target_docs with distractors
# (evals/gen_large_mem.py:377-416). Pointing RAG at the raw corpus with --max_docs 512 hands it
# 512 mostly-irrelevant documents and it scores ~0 — the memory model would "win" purely because
# it was given the answers and RAG was not. datagen/musique/build_musique_c512_rag_corpus.py
# reproduces the eval's selection (verified: 512 docs = 293 gold + 219 distractor, all golds
# present) and both systems read it.
set -uo pipefail
set -a; . "$HOME/.env"; set +a
export PATH="$HOME/.local/bin:$PATH"
cd "${REPO_DIR:-$HOME/memory-layers}"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL:?GCS_USER_EMAIL not set in ~/.env}/adc.json"
export GCLOUD_PROJECT="${GCS_BUCKET_PROJECT}"

CKPT="${CKPT:?set CKPT to a gs://…/qwen3_mem_embed/<step> path}"
RAG_CORPUS="${RAG_CORPUS:-${HF_USERNAME}/musique-c512-rag-corpus}"
QUERY_DS="${QUERY_DS:-${HF_USERNAME}/msa-musique-qa-with-ids}"
NUM_QUERIES="${NUM_QUERIES:-128}"          # must match the memory eval's eval.num_samples
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1280}"   # memory half; see eval_musique_sft_midtrain.sh
TP="${TP:-$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]*$')}"; TP="${TP:-4}"

# Fairness knobs, both deliberate:
#   max_doc_length=256 — the memory eval indexes ONE 256-token chunk per doc
#                        (max_chunks_per_doc: 1, chunk_size: 256). The rag_eval.yaml default of
#                        512 would let RAG read twice as much of each document.
#   embedding_model    — Qwen3-Embedding-0.6B, the same tower the memory model embeds docs with,
#                        so the retrieval comparison is about the mechanism, not the encoder.
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-256}"

echo "[rag] ckpt=$CKPT"
echo "[rag] corpus=$RAG_CORPUS  queries=$QUERY_DS n=$NUM_QUERIES  tp=$TP  max_doc_len=$MAX_DOC_LENGTH"

echo "[rag] pinning vllm HTTP server deps..."
uv pip install --quiet fastapi==0.115.6 starlette==0.41.3 prometheus-fastapi-instrumentator==7.0.0
pkill -f "[v]llm serve" 2>/dev/null || true
sudo fuser -k -9 /dev/vfio/[0-9]* 2>/dev/null || true
sleep 3

# Hydra note: doc_dataset/query_dataset/query_column/query_gt_column are NOT in
# configs/rag_eval.yaml's rag: block, so they need the "+" append form. The rest (num_queries,
# max_docs, max_doc_length, gen_*, judge_*) DO exist there and must NOT have it.
#
# NOTE rag_eval.py wraps the RAG stage in a bare `except Exception: print(...)` and continues, so
# a RAG failure exits 0 with memory-only metrics. Grep the output for "RAG pipeline failed" —
# absence of rag_accuracy in the final block means it did NOT run, not that it scored zero.
uv run --no-sync rag_eval.py \
    checkpoint_dir="$CKPT" \
    '~eval_set@evals=pretraining' \
    "+eval/tasks@evals.musique=gen_large_mem_musique_c512" \
    "evals.musique.eval.num_samples=$NUM_QUERIES" \
    "evals.musique.eval.max_new_tokens=$MAX_NEW_TOKENS" \
    "+evals.musique.eval.metrics.llm_judge_accuracy.model_id=Qwen/Qwen3-4B" \
    "+evals.musique.eval.metrics.llm_judge_accuracy.tensor_parallel_size=$TP" \
    "+rag.doc_dataset=$RAG_CORPUS" \
    "+rag.query_dataset=$QUERY_DS" \
    "+rag.query_column=question" \
    "+rag.query_gt_column=answer" \
    "rag.num_queries=$NUM_QUERIES" \
    "rag.max_docs=512" \
    "rag.max_doc_length=$MAX_DOC_LENGTH" \
    "rag.gen_model=Qwen/Qwen3-4B" \
    "rag.gen_tensor_parallel_size=$TP" \
    "rag.judge_model=Qwen/Qwen3-4B" \
    "rag.judge_tensor_parallel_size=$TP"
