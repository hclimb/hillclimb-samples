#!/bin/bash
# Runs ON a box. Throughput axis of the Pareto plot: RAG-equivalent vs the memory-layer
# optimization ladder, all from one harness (scripts/embed/bench_pareto_throughput.py).
#
#   bash scripts/embed/sweep_pareto_throughput.sh
#
# Emits one JSON line per point to $OUT (default ~/pareto_throughput.jsonl) and copies it to GCS.
#
# WHAT IS BEING COMPARED
# Not vLLM-vs-JAX (that would measure the serving stack). Both sides are composed from the same
# measured primitives at the same shapes on the same chip; only what the architecture forces into
# the decode step differs:
#     RAG-equiv    KV length = question + k*doc_len   (k retrieved docs in context, EVERY token)
#     memory layer KV length = question               + a bank scan
#
# DOC LENGTH is the axis that matters, not corpus size: RAG carries k documents regardless of how
# big the haystack is, so its throughput is flat in corpus size — only its accuracy decays there.
# Its throughput cost scales with DOCUMENT LENGTH, which is why the sweep walks doc_len.
set -uo pipefail
set -a; . "$HOME/.env" 2>/dev/null; set +a
export PATH="$HOME/.local/bin:$PATH"
cd "${REPO_DIR:-$HOME/memory-layers}"
export JAX_PLATFORMS=tpu PYTHONPATH=.

OUT="${OUT:-$HOME/pareto_throughput.jsonl}"
: > "$OUT"
Q=64                      # question tokens in the prompt
K="${K:-5}"               # docs in prompt for RAG
DOC_LENS="${DOC_LENS:-256 1024 4096 16384}"; DOC_LENS="${DOC_LENS//_/ }"
BANKS="${BANKS:-131072 524288 2097152}";     BANKS="${BANKS//_/ }"
NML="${N_MEM_LAYERS:-4}"  # ground4layer has 4 memory layers; hard-neg has 1

run () {  # run <label> <env assignments...>
  local label="$1"; shift
  echo "[tp] $label"
  local line
  line=$(env "$@" uv run --no-sync python scripts/embed/bench_pareto_throughput.py 2>/dev/null \
         | grep -o 'PARETO_TP_JSON .*' | sed 's/^PARETO_TP_JSON //')
  if [ -n "$line" ]; then
    python3 -c "
import json,sys
d=json.loads('''$line'''); d['label']='$label'
print(json.dumps(d))" >> "$OUT"
    echo "$line" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"      {d['decode_tok_s']} tok/s  (step {d['us']['STEP_TOTAL']}us, mem {d['memory_pct_of_step']}%)\")"
  else
    echo "      FAILED"
  fi
}

echo "=== RAG-equivalent: k=$K docs in context, varying doc length ==="
for dl in $DOC_LENS; do
  kv=$(( Q + K * dl ))
  run "rag_k${K}_doclen${dl}" MODE=rag KVLEN=$kv
done

echo "=== memory layer: prompt = question only, ladder over bank size ==="
for M in $BANKS; do
  run "mem_M${M}_exact_bf16"  MODE=mem KVLEN=$Q MEMM=$M TOPK_MODE=exact  KEY_DTYPE=bf16 N_MEM_LAYERS=$NML
  run "mem_M${M}_approx_bf16" MODE=mem KVLEN=$Q MEMM=$M TOPK_MODE=approx KEY_DTYPE=bf16 N_MEM_LAYERS=$NML
  run "mem_M${M}_approx_int8" MODE=mem KVLEN=$Q MEMM=$M TOPK_MODE=approx KEY_DTYPE=int8 N_MEM_LAYERS=$NML
done

echo "=== SUMMARY ==="
python3 - "$OUT" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
for r in sorted(rows, key=lambda r: -r["decode_tok_s"]):
    c = r["config"]
    extra = f"bank={c['bank_slots']:,} {c['topk_mode']}/{c['key_dtype']} x{c['n_mem_layers']}L" if r["mode"] == "mem" else f"kv={c['kvlen']:,}"
    print(f"  {r['decode_tok_s']:>8.1f} tok/s  {r['label']:<28} {extra}")
PY
gsutil -q cp "$OUT" "gs://${CKPT_BUCKET:-memory-layers-training-usc1}/pareto/throughput.jsonl" 2>/dev/null \
  && echo "[tp] -> gs://${CKPT_BUCKET:-memory-layers-training-usc1}/pareto/throughput.jsonl"
echo "[tp] DONE"
