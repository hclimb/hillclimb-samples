#!/bin/bash
# Runs ON a box. END-TO-END throughput axis of the Pareto plot: queries/sec for a QA workload
# (prefill the prompt, then generate GEN tokens), RAG-equivalent vs the memory-layer optimization
# ladder, all from one harness (scripts/embed/bench_pareto_throughput.py).
#
#   bash scripts/embed/sweep_pareto_e2e.sh
#
# WHY END-TO-END AND NOT tok/s. Per-token decode throughput amortises the prompt away and hides
# the entire architectural asymmetry: measured decode-only, RAG loses just 17% going from doc_len
# 256 to 16384, because attention over the KV cache is a small share of a 36-layer step. The cost
# RAG actually pays is PREFILL -- k*doc_len tokens pushed through the model once per query, before
# a single output token appears. A QA query generates ~100 tokens, so prefill is a first-class
# term, and only an end-to-end metric shows it.
#
# The memory layer's prompt is the question alone (~64 tokens) regardless of document length, so
# its prefill is flat; it pays instead on every decoded token (a bank scan per memory layer). That
# is the trade, and the crossover is the number this sweep exists to find.
set -uo pipefail
set -a; . "$HOME/.env" 2>/dev/null; set +a
export PATH="$HOME/.local/bin:$PATH"; cd "${REPO_DIR:-$HOME/memory-layers}"
export JAX_PLATFORMS=tpu PYTHONPATH=.

OUT="${OUT:-$HOME/pareto_e2e.jsonl}"; : > "$OUT"
Q=64                       # question tokens in the prompt
K="${K:-5}"                # docs retrieved into the prompt for RAG
GEN="${GEN:-100}"          # generated tokens per query
NML="${N_MEM_LAYERS:-4}"   # ground4layer has 4 memory layers; hard-neg has 1
DOC_LENS="${DOC_LENS:-256 1024 2048 4096 8192 16384}"; DOC_LENS="${DOC_LENS//_/ }"
BANKS="${BANKS:-131072 524288 2097152}";               BANKS="${BANKS//_/ }"

run () {  # run <label> <env assignments...>
  local label="$1"; shift
  local line
  line=$(env "$@" GEN_TOKENS=$GEN uv run --no-sync python scripts/embed/bench_pareto_throughput.py 2>/dev/null \
         | grep -o 'PARETO_TP_JSON .*' | sed 's/^PARETO_TP_JSON //')
  if [ -n "$line" ]; then
    python3 -c "
import json; d=json.loads('''$line'''); d['label']='$label'; print(json.dumps(d))" >> "$OUT"
    echo "$line" | python3 -c "
import json,sys; d=json.load(sys.stdin); e=d['end_to_end']
print(f\"  {'$label':<26} prefill={e['prefill_us']/1000:>9.1f}ms  decode={e['decode_us_per_tok']:>6.0f}us/tok  query={e['query_us']/1000:>9.1f}ms  {e['queries_per_s']:>6.3f} q/s\")"
  else
    echo "  $label FAILED"
  fi
}

echo "=== END-TO-END: prefill + ${GEN} generated tokens, k=$K, ${NML} memory layers ==="
echo "--- RAG-equivalent: prompt = question + k*doc_len ---"
for dl in $DOC_LENS; do run "rag_k${K}_doclen${dl}" MODE=rag KVLEN=$(( Q + K * dl )); done

# Ladder is cumulative: exact/bf16 baseline -> + approx top-k -> + int8 keys. Both optimizations
# touch the bank scan only, so they separate visibly only at large banks (decode is transformer-
# bound, not scan-bound) -- report the bank size each gain came from, not the gain alone.
echo "--- memory layer: prompt = question only, optimization ladder over bank size ---"
for M in $BANKS; do
  run "mem_M${M}_exact_bf16"  MODE=mem KVLEN=$Q MEMM=$M TOPK_MODE=exact  KEY_DTYPE=bf16 N_MEM_LAYERS=$NML
  run "mem_M${M}_approx_bf16" MODE=mem KVLEN=$Q MEMM=$M TOPK_MODE=approx KEY_DTYPE=bf16 N_MEM_LAYERS=$NML
  run "mem_M${M}_approx_int8" MODE=mem KVLEN=$Q MEMM=$M TOPK_MODE=approx KEY_DTYPE=int8 N_MEM_LAYERS=$NML
done

echo "=== SUMMARY (end-to-end queries/sec) ==="
python3 - "$OUT" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
for r in sorted(rows, key=lambda r: -r["end_to_end"]["queries_per_s"]):
    c, e = r["config"], r["end_to_end"]
    extra = (f"bank={c['bank_slots']:,} {c['topk_mode']}/{c['key_dtype']} x{c['n_mem_layers']}L"
             if r["mode"] == "mem" else f"prompt={c['kvlen']:,} tok")
    print(f"  {e['queries_per_s']:>7.3f} q/s  {r['label']:<28} prefill={e['prefill_us']/1000:>9.1f}ms  {extra}")
PY
gsutil -q cp "$OUT" "gs://${CKPT_BUCKET:-memory-layers-training-usc1}/pareto/e2e.jsonl" \
  && echo "-> gs://${CKPT_BUCKET:-memory-layers-training-usc1}/pareto/e2e.jsonl"
echo "E2E DONE"
