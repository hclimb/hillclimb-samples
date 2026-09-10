#!/bin/bash
# Long-context throughput: the shapes LongHealth / QASPER actually imply.
# Throughput only -- no accuracy, no model quality claim (deliberate: the question is whether
# RAG's prefill collapses at these context sizes, which is independent of whether any checkpoint
# can answer the questions).
set -uo pipefail
set -a; . "$HOME/.env" 2>/dev/null; set +a
export PATH="$HOME/.local/bin:$PATH"; cd "$HOME/memory-layers"
export JAX_PLATFORMS=tpu PYTHONPATH=.
OUT="$HOME/longctx.jsonl"; : > "$OUT"
GEN=100

run () { local label="$1"; shift
  local line
  line=$(env "$@" GEN_TOKENS=$GEN uv run --no-sync python scripts/embed/bench_pareto_throughput.py 2>/dev/null \
         | grep -o 'PARETO_TP_JSON .*' | sed 's/^PARETO_TP_JSON //')
  if [ -n "$line" ]; then
    python3 -c "
import json; d=json.loads('''$line'''); d['label']='$label'; print(json.dumps(d))" >> "$OUT"
    echo "$line" | python3 -c "
import json,sys; d=json.load(sys.stdin); e=d['end_to_end']
print(f\"  {'$label':<28} prefill={e['prefill_us']/1e6:>8.2f}s  decode={e['decode_us_per_tok']:>6.0f}us/tok  query={e['query_us']/1e6:>8.2f}s  {e['queries_per_s']:>7.4f} q/s\")"
  else
    echo "  $label FAILED"
  fi
}

echo "=== RAG / long-context: whole documents in the prompt ==="
run "ctx7k_one_paper"        MODE=rag KVLEN=7000      # QASPER native, LongHealth task 1
run "ctx35k_5docs"           MODE=rag KVLEN=35000     # top-5 whole docs retrieved
run "ctx70k_10docs"          MODE=rag KVLEN=70000
run "ctx140k_longhealth_t2"  MODE=rag KVLEN=140000    # all 20 patient records
run "ctx280k_stress"         MODE=rag KVLEN=280000

echo "=== memory layer at the MATCHED bank (corpus tokens = bank slots) ==="
run "mem_bank7k"    MODE=mem KVLEN=64 MEMM=7168   TOPK_MODE=approx KEY_DTYPE=int8 N_MEM_LAYERS=4
run "mem_bank140k"  MODE=mem KVLEN=64 MEMM=143360 TOPK_MODE=approx KEY_DTYPE=int8 N_MEM_LAYERS=4
run "mem_bank280k"  MODE=mem KVLEN=64 MEMM=286720 TOPK_MODE=approx KEY_DTYPE=int8 N_MEM_LAYERS=4

echo "=== SUMMARY ==="
python3 - "$OUT" <<'PY'
import json, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
for r in sorted(rows, key=lambda r:-r["end_to_end"]["queries_per_s"]):
    e=r["end_to_end"]
    print(f"  {e['queries_per_s']:>8.4f} q/s  {r['label']:<28} prefill={e['prefill_us']/1e6:>8.2f}s")
PY
echo "LONGCTX DONE"
