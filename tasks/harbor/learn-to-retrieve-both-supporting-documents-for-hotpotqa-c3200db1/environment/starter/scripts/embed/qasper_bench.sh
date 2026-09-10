#!/bin/bash
# QASPER, matched conditions: both systems have the full 1,169-paper corpus available.
#   RAG@5   : retrieves 5 WHOLE papers -> 64 + 5*5404 = 27,084 token prompt, prefill + 100 gen
#   ours    : question-only prompt (64 tok), whole corpus in bank (6,318,182 slots), 1 mem layer
# Bank is 3x larger than anything measured so far, so the ladder is swept rather than assumed.
set -uo pipefail
set -a; . "$HOME/.env" 2>/dev/null; set +a
export PATH="$HOME/.local/bin:$PATH"; cd "$HOME/memory-layers"
export JAX_PLATFORMS=tpu PYTHONPATH=.
OUT="$HOME/qasper_bench.jsonl"; : > "$OUT"
BANK=6318182
RAGKV=27084

run () { local label="$1"; shift
  local line
  line=$(env "$@" GEN_TOKENS=100 uv run --no-sync python scripts/embed/bench_pareto_throughput.py 2>/dev/null \
         | grep -o 'PARETO_TP_JSON .*' | sed 's/^PARETO_TP_JSON //')
  if [ -n "$line" ]; then
    python3 -c "
import json; d=json.loads('''$line'''); d['label']='$label'; print(json.dumps(d))" >> "$OUT"
    echo "$line" | python3 -c "
import json,sys; d=json.load(sys.stdin); e=d['end_to_end']
print(f\"  {'$label':<26} prefill={e['prefill_us']/1e6:>7.3f}s  decode={e['decode_us_per_tok']:>6.0f}us/tok  total={e['query_us']/1e6:>7.3f}s  {e['queries_per_s']:>6.3f} q/s\")"
  else
    echo "  $label FAILED"
  fi
}

echo "=== RAG@5 whole papers (27,084 tok prompt) ==="
run "qasper_rag_k5" MODE=rag KVLEN=$RAGKV

echo "=== ours: 1 memory layer, whole corpus in bank (6.32M slots) ==="
run "qasper_mem_1L_exact_bf16"  MODE=mem KVLEN=64 MEMM=$BANK TOPK_MODE=exact  KEY_DTYPE=bf16 N_MEM_LAYERS=1
run "qasper_mem_1L_approx_bf16" MODE=mem KVLEN=64 MEMM=$BANK TOPK_MODE=approx KEY_DTYPE=bf16 N_MEM_LAYERS=1
run "qasper_mem_1L_approx_int8" MODE=mem KVLEN=64 MEMM=$BANK TOPK_MODE=approx KEY_DTYPE=int8 N_MEM_LAYERS=1

echo "=== reference: 4 memory layers, same bank ==="
run "qasper_mem_4L_approx_int8" MODE=mem KVLEN=64 MEMM=$BANK TOPK_MODE=approx KEY_DTYPE=int8 N_MEM_LAYERS=4

echo "=== SUMMARY ==="
python3 - "$OUT" <<'PY'
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1])]
rag=[r for r in rows if r["mode"]=="rag"]
base=rag[0]["end_to_end"]["queries_per_s"] if rag else None
for r in sorted(rows,key=lambda r:-r["end_to_end"]["queries_per_s"]):
    e=r["end_to_end"]; rel=f"{e['queries_per_s']/base:5.2f}x RAG@5" if base else ""
    print(f"  {e['queries_per_s']:>7.3f} q/s  {r['label']:<28} total={e['query_us']/1e6:>7.3f}s  {rel}")
PY
echo "QASPER BENCH DONE"
