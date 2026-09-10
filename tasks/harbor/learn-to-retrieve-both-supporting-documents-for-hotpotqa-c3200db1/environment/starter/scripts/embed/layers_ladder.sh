#!/bin/bash
# Does the optimization ladder still pay once n_mem_layers drops from 4 to 1, at LongHealth's
# actual bank size (233,041 tokens)?
#
# PREDICTION (recorded before the run so it can be falsified):
#   4 layers: exact 0.44 -> approx 0.48 -> int8 0.48   (~9% spread, marginal)
#   1 layer : exact 0.60 -> approx 0.61 -> int8 0.62   (~3% spread, inside the ~9% noise floor)
# Reasoning: the ladder only accelerates the bank scan, which is ~26% of the step at 4 layers and
# ~8% at 1 layer, so cutting layers removes most of what the ladder had to optimize.
set -uo pipefail
set -a; . "$HOME/.env" 2>/dev/null; set +a
export PATH="$HOME/.local/bin:$PATH"; cd "$HOME/memory-layers"
export JAX_PLATFORMS=tpu PYTHONPATH=.
OUT="$HOME/layers_ladder.jsonl"; : > "$OUT"
M=233041   # LongHealth corpus, exact token count

run () { local label="$1"; shift
  local line
  line=$(env "$@" GEN_TOKENS=100 uv run --no-sync python scripts/embed/bench_pareto_throughput.py 2>/dev/null \
         | grep -o 'PARETO_TP_JSON .*' | sed 's/^PARETO_TP_JSON //')
  if [ -n "$line" ]; then
    python3 -c "
import json; d=json.loads('''$line'''); d['label']='$label'; print(json.dumps(d))" >> "$OUT"
    echo "$line" | python3 -c "
import json,sys; d=json.load(sys.stdin); e=d['end_to_end']; m=d.get('memory_ops_us',{})
print(f\"  {'$label':<26} decode={e['decode_us_per_tok']:>6.0f}us/tok  mem={d['memory_pct_of_step']:>5.1f}%  {e['queries_per_s']:>6.3f} q/s\")"
  else
    echo "  $label FAILED"
  fi
}

for L in 4 1; do
  echo "=== n_mem_layers=$L, bank=$M ==="
  run "L${L}_exact_bf16"  MODE=mem KVLEN=64 MEMM=$M TOPK_MODE=exact  KEY_DTYPE=bf16 N_MEM_LAYERS=$L
  run "L${L}_approx_bf16" MODE=mem KVLEN=64 MEMM=$M TOPK_MODE=approx KEY_DTYPE=bf16 N_MEM_LAYERS=$L
  run "L${L}_approx_int8" MODE=mem KVLEN=64 MEMM=$M TOPK_MODE=approx KEY_DTYPE=int8 N_MEM_LAYERS=$L
done

echo "=== LADDER SPREAD PER LAYER COUNT ==="
python3 - "$OUT" <<'PY'
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1])]
for L in (4,1):
    r=[x for x in rows if x["config"]["n_mem_layers"]==L]
    if not r: continue
    q=[x["end_to_end"]["queries_per_s"] for x in r]
    lo,hi=min(q),max(q)
    print(f"  {L} layer(s): {' -> '.join(f'{x:.3f}' for x in q)}   spread {100*(hi-lo)/lo:5.1f}%  "
          f"{'INSIDE ~9% noise' if 100*(hi-lo)/lo < 9 else 'resolvable'}")
PY
echo "LADDER DONE"
