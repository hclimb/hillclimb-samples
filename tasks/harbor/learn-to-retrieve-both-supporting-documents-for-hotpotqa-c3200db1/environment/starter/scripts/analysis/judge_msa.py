"""Phase 2: LLM-judge all MSA output JSONs in ONE vLLM session.

Finds the latest msa_*.json per dataset under outputs/, runs the llm_judge_accuracy
metric (starts the vLLM Qwen3-8B judge once, reused across datasets), writes the
score back into each JSON, and prints a summary table.

    cd memory-layers && set -a && source .env && set +a && \
    PYTHONPATH=. .venv/bin/python scripts/embed/judge_msa.py
"""
import os, sys, glob, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from evals.metrics import run_metrics

root = sys.argv[1] if len(sys.argv) > 1 else 'outputs'
files = glob.glob(os.path.join(root, '**', 'msa_*.json'), recursive=True)
latest = {}
for f in files:
    name = os.path.basename(f)                 # msa_popqa.json
    mt = os.path.getmtime(f)
    if name not in latest or mt > latest[name][0]:
        latest[name] = (mt, f)

summary = {}
for name, (mt, f) in sorted(latest.items()):
    d = json.load(open(f))
    samples = d.get('samples', [])
    if not samples:
        print(f"{name}: SKIP (no samples)", flush=True); continue
    ann, agg = run_metrics(samples, {'llm_judge_accuracy': {}})
    d['samples'] = ann
    d['metrics'].update(agg)
    json.dump(d, open(f, 'w'), indent=2)
    acc = agg.get('llm_judge_accuracy')
    summary[name] = acc
    print(f"{name}: llm_judge_accuracy={acc:.4f} (n={len(samples)})", flush=True)

print("\n==== MSA-4B LLM-JUDGE SUMMARY ====", flush=True)
for k, v in sorted(summary.items()):
    print(f"  {k:32s} {v:.4f}", flush=True)
if summary:
    print(f"  {'AVERAGE':32s} {sum(summary.values())/len(summary):.4f}", flush=True)
print("JUDGE_DONE", flush=True)
