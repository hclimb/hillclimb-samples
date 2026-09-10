"""Unified LLM-judge over a pool of QA-output JSONs (one vLLM session).

Runs BOTH metrics on every *.json in the pool dir:
  - llm_judge_score    (MSA-paper 0-5 rubric, Appendix A)
  - llm_judge_accuracy (binary 0/1 match)
Per-sample scores are written back into each JSON (samples[i][metric]) so std
can be computed downstream. Prints per-file aggregate means.

    PYTHONPATH=. .venv/bin/python scripts/embed/judge_pool.py /tmp/judge_pool
"""
import os, sys, glob, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from evals.metrics import run_metrics

root = sys.argv[1] if len(sys.argv) > 1 else 'outputs'
files = sorted(glob.glob(os.path.join(root, '*.json')))
summary = {}
for f in files:
    name = os.path.basename(f)
    d = json.load(open(f))
    samples = d.get('samples', [])
    if not samples:
        print(f"{name}: SKIP (no samples)", flush=True); continue
    ann, agg = run_metrics(samples, {'llm_judge_score': {}, 'llm_judge_accuracy': {}})
    d['samples'] = ann
    d.setdefault('metrics', {}).update(agg)
    json.dump(d, open(f, 'w'), indent=2)
    summary[name] = agg
    print(f"{name}: score={agg.get('llm_judge_score'):.3f} acc={agg.get('llm_judge_accuracy'):.3f} n={len(samples)}", flush=True)

print("\n==== POOL JUDGE SUMMARY ====", flush=True)
for k in sorted(summary):
    a = summary[k]
    print(f"  {k:36s} score={a.get('llm_judge_score'):.3f}  acc={a.get('llm_judge_accuracy'):.3f}", flush=True)
print("POOL_JUDGE_DONE", flush=True)
