"""MSA-paper 0-5 LLM-judge over MSA output JSONs (one vLLM session).

Uses the exact 0-5 scoring prompt from the MSA paper (Appendix A) via the
`llm_judge_score` metric. Writes llm_judge_score back into each JSON and prints
a per-dataset average (0-5 scale, comparable to the paper's Table 2).

    PYTHONPATH=. .venv/bin/python scripts/embed/judge_msa_score.py /tmp/judge_pool
"""
import os, sys, glob, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from evals.metrics import run_metrics

root = sys.argv[1] if len(sys.argv) > 1 else 'outputs'
files = glob.glob(os.path.join(root, '**', 'msa_*.json'), recursive=True)
latest = {}
for f in files:
    name = os.path.basename(f)
    mt = os.path.getmtime(f)
    if name not in latest or mt > latest[name][0]:
        latest[name] = (mt, f)

summary = {}
for name, (mt, f) in sorted(latest.items()):
    d = json.load(open(f))
    samples = d.get('samples', [])
    if not samples:
        print(f"{name}: SKIP (no samples)", flush=True); continue
    ann, agg = run_metrics(samples, {'llm_judge_score': {}})
    d['samples'] = ann
    d['metrics'].update(agg)
    json.dump(d, open(f, 'w'), indent=2)
    s = agg.get('llm_judge_score')
    summary[name] = s
    print(f"{name}: llm_judge_score={s:.3f} (0-5, n={len(samples)})", flush=True)

print("\n==== MSA-4B 0-5 LLM-JUDGE SUMMARY (paper rubric) ====", flush=True)
for k, v in sorted(summary.items()):
    print(f"  {k:32s} {v:.3f}", flush=True)
if summary:
    print(f"  {'AVERAGE':32s} {sum(summary.values())/len(summary):.3f}", flush=True)
print("SCORE_JUDGE_DONE", flush=True)
