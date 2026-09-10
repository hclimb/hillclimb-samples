"""Compute mean +/- std per dataset/model from judged pool JSONs.

Reads /tmp/judge_pool_judged/*.json (files named msa__<ds>.json / mb__<ds>.json
with per-sample 'llm_judge_score' and 'llm_judge_accuracy' fields). Prints
markdown tables (mean +/- pstdev over the per-sample scores) for both metrics.

    python scripts/embed/analyze_pool.py <pool_dir>
"""
import sys, os, glob, json, statistics

root = sys.argv[1] if len(sys.argv) > 1 else 'results/judged_pool'
DS_ORDER = ['msmarco_v1','natural_questions','narrativeqa','2wikimultihopqa',
            'hotpotqa','musique','dureader','popqa','triviaqa_10m']
DS_LABEL = {'msmarco_v1':'MS MARCO v1','natural_questions':'Natural Questions',
            'narrativeqa':'NarrativeQA','2wikimultihopqa':'2WikiMultiHopQA',
            'hotpotqa':'HotpotQA','musique':'MuSiQue','dureader':'DuReader',
            'popqa':'PopQA','triviaqa_10m':'TriviaQA (10M)'}

def stats(vals):
    """mean and standard ERROR of the mean (sample sd / sqrt(n))."""
    vals = [v for v in vals if v is not None]
    if not vals: return None
    n = len(vals)
    m = statistics.mean(vals)
    sem = (statistics.stdev(vals) / (n ** 0.5)) if n > 1 else 0.0
    return m, sem, n

# data[ds][model][metric] = (mean,std,n)
data = {}
for f in sorted(glob.glob(os.path.join(root, '*.json'))):
    name = os.path.basename(f)
    model, ds = name.replace('.json','').split('__', 1)
    d = json.load(open(f)); samples = d.get('samples', [])
    for metric in ('llm_judge_score','llm_judge_accuracy'):
        vals = [s.get(metric) for s in samples]
        st = stats(vals)
        if st: data.setdefault(ds, {}).setdefault(model, {})[metric] = st

def cell(ds, model, metric):
    st = data.get(ds, {}).get(model, {}).get(metric)
    if not st: return '—'
    m, s, n = st
    return f"{m:.3f} ± {s:.3f}"

def avg_row(model, metric):
    means = [data[ds][model][metric][0] for ds in DS_ORDER
             if model in data.get(ds, {}) and metric in data[ds][model]]
    if not means: return '—'
    m = statistics.mean(means)
    sem = (statistics.stdev(means) / (len(means) ** 0.5)) if len(means) > 1 else 0.0
    return f"{m:.3f} ± {sem:.3f}"

for metric, title in (('llm_judge_score','### 0–5 LLM-judge (paper rubric)'),
                      ('llm_judge_accuracy','### Binary LLM-judge accuracy')):
    print(f"\n{title}\n")
    print("| Dataset | MSA-4B | qwen3_mem_embed |")
    print("|---|---|---|")
    for ds in DS_ORDER:
        print(f"| {DS_LABEL[ds]} | {cell(ds,'msa',metric)} | {cell(ds,'mb',metric)} |")
    print(f"| **Average** | {avg_row('msa',metric)} | {avg_row('mb',metric)} |")
