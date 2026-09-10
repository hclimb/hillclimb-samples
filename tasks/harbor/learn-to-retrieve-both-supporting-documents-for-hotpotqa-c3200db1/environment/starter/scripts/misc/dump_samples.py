"""Dump readable side-by-side eval outputs (MSA vs membed) per dataset.

Reads results/judged_pool/{msa,mb}__<ds>.json (paired, same questions) and writes
a markdown file with N examples per dataset: question, ground truth, each model's
answer + 0-5 score + binary verdict.

    python scripts/embed/dump_samples.py [N] [out.md]
"""
import sys, os, json, re

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUT = sys.argv[2] if len(sys.argv) > 2 else 'claude/eval_sample_outputs.md'
root = 'results/judged_pool'

DS = ['msmarco_v1', 'natural_questions', 'narrativeqa', '2wikimultihopqa',
      'hotpotqa', 'musique', 'dureader', 'popqa', 'triviaqa_10m']
LABEL = {'msmarco_v1': 'MS MARCO v1', 'natural_questions': 'Natural Questions',
         'narrativeqa': 'NarrativeQA', '2wikimultihopqa': '2WikiMultiHopQA',
         'hotpotqa': 'HotpotQA', 'musique': 'MuSiQue', 'dureader': 'DuReader',
         'popqa': 'PopQA', 'triviaqa_10m': 'TriviaQA (10M)'}


def question(prompt):
    m = re.search(r'<\|im_start\|>user\s*(.*?)<\|im_end\|>', prompt, re.S)
    q = (m.group(1) if m else prompt).strip()
    return q.replace('\n', ' ')


def gt(s):
    g = s.get('ground_truth', '')
    try:
        v = json.loads(g)
        if isinstance(v, list):
            return ' | '.join(map(str, v))
    except Exception:
        pass
    return str(g).replace('\n', ' ')


def ans(s):
    a = (s.get('generated_answer') or '').strip()
    if not a:
        a = (s.get('generated') or '').strip()
    return a.replace('\n', ' ')


lines = ["# Eval sample outputs — MSA-4B vs qwen3_mem_embed",
         f"\n{N} examples/dataset. Same questions for both models. "
         "score = paper 0–5 rubric; ✓/✗ = binary judge.\n"]

for ds in DS:
    mp = os.path.join(root, f'msa__{ds}.json')
    bp = os.path.join(root, f'mb__{ds}.json')
    mb = json.load(open(bp))['samples'] if os.path.exists(bp) else []
    ms = json.load(open(mp))['samples'] if os.path.exists(mp) else []
    lines.append(f"\n## {LABEL[ds]}\n")
    for i in range(min(N, len(mb) or len(ms))):
        b = mb[i] if i < len(mb) else None
        m = ms[i] if i < len(ms) else None
        ref = b or m
        lines.append(f"**Q{i+1}:** {question(ref['prompt'])}")
        lines.append(f"**Ground truth:** {gt(ref)}\n")
        if m:
            v = '✓' if m.get('llm_judge_accuracy') else '✗'
            lines.append(f"- **MSA-4B** [{m.get('llm_judge_score')}/5 {v}]: {ans(m)}")
        else:
            lines.append("- **MSA-4B**: — (no output / OOM)")
        if b:
            v = '✓' if b.get('llm_judge_accuracy') else '✗'
            lines.append(f"- **membed** [{b.get('llm_judge_score')}/5 {v}]: {ans(b)}")
        lines.append("")

open(OUT, 'w').write('\n'.join(lines))
print(f"wrote {OUT} ({len(lines)} lines)")
