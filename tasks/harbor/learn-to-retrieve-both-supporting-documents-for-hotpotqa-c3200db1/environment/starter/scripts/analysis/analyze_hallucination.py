"""Hallucination analysis for qwen3_mem_embed: grounding vs answer correctness.

For each membed dataset (results/judged_pool/mb__<ds>.json) we have per-sample:
  - doc_token_hit_rate  : fraction of GENERATED tokens whose lookups hit the gold
                          doc (how much the generation actually attended evidence)
                          -> the GRADED grounding signal used here.
  - doc_hit_rate        : 1.0 if ANY lookup hit the gold doc (~1 for every sample,
                          so it is degenerate as a threshold -> not used).
  - llm_judge_accuracy  : 1.0 if the answer is correct.

"Grounded" := doc_token_hit_rate >= threshold. The threshold defaults to the
PER-DATASET MEAN of doc_token_hit_rate (so each dataset is split into
above/below-average grounding); pass a float to override with a fixed threshold.

Decomposition (2x2 of grounded x correct):
  - grounded_correct   : grounded AND correct
  - hallucinated       : grounded AND wrong   (attended evidence, still wrong)
  - ungrounded_correct : NOT grounded AND correct (right despite low grounding)
  - ungrounded_wrong   : NOT grounded AND wrong

Headline hallucination rate = P(wrong | grounded). We also report accuracy among
grounded vs ungrounded samples: if grounding barely lifts accuracy, the model is
not using the evidence it attends.

    python scripts/embed/analyze_hallucination.py [threshold|mean]
"""
import sys, os, json, statistics

ARG = sys.argv[1] if len(sys.argv) > 1 else 'mean'
USE_MEAN = (ARG == 'mean')
FIXED = None if USE_MEAN else float(ARG)
root = 'results/judged_pool'

DS = ['msmarco_v1', 'natural_questions', 'narrativeqa', '2wikimultihopqa',
      'hotpotqa', 'musique', 'dureader', 'popqa', 'triviaqa_10m']

def m(xs): return statistics.mean(xs) if xs else float('nan')

agg = {'gc': 0, 'hal': 0, 'uc': 0, 'uw': 0}
print(f"grounded := doc_token_hit_rate >= {'per-dataset mean' if USE_MEAN else FIXED}\n")
hdr = (f"{'dataset':18s} {'n':>4s} {'thr':>5s} {'gnd%':>5s} {'acc':>5s} | "
       f"{'halluc':>7s} | {'acc|gnd':>8s} {'acc|ungnd':>9s}")
print(hdr); print('-' * len(hdr))
for ds in DS:
    p = os.path.join(root, f'mb__{ds}.json')
    if not os.path.exists(p):
        continue
    S = json.load(open(p))['samples']
    ths = [s.get('doc_token_hit_rate') for s in S if s.get('doc_token_hit_rate') is not None]
    thr = m(ths) if USE_MEAN else FIXED
    gc = hal = uc = uw = 0
    for s in S:
        th = s.get('doc_token_hit_rate') or 0.0
        gnd = th >= thr
        corr = bool(s.get('llm_judge_accuracy'))
        if gnd and corr: gc += 1
        elif gnd and not corr: hal += 1
        elif (not gnd) and corr: uc += 1
        else: uw += 1
    n = len(S)
    gnd_n = gc + hal
    halluc = hal / gnd_n if gnd_n else float('nan')
    acc = (gc + uc) / n
    acc_gnd = gc / gnd_n if gnd_n else float('nan')
    acc_ungnd = uc / (uc + uw) if (uc + uw) else float('nan')
    agg['gc'] += gc; agg['hal'] += hal; agg['uc'] += uc; agg['uw'] += uw
    print(f"{ds:18s} {n:>4d} {thr:>5.2f} {gnd_n/n*100:>4.0f}% {acc:>5.2f} | "
          f"{halluc*100:>6.0f}% | {acc_gnd:>8.2f} {acc_ungnd:>9.2f}")

N = sum(agg.values()); gnd = agg['gc'] + agg['hal']
print('-' * len(hdr))
print(f"{'POOLED':18s} {N:>4d} {'':>5s} {gnd/N*100:>4.0f}% {(agg['gc']+agg['uc'])/N:>5.2f} | "
      f"{agg['hal']/gnd*100:>6.0f}% | {agg['gc']/gnd:>8.2f} "
      f"{agg['uc']/(agg['uc']+agg['uw']):>9.2f}")

print(f"\nPooled 2x2 (grounded[>=mean token-hit] x correct), n={N}:")
print(f"  grounded_correct   : {agg['gc']:4d}  ({agg['gc']/N*100:.1f}%)")
print(f"  HALLUCINATED       : {agg['hal']:4d}  ({agg['hal']/N*100:.1f}%)   P(wrong|grounded)={agg['hal']/gnd*100:.0f}%")
print(f"  ungrounded_correct : {agg['uc']:4d}  ({agg['uc']/N*100:.1f}%)")
print(f"  ungrounded_wrong   : {agg['uw']:4d}  ({agg['uw']/N*100:.1f}%)")
print(f"\nAccuracy | grounded={agg['gc']/gnd:.3f}  ungrounded={agg['uc']/(agg['uc']+agg['uw']):.3f}")
