import math
import random

from utils.protocol import REPETITIONS, TIMED_CALLS, WARMUP_CALLS
from utils.roofline import MODEL


def fixed_score_case():
    batch, queries, heads, capacity = 128, 3, 128, 32768
    scopes = [dict(nominal=capacity, topk=topk, page=page, lengths=[capacity] * batch,
                   topk_lengths=None, padded_rows=[], capacity=capacity,
                   pages=batch * capacity // page,
                   page_stride=math.ceil(page * 584 / 576) * 576,
                   valid_counts=[[topk] * queries for _ in range(batch)])
              for topk, page in [(128, 256), (1024, 64)]]
    return dict(name='roofline-b128-q3-h128-k1152-c32768', batch=batch, queries=queries,
                heads=heads, scored=True, window=True, sinks=True, scopes=scopes,
                sink_categories=[0] * heads, throughput_weight=1)


def make_manifest():
    randomizer = random.Random(20260904)
    cases = []
    configurations = [(64, 512, 64, False), (128, 1024, 64, False),
                      (64, 1024, 2, True), (128, 1024, 2, True)]
    for config_id, (heads, extra_topk, extra_page, extra_length) in enumerate(configurations):
        for batch, queries in [(2, 1), (74, 2), (128, 3)]:
            cases.append(dict(name=f'production-{config_id + 1}-b{batch}-q{queries}',
                              batch=batch, queries=queries, heads=heads, scored=True,
                              window=True, sinks=True,
                              scopes=[(16384, 128, 256, False),
                                      (16384, extra_topk, extra_page, extra_length)]))
    for heads in [64, 128]:
        cases.append(dict(name=f'long-h{heads}', batch=148, queries=2, heads=heads,
                          scored=True, window=False, sinks=True,
                          scopes=[(32768, 16384, 64, False)]))
    for heads in [64, 128]:
        for extra in [False, True]:
            for sinks in [False, True]:
                cases.append(dict(name=f'fixture-h{heads}-extra{int(extra)}-sink{int(sinks)}',
                                  batch=4, queries=3, heads=heads, scored=False,
                                  window=False, sinks=sinks,
                                  scopes=[(650, 576, 53, True)] +
                                  ([(512, 64, 61, True)] if extra else [])))
    for case in cases:
        batch, queries = case['batch'], case['queries']
        case['sink_categories'] = [
            (-1 if category < -0.5 else 1 if category > 0.5 else 0)
            for category in [randomizer.normalvariate(0, 1) for _ in range(case['heads'])]
        ] if case['sinks'] else None
        scopes = []
        for scope_id, (nominal, topk, page, masked) in enumerate(case['scopes']):
            lengths = [int(max(randomizer.normalvariate(nominal, nominal / 2), queries))
                       for _ in range(batch)]
            selected = [randomizer.randrange(topk + 1) for _ in range(batch)] if masked else None
            padded_rows = []
            if not case['scored']:
                lengths = [0, 7, 650, 317] if scope_id == 0 else [0, 5, 512, 129]
                selected = [0, topk, topk // 2 + 1, 0]
                padded_rows = [[2, 1]]
            capacity = max(1, math.ceil(max(lengths) / (4 * page))) * 4 * page
            valid = []
            for batch_id, length in enumerate(lengths):
                row_counts = []
                for row in range(queries):
                    available = length - queries + row + 1 if case['window'] and scope_id == 0 else length
                    count = min(max(available, 0), topk)
                    if selected is not None:
                        count = min(count, selected[batch_id])
                    if [batch_id, row] in padded_rows:
                        count = 0
                    row_counts.append(count)
                valid.append(row_counts)
            scopes.append(dict(nominal=nominal, topk=topk, page=page, lengths=lengths,
                               topk_lengths=selected, padded_rows=padded_rows,
                               capacity=capacity, pages=batch * capacity // page,
                               page_stride=math.ceil(page * 584 / 576) * 576,
                               valid_counts=valid))
        case['scopes'] = scopes
    for case in cases:
        # Preserve the original profiles as correctness gates, not reward weights.
        case['throughput_weight'] = 0
    cases.append(fixed_score_case())
    return dict(version=5, profile_seed=20260904, scale=512 ** -0.55,
                metric='paired_throughput_ratio', roofline_model=MODEL,
                reward_normalization=dict(formula='candidate_rate / baseline_rate',
                                          baseline='remeasured frozen incumbent on the same GPU'),
                warmup_calls=WARMUP_CALLS, timed_calls=TIMED_CALLS,
                repetitions=REPETITIONS, warmup_seconds=30, workload_seconds=120,
                output_tolerance=[1e-3, 2.01 / 128, 5e-6],
                lse_tolerance=[1e-6, 8.01 / 65536, 1e-7], cases=cases)
