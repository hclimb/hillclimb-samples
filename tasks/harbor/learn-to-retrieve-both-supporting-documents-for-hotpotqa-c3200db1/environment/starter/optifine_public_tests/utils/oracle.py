import math


TARGET_NDCG = 0.715285038072705
SCORE_MARGIN = 0.01


def reward(candidate, baseline):
    if not math.isfinite(candidate) or not 0 <= candidate <= 1:
        raise ValueError('Invalid candidate nDCG')
    if not math.isfinite(baseline) or not 0 <= baseline < TARGET_NDCG:
        raise ValueError('Starter nDCG must be finite and below the target')
    progress = (candidate - baseline) / (TARGET_NDCG - baseline)
    return min(1.0, max(0.0, (progress - SCORE_MARGIN) / (1 - 2 * SCORE_MARGIN)))


def score(results, queries, gold, slices, corpus_count):
    expected = {row['id']: row['question'] for row in queries}
    targets = {row['id']: set(row['pos_doc_ids']) for row in gold}
    types = {row['id']: row['type'] for row in slices}
    if len(expected) != len(queries) or not expected or targets.keys() != expected.keys() or types.keys() != expected.keys():
        raise ValueError('Invalid trusted panel joins')
    if len(gold) != len(targets) or any(len(values) != 2 for values in targets.values()):
        raise ValueError('Expected exactly two gold supports')
    if len(results) != len(queries):
        raise ValueError('Wrong result count')
    seen, cases = set(), []
    ideal = 1 + 1 / math.log2(3)
    for row in results:
        identifier = row['query_id']
        if identifier not in expected or identifier in seen or row['query'] != expected[identifier]:
            raise ValueError('Duplicate, unknown or mismatched query')
        seen.add(identifier)
        ids = [doc['doc_index'] for doc in row['retrieved']]
        if len(ids) != 10 or len(set(ids)) != 10:
            raise ValueError('Expected ten distinct ranked documents')
        if any(type(index) is not int or not 0 <= index < corpus_count for index in ids):
            raise ValueError('Invalid document ID')
        supports = targets[identifier]
        ranks = [rank for rank, index in enumerate(ids, 1) if index in supports]
        cases.append(dict(id=identifier, type=types[identifier], support_ranks=ranks,
                          ndcg_at10=sum(1 / math.log2(rank + 1) for rank in ranks) / ideal,
                          both_supports_at2=int(supports <= set(ids[:2]))))

    def aggregate(selected):
        return dict(count=len(selected), **{key: sum(case[key] for case in selected) / len(selected)
                                          for key in ('ndcg_at10', 'both_supports_at2')})

    return dict(**aggregate(cases), cases=cases,
                slices={kind: aggregate([case for case in cases if case['type'] == kind])
                        for kind in sorted({case['type'] for case in cases})})
