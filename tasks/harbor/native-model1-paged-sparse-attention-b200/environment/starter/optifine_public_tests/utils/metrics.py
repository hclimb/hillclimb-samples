import math
import statistics

from utils.protocol import REPETITIONS, TIMED_CALLS, TIMED_OUTPUT_CHECKS, WARMUP_CALLS
from utils.roofline import estimate


def require_numerical(report):
    if set(report) != {'output', 'lse'} or any(item.get('passed') is not True for item in report.values()):
        raise ValueError('Missing or failed output/LSE correctness evidence')


def measure(reports, cases, diagnostic_only=False):
    names = {case['name'] for case in cases}
    if not names or len(names) != len(cases):
        raise ValueError('Expected nonempty, unique correctness cases')
    if len(reports) != REPETITIONS:
        raise ValueError(f'Expected {REPETITIONS} successful runs')
    scored = [case for case in cases if case['throughput_weight'] > 0]
    if len(scored) != 1 and not (diagnostic_only and not scored):
        raise ValueError('Expected one fixed scored workload')
    for run in reports:
        if set(run['cases']) != names:
            raise ValueError('Every run must cover every requested correctness case')
        for case in cases:
            item = run['cases'][case['name']]
            require_numerical(item['numerical'])
            require_numerical(item['same_storage'])
    if not scored:
        return dict(correctness_passed=True, diagnostic_only=True)
    case = scored[0]
    measurements = [run['cases'][case['name']] for run in reports]
    for item in measurements:
        if (item.get('timed_calls') != TIMED_CALLS or item.get('warmup_calls') != WARMUP_CALLS or
                item.get('distinct_queries') != TIMED_CALLS or item.get('snapshot_copies') != TIMED_OUTPUT_CHECKS):
            raise ValueError('Expected warmed fresh-query throughput blocks')
        checks = item.get('timed_output_checks', {})
        if len(checks) != TIMED_OUTPUT_CHECKS or str(TIMED_CALLS - 1) not in checks:
            raise ValueError('Missing sampled timed-output correctness')
        if any(not key.isdigit() or not 0 <= int(key) < TIMED_CALLS for key in checks):
            raise ValueError('Invalid timed-output sample index')
        for check in checks.values():
            require_numerical(check)
        seconds = item['seconds']
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Expected positive finite timings')
        if not math.isclose(item.get('block_seconds', 0), seconds * TIMED_CALLS, rel_tol=1e-9):
            raise ValueError('Block timing does not match per-call timing')
    seconds = statistics.median(item['seconds'] for item in measurements)
    roofline = estimate(case)
    rate = case['batch'] * case['queries'] / seconds
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('Expected positive finite throughput')
    efficiency = roofline['ideal_seconds'] / seconds
    return dict(median_seconds=seconds, rate=rate, scored_case=case['name'],
                roofline=roofline, estimated_roofline_efficiency=efficiency,
                exceeds_estimated_roofline=efficiency > 1,
                correctness_passed=True, diagnostic_only=diagnostic_only)
