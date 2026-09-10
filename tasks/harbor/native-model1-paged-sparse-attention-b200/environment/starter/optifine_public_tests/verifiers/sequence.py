"""Fresh-query timing; all generation/reference work precedes warmup and measurement."""

import random
import time

import torch

from utils.native_generate import non_contiguousify
from utils.numerics import compare
from utils.protocol import TIMED_CALLS, TIMED_OUTPUT_CHECKS, WARMUP_CALLS
from utils.workload import call_native
from verifiers.reference import reference


def query_sequence(query, seed, count):
    # Local RNG: candidate torch.manual_seed() calls cannot determine future queries.
    generator = torch.Generator(device=query.device).manual_seed(seed)
    return [non_contiguousify(torch.randn(query.shape, dtype=query.dtype, device=query.device,
                                         generator=generator).clamp_(-1, 1)) for _ in range(count)]


def time_sequence(implementation, calls, scheduler, snapshots):
    torch.cuda.synchronize()
    started = time.perf_counter()
    for index, inputs in enumerate(calls):
        answer = call_native(implementation, inputs, scheduler)
        if index in snapshots:
            output, lse = answer
            target_output, target_lse = snapshots[index]
            if (output.shape != target_output.shape or output.dtype != target_output.dtype or
                    lse.shape != target_lse.shape or lse.dtype != target_lse.dtype):
                raise AssertionError('Timed output shape/dtype mismatch')
            # Charge these three preallocated copies. This also supports kernels
            # that legitimately reuse output buffers between invocations.
            target_output.copy_(output)
            target_lse.copy_(lse)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / len(calls)


def benchmark_sequence(implementation, inputs, preserved, scheduler, seed, tolerances):
    queries = query_sequence(inputs['q'], seed + 900001, WARMUP_CALLS + TIMED_CALLS)
    calls = [dict(inputs, q=query) for query in queries]
    check_indices = sorted(random.Random(seed).sample(range(TIMED_CALLS - 1), TIMED_OUTPUT_CHECKS - 1)
                           + [TIMED_CALLS - 1])
    expected = {index: reference(dict(preserved, q=queries[WARMUP_CALLS + index].clone()))
                for index in check_indices}
    snapshots = {index: tuple(torch.empty_like(value) for value in answer)
                 for index, answer in expected.items()}
    for inputs in calls[:WARMUP_CALLS]:
        call_native(implementation, inputs, scheduler)
    seconds = time_sequence(implementation, calls[WARMUP_CALLS:], scheduler, snapshots)
    numerical = {str(index): compare(snapshots[index], expected[index], tolerances)
                 for index in check_indices}
    return dict(seconds=seconds, block_seconds=seconds * TIMED_CALLS,
                timed_calls=TIMED_CALLS, warmup_calls=WARMUP_CALLS,
                distinct_queries=TIMED_CALLS, timed_output_checks=numerical,
                snapshot_copies=TIMED_OUTPUT_CHECKS,
                query_tokens_per_second=inputs['q'].shape[0] * inputs['q'].shape[1] / seconds)
