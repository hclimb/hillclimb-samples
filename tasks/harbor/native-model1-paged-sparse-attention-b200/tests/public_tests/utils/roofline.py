"""A declared dense-BF16 roofline estimate, not an attainable or universal ceiling."""

from utils.protocol import TIMED_CALLS


MODEL = dict(name='estimated_dense_bf16_roofline', dense_bf16_flops_per_second=2.25e15,
             hbm_bytes_per_second=8e12)
# Dense, not sparse, BF16: https://github.com/NVIDIA/dgxc-benchmarking#gpu-specifications
# HBM: https://www.nvidia.com/en-us/data-center/hgx/ (64 TB/s across eight B200s).
# Decimal SI units. Rates are declared constants, never fitted to a submission.


def estimate(case):
    batch, queries, heads, dimension = case['batch'], case['queries'], case['heads'], 512
    selected = sum(scope['topk'] for scope in case['scopes'])
    if min(batch, queries, heads, selected) <= 0:
        raise ValueError('Roofline dimensions must be positive')
    if any(any(count != scope['topk'] for count in row)
           for scope in case['scopes'] for row in scope['valid_counts']):
        raise ValueError('Roofline workload requires fully populated selections')
    tokens = batch * queries
    flops = 4 * tokens * heads * selected * dimension  # QK and PV; multiply-add = 2.
    # Optimistic reuse across heads, query rows and the entire measured block.
    # Sliding windows add Q-1 unique entries; sparse rows can actually select more
    # distinct tokens than this minimum union. This is a model, not measured traffic.
    minimum_union = sum(scope['topk'] + (queries - 1 if case['window'] and index == 0 else 0)
                        for index, scope in enumerate(case['scopes']))
    kv_bytes_per_block = batch * minimum_union * 584
    query_output_bytes = tokens * heads * dimension * 4  # BF16 query read + output write.
    lse_bytes = tokens * heads * 4
    index_bytes = tokens * selected * 4
    sink_bytes = heads * 4 if case['sinks'] else 0
    bytes_per_call = (query_output_bytes + lse_bytes + index_bytes + sink_bytes +
                      kv_bytes_per_block / TIMED_CALLS)
    compute_seconds = flops / MODEL['dense_bf16_flops_per_second']
    memory_seconds = bytes_per_call / MODEL['hbm_bytes_per_second']
    ideal_seconds = max(compute_seconds, memory_seconds)
    return dict(**MODEL, attention_flops=flops, modeled_bytes_per_call=bytes_per_call,
                minimum_kv_bytes_per_block=kv_bytes_per_block,
                compute_seconds=compute_seconds, memory_seconds=memory_seconds,
                ideal_seconds=ideal_seconds, query_tokens_per_second=tokens / ideal_seconds,
                limiting_resource='compute' if compute_seconds >= memory_seconds else 'memory',
                assumptions='Dense BF16 QK/PV; ideal reuse; excludes softmax, dequantization, '
                            'scheduling, padding and snapshot overhead. Not a universal bound; '
                            'valid lower-precision implementations may exceed it.')
