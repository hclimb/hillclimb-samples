import torch

from utils.native_generate import gen_non_contiguous_randn_tensor, non_contiguousify
from utils.native_quant import FP8KVCacheLayout, abs_indices2indices_in_kvcache, quantize_k_cache
from utils.native_sampling import _randperm_batch


def logical_indices(case, scope, scope_id):
    batch, queries, topk = case['batch'], case['queries'], scope['topk']
    lengths = torch.tensor(scope['lengths'], dtype=torch.int32)
    if case['window'] and scope_id == 0:
        indices = torch.full((batch, queries, topk), -1, dtype=torch.int32)
        for batch_id, length in enumerate(scope['lengths']):
            for row in range(queries):
                end = length - queries + row + 1
                start = max(0, end - topk)
                indices[batch_id, row, :end - start] = torch.arange(start, end)
    else:
        indices = _randperm_batch(batch * queries, lengths.repeat_interleave(queries),
                                  topk, [-1]).view(batch, queries, topk)
    for batch_id, row in scope['padded_rows']:
        indices[batch_id, row].fill_(-1)
    return indices


def build_inputs(case, realization_seed, value_seed, device='cuda'):
    torch.manual_seed(value_seed)
    query = torch.randn((case['batch'], case['queries'], case['heads'], 512),
                        dtype=torch.bfloat16, device=device).clamp_(-1, 1)
    query = non_contiguousify(query)
    sink = None
    if case['sinks']:
        sink = torch.randn(case['heads'], dtype=torch.float32, device=device)
        categories = torch.tensor(case['sink_categories'], device=device)
        sink[categories < 0] = -float('inf')
        sink[categories > 0] = float('inf')
    inputs = dict(q=query, sink=sink, scopes=[])
    for scope_id, scope in enumerate(case['scopes']):
        torch.manual_seed(realization_seed + scope_id * 997)
        page, pages = scope['page'], scope['pages']
        block_table = torch.randperm(pages, dtype=torch.int64).to(torch.int32).view(case['batch'], -1)
        indices = abs_indices2indices_in_kvcache(logical_indices(case, scope, scope_id), block_table, page)
        valid = indices >= 0
        length = None
        if scope['topk_lengths'] is not None:
            length = torch.tensor(scope['topk_lengths'], dtype=torch.int32)
            valid &= torch.arange(scope['topk']).view(1, 1, -1) < length.view(-1, 1, 1)
        if valid.sum(-1).tolist() != scope['valid_counts']:
            raise AssertionError('Frozen work profile drift')
        used = torch.zeros(pages * page, dtype=torch.bool)
        used[indices[valid].long()] = True
        zero_token = int(indices[1, 0, 0]) if not case['scored'] else -1
        torch.manual_seed(value_seed + scope_id * 1237 + 13)
        storage = torch.empty((pages, scope['page_stride']), dtype=torch.float8_e4m3fn, device=device)
        cache = storage[:, :page * 584].view(pages, page, 1, 584)
        for first_page in range(0, pages, 1024):
            stop_page = min(first_page + 1024, pages)
            values = gen_non_contiguous_randn_tensor((stop_page - first_page, page, 1, 512),
                                                     dtype=torch.bfloat16, device=device) / 10
            values.clamp_(-1, 1)
            keep = used[first_page * page:stop_page * page].to(device).view(-1, page)
            values[~keep] = float('nan')
            if first_page * page <= zero_token < stop_page * page:
                values[zero_token // page - first_page, zero_token % page].zero_()
            packed = quantize_k_cache(values, FP8KVCacheLayout.MODEL1_FP8Sparse)
            cache[first_page:stop_page].copy_(packed)
        inputs['scopes'].append(dict(cache=cache, indices=non_contiguousify(indices.to(device)),
                                     length=length.to(device) if length is not None else None))
    return inputs


def replace_values_in_place(inputs, replacement):
    inputs['q'].copy_(replacement['q'])
    if inputs['sink'] is not None:
        inputs['sink'].copy_(replacement['sink'])
    for original, updated in zip(inputs['scopes'], replacement['scopes']):
        if not torch.equal(original['indices'], updated['indices']):
            raise AssertionError('Same-storage regression changed scheduling invariants')
        original['cache'].copy_(updated['cache'])


def preserve_inputs(inputs):
    return dict(q=inputs['q'].clone(), sink=inputs['sink'].clone() if inputs['sink'] is not None else None,
                scopes=[{name: value.clone() if value is not None else None
                         for name, value in scope.items()} for scope in inputs['scopes']])


def call_native(implementation, inputs, scheduler):
    primary = inputs['scopes'][0]
    extra = inputs['scopes'][1] if len(inputs['scopes']) == 2 else None
    return implementation.flash_mla_with_kvcache(
        inputs['q'], primary['cache'], None, None, 512, scheduler, None,
        512 ** -0.55, False, True, primary['indices'], inputs['sink'],
        extra['cache'] if extra else None, extra['indices'] if extra else None,
        primary['length'], extra['length'] if extra else None)
