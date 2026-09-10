import torch


def decode_selected(cache, indices):
    page_size = cache.shape[1]
    pages = indices.long() // page_size
    slots = indices.long() % page_size
    raw = cache.view(torch.uint8).view(cache.shape[0], -1)
    records = raw[:, :page_size * 576].view(-1, page_size, 576)
    selected = records[pages, slots].contiguous()
    scales = raw[:, page_size * 576:].view(-1, page_size, 8)[pages, slots, :7]
    scale_values = torch.exp2(scales.float() - 127)
    scale_values[scales == 255] = float('nan')
    nope = selected[:, :448].contiguous().view(torch.float8_e4m3fn).float()
    nope = (nope.view(-1, 7, 64) * scale_values.unsqueeze(-1)).reshape(-1, 448)
    rope = selected[:, 448:].contiguous().view(torch.bfloat16).float()
    return torch.cat([nope, rope], dim=-1)


@torch.inference_mode()
def reference(inputs):
    query = inputs['q']
    batch, queries, heads, _ = query.shape
    output = torch.zeros_like(query)
    lse = torch.full((batch, heads, queries), float('inf'), device=query.device, dtype=torch.float32)
    for batch_id in range(batch):
        for row in range(queries):
            gathered = []
            for scope in inputs['scopes']:
                indices = scope['indices'][batch_id, row]
                if scope['length'] is not None:
                    indices = indices[:int(scope['length'][batch_id])]
                indices = indices[indices >= 0]
                if indices.numel():
                    gathered.append(decode_selected(scope['cache'], indices))
            if not gathered:
                continue
            values = torch.cat(gathered)
            if not torch.isfinite(values).all():
                raise AssertionError('Input construction poisoned a valid selected token')
            logits = query[batch_id, row].float() @ values.T * (512 ** -0.55)
            row_lse = torch.logsumexp(logits, dim=-1)
            weights = torch.exp(logits - row_lse[:, None])
            result = weights @ values
            if inputs['sink'] is not None:
                result *= torch.sigmoid(row_lse - inputs['sink'])[:, None]
            output[batch_id, row] = result.to(torch.bfloat16)
            lse[batch_id, :, row] = row_lse
    return output, lse
