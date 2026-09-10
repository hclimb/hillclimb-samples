from typing import List
import torch

def _randperm_batch(batch_size: int, perm_range: torch.Tensor, perm_size: int, paddings: List[int]) -> torch.Tensor:
    """
    Generate random permutations in batch
    The return tensor, denoted as `res`, has a shape of [batch_size, perm_size]. `0 <= res[i, :] < perm_range[i]` holds.
    Values within each row are unique.
    If, for some `i`, `perm_range[i] < perm_size` holds, then `res[i, :]` contains values in `[0, perm_range[i])` as many as possible, and the rest are filled with `padding`.
    """
    assert not torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    perm_range_max = max(int(torch.max(perm_range).item()), perm_size)
    rand = torch.rand(batch_size, perm_range_max, dtype=torch.float32)
    rand[torch.arange(0, perm_range_max).broadcast_to(batch_size, perm_range_max) >= perm_range.view(batch_size, 1)] = float("-inf")    # Fill invalid positions, so that the following `topk` operators will select positions within `perm_range` first
    res = rand.topk(perm_size, dim=-1, sorted=True).indices.to(torch.int32)
    if len(paddings) == 1:
        res[res >= perm_range.view(batch_size, 1)] = paddings[0]
    else:
        fillers = torch.tensor(paddings, dtype=torch.int32).index_select(0, torch.randint(0, len(paddings), (res.numel(), ), dtype=torch.int32))
        res.masked_scatter_(res >= perm_range.view(batch_size, 1), fillers)
    torch.use_deterministic_algorithms(False)
    return res
