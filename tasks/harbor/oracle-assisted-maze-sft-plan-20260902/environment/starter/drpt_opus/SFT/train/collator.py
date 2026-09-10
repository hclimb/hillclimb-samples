"""Collator that carries a per-example source index alongside model inputs.

`meta_idx` is the row's position in the raw training pool. It must never reach
`model(**batch)`, so this wrapper keeps it out of the delegated collation and
re-attaches it afterwards; the trainer pops it before the forward pass.

This is what makes per-domain selection rates possible: the curation strategies
report which *in-batch* positions they selected, and `meta_idx` maps those back
to `{id, dataset, domain}` in the pool.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

META_IDX_KEY = "meta_idx"


class MetaIdxCollator:
    """Wrap any collator, passing `meta_idx` around it instead of through it."""

    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        meta_indices = []
        stripped = []
        for feature in features:
            if META_IDX_KEY in feature:
                meta_indices.append(int(feature[META_IDX_KEY]))
                feature = {k: v for k, v in feature.items() if k != META_IDX_KEY}
            stripped.append(feature)

        batch = self.base_collator(stripped)
        if meta_indices and len(meta_indices) == len(features):
            batch[META_IDX_KEY] = torch.tensor(meta_indices, dtype=torch.long)
        return batch

    def __getattr__(self, name):
        # Trainer inspects collator attributes (e.g. `tokenizer`) in places;
        # forward anything we don't define to the wrapped collator.
        return getattr(self.base_collator, name)
