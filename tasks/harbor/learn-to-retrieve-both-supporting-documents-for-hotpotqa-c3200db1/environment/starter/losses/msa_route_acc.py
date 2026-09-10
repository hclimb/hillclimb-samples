import jax.numpy as jnp
from .registry import register_aux_loss


@register_aux_loss("msa_route_acc")
def compute_msa_route_acc(aux_data, mask, input_mask, inputs, **kwargs):
    """Routing-separation metric (weight 0 — logging only).

    Fraction of positive doc slots whose routing score exceeds the max score over the
    query's valid negative slots, averaged over router layers. 1.0 ⇒ every positive is
    ranked above all of its hard negatives.
    """
    scores_list = aux_data.get("route_scores")
    if not scores_list:
        return 0.0

    pos = input_mask["pos_doc_mask"].astype(bool)          # [B, M]
    slot_valid = aux_data.get("slot_valid")
    slot_valid = jnp.ones_like(pos) if slot_valid is None else slot_valid.astype(bool)
    pos_m = pos & slot_valid
    neg_m = (~pos) & slot_valid
    neg_big = jnp.finfo(jnp.float32).min

    total_corr, total = 0.0, 0.0
    for s in scores_list:
        s = s.astype(jnp.float32)
        max_neg = jnp.where(neg_m, s, neg_big).max(axis=-1, keepdims=True)   # [B,1]
        correct = (s > max_neg) & pos_m
        total_corr += correct.sum()
        total += pos_m.sum()

    return total_corr / (total + 1e-9)
