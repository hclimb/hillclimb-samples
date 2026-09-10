import jax
import jax.numpy as jnp
from .registry import register_aux_loss


@register_aux_loss("msa_route_loss")
def compute_msa_route_loss(aux_data, mask, input_mask, inputs, temperature=0.1, **kwargs):
    """MSA decoupled supervised-contrastive routing loss (paper Eq 5, msa.pdf §3.3.1).

    For each query, each router layer produces a doc-level score s_m over the query's
    own M doc-chunk slots. With positives P and negatives N = (valid slots) \\ P:

        L = -(1/|P|) * sum_{i in P} log( exp(s_i/τ) / (exp(s_i/τ) + sum_{j in N} exp(s_j/τ)) )
          =  (1/|P|) * sum_{i in P} softplus( logsumexp_{j in N}(s_j/τ) - s_i/τ )

    Averaged over queries (with >=1 positive) and over router layers. Negatives are the
    query's OWN hard negatives (not global in-batch) — faithful to the paper's D = the
    query's associated document set.
    """
    scores_list = aux_data.get("route_scores")
    if not scores_list:
        return 0.0

    pos = input_mask["pos_doc_mask"].astype(bool)          # [B, M]
    slot_valid = aux_data.get("slot_valid")
    slot_valid = jnp.ones_like(pos) if slot_valid is None else slot_valid.astype(bool)
    pos_m = pos & slot_valid                                # positives (valid)
    neg_m = (~pos) & slot_valid                             # negatives (valid)
    tau = temperature
    NEG = -1e9   # finite sentinel for masked-out slots (avoids -inf grad from where=)

    total = 0.0
    for s in scores_list:
        s = s.astype(jnp.float32) / tau                     # [B, M]
        # logsumexp over the query's own negatives; masked slots set to a finite
        # large-negative so a query with no negatives gives a finite (~NEG) result.
        s_neg = jnp.where(neg_m, s, NEG)
        lse_neg = jax.nn.logsumexp(s_neg, axis=-1)          # [B]
        # jnp.where (NOT multiply) to mask: a fully-invalid slot can carry s=-inf →
        # softplus(+inf)=+inf, and inf*0 would be NaN (and NaN grad). where() selects
        # the 0 branch for non-positive slots so neither value nor gradient sees inf.
        term = jnp.where(pos_m, jax.nn.softplus(lse_neg[:, None] - s), 0.0)   # [B, M]
        n_pos = pos_m.sum(axis=-1)                          # [B]
        per_q = term.sum(axis=-1) / jnp.clip(n_pos, 1.0)    # [B]
        has_pos = n_pos > 0
        total += (per_q * has_pos).sum() / (has_pos.sum() + 1e-9)

    return total / len(scores_list)
