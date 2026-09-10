import jax
import jax.numpy as jnp
from .registry import register_aux_loss


@register_aux_loss("doc_access_per_query_loss")
def compute_doc_access_per_query_loss(aux_data, mask, input_mask, inputs, temperature=1.0, **kwargs):
    """
    doc_access_loss's exact contrastive objective (log_z - log_pos, i.e. softmax cross-entropy
    against a uniform-over-positives target), restricted to a query's OWN m_per_query slots —
    for `mem_batched_isolation` (models/memory.py::mem_lookup_batched), where a query
    structurally can't see another query's docs at all.

    Consumes `aux_data["mem_scores"]` populated by `mem_lookup_batched` when
    `cfg.mem_collect_full_scores=True`: the FULL per-row score grid, [B, T, N, m_per_query],
    BEFORE top-k truncation. This is the same [B,T,N,·] shape family `doc_access_loss` consumes,
    just `m_per_query` instead of the batch-global `B*m_per_query` in the last axis — that's what
    makes exposing it affordable here (same order as the retrieval kernel's own peak) where it
    never was for the cross-batch case (that's the whole point of `mem_batched_isolation`).

    Unlike `doc_access_loss`, there is no cross-query positive-mask construction (no `block_eye`):
    a query's own slots can only ever contain its own docs under per-row isolation, so the "no
    cross-query contamination" guarantee `doc_access_loss` builds via block_eye is structural here,
    not something this loss has to enforce.

    Dtype: deliberately does no fp32 upcasting of the score tensor — `mem_scores` arrives in
    whatever dtype the memory layer computes in (bf16), and `jax.nn.logsumexp`'s max-subtraction
    is numerically stable regardless. Only the scalar loss accumulator is fp32, matching
    `doc_access_loss`'s own convention (a scalar has no memory cost either way).
    """
    mem_scores_list = aux_data.get("mem_scores")
    if not mem_scores_list or len(mem_scores_list) == 0:
        return 0.0

    if input_mask is None or "pos_doc_mask" not in input_mask:
        raise ValueError(
            "doc_access_per_query_loss requires input_mask['pos_doc_mask'] [B, docs_per_query] "
            "-- there is no batch-global fallback (unlike doc_access_loss's diagonal fallback), "
            "since per-query isolation has no meaningful 'assume doc i is query i's positive'.")
    pos_mask_local = input_mask["pos_doc_mask"]  # [B, docs_per_query]
    docs_per_query = pos_mask_local.shape[1]

    loss_mask = mask  # [B, T]
    total_loss = 0.0

    for scores_item in mem_scores_list:
        logits = scores_item[0] if isinstance(scores_item, tuple) else scores_item  # [B,T,N,M]
        B, T, N, M = logits.shape
        if M % docs_per_query != 0:
            raise ValueError(f"doc_access_per_query_loss: m_per_query={M} not divisible by "
                              f"docs_per_query={docs_per_query}")
        doc_len = M // docs_per_query

        # Per-row validity: effective_mem_mask (set once, globally, in qwen3_mem_embed.py::forward)
        # is the flat [B*m_per_query] post-conv mask; reshape to this loss's [B, docs_per_query,
        # doc_len]. Fall back to the pre-conv docs_mask (same per-row layout, just raw_doc_len)
        # if the post-conv mask wasn't threaded through for some reason.
        effective_mem_mask = aux_data.get("effective_mem_mask", None)
        if effective_mem_mask is not None:
            valid = (effective_mem_mask.reshape(B, docs_per_query, doc_len) > 0)
        else:
            docs_mask = input_mask["docs_mask"]  # [B*docs_per_query, raw_doc_len]
            valid = (docs_mask.reshape(B, docs_per_query, -1) > 0)
            doc_len = valid.shape[-1]

        pos_valid = (pos_mask_local[:, :, None] > 0) & valid  # [B, docs_per_query, doc_len]

        scaled_logits = (logits / temperature).reshape(B, T, N, docs_per_query, doc_len)

        # T-first layout + scan, exactly like doc_access_loss: avoids materializing the full
        # (B,T,N,docs_per_query,doc_len) tensor at once.
        logits_T_first = jnp.moveaxis(scaled_logits, 1, 0)          # [T, B, N, docs_per_query, doc_len]
        loss_mask_T_first = jnp.moveaxis(loss_mask, 1, 0)            # [T, B]

        # Insert the head (N) axis at position 1 -- valid/pos_valid have no N dependence, so this
        # is the broadcast [B, docs_per_query, doc_len] -> [B, 1, docs_per_query, doc_len] needs
        # to align against logits_t's [B, N, docs_per_query, doc_len] (trailing-align would
        # mismatch docs_per_query against N instead).
        valid_b = valid[:, None, :, :]
        pos_valid_b = pos_valid[:, None, :, :]

        def scan_body(carry, x):
            logits_t, loss_mask_t = x  # [B, N, docs_per_query, doc_len], [B]
            log_z_t = jax.nn.logsumexp(
                logits_t, axis=(-2, -1), where=valid_b,
            )  # [B, N]
            log_pos_t = jax.nn.logsumexp(
                logits_t, axis=(-2, -1), where=pos_valid_b,
            )  # [B, N]
            log_z_t = jnp.where(jnp.isfinite(log_z_t), log_z_t, 0.0)
            log_pos_t = jnp.where(jnp.isfinite(log_pos_t), log_pos_t, 0.0)
            layer_loss_t = log_z_t - log_pos_t  # [B, N], >= 0
            carry = carry + jnp.sum(layer_loss_t.astype(jnp.float32) * loss_mask_t[:, None])
            return carry, None

        loss_sum, _ = jax.lax.scan(
            scan_body,
            init=jnp.zeros((), dtype=jnp.float32),
            xs=(logits_T_first, loss_mask_T_first),
        )

        total_loss = total_loss + loss_sum / (jnp.sum(loss_mask) * N + 1e-9)

    return total_loss / len(mem_scores_list)
