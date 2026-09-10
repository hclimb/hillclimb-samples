"""
Scaling test for encoder-gradient memory under two qwen3_mem_embed-style strategies:

  1. Baseline full-grad embed:
     - Encode every document with gradients enabled.
     - Run two-pass retrieval over the full memory bank.

  2. Restricted re-embed:
     - Encode every document once under stop_gradient for pass-1 selection.
     - Re-encode only the union of selected docs and positive docs with gradients enabled.
     - Run pass-2 retrieval only on that compact differentiable bank.

This test is designed to answer a specific asymptotic question:
does the modeled end-to-end encoder path become sub-O(M), or is it merely a smaller-O(M)
than the baseline because the no-grad full-doc pass still scales with total memory-bank size M?

The synthetic batch is constructed so the selected-doc union stays essentially fixed
as M grows: only one positive doc per query carries signal; all other docs are low-norm
noise. That makes it possible to observe whether the gradient-bearing encoder graph is
truly sub-O(M).

Running:
    env UV_CACHE_DIR=/tmp/uv-cache JAX_PLATFORMS=cpu \
        uv run python tests/test_restricted_reembed_scaling.py
"""

import os
import sys
from dataclasses import dataclass

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_LOG_COMPILES", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from models.memory import mem_lookup_chunked, mem_lookup_two_pass
from models.qwen3 import rms_norm


@dataclass(frozen=True)
class BenchCfg:
    batch_size: int = 4
    seq_len: int = 32
    doc_len: int = 64
    input_dim: int = 128
    hidden_dim: int = 192
    mem_heads: int = 2
    mem_k_dim: int = 64
    mem_v_dim: int = 64
    mem_top_k: int = 8
    mem_lookup_chunk_size: int = 256
    depth: int = 3

    def memory_cfg(self):
        return {
            "rms_norm_eps": 1e-6,
            "mem_top_k": self.mem_top_k,
            "mem_lookup_chunk_size": self.mem_lookup_chunk_size,
            "mem_k_prenormed": False,
            "two_pass_topk": True,
        }


def make_params(key, cfg: BenchCfg):
    keys = jax.random.split(key, cfg.depth + 3)
    return {
        "in_proj": jax.random.normal(keys[0], (cfg.input_dim, cfg.hidden_dim), dtype=jnp.float32) * 0.05,
        "layers": [
            jax.random.normal(keys[i + 1], (cfg.hidden_dim, cfg.hidden_dim), dtype=jnp.float32) * 0.05
            for i in range(cfg.depth)
        ],
        "k_proj": jax.random.normal(keys[-2], (cfg.hidden_dim, cfg.mem_k_dim), dtype=jnp.float32) * 0.05,
        "v_proj": jax.random.normal(keys[-1], (cfg.hidden_dim, cfg.mem_v_dim), dtype=jnp.float32) * 0.05,
    }


def embed_docs(params, docs):
    x = jnp.einsum("dli,ih->dlh", docs, params["in_proj"])
    x = jax.nn.gelu(x)
    for w in params["layers"]:
        x = jnp.einsum("dli,ih->dlh", x, w)
        x = jax.nn.gelu(x)
    mem_k = jnp.einsum("dli,ih->dlh", x, params["k_proj"])
    mem_v = jnp.einsum("dli,iv->dlv", x, params["v_proj"])
    return mem_k.reshape(-1, mem_k.shape[-1]), mem_v.reshape(-1, mem_v.shape[-1])


def make_positive_slot_indices(batch_size: int, docs_per_query: int, doc_len: int):
    pos_doc_ids = jnp.arange(batch_size, dtype=jnp.int32) * docs_per_query
    return pos_doc_ids[:, None] * doc_len + jnp.arange(doc_len, dtype=jnp.int32)[None, :]


def make_inputs(cfg: BenchCfg, docs_per_query: int):
    total_docs = cfg.batch_size * docs_per_query
    key = jax.random.PRNGKey(0)
    docs_key, params_key, pos_docs_key, noise_key = jax.random.split(key, 4)

    # Low-norm distractors; only the designated positive doc per query carries signal.
    docs = 0.001 * jax.random.normal(
        docs_key,
        (total_docs, cfg.doc_len, cfg.input_dim),
        dtype=jnp.float32,
    )
    pos_doc_ids = jnp.arange(cfg.batch_size, dtype=jnp.int32) * docs_per_query
    pos_docs = jax.random.normal(
        pos_docs_key,
        (cfg.batch_size, cfg.doc_len, cfg.input_dim),
        dtype=jnp.float32,
    )
    docs = docs.at[pos_doc_ids].set(pos_docs)

    params = make_params(params_key, cfg)
    mem_k0, _ = embed_docs(params, docs)
    mem_k0 = mem_k0.reshape(total_docs, cfg.doc_len, cfg.mem_k_dim)

    tok_idx = jnp.arange(cfg.seq_len, dtype=jnp.int32) % cfg.doc_len
    q_base = mem_k0[pos_doc_ids[:, None], tok_idx[None, :]]
    q_base = jnp.broadcast_to(q_base[:, :, None, :], (cfg.batch_size, cfg.seq_len, cfg.mem_heads, cfg.mem_k_dim))
    q = q_base + 0.01 * jax.random.normal(noise_key, q_base.shape, dtype=jnp.float32)

    docs_mask = jnp.ones((total_docs, cfg.doc_len), dtype=jnp.bool_)
    loss_mask = jnp.ones((cfg.batch_size, cfg.seq_len), dtype=jnp.bool_)
    pos_slot_indices = make_positive_slot_indices(cfg.batch_size, docs_per_query, cfg.doc_len)
    return docs, params, q, docs_mask, loss_mask, pos_slot_indices


def retrieval_loss(top_k_scores, mem_v_k):
    out = jnp.einsum("bntk,bntkd->bntd", top_k_scores, mem_v_k)
    return jnp.mean(out.astype(jnp.float32) ** 2)


def topk_positive_contrastive_loss(top_k_logits, pos_logits, loss_mask):
    pool_logits = jnp.concatenate([top_k_logits, pos_logits], axis=-1)
    log_z = jax.nn.logsumexp(pool_logits, axis=-1)
    log_pos = jax.nn.logsumexp(pos_logits, axis=-1)
    per_token = (log_z - log_pos).astype(jnp.float32)
    return jnp.sum(per_token * loss_mask[:, None, :]) / (
        jnp.sum(loss_mask) * top_k_logits.shape[1] + 1e-9
    )


def select_doc_ids(top_k_indices, pos_slot_indices, total_docs: int, doc_len: int, selected_doc_cap: int):
    top_doc_ids = (top_k_indices.reshape(-1) // doc_len).astype(jnp.int32)
    pos_doc_ids = (pos_slot_indices.reshape(-1) // doc_len).astype(jnp.int32)
    selected_mask = jnp.zeros((total_docs,), dtype=jnp.int32)
    selected_mask = selected_mask.at[top_doc_ids].set(1)
    selected_mask = selected_mask.at[pos_doc_ids].set(1)
    _, selected_doc_ids = jax.lax.top_k(selected_mask, selected_doc_cap)
    return selected_doc_ids, selected_mask


def remap_slot_indices(global_slot_indices, selected_doc_ids, total_docs: int, doc_len: int):
    local_doc_map = -jnp.ones((total_docs,), dtype=jnp.int32)
    local_doc_map = local_doc_map.at[selected_doc_ids].set(jnp.arange(selected_doc_ids.shape[0], dtype=jnp.int32))
    global_doc_ids = (global_slot_indices // doc_len).astype(jnp.int32)
    slot_offsets = (global_slot_indices % doc_len).astype(jnp.int32)
    local_doc_ids = local_doc_map[global_doc_ids]
    return local_doc_ids * doc_len + slot_offsets


def fixed_indices_pass2(q, mem_k, mem_v, mem_k_norm, top_k_indices_local, pos_slot_indices_local, eps: float):
    mem_k_normed = rms_norm(mem_k, mem_k_norm, eps)
    mem_k_k = mem_k_normed[top_k_indices_local]
    mem_v_k = mem_v[top_k_indices_local]
    q_t = jnp.transpose(q, (0, 2, 1, 3))
    scale = jnp.sqrt(jnp.array(q.shape[-1], dtype=jnp.float32))
    top_k_logits = jnp.einsum("bntd,bntkd->bntk", q_t, mem_k_k) / scale
    top_k_scores = jax.nn.softmax(top_k_logits, axis=-1).astype(q.dtype)
    mem_k_pos = mem_k_normed[pos_slot_indices_local]
    pos_logits = jnp.einsum("bntd,bpd->bntp", q_t, mem_k_pos) / scale
    return top_k_scores, mem_v_k, top_k_logits, pos_logits


def estimate_selected_doc_count(params, docs, q, docs_mask, pos_slot_indices, cfg: BenchCfg, docs_per_query: int):
    total_docs = cfg.batch_size * docs_per_query
    mem_k, mem_v = embed_docs(params, docs)
    w = {
        "mem_k": mem_k,
        "mem_v": mem_v,
        "mem_k_norm": jnp.ones((cfg.mem_k_dim,), dtype=mem_k.dtype),
        "mem_mask": docs_mask.reshape(-1),
    }
    _, _, aux = mem_lookup_chunked(
        jax.lax.stop_gradient(q),
        jax.tree_util.tree_map(jax.lax.stop_gradient, w),
        cfg.memory_cfg(),
        collect_aux=True,
        keys_only=True,
    )
    selected_doc_ids, selected_mask = select_doc_ids(
        aux["mem_top_k_indices"], pos_slot_indices, total_docs, cfg.doc_len, total_docs
    )
    selected_count = int(jnp.sum(selected_mask))
    return selected_count, np.array(selected_doc_ids[:selected_count]).tolist()


def build_losses(docs, q, docs_mask, loss_mask, pos_slot_indices, cfg: BenchCfg, docs_per_query: int, selected_doc_cap: int):
    total_docs = cfg.batch_size * docs_per_query
    mem_cfg = cfg.memory_cfg()
    mem_k_norm = jnp.ones((cfg.mem_k_dim,), dtype=jnp.float32)
    effective_mem_mask = docs_mask.reshape(-1)

    def baseline_loss(params):
        mem_k, mem_v = embed_docs(params, docs)
        w = {
            "mem_k": mem_k,
            "mem_v": mem_v,
            "mem_k_norm": mem_k_norm,
            "mem_mask": effective_mem_mask,
        }
        top_k_scores, mem_v_k, aux = mem_lookup_two_pass(
            q, w, mem_cfg, collect_aux=True, pos_slot_indices=pos_slot_indices
        )
        return retrieval_loss(top_k_scores, mem_v_k) + topk_positive_contrastive_loss(
            aux["mem_top_k_logits"], aux["mem_pos_logits"], loss_mask
        )

    def restricted_loss(params):
        params_sg = jax.tree_util.tree_map(jax.lax.stop_gradient, params)
        mem_k_full_sg, mem_v_full_sg = embed_docs(params_sg, docs)
        w_full_sg = {
            "mem_k": mem_k_full_sg,
            "mem_v": mem_v_full_sg,
            "mem_k_norm": mem_k_norm,
            "mem_mask": effective_mem_mask,
        }
        _, _, pass1_aux = mem_lookup_chunked(
            jax.lax.stop_gradient(q),
            w_full_sg,
            mem_cfg,
            collect_aux=True,
            keys_only=True,
        )
        top_k_indices_global = jax.lax.stop_gradient(pass1_aux["mem_top_k_indices"])
        selected_doc_ids, _ = select_doc_ids(
            top_k_indices_global, pos_slot_indices, total_docs, cfg.doc_len, selected_doc_cap
        )
        selected_doc_ids = jax.lax.stop_gradient(selected_doc_ids)

        docs_selected = docs[selected_doc_ids]
        mem_k_sel, mem_v_sel = embed_docs(params, docs_selected)

        top_k_indices_local = remap_slot_indices(
            top_k_indices_global, selected_doc_ids, total_docs, cfg.doc_len
        )
        pos_slot_indices_local = remap_slot_indices(
            pos_slot_indices, selected_doc_ids, total_docs, cfg.doc_len
        )
        top_k_scores, mem_v_k, top_k_logits, pos_logits = fixed_indices_pass2(
            q, mem_k_sel, mem_v_sel, mem_k_norm, top_k_indices_local, pos_slot_indices_local, mem_cfg["rms_norm_eps"]
        )
        return retrieval_loss(top_k_scores, mem_v_k) + topk_positive_contrastive_loss(
            top_k_logits, pos_logits, loss_mask
        )

    return baseline_loss, restricted_loss


def cost_bytes(compiled) -> float:
    c = compiled.cost_analysis()
    if isinstance(c, list):
        return sum(x.get("bytes accessed", 0) for x in c)
    return c.get("bytes accessed", 0)


def memory_stats(compiled):
    if not hasattr(compiled, "memory_analysis"):
        return {}
    analysis = compiled.memory_analysis()
    if analysis is None:
        return {}
    stats = {}
    for name in ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes"):
        if hasattr(analysis, name):
            stats[name] = getattr(analysis, name)
    return stats


def test_encoder_reembed_scaling():
    cfg = BenchCfg()
    docs_per_query_values = [8, 16, 32, 64]

    baseline_costs = []
    restricted_costs = []
    baseline_temps = []
    restricted_temps = []
    total_slots = []
    selected_counts = []

    print("\nEncoder scaling:")
    print(f"{'docs/q':>6}  {'M slots':>8}  {'sel docs':>8}  {'base MB':>10}  {'restr MB':>10}  {'base tmp MB':>12}  {'restr tmp MB':>13}")
    print("-" * 86)

    for docs_per_query in docs_per_query_values:
        docs, params, q, docs_mask, loss_mask, pos_slot_indices = make_inputs(cfg, docs_per_query)
        selected_doc_cap, selected_doc_ids = estimate_selected_doc_count(
            params, docs, q, docs_mask, pos_slot_indices, cfg, docs_per_query
        )
        baseline_loss, restricted_loss = build_losses(
            docs, q, docs_mask, loss_mask, pos_slot_indices, cfg, docs_per_query, selected_doc_cap
        )

        compiled_baseline = jax.jit(jax.grad(baseline_loss)).lower(params).compile()
        compiled_restrict = jax.jit(jax.grad(restricted_loss)).lower(params).compile()

        baseline_cost = cost_bytes(compiled_baseline)
        restricted_cost = cost_bytes(compiled_restrict)
        baseline_mem = memory_stats(compiled_baseline)
        restricted_mem = memory_stats(compiled_restrict)

        M = cfg.batch_size * docs_per_query * cfg.doc_len
        total_slots.append(M)
        selected_counts.append(selected_doc_cap)
        baseline_costs.append(baseline_cost)
        restricted_costs.append(restricted_cost)
        baseline_temps.append(baseline_mem.get("temp_size_in_bytes", 0))
        restricted_temps.append(restricted_mem.get("temp_size_in_bytes", 0))

        print(
            f"{docs_per_query:>6}  {M:>8}  {selected_doc_cap:>8}  "
            f"{baseline_cost/1e6:>10.1f}  {restricted_cost/1e6:>10.1f}  "
            f"{baseline_temps[-1]/1e6:>12.1f}  {restricted_temps[-1]/1e6:>13.1f}"
        )
        assert selected_doc_cap <= cfg.batch_size + 2, (
            f"Selected-doc union grew unexpectedly: docs/q={docs_per_query}, "
            f"selected={selected_doc_cap}, ids={selected_doc_ids}"
        )

    baseline_cost_slope = np.polyfit(total_slots, baseline_costs, 1)[0]
    restricted_cost_slope = np.polyfit(total_slots, restricted_costs, 1)[0]
    baseline_temp_slope = np.polyfit(total_slots, baseline_temps, 1)[0]
    restricted_temp_slope = np.polyfit(total_slots, restricted_temps, 1)[0]

    print("\nSlope summary:")
    print(f"  baseline grad bytes slope      : {baseline_cost_slope:.2f} bytes / slot")
    print(f"  restricted grad bytes slope    : {restricted_cost_slope:.2f} bytes / slot")
    print(f"  baseline temp-mem slope        : {baseline_temp_slope:.2f} bytes / slot")
    print(f"  restricted temp-mem slope      : {restricted_temp_slope:.2f} bytes / slot")
    print("  interpretation                 : restricted re-embed is cheaper,")
    print("                                   but still grows with M because the")
    print("                                   full no-grad encoder pass is dense")
    print("                                   in docs and not streamed/chunked.")

    # Baseline should scale clearly with M; restricted path should be much flatter,
    # but this synthetic model shows it is still O(M) end-to-end.
    assert baseline_cost_slope > 500.0, f"Baseline slope too small: {baseline_cost_slope:.2f}"
    assert restricted_cost_slope > 0.0, "Restricted path unexpectedly appeared flat in bytes-accessed"
    assert restricted_cost_slope < baseline_cost_slope / 3.0, (
        f"Restricted cost slope not sufficiently smaller: {restricted_cost_slope:.2f} vs {baseline_cost_slope:.2f}"
    )
    assert baseline_temp_slope > 100.0, f"Baseline temp slope too small: {baseline_temp_slope:.2f}"
    assert restricted_temp_slope > 0.0, "Restricted path unexpectedly appeared flat in temp memory"
    assert restricted_temp_slope < baseline_temp_slope / 5.0, (
        f"Restricted temp slope not sufficiently smaller: {restricted_temp_slope:.2f} vs {baseline_temp_slope:.2f}"
    )
    print("PASS test_encoder_reembed_scaling")


if __name__ == "__main__":
    print("=" * 72)
    print("test_restricted_reembed_scaling.py")
    print("=" * 72)
    test_encoder_reembed_scaling()
    print("=" * 72)
    print("All tests passed.")
