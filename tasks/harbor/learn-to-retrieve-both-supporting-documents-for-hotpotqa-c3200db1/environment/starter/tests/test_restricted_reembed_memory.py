"""
Compare two embed-model training strategies for qwen3_mem_embed-style batches:

  1. Current full-embed two-pass:
     - Embed every document in the mini-batch with gradients enabled.
     - Run mem_lookup_two_pass over the full memory bank.

  2. Restricted re-embed two-pass:
     - Embed every document once under stop_gradient to get pass-1 top-k slots.
     - Take the union of selected docs plus positive docs.
     - Re-embed only that restricted doc set with gradients enabled.
     - Reuse the original pass-1 selections, remapped into the restricted bank.

This test answers two questions:
  a) Validity: does restricted re-embed preserve the same loss / parameter grads
     as the current approach when the restricted set includes all docs needed by
     the pass-2 objective (selected docs + positives)?
  b) Cost: how do compile-time memory proxies and wall-clock throughput compare?

Running:
    env UV_CACHE_DIR=/tmp/uv-cache JAX_PLATFORMS=cpu \
        uv run python tests/test_restricted_reembed_memory.py
"""

import os
import sys
import time
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
    docs_per_query: int = 8
    doc_len: int = 64
    input_dim: int = 128
    hidden_dim: int = 192
    mem_heads: int = 2
    mem_k_dim: int = 64
    mem_v_dim: int = 64
    mem_top_k: int = 8
    mem_lookup_chunk_size: int = 256
    depth: int = 3
    iters: int = 5

    @property
    def total_docs(self) -> int:
        return self.batch_size * self.docs_per_query

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
    params = {
        "in_proj": jax.random.normal(keys[0], (cfg.input_dim, cfg.hidden_dim), dtype=jnp.float32) * 0.05,
        "layers": [
            jax.random.normal(keys[i + 1], (cfg.hidden_dim, cfg.hidden_dim), dtype=jnp.float32) * 0.05
            for i in range(cfg.depth)
        ],
        "k_proj": jax.random.normal(keys[-2], (cfg.hidden_dim, cfg.mem_k_dim), dtype=jnp.float32) * 0.05,
        "v_proj": jax.random.normal(keys[-1], (cfg.hidden_dim, cfg.mem_v_dim), dtype=jnp.float32) * 0.05,
    }
    return params


def embed_docs(params, docs):
    x = jnp.einsum("dli,ih->dlh", docs, params["in_proj"])
    x = jax.nn.gelu(x)
    for w in params["layers"]:
        x = jnp.einsum("dli,ih->dlh", x, w)
        x = jax.nn.gelu(x)
    mem_k = jnp.einsum("dli,ih->dlh", x, params["k_proj"])
    mem_v = jnp.einsum("dli,iv->dlv", x, params["v_proj"])
    return mem_k.reshape(-1, mem_k.shape[-1]), mem_v.reshape(-1, mem_v.shape[-1])


def make_inputs(cfg: BenchCfg):
    key = jax.random.PRNGKey(0)
    docs_key, params_key, noise_key = jax.random.split(key, 3)
    docs = 0.001 * jax.random.normal(
        docs_key,
        (cfg.total_docs, cfg.doc_len, cfg.input_dim),
        dtype=jnp.float32,
    )
    params = make_params(params_key, cfg)

    pos_doc_ids = jnp.arange(cfg.batch_size, dtype=jnp.int32) * cfg.docs_per_query
    pos_docs = jax.random.normal(
        jax.random.PRNGKey(7),
        (cfg.batch_size, cfg.doc_len, cfg.input_dim),
        dtype=jnp.float32,
    )
    docs = docs.at[pos_doc_ids].set(pos_docs)

    # Build queries that strongly target one positive doc per query so the
    # selected-doc union stays much smaller than the full mini-batch doc set.
    mem_k0, _ = embed_docs(params, docs)
    mem_k0 = mem_k0.reshape(cfg.total_docs, cfg.doc_len, cfg.mem_k_dim)
    tok_idx = jnp.arange(cfg.seq_len, dtype=jnp.int32) % cfg.doc_len
    q_base = mem_k0[pos_doc_ids[:, None], tok_idx[None, :]]
    q_base = jnp.broadcast_to(q_base[:, :, None, :], (cfg.batch_size, cfg.seq_len, cfg.mem_heads, cfg.mem_k_dim))
    q = q_base + 0.01 * jax.random.normal(noise_key, q_base.shape, dtype=jnp.float32)

    docs_mask = jnp.ones((cfg.total_docs, cfg.doc_len), dtype=jnp.bool_)
    loss_mask = jnp.ones((cfg.batch_size, cfg.seq_len), dtype=jnp.bool_)
    pos_doc_mask = jnp.zeros((cfg.batch_size, cfg.docs_per_query), dtype=jnp.bool_).at[:, 0].set(True)
    pos_slot_indices = make_positive_slot_indices(cfg)
    return docs, params, q, docs_mask, loss_mask, pos_slot_indices, pos_doc_mask


def make_positive_slot_indices(cfg: BenchCfg):
    global_doc_idx = jnp.arange(cfg.batch_size, dtype=jnp.int32) * cfg.docs_per_query
    return (
        global_doc_idx[:, None] * cfg.doc_len
        + jnp.arange(cfg.doc_len, dtype=jnp.int32)[None, :]
    )


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
    for name in (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
        "generated_code_size_in_bytes",
    ):
        if hasattr(analysis, name):
            stats[name] = getattr(analysis, name)
    return stats


def _timeit(fn, *args, iters=3):
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - start)
    return float(np.mean(times))


def select_doc_ids(top_k_indices, pos_slot_indices, cfg: BenchCfg, selected_doc_cap: int):
    top_doc_ids = (top_k_indices.reshape(-1) // cfg.doc_len).astype(jnp.int32)
    pos_doc_ids = (pos_slot_indices.reshape(-1) // cfg.doc_len).astype(jnp.int32)
    selected_mask = jnp.zeros((cfg.total_docs,), dtype=jnp.int32)
    selected_mask = selected_mask.at[top_doc_ids].set(1)
    selected_mask = selected_mask.at[pos_doc_ids].set(1)
    _, selected_doc_ids = jax.lax.top_k(selected_mask, selected_doc_cap)
    return selected_doc_ids, selected_mask


def remap_slot_indices(global_slot_indices, selected_doc_ids, cfg: BenchCfg):
    local_doc_map = -jnp.ones((cfg.total_docs,), dtype=jnp.int32)
    local_doc_map = local_doc_map.at[selected_doc_ids].set(jnp.arange(selected_doc_ids.shape[0], dtype=jnp.int32))
    global_doc_ids = (global_slot_indices // cfg.doc_len).astype(jnp.int32)
    slot_offsets = (global_slot_indices % cfg.doc_len).astype(jnp.int32)
    local_doc_ids = local_doc_map[global_doc_ids]
    return local_doc_ids * cfg.doc_len + slot_offsets


def fixed_indices_pass2(q, mem_k, mem_v, mem_k_norm, top_k_indices_local, pos_slot_indices_local, cfg: BenchCfg):
    mem_k_normed = rms_norm(mem_k, mem_k_norm, cfg.memory_cfg()["rms_norm_eps"])
    mem_k_k = mem_k_normed[top_k_indices_local]
    mem_v_k = mem_v[top_k_indices_local]
    q_t = jnp.transpose(q, (0, 2, 1, 3))
    scale = jnp.sqrt(jnp.array(q.shape[-1], dtype=jnp.float32))
    top_k_logits = jnp.einsum("bntd,bntkd->bntk", q_t, mem_k_k) / scale
    top_k_scores = jax.nn.softmax(top_k_logits, axis=-1).astype(q.dtype)

    mem_k_pos = mem_k_normed[pos_slot_indices_local]
    pos_logits = jnp.einsum("bntd,bpd->bntp", q_t, mem_k_pos) / scale
    return top_k_scores, mem_v_k, top_k_logits, pos_logits


def estimate_selected_doc_cap(params, docs, q, docs_mask, pos_slot_indices, cfg: BenchCfg):
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
    top_k_indices = aux["mem_top_k_indices"]
    selected_doc_ids, selected_mask = select_doc_ids(top_k_indices, pos_slot_indices, cfg, cfg.total_docs)
    selected_count = int(jnp.sum(selected_mask))
    selected_doc_ids = np.array(selected_doc_ids[:selected_count])
    return selected_count, selected_doc_ids.tolist()


def build_losses(docs, q, docs_mask, loss_mask, pos_slot_indices, cfg: BenchCfg, selected_doc_cap: int):
    mem_cfg = cfg.memory_cfg()
    mem_k_norm = jnp.ones((cfg.mem_k_dim,), dtype=jnp.float32)
    effective_mem_mask = docs_mask.reshape(-1)

    def current_loss(params):
        mem_k, mem_v = embed_docs(params, docs)
        w = {
            "mem_k": mem_k,
            "mem_v": mem_v,
            "mem_k_norm": mem_k_norm,
            "mem_mask": effective_mem_mask,
        }
        top_k_scores, mem_v_k, aux = mem_lookup_two_pass(
            q,
            w,
            mem_cfg,
            collect_aux=True,
            pos_slot_indices=pos_slot_indices,
        )
        return retrieval_loss(top_k_scores, mem_v_k) + topk_positive_contrastive_loss(
            aux["mem_top_k_logits"],
            aux["mem_pos_logits"],
            loss_mask,
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
            top_k_indices_global,
            pos_slot_indices,
            cfg,
            selected_doc_cap,
        )
        selected_doc_ids = jax.lax.stop_gradient(selected_doc_ids)

        docs_selected = docs[selected_doc_ids]
        mem_k_sel, mem_v_sel = embed_docs(params, docs_selected)

        top_k_indices_local = remap_slot_indices(top_k_indices_global, selected_doc_ids, cfg)
        pos_slot_indices_local = remap_slot_indices(pos_slot_indices, selected_doc_ids, cfg)

        top_k_scores, mem_v_k, top_k_logits, pos_logits = fixed_indices_pass2(
            q,
            mem_k_sel,
            mem_v_sel,
            mem_k_norm,
            top_k_indices_local,
            pos_slot_indices_local,
            cfg,
        )

        return retrieval_loss(top_k_scores, mem_v_k) + topk_positive_contrastive_loss(
            top_k_logits,
            pos_logits,
            loss_mask,
        )

    return current_loss, restricted_loss


def test_restricted_reembed_matches_current():
    cfg = BenchCfg()
    docs, params, q, docs_mask, loss_mask, pos_slot_indices, _ = make_inputs(cfg)
    selected_doc_cap, selected_docs = estimate_selected_doc_cap(
        params,
        docs,
        q,
        docs_mask,
        pos_slot_indices,
        cfg,
    )
    current_loss, restricted_loss = build_losses(
        docs,
        q,
        docs_mask,
        loss_mask,
        pos_slot_indices,
        cfg,
        selected_doc_cap,
    )

    val_current = current_loss(params)
    val_restrict = restricted_loss(params)
    grads_current = jax.grad(current_loss)(params)
    grads_restrict = jax.grad(restricted_loss)(params)

    loss_diff = float(jnp.abs(val_current - val_restrict))
    max_grad_diff = max(
        float(jnp.max(jnp.abs(a - b)))
        for a, b in zip(
            jax.tree_util.tree_leaves(grads_current),
            jax.tree_util.tree_leaves(grads_restrict),
        )
    )

    print("\nValidity check:")
    print(f"  total docs                  : {cfg.total_docs}")
    print(f"  selected docs in union      : {selected_doc_cap}")
    print(f"  selected doc ids            : {selected_docs}")
    print(f"  loss abs diff               : {loss_diff:.3e}")
    print(f"  max param-grad abs diff     : {max_grad_diff:.3e}")

    assert selected_doc_cap < cfg.total_docs, "Synthetic batch did not yield a restricted doc subset"
    assert loss_diff < 1e-5, f"loss mismatch too large: {loss_diff:.3e}"
    assert max_grad_diff < 2e-5, f"grad mismatch too large: {max_grad_diff:.3e}"
    print("PASS test_restricted_reembed_matches_current")


def test_restricted_reembed_cost_and_throughput():
    cfg = BenchCfg()
    docs, params, q, docs_mask, loss_mask, pos_slot_indices, _ = make_inputs(cfg)
    selected_doc_cap, _ = estimate_selected_doc_cap(
        params,
        docs,
        q,
        docs_mask,
        pos_slot_indices,
        cfg,
    )
    current_loss, restricted_loss = build_losses(
        docs,
        q,
        docs_mask,
        loss_mask,
        pos_slot_indices,
        cfg,
        selected_doc_cap,
    )

    current_grad = jax.jit(jax.grad(current_loss))
    restricted_grad = jax.jit(jax.grad(restricted_loss))

    compiled_current = current_grad.lower(params).compile()
    compiled_restrict = restricted_grad.lower(params).compile()
    cost_current = cost_bytes(compiled_current)
    cost_restrict = cost_bytes(compiled_restrict)
    mem_current = memory_stats(compiled_current)
    mem_restrict = memory_stats(compiled_restrict)

    def grad_sum(fn, p):
        grads = fn(p)
        return sum(jnp.sum(jnp.abs(x).astype(jnp.float32)) for x in jax.tree_util.tree_leaves(grads))

    current_sum = jax.jit(lambda p: grad_sum(current_grad, p))
    restrict_sum = jax.jit(lambda p: grad_sum(restricted_grad, p))

    t_current = _timeit(current_sum, params, iters=cfg.iters)
    t_restrict = _timeit(restrict_sum, params, iters=cfg.iters)

    print("\nCost and throughput:")
    print(f"  total docs                  : {cfg.total_docs}")
    print(f"  restricted docs             : {selected_doc_cap}")
    print(f"  doc fraction re-embedded    : {selected_doc_cap / cfg.total_docs:.3f}")
    print(f"  grad bytes accessed current : {cost_current / 1e6:8.1f} MB")
    print(f"  grad bytes accessed restr.  : {cost_restrict / 1e6:8.1f} MB")
    print(f"  cost ratio current/restr.   : {cost_current / cost_restrict:8.3f}")
    if mem_current and mem_restrict:
        tc = mem_current.get("temp_size_in_bytes", 0) / 1e6
        tr = mem_restrict.get("temp_size_in_bytes", 0) / 1e6
        print(f"  temp memory current         : {tc:8.1f} MB")
        print(f"  temp memory restr.          : {tr:8.1f} MB")
        if tr > 0:
            print(f"  temp-mem ratio current/restr: {tc / tr:8.3f}")
    print(f"  step time current           : {t_current:8.4f} s")
    print(f"  step time restr.            : {t_restrict:8.4f} s")
    print(f"  throughput ratio current/restr.: {t_current / t_restrict:8.3f}")

    assert selected_doc_cap < cfg.total_docs, "Expected a strict subset of docs"
    assert cost_restrict < cost_current, "Restricted re-embed should lower gradient cost"
    if mem_current and mem_restrict:
        assert mem_restrict.get("temp_size_in_bytes", 0) <= mem_current.get("temp_size_in_bytes", 0)
    print("PASS test_restricted_reembed_cost_and_throughput")


if __name__ == "__main__":
    print("=" * 72)
    print("test_restricted_reembed_memory.py")
    print("=" * 72)
    test_restricted_reembed_matches_current()
    test_restricted_reembed_cost_and_throughput()
    print("=" * 72)
    print("All tests passed.")
