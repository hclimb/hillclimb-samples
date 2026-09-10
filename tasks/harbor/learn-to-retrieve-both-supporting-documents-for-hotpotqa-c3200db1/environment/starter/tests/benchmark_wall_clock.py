import os
import sys
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType
from jax.sharding import PartitionSpec as P

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses.doc_access_top_k_loss import compute_doc_access_top_k_loss
from models.memory import mem_lookup_chunked, mem_lookup_two_pass
from models.qwen3 import rms_norm


@dataclass(frozen=True)
class BenchCfg:
    batch_size: int = 512
    grad_accum_steps: int = 32
    seq_len: int = 512
    docs_per_query: int = 4
    doc_chunk_seq_len: int = 256
    mem_heads: int = 4
    mem_k_dim: int = 1024
    mem_v_dim: int = 1024
    mem_top_k: int = 128
    mem_lookup_chunk_size: int = 16384
    two_pass_topk: bool = True
    sparse_grads: bool = False
    mem_k_prenormed: bool = False
    rms_norm_eps: float = 1e-6
    iters: int = 3

    @property
    def micro_batch_size(self) -> int:
        return self.batch_size // self.grad_accum_steps

    @property
    def total_docs(self) -> int:
        return self.batch_size * self.docs_per_query

    @property
    def total_mem_slots(self) -> int:
        return self.total_docs * self.doc_chunk_seq_len

    @property
    def pos_slots_per_query(self) -> int:
        return self.docs_per_query * self.doc_chunk_seq_len

    def memory_cfg(self):
        return {
            "rms_norm_eps": self.rms_norm_eps,
            "mem_top_k": self.mem_top_k,
            "mem_lookup_chunk_size": self.mem_lookup_chunk_size,
            "mem_k_prenormed": self.mem_k_prenormed,
            "two_pass_topk": self.two_pass_topk,
            "sparse_grads": self.sparse_grads,
        }


def _timeit(fn, *args, iters=3):
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - start)
    return float(np.mean(times))


def make_inputs(cfg: BenchCfg):
    dtype = jnp.bfloat16
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 6)

    q = jax.random.normal(
        keys[0],
        (cfg.micro_batch_size, cfg.seq_len, cfg.mem_heads, cfg.mem_k_dim),
        dtype=dtype,
    )
    mem_k = jax.random.normal(
        keys[1],
        (cfg.total_mem_slots, cfg.mem_k_dim),
        dtype=dtype,
    )
    mem_v = jax.random.normal(
        keys[2],
        (cfg.total_mem_slots, cfg.mem_v_dim),
        dtype=dtype,
    )
    mem_k_norm = jnp.ones((cfg.mem_k_dim,), dtype=dtype)
    docs_mask = jnp.ones((cfg.total_docs, cfg.doc_chunk_seq_len), dtype=jnp.bool_)
    pos_doc_mask = jnp.ones((cfg.micro_batch_size, cfg.docs_per_query), dtype=jnp.bool_)
    loss_mask = jnp.ones((cfg.micro_batch_size, cfg.seq_len), dtype=jnp.bool_)

    w = {
        "mem_k": mem_k,
        "mem_v": mem_v,
        "mem_k_norm": mem_k_norm,
        "mem_mask": docs_mask.reshape(-1),
    }
    input_mask = {
        "docs_mask": docs_mask,
        "pos_doc_mask": pos_doc_mask,
    }
    return q, w, input_mask, loss_mask


def make_pos_slot_indices(cfg: BenchCfg):
    global_query_idx = jnp.arange(cfg.micro_batch_size, dtype=jnp.int32)
    global_doc_idx = (
        global_query_idx[:, None] * cfg.docs_per_query
        + jnp.arange(cfg.docs_per_query, dtype=jnp.int32)[None, :]
    )
    pos_slot_indices = (
        global_doc_idx[:, :, None] * cfg.doc_chunk_seq_len
        + jnp.arange(cfg.doc_chunk_seq_len, dtype=jnp.int32)[None, None, :]
    ).reshape(cfg.micro_batch_size, -1)
    return pos_slot_indices


def build_aux_data(cfg: BenchCfg, top_k_indices, top_k_logits, pos_slot_indices, q, w):
    q_t = jnp.transpose(q, (0, 2, 1, 3))
    mem_k_normed = rms_norm(w["mem_k"], w["mem_k_norm"], cfg.rms_norm_eps)
    mem_k_pos = mem_k_normed.at[pos_slot_indices].get()
    pos_logits = jnp.einsum("bntd,bpd->bntp", q_t, mem_k_pos) / jnp.sqrt(
        jnp.array(q.shape[-1], dtype=jnp.float32)
    )
    pos_indices_expanded = jnp.broadcast_to(
        pos_slot_indices[:, None, None, :],
        (cfg.micro_batch_size, cfg.mem_heads, cfg.seq_len, cfg.pos_slots_per_query),
    )
    pos_indices_expanded = jax.sharding.reshard(pos_indices_expanded, P("data", "model", None, None))
    pos_logits = jax.sharding.reshard(pos_logits, P("data", "model", None, None))
    return {
        "mem_top_k_indices": (top_k_indices,),
        "mem_top_k_logits": (top_k_logits,),
        "mem_pos_indices": (pos_indices_expanded,),
        "mem_pos_logits": (pos_logits,),
        "effective_doc_len": cfg.doc_chunk_seq_len,
        "effective_mem_mask": w["mem_mask"],
    }


def normalize_aux(aux):
    out = dict(aux)
    for key in ("mem_top_k_indices", "mem_top_k_logits", "mem_pos_indices", "mem_pos_logits"):
        if key in out and out[key] is not None and not isinstance(out[key], tuple):
            out[key] = (out[key],)
    return out


def main():
    cfg = BenchCfg()
    tp_devices = 1
    fsdp_devices = jax.device_count() // tp_devices
    mesh = jax.make_mesh(
        (fsdp_devices, tp_devices),
        ("data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    jax.set_mesh(mesh)

    print(f"JAX devices: {jax.devices()}")
    print(f"JAX mesh: {mesh}")
    print("Benchmark geometry:")
    print(
        f"  batch={cfg.batch_size}, grad_accum_steps={cfg.grad_accum_steps}, "
        f"micro_batch={cfg.micro_batch_size}, seq_len={cfg.seq_len}"
    )
    print(
        f"  docs_per_query={cfg.docs_per_query}, total_docs={cfg.total_docs}, "
        f"doc_len={cfg.doc_chunk_seq_len}, total_mem_slots={cfg.total_mem_slots}"
    )
    print(
        f"  mem_heads={cfg.mem_heads}, k_dim={cfg.mem_k_dim}, v_dim={cfg.mem_v_dim}, "
        f"top_k={cfg.mem_top_k}, chunk_size={cfg.mem_lookup_chunk_size}"
    )

    q, w, input_mask, loss_mask = make_inputs(cfg)
    pos_slot_indices = make_pos_slot_indices(cfg)
    mem_cfg = cfg.memory_cfg()

    @jax.jit
    def pass1_only(q, w):
        scores, _, aux = mem_lookup_chunked(q, w, mem_cfg, collect_aux=True, keys_only=True)
        return scores, aux["mem_top_k_indices"]

    @jax.jit
    def two_pass_only(q, w, pos_slot_indices):
        scores, _, aux = mem_lookup_two_pass(q, w, mem_cfg, collect_aux=True, pos_slot_indices=pos_slot_indices)
        return scores, aux["mem_top_k_indices"], aux["mem_top_k_logits"]

    @jax.jit
    def pass2_plus_aux_only(q, w, pos_slot_indices):
        _, _, aux = mem_lookup_two_pass(q, w, mem_cfg, collect_aux=True, pos_slot_indices=pos_slot_indices)
        aux = normalize_aux(aux)
        loss = compute_doc_access_top_k_loss(aux, loss_mask, input_mask, q, temperature=1.0)
        return loss

    top_k_scores, top_k_indices = pass1_only(q, w)
    top_k_scores.block_until_ready()
    top_k_logits = jnp.log(jnp.maximum(top_k_scores, 1e-9))
    aux_data = build_aux_data(cfg, top_k_indices, top_k_logits, pos_slot_indices, q, w)

    @jax.jit
    def aux_only(aux_data):
        return compute_doc_access_top_k_loss(aux_data, loss_mask, input_mask, q, temperature=1.0)

    @jax.jit
    def retrieval_fwd_bwd(q, w, pos_slot_indices):
        def loss_fn(q, mem_k, mem_v):
            w_local = {
                "mem_k": mem_k,
                "mem_v": mem_v,
                "mem_k_norm": w["mem_k_norm"],
                "mem_mask": w["mem_mask"],
            }
            scores, values, aux = mem_lookup_two_pass(
                q, w_local, mem_cfg, collect_aux=True, pos_slot_indices=pos_slot_indices
            )
            aux = normalize_aux(aux)
            retrieval_loss = jnp.mean(jnp.einsum("bntk,bntkd->bntd", scores, values).astype(jnp.float32) ** 2)
            aux_loss = compute_doc_access_top_k_loss(aux, loss_mask, input_mask, q, temperature=1.0)
            return retrieval_loss + aux_loss

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, w["mem_k"], w["mem_v"])
        return sum(jnp.sum(jnp.abs(x).astype(jnp.float32)) for x in grads)

    pass1_s = _timeit(pass1_only, q, w, iters=cfg.iters)
    two_pass_s = _timeit(two_pass_only, q, w, pos_slot_indices, iters=cfg.iters)
    aux_s = _timeit(aux_only, aux_data, iters=cfg.iters)
    pass2_aux_s = _timeit(pass2_plus_aux_only, q, w, pos_slot_indices, iters=cfg.iters)
    retrieval_bwd_s = _timeit(retrieval_fwd_bwd, q, w, pos_slot_indices, iters=cfg.iters)

    micro_est_s = retrieval_bwd_s
    full_step_est_s = micro_est_s * cfg.grad_accum_steps

    print("\nMeasured wall clock:")
    print(f"  pass1 chunked top-k only           : {pass1_s:8.3f} s / microbatch")
    print(f"  full two-pass retrieval forward    : {two_pass_s:8.3f} s / microbatch")
    print(f"  aux loss only                      : {aux_s:8.3f} s / microbatch")
    print(f"  two-pass retrieval + aux forward   : {pass2_aux_s:8.3f} s / microbatch")
    print(f"  retrieval+aux forward/backward     : {retrieval_bwd_s:8.3f} s / microbatch")

    print("\nBack-of-envelope step estimate:")
    print(f"  microbatch retrieval share         : {micro_est_s:8.3f} s")
    print(f"  x grad_accum_steps ({cfg.grad_accum_steps:>2})             : {full_step_est_s:8.3f} s / train step")
    print("  note: this excludes embed model forward/backward and the rest of the main model")


if __name__ == "__main__":
    main()
