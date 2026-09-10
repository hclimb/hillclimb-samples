"""Benchmark: memory-retrieval forward+backward at the multihop_hard_neg_full geometry (200
hard negatives -> num_chunks_per_doc=256, doc_chunk_seq_len=256 -> m_per_query=65,536 doc-token
slots/query). Compares the masked full-cross-batch path (current default, M_total = B*m_per_query)
against mem_lookup_batched (true per-row retrieval, M stays m_per_query regardless of B) and the
two-pass/chunked paths, isolating the retrieval step exactly like benchmarks/bench_sharded_retrieval.py
and tests/benchmark_wall_clock.py (random query + bank, no Qwen). See
wiki/implementations/2026-08-02-hard-neg-full-efficient-retrieval.md.

Run ONE (B, mode) per process (peak HBM is per-process); sweep via a shell loop, e.g.:
  for MODE in full_masked batched two_pass; do for B in 4 8 16; do
    MEM_BENCH_B=$B MEM_BENCH_MODE=$MODE .venv/bin/python benchmarks/bench_hard_neg_full_retrieval.py
  done; done
"""
import os, sys, time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, AxisType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root: utils.py isn't packaged

from utils import init_jax_distributed
from models.memory import mem_lookup, mem_lookup_batched, mem_lookup_chunked, mem_lookup_two_pass

# Every entry point that touches JAX must call this (see utils.py::init_jax_distributed) — on a
# single GCE host it's a no-op; on the 2x4 flex slice (both hosts launched together) it forms the
# real 8-chip mesh. Set JAX_FORCE_SINGLE_HOST=1 for a quick 4-chip sweep on one host standalone.
init_jax_distributed()

# ---- config (multihop_hard_neg_full.yaml geometry by default) ----
B          = int(os.environ.get("MEM_BENCH_B", 8))
N          = int(os.environ.get("MEM_BENCH_N", 4))          # mem_num_heads (qwen3_mem_embed default)
T          = int(os.environ.get("MEM_BENCH_T", 512))        # seq_len
D          = int(os.environ.get("MEM_BENCH_D", 1024))       # mem_k_dim
DV         = int(os.environ.get("MEM_BENCH_DV", 1024))      # mem_v_dim
TOPK       = int(os.environ.get("MEM_BENCH_TOPK", 128))     # mem_top_k
DOC_CHUNKS = int(os.environ.get("MEM_BENCH_DOC_CHUNKS", 256))   # num_chunks_per_doc (total budget/query)
DOC_LEN    = int(os.environ.get("MEM_BENCH_DOC_LEN", 256))      # doc_chunk_seq_len
MODE       = os.environ.get("MEM_BENCH_MODE", "batched")    # full_masked | batched | two_pass | two_pass_kchunk | chunked
TP_DEVICES = int(os.environ.get("MEM_BENCH_TP", 1))         # mesh 'model' axis size
CHUNK      = int(os.environ.get("MEM_BENCH_CHUNK", 8192))   # mem_lookup_chunk_size / two-pass pass-1 scan chunk
KCHUNK     = int(os.environ.get("MEM_BENCH_KCHUNK", 16))    # two_pass_kchunk: mem_value_read_kchunk
T_CHUNK    = int(os.environ.get("MEM_BENCH_T_CHUNK", 0))    # mem_t_chunk (batched/gqa); 0 = off
APPROX     = os.environ.get("MEM_BENCH_APPROX", "1") == "1"
ITERS      = int(os.environ.get("MEM_BENCH_ITERS", 10))

M_PER_QUERY = DOC_CHUNKS * DOC_LEN
M_TOTAL = B * M_PER_QUERY
ndev = jax.device_count()


def _peak_gb():
    try:
        return jax.local_devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 1e9
    except Exception:
        return -1.0


def make_inputs():
    q = jax.random.normal(jax.random.PRNGKey(0), (B, T, N, D), dtype=jnp.bfloat16)
    mem_k = jax.random.normal(jax.random.PRNGKey(1), (M_TOTAL, D), dtype=jnp.bfloat16)
    mem_v = jax.random.normal(jax.random.PRNGKey(2), (M_TOTAL, DV), dtype=jnp.bfloat16)
    mem_k_norm = jnp.ones((D,), dtype=jnp.bfloat16)
    mem_mask = jnp.ones((M_TOTAL,), dtype=jnp.float32)
    return q, mem_k, mem_v, mem_k_norm, mem_mask


def base_cfg():
    return {
        "rms_norm_eps": 1e-6,
        "mem_top_k": TOPK,
        "mem_approx_topk": APPROX,
        "mem_approx_recall": 0.99,
        "mem_score_activation": "softmax",
        "mem_softmax_temp": 1.0,
        "mem_phantom_log_n": 0.0,
    }


def loss_fn_full_masked(mem_k, mem_v, q, mem_k_norm, mem_mask):
    w = {"mem_k": mem_k, "mem_v": mem_v, "mem_k_norm": mem_k_norm, "mem_mask": mem_mask}
    cfg = {**base_cfg(), "per_query_isolation": True, "isolation_group_size": 1}
    scores, values, _ = mem_lookup(q, w, cfg, collect_aux=False)
    return jnp.einsum("bntk,bntkv->btnv", scores, values).astype(jnp.float32).sum()


def loss_fn_batched(mem_k, mem_v, q, mem_k_norm, mem_mask):
    w = {"mem_k": mem_k, "mem_v": mem_v, "mem_k_norm": mem_k_norm, "mem_mask": mem_mask}
    cfg = {**base_cfg(), "per_query_isolation": True, "isolation_group_size": 1,
           "mem_t_chunk": T_CHUNK}
    scores, values, _ = mem_lookup_batched(q, w, cfg, collect_aux=False)
    return jnp.einsum("bntk,bntkv->btnv", scores, values).astype(jnp.float32).sum()


def loss_fn_chunked(mem_k, mem_v, q, mem_k_norm, mem_mask):
    w = {"mem_k": mem_k, "mem_v": mem_v, "mem_k_norm": mem_k_norm, "mem_mask": mem_mask}
    cfg = {**base_cfg(), "mem_lookup_chunk_size": CHUNK}
    scores, values, _ = mem_lookup_chunked(q, w, cfg, collect_aux=False)
    return jnp.einsum("bntk,bntkv->btnv", scores, values).astype(jnp.float32).sum()


def loss_fn_two_pass(mem_k, mem_v, q, mem_k_norm, mem_mask, kchunk):
    w = {"mem_k": mem_k, "mem_v": mem_v, "mem_k_norm": mem_k_norm, "mem_mask": mem_mask}
    cfg = {**base_cfg(), "two_pass_topk": True, "mem_lookup_chunk_size": CHUNK,
           "mem_value_read_kchunk": kchunk}
    scores, values, _ = mem_lookup_two_pass(q, w, cfg, collect_aux=False)
    # kchunk>0 always returns the already-reduced [B,N,T,Dv] read in the values slot regardless of
    # collect_aux (models/memory.py::mem_lookup_two_pass) — memory_layer's own dispatch branches on
    # cfg, not on aux_data (which is only populated when collect_aux=True); mirror that here rather
    # than checking aux (a bug in an earlier version of this script: aux is None at collect_aux=False,
    # so `aux.get("mem_fused_read")` never fired and this always took the wrong (shape-mismatched)
    # einsum branch for kchunk>0, erroring "Einstein sum subscript 'bntkv' does not contain the
    # correct number of indices for operand 1").
    if kchunk > 0:
        return values.astype(jnp.float32).sum()  # already reduced [B,N,T,Dv]
    return jnp.einsum("bntk,bntkv->btnv", scores, values).astype(jnp.float32).sum()


def main():
    fsdp_devices = max(1, ndev // TP_DEVICES)
    mesh = jax.make_mesh((fsdp_devices, TP_DEVICES), ("data", "model"),
                          axis_types=(AxisType.Explicit, AxisType.Explicit))
    jax.set_mesh(mesh)

    q, mem_k, mem_v, mem_k_norm, mem_mask = make_inputs()

    if MODE == "full_masked":
        lf = loss_fn_full_masked
    elif MODE == "batched":
        lf = loss_fn_batched
    elif MODE == "chunked":
        lf = loss_fn_chunked
    elif MODE == "two_pass":
        lf = lambda *a: loss_fn_two_pass(*a, kchunk=0)
    elif MODE == "two_pass_kchunk":
        lf = lambda *a: loss_fn_two_pass(*a, kchunk=KCHUNK)
    else:
        raise ValueError(f"unknown MODE={MODE}")

    grad_fn = jax.jit(jax.value_and_grad(lf, argnums=(0, 1)))

    try:
        v, (gk, gv) = grad_fn(mem_k, mem_v, q, mem_k_norm, mem_mask)
        jax.block_until_ready((v, gk, gv))
    except Exception as e:
        print(f"MODE={MODE} B={B} m_per_query={M_PER_QUERY} M_total={M_TOTAL} TP={TP_DEVICES} "
              f"-> OOM/ERROR: {str(e)[:200]}")
        return

    t0 = time.perf_counter()
    for _ in range(ITERS):
        v, (gk, gv) = grad_fn(mem_k, mem_v, q, mem_k_norm, mem_mask)
    jax.block_until_ready((v, gk, gv))
    dt = (time.perf_counter() - t0) / ITERS

    toks = B * T
    print(f"MODE={MODE:16s} B={B:3d} m_per_query={M_PER_QUERY:7d} M_total={M_TOTAL:8d} "
          f"TP={TP_DEVICES} N={N} topk={TOPK} approx={APPROX} t_chunk={T_CHUNK} kchunk={KCHUNK} | "
          f"step={dt*1000:8.1f}ms  throughput={toks/dt:9,.0f} tok/s  peakHBM={_peak_gb():6.2f}G  "
          f"({ndev} chips)")


if __name__ == "__main__":
    main()
