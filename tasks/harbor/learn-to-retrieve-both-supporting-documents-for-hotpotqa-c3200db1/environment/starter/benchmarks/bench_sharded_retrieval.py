"""Benchmark: emulated TRAINING memory-retrieval forward+backward, comparing the naive full
score-matrix path (current training, OOMs) vs sharded_top_k_ip (the inference kernel, bank
sharded across the model axis). Isolates the retrieval bottleneck — random query + bank, no Qwen.

The training bank scales with batch (M ~= B * chunks_per_query), so the score tensor [B,N,T,M]
is ~B^2. This measures peak HBM + step time as B grows, for each path, to see how much bigger a
batch/bank the sharded path fits.

Run ONE (B, mode) per process (peak HBM is per-process), sweep via the shell loop in __main__ help:
  for MODE in full sharded; do for B in 16 32 64 128; do
    MEM_BENCH_B=$B MEM_BENCH_MODE=$MODE .venv/bin/python benchmarks/bench_sharded_retrieval.py
  done; done
"""
import os, time, functools
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P, AxisType

from models.retrieval_ops import sharded_top_k_ip, set_global_mesh, mem_weight_from_logits
from models.memory_utils import bank_top_k

# ---- config (emulates hparams/hp32 training retrieval) ----
B       = int(os.environ.get("MEM_BENCH_B", 16))
N       = int(os.environ.get("MEM_BENCH_N", 16))     # query heads (16=hparams, 32=hp32, 4=baseline)
T       = int(os.environ.get("MEM_BENCH_T", 512))    # seq len
D       = int(os.environ.get("MEM_BENCH_D", 128))    # query/key head_dim (mem_k_dim)
DV      = int(os.environ.get("MEM_BENCH_DV", 128))   # value head_dim (mem_v_dim)
TOPK    = int(os.environ.get("MEM_BENCH_TOPK", 64))
CHUNKS  = int(os.environ.get("MEM_BENCH_CHUNKS", 64))  # bank chunks per batch item -> M = B*CHUNKS
MODE    = os.environ.get("MEM_BENCH_MODE", "sharded")  # full | sharded | twopass | twopassfused
CHUNK   = int(os.environ.get("MEM_BENCH_CHUNK", 2048))  # twopass pass-1 scan chunk (lower = less mem/B)
KCHUNK  = int(os.environ.get("MEM_BENCH_KCHUNK", 8))     # twopassfused: K-block per scan step
ITERS   = int(os.environ.get("MEM_BENCH_ITERS", 20))
M       = B * CHUNKS

ndev = jax.device_count()


def _peak_gb():
    try:
        return jax.local_devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 1e9
    except Exception:
        return -1.0


def make_inputs(mesh, sharded):
    q  = jax.random.normal(jax.random.PRNGKey(0), (B, N, T, D), dtype=jnp.bfloat16)  # [B,N,T,D]
    mk = jax.random.normal(jax.random.PRNGKey(1), (M, D),  dtype=jnp.bfloat16)    # [M,D]
    mv = jax.random.normal(jax.random.PRNGKey(2), (M, DV), dtype=jnp.bfloat16)    # [M,Dv]
    if sharded:
        q  = jax.device_put(q,  NamedSharding(mesh, P(None, None, None, None)))   # replicated queries
        mk = jax.device_put(mk, NamedSharding(mesh, P("model", None)))            # KEYS sharded (score matrix distributed)
        mv = jax.device_put(mv, NamedSharding(mesh, P(None, None)))               # VALUES replicated (gather final K locally)
    else:
        q  = jax.device_put(q,  NamedSharding(mesh, P(None, None, None, None)))
        mk = jax.device_put(mk, NamedSharding(mesh, P(None, None)))               # replicated bank
        mv = jax.device_put(mv, NamedSharding(mesh, P(None, None)))
    return q, mk, mv


def read_full(q, mk, mv):
    # Naive full score matrix [B,N,T,M] (the current training path that OOMs)
    logits = jnp.einsum("bntd,md->bntm", q, mk, preferred_element_type=jnp.float32)
    logits = logits / jnp.sqrt(jnp.array(D, jnp.float32))
    tk_logits, tk_idx = bank_top_k(logits, TOPK)                 # [B,N,T,K]
    w = jax.nn.softmax(tk_logits, axis=-1).astype(jnp.bfloat16)  # [B,N,T,K]
    vals = mv[tk_idx]                                            # [B,N,T,K,Dv]
    read = jnp.einsum("bntk,bntkd->bntd", w, vals)              # [B,N,T,Dv]
    return read


def read_sharded(q, mk, mv, mesh):
    # keys_only: shard the SCORE/top-k over the sharded keys (the big [B,N,T,M] work), skip the
    # inline n_shards*K value merge (that intermediate is what blows up at training scale). Then
    # a SINGLE gather of the final K values from the (replicated) value bank -> [B,N,T,K,Dv].
    scores, _, idx, _ = sharded_top_k_ip(
        q, mk, mv, TOPK, mem_mask=None, concrete_mesh=mesh, shard_axis="model",
        return_all_scores=False, keys_only=True)
    vals = mv[idx]                                                     # [B,N,T,K,Dv], values replicated -> local
    read = jnp.einsum("bntk,bntkd->bntd", scores.astype(jnp.bfloat16), vals.astype(jnp.bfloat16))
    return read


def read_twopass(q, mk, mv, mesh):
    # Two-pass (the O(K) training path): pass-1 NO-GRAD chunked scan for top-k INDICES with
    # return_all_scores=False -> NEVER builds [B,N,T,M]. pass-2 gathers only the K selected keys/
    # values and grads over [B,N,T,K]. Bank replicated (model=1 mesh -> replicated chunked scan),
    # so no cross-device comms. This is the "two_pass_topk + chunk_size" cheap lever.
    # pass-1 scan chunk: per-chunk score is [B,N,T,chunk] (O(B*chunk)) — lower chunk at large B so
    # the pass-1 term stays small; then twopass is bounded by the pass-2 value-read O(B*K).
    q_sg = jax.lax.stop_gradient(q)
    _, _, idx, _ = sharded_top_k_ip(
        q_sg, jax.lax.stop_gradient(mk), mv, TOPK, mem_mask=None, chunk_size=CHUNK,
        concrete_mesh=mesh, shard_axis="model", return_all_scores=False, keys_only=True)
    idx = jax.lax.stop_gradient(idx)                                   # [B,N,T,K]
    rep5 = P(None, None, None, None, None)                             # replicated gather output (Explicit axes need it)
    mk_k = mk.at[idx].get(out_sharding=rep5)                           # [B,N,T,K,D]  grad -> mk (sparse)
    mv_k = mv.at[idx].get(out_sharding=rep5)                           # [B,N,T,K,Dv]
    logits = jnp.einsum("bntd,bntkd->bntk", q, mk_k, preferred_element_type=jnp.float32) / jnp.sqrt(
        jnp.array(D, jnp.float32))
    w = jax.nn.softmax(logits, axis=-1).astype(jnp.bfloat16)          # [B,N,T,K]
    return jnp.einsum("bntk,bntkd->bntd", w, mv_k.astype(jnp.bfloat16))


def read_twopass_fused(q, mk, mv, mesh):
    # Same two-pass indices as read_twopass, but the value-read is a REMAT chunk-scan over K so the
    # [B,N,T,K,D] / [B,N,T,K,Dv] gather intermediates NEVER materialize (fwd or bwd). Per scan step
    # holds only [B,N,T,KCHUNK,·]; jax.remat recomputes the gather in the backward instead of storing
    # every step. Peak -> ~[B,N,T,Dv] accumulator + [B,N,T,K] logits (both small). This is the fix
    # that should clear the B256 value-read OOM.
    q_sg = jax.lax.stop_gradient(q)
    _, _, idx, _ = sharded_top_k_ip(
        q_sg, jax.lax.stop_gradient(mk), mv, TOPK, mem_mask=None, chunk_size=CHUNK,
        concrete_mesh=mesh, shard_axis="model", return_all_scores=False, keys_only=True)
    idx = jax.lax.stop_gradient(idx)                                  # [B,N,T,K]
    nkc = TOPK // KCHUNK
    rep = P(None, None, None, None, None)
    idx_s = jnp.moveaxis(idx.reshape(B, N, T, nkc, KCHUNK), 3, 0)     # [nkc, B,N,T,KCHUNK]

    # Pass A: logits over K, one K-block per step -> logits [B,N,T,K] (no Dv, small).
    @jax.remat
    def score_step(_, ii):                                           # ii: [B,N,T,KCHUNK]
        mk_c = mk.at[ii].get(out_sharding=rep)                       # [B,N,T,KCHUNK,D]
        lg = jnp.einsum("bntd,bntcd->bntc", q, mk_c,
                        preferred_element_type=jnp.float32) / jnp.sqrt(jnp.array(D, jnp.float32))
        return None, lg
    _, logits_s = jax.lax.scan(score_step, None, idx_s)              # [nkc, B,N,T,KCHUNK]
    logits = jnp.moveaxis(logits_s, 0, 3).reshape(B, N, T, TOPK)
    w = jax.nn.softmax(logits, axis=-1)
    w_s = jnp.moveaxis(w.reshape(B, N, T, nkc, KCHUNK), 3, 0).astype(jnp.bfloat16)  # [nkc,B,N,T,KCHUNK]

    # Pass B: accumulate read, one K-block per step -> acc [B,N,T,Dv].
    @jax.remat
    def read_step(acc, ii_w):
        ii, wc = ii_w                                               # ii:[B,N,T,KCHUNK], wc:[B,N,T,KCHUNK]
        mv_c = mv.at[ii].get(out_sharding=rep).astype(jnp.bfloat16)  # [B,N,T,KCHUNK,Dv]
        return acc + jnp.einsum("bntc,bntcd->bntd", wc, mv_c), None
    read, _ = jax.lax.scan(read_step, jnp.zeros((B, N, T, DV), jnp.bfloat16), (idx_s, w_s))
    return read


def main():
    # sharded/full: (data=1, model=ndev) so the bank shards model-wise (or replicates for full).
    # twopass: (data=ndev, model=1) -> model=1 makes sharded_top_k_ip use the REPLICATED chunked
    # scan (no comms); bank replicated. Explicit axes required for reshard inside the kernel.
    if MODE in ("twopass", "twopassfused"):
        mesh = jax.make_mesh((ndev, 1), ("data", "model"),
                             axis_types=(AxisType.Explicit, AxisType.Explicit))
    else:
        mesh = jax.make_mesh((1, ndev), ("data", "model"),
                             axis_types=(AxisType.Explicit, AxisType.Explicit))
    set_global_mesh(mesh)
    jax.set_mesh(mesh)   # abstract-mesh context so reshard/PartitionSpec inside sharded_top_k_ip work

    q, mk, mv = make_inputs(mesh, sharded=(MODE == "sharded"))

    if MODE == "sharded":
        def loss_fn(mk_, q_):
            return read_sharded(q_, mk_, mv, mesh).astype(jnp.float32).sum()
    elif MODE == "twopass":
        def loss_fn(mk_, q_):
            return read_twopass(q_, mk_, mv, mesh).astype(jnp.float32).sum()
    elif MODE == "twopassfused":
        def loss_fn(mk_, q_):
            return read_twopass_fused(q_, mk_, mv, mesh).astype(jnp.float32).sum()
    else:
        def loss_fn(mk_, q_):
            return read_full(q_, mk_, mv).astype(jnp.float32).sum()

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))

    # warmup / compile
    try:
        v, (gk, gq) = grad_fn(mk, q)
        jax.block_until_ready((v, gk, gq))
    except Exception as e:
        print(f"MODE={MODE} B={B} M={M} N={N} -> OOM/ERROR: {str(e)[:140]}")
        return

    t0 = time.perf_counter()
    for _ in range(ITERS):
        v, (gk, gq) = grad_fn(mk, q)
    jax.block_until_ready((v, gk, gq))
    dt = (time.perf_counter() - t0) / ITERS

    toks = B * T
    print(f"MODE={MODE} B={B} M={M} N={N} T={T} D={D} | "
          f"step={dt*1000:.1f}ms  throughput={toks/dt:,.0f} tok/s  peakHBM={_peak_gb():.2f}G  "
          f"(fwd+bwd, {ndev} chips)")


if __name__ == "__main__":
    main()
