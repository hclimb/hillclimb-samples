#!/usr/bin/env python3
"""INDEXING cost: what it takes to encode a corpus into the memory bank.

Every throughput number in the Pareto work allocates the bank as random tensors and never pays to
build it. That hides a real cost, and this measures it.

WHAT IS BEING MEASURED. models/qwen3_mem_embed.py::embed_forward runs a FULL 36-layer
qwen3_forward over the document chunks and then projects the last hidden states through
mem_k_proj / mem_v_proj. So indexing is a genuine forward pass over every corpus token.

WHY IT IS NOT THE SAME AS A LONG-CONTEXT PREFILL. Documents are encoded as INDEPENDENT
chunk_size-token chunks, so attention is confined within a chunk (S=256) and the cost is linear in
corpus tokens. Full-context RAG prefilling the same corpus pays attention across the whole
sequence, which is quadratic. Identical token counts, very different curves -- that difference is
the entire amortization argument, so it gets measured rather than assumed.

The honest framing this feeds: the memory layer pays indexing ONCE and then serves at its measured
q/s; full-context RAG pays its prefill on EVERY query. Below some number of queries the memory
layer has not repaid its indexing cost, and that break-even is worth stating explicitly.

Env:
  CORPUS_TOKENS  total tokens to index      (default 148584 = LongHealth)
  CHUNK          doc chunk size             (default 256)
  BATCH_CHUNKS   chunks per forward         (default 32)
  VALUE_MODEL    1 if a separate base-LM value trunk also runs (doubles the pass; default 0)

Run:  CORPUS_TOKENS=148584 uv run --no-sync python scripts/embed/bench_embed_cost.py
"""
import json
import math
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

dev = jax.devices()[0]

# Qwen3-4B, matching bench_pareto_throughput.py so the numbers compose.
D = 2560; L = 36; NH = 32; NKV = 8; HD = 128; INTER = 9728
KD = 1024; VD = 1024
REPS = int(os.environ.get("REPS", "20"))

CORPUS_TOKENS = int(os.environ.get("CORPUS_TOKENS", "148584"))
CHUNK = int(os.environ.get("CHUNK", "256"))
BATCH_CHUNKS = int(os.environ.get("BATCH_CHUNKS", "32"))
VALUE_MODEL = int(os.environ.get("VALUE_MODEL", "0"))
bf16 = jnp.bfloat16


def rnd(*shape):
    return jax.device_put(jnp.asarray(np.random.randn(*shape) * 0.02, dtype=bf16), dev)


def timeit(fn, *args, reps=REPS):
    f = jax.jit(fn)
    jax.block_until_ready(f(*args))
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); jax.block_until_ready(f(*args)); ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2] * 1e6


B = BATCH_CHUNKS
x = rnd(B, CHUNK, D)
Wq = rnd(D, NH * HD); Wk = rnd(D, NKV * HD); Wv = rnd(D, NKV * HD); Wo = rnd(NH * HD, D)
Wg = rnd(D, INTER); Wu = rnd(D, INTER); Wd = rnd(INTER, D)
Wkp = rnd(D, KD); Wvp = rnd(D, VD)


def attn_chunk(x, Wq, Wk, Wv, Wo):
    """Self-attention WITHIN one chunk. No cross-chunk attention: that is what makes indexing
    linear in corpus size while a long-context prefill is quadratic."""
    q = (x @ Wq).reshape(B, CHUNK, NH, HD).transpose(0, 2, 1, 3)
    k = (x @ Wk).reshape(B, CHUNK, NKV, HD).transpose(0, 2, 1, 3)
    v = (x @ Wv).reshape(B, CHUNK, NKV, HD).transpose(0, 2, 1, 3)
    q = q.reshape(B, NKV, NH // NKV, CHUNK, HD)
    s = jnp.einsum('bkgqh,bkth->bkgqt', q, k) / jnp.sqrt(HD)
    a = jax.nn.softmax(s, axis=-1)
    o = jnp.einsum('bkgqt,bkth->bkgqh', a, v).reshape(B, NH, CHUNK, HD)
    return o.transpose(0, 2, 1, 3).reshape(B, CHUNK, NH * HD) @ Wo


def mlp(x, Wg, Wu, Wd):
    return (jax.nn.silu(x @ Wg) * (x @ Wu)) @ Wd


def kv_proj(h, Wkp, Wvp):
    return jnp.einsum('bsh,hd->bsd', h, Wkp), jnp.einsum('bsh,hd->bsd', h, Wvp)


t_attn = timeit(attn_chunk, x, Wq, Wk, Wv, Wo)
t_mlp = timeit(mlp, x, Wg, Wu, Wd)
t_proj = timeit(kv_proj, x, Wkp, Wvp)

t_batch = L * (t_attn + t_mlp) + t_proj              # one forward over BATCH_CHUNKS chunks
if VALUE_MODEL:
    t_batch += L * (t_attn + t_mlp)                  # separate base-LM value trunk

n_chunks = math.ceil(CORPUS_TOKENS / CHUNK)
n_batches = n_chunks / BATCH_CHUNKS
t_total_us = t_batch * n_batches

out = {
    "corpus_tokens": CORPUS_TOKENS,
    "chunk": CHUNK,
    "batch_chunks": BATCH_CHUNKS,
    "n_chunks": n_chunks,
    "value_model": bool(VALUE_MODEL),
    "us": {"attn_per_batch": round(t_attn, 1), "mlp_per_batch": round(t_mlp, 1),
           "kv_proj_per_batch": round(t_proj, 1), "forward_per_batch": round(t_batch, 1)},
    "index_total_s": round(t_total_us / 1e6, 3),
    "tokens_per_s": round(CORPUS_TOKENS / (t_total_us / 1e6), 1),
    "regime": "single-chip no-TP, Qwen3-4B 36L, intra-chunk attention only",
}
print("EMBED_COST_JSON " + json.dumps(out))
print(f"  index {CORPUS_TOKENS:,} tokens ({n_chunks:,} chunks of {CHUNK}) "
      f"-> {out['index_total_s']:.2f} s  ({out['tokens_per_s']:,.0f} tok/s)")
