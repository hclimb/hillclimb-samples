"""Decode throughput: memory-layer (+optimization ladder) vs RAG-equivalent, ONE harness.

WHY NOT TIME THE REAL SYSTEMS AGAINST EACH OTHER
RAG generates through vLLM (evals/rag/generator.py); the memory model runs the repo's JAX decode
loop. Timing those head-to-head measures the SERVING STACK, not the architecture — vLLM would
likely win on engineering alone, which says nothing about memory layers. So both are composed
here from the SAME measured primitives at the SAME shapes on the SAME chip, and only what each
architecture forces into the decode step differs:

    RAG-equivalent : 36 transformer layers with KV length = question + k*doc_len,  NO memory ops
    memory layer   : 36 transformer layers with KV length = question (~64),        + memory ops

That is the real asymmetry. RAG must carry k retrieved documents in its context on every decoded
token; the memory layer keeps the prompt to the question and pays a bank scan instead.

Built on scripts/embed/profile_bs1.py (same Qwen3-4B shapes, same timing method, same B=1
single-chip no-TP regime), extended with the levers the Pareto plot needs:

    TOPK_MODE = exact | approx     jax.lax.top_k vs approx_max_k (recall_target)
    KEY_DTYPE = bf16  | int8       quantized contiguous score-scan (see quant_bench_bs1.py)

⚠️ SCALE CAVEAT, from the existing plot's own notes: at B=1 decode is transformer-bound, so int8
on the key scan buys ~1% at a 512k bank and ~4% at 4M. The ladder only separates visibly at large
banks. Report the bank size with every point; do not present a small-bank gain as general.

Run:  MODE=mem  MEMM=512000 KVLEN=64   TOPK_MODE=approx KEY_DTYPE=int8 \
        JAX_PLATFORMS=tpu PYTHONPATH=. python scripts/embed/bench_pareto_throughput.py
      MODE=rag  KVLEN=5160                                            (= 64 + 5*1024)
"""
import json
import math
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_platforms", os.environ.get("JAX_PLATFORMS", "tpu"))
dev = jax.devices()[0]

# Qwen3-4B, matching profile_bs1.py exactly so numbers are comparable across both scripts.
D = 2560; L = 36; NH = 32; NKV = 8; HD = 128; INTER = 9728; V = 151936
N = 4; KD = 1024; VD = 1024
B = 1
REPS = int(os.environ.get("REPS", "50"))

MODE      = os.environ.get("MODE", "mem")            # mem | rag
GEN_TOKENS = int(os.environ.get("GEN_TOKENS", "0"))  # >0 => also report END-TO-END queries/sec
M         = int(os.environ.get("MEMM", "512000"))    # bank slots (mem only)
TOPK      = int(os.environ.get("TOPK", "128"))
KVLEN     = int(os.environ.get("KVLEN", "64"))       # context tokens in the KV cache
TOPK_MODE = os.environ.get("TOPK_MODE", "approx")    # exact | approx
KEY_DTYPE = os.environ.get("KEY_DTYPE", "bf16")      # bf16 | int8
RECALL    = float(os.environ.get("RECALL", "0.95"))
N_MEM_LAYERS = int(os.environ.get("N_MEM_LAYERS", "1"))  # 1 (hard-neg) or 4 (ground4layer)
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
    return ts[len(ts) // 2] * 1e6          # median microseconds


x = rnd(B, 1, D)

# ---------------- transformer decode layer (attention over KVLEN + SwiGLU MLP) --------------
Wq = rnd(D, NH*HD); Wk = rnd(D, NKV*HD); Wv = rnd(D, NKV*HD); Wo = rnd(NH*HD, D)
Kc = rnd(B, NKV, KVLEN, HD); Vc = rnd(B, NKV, KVLEN, HD)
Wg = rnd(D, INTER); Wu = rnd(D, INTER); Wd = rnd(INTER, D)


def attn(x, Wq, Wk, Wv, Wo, Kc, Vc):
    q = (x @ Wq).reshape(B, 1, NH, HD)
    k = (x @ Wk).reshape(B, 1, NKV, HD); v = (x @ Wv).reshape(B, 1, NKV, HD)
    K = jnp.concatenate([Kc, k.transpose(0, 2, 1, 3)], axis=2)
    Vv = jnp.concatenate([Vc, v.transpose(0, 2, 1, 3)], axis=2)
    q = q.transpose(0, 2, 1, 3).reshape(B, NKV, NH//NKV, 1, HD)
    s = jnp.einsum('bkgqh,bkth->bkgqt', q, K) / jnp.sqrt(HD)
    a = jax.nn.softmax(s, axis=-1)
    o = jnp.einsum('bkgqt,bkth->bkgqh', a, Vv).reshape(B, NH, 1, HD).transpose(0, 2, 1, 3).reshape(B, 1, NH*HD)
    return o @ Wo


def mlp(x, Wg, Wu, Wd):
    return (jax.nn.silu(x @ Wg) * (x @ Wu)) @ Wd


t_attn = timeit(attn, x, Wq, Wk, Wv, Wo, Kc, Vc)
t_mlp  = timeit(mlp, x, Wg, Wu, Wd)
t_layer = t_attn + t_mlp

Wemb = rnd(V, D); Wlm = rnd(D, V); tok = jax.device_put(jnp.zeros((B, 1), jnp.int32), dev)
t_emb = timeit(lambda t, W: W[t], tok, Wemb)
t_lm  = timeit(lambda x, W: x @ W, x, Wlm)

# ---------------- memory ops (only for MODE=mem) --------------------------------------------
t_mem = 0.0
mem_detail = {}
if MODE == "mem":
    Wmq = rnd(N, KD, D); Wmo = rnd(D, N, VD)
    mem_v = rnd(M, VD)
    if KEY_DTYPE == "int8":
        # Store keys as int8 + a per-row scale; the scan becomes an int8 matmul over a
        # contiguous buffer (1 byte/elem vs 2), which is the only lever that helps a scan that
        # is HBM-bandwidth-bound. Accuracy cost is real but small: quant_bench_bs1.py measures
        # recall@128 ~0.97 against the exact bf16 top-128.
        kf = np.random.randn(M, KD).astype(np.float32) * 0.02
        scale = np.abs(kf).max(axis=1, keepdims=True) / 127.0
        mem_k_i8 = jax.device_put(jnp.asarray(np.round(kf / scale).astype(np.int8)), dev)
        mem_scale = jax.device_put(jnp.asarray(scale.astype(np.float32)), dev)

        def mem_score(q, k_i8, sc):
            qi = jnp.asarray(q, jnp.int8)                       # shape-accurate cost model
            s = jnp.einsum('btnh,mh->bntm', qi, k_i8, preferred_element_type=jnp.int32)
            return s.astype(jnp.float32) * sc.reshape(1, 1, 1, -1) / jnp.sqrt(KD)
        score_args = (rnd(B, 1, N, KD), mem_k_i8, mem_scale)
    else:
        mem_k = rnd(M, KD)

        def mem_score(q, k, _unused=None):
            return jnp.einsum('btnh,mh->bntm', q, k) / jnp.sqrt(KD)
        score_args = (rnd(B, 1, N, KD), mem_k, None)

    def _score_call(*a):
        return mem_score(*a) if a[-1] is not None else mem_score(a[0], a[1])

    q0 = score_args[0]
    scores = _score_call(*score_args)

    if TOPK_MODE == "exact":
        topk_fn = lambda s: jax.lax.top_k(s, TOPK)
    else:
        topk_fn = lambda s: jax.lax.approx_max_k(s, TOPK, recall_target=RECALL)
    sk, idx = topk_fn(scores)

    t_mq    = timeit(lambda x, W: jnp.einsum('btd,nhd->btnh', x, W), x, rnd(N, KD, D))
    t_score = timeit(_score_call, *score_args)
    t_topk  = timeit(topk_fn, scores)
    t_gath  = timeit(lambda v, i: v[i], mem_v, idx)
    valsk   = mem_v[idx]
    t_comb  = timeit(lambda s, v: jnp.einsum('bntk,bntkv->btnv', jax.nn.softmax(s, axis=-1), v), sk, valsk)
    t_mo    = timeit(lambda y, W: jnp.einsum('btnv,dnv->btd', y, W),
                     jnp.einsum('bntk,bntkv->btnv', jax.nn.softmax(sk, axis=-1), valsk), Wmo)

    t_mem_one = t_mq + t_score + t_topk + t_gath + t_comb + t_mo
    t_mem = t_mem_one * N_MEM_LAYERS       # ground4layer has memory at 4 layers, not 1
    mem_detail = {"q_proj": round(t_mq, 1), "score_scan": round(t_score, 1),
                  "topk": round(t_topk, 1), "gather_v": round(t_gath, 1),
                  "softmax_combine": round(t_comb, 1), "o_proj": round(t_mo, 1),
                  "per_mem_layer": round(t_mem_one, 1), "n_mem_layers": N_MEM_LAYERS}

total = L * t_layer + t_emb + t_lm + t_mem

# ---------------- PREFILL, and why it matters ----------------------------------------------
# Per-token DECODE throughput amortises the prompt away, which flatters RAG enormously: at
# doc_len 16384 it must process 5*16384 = 81,920 prompt tokens ONCE per query before emitting
# anything, while the memory layer processes ~64. A QA workload generates ~100 tokens per query,
# so prefill dominates the end-to-end cost and a tok/s-decode plot hides the entire asymmetry.
# Prefill is compute-bound (a full forward over KVLEN tokens) rather than memory-bound, so it is
# modelled here as KVLEN parallel token-steps through the 36 layers, measured at the real shape.
t_prefill = 0.0
prefill_model = None
if GEN_TOKENS > 0:
    PF = max(1, KVLEN)
    # CHUNKED CAUSAL prefill, the way a real serving stack does it: query chunk c attends only to
    # the keys written so far, and nothing ever materialises a PF x PF score matrix.
    #
    # The earlier single-shot version did materialise it, and that was wrong in RAG's DISFAVOUR
    # twice over: it charged full non-causal attention (~2x the real FLOPs) plus the HBM traffic of
    # a 27 GB score tensor at doc_len 4096, and it OOM'd outright at doc_len 16384 (the score
    # tensor would have been ~430 TB). Both errors inflate RAG's prefill, i.e. they flatter the
    # memory layer -- which is exactly the direction this comparison must not be wrong in.
    CHUNK = min(int(os.environ.get("PREFILL_CHUNK", "1024")), PF)
    nchunks = math.ceil(PF / CHUNK)

    xpf = rnd(B, CHUNK, D)

    def attn_pf(x, Wq, Wk, Wv, Wo, Kc, Vc):
        q = (x @ Wq).reshape(B, CHUNK, NH, HD).transpose(0, 2, 1, 3)
        q = q.reshape(B, NKV, NH//NKV, CHUNK, HD)
        s = jnp.einsum('bkgqh,bkth->bkgqt', q, Kc) / jnp.sqrt(HD)
        a = jax.nn.softmax(s, axis=-1)
        o = jnp.einsum('bkgqt,bkth->bkgqh', a, Vc).reshape(B, NH, CHUNK, HD)
        return o.transpose(0, 2, 1, 3).reshape(B, CHUNK, NH*HD) @ Wo

    # Per-chunk attention time is linear in the KV length that chunk sees, so probe a few lengths
    # and fit rather than timing all nchunks (40 of them at doc_len 16384) separately.
    probes = sorted({min(PF, CHUNK * m)
                     for m in {1, max(1, nchunks // 4), max(1, nchunks // 2), nchunks}})
    t_probe = []
    for Klen in probes:
        Kpf = rnd(B, NKV, Klen, HD); Vpf = rnd(B, NKV, Klen, HD)
        t_probe.append(timeit(attn_pf, xpf, Wq, Wk, Wv, Wo, Kpf, Vpf, reps=max(5, REPS // 10)))
        del Kpf, Vpf

    # The last chunk is usually PARTIAL (PF is rarely a multiple of CHUNK), and charging it as a
    # full chunk overcharges RAG worst at SHORT doc lengths -- 1.52x at doc_len 256 (PF=1344 billed
    # as 2048 query-tokens) versus 1.01x at doc_len 16384. That is the memory layer's favour again,
    # precisely where RAG should be winning, so weight each chunk by its real token count. Both
    # attention and MLP are linear in the number of query tokens, hence the plain size ratio.
    def chunk_tokens(c):
        return min(CHUNK, PF - c * CHUNK)

    if len(probes) > 1:
        fit = np.polyfit(probes, t_probe, 1)
        t_attn_pf = sum(float(np.polyval(fit, min((c + 1) * CHUNK, PF))) * chunk_tokens(c) / CHUNK
                        for c in range(nchunks))
    else:
        t_attn_pf = t_probe[0] * sum(chunk_tokens(c) / CHUNK for c in range(nchunks))
    t_mlp_pf = timeit(mlp, xpf, Wg, Wu, Wd, reps=max(5, REPS // 10)) * (PF / CHUNK)
    t_prefill = L * (t_attn_pf + t_mlp_pf)
    prefill_model = {"chunk": CHUNK, "n_chunks": nchunks, "causal": True,
                     "prompt_tokens": PF, "billed_tokens": round(CHUNK * sum(
                         chunk_tokens(c) / CHUNK for c in range(nchunks)), 1),
                     "probe_kv_lens": probes, "probe_us": [round(t, 1) for t in t_probe]}

    out_e2e = {
        "prefill_us": round(t_prefill, 1),
        "decode_us_per_tok": round(total, 1),
        "gen_tokens": GEN_TOKENS,
        "query_us": round(t_prefill + GEN_TOKENS * total, 1),
        "queries_per_s": round(1e6 / (t_prefill + GEN_TOKENS * total), 3),
        "prefill_model": prefill_model,
    }

out = {
    "mode": MODE,
    "config": {"kvlen": KVLEN, "bank_slots": M if MODE == "mem" else 0, "topk": TOPK,
               "topk_mode": TOPK_MODE if MODE == "mem" else None,
               "key_dtype": KEY_DTYPE if MODE == "mem" else None,
               "n_mem_layers": N_MEM_LAYERS if MODE == "mem" else 0,
               "regime": "B=1 single-chip no-TP, Qwen3-4B 36L"},
    "us": {"one_full_layer": round(t_layer, 1), "all_36_layers": round(L * t_layer, 1),
           "memory_total": round(t_mem, 1), "embed": round(t_emb, 1), "lm_head": round(t_lm, 1),
           "STEP_TOTAL": round(total, 1)},
    "memory_pct_of_step": round(100 * t_mem / total, 2) if t_mem else 0.0,
    "decode_tok_s": round(1e6 / total, 1),
}
if mem_detail:
    out["memory_ops_us"] = mem_detail
if GEN_TOKENS > 0:
    out["end_to_end"] = out_e2e
print("PARETO_TP_JSON " + json.dumps(out))
