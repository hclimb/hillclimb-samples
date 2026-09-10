"""BS=1, single-chip, no-TP per-component profile of the memory-layer decode step.

Micro-benchmarks each op at the decode shape [B=1, 1 token] with a realistic KV length,
using real shapes (Qwen3-4B + memory layer) and dummy bf16 weights (timing is shape-, not
value-dependent). Composes: layers BEFORE memory (0..13), the memory-layer sub-ops (layer 14),
layers AFTER memory (15..35) + final norm + lm_head. Single device (no mesh) => true B=1, no TP.

Run: JAX_PLATFORMS=tpu PYTHONPATH=. python scripts/embed/profile_bs1.py
"""
import os, time, json
import numpy as np
import jax, jax.numpy as jnp

# pin to ONE chip (true BS=1, no TP/data-parallel)
jax.config.update("jax_platforms", os.environ.get("JAX_PLATFORMS", "tpu"))
dev = jax.devices()[0]

# ---- Qwen3-4B config ----
D = 2560; L = 36; NH = 32; NKV = 8; HD = 128; INTER = 9728; V = 151936
MEM_LAYER = 14; N = 4; KD = 1024; VD = 1024; M = int(os.environ.get("MEMM", "512000")); TOPK = 128
KVLEN = int(os.environ.get("KVLEN", "2560"))   # prompt/context length in the KV cache
B = 1
REPS = 50
bf16 = jnp.bfloat16

def rnd(*shape):
    return jax.device_put(jnp.asarray(np.random.randn(*shape) * 0.02, dtype=bf16), dev)

def timeit(fn, *args, reps=REPS):
    f = jax.jit(fn)
    jax.block_until_ready(f(*args))  # warmup/compile
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); jax.block_until_ready(f(*args)); ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts)//2] * 1e6  # median microseconds

x = rnd(B, 1, D)  # one decode token

# ---- transformer decode layer (attention over KVLEN + SwiGLU MLP) ----
Wq = rnd(D, NH*HD); Wk = rnd(D, NKV*HD); Wv = rnd(D, NKV*HD); Wo = rnd(NH*HD, D)
Kc = rnd(B, NKV, KVLEN, HD); Vc = rnd(B, NKV, KVLEN, HD)
Wg = rnd(D, INTER); Wu = rnd(D, INTER); Wd = rnd(INTER, D)

def attn(x, Wq, Wk, Wv, Wo, Kc, Vc):
    q = (x @ Wq).reshape(B, 1, NH, HD)
    k = (x @ Wk).reshape(B, 1, NKV, HD); v = (x @ Wv).reshape(B, 1, NKV, HD)
    K = jnp.concatenate([Kc, k.transpose(0,2,1,3)], axis=2)
    Vv = jnp.concatenate([Vc, v.transpose(0,2,1,3)], axis=2)
    q = q.transpose(0,2,1,3).reshape(B, NKV, NH//NKV, 1, HD)
    s = jnp.einsum('bkgqh,bkth->bkgqt', q, K) / jnp.sqrt(HD)
    a = jax.nn.softmax(s, axis=-1)
    o = jnp.einsum('bkgqt,bkth->bkgqh', a, Vv).reshape(B, NH, 1, HD).transpose(0,2,1,3).reshape(B,1,NH*HD)
    return o @ Wo

def mlp(x, Wg, Wu, Wd):
    return (jax.nn.silu(x @ Wg) * (x @ Wu)) @ Wd

def layer(x, Wq,Wk,Wv,Wo,Kc,Vc, Wg,Wu,Wd):
    return x + attn(x, Wq,Wk,Wv,Wo,Kc,Vc) + mlp(x + attn(x,Wq,Wk,Wv,Wo,Kc,Vc), Wg,Wu,Wd)

t_attn = timeit(attn, x, Wq,Wk,Wv,Wo,Kc,Vc)
t_mlp  = timeit(mlp,  x, Wg,Wu,Wd)
t_layer = t_attn + t_mlp

# ---- memory layer sub-ops (layer 14) ----
Wmq = rnd(N, KD, D); Wmo = rnd(D, N, VD)
mem_k = rnd(M, KD); mem_v = rnd(M, VD)
gnorm = rnd(KD)

def mem_qproj(x, Wmq):
    return jnp.einsum('btd,nhd->btnh', x, Wmq)
def mem_score(q, mem_k):
    return jnp.einsum('btnh,mh->bntm', q, mem_k) / jnp.sqrt(KD)
def mem_topk(scores):
    return jax.lax.approx_max_k(scores, TOPK, recall_target=0.95)
def mem_gather(mem_v, idx):
    return mem_v[idx]  # [B,N,1,TOPK,VD]
def mem_combine(scoresk, valsk):
    w = jax.nn.softmax(scoresk, axis=-1)
    return jnp.einsum('bntk,bntkv->btnv', w, valsk)
def mem_oproj(y, Wmo):
    return jnp.einsum('btnv,dnv->btd', y, Wmo)

q = jnp.einsum('btd,nhd->btnh', x, Wmq)               # [B,1,N,KD]
scores = jnp.einsum('btnh,mh->bntm', q, mem_k)        # [B,N,1,M]
sk, idx = jax.lax.approx_max_k(scores, TOPK, recall_target=0.95)
valsk = mem_v[idx]                                    # [B,N,1,TOPK,VD]

t_mq   = timeit(mem_qproj, x, Wmq)
t_score= timeit(mem_score, q, mem_k)
t_topk = timeit(lambda s: jax.lax.approx_max_k(s, TOPK, recall_target=0.95), scores)
t_gath = timeit(mem_gather, mem_v, idx)
t_comb = timeit(mem_combine, sk, valsk)
t_mo   = timeit(mem_oproj, mem_combine(sk, valsk), Wmo)
t_mem  = t_mq + t_score + t_topk + t_gath + t_comb + t_mo

# ---- embed + lm_head ----
Wemb = rnd(V, D); Wlm = rnd(D, V); tok = jax.device_put(jnp.zeros((B,1), jnp.int32), dev)
t_emb = timeit(lambda t, W: W[t], tok, Wemb)
t_lm  = timeit(lambda x, W: x @ W, x, Wlm)

before = 14 * t_layer
after  = 21 * t_layer + t_lm     # layers 15..35 + lm_head
total  = before + t_layer + t_mem + after + t_emb  # layer14 = t_layer(attn+mlp) + t_mem

out = {
  "regime": f"B=1 single-chip no-TP, KVLEN={KVLEN}, Qwen3-4B 36L, mem@L14, M={M}, topk={TOPK}",
  "per_op_us": {"attn_layer": round(t_attn,1), "mlp_layer": round(t_mlp,1), "one_full_layer": round(t_layer,1),
                "mem_q_proj": round(t_mq,1), "mem_score_scan": round(t_score,1), "mem_approx_topk": round(t_topk,1),
                "mem_gather_v": round(t_gath,1), "mem_softmax_combine": round(t_comb,1), "mem_o_proj": round(t_mo,1),
                "embed": round(t_emb,1), "lm_head": round(t_lm,1)},
  "grouped_us": {"layers_0_13_before": round(before,1),
                 "layer14_transformer(attn+mlp)": round(t_layer,1),
                 "layer14_memory_TOTAL": round(t_mem,1),
                 "layers_15_35_after+lmhead": round(after,1),
                 "embed": round(t_emb,1),
                 "STEP_TOTAL": round(total,1)},
  "memory_pct_of_step": round(100*t_mem/total,2),
  "decode_tok_s_est": round(1e6/total,1),
}
print("PROFILE_JSON " + json.dumps(out))
