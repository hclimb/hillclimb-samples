"""Quantized contiguous-scan retrieval vs brute bf16, BS=1 single-chip.

The score_scan is HBM-bandwidth-bound at near-peak. Only lever that beats it = FEWER BYTES,
STAYING CONTIGUOUS (no dynamic gather). Two families, selected by MODE:

  int8 / int4 : store mem_k quantized -> contiguous int matmul (1B / 0.5B per elem vs bf16 2B).
  pq64 / pq128: product-quantize keys to m 1-byte codes -> ADC (precompute q.codebook LUT,
                scan codes contiguously, sum LUT lookups). Heavy scan = M*m bytes, no vector gather.

Recall = overlap of top-128 vs EXACT bf16 top-128 (quant error is real on any data).

Run: MODE=int8 MEMM=2000000 JAX_PLATFORMS=tpu $VENV/python scripts/embed/quant_bench_bs1.py
"""
import os, time, json
import numpy as np
import jax, jax.numpy as jnp

jax.config.update("jax_platforms", os.environ.get("JAX_PLATFORMS", "tpu"))
dev = jax.devices()[0]

MODE = os.environ.get("MODE", "int8")
M    = int(os.environ.get("MEMM", "2000000"))
KD   = 1024; VD = 1024; N = 4; TOPK = 128
KM_PQ = int(os.environ.get("KM_PQ", "2"))
REPS = 30
bf16 = jnp.bfloat16
rng  = np.random.default_rng(0)

def timeit(fn, *a, reps=REPS):
    f = jax.jit(fn); jax.block_until_ready(f(*a))
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); jax.block_until_ready(f(*a)); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts) // 2] * 1e6

mem_k_np = (rng.standard_normal((M, KD)) * 0.02).astype(np.float32)
q_np     = (rng.standard_normal((N, KD)) * 0.02).astype(np.float32)
q  = jax.device_put(jnp.asarray(q_np, bf16), dev)

# ---- brute bf16 baseline ----
mem_k = jax.device_put(jnp.asarray(mem_k_np, bf16), dev)
def brute(q, mem_k):
    s = jnp.einsum('nh,mh->nm', q, mem_k) / jnp.sqrt(KD)
    return jax.lax.approx_max_k(s, TOPK, recall_target=0.95)
t_brute = timeit(brute, q, mem_k)

# ---- exact bf16 top-k ground truth (recall ref) ----
Qn = 64
qq_np = (rng.standard_normal((Qn, KD)) * 0.02).astype(np.float32)
qq = jax.device_put(jnp.asarray(qq_np, bf16), dev)
def exact_topk(qq, mem_k):
    return jax.lax.top_k(jnp.einsum('qh,mh->qm', qq, mem_k), TOPK)[1]
gt = np.asarray(jax.jit(exact_topk)(qq, mem_k))
def recall_of(ids):
    return float(np.mean([len(set(gt[i]) & set(ids[i])) / TOPK for i in range(Qn)]))

info = {"mode": MODE, "M": M, "TOPK": TOPK, "brute_us": round(t_brute, 1)}

if MODE in ("int8", "int4"):
    qmax = 127 if MODE == "int8" else 7
    idt  = jnp.int8 if MODE == "int8" else jnp.int4
    sk   = (np.abs(mem_k_np).max(1, keepdims=True) / qmax).astype(np.float32)   # per-row scale [M,1]
    k_q  = np.round(mem_k_np / sk).clip(-qmax, qmax).astype(np.int8)
    mem_k_q = jax.device_put(jnp.asarray(k_q).astype(idt), dev)
    sk_j = jax.device_put(jnp.asarray(sk[:, 0], bf16), dev)                     # [M]
    def qscore(qf, mem_k_q, sk_j):
        qs  = jnp.abs(qf).max() / qmax
        q_q = jnp.round(qf / qs).astype(idt)
        s   = jnp.einsum('nh,mh->nm', q_q, mem_k_q, preferred_element_type=jnp.int32)
        s   = s.astype(jnp.float32) * (qs.astype(jnp.float32) * sk_j.astype(jnp.float32)[None, :]) / jnp.sqrt(KD)
        return jax.lax.approx_max_k(s, TOPK, recall_target=0.95)
    t_m = timeit(qscore, q, mem_k_q, sk_j)
    # recall (batch)
    def qscore_b(qf, mem_k_q, sk_j):
        qs = jnp.abs(qf).max() / qmax
        q_q = jnp.round(qf / qs).astype(idt)
        s = jnp.einsum('qh,mh->qm', q_q, mem_k_q, preferred_element_type=jnp.int32)
        s = s.astype(jnp.float32) * (qs.astype(jnp.float32) * sk_j.astype(jnp.float32)[None, :])
        return jax.lax.top_k(s, TOPK)[1]
    ids = np.asarray(jax.jit(qscore_b)(qq, mem_k_q, sk_j))
    bytes_scanned = M * KD * (1 if MODE == "int8" else 0.5)
    info.update(method_us=round(t_m, 1), speedup=round(t_brute / t_m, 2),
                recall_at_topk=round(recall_of(ids), 4),
                bytes_scanned_MB=round(bytes_scanned / 1e6, 1),
                bytes_reduction=round((M * KD * 2) / bytes_scanned, 1))

elif MODE in ("pq64", "pq128"):
    m   = 64 if MODE == "pq64" else 128
    sub = KD // m
    Xr  = mem_k_np.reshape(M, m, sub)
    cb  = np.zeros((m, 256, sub), np.float32)
    codes_cm = np.zeros((m, M), np.uint8)   # column-major: codes_cm[j] contiguous [M]

    @jax.jit
    def assign_sub(xj, cj):                  # xj [chunk,sub], cj [256,sub] -> nearest idx
        d = -2 * jnp.einsum('xs,cs->xc', xj, cj) + jnp.sum(cj * cj, 1)[None, :]
        return jnp.argmin(d, 1).astype(jnp.uint8)
    t0 = time.perf_counter(); CH = 200000
    for j in range(m):
        Xj = Xr[:, j, :]
        cj = Xj[rng.choice(M, 256, replace=False)].astype(np.float32).copy()
        for _ in range(KM_PQ):
            cj_j = jax.device_put(jnp.asarray(cj, bf16), dev)
            asg = np.empty(M, np.uint8)
            for s in range(0, M, CH):
                e = min(M, s + CH)
                asg[s:e] = np.asarray(assign_sub(jax.device_put(jnp.asarray(Xj[s:e], bf16), dev), cj_j))
            sums = np.zeros((256, sub), np.float32); np.add.at(sums, asg, Xj)
            cnt = np.bincount(asg, minlength=256).astype(np.float32); nz = cnt > 0
            cj[nz] = sums[nz] / cnt[nz, None]
        cb[j] = cj; codes_cm[j] = asg
    build_s = time.perf_counter() - t0

    cb_j = jax.device_put(jnp.asarray(cb, bf16), dev)               # [m,256,sub]
    codes_j = jax.device_put(jnp.asarray(codes_cm), dev)           # [m,M] uint8

    def adc(qf, cb, codes):                                         # qf [N,KD]
        qr = qf.reshape(N, m, sub)
        LUT = jnp.einsum('nms,mcs->nmc', qr, cb)                    # [N,m,256]
        s = jnp.zeros((N, M), jnp.float32)
        for j in range(m):                                          # unrolled: m tiny-LUT gathers
            s = s + LUT[:, j, :][:, codes[j]]                       # [N,256] gather by codes[j] [M] -> [N,M]
        return jax.lax.approx_max_k(s / jnp.sqrt(KD), TOPK, recall_target=0.95)
    t_m = timeit(adc, q, cb_j, codes_j)

    def adc_b(qf, cb, codes):
        qr = qf.reshape(qf.shape[0], m, sub)
        LUT = jnp.einsum('nms,mcs->nmc', qr, cb)
        s = jnp.zeros((qf.shape[0], M), jnp.float32)
        for j in range(m):
            s = s + LUT[:, j, :][:, codes[j]]
        return jax.lax.top_k(s, TOPK)[1]
    ids = np.asarray(jax.jit(adc_b)(qq, cb_j, codes_j))
    bytes_scanned = M * m
    info.update(m=m, sub=sub, method_us=round(t_m, 1), speedup=round(t_brute / t_m, 2),
                recall_at_topk=round(recall_of(ids), 4),
                code_storage_MB=round(codes_cm.nbytes / 1e6, 1),
                bytes_scanned_MB=round(bytes_scanned / 1e6, 1),
                bytes_reduction=round((M * KD * 2) / bytes_scanned, 1),
                build_s=round(build_s, 1))

print("QUANT_JSON " + json.dumps(info))
