"""IVF-Flat prototype vs brute-force bank scan, BS=1 single-chip.

Compares the memory-layer's current retrieval (brute-force: score ALL M keys -> approx_max_k)
against an IVF-Flat index (k-means partition -> score C centroids -> probe nprobe clusters ->
score only that subset -> exact top_k), at the decode shape [B=1, N=4 mem heads].

Reuses the ORIGINAL bank for member vectors (gather by id) -> IVF adds only member_ids [C,cap]
(tiny int32) as storage, not a reordered copy. Reports median us, speedup, recall@TOPK vs exact,
extra storage, and index build time.

Run: MEMM=512000 NPROBE=16 JAX_PLATFORMS=tpu PYTHONPATH=. python scripts/embed/ivf_bench_bs1.py
"""
import os, time, json
import numpy as np
import jax, jax.numpy as jnp

jax.config.update("jax_platforms", os.environ.get("JAX_PLATFORMS", "tpu"))
dev = jax.devices()[0]

M      = int(os.environ.get("MEMM", "512000"))
KD     = 1024
VD     = 1024
N      = 4                          # mem heads (query batch at decode, B=1)
TOPK   = 128
NPROBE = int(os.environ.get("NPROBE", "16"))
C      = int(round(M ** 0.5))       # sqrt(M) centroids (classic IVF default)
KMEANS_ITERS = int(os.environ.get("KMEANS_ITERS", "3"))
CAP    = int(np.ceil(2.0 * M / C))  # per-cluster capacity (pad); ~2x mean for ~uniform k-means
REPS   = 30
bf16   = jnp.bfloat16
rng    = np.random.default_rng(0)

def timeit(fn, *a, reps=REPS):
    f = jax.jit(fn)
    jax.block_until_ready(f(*a))
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); jax.block_until_ready(f(*a)); ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2] * 1e6

# ---- bank (dummy bf16; timing is shape- not value-dependent; recall uses same data both ways) ----
mem_k_np = (rng.standard_normal((M, KD)) * 0.02).astype(np.float32)
mem_k = jax.device_put(jnp.asarray(mem_k_np, bf16), dev)
mem_v = jax.device_put(jnp.asarray(rng.standard_normal((M, VD)) * 0.02, bf16), dev)
q     = jax.device_put(jnp.asarray(rng.standard_normal((N, KD)) * 0.02, bf16), dev)

# ================= brute force (current memory layer) =================
def brute(q, mem_k):
    s = jnp.einsum('nh,mh->nm', q, mem_k) / jnp.sqrt(KD)          # [N,M]  scans WHOLE bank
    sk, idx = jax.lax.approx_max_k(s, TOPK, recall_target=0.95)
    return sk, idx
t_brute = timeit(brute, q, mem_k)

# ================= build IVF index (offline, untimed except build clock) =================
t0 = time.perf_counter()
cent = mem_k_np[rng.choice(M, C, replace=False)].astype(np.float32).copy()

@jax.jit
def assign_chunk(xc, cent):
    d = jnp.einsum('ch,kh->ck', xc, cent)   # MIPS: max inner product
    return jnp.argmax(d, axis=1).astype(jnp.int32)

labels = np.zeros(M, np.int32)
CH = 200000
for _ in range(KMEANS_ITERS):
    cent_j = jax.device_put(jnp.asarray(cent, bf16), dev)
    for s in range(0, M, CH):
        e = min(M, s + CH)
        labels[s:e] = np.asarray(assign_chunk(jax.device_put(jnp.asarray(mem_k_np[s:e], bf16), dev), cent_j))
    sums = np.zeros((C, KD), np.float32)
    np.add.at(sums, labels, mem_k_np)
    cnt = np.bincount(labels, minlength=C).astype(np.float32)
    nz = cnt > 0
    cent[nz] = sums[nz] / cnt[nz, None]     # empty clusters keep prior centroid

# pack member ids into [C, CAP] (pad w/ 0, mask via valid)
order = np.argsort(labels, kind='stable')
sl = labels[order]
pos = np.arange(M) - np.searchsorted(sl, sl)      # rank within cluster
keep = pos < CAP
rows, cols, vids = sl[keep], pos[keep], order[keep]
member_ids = np.zeros((C, CAP), np.int32)
valid = np.zeros((C, CAP), bool)
member_ids[rows, cols] = vids
valid[rows, cols] = True
dropped = int((~keep).sum())                       # points past CAP (lost from index)
build_s = time.perf_counter() - t0

cent_j = jax.device_put(jnp.asarray(cent, bf16), dev)
mid_j  = jax.device_put(jnp.asarray(member_ids), dev)
vld_j  = jax.device_put(jnp.asarray(valid), dev)

# ================= IVF query =================
def ivf(q, mem_k, cent, mid, vld):
    cs = jnp.einsum('nh,ch->nc', q, cent)          # [N,C]  score centroids (cheap)
    _, cl = jax.lax.top_k(cs, NPROBE)              # [N,NPROBE] clusters to probe
    ids = mid[cl].reshape(N, -1)                    # [N, NPROBE*CAP]
    vf  = vld[cl].reshape(N, -1)
    mk  = mem_k[ids]                                # gather ONLY subset from original bank
    s   = jnp.einsum('nph,nh->np', mk, q) / jnp.sqrt(KD)
    s   = jnp.where(vf, s, -jnp.inf)
    sk, loc = jax.lax.top_k(s, TOPK)               # exact top_k over small candidate set
    fin = jnp.take_along_axis(ids, loc, axis=1)    # [N,TOPK] final ids
    return sk, fin
t_ivf = timeit(ivf, q, mem_k, cent_j, mid_j, vld_j)

# ================= recall@TOPK vs exact (batch of queries) =================
Q = 64
qq = jax.device_put(jnp.asarray(rng.standard_normal((Q, KD)) * 0.02, bf16), dev)

def exact_topk(qq, mem_k):
    s = jnp.einsum('qh,mh->qm', qq, mem_k) / jnp.sqrt(KD)
    _, idx = jax.lax.top_k(s, TOPK)
    return idx

def ivf_q(qq, mem_k, cent, mid, vld):
    cs = jnp.einsum('qh,ch->qc', qq, cent)
    _, cl = jax.lax.top_k(cs, NPROBE)
    ids = mid[cl].reshape(qq.shape[0], -1)
    vf  = vld[cl].reshape(qq.shape[0], -1)
    mk  = mem_k[ids]
    s   = jnp.einsum('qph,qh->qp', mk, qq) / jnp.sqrt(KD)
    s   = jnp.where(vf, s, -jnp.inf)
    _, loc = jax.lax.top_k(s, TOPK)
    return jnp.take_along_axis(ids, loc, axis=1)

gt   = np.asarray(jax.jit(exact_topk)(qq, mem_k))
ivfr = np.asarray(jax.jit(ivf_q)(qq, mem_k, cent_j, mid_j, vld_j))
recall = float(np.mean([len(set(gt[i]) & set(ivfr[i])) / TOPK for i in range(Q)]))

brute_bytes = M * KD * 2
ivf_bytes   = N * NPROBE * CAP * KD * 2 + C * KD * 2   # gather subset + centroids
out = {
    "M": M, "C": C, "CAP": CAP, "NPROBE": NPROBE, "TOPK": TOPK, "kmeans_iters": KMEANS_ITERS,
    "brute_us": round(t_brute, 1), "ivf_us": round(t_ivf, 1),
    "speedup": round(t_brute / t_ivf, 2),
    "recall_at_topk": round(recall, 4),
    "brute_bytes_scanned_MB": round(brute_bytes / 1e6, 1),
    "ivf_bytes_scanned_MB": round(ivf_bytes / 1e6, 1),
    "bytes_reduction": round(brute_bytes / ivf_bytes, 1),
    "extra_storage_member_ids_MB": round(member_ids.nbytes / 1e6, 2),
    "dropped_past_cap": dropped,
    "build_s": round(build_s, 1),
}
print("IVF_JSON " + json.dumps(out))
