"""Contiguous-layout IVF vs brute-force, BS=1 single-chip.

Fix for the scatter-gather tax: after k-means, REORDER the bank so each cluster's vectors are
a contiguous block. Store keys as [C, CAP, KD] (cluster-major, padded). Probing nprobe clusters
then reads nprobe CONTIGUOUS [CAP,KD] blocks (full-BW streaming) instead of gathering CAP scattered
rows from the original bank. mem_v stays as the original [M,VD] (final gather is only TOPK rows).

Run: MEMM=2000000 NPROBE=16 JAX_PLATFORMS=tpu $VENV/python scripts/embed/ivf_contig_bench_bs1.py
"""
import os, time, json
import numpy as np
import jax, jax.numpy as jnp

jax.config.update("jax_platforms", os.environ.get("JAX_PLATFORMS", "tpu"))
dev = jax.devices()[0]

M      = int(os.environ.get("MEMM", "2000000"))
KD     = 1024
VD     = 1024
N      = 4
TOPK   = 128
NPROBE = int(os.environ.get("NPROBE", "16"))
C      = int(round(M ** 0.5))
KMEANS_ITERS = int(os.environ.get("KMEANS_ITERS", "3"))
CAPF   = float(os.environ.get("CAPF", "1.35"))     # cap = CAPF * mean cluster size
CAP    = int(np.ceil(CAPF * M / C))
RECALL = M <= 2_000_000                             # skip exact-ref recall above 2M (HBM)
REPS   = 30
bf16   = jnp.bfloat16
rng    = np.random.default_rng(0)

def timeit(fn, *a, reps=REPS):
    f = jax.jit(fn); jax.block_until_ready(f(*a))
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); jax.block_until_ready(f(*a)); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts) // 2] * 1e6

mem_k_np = (rng.standard_normal((M, KD)) * 0.02).astype(np.float32)
mem_v = jax.device_put(jnp.asarray(rng.standard_normal((M, VD)) * 0.02, bf16), dev)
q     = jax.device_put(jnp.asarray(rng.standard_normal((N, KD)) * 0.02, bf16), dev)

# ================= brute (current) =================
mem_k = jax.device_put(jnp.asarray(mem_k_np, bf16), dev)
def brute(q, mem_k):
    s = jnp.einsum('nh,mh->nm', q, mem_k) / jnp.sqrt(KD)
    return jax.lax.approx_max_k(s, TOPK, recall_target=0.95)
t_brute = timeit(brute, q, mem_k)

# ================= build: k-means + CONTIGUOUS cluster blocks =================
t0 = time.perf_counter()
cent = mem_k_np[rng.choice(M, C, replace=False)].astype(np.float32).copy()

@jax.jit
def assign_chunk(xc, cent):
    d = jnp.einsum('ch,kh->ck', xc, cent)
    return jnp.argmax(d, axis=1).astype(jnp.int32)

labels = np.zeros(M, np.int32); CH = 200000
for _ in range(KMEANS_ITERS):
    cent_j = jax.device_put(jnp.asarray(cent, bf16), dev)
    for s in range(0, M, CH):
        e = min(M, s + CH)
        labels[s:e] = np.asarray(assign_chunk(jax.device_put(jnp.asarray(mem_k_np[s:e], bf16), dev), cent_j))
    sums = np.zeros((C, KD), np.float32); np.add.at(sums, labels, mem_k_np)
    cnt = np.bincount(labels, minlength=C).astype(np.float32); nz = cnt > 0
    cent[nz] = sums[nz] / cnt[nz, None]

order = np.argsort(labels, kind='stable'); sl = labels[order]
pos = np.arange(M) - np.searchsorted(sl, sl)
keep = pos < CAP
rows, cols, vids = sl[keep], pos[keep], order[keep]
dropped = int((~keep).sum())

# contiguous blocks: keys placed cluster-major; id_blk maps back to original row
kb_np = np.zeros((C, CAP, KD), np.float32)
id_blk = np.zeros((C, CAP), np.int32)
valid  = np.zeros((C, CAP), bool)
kb_np[rows, cols] = mem_k_np[vids]
id_blk[rows, cols] = vids
valid[rows, cols] = True
build_s = time.perf_counter() - t0

cent_j = jax.device_put(jnp.asarray(cent, bf16), dev)
kb_j   = jax.device_put(jnp.asarray(kb_np, bf16), dev)      # [C,CAP,KD] contiguous
idb_j  = jax.device_put(jnp.asarray(id_blk), dev)
vld_j  = jax.device_put(jnp.asarray(valid), dev)
del kb_np

# ================= contiguous-IVF query =================
def ivf_c(q, kb, idb, vld, cent, mem_v):
    cs = jnp.einsum('nh,ch->nc', q, cent)
    _, cl = jax.lax.top_k(cs, NPROBE)              # [N,NPROBE]
    kk = kb[cl]                                     # [N,NPROBE,CAP,KD]  <- contiguous block gather
    s  = jnp.einsum('npch,nh->npc', kk, q) / jnp.sqrt(KD)
    vf = vld[cl]
    s  = jnp.where(vf, s, -jnp.inf).reshape(N, -1)  # [N, NPROBE*CAP]
    ids = idb[cl].reshape(N, -1)
    sk, loc = jax.lax.top_k(s, TOPK)
    fin = jnp.take_along_axis(ids, loc, axis=1)      # [N,TOPK] original ids
    vals = mem_v[fin]                                # tiny final gather (128 rows)
    return sk, fin
t_ivf = timeit(ivf_c, q, kb_j, idb_j, vld_j, cent_j, mem_v)

# ================= recall vs exact (<=2M only) =================
recall = -1.0
if RECALL:
    Q = 64
    qq = jax.device_put(jnp.asarray(rng.standard_normal((Q, KD)) * 0.02, bf16), dev)
    def exact_topk(qq, mem_k):
        s = jnp.einsum('qh,mh->qm', qq, mem_k) / jnp.sqrt(KD)
        return jax.lax.top_k(s, TOPK)[1]
    def ivf_q(qq, kb, idb, vld, cent):
        cs = jnp.einsum('qh,ch->qc', qq, cent)
        _, cl = jax.lax.top_k(cs, NPROBE)
        kk = kb[cl]
        s  = jnp.einsum('qpch,qh->qpc', kk, qq) / jnp.sqrt(KD)
        s  = jnp.where(vld[cl], s, -jnp.inf).reshape(qq.shape[0], -1)
        ids = idb[cl].reshape(qq.shape[0], -1)
        return jnp.take_along_axis(ids, jax.lax.top_k(s, TOPK)[1], axis=1)
    gt   = np.asarray(jax.jit(exact_topk)(qq, mem_k))
    ivfr = np.asarray(jax.jit(ivf_q)(qq, kb_j, idb_j, vld_j, cent_j))
    recall = float(np.mean([len(set(gt[i]) & set(ivfr[i])) / TOPK for i in range(Q)]))

ivf_bytes = N * NPROBE * CAP * KD * 2 + C * KD * 2
out = {
    "M": M, "C": C, "CAP": CAP, "NPROBE": NPROBE, "TOPK": TOPK, "layout": "contiguous_blocks",
    "brute_us": round(t_brute, 1), "ivf_us": round(t_ivf, 1), "speedup": round(t_brute / t_ivf, 2),
    "recall_at_topk": round(recall, 4),
    "ivf_bytes_scanned_MB": round(ivf_bytes / 1e6, 1),
    "bytes_reduction": round((M * KD * 2) / ivf_bytes, 1),
    "block_storage_GB": round(C * CAP * KD * 2 / 1e9, 2),
    "dropped_past_cap": dropped, "build_s": round(build_s, 1),
}
print("IVFC_JSON " + json.dumps(out))
