"""RAG same-engine decode throughput at BS=1 (repo JAX, single v6e-8).

RAG's generator is the BASE Qwen3-4B reading the top-k retrieved passages in its
context (~2-3k tokens) — no memory bank. To compare RAG against the memory-layer /
MSA on the SAME engine (repo JAX, not vLLM), we time base Qwen3-4B decode here.

Two weight layouts:
  * REPLICATED  P()                -> no cross-chip collective  (the fair "no sharding"
                                      RAG point; matches the corrected §2b number)
  * SHARDED     get_sharding_safe  -> qwen3.load's default; shards the matmul
                                      contraction on the 8-chip 'data' axis ->
                                      all-reduce per matmul per decode step (the §2b bug)

Usage:
    PYTHONPATH=. python scripts/embed/rag_repo_speed_bs1.py [ctx_len] [max_new] [reps]
Writes results/membench_bs1/bench_rag.json (when BENCH_ROOT set, writes there).
"""
import os, sys, json, time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

import models.qwen3 as q
from inference import _generate_tokens

MODEL_ID = os.environ.get("RAG_MODEL_ID", "Qwen/Qwen3-4B")
CTX_LEN  = int(sys.argv[1]) if len(sys.argv) > 1 else 2500   # top-5 passages + question
MAX_NEW  = int(sys.argv[2]) if len(sys.argv) > 2 else 512
REPS     = int(sys.argv[3]) if len(sys.argv) > 3 else 5
B        = int(os.environ.get("RAG_BS", "1"))


def replicate(weights, mesh):
    return {k: jax.device_put(v, NamedSharding(mesh, P())) for k, v in weights.items()}


def time_decode(model, prompt, pad_mask, mesh):
    # extended pad_mask for the prefill-only forward; zeros must carry the SAME
    # P('data', None) sharding as pad_mask or concatenate raises a ShardingTypeError.
    ext_zeros = jax.device_put(jnp.zeros((B, MAX_NEW), pad_mask.dtype),
                               NamedSharding(mesh, P("data", None)))
    pad_mask_ext = jnp.concatenate([pad_mask, ext_zeros], axis=1)

    def _gen():
        return _generate_tokens(
            model.forward, model.init_kv, model.weights, prompt,
            MAX_NEW, pad_mask=pad_mask, temperature=0.0, top_k=20, top_p=0.8,
        )
    # warmup (compile)
    jax.block_until_ready(_gen())
    e2e, pf = [], []
    for _ in range(REPS):
        t = time.perf_counter(); jax.block_until_ready(_gen()); e2e.append(time.perf_counter() - t)
        t = time.perf_counter()
        out = model.forward(prompt, model.weights, kv=model.init_kv(B, prompt.shape[1] + MAX_NEW),
                            pos=0, pad_mask=pad_mask_ext)
        jax.block_until_ready(out); pf.append(time.perf_counter() - t)
    e2e_m = float(np.mean(e2e[1:] or e2e)); pf_m = float(np.mean(pf[1:] or pf))
    dec_tps = MAX_NEW * B / (e2e_m - pf_m)
    return {"gen_e2e_s": round(e2e_m, 4), "prefill_s": round(pf_m, 4),
            "decode_tokens_per_s": round(dec_tps, 1), "B": B,
            "ctx_len": CTX_LEN, "max_new_tokens": MAX_NEW}


def main():
    print(f"RAG repo-engine speed: model={MODEL_ID} B={B} ctx={CTX_LEN} max_new={MAX_NEW} reps={REPS}")
    print(f"devices={jax.device_count()}")
    model = q.load(MODEL_ID, tp_devices=1)
    mesh = next(v.sharding.mesh for v in model.weights.values()
                if hasattr(getattr(v, "sharding", None), "mesh"))
    vocab = model.cfg["vocab_size"]
    rng = np.random.default_rng(0)
    prompt = jax.device_put(jnp.array(rng.integers(0, vocab, size=(B, CTX_LEN), dtype=np.int32)),
                            NamedSharding(mesh, P("data", None)))
    pad_mask = jax.device_put(jnp.ones((B, CTX_LEN), dtype=jnp.bool_),
                              NamedSharding(mesh, P("data", None)))

    out = {"model": MODEL_ID, "engine": "repo JAX (single v6e-8)"}

    sharded_weights = model.weights
    print("\n=== SHARDED (qwen3.load default; §2b bug) ===")
    out["sharded"] = time_decode(model, prompt, pad_mask, mesh)
    print(json.dumps(out["sharded"], indent=2))

    print("\n=== REPLICATED P() (no sharding; fair RAG point) ===")
    model.weights = replicate(sharded_weights, mesh)
    out["replicated"] = time_decode(model, prompt, pad_mask, mesh)
    print(json.dumps(out["replicated"], indent=2))

    root = os.environ.get("BENCH_ROOT", "results/membench_bs1")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "bench_rag.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[RAG][BENCH] wrote {path}")
    print(f"\nRAG decode tok/s (replicated, BS={B}) = {out['replicated']['decode_tokens_per_s']}")


if __name__ == "__main__":
    main()
