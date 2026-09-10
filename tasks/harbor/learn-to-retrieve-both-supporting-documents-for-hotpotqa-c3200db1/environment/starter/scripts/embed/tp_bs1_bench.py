"""BS=1 decode throughput vs tensor-parallelism for the base Qwen3-4B transformer.

At B=1 decode is weight-bandwidth-bound (load ~8GB/step). tp_devices=8 shards the weights
across 8 chips (mesh data:1, model:8) -> ~1GB/step/chip, at the cost of a per-layer all-reduce.
Only tp=1 (single chip, SINGLE_DEVICE) and tp=8 (data:1) permit a true global B=1.

Usage: RAG_TP=<1|8> [SINGLE_DEVICE=1 for tp=1] PYTHONPATH=. python scripts/embed/tp_bs1_bench.py
"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import models.qwen3 as q
from inference import _generate_tokens

MODEL_ID = os.environ.get("RAG_MODEL_ID", "Qwen/Qwen3-4B")
TP   = int(os.environ.get("RAG_TP", "1"))
B    = int(os.environ.get("RAG_BS", "1"))
CTX  = int(os.environ.get("CTX", "2500"))
NEW  = int(os.environ.get("NEW", "256"))
REPS = int(os.environ.get("REPS", "6"))

def main():
    print(f"tp_bs1: model={MODEL_ID} tp={TP} B={B} ctx={CTX} new={NEW} single_device={os.environ.get('SINGLE_DEVICE')}")
    model = q.load(MODEL_ID, tp_devices=TP)
    mesh = next(v.sharding.mesh for v in model.weights.values()
                if hasattr(getattr(v, "sharding", None), "mesh"))
    print(f"mesh={mesh.shape}")
    vocab = model.cfg["vocab_size"]
    rng = np.random.default_rng(0)
    prompt = jax.device_put(jnp.array(rng.integers(0, vocab, size=(B, CTX), dtype=np.int32)),
                            NamedSharding(mesh, P("data", None)))
    pad = jax.device_put(jnp.ones((B, CTX), dtype=jnp.bool_), NamedSharding(mesh, P("data", None)))
    ext = jax.device_put(jnp.zeros((B, NEW), jnp.bool_), NamedSharding(mesh, P("data", None)))
    pad_ext = jnp.concatenate([pad, ext], axis=1)

    def gen():
        return _generate_tokens(model.forward, model.init_kv, model.weights, prompt,
                                NEW, pad_mask=pad, temperature=0.0, top_k=20, top_p=0.8)
    def prefill():
        return model.forward(prompt, model.weights, kv=model.init_kv(B, CTX+NEW), pos=0, pad_mask=pad_ext)
    jax.block_until_ready(gen())       # warmup decode compile
    jax.block_until_ready(prefill())   # warmup prefill compile (avoid recompile in timing loop)
    e2e, pf = [], []
    for _ in range(REPS):
        t = time.perf_counter(); jax.block_until_ready(gen()); e2e.append(time.perf_counter()-t)
        t = time.perf_counter(); jax.block_until_ready(prefill()); pf.append(time.perf_counter()-t)
    em = float(np.mean(e2e[1:] or e2e)); pm = float(np.mean(pf[1:] or pf))
    tps = NEW*B/(em-pm)
    print(f"TP_BS1_RESULT tp={TP} B={B} decode_tok_s={tps:.1f} e2e={em:.4f} prefill={pm:.4f}")

if __name__ == "__main__":
    main()
