
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from models.memory import mem_lookup_two_pass

def cost_analysis():
    B, T, N, K, H, Dv = 16, 512, 4, 128, 1024, 1024
    M = 524288
    dtype = jnp.bfloat16

    q_shape = jax.ShapeDtypeStruct((B, T, N, H), dtype)
    w_shape = {
        "mem_k": jax.ShapeDtypeStruct((M, H), dtype),
        "mem_k_norm": jax.ShapeDtypeStruct((H,), dtype),
        "mem_v": jax.ShapeDtypeStruct((M, Dv), dtype),
    }
    
    cfg_base = {
        "rms_norm_eps": 1e-6,
        "mem_top_k": K,
        "mem_lookup_chunk_size": 16384,
        "mem_k_prenormed": False,
        "two_pass_topk": True,
    }

    for name, sparse_grads in [("Dense", False), ("Sparse", True)]:
        cfg = {**cfg_base, "sparse_grads": sparse_grads}
        
        def loss_fn(q, w):
            s, v, _ = mem_lookup_two_pass(q, w, cfg)
            return jnp.mean(jnp.einsum("bntk,bntkd->bntd", s, v) ** 2)

        grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))
        compiled = grad_fn.lower(q_shape, w_shape).compile()
        
        # In newer JAX versions, cost_analysis() returns a dict or list
        cost = compiled.cost_analysis()
        print(f"DEBUG {name} keys: {cost[0].keys() if isinstance(cost, list) else cost.keys()}")
        if isinstance(cost, list):
            peak_bytes = max(c.get("bytes accessed", 0) for c in cost)
        else:
            peak_bytes = cost.get("bytes accessed", 0)
            
        print(f"{name} Peak Bytes Accessed: {peak_bytes / 1e9:.2f} GB")

if __name__ == "__main__":
    cost_analysis()
