
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import time
import numpy as np
from functools import partial
from jax.sharding import PartitionSpec as P, Mesh, AxisType
from models.qwen3_mem_embed import embed_forward, main_forward
from models.retrieval_ops import set_global_mesh

def benchmark_grad_accum():
    # Setup Mesh for 8-way parallel sharding
    devices = jax.devices()
    mesh = Mesh(np.array(devices).reshape(1, 8), ('data', 'model'), axis_types=(AxisType.Explicit, AxisType.Explicit))
    jax.set_mesh(mesh)
    set_global_mesh(mesh)
    
    print(f"Using device mesh: {mesh}")
    
    # Scale from user's 512k setup
    B_total = 512
    T = 512
    N = 8
    K = 128
    H = 2048
    Dv = 2048
    M = 524288
    dtype = jnp.bfloat16

    # Mock weights and inputs
    key = jax.random.PRNGKey(0)
    # Memory bank weights (M is sharded on 'model' axis)
    embed_weights = {
        "mem_k_proj": jax.random.normal(key, (H, H), dtype=dtype),
        "mem_v_proj": jax.random.normal(key, (H, Dv), dtype=dtype),
        # Hidden states from embed model for M tokens
        "hidden": jax.random.normal(key, (M // 512, 512, H), dtype=dtype) 
    }
    
    cfg = {
        "main_model": {
            "num_hidden_layers": 2, # Reduced for faster benchmarking
            "num_attention_heads": N,
            "num_key_value_heads": N,
            "head_dim": H // N,
            "rope_theta": 10000.0,
            "mem_layers": [1],
            "rms_norm_eps": 1e-6,
            "two_pass_topk": True,
            "mem_top_k": K,
            "mem_lookup_chunk_size": 16384,
            "mem_k_prenormed": False,
            "hidden_size": H,
            "tie_word_embeddings": True,
            "mask_type": "causal",
            "mem_placement": "after_attention",
        },
        "embed_model": {
            "num_hidden_layers": 2, # Reduced
            "rms_norm_eps": 1e-6,
            "embed_conv": False,
        }
    }

    main_weights = {
        "embed_tokens": jax.random.normal(key, (1000, H), dtype=dtype), # Dummy
        "norm": jnp.ones((H,), dtype=dtype),
        "input_layernorm": jnp.ones((H,), dtype=dtype),
    }
    # Per-layer weights
    for i in range(cfg["main_model"]["num_hidden_layers"]):
        main_weights[f"layers.{i}.input_layernorm"] = jnp.ones((H,), dtype=dtype)
        main_weights[f"layers.{i}.q_proj"] = jax.random.normal(key, (N, H // N, H), dtype=dtype)
        main_weights[f"layers.{i}.k_proj"] = jax.random.normal(key, (N, H // N, H), dtype=dtype)
        main_weights[f"layers.{i}.v_proj"] = jax.random.normal(key, (N, H // N, H), dtype=dtype)
        main_weights[f"layers.{i}.q_norm"] = jnp.ones((N, H // N), dtype=dtype)
        main_weights[f"layers.{i}.k_norm"] = jnp.ones((N, H // N), dtype=dtype)
        main_weights[f"layers.{i}.o_proj"] = jax.random.normal(key, (H, N, H // N), dtype=dtype)
        main_weights[f"layers.{i}.gate_proj"] = jax.random.normal(key, (H * 4, H), dtype=dtype)
        main_weights[f"layers.{i}.up_proj"] = jax.random.normal(key, (H * 4, H), dtype=dtype)
        main_weights[f"layers.{i}.down_proj"] = jax.random.normal(key, (H, H * 4), dtype=dtype)
        main_weights[f"layers.{i}.post_attention_layernorm"] = jnp.ones((H,), dtype=dtype)
        
        if i in cfg["main_model"]["mem_layers"]:
            main_weights[f"layers.{i}.mem_layernorm"] = jnp.ones((H,), dtype=dtype)
            main_weights[f"layers.{i}.mem_q_proj"] = jax.random.normal(key, (N, H // N, H), dtype=dtype)
            main_weights[f"layers.{i}.mem_q_norm"] = jnp.ones((N, H // N), dtype=dtype)
            main_weights[f"layers.{i}.mem_o_proj"] = jax.random.normal(key, (H, N, H // N), dtype=dtype)
            main_weights[f"layers.{i}.mem_k_norm"] = jnp.ones((H // N,), dtype=dtype)

    def full_step(n_accum):
        mini_bs = B_total // n_accum
        
        # 1. Embed forward (mocked to just the projections)
        mem_k = jax.random.normal(key, (M * N, H // N), dtype=dtype)
        mem_v = jax.random.normal(key, (M * N, H // N), dtype=dtype)
        
        mem_k = jax.sharding.reshard(mem_k, P('model', None))
        mem_v = jax.sharding.reshard(mem_v, P('model', None))
        
        mem_mask = jnp.ones((M * N,), dtype=bool)
        mem_mask = jax.sharding.reshard(mem_mask, P('model'))
        
        # 2. Accumulate steps
        weights = {**main_weights, "mem_k": mem_k, "mem_v": mem_v, "mem_mask": mem_mask}
        
        def mini_batch_step(carry, i):
            # Mock mini-batch input
            x_m = jnp.zeros((mini_bs, T), dtype=jnp.int32)
            # We use jax.grad here to measure backward pass time too
            def loss_fn(w):
                logits, _, _ = main_forward(cfg["main_model"], x_m, w, collect_aux=False)
                return jnp.mean(logits**2)

            # Filter for inexact weights for grad
            inexact_weights = {k: v for k, v in weights.items() if jnp.issubdtype(v.dtype, jnp.inexact)}
            other_weights = {k: v for k, v in weights.items() if not jnp.issubdtype(v.dtype, jnp.inexact)}

            def loss_wrapper(iw):
                return loss_fn({**iw, **other_weights})

            grad = jax.grad(loss_wrapper)(inexact_weights)
            return carry, grad
        # We use scan to mimic GradAccumTrainer logic
        _, grads = jax.lax.scan(mini_batch_step, None, jnp.arange(n_accum))
        return grads

    for n_accum in [8, 16, 32]:
        mini_bs = B_total // n_accum
        print(f"\nBenchmarking B_total={B_total}, n_accum={n_accum} (B_mini={mini_bs})...")
        
        step_jit = jax.jit(partial(full_step, n_accum))
        
        # Warmup
        try:
            start_compile = time.perf_counter()
            _ = step_jit()
            jax.block_until_ready(_)
            print(f"  Compiled in {time.perf_counter() - start_compile:.2f}s")
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        times = []
        for _ in range(3):
            start = time.perf_counter()
            _ = step_jit()
            jax.block_until_ready(_)
            times.append(time.perf_counter() - start)
        
        avg_s = np.mean(times)
        tps = (B_total * T) / avg_s
        print(f"  Avg Step Time: {avg_s:.2f} s")
        print(f"  Throughput: {tps:.0f} tokens/sec")

if __name__ == "__main__":
    benchmark_grad_accum()
