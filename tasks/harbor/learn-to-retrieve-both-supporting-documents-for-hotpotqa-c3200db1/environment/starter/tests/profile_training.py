"""
Profile forward and backward pass for each layer to identify bottlenecks
Includes CPU-TPU transfer overhead and compilation time
"""
import argparse
import json
import time
import os
from pathlib import Path
from functools import partial
from contextlib import contextmanager

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
import optax
from tqdm import tqdm

from models import qwen3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@contextmanager
def timed_section(name):
    """Context manager for timing code sections"""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"  {name}: {(end - start)*1000:.2f} ms")


def profile_with_transfers(model, cfg, x_host, weights):
    """
    Profile with explicit transfer measurements
    """
    from jax.sharding import PartitionSpec as P
    
    print("\n" + "="*80)
    print("DETAILED PROFILING WITH TRANSFERS")
    print("="*80)
    
    # 1. Measure input transfer (CPU → TPU)
    with timed_section("Input transfer (CPU → TPU)"):
        x_device = jax.device_put(x_host, P('data', None))
        x_device.block_until_ready()
    
    # 2. Measure first forward (includes compilation)
    print("\nFirst forward pass (includes compilation):")
    with timed_section("  Forward with compilation"):
        logits = model.forward(x_device, weights)
        logits.block_until_ready()
    
    # 3. Measure subsequent forward (no compilation)
    print("\nSubsequent forward passes (compiled):")
    times = []
    for i in range(10):
        start = time.perf_counter()
        logits = model.forward(x_device, weights)
        logits.block_until_ready()
        end = time.perf_counter()
        times.append(end - start)
    
    avg_forward = np.mean(times)
    std_forward = np.std(times)
    print(f"  Average: {avg_forward*1000:.2f} ms ± {std_forward*1000:.2f} ms")
    
    # 4. Measure output transfer (TPU → CPU)
    with timed_section("Output transfer (TPU → CPU)"):
        logits_host = np.array(logits)
    
    # 5. Measure backward pass
    def loss_fn(weights):
        logits = model.forward(x_device, weights)
        return jnp.mean(logits ** 2)
    
    grad_fn = jax.value_and_grad(loss_fn)
    
    print("\nFirst backward pass (includes compilation):")
    with timed_section("  Backward with compilation"):
        loss, grads = grad_fn(weights)
        loss.block_until_ready()
        jax.tree.map(lambda g: g.block_until_ready(), grads)
    
    print("\nSubsequent backward passes (compiled):")
    times = []
    for i in range(10):
        start = time.perf_counter()
        loss, grads = grad_fn(weights)
        loss.block_until_ready()
        jax.tree.map(lambda g: g.block_until_ready(), grads)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_backward = np.mean(times)
    std_backward = np.std(times)
    print(f"  Average: {avg_backward*1000:.2f} ms ± {std_backward*1000:.2f} ms")
    print(f"  Backward only: {(avg_backward - avg_forward)*1000:.2f} ms")
    
    # 6. Measure gradient transfer
    with timed_section("Gradient transfer (TPU → CPU)"):
        grads_host = jax.tree.map(lambda g: np.array(g), grads)
    
    # 7. Full training step with optimizer
    optimizer = optax.adamw(learning_rate=2e-5)
    opt_state = optimizer.init(weights)
    
    @jax.jit
    def train_step(params, opt_state, x):
        def loss_fn(params):
            logits = model.forward(x, params)
            return jnp.mean(logits ** 2)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    
    print("\nFirst full training step (with optimizer, includes compilation):")
    with timed_section("  Train step with compilation"):
        new_weights, new_opt_state, loss = train_step(weights, opt_state, x_device)
        loss.block_until_ready()
        jax.tree.map(lambda w: w.block_until_ready(), new_weights)
    
    print("\nSubsequent training steps:")
    times = []
    for i in range(10):
        start = time.perf_counter()
        new_weights, new_opt_state, loss = train_step(weights, opt_state, x_device)
        loss.block_until_ready()
        jax.tree.map(lambda w: w.block_until_ready(), new_weights)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_train_step = np.mean(times)
    std_train_step = np.std(times)
    print(f"  Average: {avg_train_step*1000:.2f} ms ± {std_train_step*1000:.2f} ms")
    
    return {
        'forward': avg_forward,
        'backward_only': avg_backward - avg_forward,
        'train_step': avg_train_step,
    }


def profile_cross_device_communication():
    """Profile cross-device communication overhead"""
    if jax.device_count() <= 1:
        print("\nSkipping cross-device communication profiling (only 1 device)")
        return
    
    print("\n" + "="*80)
    print("CROSS-DEVICE COMMUNICATION PROFILING")
    print("="*80)
    
    # Skip detailed profiling if there's already a mesh context from model loading
    # Just do a simple collective operation test
    try:
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 1024, 1024))
        
        # Simple replicated sum that works with existing mesh
        def simple_sum(x):
            return jnp.sum(x)
        
        # Warmup
        for _ in range(5):
            result = jax.jit(simple_sum)(x)
            result.block_until_ready()
        
        # Profile
        times = []
        for _ in range(20):
            start = time.perf_counter()
            result = jax.jit(simple_sum)(x)
            result.block_until_ready()
            end = time.perf_counter()
            times.append(end - start)
        
        print(f"Simple collective operation (4MB): {np.mean(times)*1000:.2f} ms ± {np.std(times)*1000:.2f} ms")
        print("(Note: Detailed cross-device profiling skipped due to existing mesh context)")
    except Exception as e:
        print(f"Skipping cross-device profiling: {e}")


def profile_layer_forward(cfg, layer_idx, x, layer_weights, kv, pos):
    from models.qwen3 import forward_layer
    
    # Warmup
    for _ in range(3):
        x_out, kv_out = forward_layer(cfg, x, layer_weights, kv, pos)
        x_out.block_until_ready()
    
    # Profile
    start = time.time()
    num_iters = 10
    for _ in range(num_iters):
        x_out, kv_out = forward_layer(cfg, x, layer_weights, kv, pos)
        x_out.block_until_ready()
    end = time.time()
    
    avg_time = (end - start) / num_iters
    return avg_time, x_out, kv_out


def profile_layer_backward(cfg, layer_idx, x, layer_weights, kv, pos):
    """Profile a single layer forward + backward pass"""
    from models.qwen3 import forward_layer
    
    def loss_fn(weights, x, kv, pos):
        x_out, _ = forward_layer(cfg, x, weights, kv, pos)
        return jnp.mean(x_out ** 2)
    
    grad_fn = jax.value_and_grad(loss_fn)
    
    # Warmup
    for _ in range(3):
        loss, grads = grad_fn(layer_weights, x, kv, pos)
        loss.block_until_ready()
        jax.tree.map(lambda g: g.block_until_ready(), grads)
    
    # Profile
    start = time.time()
    num_iters = 10
    for _ in range(num_iters):
        loss, grads = grad_fn(layer_weights, x, kv, pos)
        loss.block_until_ready()
        jax.tree.map(lambda g: g.block_until_ready(), grads)
    end = time.time()
    
    avg_time = (end - start) / num_iters
    return avg_time


def profile_full_forward(model, cfg, x, weights):
    """Profile full forward pass"""
    # Warmup
    for _ in range(3):
        logits = model.forward(x, weights)
        logits.block_until_ready()
    
    # Profile
    start = time.time()
    num_iters = 10
    for _ in range(num_iters):
        logits = model.forward(x, weights)
        logits.block_until_ready()
    end = time.time()
    
    return (end - start) / num_iters


def profile_full_backward(model, cfg, x, weights):
    """Profile full forward + backward pass"""
    
    def loss_fn(weights):
        logits = model.forward(x, weights)
        return jnp.mean(logits ** 2)
    
    grad_fn = jax.value_and_grad(loss_fn)
    
    # Warmup
    for _ in range(3):
        loss, grads = grad_fn(weights)
        loss.block_until_ready()
        jax.tree.map(lambda g: g.block_until_ready(), grads)
    
    # Profile
    start = time.time()
    num_iters = 10
    for _ in range(num_iters):
        loss, grads = grad_fn(weights)
        loss.block_until_ready()
        jax.tree.map(lambda g: g.block_until_ready(), grads)
    end = time.time()
    
    return (end - start) / num_iters


def profile_embedding_forward(weights, x):
    """Profile embedding lookup"""
    from jax.sharding import PartitionSpec as P
    
    # Warmup
    for _ in range(3):
        out = weights['embed_tokens'].at[x, :].get(out_sharding=P('data', None, None))
        out.block_until_ready()
    
    start = time.time()
    num_iters = 10
    for _ in range(num_iters):
        out = weights['embed_tokens'].at[x, :].get(out_sharding=P('data', None, None))
        out.block_until_ready()
    end = time.time()
    
    return (end - start) / num_iters


def profile_logits_computation(cfg, x, weights):
    """Profile final logits computation"""
    from models.qwen3 import rms_norm
    
    def compute_logits(x):
        out_embed = weights['embed_tokens'] if cfg['tie_word_embeddings'] else weights['lm_head']
        x = rms_norm(x, weights['norm'], cfg['rms_norm_eps'])
        logits = jnp.einsum('btd,vd->btv', x, out_embed)
        return logits
    
    # Warmup
    for _ in range(3):
        logits = compute_logits(x)
        logits.block_until_ready()
    
    start = time.time()
    num_iters = 10
    for _ in range(num_iters):
        logits = compute_logits(x)
        logits.block_until_ready()
    end = time.time()
    
    return (end - start) / num_iters


def main():
    parser = argparse.ArgumentParser(description="Profile training performance")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--use_memory_layer", action="store_true")
    parser.add_argument("--reset_mlp", action="store_true")
    parser.add_argument("--memory_config", type=str, default="configs/memory_layer_config.json")
    parser.add_argument("--profile_per_layer", action="store_true", help="Profile each layer individually (slower but detailed)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("PROFILING TRAINING PERFORMANCE")
    print("="*80)
    print(f"Model: {args.model_id}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Memory layer: {args.use_memory_layer}")
    print(f"Devices: {jax.device_count()} ({jax.devices()[0].platform})")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    token = os.environ.get("HF_TOKEN")
    model = qwen3.load(model_id=args.model_id, token=token)
    
    actual_model_id = args.model_id
    hf_ckpt_dir = os.path.expanduser('~/weights/huggingface')
    model_ckpt_dir = Path(hf_ckpt_dir) / actual_model_id
    
    cfg = json.loads((model_ckpt_dir/'config.json').read_text())
    
    # Apply memory layer if requested
    if args.use_memory_layer:
        print(f"Applying memory layer from {args.memory_config}...")
        with open(args.memory_config, 'r') as f:
            mem_cfg = json.load(f)
        cfg.update(mem_cfg)
        
        if args.reset_mlp:
            model.weights = qwen3.reset_mlps(cfg, model.weights)
        
        model.weights = qwen3.add_memory_layer(cfg, model.weights)
        model.forward = partial(qwen3.forward, cfg)
    
    # Create dummy input
    print("\nGenerating dummy input...")
    key = jax.random.PRNGKey(42)
    x_host = np.random.randint(0, cfg['vocab_size'], (args.batch_size, args.seq_len), dtype=np.int32)
    
    print(f"Input shape: {x_host.shape}")
    print("\nStarting profiling (this will take a few minutes)...\n")
    
    # Profile with explicit transfers
    transfer_stats = profile_with_transfers(model, cfg, x_host, model.weights)
    
    # Profile cross-device communication
    profile_cross_device_communication()
    
    # Convert to JAX array for other profiling
    x = jnp.array(x_host)
    
    # Profile embedding
    print("Profiling embedding layer...")
    embed_time = profile_embedding_forward(model.weights, x)
    print(f"  Embedding: {embed_time*1000:.2f} ms")
    
    # Profile full forward pass
    print("\nProfiling full forward pass...")
    forward_time = profile_full_forward(model, cfg, x, model.weights)
    print(f"  Full forward: {forward_time*1000:.2f} ms")
    
    # Profile full backward pass
    print("\nProfiling full forward + backward pass...")
    backward_time = profile_full_backward(model, cfg, x, model.weights)
    print(f"  Full forward + backward: {backward_time*1000:.2f} ms")
    print(f"  Backward only (estimated): {(backward_time - forward_time)*1000:.2f} ms")
    
    # Profile logits computation
    print("\nProfiling logits computation...")
    # Create dummy hidden states
    dummy_hidden = jax.random.normal(key, (args.batch_size, args.seq_len, cfg['hidden_size']))
    logits_time = profile_logits_computation(cfg, dummy_hidden, model.weights)
    print(f"  Logits computation: {logits_time*1000:.2f} ms")
    
    # Per-layer profiling (optional, slower)
    if args.profile_per_layer:
        print("\n" + "="*80)
        print("PER-LAYER PROFILING")
        print("="*80)
        
        # Prepare layer inputs
        x_layer = model.weights['embed_tokens'].at[x, :].get(out_sharding=P('data', None, None)).astype(jnp.bfloat16)
        
        layer_forward_times = []
        layer_backward_times = []
        
        for i in tqdm(range(cfg['num_hidden_layers']), desc="Profiling layers"):
            # Get layer weights
            layer_weights = {k.replace(prefix, ''):v for k,v in model.weights.items() if (prefix:=f'layers.{i}.') in k}
            
            if "mem_layers" in cfg and i in cfg["mem_layers"]:
                layer_weights.update({"mem_k": model.weights["mem_k"], "mem_v": model.weights["mem_v"]})
                layer_type = "MEMORY"
            else:
                layer_type = "MLP"
            
            # Profile forward
            fwd_time, x_layer, _ = profile_layer_forward(cfg, i, x_layer, layer_weights, None, 0)
            layer_forward_times.append((i, layer_type, fwd_time))
            
            # Profile backward
            bwd_time = profile_layer_backward(cfg, i, x_layer, layer_weights, None, 0)
            layer_backward_times.append((i, layer_type, bwd_time))
        
        # Print per-layer results
        print("\n" + "="*80)
        print("FORWARD PASS - PER LAYER")
        print("="*80)
        print(f"{'Layer':<8} {'Type':<10} {'Time (ms)':<12} {'% of Total':<12}")
        print("-"*80)
        
        total_layer_time = sum(t for _, _, t in layer_forward_times)
        for layer_idx, layer_type, layer_time in layer_forward_times:
            pct = (layer_time / total_layer_time * 100) if total_layer_time > 0 else 0
            print(f"{layer_idx:<8} {layer_type:<10} {layer_time*1000:<12.2f} {pct:<12.1f}")
        
        print("\n" + "="*80)
        print("FORWARD + BACKWARD PASS - PER LAYER")
        print("="*80)
        print(f"{'Layer':<8} {'Type':<10} {'Time (ms)':<12} {'% of Total':<12}")
        print("-"*80)
        
        total_bwd_time = sum(t for _, _, t in layer_backward_times)
        for layer_idx, layer_type, layer_time in layer_backward_times:
            pct = (layer_time / total_bwd_time * 100) if total_bwd_time > 0 else 0
            print(f"{layer_idx:<8} {layer_type:<10} {layer_time*1000:<12.2f} {pct:<12.1f}")
        
        # Show slowest layers
        print("\n" + "="*80)
        print("TOP 5 SLOWEST LAYERS (Forward + Backward)")
        print("="*80)
        sorted_layers = sorted(layer_backward_times, key=lambda x: x[2], reverse=True)
        for i, (layer_idx, layer_type, layer_time) in enumerate(sorted_layers[:5], 1):
            print(f"{i}. Layer {layer_idx} ({layer_type}): {layer_time*1000:.2f} ms")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Embedding:              {embed_time*1000:>10.2f} ms")
    print(f"Full forward pass:      {transfer_stats['forward']*1000:>10.2f} ms")
    print(f"Full backward pass:     {transfer_stats['backward_only']*1000:>10.2f} ms")
    print(f"Logits computation:     {logits_time*1000:>10.2f} ms")
    print(f"Total train step:       {transfer_stats['train_step']*1000:>10.2f} ms")
    print(f"\nThroughput:             {args.batch_size * args.seq_len / transfer_stats['train_step']:.0f} tokens/sec")
    print(f"Steps per hour:         {3600 / transfer_stats['train_step']:.0f}")
    print("="*80)


if __name__ == "__main__":
    main()
