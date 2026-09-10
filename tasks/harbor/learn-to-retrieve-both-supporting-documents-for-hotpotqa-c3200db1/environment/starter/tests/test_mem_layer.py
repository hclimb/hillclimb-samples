import jax
import jax.numpy as jnp
from models import qwen3
import json
from pathlib import Path
import os
import numpy as np
from functools import partial

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    model_id = "Qwen/Qwen3-0.6B-Base"
    print(f"Loading model: {model_id}")
    
    # Load model
    try:
        model = qwen3.load(model_id=model_id)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load config manually to update it for the memory layer
    hf_ckpt_dir = os.path.expanduser('~/weights/huggingface')
    model_ckpt_dir = Path(hf_ckpt_dir) / model_id
    config_path = model_ckpt_dir / 'config.json'
    
    if not config_path.exists():
         print(f"Config file not found at {config_path}.")
         return
    
    cfg = json.loads(config_path.read_text())
    
    # Update cfg with memory layer parameters
    cfg["mem_layers"] = [2, 4] 
    cfg["mem_size"] = 1024
    cfg["num_mem_heads"] = 4
    cfg["mem_k_dim"] = 64
    cfg["mem_v_dim"] = 64
    cfg["top_k"] = 32
    
    print("Adding memory layer to weights...")
    model.weights = qwen3.add_memory_layer(cfg, model.weights)
    
    # Construct a new forward function with the updated config
    new_forward = partial(qwen3.forward, cfg)
    
    print("\n--- Structural Verification ---")
    
    # Check if MLP projection was removed for a memory layer
    layer_idx = cfg["mem_layers"][0]
    gate_proj_key = f'layers.{layer_idx}.gate_proj'
    if gate_proj_key not in model.weights:
        print(f"PASS: {gate_proj_key} removed.")
    else:
        print(f"FAIL: {gate_proj_key} still present.")

    # Check if memory projections were added
    mem_q_proj_key = f'layers.{layer_idx}.mem_q_proj'
    if mem_q_proj_key in model.weights:
        print(f"PASS: {mem_q_proj_key} added.")
    else:
        print(f"FAIL: {mem_q_proj_key} missing.")

    print("\n--- Functional Verification (Forward Pass) ---")
    
    # Create dummy input
    B, T = 4, 8
    dummy_input = jnp.zeros((B, T), dtype=jnp.int32)
    
    print("Running forward pass...")
    try:
        logits = new_forward(dummy_input, model.weights, kv=None)
        print("PASS: Forward pass completed successfully.")
        print(f"Logits shape: {logits.shape}")
    except Exception as e:
        print(f"FAIL: Forward pass crashed.")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()