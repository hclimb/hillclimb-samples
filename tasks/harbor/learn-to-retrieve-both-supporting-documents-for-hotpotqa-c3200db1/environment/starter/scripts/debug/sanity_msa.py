"""Sanity 1: load MSA-4B, run the plain backbone (no memory), generate text.

Validates weight conversion / dims / RoPE / tokenizer before exercising the MSA
routing path. Run on a TPU VM:
    cd memory-layers && set -a && source .env && set +a && \
    .venv/bin/python scripts/embed/sanity_msa.py
"""
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from models import get_model
from inference import _generate_tokens

cfg = OmegaConf.create({
    "name": "qwen3_msa",
    "main_model": {"model_id": "EverMind-AI/MSA-4B", "load_weights": True},
})

print("Loading MSA-4B ...", flush=True)
model = get_model(cfg, tp_devices=1)
print("num_layers:", model.cfg["num_hidden_layers"],
      "router_layers:", model.cfg["msa"]["router_layers"], flush=True)
print("has router_q_proj L18:", "layers.18.router_q_proj" in model.weights, flush=True)
print("has temperature:", "temperature" in model.weights, flush=True)
print("q_proj L0 shape:", model.weights["layers.0.q_proj"].shape, flush=True)
print("router_q_proj L18 shape:", model.weights["layers.18.router_q_proj"].shape, flush=True)
print("router_k_proj L18 shape:", model.weights["layers.18.router_k_proj"].shape, flush=True)

tok = model.tokenizer
prompt = "<|im_start|>user\nWhat is the capital of France? Answer in one word.<|im_end|>\n<|im_start|>assistant\n"
ids = tok(prompt, return_tensors="np", add_special_tokens=False)["input_ids"]
ids = np.repeat(ids, jax.device_count(), axis=0)   # batch divisible by data mesh
B, T = ids.shape
prompt_jax = jax.device_put(jnp.array(ids))
pad = jax.device_put(jnp.ones((B, T), dtype=bool))

print("Generating (plain backbone)...", flush=True)
gen = _generate_tokens(
    model.forward, model.init_kv, model.weights, prompt_jax, 32,
    pad_mask=pad, temperature=0.0, top_k=20, top_p=0.8,
)
gen = np.array(gen)
out = tok.decode(gen[0], skip_special_tokens=True)
print("=== GENERATION ===", flush=True)
print(repr(out), flush=True)
print("SANITY1_DONE", flush=True)
