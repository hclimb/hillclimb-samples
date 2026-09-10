import os
import json
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from safetensors import safe_open
from dataclasses import dataclass
from collections import defaultdict
from typing import Callable, Dict
from functools import partial, reduce
from huggingface_hub import snapshot_download
from jax.sharding import PartitionSpec as P, AxisType
from transformers import PreTrainedTokenizerFast, AddedToken

def _sample(logits, key, temperature=0.7, top_k=0, top_p=1.0):
    # Gather logits to replicated so slicing/sampling works regardless of sharding
    logits = jax.sharding.reshard(logits, P('data', None))

    if temperature == 0.0:
        return jnp.argmax(logits, axis=-1, keepdims=True)

    # Top-K
    if top_k > 0:
        top_k_val, _ = jax.lax.top_k(logits, top_k)
        logits = jnp.where(logits < top_k_val[..., -1:], -jnp.inf, logits)

    # Top-P
    if top_p < 1.0:
        sorted_logits, indices = jax.lax.top_k(logits, k=logits.shape[-1])
        sorted_probs = jax.nn.softmax(sorted_logits / temperature, axis=-1)
        cum_probs = jnp.cumsum(sorted_probs, axis=-1)
        sorted_mask = cum_probs > top_p
        sorted_mask = jnp.concatenate([jnp.zeros_like(sorted_mask[..., :1]), sorted_mask[..., :-1]], axis=-1)
        sorted_logits = jnp.where(sorted_mask, -jnp.inf, sorted_logits)
        
        # Sample from sorted
        idx = jax.random.categorical(key, sorted_logits / temperature, axis=-1)
        # Map back
        return indices.at[jnp.arange(indices.shape[0])[:, None], idx[..., None]].get(mode='fill', fill_value=0, out_sharding=P('data', None))

    return jax.random.categorical(key, logits / temperature, axis=-1)[..., None]


@partial(jax.jit, static_argnames=("forward", "init_kv", "max_new_tokens", "temperature", "top_k", "top_p"))
def _generate_tokens(forward, init_kv, params, prompt_tokens, max_new_tokens=32, pad_mask=None, key=None, temperature=0.7, top_k=20, top_p=0.8, **kwargs):
    if isinstance(prompt_tokens, dict):
        B, T = prompt_tokens["batch"].shape
        dtype = prompt_tokens["batch"].dtype
    else:
        B, T = prompt_tokens.shape
        dtype = prompt_tokens.dtype
    
    max_seq_len = T + max_new_tokens
    
    # Initialize KV cache
    kv = init_kv(B, max_seq_len)
    
    # Shard input. Under jit (this fn is jitted) this is a sharding constraint on the traced
    # argument, which is multi-host safe — callers pass globally-placed arrays as arguments.
    prompt_tokens = jax.device_put(prompt_tokens, P('data', None))
    
    # Extend pad_mask to match KV cache size for prefill
    # Positions [T:max_seq_len] are unfilled, so mask them as invalid (0)
    if pad_mask is not None:
        prefill_pad_mask = jnp.concatenate([
            pad_mask,
            jnp.zeros((B, max_new_tokens), dtype=pad_mask.dtype)
        ], axis=1)
    else:
        prefill_pad_mask = None
    
    # Prefill
    output = forward(prompt_tokens, params, kv=kv, pos=0, pad_mask=prefill_pad_mask, **kwargs)
    logits, kv = output.logits, output.kv
    next_logits = logits[:, -1, :] 
    
    if key is None:
        key = jax.random.PRNGKey(42)

    if temperature > 0.0:
        key, subkey = jax.random.split(key)
        next_token = _sample(next_logits, subkey, temperature, top_k, top_p)
    else:
        next_token = _sample(next_logits, None, 0.0, top_k, top_p)
        
    next_token = next_token.astype(dtype)
    
    # Store generated tokens
    generated_tokens = jnp.zeros((B, max_new_tokens), dtype=dtype)
    generated_tokens = generated_tokens.at[:, 0].set(next_token[:, 0], out_sharding=P('data', None))
    
    # For generation loop, we'll update the mask dynamically
    # Start with the prefill mask and update positions as we generate
    if pad_mask is not None:
        # Initialize with prompt mask + zeros for future positions
        extended_pad_mask = prefill_pad_mask
    else:
        extended_pad_mask = None
    
    # Generation loop
    def step_fn(i, state):
        kv, current_token, key, gen_seq, gen_mask = state
        pos = T + i 
        
        # Update mask to mark this position as valid (we're generating a real token)
        if gen_mask is not None:
            gen_mask = gen_mask.at[:, pos].set(True)
        
        output = forward(current_token, params, kv=kv, pos=pos, pad_mask=gen_mask, **kwargs)
        logits, kv = output.logits, output.kv
        next_logits = logits[:, 0, :]
        
        if temperature > 0.0:
            key, subkey = jax.random.split(key)
            token = _sample(next_logits, subkey, temperature, top_k, top_p)
        else:
            token = _sample(next_logits, None, 0.0, top_k, top_p)
            
        token = token.astype(current_token.dtype)
        gen_seq = gen_seq.at[:, i+1].set(token[:, 0], out_sharding=P('data', None))
        
        return kv, token, key, gen_seq, gen_mask

    init_state = (kv, next_token, key, generated_tokens, extended_pad_mask)
    final_state = jax.lax.fori_loop(0, max_new_tokens - 1, step_fn, init_state)
    
    return final_state[3]


def generate(forward, init_kv, tokenizer, params, prompts, chat_template=True, max_new_tokens=32, key=None, temperature=0.7, top_k=20, top_p=0.8, **kwargs):
    tokenized_prompts = []
    for p in prompts:
        if chat_template:
            messages = [{"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        else:
            text = p
            
        tokenized_prompts.append(tokenizer(text, return_tensors="np")["input_ids"][0])

    max_len = max(len(t) for t in tokenized_prompts)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    padded_input = []
    for t in tokenized_prompts:
        pad_len = max_len - len(t)
        padded_t = np.concatenate([np.full(pad_len, pad_id, dtype=t.dtype), t])
        padded_input.append(padded_t)
        
    batch_input_ids = jnp.array(np.stack(padded_input))
    
    # Create pad mask: 1 for real tokens, 0 for padding (left-padded)
    # Use boolean dtype for compatibility with bitwise operations in create_mask
    pad_mask = jnp.array(np.stack([
        np.concatenate([np.zeros(max_len - len(t), dtype=np.bool_), np.ones(len(t), dtype=np.bool_)])
        for t in tokenized_prompts
    ]))
    
    generated_tokens = _generate_tokens(forward, init_kv, params, batch_input_ids, max_new_tokens, pad_mask=pad_mask, key=key, temperature=temperature, top_k=top_k, top_p=top_p, **kwargs)
    
    # Decode to text
    return [tokenizer.decode(t, skip_special_tokens=True) for t in generated_tokens]
