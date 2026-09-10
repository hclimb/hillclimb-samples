'''
Adapted from https://github.com/martin-marek/jax-llm
'''
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

from .qwen3 import Model, apply_rope, rms_norm, self_attention, mlp, create_mask
from .qwen3 import load as load_qwen3
from .memory_utils import add_memory_layer
from .memory import memory_layer
from .output import ModelOutput


def forward_layer(cfg, x, w, attn_mask, kv=None, pos=0, collect_aux=False):
    x, kv = self_attention(cfg, x, w, attn_mask, kv, pos)
    placement = cfg.get("mem_placement")
    aux_data = None
    
    if "mem_q_proj" in w:
        if placement in ("replace_mlp"):
            x, aux_data = memory_layer(cfg, x, w, collect_aux=collect_aux)
        if placement == "after_attention":
            x, aux_data = memory_layer(cfg, x, w, collect_aux=collect_aux)
            x = mlp(cfg, x, w)
        if placement == "after_mlp":
            x = mlp(cfg, x, w)
            x, aux_data = memory_layer(cfg, x, w, collect_aux=collect_aux)
    else:
        x = mlp(cfg, x, w)
    
    return x, kv, aux_data

def forward(cfg, x, weights, pad_mask=None, kv=None, pos=0, collect_aux=False):
    """
    Forward pass through the model.
    
    Args:
        cfg: Model config
        x: Input token ids [B, T]
        weights: Model weights dict
        pad_mask: Padding mask [B, T]
        kv: KV cache (optional)
        pos: Position offset for KV cache
        collect_aux: Whether to collect auxiliary data for aux losses
    
    Returns:
        ModelOutput with logits, optional kv, and optional aux data
    """
    # Prepare attention mask
    attn_mask = create_mask(cfg, x, kv, pad_mask, pos)

    # embedding
    x = jax.device_put(x, P('data', None))
    x = weights['embed_tokens'].at[x, :].get(out_sharding=P('data', None, None)).astype(jnp.bfloat16)
    
    # iterate over hidden layers
    return_kv = kv is not None
    if kv is None: kv = defaultdict(lambda: None)
    
    # Collect auxiliary data from all memory layers
    all_aux_data = {"mem_scores": []} if collect_aux else None
    
    for i in range(cfg['num_hidden_layers']):
        layer_weights = {k.replace(prefix, ''):v for k,v in weights.items() if (prefix:=f'layers.{i}.') in k}
        if "mem_layers" in cfg and i in cfg["mem_layers"]:
            layer_weights.update({"mem_k": weights["mem_k"], "mem_v": weights["mem_v"]})
        
        # partial captures collect_aux as a static Python value (not traced)
        x, kv[i], layer_aux = jax.remat(
            partial(forward_layer, cfg, collect_aux=collect_aux)
        )(x, layer_weights, attn_mask, kv[i], pos)
        
        # Aggregate auxiliary data from memory layers
        if collect_aux and layer_aux is not None:
            if "mem_scores" in layer_aux and layer_aux["mem_scores"] is not None:
                all_aux_data["mem_scores"].append(layer_aux["mem_scores"])

    # logits
    out_embed = weights['embed_tokens'] if cfg['tie_word_embeddings'] else weights['lm_head']
    x = rms_norm(x, weights['norm'], cfg['rms_norm_eps'])
    logits = jnp.einsum('btd,vd->btv', x, out_embed, preferred_element_type=x.dtype, out_sharding=P('data', None, 'model'))

    return ModelOutput(
        logits=logits,
        kv=kv if return_kv else None,
        aux=all_aux_data
    )

def load(cfg, tp_devices=1, hf_ckpt_dir='~/weights/huggingface'):
    model_id = cfg.main_model.model_id
    load_weights = cfg.main_model.load_weights
    mask_type = cfg.main_model.get("mask_type", "causal")
    lora_cfg = cfg.main_model.get("lora")
    
    # Load base model from qwen3.py
    model = load_qwen3(model_id, tp_devices, load_weights, hf_ckpt_dir, mask_type, lora_cfg)

    # Add memory layers
    model.weights, model.cfg = add_memory_layer(cfg, model.cfg, model.weights)
    
    model.forward = partial(forward, model.cfg)
    
    return model

def init(cfg, tp_devices):
    model = load(cfg, tp_devices=tp_devices)
    return model