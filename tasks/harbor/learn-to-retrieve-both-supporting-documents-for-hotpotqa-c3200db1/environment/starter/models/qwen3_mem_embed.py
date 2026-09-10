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

from .utils import merge_weights, split_weights, merge_configs
from .qwen3 import Model, apply_rope, rms_norm, self_attention, mlp, create_mask, forward as qwen3_forward
from .qwen3 import load as load_qwen3
from .memory_utils import add_memory_layer, add_kv_head, add_embed_conv
from .memory import memory_layer
from .conv_utils import apply_conv1d, pool_pad_mask
from .output import ModelOutput

def forward_layer(cfg, x, w, attn_mask, kv=None, pos=0, collect_aux=False, pos_slot_indices=None):
    x, kv = self_attention(cfg, x, w, attn_mask, kv, pos)
    placement = cfg.get("mem_placement")
    aux_data = None

    if "mem_q_proj" in w:
        if placement in ("replace_mlp"):
            x, aux_data = memory_layer(cfg, x, w, collect_aux=collect_aux, pos_slot_indices=pos_slot_indices)
        if placement == "after_attention":
            x, aux_data = memory_layer(cfg, x, w, collect_aux=collect_aux, pos_slot_indices=pos_slot_indices)
            x = mlp(cfg, x, w)
        if placement == "after_mlp":
            x = mlp(cfg, x, w)
            x, aux_data = memory_layer(cfg, x, w, collect_aux=collect_aux, pos_slot_indices=pos_slot_indices)
    else:
        x = mlp(cfg, x, w)

    return x, kv, aux_data

def embed_forward(cfg, x, weights, pad_mask):
    last_hidden_states = qwen3_forward(cfg, x, weights, pad_mask, return_hidden=True)

    # Optional 1D conv to compress sequence dimension (separate convs for k and v)
    use_conv = cfg.get('embed_conv', False)
    if use_conv:
        kernel_size = cfg['embed_conv_kernel_size']
        stride = cfg['embed_conv_stride']
        hidden_k = apply_conv1d(last_hidden_states, weights['embed_proj_conv_k_weight'], weights['embed_proj_conv_k_bias'], stride)
        hidden_v = apply_conv1d(last_hidden_states, weights['embed_proj_conv_v_weight'], weights['embed_proj_conv_v_bias'], stride)
        pad_mask = pool_pad_mask(pad_mask, kernel_size, stride)
    else:
        hidden_k = last_hidden_states
        hidden_v = last_hidden_states

    if weights['mem_k_proj'].ndim == 3:
        # GQA per-kv-head bank: mem_{k,v}_proj [d_embed, Nkv, dim] -> bank [M, Nkv, dim].
        mem_k = jnp.einsum('bsh,hnd->bsnd', hidden_k, weights['mem_k_proj']) # (num_docs, seq_len, Nkv, k_dim)
        mem_v = jnp.einsum('bsh,hnd->bsnd', hidden_v, weights['mem_v_proj']) # (num_docs, seq_len, Nkv, v_dim)
        effective_doc_len = mem_k.shape[1]
        nkv = mem_k.shape[2]
        mem_k = mem_k.reshape(-1, nkv, mem_k.shape[-1]) # (total_mem_vectors, Nkv, k_dim)
        mem_v = mem_v.reshape(-1, nkv, mem_v.shape[-1]) # (total_mem_vectors, Nkv, v_dim)
    else:
        mem_k = jnp.einsum('bsh,hd->bsd', hidden_k, weights['mem_k_proj']) # (num_docs, seq_len, k_dim)
        mem_v = jnp.einsum('bsh,hd->bsd', hidden_v, weights['mem_v_proj']) # (num_docs, seq_len, v_dim)
        # Doc-code: mean-pool the chunk's hidden states (pad-masked) into one identity
        # vector, project to code_dim, and concatenate onto EVERY token key of the chunk.
        # Bank key becomes [token_key ; doc_code]; q·k = token-similarity + doc-affinity,
        # giving value-position slots a subject binding their local context lacks.
        if 'doc_code_proj' in weights:   # weight presence is authoritative (embed cfg may not carry memory keys)
            m = pad_mask.astype(hidden_k.dtype)                                # (num_docs, seq_len)
            pooled = (hidden_k * m[:, :, None]).sum(axis=1) / \
                     (m.sum(axis=1, keepdims=True) + 1e-6)                     # (num_docs, d_embed)
            code = jnp.einsum('bh,hd->bd', pooled, weights['doc_code_proj'])   # (num_docs, code_dim)
            code = jnp.broadcast_to(code[:, None, :], mem_k.shape[:2] + code.shape[-1:])
            mem_k = jnp.concatenate([mem_k, code], axis=-1)                    # (num_docs, seq_len, k_dim + code_dim)
        effective_doc_len = mem_k.shape[1]  # post-conv doc length (or original if no conv)
        mem_k = mem_k.reshape(-1, mem_k.shape[-1]) # (total_mem_vectors, k_dim [+ code_dim])
        mem_v = mem_v.reshape(-1, mem_v.shape[-1]) # (total_mem_vectors, v_dim)

    pad_mask = pad_mask.reshape(-1) # (total_mem_vectors,)
    return mem_k, mem_v, pad_mask, effective_doc_len

def value_forward(cfg, x, weights, pad_mask):
    """Stage 2: memory VALUES from a SEPARATE (base-LM) value model. Base-LM per-token
    hidden states carry surface/next-token info — the right starting point for value
    reconstruction — whereas the contrastive embedding trunk homogenizes them. No conv on
    the value path (keeps it simple; 1 doc token = 1 value slot, aligned with the key path
    at conv kernel/stride 1). Returns flat mem_v [total_mem_vectors, v_dim]."""
    last_hidden_states = qwen3_forward(cfg, x, weights, pad_mask, return_hidden=True)
    mem_v = jnp.einsum('bsh,hd->bsd', last_hidden_states, weights['mem_v_proj'])
    mem_v = mem_v.reshape(-1, mem_v.shape[-1])
    return mem_v

def _kv_bank_telemetry(mem_k, mem_v, mem_mask):
    """Stage-2 fidelity telemetry on the built banks (weight-0). cos(K,V) for the same doc
    token should DROP as the key/value models specialize (the direct decouple success metric);
    value anisotropy (mean pairwise cosine of values) quantifies whether values stay spread
    (base-LM) vs collapse toward a pooled summary (embedding trunk). Reductions over the flat
    bank axis all-reduce cleanly; the per-vector norm is over the (unsharded) feature axis."""
    f32 = jnp.float32
    m = mem_mask.astype(f32)                               # [total] valid=1
    n = m.sum()
    out = {}
    if mem_k.shape[-1] == mem_v.shape[-1]:
        kn = mem_k.astype(f32); kn = kn / (jnp.linalg.norm(kn, axis=-1, keepdims=True) + 1e-9)
        vn = mem_v.astype(f32); vn = vn / (jnp.linalg.norm(vn, axis=-1, keepdims=True) + 1e-9)
        out["mem_kv_cos"] = (jnp.sum(kn * vn, axis=-1) * m).sum() / (n + 1e-6)
    vv = mem_v.astype(f32); vv = vv / (jnp.linalg.norm(vv, axis=-1, keepdims=True) + 1e-9)
    s = (vv * m[:, None]).sum(axis=0)                      # [v_dim]
    out["mem_value_anisotropy"] = (jnp.square(s).sum() - n) / (n * (n - 1) + 1e-9)
    return out

def main_forward(cfg, x, weights, pad_mask=None, kv=None, pos=0, collect_aux=False, pos_slot_indices=None):
    # Prepare attention mask
    attn_mask = create_mask(cfg, x, kv, pad_mask, pos)

    # embedding
    x = jax.device_put(x, P('data', None))
    x = weights['embed_tokens'].at[x, :].get(out_sharding=P('data', None, None)).astype(jnp.bfloat16)

    # iterate over hidden layers
    return_kv = kv is not None
    if kv is None: kv = defaultdict(lambda: None)

    # Collect auxiliary data from all memory layers
    all_aux_data = {} if collect_aux else None

    for i in range(cfg['num_hidden_layers']):
        layer_weights = {k.replace(prefix, ''):v for k,v in weights.items() if (prefix:=f'layers.{i}.') in k}
        if "mem_layers" in cfg and i in cfg["mem_layers"]:
            layer_weights.update({"mem_k": weights["mem_k"], "mem_v": weights["mem_v"], "mem_mask": weights["mem_mask"]})

        # partial captures collect_aux and pos_slot_indices as closed-over values.
        # _mem_layer_idx is a static per-layer tag used by the DLA ablation hook in memory_layer.
        layer_cfg = {**cfg, "_mem_layer_idx": i}
        x, kv[i], layer_aux = jax.remat(
            partial(forward_layer, layer_cfg, collect_aux=collect_aux, pos_slot_indices=pos_slot_indices)
        )(x, layer_weights, attn_mask, kv[i], pos)

        # Aggregate auxiliary data from memory layers
        if collect_aux and layer_aux is not None:
            for k, v in layer_aux.items():
                if k not in all_aux_data:
                    all_aux_data[k] = []
                all_aux_data[k].append(v)

    # logits
    out_embed = weights['embed_tokens'] if cfg['tie_word_embeddings'] else weights['lm_head']
    x = rms_norm(x, weights['norm'], cfg['rms_norm_eps'])
    logits = jnp.einsum('btd,vd->btv', x, out_embed, preferred_element_type=x.dtype, out_sharding=P('data', None, 'model'))
    
    return logits, kv, all_aux_data


def forward(cfg, x, weights, pad_mask=None, kv=None, pos=0, collect_aux=False):
    # ── Fp32-storage weight-norm telemetry (Bug 2 detector) ──────────────────────────
    # Compute BEFORE the bf16 cast so norms reflect the true fp32 storage. Reading them
    # post-cast would see bf16-rounded values and mask sub-ULP fp32 drift — exactly the
    # signal this probe exists to detect.
    telemetry_norms = None
    if collect_aux:
        main_cfg_ro = cfg['main_model']
        mem_layers = main_cfg_ro.get('mem_layers', [])
        telemetry_norms = {}
        for name in ('mem_q_proj', 'mem_q_norm', 'mem_o_norm', 'mem_layernorm', 'mem_layer_scale'):
            vals = []
            for li in mem_layers:
                key = f'main_model.layers.{li}.{name}'
                if key in weights:
                    vals.append(jnp.linalg.norm(weights[key].astype(jnp.float32)))
            if vals:
                telemetry_norms[f'{name}_norm'] = vals
        for name in ('mem_k_proj', 'mem_v_proj'):
            key = f'embed_model.{name}'
            if key in weights:
                telemetry_norms[f'embed_{name}_norm'] = jnp.linalg.norm(
                    weights[key].astype(jnp.float32)
                )

    # fp32 master weights: promote_trainable_to_fp32 stores every trainable leaf in fp32
    # to escape the bf16-ULP trap on adamw updates (utils.py::promote_trainable_to_fp32).
    # The forward pass must still run in bf16 for throughput + parity with pretrained
    # activations. Cast every fp32 weight down to bf16 here, at the boundary. VJP of
    # astype is identity, so gradients still flow to the fp32 params; adamw sees fp32
    # grads + fp32 weights + fp32 moments — the whole promotion round-trip stays lossless.
    # int weights (mem_mask) pass through unchanged.
    weights = jax.tree_util.tree_map(
        lambda w: w.astype(jnp.bfloat16) if getattr(w, 'dtype', None) == jnp.float32 else w,
        weights,
    )

    # Stage 2: an optional separate value_model namespace (base LM) supplies memory VALUES;
    # embed_model stays the KEY model. Gated on presence so control / Stage 1 are unchanged.
    value_present = 'value_model' in cfg
    if value_present:
        main_weights, embed_weights, value_weights = split_weights(weights, ['main_model', 'embed_model', 'value_model'])
        value_cfg = cfg['value_model']
    else:
        main_weights, embed_weights = split_weights(weights, ['main_model', 'embed_model'])
        value_weights, value_cfg = None, None
    main_cfg, embed_cfg = cfg['main_model'], cfg['embed_model']
    return_kv = kv is not None
    effective_doc_len = None

    if isinstance(x, dict) and "docs" in x:
        # Always compute memory dynamically from batch docs when docs are provided.
        # (The static mem_k/mem_v in model weights are only used when no docs are passed,
        # e.g. during autoregressive generation with a pre-built retrieval index.)
        if "batch" not in x:
            raise ValueError("x must have key 'batch' when 'docs' is present")
        if "batch_mask" not in pad_mask or "docs_mask" not in pad_mask:
            raise ValueError("pad_mask must have keys 'batch_mask' and 'docs_mask' when docs are provided")

        x_batch, docs = x["batch"], x["docs"]
        pad_mask_batch, docs_mask = pad_mask["batch_mask"], pad_mask["docs_mask"]

        mem_k, mem_v_embed, mem_mask, effective_doc_len = embed_forward(embed_cfg, docs, embed_weights, docs_mask)
        # Stage 2: values come from the separate value model when present (keys always from embed model).
        mem_v = value_forward(value_cfg, docs, value_weights, docs_mask) if value_present else mem_v_embed
        # Value slots must align 1:1 with key slots — indices/mem_mask/effective_doc_len all come
        # from the (post-conv) key path. value_forward does no conv, so this holds iff the key
        # conv doesn't downsample (kernel/stride 1). Guard against a silent cross-doc value read.
        if value_present:
            assert mem_k.shape[0] == mem_v.shape[0], (
                f"value bank ({mem_v.shape[0]}) misaligned with key bank ({mem_k.shape[0]}); "
                "a downsampling embed_conv on the key model is incompatible with a separate value_model")
        main_weights["mem_k"], main_weights["mem_v"], main_weights["mem_mask"] = mem_k, mem_v, mem_mask

        # Compute flat memory slot indices for every doc in the batch so that
        # mem_lookup_two_pass can always compute logits for positive docs, even when
        # pass-1 top-k retrieval misses them.  pos_slot_indices[b, d*L + t] is the
        # flat memory slot for doc d (of query b) at position t.
        pos_doc_mask = pad_mask.get("pos_doc_mask") if isinstance(pad_mask, dict) else None
        pos_slot_indices = None
        if pos_doc_mask is not None:
            B_mini = pos_doc_mask.shape[0]
            docs_per_query = pos_doc_mask.shape[1]
            mini_batch_offset = pad_mask.get("mini_batch_offset", 0) if isinstance(pad_mask, dict) else 0
            global_query_idx = mini_batch_offset * B_mini + jnp.arange(B_mini)  # [B]
            global_doc_idx = (
                global_query_idx[:, None] * docs_per_query
                + jnp.arange(docs_per_query)[None, :]
            )  # [B, docs_per_query]
            pos_slot_indices = (
                global_doc_idx[:, :, None] * effective_doc_len
                + jnp.arange(effective_doc_len)[None, None, :]
            ).reshape(B_mini, -1)  # [B, docs_per_query * effective_doc_len]

        x = x_batch
        pad_mask = pad_mask_batch
    elif "mem_k" not in main_weights or jnp.size(main_weights["mem_k"]) == 0:
        raise ValueError("x must provide 'docs' or model must have pre-computed static memory")
    else:
        pos_slot_indices = None

    # Stage 3 span readout needs the per-doc length to keep windows inside a document.
    if effective_doc_len is not None:
        main_cfg = {**main_cfg, "effective_doc_len": int(effective_doc_len)}

    logits, kv, aux_data = main_forward(main_cfg, x, main_weights, pad_mask, kv, pos, collect_aux, pos_slot_indices)
    
    # Pass effective_doc_len and post-conv mem_mask through aux_data for doc_access_acc
    if collect_aux and aux_data is not None and effective_doc_len is not None:
        aux_data['effective_doc_len'] = effective_doc_len
        aux_data['effective_mem_mask'] = main_weights["mem_mask"]
        aux_data.update(_kv_bank_telemetry(main_weights["mem_k"], main_weights["mem_v"], main_weights["mem_mask"]))

    # Fp32-storage weight-norm telemetry: merge in the pre-cast norms computed at forward()
    # entry. These read the TRUE fp32 storage, not the bf16 view the forward runs on — so
    # sub-ULP fp32 drift is visible instead of being masked by the cast.
    if collect_aux and aux_data is not None and telemetry_norms is not None:
        aux_data.update(telemetry_norms)

    return ModelOutput(
        logits=logits,
        kv=kv if return_kv else None,
        aux=aux_data
    )

def load(cfg, tp_devices=1, hf_ckpt_dir='~/weights/huggingface'):
    main_model_id = cfg.main_model.model_id
    embed_model_id = cfg.embed_model.model_id
    main_mask_type = cfg.main_model.get("mask_type", "causal")
    embed_mask_type = cfg.embed_model.get("mask_type", "causal")
    main_lora_cfg = cfg.main_model.get("lora")
    embed_lora_cfg = cfg.embed_model.get("lora")

    # Load base main model
    main_model = load_qwen3(main_model_id, tp_devices, cfg.main_model.load_weights, hf_ckpt_dir, main_mask_type, main_lora_cfg)

    # Add memory layers
    main_model.weights, main_model.cfg = add_memory_layer(cfg, main_model.cfg, main_model.weights, init_empty=True)
    
    # Load embed model (KEY model)
    embed_model = load_qwen3(embed_model_id, tp_devices, cfg.embed_model.load_weights, hf_ckpt_dir, embed_mask_type, embed_lora_cfg)
    embed_model.weights = add_kv_head(embed_model.weights, main_model.cfg)
    embed_model.weights, embed_model.cfg = add_embed_conv(cfg, embed_model.cfg, embed_model.weights)

    # Stage 2: optional separate VALUE model (base LM). add_kv_head stamps mem_v_proj onto
    # its own weights (mem_k_proj also added but unused on this model). No conv on values.
    value_model = None
    if hasattr(cfg, 'value_model') and cfg.value_model is not None:
        value_model_id = cfg.value_model.model_id
        value_mask_type = cfg.value_model.get("mask_type", "bidirectional")
        value_lora_cfg = cfg.value_model.get("lora")
        value_model = load_qwen3(value_model_id, tp_devices, cfg.value_model.load_weights, hf_ckpt_dir, value_mask_type, value_lora_cfg)
        value_model.weights = add_kv_head(value_model.weights, main_model.cfg)

    # Combine weights and configs
    if value_model is not None:
        model_weights = merge_weights(["main_model", "embed_model", "value_model"], [main_model.weights, embed_model.weights, value_model.weights])
        model_cfg = merge_configs(["main_model", "embed_model", "value_model"], [main_model.cfg, embed_model.cfg, value_model.cfg])
    else:
        model_weights = merge_weights(["main_model", "embed_model"], [main_model.weights, embed_model.weights])
        model_cfg = merge_configs(["main_model", "embed_model"], [main_model.cfg, embed_model.cfg])

    # Create forward function
    model_forward = partial(forward, model_cfg)
    
    return Model(model_weights, model_forward, main_model.init_kv, main_model.tokenizer, model_cfg)

def init(cfg, tp_devices):
    model = load(cfg, tp_devices=tp_devices)
    return model