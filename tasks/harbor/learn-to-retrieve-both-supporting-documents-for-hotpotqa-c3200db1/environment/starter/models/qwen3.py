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

from .output import ModelOutput
from .utils import init_lora

@dataclass
class Model:
    weights: Dict
    forward: Callable
    init_kv: Callable
    tokenizer: Callable
    cfg: Dict


def apply_rope(x, theta, pos):
    B, T, N, H = x.shape
    positions = pos + jnp.broadcast_to(jnp.arange(T)[None, :], [B, T])
    freq = 1.0 / (theta ** (jnp.arange(0, H, 2, dtype=jnp.float32) / H))
    inp = jnp.einsum('bt,h->bth', positions, freq, precision=jax.lax.Precision.HIGHEST)
    sin, cos = jnp.sin(inp).astype(x.dtype), jnp.cos(inp).astype(x.dtype)
    x1, x2 = x[:, :, :, :H//2], x[:, :, :, H//2:]
    sin, cos = sin[:, :, None, :], cos[:, :, None, :] # [B, T, 1, H/2]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def rms_norm(x, gamma, eps):
    rms = jnp.sqrt(jnp.mean(x.astype(jnp.float32)**2, axis=-1, keepdims=True) + eps)
    return (gamma * x / rms).astype(x.dtype)


def self_attention(cfg, x, w, attn_mask, kv=None, pos=0):
    B, T, D = x.shape

    # input norm
    x_norm = rms_norm(x, w['input_layernorm'], cfg['rms_norm_eps'])
    
    # QKV projection
    q = jnp.einsum('btd,nhd->btnh', x_norm, w['q_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, 'model', None))
    k = jnp.einsum('bsd,khd->bskh', x_norm, w['k_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, 'model', None))
    v = jnp.einsum('bsd,khd->bskh', x_norm, w['v_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, 'model', None))

    # Q/K norm
    q = rms_norm(q, w['q_norm'], cfg['rms_norm_eps'])
    k = rms_norm(k, w['k_norm'], cfg['rms_norm_eps'])
    
    # RoPE
    q = apply_rope(q, cfg['rope_theta'], pos)
    k = apply_rope(k, cfg['rope_theta'], pos)

    # load kv cache
    if kv is not None:
        kv = jax.lax.dynamic_update_slice(kv, jnp.stack([k, v]), (0, 0, pos, 0, 0))
        k, v = kv

    # grouped-query attention
    attn_out = jax.nn.dot_product_attention(q, k, v, mask=attn_mask)

    # O projection
    o = jnp.einsum('btnh,dnh->btd', attn_out, w['o_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, None, None))
    x += o

    return x, kv


def mlp(cfg, x, w):
    x_norm = rms_norm(x, w['post_attention_layernorm'], cfg['rms_norm_eps'])
    # LoRA is applied only when this model was loaded with lora.enabled=true (load() sets cfg['lora']
    # and init_lora creates the *_a_proj/*_b_proj adapters). Non-LoRA models have cfg.get('lora')==None
    # -> every branch below is skipped -> identical to the base forward. B is zero-init so a fresh
    # adapter starts as a no-op.
    lora_cfg = cfg.get('lora')
    alpha = lora_cfg.get('alpha', 32) if lora_cfg else 32

    # Gate proj
    gate_base = jnp.einsum('btd,fd->btf', x_norm, w['gate_proj'], preferred_element_type=jnp.float32, out_sharding=P('data', None, 'model'))
    if lora_cfg is not None and 'gate_a_proj' in w:
        rank = w['gate_a_proj'].shape[0]
        scaling = alpha / rank
        lora_gate = jnp.einsum('btd,rd->btr', x_norm, w['gate_a_proj'], preferred_element_type=jnp.float32)
        lora_gate = jnp.einsum('btr,fr->btf', lora_gate, w['gate_b_proj'], preferred_element_type=jnp.float32, out_sharding=P('data', None, 'model'))
        gate_base = gate_base + lora_gate * scaling
    gate = jax.nn.silu(gate_base)

    # Up proj
    up_base = jnp.einsum('btd,fd->btf', x_norm, w['up_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, 'model'))
    if lora_cfg is not None and 'up_a_proj' in w:
        rank = w['up_a_proj'].shape[0]
        scaling = alpha / rank
        lora_up = jnp.einsum('btd,rd->btr', x_norm, w['up_a_proj'], preferred_element_type=jnp.float32)
        lora_up = jnp.einsum('btr,fr->btf', lora_up, w['up_b_proj'], preferred_element_type=jnp.float32, out_sharding=P('data', None, 'model'))
        up_base = up_base + (lora_up * scaling).astype(up_base.dtype)
    up = up_base

    # Down proj
    down_base = jnp.einsum('btf,df->btd', gate * up, w['down_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, None))
    if lora_cfg is not None and 'down_a_proj' in w:
        rank = w['down_a_proj'].shape[0]
        scaling = alpha / rank
        lora_down = jnp.einsum('btf,rf->btr', gate * up, w['down_a_proj'], preferred_element_type=jnp.float32)
        lora_down = jnp.einsum('btr,dr->btd', lora_down, w['down_b_proj'], preferred_element_type=jnp.float32, out_sharding=P('data', None, None))
        down_base = down_base + (lora_down * scaling).astype(down_base.dtype)

    x += down_base.astype(x.dtype)
    return x


def forward_layer(cfg, x, w, attn_mask, kv=None, pos=0):
    x, kv = self_attention(cfg, x, w, attn_mask, kv, pos)
    x = mlp(cfg, x, w)
    return x, kv


def forward_window_scores(cfg, x, weights, kv, pos, pad_mask=None):
    """SnapKV observation-window probe (additive helper for scripts/embed/snapkv_stream.py;
    the standard forward()/self_attention paths are untouched).

    Runs a normal forward of the window chunk `x` — updating `kv` at `pos` exactly like
    forward() would — and ALSO returns, per layer, the softmax attention mass each cached KV
    position received from the window's queries, aggregated over window positions and the
    query heads of each KV group. jax.nn.dot_product_attention cannot expose these scores,
    hence the duplicated layer loop.

    Returns (kv, scores): kv is the updated cache; scores is a list over layers of
    [B, num_kv_heads, S] float32 (S = cache length).
    """
    attn_mask = create_mask(cfg, x, kv, pad_mask, pos)
    x = jax.device_put(x, P('data', None))
    x = weights['embed_tokens'].at[x, :].get(out_sharding=P('data', None, None)).astype(jnp.bfloat16)
    scores = []
    for i in range(cfg['num_hidden_layers']):
        w = {k.replace(prefix, ''): v for k, v in weights.items() if (prefix := f'layers.{i}.') in k}
        x_norm = rms_norm(x, w['input_layernorm'], cfg['rms_norm_eps'])
        q = jnp.einsum('btd,nhd->btnh', x_norm, w['q_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, 'model', None))
        k = jnp.einsum('bsd,khd->bskh', x_norm, w['k_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, 'model', None))
        v = jnp.einsum('bsd,khd->bskh', x_norm, w['v_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, 'model', None))
        q = rms_norm(q, w['q_norm'], cfg['rms_norm_eps'])
        k = rms_norm(k, w['k_norm'], cfg['rms_norm_eps'])
        q = apply_rope(q, cfg['rope_theta'], pos)
        k = apply_rope(k, cfg['rope_theta'], pos)
        kv[i] = jax.lax.dynamic_update_slice(kv[i], jnp.stack([k, v]), (0, 0, pos, 0, 0))
        k_full, v_full = kv[i]

        Bq, Tq, N, Hd = q.shape
        Kh = k_full.shape[2]
        qg = q.reshape(Bq, Tq, Kh, N // Kh, Hd).astype(jnp.float32)
        logit = jnp.einsum('btkgh,bskh->bkgts', qg, k_full.astype(jnp.float32)) / jnp.sqrt(float(Hd))
        # attn_mask from create_mask is [B/1, 1, T, S] (heads dim already present) ->
        # insert only the group dim to broadcast against logit [B, K, G, T, S]
        logit = jnp.where(attn_mask[:, :, None, :, :], logit, jnp.float32(-1e30))
        p = jax.nn.softmax(logit, axis=-1)
        # label-driven reduction: identical to sum(axis=(2,3)) when p is [B,K,G,T,S], but
        # errors loudly with the true shape if anything upstream disagrees
        scores.append(jnp.einsum('bkgts->bks', p))                 # [B, Kh, S]

        attn_out = jax.nn.dot_product_attention(q, k_full, v_full, mask=attn_mask)
        o = jnp.einsum('btnh,dnh->btd', attn_out, w['o_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, None, None))
        x = x + o
        x = mlp(cfg, x, w)
    return kv, scores


def create_mask(cfg, x, kv, pad_mask, pos=0):
    T = x.shape[1]
    S = kv[0].shape[2] if (kv is not None and kv[0] is not None) else T
    
    if cfg['mask_type'] == 'causal':
        rows = jnp.arange(T)[:, None]
        cols = jnp.arange(S)[None, :]
        attn_mask = (cols <= rows + pos)[None]
    elif cfg['mask_type'] == 'bidirectional':
        attn_mask = jnp.ones((T, S), dtype=bool)[None]
    else:
        raise ValueError(f"Unknown mask type: {cfg['mask_type']}")
    
    if pad_mask is not None:
        assert pad_mask.ndim == 2 and pad_mask.shape[1] == S, f"pad_mask.shape={pad_mask.shape}, S={S}, should be shape (B, S)"
        attn_mask = attn_mask & pad_mask[:, None, :]
    
    attn_mask = attn_mask[:, None].astype(jnp.bool_)
    return attn_mask


def forward(cfg, x, weights, pad_mask=None, kv=None, pos=0, return_hidden=False, collect_aux=False):
    """
    Forward pass through the model.
    
    Args:
        cfg: Model config
        x: Input token ids [B, T]
        weights: Model weights dict
        pad_mask: Padding mask [B, T]
        kv: KV cache (optional)
        pos: Position offset for KV cache
        return_hidden: Return hidden states instead of logits
        collect_aux: Whether to collect auxiliary data (ignored for base model)
    
    Returns:
        ModelOutput with logits, optional kv (base model has no aux data)
    """
    # Prepare attention mask
    attn_mask = create_mask(cfg, x, kv, pad_mask, pos)

    # embedding
    x = jax.device_put(x, P('data', None))
    x = weights['embed_tokens'].at[x, :].get(out_sharding=P('data', None, None)).astype(jnp.bfloat16)
    
    # iterate over hidden layers
    return_kv = kv is not None
    if kv is None: kv = defaultdict(lambda: None)
    for i in range(cfg['num_hidden_layers']):
        layer_weights = {k.replace(prefix, ''):v for k,v in weights.items() if (prefix:=f'layers.{i}.') in k}
        x, kv[i] = jax.remat(partial(forward_layer, cfg))(x, layer_weights, attn_mask, kv=kv[i], pos=pos)

    # logits
    out_embed = weights['embed_tokens'] if cfg['tie_word_embeddings'] else weights['lm_head']
    x = rms_norm(x, weights['norm'], cfg['rms_norm_eps'])
    if return_hidden:
        return x

    logits = jnp.einsum('btd,vd->btv', x, out_embed, preferred_element_type=x.dtype, out_sharding=P('data', None, 'model'))

    return ModelOutput(
        logits=logits,
        kv=kv if return_kv else None,
        aux=None
    )


def get_sharding(key):
    # Standard Qwen layers (2D weights from checkpoint)
    if any(k in key for k in ('q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj')):
        if key.endswith('_a_proj'): return P(None, 'data')
        if key.endswith('_b_proj'): return P('model', None)
        return P('model', 'data')
    if any(k in key for k in ('o_proj', 'down_proj')):
        if key.endswith('_a_proj'): return P(None, 'model')
        if key.endswith('_b_proj'): return P('data', None)
        return P('data', 'model')
    if any(k in key for k in ('embed_tokens', 'lm_head')):
        return P('model', 'data')

    return P()


def get_sharding_safe(key, shape, tp_devices):
    """Like get_sharding but falls back to replicated for non-divisible dims."""
    spec = get_sharding(key)
    for i, axis in enumerate(spec):
        if axis is not None and i < len(shape) and shape[i] % tp_devices != 0:
            return P()
    return spec


def init_kv(L, K, H, B, T):
    sharding = P(None, 'data', None, 'model', None)
    kv = [jnp.zeros((2, B, T, K, H), dtype=jnp.bfloat16, out_sharding=sharding) for _ in range(L)]
    return kv


def load(model_id='Qwen/Qwen3-0.6B-Base', tp_devices=1, load_weights=True, hf_ckpt_dir='~/weights/huggingface', mask_type='causal', lora_cfg=None):
    """loads huggingface checkpoint"""
    hf_token = os.environ.get("HF_TOKEN")

    model_ckpt_dir = Path(hf_ckpt_dir).expanduser() / model_id

    # download checkpoint
    if not model_ckpt_dir.exists():
        snapshot_download(repo_id=model_id, local_dir=model_ckpt_dir, token=hf_token)

    # load tokenizer
    tokenizer_config = json.loads((model_ckpt_dir/'tokenizer_config.json').read_text())
    tokenizer_config['added_tokens_decoder'] = {int(k): AddedToken(**v) for k, v in tokenizer_config['added_tokens_decoder'].items()}
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(model_ckpt_dir/'tokenizer.json'), **tokenizer_config)

    # load model config
    cfg = json.loads((model_ckpt_dir/'config.json').read_text())
    L, N, K, H, D = cfg['num_hidden_layers'], cfg['num_attention_heads'], cfg['num_key_value_heads'], cfg['head_dim'], cfg['hidden_size']

    # define sharding (FSDP + TP)
    # SINGLE_DEVICE=1: build a 1x1 mesh on a single chip so there is NO parallelism of
    # any kind (no data-parallel batch sharding, no bank sharding) and a true global
    # batch of 1 is expressible — P('data'/'model', ...) become no-ops over size-1 axes.
    # This is the "no sharding, BS=1, single-stream" regime; it also makes the memory
    # bank inherently unsharded (full bank on the one chip), matching MSA/RAG.
    if os.environ.get("SINGLE_DEVICE"):
        mesh = jax.make_mesh((1, 1), ('data', 'model'),
                             axis_types=(AxisType.Explicit, AxisType.Explicit),
                             devices=jax.devices()[:1])
    else:
        fsdp_devices = jax.device_count() // tp_devices
        mesh = jax.make_mesh((fsdp_devices, tp_devices), ('data', 'model'),
                             axis_types=(AxisType.Explicit, AxisType.Explicit))
    jax.set_mesh(mesh)

    # load or init weights
    weights = {}
    if load_weights:
        for file in model_ckpt_dir.glob('*.safetensors'):
            with safe_open(file, framework='numpy') as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    weights[key] = jax.device_put(tensor, get_sharding_safe(key, tensor.shape, tp_devices))
    else:
        rng_key = jax.random.PRNGKey(42)
        for file in model_ckpt_dir.glob('*.safetensors'):
            with safe_open(file, framework='numpy') as f:
                for key in f.keys():
                    rng_key, subkey = jax.random.split(rng_key)
                    shape = f.get_tensor(key).shape
                    weights[key] = jax.device_put(jax.nn.initializers.he_normal()(subkey, shape, jnp.bfloat16), get_sharding_safe(key, shape, tp_devices))

    # define mask type
    cfg['mask_type'] = mask_type

    # shorten layer keys
    substrings = ['model.', 'self_attn.', 'mlp.', '.weight']
    weights = {reduce(lambda k, s: k.replace(s, ''), substrings, k):v for k,v in weights.items()}
    
    # split head dimension
    for key in weights.keys():
        if 'q_proj' in key and not (key.endswith('_a_proj') or key.endswith('_b_proj')): weights[key] = weights[key].reshape([N, H, D])
        if 'k_proj' in key and not (key.endswith('_a_proj') or key.endswith('_b_proj')): weights[key] = weights[key].reshape([K, H, D])
        if 'v_proj' in key and not (key.endswith('_a_proj') or key.endswith('_b_proj')): weights[key] = weights[key].reshape([K, H, D])
        if 'o_proj' in key and not (key.endswith('_a_proj') or key.endswith('_b_proj')): weights[key] = weights[key].reshape([D, N, H])

    # Initialize LoRA weights if enabled
    if lora_cfg and lora_cfg.get('enabled', False):
        weights = init_lora(weights, lora_cfg, get_sharding)
        cfg['lora'] = lora_cfg
    
    model_forward = partial(forward, cfg)
    model_init_kv = partial(init_kv, L, K, H)
    
    return Model(weights, model_forward, model_init_kv, tokenizer, cfg)


def init(cfg, tp_devices):
    model = load(cfg.main_model.model_id, tp_devices=tp_devices, load_weights=cfg.main_model.load_weights, mask_type=cfg.main_model.get("mask_type", "causal"))
    return model