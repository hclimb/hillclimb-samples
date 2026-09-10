# Base Transformer (`qwen3`)

`models/qwen3.py` — the Qwen3 transformer in JAX. Every variant loads this and splices in
extra behavior. Baseline model; all params trainable. (Adapted from martin-marek/jax-llm.)

## Components
| Fn | What |
|----|------|
| `apply_rope(x, theta, pos)` | Rotary position embedding (rotate-half), applied to q & k |
| `rms_norm(x, gamma, eps)` | RMSNorm (fp32 reduction) |
| `self_attention` | RMSNorm → q/k/v proj → **q/k RMSNorm** → RoPE → KV-cache update → `jax.nn.dot_product_attention` (**GQA**, `K` kv-heads ≤ `N` q-heads) → o-proj → residual |
| `mlp` | RMSNorm → **SwiGLU** `down(silu(gate)·up)` (+ optional LoRA) → residual |
| `create_mask` | `causal` (lower-triangular incl. `pos` offset) or `bidirectional`; AND-ed with `pad_mask` |
| `forward` | embed → `num_hidden_layers` × `remat(forward_layer)` → final `rms_norm` → `lm_head` (tied to `embed_tokens` iff `tie_word_embeddings`) |

## Forward pass
```
ids [B,T] → embed_tokens[ids] → for each layer: attn(+RoPE,GQA) ; mlp(SwiGLU)
          → rms_norm → einsum('btd,vd->btv') logits   (return_hidden short-circuits before logits)
```

## Inference / loading
- **KV cache:** `init_kv(L,K,H,B,T)` allocates `[2,B,T,K,H]` per layer; `self_attention` writes
  via `dynamic_update_slice` at `pos`.
- **`load(model_id, tp_devices, load_weights, hf_ckpt_dir, mask_type, lora_cfg)`** — builds the
  mesh, downloads HF safetensors, shortens keys, reshapes q/k/v/o to per-head, shards
  (`get_sharding_safe`), optionally inits LoRA. Returns a `Model`.

**Config:** `configs/model/qwen3.yaml`  ·  **Conventions:** [conventions.md](conventions.md)
