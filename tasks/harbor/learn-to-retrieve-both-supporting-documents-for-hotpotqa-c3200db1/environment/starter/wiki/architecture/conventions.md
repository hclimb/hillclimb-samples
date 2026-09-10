# Model Conventions

Shared contracts every `models/*.py` follows. Read this before editing a model.

## The `Model` object
`qwen3.py::Model` is a dataclass: `weights`, `forward`, `init_kv`, `tokenizer`, `cfg`.
`get_model` / each `init()` returns one.

## Forward signature
`forward(cfg, x, weights, pad_mask=None, kv=None, pos=0, collect_aux=False)` → `ModelOutput`.
- `qwen3.forward` also takes `return_hidden=True` → returns final hidden states instead of
  logits (the embedding path depends on this).
- `x` is token ids `[B,T]`, or a dict `{"batch", "docs", ...}` for the embed/distill variants.
- `pos`/`kv` drive the decode-time KV cache; `collect_aux=True` populates `aux` for losses.

## Weights are plain nested dicts
Keys like `embed_tokens`, `layers.{i}.<name>`, `norm`, `lm_head`, `mem_k`. `load()` shortens
HF keys by stripping `model.` / `self_attn.` / `mlp.` / `.weight`. Multi-model variants
namespace with `merge_weights(['main_model','embed_model'], [...])` → `main_model.<key>`, and
recover via `split_weights(w, [...])`; configs likewise (`merge_configs`). (`models/utils.py`)

## `ModelOutput` (`models/output.py`)
`logits` / `kv` / `aux`. Iterable as `(logits, kv)`. `aux` carries auxiliary-loss inputs
(`mem_scores`, `mem_top_k_*`, `teacher_logits`…) and weight-0 telemetry.

## Sharding
Mesh built in `qwen3.load()` with axes `('data','model')` = **FSDP × TP**: `data =
device_count // tp_devices`, `model = tp_devices`. Every projection einsum passes
`out_sharding=P('data', …, 'model')`; weight→spec map is `get_sharding` /
`get_sharding_safe` (falls back to replicated when a dim isn't divisible by `tp_devices`).
`SINGLE_DEVICE=1` forces a 1×1 mesh (true BS=1, unsharded bank). See
[sharded-retrieval.md](sharded-retrieval.md).

## Other invariants
- **`jax.remat`** wraps every layer forward (activation checkpointing); `collect_aux` is
  captured as a *static* value via `partial`.
- **dtype:** bf16 activations, fp32 matmul accumulation (`preferred_element_type`).
- **LoRA** (`models/utils.py::init_lora`, applied in `qwen3.mlp`): adds `*_a_proj` (normal,
  σ=1/rank) + `*_b_proj` (zeros → starts as a no-op) when `cfg['lora']` is set.
- **Weight loading** (`qwen3.load`): `snapshot_download` HF safetensors, reshape q/k/v/o to
  per-head, `device_put` with sharding; `mask_type` ∈ `causal`/`bidirectional`.
