# Memory Bank Construction

Splicing memory projections, banks, and the embed-path conv onto a loaded model. All in
`models/memory_utils.py` (+ `conv_utils.py`); each returns `(weights, model_cfg)`.

## `add_memory_layer(cfg, model_cfg, weights, init_empty=False)`
For each layer in `mem_layers`:
- `replace_mlp` → deletes that layer's `gate/up/down_proj`; `after_attention` → adds a
  `mem_layernorm`.
- Adds `mem_q_proj` `[N,k_dim,D]`, `mem_o_proj` `[D,N,v_dim]` (init σ=0.02, or **zeros** when
  `mem_o_proj_zero_init` — memory starts as identity so any CE drop is provably memory-sourced),
  plus `mem_q_norm`/`mem_k_norm`/`mem_o_norm` (ones) and scalar `mem_layer_scale=0.1`.
- Optional gating weights (`mem_gate_proj`, `mem_up_proj`) when `mem_use_gating`.
- **Static bank** (`qwen3_mem`): `mem_k`, `mem_v` `[mem_size, dim]` (product keys →
  `[2, √mem_size, k_dim/2]`, requires perfect-square `mem_size`). `init_empty=True`
  (`qwen3_mem_embed`) stores empty `mem_k/mem_v` — the bank is built per-step from docs.

## `add_kv_head(weights, cfg)` — embed/value model projections
Adds `mem_k_proj`, `mem_v_proj` mapping encoder hidden → bank vectors.
- `mem_num_kv_heads (nkv) > 0` → **3-D** `[d_embed, Nkv, dim]` = GQA per-kv-head bank.
- else 2-D `[d_embed, dim]` single shared bank.

## `add_embed_conv(cfg, model_cfg, weights)` — sequence compression
When `embed_conv`, adds separate k/v 1D convs (`embed_proj_conv_{k,v}_{weight,bias}`,
`[d_embed, d_embed, kernel]`). Mechanics in `conv_utils.py`: `apply_conv1d`
(`conv_general_dilated`, channel-first) and `pool_pad_mask` (max-pool the pad mask to the new
length). Stride>1 downsamples — incompatible with a separate value model
(see [qwen3-mem-embed.md](qwen3-mem-embed.md)).

`apply_conv1d` casts `weight`/`bias` to the input activation's dtype before the conv — needed
because `lax.conv_general_dilated` (unlike `jnp.einsum` elsewhere in this codebase) does **not**
implicitly promote mismatched dtypes and raises instead. This matters whenever
`.*embed_proj_conv.*` is in a trainable regex: `utils.py::promote_trainable_to_fp32` promotes
those leaves to fp32 for training, per its own contract ("cast to bf16 only inside the forward
pass") — `apply_conv1d` is where that cast-back happens for this weight family. See
[implementation note](../implementations/2026-08-13-conv1d-mixed-dtype-cast.md).

## Also here
`get_memory_sharding(name)` (weight→`PartitionSpec`), `bank_top_k` (approx-top-k toggle),
`chunked_memory_top_k_retrieval` (single-device scan retrieval), `_pad_memory_to_chunk_multiple`.
