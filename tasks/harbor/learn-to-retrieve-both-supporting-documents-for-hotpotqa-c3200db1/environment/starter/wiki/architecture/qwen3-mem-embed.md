# `qwen3_mem_embed` — Embedding-Retrieval Memory

`models/qwen3_mem_embed.py` — **the primary variant**. A second (embedding) model encodes
the batch's documents into a K/V bank *every step*; the main LLM retrieves from that dynamic
bank at `mem_layers`. Trains memory + embedding params (`.*mem_.*`, `.*embed_model.*`).
This file has grown past the repo README — the newer machinery is below.

## Sub-models (namespaced weights)
- `main_model` — the LLM that reads memory (`memory_layer` at `mem_layers`).
- `embed_model` — the **KEY** model; bidirectional attention over docs.
- `value_model` *(optional, Stage 2)* — a separate base LM that supplies memory **VALUES**;
  keys still come from the embed model. Gated on `cfg.value_model` (control/Stage 1 unchanged).

## `forward` flow (when `x` has `docs`)
1. `split_weights` into main/embed(/value).
2. `embed_forward(embed_cfg, docs, …)` → `mem_k, mem_v_embed, mem_mask, effective_doc_len`.
3. If a value model is present, `value_forward` → `mem_v` (asserts 1:1 slot alignment with
   keys — a downsampling key-side conv is therefore incompatible with a value model).
4. Build `pos_slot_indices` (flat bank slot per positive-doc token) from `pos_doc_mask`, for
   the two-pass read ([staged-readout.md](staged-readout.md)).
5. Stash `mem_k/mem_v/mem_mask` into `main_weights`, run `main_forward` (= base transformer
   with `memory_layer` spliced in), passing `effective_doc_len` for span readout.
- No `docs` → uses the **static** `mem_k/mem_v` already in weights (pre-built retrieval index,
  e.g. autoregressive generation).

## `embed_forward` details
`qwen3_forward(return_hidden=True)` → optional **1D conv** compression (separate k/v convs,
`embed_conv`) → project to bank. Projection shape decides the bank:
- 2-D `mem_{k,v}_proj [d_embed, dim]` → flat bank `[M, dim]`.
- 3-D `mem_{k,v}_proj [d_embed, Nkv, dim]` → **GQA per-kv-head** bank `[M, Nkv, dim]`
  (see `mem_lookup_gqa` in [retrieval-modes.md](retrieval-modes.md)).

## Telemetry
`_kv_bank_telemetry` (weight-0): `mem_kv_cos` (key/value cosine — should drop as key/value
models specialize) and `mem_value_anisotropy`.

## `load`
base main model → `add_memory_layer(init_empty=True)` (bank built per-step, not stored) →
embed model → `add_kv_head` + `add_embed_conv` → optional value model
(`add_kv_head`) → merge. See [memory-bank-construction.md](memory-bank-construction.md).

**Config:** `configs/model/qwen3_mem_embed.yaml`
**Knobs (beyond `qwen3_mem`):** `embed_conv`(+`_kernel_size`/`_stride`), `embed_model.lora`,
`mask_type: bidirectional`, `mem_num_kv_heads`, `two_pass_topk`, `span_window`, `value_model`.
