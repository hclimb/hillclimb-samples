# The Memory Layer

`models/memory.py::memory_layer(cfg, x, w, collect_aux, pos_slot_indices)` — the memory
forward spliced in at `mem_layers`. Writes an additive `o` into the residual stream.

## Pipeline
1. `MEM_TOP_K` env override (inference knob).
2. **Norm:** `mem_layernorm` if `mem_placement=="after_attention"`, else `post_attention_layernorm`.
3. **Query:** `q = mem_q_proj(x_norm)` `[B,T,N,H]` → per-head `mem_q_norm` (RMSNorm).
4. **Retrieval mode** → `(top_k_scores, top_k_values, aux)`; dispatch (first match wins):

   | Condition | Mode |
   |-----------|------|
   | `mem_k.ndim == 3` | `mem_lookup_gqa` (per-kv-head bank) |
   | `two_pass_topk` | `mem_lookup_two_pass` |
   | `mem_use_product_keys` | `product_key_lookup` |
   | `mem_lookup_chunk_size` set | `mem_lookup_chunked` |
   | else | `mem_lookup` (full matrix) |

   See [retrieval-modes.md](retrieval-modes.md). GQA takes priority and ignores the
   span/two-pass/product-key/chunk knobs.
5. **Span readout** (`span_window>0`): replace values with a doc-bounded neighbor-window mean
   ([staged-readout.md](staged-readout.md)).
6. **Value read:** `y = einsum('bntk,bntkv->btnv', scores, values)` — *unless* two-pass already
   returned a fused, reduced read (`mem_value_read_kchunk>0`), then just transpose.
7. **Output:** gating (`mem_use_gating` → `input_gate`) or `o = einsum('btnv,dnv->btd', y,
   mem_o_proj)`.
8. `MEM_ABLATE_LAYER` DLA hook (zero this layer's write for eval attribution).
9. Telemetry (`_memory_telemetry`) when `collect_aux`; `x += o`.

## Scoring knobs (`_mem_score_opts`, env-overridable at eval)
`mem_score_activation` (`softmax`/`relu`/`sigmoid`; `MEM_SCORE_ACTIVATION`), `mem_softmax_temp`
(`MEM_SOFTMAX_TEMP`), `mem_phantom_log_n` — see `mem_weight_from_logits` in
[sharded-retrieval.md](sharded-retrieval.md). GQA/two-pass use a plain softmax (knobs N/A).

## Placement
Chosen in each variant's `forward_layer`, not here: `replace_mlp` / `after_attention`
(memory + MLP) / `after_mlp`.

> Note: `mem_o_norm` and `mem_layer_scale` are allocated by `add_memory_layer` but not
> applied in the current read path.
