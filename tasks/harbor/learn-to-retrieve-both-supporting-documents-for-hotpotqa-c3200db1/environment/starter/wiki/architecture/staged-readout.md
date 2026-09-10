# Staged Readout (two-pass value read + span readout)

Advanced value-read machinery in `models/memory.py`, added for the grounding experiments.
Both are off by default (control / early stages are byte-identical without them).

## Two-pass fused value read — `mem_lookup_two_pass` (`two_pass_topk=true`)
Gradient-efficient retrieval over a large dynamic bank.
- **Pass 1 (no-grad):** `stop_gradient` on q and a replicated `mem_k`; chunked scan
  (`keys_only`, `return_all_scores=False`) → just top-k **indices**, so the forward stays
  O(K) not O(M).
- **Pass 2 (with-grad):** gather the K keys, compute `[B,N,T,K]` logits + softmax. Gradients
  are exact (non-selected slots contribute zero).
- **`pos_slot_indices`:** also scores the positive-doc slots → `mem_pos_logits`/`mem_pos_indices`
  in aux, so `doc_access_top_k_loss` always has a positive in its pool even when pass-1 top-k
  missed every positive doc token.
- **Memory-scaling variants:**
  - default *norm-then-gather* (rms_norm backward on `[M,H]`, cheap when B·N·T·K ≫ M);
  - `sparse_grads=true` *gather-then-norm* → exposes `mem_k_k`/`mem_v_k` for a sparse
    scatter-add grad (use only when K·B·N·T < M);
  - `mem_value_read_kchunk>0` → `_fused_topk_logits` + `_fused_value_read` gather/contract K in
    `remat` chunk-scans, so `[B,N,T,K,H]`/`[B,N,T,K,V]` never materialize (fixes B256 OOM).
    Returns the already-reduced read `y [B,N,T,V]`; `memory_layer` skips the dense einsum.
- `mem_scores` from pass 1 are stop-gradient (correct loss *values*, no gradient).
- Incompatible with `span_window>0`; `kchunk` incompatible with `sparse_grads`.

## Span readout — `span_readout` (`span_window>0`, Stage 3)
Replaces each retrieved value with a boundary-respecting **mean over the `t ± span_w`
window** (keys are position-aligned with values, so a hit on a cue token drags in the adjacent
answer token). A window may not cross documents — validity = in-range **and** `mem_mask`
**and** same `effective_doc_len` block. Reports `mem_boundary_straddle` (share of window slots
that fell out-of-doc). Incompatible with two-pass sparse-grad mode.

**Knobs:** `two_pass_topk`, `sparse_grads`, `mem_value_read_kchunk`, `span_window`,
`effective_doc_len` (derived in `qwen3_mem_embed.forward`).
