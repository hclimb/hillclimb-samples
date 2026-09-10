# `apply_conv1d` casts weight/bias to the activation's dtype (fp32-promoted conv weights crashed)

**Date:** 2026-08-13 · **Author:** rohunagrawal (with Claude) · **Status:** done · **Commit/PR:**
none yet (working tree on `multihop-finetuning`).

## What changed

`models/conv_utils.py::apply_conv1d` now casts `weight` (and `bias`, against the conv's output
dtype) to the input activation's dtype before calling `lax.conv_general_dilated`. Previously it
passed both straight through, which crashes whenever the conv weight and activation dtypes
differ.

## Motivation & context

Asked to run the hybrid hotpotqa eval against
`qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45`
(step 100000). It crashed while building the document bank:

```
TypeError: lax.conv_general_dilated requires arguments to have the same dtypes,
got bfloat16, float32.
```

Root cause: this checkpoint's trainable regex includes `.*embed_proj_conv.*` (confirmed in its
`.hydra/config.yaml`), so `utils.py::promote_trainable_to_fp32` promoted
`embed_proj_conv_{k,v}_weight/bias` to fp32 for training (its own documented contract: "store
trainable weights in fp32, cast to bf16 only inside the forward pass" — the whole reason it
exists is the bf16-ULP-freeze family of bugs this repo has chased repeatedly, e.g.
[2026-08-13-warmstart-restore-dtype-fix.md](2026-08-13-warmstart-restore-dtype-fix.md)). The rest
of the embed backbone (frozen) stays bf16, so `embed_forward`'s `last_hidden_states` is bf16 while
the promoted conv weight is fp32 — and `apply_conv1d` calls the raw `lax.conv_general_dilated`
primitive directly with no cast.

Every other promoted-weight consumption path in this codebase tolerates the mismatch silently:
`jnp.einsum` (used everywhere else — `models/qwen3.py`, `models/memory.py`) does NumPy-style
implicit type promotion on mismatched operand dtypes before dispatching to the underlying dot,
so a stray fp32 leaf next to bf16 activations just works (with `preferred_element_type=x.dtype`
controlling the *output* accumulation dtype, not requiring matching *inputs*). `lax.conv_general_dilated`
is a lower-level primitive with no such promotion — hence this is the one place in the whole
forward pass that a fp32-promoted leaf actually breaks instead of silently working. First hit
because no eval had exercised `embed_conv: true` + a promoted `embed_proj_conv` weight together
before.

## Options weighed

1. **Cast the activation up to the weight's dtype** (fp32 compute). Rejected: contradicts the
   established, deliberate "everything memory-layer-related stays bf16 at eval/inference time"
   convention elsewhere in this exact file family (e.g. `gen_large_mem_rag_hybrid.py`'s
   `_put_replicated(..., dtype=jnp.bfloat16)` explicitly downcasts the bank regardless of stored
   checkpoint dtype) — would silently double this op's memory/compute cost and diverge from every
   other computation path's behavior.
2. **Cast the weight/bias down to the activation's dtype** (chosen). Matches
   `promote_trainable_to_fp32`'s own stated contract exactly ("cast to bf16 only inside the
   forward pass") and matches what `jnp.einsum` already does implicitly everywhere else in the
   forward pass — this fix just makes the one primitive that doesn't auto-promote behave the same
   way. No-op when weight/bias already match `x`'s dtype (the common case for non-pf32
   checkpoints), so every existing run is unaffected.

## How it was built & integrated

- `models/conv_utils.py::apply_conv1d`: `weight = weight.astype(x_t.dtype)` before the
  `conv_general_dilated` call; `bias` cast to `out.dtype` (the conv output's dtype, which now
  matches `x`'s) before the add. `pool_pad_mask` untouched (already dtype-safe — casts internally
  and casts back).
- No caller changes needed — `embed_forward` (`models/qwen3_mem_embed.py`) passes weights straight
  through regardless of their stored dtype, same as before.
- `tests/test_conv_utils_dtype.py` (new): builds fp32 weight/bias against bf16 `x` (the
  regression case — previously raised), asserts it now succeeds, returns `x`'s dtype, and matches
  a reference call with weight/bias pre-cast to bf16; also asserts the untouched same-dtype path
  (fp32 weight against fp32 `x`) is unaffected. Along the way, confirmed (empirically, via a shape
  error) that `apply_conv1d`'s own docstring is wrong about the weight layout: the actual
  `dimension_numbers=('NCH','IOH','NCH')` requires `(in_channels, out_channels, kernel)`, not
  `(out_channels, in_channels, kernel)` as the comment claims — harmless in every real checkpoint
  today because `add_embed_conv` always allocates `[d_embed, d_embed, kernel]` (square, so the
  ordering is invisible), but worth fixing the comment if this function is ever touched again.

## Reference pages updated

- [architecture/memory-bank-construction.md](../architecture/memory-bank-construction.md):
  `add_embed_conv` section now documents the dtype-cast behavior and why it's needed.

## Tests

`JAX_PLATFORMS=cpu uv run python tests/test_conv_utils_dtype.py` (run via the launcher on
`tn-v6e-8-0`, no local accelerator):

```
[PASS] seed=0: dtype_ok=True match_ok=True same_dtype_ok=True
[PASS] seed=1: dtype_ok=True match_ok=True same_dtype_ok=True
[PASS] seed=2: dtype_ok=True match_ok=True same_dtype_ok=True
[PASS] seed=3: dtype_ok=True match_ok=True same_dtype_ok=True
[PASS] seed=4: dtype_ok=True match_ok=True same_dtype_ok=True

ALL TESTS PASSED
```

Also reran `tests/test_mem_lookup_batched.py` (the other retrieval-path fix from earlier the same
day) to confirm no interaction — still all-PASS, unchanged.

Real-hardware confirmation: re-ran the eval that surfaced this bug
(`qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45`
step 100000, hybrid hotpotqa, `tn-v6e-8-0`) with the fix — see
[wiki/experiments/2026-08-13-hotpotqa-hybrid-pf32-indexed-lrmasked-checkpoint.md](../experiments/2026-08-13-hotpotqa-hybrid-pf32-indexed-lrmasked-checkpoint.md)
for the eval result.

## Follow-ups & risks

- The docstring/shape-comment fix noted above (`apply_conv1d`'s weight-layout comment) is cosmetic
  and left as-is — not touched here since it doesn't affect correctness for any real checkpoint.
- No other `lax.*` primitive in this codebase was audited for the same "doesn't auto-promote
  mismatched dtypes" gap; `apply_conv1d` was the one this session's eval actually exercised. Worth
  a quick sweep if another `.*embed_proj_conv.*`-style promoted-weight-through-a-raw-lax-primitive
  pattern shows up elsewhere.
