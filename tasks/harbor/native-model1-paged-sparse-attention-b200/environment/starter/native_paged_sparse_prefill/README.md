# SGLang FlashMLA Prefill Experiments

This subtree tracks a clean SGLang/FlashMLA-based DeepSeek V4 prefill
experiment. The old standalone `flash_mla_paged_sparse_prefill` prototype has
been removed from the refreshed upstream trees.

## Scope

The active implementation keeps SGLang on its current FlashMLA sparse kvcache
path and adds an optional prefill row-packing mode:

- no BF16 materialization workspace is introduced,
- SWA and compressed sparse indices stay native to SGLang/FlashMLA,
- invalid short-prefix entries stay represented by `-1` padded indices,
- row packing uses FlashMLA's existing `s_q` dimension instead of a separate
  custom operator.

## Workspace Contract

- `upstreams/sglang` is a refreshed SGLang tree with the experiment patch.
- `upstreams/FlashMLA` is a refreshed FlashMLA tree with no standalone native
  paged-prefill operator.
- `upstreams/vllm` is refreshed for baseline comparison only.

## Layout

```text
native_paged_sparse_prefill/
  upstreams/
    FlashMLA/
    sglang/
    vllm/
  validation/
```

## Runtime Switches

- `SGLANG_DSV4_FLASHMLA_PREFILL_ROW_TILE_M=1|2|4|8`
  controls optional row packing. `1` is the upstream behavior and default.
- `SGLANG_DSV4_FLASHMLA_PREFILL_ROW_TILE_REQUIRED=1`
  makes an unpackable batch fail instead of silently using the upstream shape.
