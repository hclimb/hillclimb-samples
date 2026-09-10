# Architecture

How the models are built in JAX, so a future agent can read, modify, and extend them.
Every variant is a thin wrapper over one **base transformer** (`qwen3`) plus a pluggable
**memory layer**. Weights are plain nested dicts; forwards are pure functions `forward(cfg,
x, weights, ...)` returning a [`ModelOutput`](conventions.md). Start with
[conventions.md](conventions.md) before editing any model file.

## Model variants — `models/__init__.py :: get_model(model_cfg, tp_devices)`

| `model.name` | Page | What it is |
|--------------|------|------------|
| `qwen3` | [base-transformer.md](base-transformer.md) | Vanilla Qwen3 transformer (baseline, all-trainable) |
| `qwen3_mem (DEPRECATED)` |  | + a **static** learned K/V memory bank in chosen layers |
| `qwen3_mem_embed` | [qwen3-mem-embed.md](qwen3-mem-embed.md) | + **dynamic** memory encoded from docs by an embedding model (**primary variant**) |
| `qwen3_distill (DEPRECATED)` | | Frozen teacher (`qwen3`) + student (`qwen3_mem_embed`) with KL distillation |
| `qwen3_msa` | [qwen3-msa.md](qwen3-msa.md) | Memory Sparse Attention port |

## Memory internals (`qwen3_mem_embed`)

- [memory-layer.md](memory-layer.md) — the memory-layer forward pass, placement, gating, telemetry
- [retrieval-modes.md](retrieval-modes.md) — the `mem_lookup*` family and which config selects each
- [sharded-retrieval.md](sharded-retrieval.md) — distributed top-k in `retrieval_ops.py`
- [memory-bank-construction.md](memory-bank-construction.md) — building/initializing banks, projections, conv
- [staged-readout.md](staged-readout.md) — Stage-2 two-pass value read + Stage-3 span readout

## How the pieces fit

`get_model` → base transformer (`qwen3.load`) → `add_memory_layer` splices a memory layer
into `mem_layers` → at those layers `memory_layer()` runs query-proj → **retrieval mode** →
value read → output-proj. `qwen3_mem_embed` additionally runs an embedding model over the
batch's docs each step to *build* the bank dynamically.