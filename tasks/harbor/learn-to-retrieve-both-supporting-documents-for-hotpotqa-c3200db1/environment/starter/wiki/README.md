# Memory Layers — Wiki

Agent-facing documentation for the **memory-layers** codebase: a JAX/TPU research framework
that augments a frozen Qwen3 LLM with a trainable **external memory** it retrieves from
(primary model: `qwen3_mem_embed` — an embedding model encodes the batch's documents into a
K/V bank each step, and the main LLM retrieves from it). These pages explain how the code works
and how to run it, grounded in the source. Start with a section's README, then drill in.

## Sections

| Section | Start here | What's inside |
|---------|-----------|---------------|
| [infrastructure](infrastructure/README.md) | [experiment-launch-instructions](infrastructure/experiment-launch-instructions.md) | Standing up TPUs, launching/resuming runs, checkpointing |
| [architecture](architecture/README.md) | [conventions](architecture/conventions.md) | The model variants + the memory-layer internals |
| [training](training/README.md) | [training-loop](training/training-loop.md) | Training loop, multi-stage schedule, optimizer, losses, configs |
| [data](data/README.md) | [batch-format](data/batch-format.md) | Datasets, the batch contract, normalizers, corpus prep |
| [evaluation](evaluation/README.md) | [two-process-design](evaluation/two-process-design.md) | Scoring checkpoints: evaluators, LLM judge, RULER, RAG |

## Orientation
- **Compute:** TRC `trc2` v6e-8 TPUs; runs launch via `multi-vm-tpu-run.sh` and checkpoint to GCS
  ([infrastructure](infrastructure/README.md)).
- **Model:** base Qwen3 transformer + a memory layer spliced into chosen layers; the bank is
  built dynamically from docs ([architecture](architecture/README.md)).
- **Convention:** every page cites the `file` / function it documents, so it doubles as a map
  into the code.
