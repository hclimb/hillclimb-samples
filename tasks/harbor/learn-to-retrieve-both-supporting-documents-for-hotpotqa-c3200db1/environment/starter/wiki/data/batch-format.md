# Batch Format

The contract between the data pipeline and the model/trainer. `QADataset.generator()` yields
`(x, masks)` tuples in one of three shapes depending on config.

## The three shapes
**With docs** (`provide_docs`, default for `qwen3_mem_embed`):
```python
x     = {"batch": inputs[B,T], "docs": flat_docs[B·M, dc]}
masks = {"batch_mask", "docs_mask", "loss_mask", "pos_doc_mask", "pos_doc_ids", "ce_enable"}
```
**Distill** (`distill=true`): `x` adds `"teacher_batch"`; `masks` adds `"teacher_mask"`,
`"student_distill_mask"`, `"teacher_distill_mask"`.
**No docs** (`provide_docs=false`): `x = inputs[B,T]`; `masks = {batch_mask, loss_mask,
pos_doc_ids, ce_enable}`.

Docs are chunked per example to `[B, M, dc]` (`M = num_chunks_per_doc`) then flattened to
`[B·M, dc]` for the embedding model.

## Field reference
| Key | Shape | Meaning |
|-----|-------|---------|
| `batch` / `inputs` | `[B,T]` | question+answer token ids |
| `docs` | `[B·M, dc]` | supporting-doc chunks (pos then neg), padded |
| `batch_mask` | `[B,T]` | 1 = real token, 0 = pad (attention) |
| `docs_mask` | `[B·M, dc]` | per-doc-token validity → becomes the bank `mem_mask` |
| `loss_mask` | `[B,T]` | tokens CE is computed over (`mask_prefix` zeros the prompt) |
| `pos_doc_mask` | `[B,M]` | 1 = positive doc chunk, 0 = neg/pad → drives `pos_slot_indices` |
| `pos_doc_ids` | `[B,M]` | corpus doc IDs (−1 if untagged) → `doc_access_acc` |
| `ce_enable` | `[B]` | per-row CE gate; 0 = retrieval-only (feeds `doc_access` not CE) |
| `teacher_*` / `*_distill_mask` | — | distill only; align teacher/student answer spans |

## How the trainer consumes it
`process_train_pairs(tokens, masks)` (`utils.py`) → `inputs, targets, input_masks, loss_masks,
ce_enable` (next-token targets + the masks the step needs). `forward(inputs, w,
pad_mask=input_masks, collect_aux=…)`; `docs_mask`/`pos_doc_mask`/`pos_doc_ids` flow into the
memory layer and losses. See [../training/training-loop.md](../training/training-loop.md) and
[../architecture/qwen3-mem-embed.md](../architecture/qwen3-mem-embed.md).
