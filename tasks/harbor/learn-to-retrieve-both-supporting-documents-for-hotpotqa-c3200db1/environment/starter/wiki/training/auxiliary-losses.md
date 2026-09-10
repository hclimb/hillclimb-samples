# Auxiliary Losses

Training signal + telemetry beyond CE. Registry pattern in `losses/`; consumed by the trainer
via `compute_aux_losses`.

## Registry (`losses/registry.py`)
- `@register_aux_loss("name")` on `fn(aux_data, mask, input_mask, inputs, **kwargs)`.
- `compute_aux_losses(aux_data, mask, input_mask, inputs, cfg)` → `{total, losses,
  weighted_losses}`. Runs only **enabled** losses; each reads what it needs from `aux_data`
  (the model's `ModelOutput.aux`).
- A loss returns a scalar, or a **dict** of named scalars (per-layer diagnostics → logged as
  `train/<name>/<subkey>`). NaNs are zeroed per-entry.
- **Weight-0 = telemetry:** the value is `stop_gradient`-ed before weighting, so it contributes
  exactly 0 to the gradient (avoids `0·NaN` poisoning the whole step). Register a new loss →
  add file + `@register_aux_loss` + import in `losses/__init__.py` + enable in the trainer config.

## Config shape
```yaml
aux_losses:
  doc_access_loss: { enabled: true, weight: 0.1, temperature: 1.0 }
```
Extra keys (beyond `enabled`/`weight`) pass through as kwargs. Per-stage `aux_losses` merge
onto this base. Total: `ce_weight·CE + Σ weight·aux`.

## Training-signal losses
| Loss | What it supervises |
|------|--------------------|
| `doc_access_loss` | Contrastive (global in-batch negatives): a doc token should retrieve **its own** doc's memory slots. Uses the full `mem_scores` `[B,T,N,M]` grid (M = the WHOLE batch's flat bank, `B·m_per_query`). `temperature`. **Needs a mode that builds that cross-batch grid** (`mem_lookup`/`mem_lookup_chunked`) — under `mem_batched_isolation` (`mem_lookup_batched`), that grid is structurally never built, so this loss silently no-ops (`aux_data.get("mem_scores")` absent → returns `0.0`). Use `doc_access_per_query_loss` instead for that mode. |
| `doc_access_top_k_loss` | Same objective over only the **K** two-pass slots (gradients exact). Uses `pos_slot_indices` so a positive is always in the pool; if a positive isn't in top-k, that sample contributes 0. |
| `doc_access_per_query_loss` | `doc_access_loss`'s exact objective (`log_z − log_pos`) restricted to a query's own `m_per_query` slots — for `mem_batched_isolation`, where cross-query contamination is structurally impossible so the `block_eye` cross-batch positive-mask `doc_access_loss` builds isn't needed at all. Consumes `mem_scores` populated by `mem_lookup_batched` when `model.memory.mem_collect_full_scores=true` — same `[B,T,N,·]` shape family as `doc_access_loss`, just `m_per_query` instead of `B·m_per_query` in the last axis (cheap here; that size *is* the whole point of `mem_batched_isolation`). `temperature`. See `configs/trainer/staged_batched_isolation.yaml` and [2026-08-02-hard-neg-full-efficient-retrieval.md](../implementations/2026-08-02-hard-neg-full-efficient-retrieval.md) (found after an entire overnight run trained on `Loss: 0.0000` — `doc_access_loss` was the only nonzero-weight loss in early stages and was silently no-op-ing). |
| `doc_access_consistency` | Consistency of access patterns across nearby tokens. |
| `mem_uniform_kl` | KL(memory-usage ‖ uniform) — spread usage across slots, avoid collapse. |
| `distillation_loss` | KL(student ‖ teacher) logits for `qwen3_distill` (`top_k`, `temperature`). |
| `msa_route_loss` / `msa_route_acc` | MSA router selection loss / accuracy (`temperature`). |

## Metrics (run at weight 0)
- `doc_access_acc` — fraction of doc-token positions that retrieved the right document.
- **`mem_telemetry.py` family** — read-channel diagnostics, many per-layer: `mem_write_norm`,
  `mem_write_ratio`, `mem_topk_entropy`, `mem_effective_slots`, `mem_top1_weight`,
  `mem_o_proj_norm`, `mem_boundary_straddle`, `mem_pos_weight_mass`, `mem_hit_rate`,
  `mem_head_query_cos`, `mem_cross_layer_cos`, `mem_kv_cos`, `mem_value_anisotropy`. Most are
  computed in `models/memory.py::_memory_telemetry` / `qwen3_mem_embed` and surfaced here.

Defaults live in `configs/trainer/standard.yaml`. Configs carrying the **full weight-0 block**:
`standard_ground` (= `standard` + telemetry), `staged_telemetry` (= `staged` + telemetry), and
`staged_ground` (a different *recipe* — 2-stage, frozen main — that also enables it).
See [trainer-configs.md](trainer-configs.md).

## The same telemetry at eval time
`mem_telemetry.py::collect_eval_telemetry(aux_data, loss_mask, input_mask)` returns the flat
layer-mean dict for the evaluators. Two gotchas:
- **It needs `input_mask`** for the positive-slot join (`mem_pos_weight_mass`, `mem_hit_rate`);
  called with `input_mask=None` it returns only `mem_topk_entropy` / `mem_effective_slots` /
  `mem_top1_weight`.
- **A corpus bank has no `pos_doc_mask`**, so `gen_large_mem` can't use that join at all: its
  positives are flat bank indices (`pos_sets`), not a per-batch doc grid. It computes
  `mem_pos_weight_mass` from `pos_sets` in numpy instead
  (`evals/gen_large_mem.py::_numpy_pos_weight_mass`), same ratio, same layer-mean.

Which evaluator's number means what — and why the corpus and in-batch reads aren't comparable —
is in [../evaluation/metrics.md](../evaluation/metrics.md).
