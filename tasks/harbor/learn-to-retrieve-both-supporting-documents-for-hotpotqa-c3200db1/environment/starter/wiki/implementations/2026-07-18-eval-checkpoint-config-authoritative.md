# Eval silently rebuilt the model from a default config — checkpoint config is now authoritative

**Date:** 2026-07-18
**Status:** fix landed; re-eval of an affected 4-layer checkpoint in progress
**Files:** `evals/shared.py`, `eval.py`, `rag_eval.py`, `evals/eval_worker.py`,
`tests/test_checkpoint_model_authoritative.py`, `scripts/misc/eval_ground_4layer.sh`

## Answer first

When `eval.py`/`rag_eval.py` load a checkpoint, they now treat the checkpoint's **saved** model
config as authoritative for architecture, and only layer *explicit* `model.*` command-line
overrides on top. Before this, the eval's own default model config silently overrode the trained
architecture — most damagingly, a checkpoint trained with `mem_layers=[9,14,20,27]` was rebuilt as
`mem_layers=[14]` and evaluated as a **1-memory-layer model**, with the other three layers' trained
weights silently discarded on load.

## The bug

`configs/eval.yaml` composes a full default model (`model: qwen3_mem_embed`), so `cfg.model` is
*never* empty. `evals/eval_worker.py` then did:

```python
effective_model_cfg = OmegaConf.merge(train_cfg.model, cfg.model) if cfg.get("model") else train_cfg.model
```

`OmegaConf.merge(A, B)` lets **B win**, so the eval default (`cfg.model`) overrode the checkpoint's
saved config (`train_cfg.model`) on every key where they differed. The `if cfg.get("model")` guard
was presumably meant to mean "did the user pass model overrides?" — but it can't, because the
composed default makes `cfg.model` truthy on every run.

Two severities, depending on the key:

- **Loud (shape mismatch → load error).** If `mem_num_heads` / `mem_k_dim` differed, layer-14's
  weight shapes wouldn't match the checkpoint and the restore errored. `qwen3_mem_embed_16q4kv.yaml`
  exists *solely* as a workaround for this — its header says to pass `model=qwen3_mem_embed_16q4kv`
  "so the default qwen3_mem_embed config … does NOT clobber this architecture via the eval's
  OmegaConf.merge." So the failure mode was already known; the workaround was per-run vigilance.
- **Silent (subset of keys → partial restore).** If only `mem_layers` differed and the per-layer
  dims matched, the built 1-layer model's weight tree is a strict *subset* of the checkpoint. With
  `PyTreeRestore(item={"weights": model.weights}, partial_restore=True)`, orbax restores only the
  requested keys, so layers 9/20/27's memory weights are ignored with **no error**. The eval ran a
  crippled model and reported plausible-looking numbers.

## Confirmed impact

`gs://memory-layers-training/ground_s1_zeroinit_4layer-2026-07-04-09-42-55` — its saved
`.hydra/config.yaml` has `mem_layers: [9,14,20,27]`, `mem_num_heads: 4`, `mem_k_dim: 1024`,
`mem_top_k: 128`. Every non-layer key matches the eval default, so it hit the **silent** path:
`ground_eval_box.py` (which passes no `model=` override) evaluated it as `mem_layers=[14]`, dropping
3 of its 4 memory layers. Any multi-layer grounding run evaluated through that box is suspect. A
re-eval through the fixed path is running (see Test record).

The same mechanism is why the current hard-neg run's evals ran at `mem_top_k=128` instead of the
trained 64: the eval default's `mem_top_k: 128` clobbered the checkpoint's 64. That surfaced as
`mem_effective_slots=115` (a participation ratio `1/Σp²`, bounded by K) at an eval where K was
"supposed" to be 64 — impossible unless K was actually 128.

## Options considered

| Option | Verdict |
|---|---|
| Reverse the merge: `merge(cfg.model, train_cfg.model)` | rejected — checkpoint wins, but *all* explicit CLI `model.*` overrides are then silently dropped: same footgun, reversed. |
| Drop the default `model` from `configs/eval.yaml` | rejected — no-checkpoint eval (fresh/random model) genuinely needs a default model to build anything. |
| Per-run workaround: always pass a matching `model=<cfg>` | rejected — this is the status quo (the 16q4kv pattern); relies on the operator remembering, which is exactly what failed. |
| **Checkpoint config authoritative + explicit overrides layered on** | **chosen** — correct by construction, and preserves the deliberate-override use case. |

## The fix

New `apply_checkpoint_model_cfg(cfg, train_cfg, cli_overrides)` in `evals/shared.py`, called by
`eval.py` and `rag_eval.py` right after `resolve_train_cfg` (both have `HydraConfig.get().overrides`).
Precedence when a checkpoint is present:

1. Explicit **`model=<group>`** on the command line → the caller deliberately swapped the whole
   model; `cfg.model` is respected unchanged (the 16q4kv workaround still works, now redundant).
2. Explicit **`model.<path>=<val>`** (and `+`/`~` variants) → a knob layered on the trained arch.
3. Everything else → comes from the checkpoint's saved config, **not** the eval default.

No checkpoint → `cfg.model` is the only source of truth, returned unchanged. It also prints a loud
line for every architecture key where the eval default *would* have differed
(`[eval] checkpoint-authoritative: memory.mem_layers=[9,14,20,27] from checkpoint (eval default [14]
would have clobbered it)`), so a train/eval mismatch can never again be silent.

`eval_worker.py`'s merge is left in place as a safety net (it is now a no-op for these paths, since
`cfg.model` is already authoritative) with a comment explaining so.

### Second bug found while re-evaluating: eval.py read GCS before setting credentials

The re-eval initially failed with `OSError: Forbidden` on the checkpoint's `.hydra/config.yaml`.
Root cause: `eval.py`'s `main` called `resolve_train_cfg` (which reads `gs://…/config.yaml`)
**before** any credential setup, so the read fell back to the box's compute service account
(`…-compute@developer.gserviceaccount.com`), which lacks access to `memory-layers-training`.
`rag_eval.py` already called `setup_gcs_credentials()` first; `eval.py` did not. Fixed by calling
`setup_gcs_credentials()` at the top of `eval.py`'s `main`. (The "intermittent" appearance was
env-dependent, not random: runs where `GOOGLE_APPLICATION_CREDENTIALS` was already exported worked;
the standalone re-eval script didn't set it, so every read 403'd.) Confirmed on box: the user's
`adc.json` exists and reads the object fine once `GOOGLE_APPLICATION_CREDENTIALS` points at it.

### Behavior change to be aware of

Evals of already-trained checkpoints now use the checkpoint's `mem_top_k` (and every other model
key), not the eval default. For the current hard-neg run that means future evals run at
`mem_top_k=64`, not 128 — correct, but not comparable to the historical 128-based numbers.

## Test record

`tests/test_checkpoint_model_authoritative.py` (standalone, no pytest) covers: no-override →
checkpoint's 4 layers win; `model.memory.mem_top_k=64` → arch from checkpoint + that field
overridden; `model=<group>` → `cfg.model` respected; no-checkpoint → unchanged.

```
uv run python tests/test_checkpoint_model_authoritative.py
  [PASS] no-override: mem_layers == [9,14,20,27] (checkpoint authoritative)
  [PASS] field-override: mem_layers still [9,14,20,27]
  [PASS] field-override: mem_top_k == 64 (explicit override applied)
  [PASS] group-swap: cfg.model left as-is ([14], caller's explicit choice)
  [PASS] no-checkpoint: cfg.model unchanged ([14])
RESULT: ALL PASS
```

On-box proof the fixed path builds all 4 layers (from the eval log):

```
[eval] checkpoint-authoritative: memory.mem_layers=[9, 14, 20, 27] from checkpoint (eval default [14] would have clobbered it)
```

### Corrected 4-layer eval — `ground_s1_zeroinit_4layer-2026-07-04-09-42-55`, step 38000, n=128

Run with `scripts/misc/eval_ground_4layer.sh` on `rohun-v6e-8-2` (v6e-8). Result JSONs:
`gs://memory-layers-training/ground_4layer_refix/ground_s1_zeroinit_4layer-2026-07-04-09-42-55/step38000/{msmarco,hotpotqa,musique}_c512.json`

| task @c512 | llm_judge_accuracy | lexical_grounding | mem_pos_weight_mass | doc_hit_rate |
|---|---|---|---|---|
| msmarco  | 0.664 | 0.494 | 0.445 | 1.000 |
| hotpotqa | 0.328 | 0.413 | 0.159 | 1.000 |
| musique  | 0.117 | 0.217 | 0.112 | 1.000 |

These reflect the true 4-memory-layer model. Any historical ground_eval number for this run
(evaluated as 1 layer) should be treated as invalid and re-run through the fixed path if needed.
A direct 1-layer-vs-4-layer A/B on the same task was not run (the historical box used different
corpus configs), so the table is the corrected baseline, not a delta.
