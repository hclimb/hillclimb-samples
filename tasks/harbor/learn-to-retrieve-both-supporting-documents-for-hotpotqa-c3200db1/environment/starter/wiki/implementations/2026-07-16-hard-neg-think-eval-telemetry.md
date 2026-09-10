# Training telemetry + a 512-doc judged eval suite on one wandb run (hard-neg think)

*2026-07-16*

> **Superseded in part** by [2026-07-16-wandb-id-per-run-dir.md](2026-07-16-wandb-id-per-run-dir.md):
> the wandb id here is derived from `run_name`, which is **not unique** — every launch of a name
> collided on one id, and the eval box's name-scoped GCS scan would have evaluated an unrelated
> April run's checkpoints into the new run's curve. Identity is now the run-dir. The telemetry,
> eval-suite and corpus-`mem_pos_weight_mass` parts of this note still stand.

## Motivation

Set up the hard-neg (think) SFT4B run (`scripts/embed/train_hard_neg_think.sh`) so that over the
course of training we see (a) the full read-channel telemetry, and (b) a judged eval suite —
MS MARCO / HotpotQA / MuSiQue (LLM-judge accuracy, lexical grounding, `mem_pos_weight_mass`) and
ScienceQA (`mem_pos_weight_mass`, NLL) at 512-doc corpus, n=128 — with everything on **one clean
wandb run**.

Three things blocked the obvious "point `trainer.evals` at a new eval_set" approach:

1. **Judge/grounding metrics cannot be produced in-loop.** `run_metrics()` (the judge +
   `lexical_grounding`) is only called by `evals/shared.py::run_metrics_pipeline`, in `eval.py`'s
   parent process after the JAX worker exits and frees the TPU for vLLM. `trainer.py::_run_evals`
   keeps only `evaluator.evaluate()`'s `inference_metrics` and never calls `run_metrics` — so a
   task's `metrics:` block is **inert during training**, silently and without error.
2. **`mem_pos_weight_mass` did not exist for corpus evals.** `gen_large_mem.py` called
   `collect_eval_telemetry(aux_gen, answer_mask, None)` — `input_mask=None`, which short-circuits
   the positive-slot join in `mem_telemetry.py`, leaving only entropy/slots/top1. Nor did
   `NLLEvaluator` compute it (nll + `doc_access_acc` only).
3. **Eval metrics had no path into the training run.** `train.py` called `wandb.init` with no
   explicit `id`, so a training run's id is random — nothing for another process to attach to.
   `ground_eval_box.py` worked around this by logging to a *companion* `<run>_eval` run.

## Options considered

| Decision | Options | Chosen |
|---|---|---|
| Where evals run | (a) in-loop; (b) dedicated eval box on checkpoints | **(b)** — (a) cannot produce the judge metrics at all, and would fight the judge's vLLM for the TPU |
| Corpus `mem_pos_weight_mass` | (a) JAX, all layers, via `collect_eval_telemetry`; (b) numpy from `pos_sets` | **(b)** — (a) needs `.at[].get(out_sharding=…)` gathers that have never run on the corpus path (its telemetry call short-circuits before them) under that path's `set_global_mesh`; (b) mirrors the proven `_numpy_doc_access_acc` convention already used there for the same join |
| One wandb run | (a) companion run + post-hoc merge; (b) shared-mode secondary writer; (c) companion only | **(b)** — verified wandb 0.28.0 on the box exposes `mode="shared"`, `x_primary`, `x_label`, `x_update_finish_state` |
| Deterministic wandb id | (a) always on; (b) opt-in config key | **(b)** — (a) would silently make every re-run of any config append to an earlier run |

## What was built

**Corpus `mem_pos_weight_mass`** (`evals/gen_large_mem.py`). `_numpy_pos_weight_mass` +
`_per_example` + layer-mean wrappers: same ratio as the train-time metric —
`Σ(w on positive slots) / Σ(w on valid slots)` over the answer span — but joining positives via
the corpus `pos_sets` (flat bank indices) instead of the batch `docs_mask`/`pos_doc_mask` grid,
which has no meaning for a shared bank (`_positive_slot_stats` derives the doc id as
`index // doc_len`). Indices+probs are sliced to the generated span **on-device before the
allgather**, so only the answer window crosses the host boundary. Written per-layer (today
`mem_layers: [14]`, one layer) and surfaced in `inference_metrics`, `file_metrics`, and
per-sample as `pos_slot_weight_mass`.

**NLL telemetry** (`evals/nll.py`). `collect_eval_telemetry(aux_data, loss_masks, input_masks)` —
the NLL path has real `input_masks` from `process_train_pairs` (`docs_mask` + `pos_doc_mask`), so
the existing join works unchanged. Gated on the aux config enabling `mem_pos_weight_mass`, which
`eval_worker` seeds from the train cfg, so a telemetry-trained checkpoint gets it free.

**Deterministic wandb id** (`utils.py`, `train.py`). `wandb_run_id_from_name(run_name)` (shared by
train + box) and `resolve_wandb_run_id(cfg)` behind `trainer.wandb_run_id` (`null` → legacy,
`"auto"` → derived, else literal). When set, `train.py` initialises as the shared-mode **primary**
writer.

**The eval box** (`scripts/misc/hard_neg_eval_box.py` + `scripts/embed/hard_neg_eval_box_run.sh`).
Adapted from `ground_eval_box.py`; differences: logs **every** scalar from the result JSON
(handling both `{"metrics":…}` from the generation evaluators and `{"stats":…}` from NLL) rather
than a `_find` heuristic that only pulled judge accuracy; attaches to the **training** run as a
shared-mode secondary (`x_primary=False`, `x_label="eval"`, `x_update_finish_state=False` — without
that last flag its per-point `run.finish()` would mark the live training run finished); keeps the
`train_step` custom x-axis (logging `step=step` drops every task after the first at a given step).

**Configs.** `configs/eval/tasks/gen_large_mem_{msmarco,hotpotqa,musique}_c512.yaml`,
`nll_science_qa_hard_neg_think_n128.yaml`, `configs/eval_set/hard_neg_think_c512.yaml`,
`configs/trainer/staged_telemetry.yaml` (= `staged` + the weight-0 block).

Two notes on the eval configs:
- Corpus sizing uses `max_docs: null` + `target_docs: 512`, **not** `max_docs: 512`.
  `data/documents.py:54` stops the doc generator at `max_docs`, which would cap the
  `inject_query_gold` scan and silently drop every gold past the cap. (`embed_corpus_1000.yaml`
  sets `max_docs: 1000` with `inject_query_gold: true` and so has this problem — left alone here,
  flagged for a separate fix.)
- **The 512-doc corpus has no analog for ScienceQA's NLL task.** `NLLEvaluator` builds memory from
  each batch's own pos+hard-neg docs; only `generation_large_mem` builds a corpus. So its
  `mem_pos_weight_mass` is the in-batch read (directly comparable to `train/mem_pos_weight_mass`),
  while the corpus tasks' is the 512-distractor read. Both are reported; they are different numbers.

## Test record

`tests/test_corpus_pos_weight_mass.py` pins the numpy math against hand-computed values (base
ratio, answer-span masking, invalid-slot exclusion from both numerator and denominator, no-positives
→ 0.0, all-masked → None, saturation at 1.0, Σnum/Σden batch pooling vs per-example ratios,
layer-mean incl. skipping an undefined layer, single-layer identity).
`scripts/embed/validate_hard_neg_setup.sh` additionally asserts config composition (telemetry block
present and all weight 0, `doc_access_loss` weight unchanged, n=128 on all four tasks,
`max_docs is None` + `target_docs == 512`, rotation window ≥ one eval cycle) and wandb-id
determinism. Both run on `rohun-v6e-8-2` (v6e-8, europe-west4-a):

```
TPU_NAME=rohun-v6e-8-2 RUN_SCRIPT_PATH=scripts/embed/validate_hard_neg_setup.sh \
  bash scripts/infrastructure/multi-vm-tpu-run.sh

=============== 1. corpus mem_pos_weight_mass unit test ===============
PASS  base / loss_mask excludes position / invalid slot excluded from denominator
PASS  no positives -> 0.0        PASS  all masked -> None      PASS  all-positive -> 1.0
PASS  pooled over batch: got=0.32499999813735486 want=0.325
PASS  per-example [0] / [1] / undefined -> None
PASS  layer-mean / layer-mean per-example / layer-mean skips undefined layer
PASS  single layer == base
ALL PASS

=============== 2. training config composes ===============
  mem_pos_weight_mass: {'enabled': True, 'weight': 0.0}   (+12 more mem_* all weight 0.0)
  trainer.evals: {} (in-loop eval OFF)
  wandb_run_id: auto   run_name: qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16
  steps: 100000  checkpoint_interval: 2000  max_to_keep: 16
  stages: 4
  rotation window: 32000 steps (~4.4h at 490ms/step)
  OK

=============== 3. eval_set composes ===============
  gen_large_mem_msmarco:  {'type': 'generation_large_mem', 'n': 128,
                           'metrics': ['lexical_grounding', 'llm_judge_accuracy'],
                           'max_docs': None, 'target_docs': 512}
  gen_large_mem_hotpotqa: {... identical ...}
  gen_large_mem_musique:  {... identical ...}
  nll_science_qa:         {'type': 'nll', 'n': 128, ...}
  OK

=============== 4. deterministic wandb id ===============
  wandb id : qa_hard_neg_think_sft4b_topk64_seq512_ch-5bd5d8f9
  OK
=============== ALL CHECKS PASSED ===============
```

The wandb id matches what a live launch printed (`W&B shared-mode primary writer, run id:
qa_hard_neg_think_sft4b_topk64_seq512_ch-5bd5d8f9`), confirming the box derives the same id.

Two defects surfaced only by running on a box, both in the test scaffolding rather than the
shipped code, and both worth remembering:
- `tests/test_corpus_pos_weight_mass.py` imported `evals.*` with no `sys.path` insert. `pyproject`
  packages only `["models", "data"]`, so `data` imports resolve and `evals` does not — other tests
  carry the insert; this one didn't.
- The test asserted `tol=1e-9` on **float32** inputs (`mem_top_k_probs` is f32 on device, so 0.7 is
  0.699999988…). Every value was right to ~8 s.f.; the tolerance was testing float32's
  representation, not the metric. Now 1e-6.
- `validate_hard_neg_setup.sh` piped `train.py --cfg job` straight to a file, but `train.py` calls
  `setup_gcs_credentials()` at module level, printing `GCloud credentials set!` ahead of the YAML
  and corrupting it. Now filtered to start at the first top-level key.

## Known gaps

- **The eval box is a fork, not an abstraction.** `hard_neg_eval_box.py` is ~80% copy-paste from
  `ground_eval_box.py` (itself from `sim_eval_box.py`). Deliberately left as a fork to avoid
  refactoring a box that other live runs depend on mid-setup — but it should be consolidated into
  one module before a third fork. Plan + config surface:
  [evaluation/eval-boxes.md](../evaluation/eval-boxes.md) ("TODO — abstract the box").
- **`trainer.py:455`**: the `val/*` wandb log is gated on `not cfg.trainer.get("evals")`, so when
  `trainer.evals` is set, `generation_embed`/`gen_large_mem` `inference_metrics` reach neither wandb
  nor console (they self-log only an artifact; `NLLEvaluator` self-logs scalars, so NLL survives).
  Not hit by this run (in-loop eval is off) — left as-is rather than changing shared trainer
  behaviour mid-setup.
- **`embed_corpus_1000.yaml`** has the `max_docs` gold-scan truncation described above.

Reference pages updated: [evaluation/eval-boxes.md](../evaluation/eval-boxes.md) (new),
[evaluation/metrics.md](../evaluation/metrics.md),
[evaluation/eval-configs.md](../evaluation/eval-configs.md),
[training/auxiliary-losses.md](../training/auxiliary-losses.md),
[training/trainer-configs.md](../training/trainer-configs.md).
