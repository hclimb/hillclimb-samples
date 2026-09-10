# Wire up `vm2825/hotpotqa-hard-neg-reasoning-embedding-modified-SFT-4B` as a standalone finetune

**Date:** 2026-08-22 · **Author:** rohunagrawal (with Claude) · **Status:** done — training completed cleanly (10000/10000 steps); checkpoint sweep shows the finetune regresses the hotpotqa hybrid eval, see [experiment write-up](../experiments/2026-08-23-hotpotqa-hard-neg-reasoning-finetune-checkpoint-sweep.md) · **Commit:** _TBD_

**Run:** `hotpotqa_hard_neg_reasoning_finetune_topk64_seq512_chunks16_bs16-2026-08-22-19-08-24` on
`rohun-v6e-8-0` (`us-east1-d`, `memorylayers` project, TRC spot). Checkpoints:
`gs://memory-layers-training/hotpotqa_hard_neg_reasoning_finetune_topk64_seq512_chunks16_bs16-2026-08-22-19-08-24/qwen3_mem_embed/`.
wandb: `johnzhang2366-columbia-university/memory-layers`, run id auto-derived from the run dir.

## What changed

Added a new dataset source + standalone dataset config wiring
`vm2825/hotpotqa-hard-neg-reasoning-embedding-modified-SFT-4B-parts-0-1` (78,755 HotpotQA rows,
1 `pos_doc` + 3 `neg_docs` each) into the existing `QADataset` machinery, plus a launch script that
warm-starts it from the hard-neg-think 4B checkpoint
(`qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45`,
step 100000 —
[wandb run](https://wandb.ai/johnzhang2366-columbia-university/memory-layers/runs/qa_hard_neg_think_sft4b_topk64_seq512_ch-2026-08-09-05-49-45)).
No loader/normalizer code changes were needed — the generic normalizer already handles this
dataset's schema.

Also added `learning_rates: {mem: 1e-4, embed: 1e-5, main: 1e-5}` to the **shared**
`configs/trainer/midtraining.yaml` (mirroring `staged.yaml`'s own group rates), at the user's
explicit request to edit that file directly rather than override per-script — see the ⚠️ in
"Options weighed" for the blast radius on other scripts that use the `midtraining*` family.

New files:
- `configs/dataset/sources/hotpotqa_hard_neg_reasoning_modified.yaml`
- `configs/dataset/hotpotqa_hard_neg_reasoning_modified_finetune.yaml`
- `scripts/misc/download_hotpotqa_hard_neg_reasoning_data.sh`
- `scripts/embed/train_hotpotqa_hard_neg_reasoning_finetune.sh`

Modified:
- `configs/trainer/midtraining.yaml` — added the `learning_rates` block above.

## Motivation & context

Requested follow-up to the `qa_hard_neg_think_sft4b` run: finetune that checkpoint further on a
newer, HotpotQA-specific hard-neg reasoning SFT set, with the explicit instruction to set
`neg_score_threshold: 0.0` "so the hard negs actually come in". Prior state:
[data/dataset-configs](../data/dataset-configs.md), [infrastructure/checkpointing](../infrastructure/checkpointing.md).

**Verified the schema and the threshold concern before writing the config**, rather than assuming
this dataset follows the sibling `vm2825/*` convention exactly: fetched the HF dataset-server
`/rows` endpoint directly (`https://datasets-server.huggingface.co/rows?dataset=vm2825%2Fhotpotqa-hard-neg-reasoning-embedding-modified-SFT-4B-parts-0-1&config=default&split=train&offset=0&length=2`).
Finding: `neg_scores` is the literal constant string `"0.0<doc_seperator>0.0<doc_seperator>0.0"`
on every row (also visible in the HF auto-viewer as `neg_scores: stringclasses 1 value`, i.e.
exactly one distinct value across all 78,755 rows) — the same "cheap negs, `neg_scores = 0.0`
placeholder" convention documented in `datagen/create_hard_neg_sft_dataset.py`. Given
`make_normalizer`'s filter keeps `score <= threshold` (`data/utils.py:153`), any
`neg_score_threshold >= 0.0` already passes every neg in this specific dataset — so the fix isn't
load-bearing for *this* data as shipped, but it is the config's honest statement of intent (there
is no real score here to threshold on) instead of silently inheriting the siblings' `0.95`, which
is tuned for a real 0-1 similarity score this dataset doesn't have.

## Options weighed

| Decision | Chosen | Rejected, and why |
|---|---|---|
| Fold into the main mix vs. standalone | Standalone dataset config (`hotpotqa_hard_neg_reasoning_modified_finetune.yaml`), same pattern as `musique_sft.yaml` / `multihop_qa_sft_midtraining.yaml` | Adding it as a `qa_hard_neg_think_sft4b.yaml` source — the request is specifically to *finetune the existing checkpoint on this dataset*, not retrain the original mix with an added source. |
| Trainer recipe | `midtraining_full_telemetry` (single cosine stage, unfreezes `mem_*`/`embed_model`/**all** of `main_model`) | `midtraining_telemetry` (layers 13-15 only) — the choice `train_musique_sft_midtrain.sh` / `train_multihop_sft_midtrain.sh` make, but the checkpoint's own lineage (`staged.yaml`) reaches its stage 3 with the *whole* `main_model` trainable, not a 3-layer neighbourhood, so continuing that trainable set as one stage matches the checkpoint's own trajectory more closely than narrowing it. `staged*` itself — its stage 0 re-freezes `main_model` and zeroes `ce_weight` for a warmup schedule meant for a *fresh* model; applying it to an already-converged checkpoint would undo that. The newer `staged_batched_isolation_docaccess_warmstart` (used by the `multihop_lrmasked_warmstart*` arms off the same checkpoint) exists specifically for `multihop_hard_neg_full`'s ~200-hard-neg/query scale (`per_query_isolation`/`mem_batched_isolation`) — this dataset has a fixed 3 negs/row, nowhere near that scale, so the plain midtraining recipe is the right fit, not the isolation machinery. |
| Learning rate | Per-group `learning_rates: {mem: 1e-4, embed: 1e-5, main: 1e-5}` (`utils.py::setup_optimizer_for_stage`'s opt-in per-group path), mirroring `staged.yaml`'s own group rates — added directly to shared `configs/trainer/midtraining.yaml`, at the user's explicit request, rather than as a per-script override | A single uniform rate (the musique/multihop midtrain scripts' `learning_rate=1e-4`) — fine when only layers 13-15 of `main_model` are trainable, but wrong once the *whole* `main_model` unfreezes: `staged.yaml`/`staged_sim` never run `main_model` above 1e-5 ("to protect the 4B"), so a uniform 1e-4 would apply the aggressive mem-only rate to the full 4B; a uniform 1e-5 would under-train `mem_*` relative to how this checkpoint's own lineage was trained. A per-script `+trainer.learning_rates.*` override on just this script — considered, and the mechanically safer choice (no effect on other callers), but not what was asked for. |
| Where the per-group rates live | Edited shared `configs/trainer/midtraining.yaml` directly, per explicit instruction | A dedicated new trainer config (e.g. `midtraining_full_telemetry_staged_lr.yaml`) layering `learning_rates` on top without touching the shared base — mechanically safer, but not what was asked for. **Consequence, not reconciled here:** `midtraining.yaml` is the base of `midtraining_telemetry`, `midtraining_frozen_telemetry`, and `midtraining_full_telemetry` alike, so this also changes the effective LR for `train_musique_sft_midtrain.sh`, `train_multihop_sft_midtrain.sh`, and `train_musique_ground4layer_midtrain.sh` on any future re-run — all three currently pass a single uniform `trainer.learning_rate=1e-4`, which becomes inert once `learning_rates` exists in the composed config. `train_musique_ground4layer_midtrain.sh`'s header specifically frames its LR choice as isolating one variable in a comparison against a sibling run; that comparison's validity on a future re-run is now unverified. |
| Dataset shape (`seq_len`/`doc_chunk_seq_len`/`num_chunks_per_doc`/`batch_size`) | Match `qa_hard_neg_think_sft4b.yaml` exactly (512/256/16/16) | Re-tuning for this dataset's smaller per-row doc count (4 docs vs. a mixed source's larger pool) — kept identical instead so the memory bank shape exactly matches the checkpoint being warm-started, no resizing risk. |
| `neg_score_threshold` | `0.0`, set explicitly | Leaving it unset (`null`, no filtering) — functionally identical today (all scores are 0.0), but `0.0` documents that the threshold was deliberately considered for this source, per the explicit instruction, rather than looking like an oversight. |
| Epoch budget | `steps=10000` × `batch_size=16` = 160,000 samples ≈ 2.03 epochs over 78,755 rows | Matching `musique_sft_midtrain`'s ~3 epochs (13k rows, comparably small) would be 4,922 steps/epoch × 3 ≈ 14,766 steps — considered, but 2 epochs was chosen as the more conservative default against overfitting on reasoning/CoT text; easy to raise via `trainer.steps=`. |

## How it was built & integrated

`configs/dataset/sources/hotpotqa_hard_neg_reasoning_modified.yaml` — `field_map`
(`question→query`, `neg_doc→neg_docs`, `answer→generated_answer`), `think_field: think`,
`doc_separator: "<doc_seperator>"` (matches the misspelling `datagen/persistent_tpu/
generate_cot_sft.py` actually writes for this dataset family — verified against
`data/utils.py::make_normalizer`, not assumed), `neg_score_threshold: 0.0`, `min_neg_docs: 2`.
Uses the generic (non-`docid`, non-`hotpotqa`) normalizer path in `data/qa.py` — no code changes.

`configs/dataset/hotpotqa_hard_neg_reasoning_modified_finetune.yaml` — standalone `sources:`
composition (one entry), `seq_len`/`doc_chunk_seq_len`/`num_chunks_per_doc`/`batch_size` =
512/256/16/16 to match `qa_hard_neg_think_sft4b.yaml`.

`scripts/misc/download_hotpotqa_hard_neg_reasoning_data.sh` — `data/download_hf_data.py
--dataset hotpotqa_hard_neg_reasoning_modified_finetune`, modeled on
`download_musique_sft_data.sh` (single-source, not epoch-balanced across sources).

`configs/trainer/midtraining.yaml` — added `learning_rates: {mem: 1e-4, embed: 1e-5, main: 1e-5}`
at the base-config level (see table above), so every `midtraining*` recipe now takes the
per-group path in `utils.py::setup_optimizer_for_stage` instead of the single-LR one. The base
cosine schedule is built off the `mem` group's peak; `embed`/`main` scale by a fixed ratio to it.
`trainer.learning_rate` (still `standard.yaml`'s default, 1e-4) is now inert everywhere this
composes — it's read only "for schedule construction sanity" per the existing code comment.

`scripts/embed/train_hotpotqa_hard_neg_reasoning_finetune.sh` — `trainer.resume_from=".../qwen3_mem_embed/100000"`
(trailing step load-bearing: selects `load_checkpoint`'s weights-only warm-start branch, not a
full resume), `model.memory.mem_top_k=64` (matches checkpoint; config default is 128),
`trainer=midtraining_full_telemetry` (inherits the new `learning_rates` from `midtraining.yaml`
above — no override needed here), `+trainer.training_stages.0.warmup_frac=0.1` (midtraining
ships with none; a warm start has no optimizer state to inherit), `steps=10000`,
`checkpoint_interval=500`/`max_to_keep=20` (10,000-step window = whole run stays evaluable),
in-loop eval off (same rationale as every other script here — `trainer._run_evals` never calls
`run_metrics`, so a task's `metrics:` block is inert in-loop; score from a separate eval box).

**OOM risk, accepted rather than mitigated pre-emptively:** unfreezing the whole `main_model`
allocates Adam moments for every one of its params. Not pre-shrunk here because this is the exact
same trainable set at the exact same bank size (`16 × 16 × 256 = 65,536` slots/batch) that the
checkpoint's own stage 3 (`staged.yaml`) already trained at to reach step 100000 — if that fit on
the box that produced it, this should too. Fallback ladder if it doesn't (documented in the
script): `dataset.batch_size` or `dataset.num_chunks_per_doc` down, not the trainable set.

## Reference pages updated

- [data/dataset-configs](../data/dataset-configs.md) — new source-family table row +
  `hotpotqa_hard_neg_reasoning_modified` section (the constant-placeholder `neg_scores` finding).
- [training/trainer-configs](../training/trainer-configs.md) — `midtraining` row now notes the
  new default `learning_rates` block and that it makes `trainer.learning_rate=` overrides inert.

## Tests

Ran `scripts/misc/check_train_config.sh` on `rohun-v6e-8-0` (`us-east1-d`, `memorylayers`) via
`multi-vm-tpu-run.sh`:

```
CHECK_OVERRIDES="model=qwen3_mem_embed^model.main_model.model_id=Qwen/Qwen3-4B^model.memory.mem_top_k=64^dataset=hotpotqa_hard_neg_reasoning_modified_finetune^trainer=midtraining_full_telemetry^trainer.resume_from=gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45/qwen3_mem_embed/100000^trainer.steps=10000^+trainer.training_stages.0.warmup_frac=0.1^trainer.checkpoint_interval=500^trainer.max_to_keep=20^trainer.log_interval=10^eval_set@trainer.evals=none" \
DATASET_ROWS=78755 bash scripts/misc/check_train_config.sh
```

Confirmed: dataset shape (`seq_len 512 / doc_chunk 256 / num_chunks_per_doc 16 / batch_size 16`,
source `field_map`/`doc_separator`/`neg_threshold=0.0 min_neg=2` exactly as written), `mem_top_k
64`, resume mode `WARM START (weights only, step 0)` (not full resume), single stage with
`trainable=['.*mem_.*', '.*embed_model.*', '.*main_model.*']`, memory bank `16 × 16 × 256 =
65,536` slots/batch, `2.03` epochs over 78,755 rows. (`check_train_config.sh` doesn't print
`trainer.learning_rates` itself, only the single `learning_rate`/`weight_decay` scalars — the
per-group values were verified by reading `configs/trainer/midtraining.yaml`, not by this tool's
output; worth extending the tool if per-group rates need the same fast sanity check regularly.)

**Then launched for real** and watched the box-side log directly (`tail -f` over SSH, filtered
for step/loss/error markers) rather than trusting the launcher's exit code alone. First attempt
failed immediately: `huggingface_hub.errors.LocalEntryNotFoundError` at `models/qwen3.py::load`
— `rohun-v6e-8-0` is a fresh-since-last-preemption box with no HF model snapshot cached, and
`HF_HUB_OFFLINE=1` (required for the offline-parquet data path) also blocks `snapshot_download`
of the model weights, a gotcha already documented in `2026-07-18-musique-sft-dataset.md`'s
"Infrastructure" section for the same class of problem. Fixed with `MODELS_ONLY=1 bash
scripts/misc/precache_hf.sh` (pulls `Qwen/Qwen3-4B` + `Qwen/Qwen3-Embedding-0.6B` to
`~/weights/huggingface/`), then relaunched. Confirmed via the box log through step 10: loss
0.1784 / CE 0.1754 at step 1 (first-step JIT compile, ~4m06s), settling to a normal per-step
rate by step 10; the `[wmon step=10]` weight-monitor diagnostic shows `main_model.layers.9/17/26`
leaves actually changing (`n_changed` 10-100% depending on the leaf), confirming the full
`main_model` unfreeze is genuinely receiving gradient, not just `mem_*`/`embed_model`.

## Follow-ups & risks

- **Fresh/recreated TRC spot boxes need `MODELS_ONLY=1 bash scripts/misc/precache_hf.sh` before
  a `HF_HUB_OFFLINE=1` training script will load the model** — hit this live on `rohun-v6e-8-0`
  (see Tests). Worth a one-line callout in the launch runbook or folding the precache into
  `download_hotpotqa_hard_neg_reasoning_data.sh` itself so this can't be missed on the next box.
- **`configs/trainer/midtraining.yaml`'s new `learning_rates` block is a shared-file change**,
  not scoped to this script. `train_musique_sft_midtrain.sh`, `train_multihop_sft_midtrain.sh`,
  and `train_musique_ground4layer_midtrain.sh` all currently pass a single uniform
  `trainer.learning_rate=1e-4`, which is now inert — a future re-run of any of them gets the
  mem=1e-4/embed=1e-5/main(-or-frozen)=1e-5 split instead of the uniform rate their own header
  comments describe and, for the ground4layer script, whose comparison methodology depends on.
  Not reconciled here (out of scope for this change); worth revisiting those three scripts'
  comments — or giving them an explicit `+trainer.learning_rates.*` override to restore their
  original uniform behavior — before anyone relies on a byte-for-bit re-run of one of them.
- **Results are in — this finetune regresses the hotpotqa hybrid eval.** Full checkpoint sweep
  (steps 500-10000) and interpretation:
  [2026-08-23-hotpotqa-hard-neg-reasoning-finetune-checkpoint-sweep](../experiments/2026-08-23-hotpotqa-hard-neg-reasoning-finetune-checkpoint-sweep.md).
  Headline: ties baseline at step 500, then collapses to a noisy plateau by step 1000-2000 and
  stays there through step 10000 — a step, not a slope, coinciding with the LR schedule reaching
  peak. Per the docs the checkpoint-comparison should watch: msmarco/hotpotqa-hybrid numbers in
  `hard_neg_think_c512` are already-in-domain-ish for HotpotQA-family checkpoints from this
  lineage — see
  [2026-08-16-multihop-training-data-vs-hotpotqa-eval-mismatch](../experiments/2026-08-16-multihop-training-data-vs-hotpotqa-eval-mismatch.md)
  for the adjacent caveat about what counts as in- vs. out-of-domain for this checkpoint family.
- **`neg_score_threshold: 0.0` is a no-op given the data as shipped** (every score is the 0.0
  placeholder). If this dataset (or a later `-parts-N-M` shard) ever ships real mined scores, the
  effective filter will silently start doing something — worth re-checking the datasets-server
  rows endpoint again before trusting a re-download blindly.
- Only `-parts-0-1` of what its name implies is a multi-part dataset is wired up; if further parts
  exist/get published, this config would need `hf_config`/split handling revisited.
