# Unfreezing the main 4B on ground_s1_zeroinit_4layer — attempt log

**Date:** 2026-07-20 · **Author:** rohunagrawal · **Status:** **blocked — no training steps ran**

## Conclusion

**The run never trained.** Resuming `ground_s1_zeroinit_4layer` @ 38000 with the main 4B unfrozen
was blocked twice, by two independent limits, and no `ce_loss` / NaN / throughput data exists.

1. **Resume is unaffordable as implemented.** Dataloader restore *replays* the stream rather than
   seeking; the first attempt spent **>2 h without producing a single batch**, heading for an OOM.
2. **The unfrozen 4B does not fit a v6e at any batch size.** ~32 GB/chip needed vs ~31 GB
   available, and halving the batch changed free HBM by 0.6 MB.

What *was* established: the mechanism works end-to-end up to the memory wall — a fresh-stream
resume reaches the trainer in minutes, stage C fires at exactly step 38000, and **150 real
main-model weight tensors** become trainable at `peak_lr=1e-05`. And the chase produced a
generally useful finding about optimizer-state allocation
([implementation note](../implementations/2026-07-20-optimizer-moment-allocation.md)).

## Motivation

`staged_ground` never trains the main model in either stage, so its Qwen3-4B is pristine. The
question was whether giving the LM itself capacity to *use* the memory-injected residual improves
grounding. A full unfreeze at 1e-4 had already regressed the LM on the MuSiQue midtrain (ce_loss
0.560 → 0.357 → past 0.560 while retrieval improved), so this used a **new stage C at 1e-5**
(`configs/trainer/staged_ground_mainunfreeze.yaml`), leaving stages A/B byte-identical so the
config still describes how the checkpoint was made.

## What happened, in order

| # | Config | Outcome |
|---|---|---|
| 1 | full resume, `num_workers` 16 | **>2 h in dataloader fast-forward, no first batch**; RAM 221 → 469 GB as workers spawned (~45 GB each, ~1 per 12 min), `sshd` refusing connections; killed to pre-empt OOM |
| 2 | `SKIP_LOADER_RESTORE=1`, reseed, `num_workers` 8, batch 16 | reached stage C in minutes; **HBM OOM** — `allocate 47.50M, 36.65M free` |
| 3 | as above, batch 8 | **HBM OOM** — `allocate 47.50M, 37.26M free` (halving the batch bought **0.6 MB**) |
| 4 | LoRA rank 16 on MLP | **HBM OOM** — `allocate 20.00M, **2.17M free**` — *less* headroom than the full unfreeze |
| 5 | LoRA + `MEM_MASKED_OPTIMIZER=1` | never reached the optimizer: `opt_state` pytree change is unrestorable against the 38000 checkpoint |

Arm 4 is the informative one: LoRA should have freed ~16 GB of moments and instead left the
tightest margin of all, which is what exposed
[the allocation bug](../implementations/2026-07-20-optimizer-moment-allocation.md).

## Fresh-stream resume (what unblocked #1)

`SKIP_LOADER_RESTORE=1` (`trainer.py`) resumes weights + optimizer but starts the stream at 0,
**paired with a changed `dataset.shuffle_seed`** — on the original seed, "start from 0" replays
exactly the samples the checkpoint already trained on, in the same order.

Reseeding is a sound approximation here: the original run consumed only **~10% of one epoch**.

> **Do not sum the 16 cursors in `dataloader_state.json`** — an easy misreading (I made it, and
> initially reported a 20:1 filter ratio from it). Each grain worker holds its **own** iterator over
> the **full** stream and takes a strided slice; `_count` increments on *every* item pulled,
> including ones the stride skips (`data/qa.py::_StreamingIterator`). The 16 values are 16
> positions in **one** stream, not disjoint shares.
>
> | quantity | value |
> |---|---|
> | stream position (cursors 670k–960k) | ~800k source items |
> | samples emitted (38000 × 16) | 608,000 → filter drops ~24% |
> | staged rows | 8,105,497 → **~10% of an epoch** |
>
> The ~13M figure often quoted is the **replay work** (16 workers × ~800k item-reads), not data.

## Reproducibility

- **Commit:** branch `multihost-v6e-slice`.
- **TPU:** v6e-8 flex slice, 2 × `ct6e-standard-4t`, 2x4, `europe-west4-a` (runbook §2.3).
- **Data:** full parquet, `GROUND_DATA_FRAC=1.0` → 70 GB / 335 shards / 8,105,497 rows per host
  (each host needs its own copy — separate disks).
- **Checkpoint:** `gs://memory-layers-training/ground_s1_zeroinit_4layer-2026-07-04-09-42-55/qwen3_mem_embed`
  (**no trailing step** — a trailing `/38000` takes the weights-only branch: fresh optimizer, step 0).
- No wandb run (`SMOKE=1` disables it); no checkpoints written.

```bash
TRANSPORT=gce ZONE=europe-west4-a PROJECT_ID=memory-layers \
RUN_ENV="SMOKE=1 SKIP_LOADER_RESTORE=1 NUM_WORKERS=8" \
bash scripts/infrastructure/multi-tpu-box-run.sh \
  tpu-v6e-slice-mig-1wjb=scripts/embed/train_ground_s1_mainunfreeze.sh \
  tpu-v6e-slice-mig-1z9d=scripts/embed/train_ground_s1_mainunfreeze.sh
```

## To actually run this

Per-chip HBM is the constraint, and it is **not** what chip count suggests: v6e ~31 GB/chip vs v5p
**~95 GB/chip**, so a v5p-4 has more usable memory for this than a v6e-8 slice.

1. **v5p-4** — science unchanged, batch 16, no config change. Blocked only on flex capacity: every
   request expired ungranted over this session (us-central1-a ×3, us-east5-a ×2).
2. **`tp_devices=2`** on the v6e slice — shards params/optimizer, ~16 GB/chip, keeps 4-way data
   parallelism. Cheapest untested option.
3. **LoRA + `MEM_MASKED_OPTIMIZER=1`** — needs the `load_checkpoint` fallback first, and is a
   **weaker experiment**: capacity confined to a rank-R MLP update, attention frozen. Not a
   substitute for a full unfreeze; report it as LoRA.

## Operational gotchas hit (all now in the runbook)

- Killing a run does **not** kill its data workers — they orphan to `ppid=1` holding hundreds of GB.
- `pkill python3` **misses** the venv binary, whose `comm` is `python`. `pgrep -c python3` then
  reports 0 while a process still holds all four chips (`/dev/vfio/*`). Match on
  `-f venv/bin/python`.
- Importing JAX unguarded on **one** host of a multi-host slice hangs in TPU backend init; use
  `JAX_PLATFORMS=cpu` for CPU-only probes.
- Under memory pressure `sshd` cannot fork and SSH fails with `return code [255]`.
