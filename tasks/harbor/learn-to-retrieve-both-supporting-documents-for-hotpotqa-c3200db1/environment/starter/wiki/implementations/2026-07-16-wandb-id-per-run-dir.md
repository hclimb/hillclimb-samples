# wandb identity keyed on the run-dir, not the run name

*2026-07-16* — supersedes the wandb-id part of
[2026-07-16-hard-neg-think-eval-telemetry.md](2026-07-16-hard-neg-think-eval-telemetry.md).

**`run_name` is not a run identity, and that note made two things depend on it as if it were.
Both are now keyed on the run-dir (`{run_name}-{date}-{time}`), which is unique per launch.**

## The bug

Caught by rohun before it corrupted an experiment. One root cause, two symptoms:

1. **wandb.** `wandb_run_id_from_name(run_name)` gave *every* launch of a name the same id, and
   `train.py` inits with `resume="allow"` — so a relaunch **appended to the earlier attempt's
   run**. Both of 2026-07-16's launches (15-14-02, 16-52-25) landed on `…ch-5bd5d8f9`.
2. **GCS.** `hard_neg_eval_box.py` scanned `prefix=f"{run_name}-"`, i.e. **every launch that ever
   used the name**, and took the max step across all of them.

Symptom 2 was the dangerous one. `qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16` already had
a **complete April run**:

```
…-2026-04-19-20-17-51 :  40000  60000  80000  100000     <- unrelated, complete
…-2026-07-16-15-14-02 :  (empty — the livelocked run)
…-2026-07-16-16-52-25 :  (empty)
```

So the box would have reported `latest_ckpt = 100000`, evaluated the **April model**, and logged
it into the **new** wandb run at `train_step=100000`, then backfilled 80k/60k/40k — a full,
plausible-looking eval curve from the wrong model, sitting next to `train/*` climbing from zero.
No error anywhere.

Ruled out by reading the code: training does **not** resume the old checkpoints.
`setup_checkpointing` only resumes when `trainer.resume_from` is set (it is `null`), and
`_build_gcs_run_dir` always mints a fresh timestamped dir — confirmed by both July dirs being
empty while the log sat at `0/100000`.

## Why the original design was wrong

An *experiment* needs an identity that is **stable across preemption-resume** (one curve) and
**unique per experiment** (no collisions). `run_name` cannot give both: it is stable, so it is by
construction not unique. The first note chose stability and got collisions. The run-dir chooses
uniqueness and gives up resume-continuity — which the repo never had anyway (before `auto`, every
launch got a random id, so resume always forked).

| | before | after |
|---|---|---|
| wandb id | `H(run_name)` — shared by every launch | derived from the run-dir — one per launch |
| eval box target | scans `prefix={run_name}-` → all dirs | one `--run-dir`; a bare name **raises** |
| eval results | `hard_neg_eval/{run_name}/…` | `hard_neg_eval/{run_dir}/…` |
| resume | continues one curve | **forks** — pass a literal `wandb_run_id` to keep one |

## Options considered

| Decision | Options | Chosen |
|---|---|---|
| Identity | (a) unique `run_name` by discipline; (b) explicit id everywhere; (c) **the run-dir** | **(c)** — (a) is one typo from repeating this; (b) is two things to set and still collides if reused; (c) is already unique and already exists |
| id == dir name? | (a) literal; (b) **derived** | **(b)** — wandb ids are capped (≤64) and a real dir is 71 chars; slug the name to 40, keep the 19-char timestamp whole (≈60). Still 1:1 and derivable by both sides. |
| Resume forking | (a) accept; (b) carry the id via `resume_launch.sh` | **(a)**, per rohun. `trainer.wandb_run_id=<literal>` is the escape hatch. |
| Eval box scope | (a) single dir; (b) list | **(a)**, per rohun. Point a second box at a resume's dir. |
| `setup_checkpointing` returns the dir? | (a) 4-tuple; (b) **pure `run_dir_name(cfg)`** | **(b)** — (a) breaks `tests/test_gcs_checkpointing.py` (six unpack sites); (b) also avoids re-running `ensure_gcs_bucket`'s side effects |

## What changed

- `utils.py`: new pure `run_dir_name(cfg)` (the identity; `_build_gcs_run_dir` now wraps it);
  `wandb_run_id_from_name` → **`wandb_run_id_from_run_dir`** (raises on a bare name);
  `resolve_wandb_run_id(cfg, run_dir)`.
- `train.py`: derives the id from `run_dir_name(cfg)` and prints the dir beside the id.
- `scripts/misc/hard_neg_eval_box.py`: takes run-dir(s); `steps_in_run_dir` / `ckpt_path_for_step`
  replace the cross-dir `_steps_by_dir` / `ckpt_dir_for_step`; results keyed by dir.
- `scripts/embed/hard_neg_eval_box_run.sh`: `RUN_DIR` (required, no default) replaces `RUN_NAME`.

## Test record

`scripts/embed/validate_hard_neg_setup.sh` on `rohun-v6e-8-1` (v6e-8, europe-west4-a). Check 4 now
pins the regression against the three **real** dirs that share this run_name:

```
=============== 4. wandb id is per-RUN-DIR, not per-run-name ===============
  qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-04-19-20-17-51
    -> qa_hard_neg_think_sft4b_topk64_seq512_ch-2026-04-19-20-17-51
  qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-07-16-15-14-02
    -> qa_hard_neg_think_sft4b_topk64_seq512_ch-2026-07-16-15-14-02
  qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-07-16-16-52-25
    -> qa_hard_neg_think_sft4b_topk64_seq512_ch-2026-07-16-16-52-25
  bare run_name correctly rejected
  OK
=============== ALL CHECKS PASSED ===============
```

Three distinct ids where the old scheme produced one (`…ch-5bd5d8f9`). Also asserted: determinism,
`gs://…/<dir>/` resolving to the same id as the bare basename, charset/length ≤64, and the
timestamp surviving truncation. Checks 1–3 (corpus `mem_pos_weight_mass`, telemetry block, eval_set
composition) still pass unchanged.

## Follow-up: `trainer.run_start_time` (same day)

Keying identity on the run-dir made the dir unknowable until `train.py` printed it — which broke
`multi-tpu-box-run.sh`, since an eval box needs the dir at launch. Fix: let the timestamp be
**pinned**, so a launcher computes the identity up front and hands the same one to every box.

- `utils.py::run_dir_name` honours `trainer.run_start_time` (validated against
  `YYYY-MM-DD-HH-MM-SS`; malformed raises at startup rather than yielding a dir that
  `wandb_run_id_from_run_dir` can't parse, or one an eval box would poll forever).
- ssh forwards no env, so there was **no channel** to a box-side script. Added one:
  `multi-vm-tpu-run.sh`'s `RUN_ENV="K=V …"` → `tmux_launch.sh` trailing `KEY=VAL` args → exported
  inside the tmux command *after* `setup_shell.sh` (forwarded value beats `.env`). Wired into the
  `DETACH=0` path too so it can't silently no-op.
- `multi-tpu-box-run.sh` mints one UTC `RUN_START_TIME` and forwards it to every box.
- `hard_neg_eval_box_run.sh` takes `RUN_DIR`, or composes it from `RUN_NAME` + `RUN_START_TIME`.
  The run_name reappears here but only to *compose* one exact dir — prefix **scanning** stays gone.

Per rohun: **no "pinned dir already exists" guard** — reusing a `RUN_START_TIME` with the same
`run_name` recreates the collision. Noted at the config key.

Verified on `rohun-v6e-8-1` (check 5, plus `RUN_ENV` reaching the box through the real launcher):

```
[run] forwarding env: RUN_START_TIME=2026-07-16-18-00-00
[tmux_launch] forwarded: RUN_START_TIME=2026-07-16-18-00-00
=============== 5. run_start_time pins the identity ===============
  pinned run_dir : qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-07-16-18-00-00
  wandb id       : qa_hard_neg_think_sft4b_topk64_seq512_ch-2026-07-16-18-00-00
  malformed run_start_time correctly rejected
  OK
=============== ALL CHECKS PASSED ===============
```

The `EXPORTS` quoting (`export K='V';` with `'` escaped) was checked separately against values
containing spaces, embedded single quotes, and the empty case.

## Known gaps

- **`…ch-5bd5d8f9` is abandoned**, not migrated — it holds the two failed 2026-07-16 attempts.
  A fresh launch gets a timestamped id automatically.
- **`ground_eval_box.py` still scans by name** and has symptom 2. Left alone (live runs depend on
  it); safe only while its run names stay unique. Folding both boxes into one module is the
  standing TODO in [../evaluation/eval-boxes.md](../evaluation/eval-boxes.md).
- **The ≤64 id cap is a conservative assumption**, not a verified limit — no box was up to check
  the SDK when this was written. 60 chars leaves margin either way.
- The eval box has still **never completed a run end-to-end**.

Reference pages updated: [evaluation/eval-boxes.md](../evaluation/eval-boxes.md),
[training/trainer-configs.md](../training/trainer-configs.md).
