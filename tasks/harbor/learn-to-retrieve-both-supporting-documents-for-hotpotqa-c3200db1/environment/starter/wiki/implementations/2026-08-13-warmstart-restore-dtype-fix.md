# Warm-start/resume checkpoint restore now preserves fp32-promoted dtype

**Date:** 2026-08-13 · **Author:** claude (session with rohunagrawal) · **Status:** done —
fix + regression test verified; overnight retraining arm in progress, see
[wiki/experiments/2026-08-13-doc-access-per-query-loss-investigation.md](../experiments/2026-08-13-doc-access-per-query-loss-investigation.md)

## What changed

`utils.py::load_checkpoint`'s `restore_and_reshard` helper (used by both the warm-start and
full-resume restore paths) now casts a restored leaf to the target array's dtype before
re-sharding. Before this change it only did `jax.device_put(val, a.sharding)` — sharding, not
dtype — so a leaf restored from a checkpoint saved in bf16 stayed bf16 even when the target
model had just promoted that same leaf to fp32 via `promote_trainable_to_fp32`. One line
(`if val.dtype != a.dtype: val = val.astype(a.dtype)`), but it silently neutralized an
already-shipped bug fix for every warm-started run to date.

## Motivation & context

`promote_trainable_to_fp32` (`utils.py`, called from `train.py` right after model init, before
`setup_optimizer`) exists to escape the bf16-ULP freeze:
[wiki/experiments/2026-08-07-bf16-ulp-freeze-empirical-confirmation.md](../experiments/2026-08-07-bf16-ulp-freeze-empirical-confirmation.md)
showed that with bf16-stored trainable weights, `optax.adamw`'s bf16 `mu`/`nu` produce updates
that round away to exactly zero for any weight whose magnitude is more than a few ULPs above the
per-step update size — which is essentially every weight in this model (RMSNorm scales at
~1.0, projections at ~0.02). The fix stores every leaf matched by the union of all
`training_stages[*].trainable_params` regexes in fp32 instead, so `optimizer.init` allocates
fp32 `nu` (and the weight array itself never rounds a small update to zero).

Tonight's task was investigating why `train_multihop_ground4layer_s1warmstart_no_multihop.sh`
(warm-started from `ground_s1_zeroinit_4layer_no_multihop` step 26000, training the new
`staged_ground_batched_isolation_docaccess_warmstart` recipe with
`doc_access_per_query_loss` weight bumped 0.1→1.0) showed that loss fluctuating 1–3 with no
downward trend over its first ~50 logged steps, even at the higher weight. Reading
`losses/doc_access_per_query_loss.py` and `models/memory.py::mem_lookup_batched` first —
the contrastive objective and its gradient path into `mem_q_proj`/`mem_k_proj` look
structurally sound (full pre-top-k logits are exposed via `mem_scores` specifically so
gradient reaches positives outside the current top-k, not just the discrete top-k selection).
That ruled out an obvious loss-formulation bug and pointed at the training dynamics instead —
which is what led back to the fp32-promotion mechanism and its interaction with warm-starting.

**Direct evidence, from the live run's own log** (`run-train_multihop_ground4layer_s1warmstart_no_multihop.log`):

```
[promote_fp32] promoted 344 leaves bf16 -> fp32:
  ...
[weight-monitor] snapshotting 21 watched weights for elementwise diff:
  main_model.layers.14.mem_q_proj   shape=[4, 1024, 2560]  dtype=bfloat16  ...  MIXED
  main_model.layers.14.mem_o_proj   shape=[2560, 4, 1024]  dtype=bfloat16  ...  OK
  embed_model.mem_k_proj            shape=[1024, 1024]     dtype=bfloat16  ...  MIXED
  embed_model.mem_v_proj            shape=[1024, 1024]     dtype=bfloat16  ...  MIXED
```

`promote_fp32` ran and promoted these exact leaves (confirmed: 344 leaves, includes
`main_model.layers.14.mem_q_proj` / `mem_o_proj`, `embed_model.mem_{k,v}_proj`) — but the
weight-monitor snapshot taken immediately after, post-restore, shows them back at
`dtype=bfloat16`. The warm-start restore path silently threw the promotion away. Whatever the
`ground_s1_zeroinit_4layer_no_multihop` checkpoint's exact provenance (its own script,
`train_ground_s1_no_multihop.sh`, isn't in this tree to inspect), its saved leaves are bf16,
and `restore_and_reshard` had no code path to upcast them to the fp32-promoted target's dtype.

Net effect: every warm-started multihop run to date (this one, and presumably the earlier
`train_multihop_ground4layer_s1warmstart.sh` lineage it mirrors) has been training its
retrieval-scoring projections under the exact bf16-ULP trap `promote_trainable_to_fp32` was
built to close — the promotion ran, logged its usual message, and looked like it worked, while
the very next thing that touches those weights (checkpoint restore) undid it.

## Options weighed & tradeoffs

- **Cast inside `restore_and_reshard` (chosen).** One shared helper, used by both the
  warm-start (`weights`-only) and full-resume (`weights` + `opt_state`) branches of
  `load_checkpoint`. Fixes both call sites for free, minimal diff, no new config surface.
- **Re-run `promote_trainable_to_fp32` again AFTER restore, instead of fixing the restore
  itself.** Rejected: papers over the bug instead of fixing it, and doesn't help
  `opt_state` restore on the full-resume path (where a stale bf16 `nu` would have the exact same
  problem if it were ever exercised — currently masked because same-lineage resumes restore
  fp32-saved `opt_state` into an fp32 target, so dtypes already match there today, but that's
  incidental, not guaranteed).
  Fixing the general-purpose `restore_and_reshard` closes the whole class, not just this
  instance.
- **Force the checkpoint saved dtype to fp32 upstream instead of casting on restore.** Not
  applicable — the whole point is warm-starting from checkpoints that may predate this run's
  own promotion decision (a different lineage, an older commit); the target run's
  `training_stages` is the authority on what should be fp32 now, not what a source checkpoint
  happened to be saved as.

## How it was built & integrated

`utils.py::load_checkpoint::restore_and_reshard(r, a)`: `a` is always a leaf of `model.weights`
(or `opt_state`) as it exists in the CURRENT run — i.e. already fp32-promoted where applicable —
and `r` is the restored value read from the checkpoint. Added a dtype check/cast right before the
existing `jax.device_put(val, a.sharding)`:

```python
if hasattr(val, 'dtype') and hasattr(a, 'dtype') and val.dtype != a.dtype:
    val = val.astype(a.dtype)
```

Idempotent when dtypes already match (the overwhelmingly common case: frozen bf16 leaves
restoring into a bf16 target, or a same-lineage full resume where both sides are already fp32).
Only changes behavior when the source and target dtypes genuinely differ, which is exactly the
warm-start-from-an-older/bf16 checkpoint case this note is about.

## Reference pages updated

- [wiki/training/optimizer.md](../training/optimizer.md) — the "`mu`/`nu` are bf16, not fp32"
  section was stale (didn't mention `promote_trainable_to_fp32` at all, even though it's already
  unconditional in `train.py` on this branch); rewrote it to describe the current promotion +
  `mu`-bf16/`nu`-fp32 split, and added the restore-path caveat this fix closes. (Aside: the
  canonical implementation note for `promote_trainable_to_fp32` itself,
  `2026-07-27-fp32-master-weights.md`, exists on the `memory-layers-fp32`/`draft` branches but
  was never merged to `main` — this branch only has the code, not that doc. Out of scope to
  reconcile tonight; flagging so it doesn't look like an accidental dangling link next time
  someone greps for it.)

## Tests

`tests/test_warmstart_restore_dtype.py` — reproduces the exact sequence with real orbax
save/restore against a temp directory (CPU-only, no TPU, no HF weights): saves a "legacy" bf16
checkpoint, builds a NEW target with `promote_trainable_to_fp32` applied (mirroring train.py's
ordering), restores through the real `load_checkpoint` warm-start path, and asserts the restored
leaves are fp32 with the source's numeric values intact (plus a frozen leaf staying bf16
unchanged, to confirm the cast is genuinely conditional, not a blanket upcast).

```
JAX_PLATFORMS=cpu uv run python tests/test_warmstart_restore_dtype.py
```

Run on `tn-v6e-8-0` (memorylayers/europe-west4-a) via
`scripts/misc/run_warmstart_restore_dtype_test.sh`:

```
[promote_fp32] promoted 2 leaves bf16 -> fp32:
  embed_model.mem_k_proj                                        shape=[16, 16]  elems=256
  main_model.layers.14.mem_q_proj                               shape=[16, 16]  elems=256
[promote_fp32] total promoted params: 512 (0.00 GB fp32; +0.00 GB vs bf16). Plus 2x for Adam mu+nu.

[PASS] 2 promoted leaves restored as fp32 with source values intact
[PASS] frozen leaf main_model.embed_tokens stays bf16, values exact

All warm-start restore dtype tests passed.
```

**Verified the test actually catches the bug**, not just a tautology: reverted the fix
(`git stash` on `utils.py` only), reran the identical test, got the expected failure —

```
AssertionError: [FAIL] main_model.layers.14.mem_q_proj restored as bfloat16, expected float32
— warm-start restore silently undid promote_trainable_to_fp32
```

— then restored the fix (`git stash pop`) and confirmed the pass again.

## Follow-ups & risks

- **The currently-running `multihop_ground4layer_s1warmstart_no_multihop_docaccess_warmstart_...`
  run (started 2026-08-13 00:32 UTC, wandb `multihop_ground4layer_s1warmstart_no_mul-2026-08-13-00-32-22`)
  trained ~130 steps under the bug before this fix landed.** It was killed and relaunched from the
  same warm-start checkpoint once the fix was synced — see the experiment doc for the relaunched
  run's id and the before/after weight-monitor comparison.
- **Every other warm-started run in this repo's history should be treated as suspect** for its
  promoted leaves specifically (not everything — frozen leaves and fresh, never-warm-started runs
  were never affected). Not re-auditing old completed runs tonight; flagging as a real prior for
  anyone interpreting old warm-start results' retrieval metrics.
- Did not touch `load_inference_checkpoint` (the eval-time restore path) — it has its own
  `restore_none` helper with the same missing-cast gap, but eval loads a checkpoint's weights
  wholesale (no separate "target dtype" to preserve, no `promote_trainable_to_fp32` call in
  `eval.py`), so there's no known live bug there. Worth a second look if eval-time dtype ever
  becomes suspect.
