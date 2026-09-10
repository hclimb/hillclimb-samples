# Indexed data loader: synthesize state on restore

**Date:** 2026-08-12 · **Author:** johnzhang · **Status:** done · **Branch:** `loader-resume-synth` (off `data_resume`, plus cherry-pick of `db492d76` from `lr`)

## What changed

`trainer._restore_loader_state` now falls back to **synthesizing** a valid
Grain iterator state at `K = step * batch_size` whenever the sibling
`dataloader_state.json` is missing, undecodable, or fails a max-index sanity
check. Any ckpt is now resumable at the correct data position — pre-fix,
post-fix, or with a corrupted state file — which closes the tpunanny
preempt-and-respawn loophole.

`QADatasetIndexed.set_loader_state` was widened to accept either raw bytes (a
real Grain state, current behavior) or an int K (queue synthesis, materialize
in `generator()` using the fresh iterator's own `get_state()` as a template).

`QADatasetIndexed.generator` no longer swallows `iterator.set_state()`
failures with a warn — those failures now raise. Silent index-0 replay was
the exact regression this design closes; hiding a set_state failure would
re-open it under a different failure mode (bad bytes rather than missing file).

## Motivation

Post-`db492d76` (the "actually save/restore Grain iterator state" fix), resume
is correct only on the *happy path*: ckpts saved after the fix, with a valid
sibling `dataloader_state.json`, resumed by code that also has the fix. Three
real failure modes remain:

1. **Pre-fix ckpts** (any ckpt saved by the buggy `8227fcab` code, or by
   `qa.py`'s original skip-under-`is_indexed` early-return): no state file
   next to the ckpt. Resume silently starts the Grain iterator at index 0
   while the trainer step counter climbs — post-resume step N produces
   batch (N − restart_step) content.
2. **Stale/wrong state file** — observed on the `e3m3` sweep run: `dataloader_state.json`
   at step 15000 held `max_seen=249` (i.e., a freshly-init loader) because
   two independent preemptions in the run's first 25k steps reset the loader
   each time, and each subsequent save captured that reset state. Reading
   the file verbatim would resume at position ~249 with a trainer step
   counter of 15001+.
3. **Corrupted or partial state file** — e.g., a save interrupted mid-write
   (`fsspec.open(path, "w")` is not atomic against a hard host kill).

The **derivation-based approach** ("sample-at-step is `step * batch_size`; the
sampler is deterministic in seed + config") was the original design premise
that `db492d76`'s commit message called "correct in principle but never wired
up." This change wires it up as the fallback that always fires when the file
path fails.

## Options weighed

1. **Always synthesize; ignore the sibling file entirely.**
   Simpler code, no case split. Rejected because the file is a useful audit:
   when it disagrees with the synthesized K, that disagreement is diagnostic
   evidence of an upstream bug (config drift, sampler-arg change without a
   fresh run, corrupted save). Logging the disagreement is cheaper than
   losing the signal.
2. **Prefer file, fall back to synthesis on missing/stale (chosen).**
   Keeps the file as the primary source of truth (so warm-start ckpts written
   with a legitimate mid-run state are respected), adds synthesis as the
   safety net for the failure modes above. Sanity check on `max(last_seen)`
   catches stale-write cases like e3m3-15k.
3. **Do the synthesis at ckpt-save time instead** (`_save_loader_state`
   writes the synthesized state).
   Rejected because save-time synth is redundant when the loader is running
   correctly (`iterator.get_state()` is the ground truth), and it wouldn't
   help pre-fix ckpts already on GCS.
4. **Hand-run Workflow B** (mid-run rescue, per `wiki/data/indexed-loader-resume.md`).
   Rejected as the durable answer: it requires human intervention on every
   tpunanny respawn from a pre-fix ckpt. Fine as an emergency lever for
   ckpts already on disk; not fine as the steady-state resume story.

## How it was built & integrated

Three files touched, one added:

- `data/qa.py::QADatasetIndexed`:
  - `set_loader_state(state_or_K)` — bytes → verbatim restore; int → queue
    synth for global position K.
  - `generator()` — if `_pending_synth_K` is queued, materialize the state
    by taking a template from `iter(pipeline).get_state()` (which populates
    `sampler` / `data_source` / `worker_count` in Grain's canonical form)
    and overwriting `last_seen_indices` / `last_worker_index` per the closed
    form (`last_seen_indices[i] = i + K − W`, `last_worker_index = W − 1`).
    Then applies it via `iterator.set_state`. `set_state` failures now
    propagate — no warn-and-continue.
  - Class docstring updated: `is_indexed` remains a type marker but no
    longer implies "trainer skips save/restore".
- `trainer/trainer.py::_restore_loader_state`:
  - Reads sibling file; if the indexed path and the decoded `max(last_seen_indices)`
    is not within `max(0.01 × K, 4 × num_workers × batch_size)` of `K − 1`,
    treats the file as stale and calls `data.set_loader_state(K)` (the
    synth path). If file is missing entirely on the indexed path, also synth.
    On the streaming path (non-indexed), behavior is unchanged.
  - Logs which path was taken: `restored from file`, `synthesized … (file=missing)`,
    or `synthesized … (file=stale)`.
- `tests/test_indexed_loader_synthesis.py` — new standalone box-test that
  runs a reference loader K batches forward, synthesizes state at K, and
  compares field-by-field (allowing prefetch slack on `max_seen`).

No new config keys. `SKIP_LOADER_RESTORE=1` still respected for warm-start intent.

## Reference pages updated

- `wiki/data/indexed-loader-resume.md` — updated to reflect that the trainer
  now performs Workflow B automatically. The manual `python + gsutil cp`
  synthesis recipe is retained under an "Emergency lever (rarely needed)"
  header, since it's still the fastest way to rescue a not-yet-relaunched ckpt.

## Tests

`tests/test_indexed_loader_synthesis.py` — standalone script; requires Grain
+ arrayrecord data (box execution).

Cannot run on the Windows dev box (no Grain runtime, no arrayrecord data).
Run on any TPU box after a `multi-vm-tpu-run.sh` sync:

```
uv run python tests/test_indexed_loader_synthesis.py \
    --indexed-uri gs://memory-layers-training/indexed/4bb340af07bdd0ef \
    --k-steps 32
```

Expected pass output includes `[OK] roundtrip: set_state(synth_bytes) recovers
the exact state` and `[PASS] all assertions succeeded`. **Actual output to be
pasted here after box run.**

## Follow-ups & risks

- The `4 × num_workers × batch_size` tolerance in the sanity check was picked
  by measurement (baseline `_pf32_indexed_lr_masked` at step 80000 shows
  `max_seen − (K−1) = 250` for batch_size=16, num_workers=16, prefetch=4 →
  well inside 1024). If a future recipe bumps prefetch or batch, the tolerance
  should be revisited.
- `_save_loader_state` still uses non-atomic `fsspec.open(path, "w")`. A
  hard host kill mid-write can leave a truncated JSON. The sanity check
  catches this (decode failure → synth), but the correct long-term fix is
  write-to-tmp + atomic rename. Deferred; not on the critical path.
- The streaming (non-indexed) resume path is unchanged. If we ever bring
  back streaming for a run that also uses `db492d76`-style save, it'll keep
  its current fast-forward-from-0 fallback on missing state.
